"""Lesion copy-paste augmentation for the rare ordinal classes (LR-1/LR-2/
LR-3 by default, see config.TRANSPLANT_DONOR_LABELS).

With only a handful of real examples of these classes (see train_metadata.csv),
training on the same static cases every epoch gives the model little chance
to generalize past their specific backgrounds. transplant_case() instead
extracts a donor case's real, correctly-labeled lesion,the full 3D patch,
across every CT phase, so its true multi-phase enhancement pattern (APHE,
washout, ...) is preserved,and pastes it into a different recipient case's
liver at a random plausible location, alpha-feathering the seam so there's no
hard boundary. The synthesized case is labeled with the donor's real label; it
keeps the lesion's true appearance while varying the surrounding parenchyma,
vasculature, and noise texture each time it's drawn.

Placement is constrained to the recipient's own liver, segmented ahead of
time by a pretrained nnUNetv2 model (scripts/segment_livers.py) rather than
approximated,pasting outside the liver would be anatomically implausible
and could hurt more than help. Cases without a liver mask on disk simply
aren't eligible recipients (see find_recipient_case_id()).
"""

from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt

from . import config, preprocessing


def _bbox_3d(mask: torch.Tensor, margin_frac: float) -> tuple:
    """3D bounding box (as a tuple of (start, stop) pairs, one per axis)
    around mask's positive voxels, expanded by margin_frac of each axis's
    extent on both sides and clamped to mask's shape. Mirrors
    preprocessing._crop_bbox_from_mask, but in 3D and without the
    square/min-size handling that function needs for a 2D model-input crop.

    Uses `.sum(dim=...) > 0` rather than `.any(dim=...)`: torch's `any`/`all`
    only accept a single int `dim` on older torch versions (unlike `sum`,
    which has always accepted a tuple), and this needs to reduce over two
    axes at once."""
    coords = [
        torch.where(mask.sum(dim=tuple(a for a in range(mask.ndim) if a != ax)) > 0)[0]
        for ax in range(mask.ndim)
    ]
    bbox = []
    for ax, c in enumerate(coords):
        lo, hi = int(c.min()), int(c.max())
        extent = hi - lo + 1
        pad = int(round(extent * margin_frac))
        lo, hi = lo - pad, hi + pad + 1
        bbox.append((max(lo, 0), min(hi, mask.shape[ax])))
    return tuple(bbox)


def extract_lesion_patch(phase_vols: dict, mask_vol: torch.Tensor, margin_frac: float = config.TRANSPLANT_MARGIN_FRAC) -> dict:
    """Crops a donor case's lesion out of every phase, at a shared 3D
    bounding box (mask's extent + margin_frac padding). Returns
    {"phases": {phase: cropped_vol}, "mask": cropped_bool_mask}. Raises
    ValueError if the mask has no positive voxels (nothing to extract)."""
    if not mask_vol.any():
        raise ValueError("mask_vol has no positive voxels,nothing to extract a lesion patch from")

    (r0, r1), (c0, c1), (z0, z1) = _bbox_3d(mask_vol, margin_frac)
    return {
        "phases": {phase: vol[r0:r1, c0:c1, z0:z1] for phase, vol in phase_vols.items()},
        "mask": mask_vol[r0:r1, c0:c1, z0:z1],
    }


def _feathered_alpha(mask: torch.Tensor, feather_vox: int) -> torch.Tensor:
    """Soft [0,1] blend weight: exactly 1 for voxels feather_vox/2 or more
    inside the mask, exactly 0 for voxels feather_vox/2 or more outside it,
    and a smooth linear ramp across the boundary in between,a signed
    distance-transform feather, so paste boundary transitions are gradual
    without a hard seam. Deliberately not a global gaussian blur: blurring a
    lesion smaller than the blur radius (routine for LR-1/LR-2, often just a
    few voxels wide) washes out most of its own signal, since blur reduces
    peak amplitude for anything comparable in size to sigma. A distance-
    transform feather instead guarantees any voxel deep enough inside the
    mask keeps alpha=1 regardless of the lesion's size.

    `mask` is a small (already lesion-cropped, see extract_lesion_patch())
    torch.Tensor; distance_transform_edt has no torch-native equivalent, so
    it's computed via a local numpy round-trip,this path only runs at
    training time (dataset.py's transplant augmentation), where scipy is
    already a required dependency, so it doesn't affect inference."""
    if feather_vox <= 0:
        return mask.float()
    mask_np = mask.numpy()
    inside_dist = distance_transform_edt(mask_np)
    outside_dist = distance_transform_edt(~mask_np)
    signed_dist = np.where(mask_np, inside_dist, -outside_dist)
    alpha = np.clip(signed_dist / feather_vox + 0.5, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(alpha)


def _fits_within(liver_mask: torch.Tensor, center: tuple, patch_shape: tuple) -> bool:
    half = tuple(s // 2 for s in patch_shape)
    lo = tuple(c - h for c, h in zip(center, half))
    hi = tuple(l + s for l, s in zip(lo, patch_shape))
    if any(l < 0 for l in lo) or any(h > s for h, s in zip(hi, liver_mask.shape)):
        return False
    region = liver_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return bool(region.all())


def choose_paste_center(
    liver_mask: torch.Tensor,
    patch_shape: tuple,
    rng: np.random.Generator,
    max_attempts: int = config.TRANSPLANT_MAX_PLACEMENT_ATTEMPTS,
) -> Optional[tuple]:
    """Random-search for a center voxel such that the full patch_shape box
    around it lands entirely inside liver_mask,i.e. the pasted lesion
    never spills past the recipient's real liver boundary. Returns None if
    no valid placement was found in max_attempts tries (e.g. the patch is
    larger than the recipient's liver, or the liver mask is empty)."""
    liver_voxels = torch.argwhere(liver_mask)
    if len(liver_voxels) == 0:
        return None

    for _ in range(max_attempts):
        center = tuple(int(v) for v in liver_voxels[rng.integers(len(liver_voxels))])
        if _fits_within(liver_mask, center, patch_shape):
            return center
    return None


def paste_lesion(
    recipient_phase_vols: dict,
    recipient_mask_vol: torch.Tensor,
    patch: dict,
    center: tuple,
    feather_vox: int = config.TRANSPLANT_FEATHER_VOX,
) -> tuple:
    """Alpha-composites `patch` (from extract_lesion_patch()) into
    recipient_phase_vols at `center`, independently per phase, and returns
    a fresh (phase_vols, mask_vol) pair,the recipient's own inputs are
    never mutated in place. The new mask_vol is the patch's lesion mask
    translated to `center` (a crisp copy, not the soft alpha,feathering
    only smooths the pasted image content's boundary, the ground-truth
    lesion shape stays exact). recipient_mask_vol is otherwise discarded:
    callers should only pass a recipient with no real lesion of its own
    (config.NO_LESION_LABEL) so a hidden, unlabeled second lesion is never
    introduced,see find_recipient_case_id().
    """
    patch_shape = patch["mask"].shape
    half = tuple(s // 2 for s in patch_shape)
    lo = tuple(c - h for c, h in zip(center, half))
    hi = tuple(l + s for l, s in zip(lo, patch_shape))
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))

    alpha = _feathered_alpha(patch["mask"], feather_vox)

    new_phase_vols = {}
    for phase, recipient_vol in recipient_phase_vols.items():
        donor_patch = patch["phases"][phase]
        out = recipient_vol.clone()
        region = out[sl]
        out[sl] = alpha * donor_patch + (1.0 - alpha) * region
        new_phase_vols[phase] = out

    new_mask_vol = torch.zeros_like(recipient_mask_vol, dtype=torch.bool)
    new_mask_vol[sl] = patch["mask"]
    return new_phase_vols, new_mask_vol


def find_recipient_case_id(df: pd.DataFrame, exclude_case_id: str, rng: np.random.Generator) -> Optional[str]:
    """Picks a random recipient case_id from df, preferring
    config.NO_LESION_LABEL cases,clean liver background with no real
    lesion of its own, so pasting can't create a hidden, unlabeled second
    lesion (see paste_lesion()). Falls back to any other case if the split
    has none, excluding `exclude_case_id` itself either way. Returns None if
    df has no other case to pick from."""
    candidates = df[df["case_id"].astype(str) != str(exclude_case_id)]
    no_lesion_candidates = candidates[candidates["lirads_score"].str.strip() == config.NO_LESION_LABEL]
    pool = no_lesion_candidates if len(no_lesion_candidates) > 0 else candidates
    if len(pool) == 0:
        return None
    return str(pool["case_id"].sample(random_state=int(rng.integers(2**31))).iloc[0])


def transplant_case(
    donor_case_dir: str,
    donor_case_id: str,
    recipient_case_dir: str,
    recipient_case_id: str,
    rng: np.random.Generator,
) -> tuple:
    """
    Loads the donor's real lesion and the recipient's raw volumes + liver
    mask from disk, and pastes the former into the latter at a random spot
    inside the recipient's liver. Returns (phase_vols, mask_vol, liver_mask),
    phase_vols/mask_vol in the same shape preprocessing.load_case_volumes()
    would for a real case, ready for
    preprocessing.build_case_tensors_from_volumes(); liver_mask is the
    recipient's own liver segmentation (unchanged by the paste, since
    paste_lesion() only touches image/lesion-mask voxels), handed straight
    through so callers can also anatomy-informed-augment this synthesized
    case around the same liver.

    Raises FileNotFoundError if the recipient has no liver.nii.gz yet (see
    scripts/segment_livers.py) and ValueError if no valid placement was
    found (patch too large for this recipient's liver, or the donor mask is
    empty),callers should catch both and fall back to the case's own real
    data rather than let a rare, hard-won training case fail outright.
    """
    donor_phase_paths = preprocessing.find_case_phase_paths(donor_case_dir, donor_case_id)
    donor_mask_path = preprocessing.find_case_mask_path(donor_case_dir)
    donor_phase_vols, donor_mask_vol, _, _ = preprocessing.load_case_volumes(donor_phase_paths, donor_mask_path)
    patch = extract_lesion_patch(donor_phase_vols, donor_mask_vol)

    recipient_phase_paths = preprocessing.find_case_phase_paths(recipient_case_dir, recipient_case_id)
    recipient_mask_path = preprocessing.find_case_mask_path(recipient_case_dir)
    liver_path = preprocessing.find_case_liver_path(recipient_case_dir)
    recipient_phase_vols, recipient_mask_vol, liver_mask, _ = preprocessing.load_case_volumes(
        recipient_phase_paths, recipient_mask_path, label=config.NO_LESION_LABEL, liver_path=liver_path,
    )
    if liver_mask is None:
        raise FileNotFoundError(f"no liver.nii.gz found for recipient {recipient_case_id!r} at {liver_path!r}")

    patch_shape = tuple(s + 2 * config.TRANSPLANT_LIVER_ERODE_MARGIN_VOX for s in patch["mask"].shape)
    center = choose_paste_center(liver_mask, patch_shape, rng)
    if center is None:
        raise ValueError(
            f"no valid placement found for donor lesion (shape={patch['mask'].shape}) "
            f"inside recipient {recipient_case_id!r}'s liver"
        )

    new_phase_vols, new_mask_vol = paste_lesion(recipient_phase_vols, recipient_mask_vol, patch, center)
    return new_phase_vols, new_mask_vol, liver_mask
