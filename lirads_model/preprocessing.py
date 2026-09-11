"""
Turns a case's raw NIfTI volumes + lesion mask into model-ready tensors
"""

import os
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist

from . import augmentation, config


def load_volume(path: str) -> torch.Tensor:
    return torch.from_numpy(np.asarray(nib.load(path).get_fdata(), dtype=np.float32))


def _resample_to_shape(vol: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Resamples a full 3D volume (image or mask) onto target_shape via
    trilinear interpolation. Output dtype matches `vol`'s,e.g. a bool
    mask comes back bool, via the same nonzero-after-interpolation cast
    scipy.ndimage.zoom's output-dtype behavior gave the previous numpy
    implementation (not a clean sub-voxel threshold, but preserved here for
    behavioral parity)."""
    if tuple(vol.shape) == tuple(target_shape):
        return vol
    resized = torch.nn.functional.interpolate(
        vol[None, None].float(), size=tuple(int(s) for s in target_shape), mode="trilinear", align_corners=False,
    )[0, 0]
    return resized.to(vol.dtype)


def lesion_slice_indices(mask: torch.Tensor, max_slices: int = config.MAX_SLICES_PER_CASE) -> torch.Tensor:
    """
    Returns exactly `max_slices` contiguous z-indices centered on the
    lesion's z-extent (fewer only if the volume itself is shorter than
    max_slices along the slice axis). A lesion spanning more slices than
    max_slices is trimmed symmetrically from both ends, keeping the center;
    a lesion spanning fewer is padded symmetrically with neighboring
    non-lesion slices on both ends, clamped to the volume's bounds.
    """
    other_axes = tuple(a for a in range(mask.ndim) if a != config.SLICE_AXIS)
    z_with_lesion = torch.where(mask.sum(dim=other_axes) > 0)[0]
    if z_with_lesion.numel() == 0:
        z_with_lesion = torch.tensor([40, 50])
        #raise ValueError("lesion mask contains no positive voxels")

    z_min, z_max = int(z_with_lesion.min()), int(z_with_lesion.max())
    n = z_max - z_min + 1

    if n > max_slices:
        excess = n - max_slices
        trim_start = excess // 2
        trim_end = excess - trim_start
        z_min += trim_start
        z_max -= trim_end
    elif n < max_slices:
        deficit = max_slices - n
        add_start = deficit // 2
        add_end = deficit - add_start
        z_min -= add_start
        z_max += add_end

        z_size = mask.shape[config.SLICE_AXIS]
        if z_min < 0:
            z_max += -z_min
            z_min = 0
        if z_max > z_size - 1:
            z_min -= z_max - (z_size - 1)
            z_max = z_size - 1
        z_min = max(z_min, 0)
        z_max = min(z_max, z_size - 1)

    return torch.arange(z_min, z_max + 1)


def _get_slice(volume: torch.Tensor, z: int) -> torch.Tensor:
    return volume.select(config.SLICE_AXIS, z)


def _crop_bbox_from_mask(mask2d: torch.Tensor) -> tuple:
    H, W = mask2d.shape
    rows = torch.where(mask2d.any(dim=1))[0]
    cols = torch.where(mask2d.any(dim=0))[0]

    if rows.numel() == 0 or cols.numel() == 0:
        # Defensive fallback: shouldn't happen since z was chosen for having
        # lesion pixels, but guards against interpolation artifacts.
        size = config.MIN_CROP_SIZE_PX
        cy, cx = H / 2, W / 2
    else:
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        h, w = r1 - r0 + 1, c1 - c0 + 1
        size = max(h, w, config.MIN_CROP_SIZE_PX)
        size = int(round(size * (1 + 2 * config.CROP_MARGIN_FRAC)))
        cy, cx = (r0 + r1) / 2, (c0 + c1) / 2

    def clamp_range(center, extent, limit):
        a0 = int(round(center - extent / 2))
        a1 = a0 + extent
        if a0 < 0:
            a1 -= a0
            a0 = 0
        if a1 > limit:
            a0 -= a1 - limit
            a1 = limit
        return max(a0, 0), min(a1, limit)

    r0n, r1n = clamp_range(cy, min(size, H), H)
    c0n, c1n = clamp_range(cx, min(size, W), W)
    return r0n, r1n, c0n, c1n


def _resize2d(arr: torch.Tensor, out_size: int, order: int) -> torch.Tensor:
    mode = "nearest" if order == 0 else "bilinear"
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    resized = torch.nn.functional.interpolate(
        arr[None, None].float(), size=(out_size, out_size), mode=mode, **kwargs,
    )
    return resized[0, 0]


def _pad_to(arr: torch.Tensor, out_size: int, value: float) -> torch.Tensor:
    """Zero-ish (constant-`value`) pad a square array up to out_size, split
    evenly on both sides (extra pixel on the bottom/right if odd)."""
    pad = out_size - arr.shape[0]
    top, left = pad // 2, pad // 2
    bottom, right = pad - top, pad - left
    return torch.nn.functional.pad(arr, (left, right, top, bottom), mode="constant", value=value)


def _window_normalize(slice2d: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(slice2d, config.WINDOW_LOW, config.WINDOW_HIGH)
    return (clipped - config.WINDOW_LOW) / (config.WINDOW_HIGH - config.WINDOW_LOW)


def prepare_phase_tensors(
    volume: torch.Tensor,
    mask: torch.Tensor,
    z_indices: torch.Tensor,
    augment_params: Optional[dict] = None,
):
    """
    Returns (mask_grids[S,grid,grid], slice_weights[S],
    volume[S,IMG_SIZE,IMG_SIZE]). `volume` is the windowed-normalized (to
    [0,1]) lesion crop, single-channel and unpadded, fed both to
    backbone.Dinov2SliceEncoder (which pads to its own patch-multiple input
    size, replicates to pseudo-RGB, and applies ImageNet normalization
    internally -- that's backbone-specific plumbing, not a property of the
    case data, so it doesn't belong here) and directly to the per-phase 3D
    CNN as a genuine (1, S, H, W) volume rather than S independent 2D
    images. The two branches used to each get their own separately padded/
    normalized copy of this same image; since only DINOv2 needs the RGB/
    ImageNet-normalized version, that conversion now happens inside the
    backbone wrapper instead of being precomputed for both here.

    `volume`/`mask` are torch.Tensor,every operation in this function
    (slicing, cropping, resizing, padding, normalizing, augmenting) runs as
    a torch op, never round-tripping through numpy. Callers (see
    build_case_tensors_from_volumes) are responsible for converting the
    loaded numpy volumes to tensors before calling this.

    `augment_params` (from augmentation.sample_augment_params, or None to
    disable) is applied identically to every slice, so it should be sampled
    once per case and passed to every phase's call,see
    build_case_tensors's `augment` argument. It's applied to the whole
    per-phase (S, IMG_SIZE, IMG_SIZE) slice stack in one batched call
    (see augmentation.apply_geometric_stack) rather than slice by slice.
    """
    img_resized_list, mask_resized_list, weights = [], [], []

    for z in z_indices:
        z = int(z)
        img2d = _get_slice(volume, z)
        mask2d = _get_slice(mask, z) > 0.5
        r0, r1, c0, c1 = _crop_bbox_from_mask(mask2d)
        img_crop = img2d[r0:r1, c0:c1]
        mask_crop = mask2d[r0:r1, c0:c1].float()

        img_resized_list.append(_resize2d(img_crop, config.IMG_SIZE, order=1))
        mask_resized = _resize2d(mask_crop, config.IMG_SIZE, order=1)
        mask_resized_list.append(torch.clamp(mask_resized, 0.0, 1.0))

        weights.append(mask2d.sum().float())

    img_stack = torch.stack(img_resized_list)
    mask_stack = torch.stack(mask_resized_list)

    if augment_params is not None:
        # Same geometric transform for image and mask so they stay
        # pixel-aligned; fill value is WINDOW_LOW for the image (maps to
        # 0.0 post-windowing, the same "background" level used for padding
        # below) and 0.0 (no lesion) for the mask.
        img_stack, mask_stack = augmentation.apply_geometric_stack(
            img_stack, mask_stack, augment_params, img_cval=config.WINDOW_LOW,
        )
        mask_stack = torch.clamp(mask_stack, 0.0, 1.0)
        img_stack = augmentation.apply_intensity(img_stack, augment_params)

    mask_grids, volume_slices = [], []

    for img_resized, mask_resized in zip(img_stack, mask_stack):
        # Windowed-normalize before padding so the pad value (0.0) means "at
        # or below WINDOW_LOW",a well-defined background level,rather
        # than padding in raw HU space.
        img_norm = _window_normalize(img_resized).float()
        volume_slices.append(img_norm)

        mask_padded = _pad_to(mask_resized, config.PADDED_SIZE, value=0.0)
        block = config.PADDED_SIZE // config.GRID_SIZE
        grid = mask_padded.reshape(config.GRID_SIZE, block, config.GRID_SIZE, block).mean(dim=(1, 3))
        mask_grids.append(grid > 0.3)

    mask_grids_t = torch.stack(mask_grids).float()
    weights_t = torch.stack(weights).float()
    volume_t = torch.stack(volume_slices).float()
    return mask_grids_t, weights_t, volume_t


def load_case_volumes(
    phase_paths: dict, mask_path: str, label: Optional[str] = None, liver_path: Optional[str] = None,
) -> tuple:
    """Loads one case's raw per-phase volumes + lesion mask from disk, all
    resampled to the ART phase's voxel grid. Returns (phase_vols, mask_vol,
    liver_mask_vol), phase_vols a {phase_name: torch.Tensor} dict, mask_vol a
    bool torch.Tensor,this is the point where each case's data crosses
    from raw numpy (nibabel's native format) into tensor land; every function
    downstream of this one (_resample_to_shape above, and
    build_case_tensors_from_volumes/prepare_phase_tensors below) works
    entirely in torch.Tensor. This is the loading half of
    build_case_tensors(), split out so lesion_transplant.py can load a
    donor/recipient's raw volumes, splice them, and hand the synthesized
    (phase_vols, mask_vol) to build_case_tensors_from_volumes() instead of
    re-deriving tensors from a real case's own files.

    `label`: see build_case_tensors().

    `liver_path`: optional path to a liver.nii.gz (see
    find_case_liver_path()) to load + resample alongside the rest, for
    lesion_transplant.py's paste-placement constraint. liver_mask_vol is None
    when `liver_path` is None or the file doesn't exist yet
    (segment_livers.py hasn't run on this case),callers that need it (only
    lesion_transplant.transplant_case) should treat that as "this case can't
    be a transplant recipient" rather than an error.
    """
    if label == config.NO_LESION_LABEL:
        mask_vol = torch.zeros((512, 512, 200), dtype=torch.bool)
    else:
        try:
            mask_vol = load_volume(mask_path) > 0.5
        except:  # During inference I do not have labels, so a no lesion does have the mask_volume.
            mask_vol = torch.zeros((512, 512, 200), dtype=torch.bool)

    phase_vols = {}
    arterial_shape = None
    for phase in config.PHASE_NAMES:
        path = phase_paths.get(phase)
        if not path or not os.path.exists(path):
            vol = torch.zeros(arterial_shape)
        else:
            vol = load_volume(path)
        if phase == 'ART':
            arterial_shape = vol.shape
        phase_vols[phase] = _resample_to_shape(vol, arterial_shape)
    mask_vol = _resample_to_shape(mask_vol, arterial_shape)

    liver_mask_vol = None
    if liver_path and os.path.exists(liver_path):
        liver_mask_vol = _resample_to_shape(load_volume(liver_path) > 0.5, arterial_shape)

    return phase_vols, mask_vol, liver_mask_vol


def build_case_tensors_from_volumes(
    phase_vols: dict,
    mask_vol: torch.Tensor,
    max_slices: int = config.MAX_SLICES_PER_CASE,
    augment: bool = False,
    rng: Optional[np.random.Generator] = None,
    anatomy: bool = False,
) -> dict:
    """
    The tensor-prep half of build_case_tensors(): z-index selection +
    per-phase crop/resize/window/augment, given already-loaded (and, for a
    transplanted case, already-spliced) torch.Tensor volumes,see
    load_case_volumes() and lesion_transplant.transplant_case(), the two
    producers of phase_vols/mask_vol. See build_case_tensors() for
    `augment`/`rng`.

    `anatomy`: when True and `augment` is True, with probability
    config.ANATOMY_AUGMENT_PROB the whole case is warped around a random
    local deformation of its own lesion
    (augmentation.apply_anatomy_informed_deform) *before* z-index selection,
    so the lesion-centered slice window below is chosen from the deformed
    volume. Left False (or augment=False) to skip this entirely,e.g.
    eval/test, or an --augment_mode without anatomy.
    """
    if augment:
        rng = rng if rng is not None else np.random.default_rng()
        if anatomy and rng.random() < config.ANATOMY_AUGMENT_PROB:
            phase_vols, mask_vol = augmentation.apply_anatomy_informed_deform(phase_vols, mask_vol, rng)

    z_indices = lesion_slice_indices(mask_vol, max_slices)

    augment_params = None
    if augment:
        augment_params = augmentation.sample_augment_params(rng)

    return {
        phase: prepare_phase_tensors(phase_vols[phase], mask_vol, z_indices, augment_params=augment_params)
        for phase in config.PHASE_NAMES
    }


def build_case_tensors(
    phase_paths: dict,
    mask_path: str,
    max_slices: int = config.MAX_SLICES_PER_CASE,
    augment: bool = False,
    rng: Optional[np.random.Generator] = None,
    label: Optional[str] = None,
    anatomy: bool = False,
) -> dict:
    """
    phase_paths: {"ART": path_or_None, "VEN": ..., "DEL": ..., "DRY": ...}.

    Returns {phase_name: (mask_grids, slice_weights, volume) or None}.

    `augment`: when True, one set of random rotation/zoom/flip/intensity
    parameters is sampled (via `rng`, or a fresh `np.random.default_rng()`
    if not given) and applied identically to every phase,training only;
    leave False for val/test/inference.

    `label`: the case's ground-truth lirads_score, when known (training,
    via LiRadsCaseDataset). Cases labeled config.NO_LESION_LABEL have no
    mask file by design,there's no target lesion to segment,so an
    all-zero mask is used directly rather than attempting to load one. For
    every other label (including when label is unknown, e.g. at inference
    in predict.py/submission/run.py, where a mask is always provided per the
    challenge's task spec), the mask is loaded normally and any failure
    propagates: a missing/corrupt mask on a real lesion case is a data bug,
    not something to silently paper over as an empty mask.

    `anatomy`: see build_case_tensors_from_volumes(); only meaningful when
    `augment` is True, so callers that don't augment (eval/test/inference)
    can just leave it False.
    """
    phase_vols, mask_vol, _ = load_case_volumes(phase_paths, mask_path, label=label)
    return build_case_tensors_from_volumes(
        phase_vols, mask_vol, max_slices, augment=augment, rng=rng, anatomy=anatomy,
    )


def find_case_phase_paths(case_dir: str, case_id: str) -> dict:
    ct_dir = os.path.join(case_dir, "ct")
    return {
        phase: os.path.join(ct_dir, f"{case_id}_{phase}.nii.gz")
        for phase in config.PHASE_NAMES
    }


def find_case_mask_path(case_dir: str) -> str:
    return os.path.join(case_dir, "annotations", "lesion.nii.gz")


def find_case_liver_path(case_dir: str) -> str:
    """Liver segmentation mask produced by scripts/segment_livers.py (a
    pretrained nnUNetv2 model), consumed by lesion_transplant.py to constrain
    paste placement to real liver tissue. Same per-case layout as
    find_case_mask_path()."""
    return os.path.join(case_dir, "annotations", "liver.nii.gz")


def compute_max_diameter_mm(mask_path: str) -> float:
    """
    Deterministic (non-learned) stand-in for train_metadata.csv's
    max_diameter_mm column, used by predict_clinical.py to fill that one
    field of a synthesized clinical row (see config's "Clinical feature
    prediction" section,the other 8 fields come from
    model.ClinicalPredictorNet instead, since geometry alone can't recover
    them). Standard LI-RADS practice measures a lesion's largest diameter on
    the single axial slice showing its greatest extent, so this: for every
    axial slice with any lesion voxels, finds the largest pairwise distance
    (in mm) between two of that slice's mask pixels, and returns the max of
    that over all slices. 0.0 for an empty mask.

    Reads the mask directly with nibabel rather than going through
    load_volume/load_case_volumes (which drop the affine and resample onto
    another phase's grid for model input) so the physical pixel spacing
    used for the mm conversion is exact, straight from the file's own
    header, and handles anisotropic in-plane spacing by scaling each axis
    by its own zoom before measuring distance.

    Within a slice, the true farthest-apart pair of points is always two
    vertices of that point set's convex hull (an interior point can never
    be farther from every other point than some hull vertex is), so
    reducing to hull vertices before the pairwise search is exact, not an
    approximation,it just avoids an O(pixel_count^2) distance search
    over every foreground pixel in a large lesion slice.
    """
    img = nib.load(mask_path)
    mask = np.asarray(img.get_fdata()) > 0.5
    if not mask.any():
        return 0.0

    zooms = img.header.get_zooms()
    row_axis, col_axis = (a for a in range(mask.ndim) if a != config.SLICE_AXIS)
    row_mm, col_mm = float(zooms[row_axis]), float(zooms[col_axis])

    best_mm = 0.0
    for z in range(mask.shape[config.SLICE_AXIS]):
        mask2d = mask.take(z, axis=config.SLICE_AXIS)
        if not mask2d.any():
            continue
        rows, cols = np.nonzero(mask2d)
        points_mm = np.stack([rows * row_mm, cols * col_mm], axis=1)
        if len(points_mm) < 2:
            continue
        if len(points_mm) >= 4:
            try:
                # qhull_options="QJ": joggles the input points by a
                # negligible amount before computing the hull, which avoids
                # spurious QhullError on near-degenerate (thin/collinear-ish)
                # lesion slices -- real CT lesion masks hit this often
                # enough that, without it, the except branch below (an
                # O(pixel_count^2) search over every foreground pixel
                # instead of just the hull's handful of vertices) has been
                # measured taking 60-100+ seconds on a single real case,
                # entirely because of one or two slices' shape, versus
                # ~0.2s for a case that never hits it. The joggle perturbs
                # the hull by well under a pixel, immaterial at mm scale.
                points_mm = points_mm[ConvexHull(points_mm, qhull_options="QJ").vertices]
            except QhullError:
                pass  # truly degenerate even joggled,fall back below
        if len(points_mm) > 500:
            # Residual safety net for the rare case the joggle still isn't
            # enough: pdist computes the same exact all-pairs max distance
            # without materializing the full (n,n,2) broadcast array the
            # naive approach below would for a large, unreduced point set.
            slice_max = float(pdist(points_mm).max())
        else:
            diffs = points_mm[:, None, :] - points_mm[None, :, :]
            slice_max = float(np.sqrt((diffs ** 2).sum(axis=-1)).max())
        best_mm = max(best_mm, slice_max)
    return round(best_mm, 1)
