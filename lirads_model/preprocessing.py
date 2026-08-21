"""
Turns a case's raw NIfTI volumes + lesion mask into model-ready tensors
"""

import os
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom

from . import augmentation, config


def load_volume(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).get_fdata(), dtype=np.float32)


def _resample_to_shape(vol: np.ndarray, target_shape: tuple) -> np.ndarray:
    if vol.shape == target_shape:
        return vol
    factors = tuple(t / s for t, s in zip(target_shape, vol.shape))
    return zoom(vol, factors, order=1)


def lesion_slice_indices(mask: np.ndarray, max_slices: int = config.MAX_SLICES_PER_CASE) -> np.ndarray:
    """
    Returns exactly `max_slices` contiguous z-indices centered on the
    lesion's z-extent (fewer only if the volume itself is shorter than
    max_slices along the slice axis). A lesion spanning more slices than
    max_slices is trimmed symmetrically from both ends, keeping the center;
    a lesion spanning fewer is padded symmetrically with neighboring
    non-lesion slices on both ends, clamped to the volume's bounds.
    """
    other_axes = tuple(a for a in range(mask.ndim) if a != config.SLICE_AXIS)
    z_with_lesion = np.where(mask.sum(axis=other_axes) > 0)[0]
    if len(z_with_lesion) == 0:
        z_with_lesion = np.array([40, 50])
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

    return np.arange(z_min, z_max + 1)


def _get_slice(volume: np.ndarray, z: int) -> np.ndarray:
    return np.take(volume, z, axis=config.SLICE_AXIS)


def _crop_bbox_from_mask(mask2d: np.ndarray) -> tuple:
    H, W = mask2d.shape
    rows = np.where(mask2d.any(axis=1))[0]
    cols = np.where(mask2d.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        # Defensive fallback: shouldn't happen since z was chosen for having
        # lesion pixels, but guards against interpolation artifacts.
        size = config.MIN_CROP_SIZE_PX
        cy, cx = H / 2, W / 2
    else:
        r0, r1 = rows.min(), rows.max()
        c0, c1 = cols.min(), cols.max()
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


def _resize2d(arr: np.ndarray, out_size: int, order: int) -> np.ndarray:
    fy, fx = out_size / arr.shape[0], out_size / arr.shape[1]
    return zoom(arr, (fy, fx), order=order)


def _pad_to(arr: np.ndarray, out_size: int, value: float) -> np.ndarray:
    """Zero-ish (constant-`value`) pad a square array up to out_size, split
    evenly on both sides (extra pixel on the bottom/right if odd)."""
    pad = out_size - arr.shape[0]
    top, left = pad // 2, pad // 2
    bottom, right = pad - top, pad - left
    return np.pad(arr, ((top, bottom), (left, right)), mode="constant", constant_values=value)


def _window_normalize(slice2d: np.ndarray) -> np.ndarray:
    clipped = np.clip(slice2d, config.WINDOW_LOW, config.WINDOW_HIGH)
    return (clipped - config.WINDOW_LOW) / (config.WINDOW_HIGH - config.WINDOW_LOW)


_IMAGENET_MEAN = np.array(config.IMAGENET_MEAN, dtype=np.float32)[:, None, None]
_IMAGENET_STD = np.array(config.IMAGENET_STD, dtype=np.float32)[:, None, None]


def prepare_phase_tensors(
    volume: np.ndarray,
    mask: np.ndarray,
    z_indices: np.ndarray,
    augment_params: Optional[dict] = None,
):
    """
    Returns (pixel_values[S,3,H,W], mask_grids[S,grid,grid], slice_weights[S],
    volume[S,IMG_SIZE,IMG_SIZE]). `volume` is the same windowed-normalized
    lesion crop as pixel_values, but single-channel and unpadded (no
    Imagenet normalization, no patch-alignment padding, no pseudo-RGB) -- fed
    to the per-phase 3D CNN as a genuine (1, S, H, W) volume rather than S
    independent 2D images.

    `augment_params` (from augmentation.sample_augment_params, or None to
    disable) is applied identically to every slice, so it should be sampled
    once per case and passed to every phase's call -- see
    build_case_tensors's `augment` argument.
    """
    pixel_values, mask_grids, weights, volume_slices = [], [], [], []

    for z in z_indices:
        img2d = _get_slice(volume, int(z))
        mask2d = _get_slice(mask, int(z)) > 0.5
        img_crop = img2d
        mask_crop = mask2d
        r0, r1, c0, c1 = _crop_bbox_from_mask(mask2d)
        img_crop = img2d[r0:r1, c0:c1]
        mask_crop = mask2d[r0:r1, c0:c1].astype(np.float32)

        img_resized = _resize2d(img_crop, config.IMG_SIZE, order=1)
        mask_resized = _resize2d(mask_crop, config.IMG_SIZE, order=1)
        mask_resized = np.clip(mask_resized, 0.0, 1.0)

        if augment_params is not None:
            # Same geometric transform for image and mask so they stay
            # pixel-aligned; fill value is WINDOW_LOW for the image (maps to
            # 0.0 post-windowing, the same "background" level used for
            # padding below) and 0.0 (no lesion) for the mask.
            img_resized = augmentation.apply_geometric(img_resized, augment_params, order=1, cval=config.WINDOW_LOW)
            mask_resized = augmentation.apply_geometric(mask_resized, augment_params, order=1, cval=0.0)
            mask_resized = np.clip(mask_resized, 0.0, 1.0)
            img_resized = augmentation.apply_intensity(img_resized, augment_params)

        # Windowed-normalize before padding so the pad value (0.0) means "at
        # or below WINDOW_LOW" -- a well-defined background level -- rather
        # than padding in raw HU space.
        img_norm = _window_normalize(img_resized).astype(np.float32)
        volume_slices.append(img_norm)

        img_padded = _pad_to(img_norm, config.PADDED_SIZE, value=0.0)
        mask_padded = _pad_to(mask_resized, config.PADDED_SIZE, value=0.0)

        chw = np.repeat(img_padded[None, :, :], 3, axis=0)
        chw = (chw - _IMAGENET_MEAN) / _IMAGENET_STD
        pixel_values.append(chw)

        block = config.PADDED_SIZE // config.GRID_SIZE
        grid = mask_padded.reshape(config.GRID_SIZE, block, config.GRID_SIZE, block).mean(axis=(1, 3))
        mask_grids.append(grid > 0.3)

        weights.append(float(mask2d.sum()))

    pixel_values_t = torch.from_numpy(np.stack(pixel_values)).float()
    mask_grids_t = torch.from_numpy(np.stack(mask_grids)).float()
    weights_t = torch.tensor(weights, dtype=torch.float32)
    volume_t = torch.from_numpy(np.stack(volume_slices)).float()
    return pixel_values_t, mask_grids_t, weights_t, volume_t


def load_case_volumes(phase_paths: dict, mask_path: str, label: Optional[str] = None) -> tuple:
    """Loads one case's raw per-phase volumes + lesion mask from disk, all
    resampled to the ART phase's voxel grid. Returns (phase_vols, mask_vol),
    phase_vols a {phase_name: np.ndarray} dict, mask_vol a bool np.ndarray.
    This is the loading half of build_case_tensors(), split out so
    lesion_transplant.py can load a donor/recipient's raw volumes, splice
    them, and hand the synthesized (phase_vols, mask_vol) to
    build_case_tensors_from_volumes() instead of re-deriving tensors from a
    real case's own files.

    `label`: see build_case_tensors().
    """
    if label == config.NO_LESION_LABEL:
        mask_vol = np.zeros((512, 512, 200), dtype=bool)
    else:
        mask_vol = load_volume(mask_path) > 0.5

    phase_vols = {}
    arterial_shape = None
    for phase in config.PHASE_NAMES:
        path = phase_paths.get(phase)
        if not path or not os.path.exists(path):
            vol = np.zeros(arterial_shape)
        else:
            vol = load_volume(path)
        if phase == 'ART':
            arterial_shape = vol.shape
        phase_vols[phase] = _resample_to_shape(vol, arterial_shape)
    mask_vol = _resample_to_shape(mask_vol, arterial_shape)
    return phase_vols, mask_vol


def build_case_tensors_from_volumes(
    phase_vols: dict,
    mask_vol: np.ndarray,
    max_slices: int = config.MAX_SLICES_PER_CASE,
    augment: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    The tensor-prep half of build_case_tensors(): z-index selection +
    per-phase crop/resize/window/augment, given already-loaded (and, for a
    transplanted case, already-spliced) raw volumes. See build_case_tensors()
    for `augment`/`rng`.
    """
    z_indices = lesion_slice_indices(mask_vol, max_slices)

    augment_params = None
    if augment:
        rng = rng if rng is not None else np.random.default_rng()
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
) -> dict:
    """
    phase_paths: {"ART": path_or_None, "VEN": ..., "DEL": ..., "DRY": ...}.

    Returns {phase_name: (pixel_values, mask_grids, slice_weights, volume) or None}.

    `augment`: when True, one set of random rotation/zoom/flip/intensity
    parameters is sampled (via `rng`, or a fresh `np.random.default_rng()`
    if not given) and applied identically to every phase -- training only;
    leave False for val/test/inference.

    `label`: the case's ground-truth lirads_score, when known (training,
    via LiRadsCaseDataset). Cases labeled config.NO_LESION_LABEL have no
    mask file by design -- there's no target lesion to segment -- so an
    all-zero mask is used directly rather than attempting to load one. For
    every other label (including when label is unknown, e.g. at inference
    in predict.py/submission/run.py, where a mask is always provided per the
    challenge's task spec), the mask is loaded normally and any failure
    propagates: a missing/corrupt mask on a real lesion case is a data bug,
    not something to silently paper over as an empty mask.
    """
    phase_vols, mask_vol = load_case_volumes(phase_paths, mask_path, label=label)
    return build_case_tensors_from_volumes(phase_vols, mask_vol, max_slices, augment=augment, rng=rng)


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
