"""
Turns a case's raw NIfTI volumes + lesion mask into model-ready tensors
"""

import os
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom

from . import config


def load_volume(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).get_fdata(), dtype=np.float32)


def _resample_to_shape(vol: np.ndarray, target_shape: tuple) -> np.ndarray:
    if vol.shape == target_shape:
        return vol
    factors = tuple(t / s for t, s in zip(target_shape, vol.shape))
    return zoom(vol, factors, order=1)


def lesion_slice_indices(mask: np.ndarray, max_slices: int = config.MAX_SLICES_PER_CASE) -> np.ndarray:
    """
    Evenly spread up to `max_slices` z-indices across the lesion's z-extent.
    """
    other_axes = tuple(a for a in range(mask.ndim) if a != config.SLICE_AXIS)
    z_with_lesion = np.where(mask.sum(axis=other_axes) > 0)[0]
    if len(z_with_lesion) == 0:
        z_with_lesion = np.array([1, 10])
        #raise ValueError("lesion mask contains no positive voxels")

    z_min, z_max = int(z_with_lesion.min()), int(z_with_lesion.max())
    full_range = np.arange(z_min, z_max + 1)

    if len(full_range) <= max_slices:
        return full_range

    picks = np.linspace(0, len(full_range) - 1, max_slices)
    picks = np.unique(np.round(picks).astype(int))
    return full_range[picks]


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


def prepare_phase_tensors(volume: np.ndarray, mask: np.ndarray, z_indices: np.ndarray):
    """
    Returns (pixel_values[S,3,H,W], mask_grids[S,grid,grid], slice_weights[S],
    volume[S,IMG_SIZE,IMG_SIZE]). `volume` is the same windowed-normalized
    lesion crop as pixel_values, but single-channel and unpadded (no
    Imagenet normalization, no patch-alignment padding, no pseudo-RGB) -- fed
    to the per-phase 3D CNN as a genuine (1, S, H, W) volume rather than S
    independent 2D images.
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


def build_case_tensors(phase_paths: dict, mask_path: str, max_slices: int = config.MAX_SLICES_PER_CASE) -> dict:
    """phase_paths: {"ART": path_or_None, "VEN": ..., "DEL": ..., "DRY": ...}.

    Returns {phase_name: (pixel_values, mask_grids, slice_weights, volume) or None}.
    """
    try:
        mask_vol = load_volume(mask_path) > 0.5
    except:
        mask_vol = np.zeros((512,512, 200))
    z_indices = lesion_slice_indices(mask_vol, max_slices)

    out = {}
    for phase in config.PHASE_NAMES:
        path = phase_paths.get(phase)
        if not path or not os.path.exists(path):
            vol = np.zeros(arterial_shape)
        else:
            vol = load_volume(path)
        if phase == 'ART':
            arterial_shape = vol.shape
        mask_vol = _resample_to_shape(mask_vol, arterial_shape)
        vol = _resample_to_shape(vol, arterial_shape)
        out[phase] = prepare_phase_tensors(vol, mask_vol, z_indices)
    return out


def find_case_phase_paths(case_dir: str, case_id: str) -> dict:
    ct_dir = os.path.join(case_dir, "ct")
    return {
        phase: os.path.join(ct_dir, f"{case_id}_{phase}.nii.gz")
        for phase in config.PHASE_NAMES
    }


def find_case_mask_path(case_dir: str) -> str:
    return os.path.join(case_dir, "annotations", "lesion.nii.gz")
