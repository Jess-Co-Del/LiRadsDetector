"""Random 2D data-augmentation transforms (rotation, zoom, flips, intensity
jitter), applied at training time only.

One parameter set is sampled per case (`sample_augment_params`) and reused
identically for every slice of every phase in that case, rather than
resampled per slice -- so the per-phase 3D-CNN volume branch still sees a
spatially coherent volume, and the CT phases stay mutually aligned (a
rotated lesion in ART must be the same rotated lesion in VEN/DEL).
"""

import numpy as np
from scipy.ndimage import rotate as ndi_rotate
from scipy.ndimage import zoom as ndi_zoom

from . import config


def sample_augment_params(rng: np.random.Generator) -> dict:
    """One set of augmentation parameters, reused for every slice of every
    phase in a case. Each transform is independently enabled with its own
    probability; disabled transforms are no-ops."""
    return {
        "rotate_deg": (
            rng.uniform(-config.AUGMENT_ROTATION_DEG, config.AUGMENT_ROTATION_DEG)
            if rng.random() < config.AUGMENT_ROTATION_PROB else 0.0
        ),
        "zoom": (
            rng.uniform(*config.AUGMENT_ZOOM_RANGE)
            if rng.random() < config.AUGMENT_ZOOM_PROB else 1.0
        ),
        "flip_h": bool(rng.random() < config.AUGMENT_FLIP_PROB),
        "flip_v": bool(rng.random() < config.AUGMENT_FLIP_PROB),
        "intensity_shift": (
            rng.uniform(-config.AUGMENT_INTENSITY_SHIFT_HU, config.AUGMENT_INTENSITY_SHIFT_HU)
            if rng.random() < config.AUGMENT_INTENSITY_PROB else 0.0
        ),
        "intensity_scale": (
            rng.uniform(*config.AUGMENT_INTENSITY_SCALE_RANGE)
            if rng.random() < config.AUGMENT_INTENSITY_PROB else 1.0
        ),
    }


def _zoom_centered(arr: np.ndarray, factor: float, order: int, cval: float) -> np.ndarray:
    """Zooms about the center, then crops or pads back to the original
    shape so callers never have to deal with a shape change."""
    if factor == 1.0:
        return arr
    h, w = arr.shape
    zoomed = ndi_zoom(arr, factor, order=order, cval=cval)
    zh, zw = zoomed.shape
    if factor >= 1.0:
        top, left = (zh - h) // 2, (zw - w) // 2
        return zoomed[top: top + h, left: left + w]
    pad_top, pad_left = (h - zh) // 2, (w - zw) // 2
    pad_bottom, pad_right = h - zh - pad_top, w - zw - pad_left
    return np.pad(zoomed, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=cval)


def apply_geometric(arr2d: np.ndarray, params: dict, order: int, cval: float) -> np.ndarray:
    """Rotation + zoom + flips. Shared by the image and its mask (with
    matching `order`/`cval` per caller) so they stay pixel-aligned."""
    out = arr2d
    if params["rotate_deg"] != 0.0:
        out = ndi_rotate(out, angle=params["rotate_deg"], reshape=False, order=order, mode="constant", cval=cval)
    if params["zoom"] != 1.0:
        out = _zoom_centered(out, params["zoom"], order=order, cval=cval)
    if params["flip_h"]:
        out = out[:, ::-1]
    if params["flip_v"]:
        out = out[::-1, :]
    return np.ascontiguousarray(out)


def apply_intensity(img2d: np.ndarray, params: dict) -> np.ndarray:
    """Multiplicative + additive HU jitter. Image only -- never applied to
    the lesion mask."""
    return img2d * params["intensity_scale"] + params["intensity_shift"]
