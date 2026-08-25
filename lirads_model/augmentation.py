"""Random 2D data-augmentation transforms (rotation, zoom, flips, intensity
jitter), applied at training time only.

One parameter set is sampled per case (`sample_augment_params`) and reused
identically for every slice of every phase in that case, rather than
resampled per slice -- so the per-phase 3D-CNN volume branch still sees a
spatially coherent volume, and the CT phases stay mutually aligned (a
rotated lesion in ART must be the same rotated lesion in VEN/DEL).

The geometric warp (rotate + zoom) is implemented with batchgeneratorsv2
(the augmentation library behind nnU-Net v2) instead of scipy.ndimage: since
every slice of a phase shares the exact same warp, the whole (S, H, W)
per-phase slice stack is treated as the "channel" dimension of a single
batchgeneratorsv2 SpatialTransform call, which builds one sampling grid and
warps all S slices in one batched torch.grid_sample -- replacing what used
to be S separate scipy.ndimage.rotate + scipy.ndimage.zoom calls (each a
single-threaded spline interpolation) per phase.

batchgeneratorsv2's own randomness (its p_* gates and RandomScalar ranges)
draws from numpy's/torch's *global* RNG, which is unsafe to rely on here:
LiRadsCaseDataset is iterated by a DataLoader with num_workers > 0, and
plain `np.random` state is not automatically reseeded per worker process
(unlike torch's, which the DataLoader does reseed) -- so relying on it could
give every worker correlated/duplicate augmentations. To avoid that, all
randomness is drawn once up front from our own `rng: np.random.Generator`
(fresh per __getitem__ call, see preprocessing.build_case_tensors_from_volumes),
and handed to the transform as fixed values / constant callables so its
internal RNG calls never influence the result.
"""

import numpy as np
import torch

from . import config


def sample_augment_params(rng: np.random.Generator) -> dict:
    """
    One set of augmentation parameters, reused for every slice of every
    phase in a case. Each transform is independently enabled with its own
    probability; disabled transforms are no-ops.
    """
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


def apply_geometric_stack(
    img_stack: torch.Tensor, mask_stack: torch.Tensor, params: dict, img_cval: float,
) -> tuple:
    """
    Rotation + zoom (one shared batchgeneratorsv2 affine warp) + flips,
    applied identically to every slice of a phase in a single batched call.
    img_stack/mask_stack: (S, H, W) float tensors, already cropped + resized
    to the same shape -- see preprocessing.prepare_phase_tensors. Returns
    (img_stack_out, mask_stack_out), still (S, H, W) tensors, pixel-aligned
    with each other.
    """
    img_t = img_stack.float()
    mask_t = mask_stack.float()

    do_rotate = params["rotate_deg"] != 0.0
    do_zoom = params["zoom"] != 1.0
    if do_rotate or do_zoom:
        # Imported lazily so training/inference paths that never augment
        # (e.g. plain, non-TTA predict.py inference) don't require this
        # dependency just to import this module.
        from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform

        h, w = img_t.shape[-2:]
        angle_rad = float(np.deg2rad(params["rotate_deg"]))
        # batchgeneratorsv2's "scaling" convention is inverted relative to
        # ours: a *larger* scaling value samples from a wider footprint of
        # the input, i.e. makes objects *smaller* -- the opposite of our
        # "zoom" (>1 == bigger/closer). Invert it so the visual effect
        # matches what sample_augment_params's AUGMENT_ZOOM_RANGE promises.
        inv_zoom = 1.0 / params["zoom"]

        img_transform = SpatialTransform(
            patch_size=(h, w), patch_center_dist_from_border=0, random_crop=False,
            p_rotation=1.0 if do_rotate else 0.0, rotation=lambda **_: angle_rad,
            p_scaling=1.0 if do_zoom else 0.0, scaling=lambda **_: inv_zoom,
            mode_image="bilinear", padding_mode_image="constant", padding_value_image=img_cval,
        )
        # get_parameters only depends on shape + the (deterministic) angle/
        # scale callables above, so computing it once and reusing it for the
        # mask guarantees an identical warp -- image and mask stay aligned.
        warp_params = img_transform.get_parameters(image=img_t)
        img_t = img_transform._apply_to_image(img_t, **warp_params)

        mask_transform = SpatialTransform(
            patch_size=(h, w), patch_center_dist_from_border=0, random_crop=False,
            mode_image="bilinear", padding_mode_image="constant", padding_value_image=0.0,
        )
        mask_t = mask_transform._apply_to_image(mask_t, **warp_params)

    if params["flip_h"]:
        img_t = torch.flip(img_t, dims=(-1,))
        mask_t = torch.flip(mask_t, dims=(-1,))
    if params["flip_v"]:
        img_t = torch.flip(img_t, dims=(-2,))
        mask_t = torch.flip(mask_t, dims=(-2,))

    return img_t, mask_t


def apply_intensity(img_stack: torch.Tensor, params: dict) -> torch.Tensor:
    """Multiplicative + additive HU jitter, applied to every slice of the
    stack identically. Image only -- never applied to the lesion mask."""
    return img_stack * params["intensity_scale"] + params["intensity_shift"]
