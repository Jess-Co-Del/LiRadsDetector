"""Shared constants for the AMPLIFAI LI-RADS classifier."""

import os

# ── Labels ───────────────────────────────────────────────────────────────────
ORDINAL_LABELS = ["LR-1", "LR-2", "LR-3", "LR-4", "LR-5"]
SPECIAL_LABELS = ["LR-M", "LR-TIV"]
VALID_LABELS = ORDINAL_LABELS + SPECIAL_LABELS

# 3-way super-category head: ordinal vs. the two special classes.
CAT_NAMES = ["ordinal", "LR-M", "LR-TIV"]

# Used by run.py if a case fails preprocessing/inference entirely.
FALLBACK_LABEL = "LR-4"

# ── CT phases ────────────────────────────────────────────────────────────────
PHASE_NAMES = ["ART", "VEN", "DEL", "DRY"]

# Axis in the NIfTI array (as returned by nibabel's get_fdata()) that indexes
# axial slices. AMPLIFAI volumes are harmonized/resampled to a consistent
# orientation, so this is fixed rather than inferred from the affine.
SLICE_AXIS = 2

# ── Slice sampling ───────────────────────────────────────────────────────────
# Cap on how many axial slices (evenly spread across the lesion's z-extent)
# are fed through the backbone per case, per phase.
MAX_SLICES_PER_CASE = 32

# ── CT windowing ─────────────────────────────────────────────────────────────
# Generic abdominal soft-tissue window (HU), applied identically to all four
# phases before normalization. Wide enough to keep both hypervascular lesion
# enhancement and liver parenchyma contrast within range.
WINDOW_CENTER = 50
WINDOW_WIDTH = 400
WINDOW_LOW = WINDOW_CENTER - WINDOW_WIDTH / 2
WINDOW_HIGH = WINDOW_CENTER + WINDOW_WIDTH / 2

# ── DINOv2 slice encoder ─────────────────────────────────────────────────────
# Loaded via torch.hub. "github" (needs internet) downloads code+weights and
# is used for training/weight export. "local" reconstructs the architecture
# from a vendored copy of the repo (see scripts/vendor_dinov2.sh) with random
# init — used at inference, where the real weights come from our own
# checkpoint's state_dict instead, so no network call is needed.
DINOV2_HUB_REPO = "facebookresearch/dinov2"
DINOV2_HUB_MODEL = "dinov2_vitl14_reg"

DINOV2_LOCAL_REPO = os.path.join(os.path.dirname(__file__), "vendor", "dinov2_repo")

PATCH_SIZE = 14
IMG_SIZE = 224            # multiple of PATCH_SIZE -> exact patch grid, no rounding
GRID_SIZE = IMG_SIZE // PATCH_SIZE   # 16
EMBED_DIM = 1024          # dinov2 ViT-L/14 hidden size (register tokens don't change this)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Fraction of the lesion's in-plane bounding-box extent added as margin on
# each side before the square crop, so the model sees some surrounding
# parenchyma rather than a tight lesion-only crop.
CROP_MARGIN_FRAC = 0.25
MIN_CROP_SIZE_PX = 32  # floor for tiny lesions, in original-slice pixels

# ── Classification head ──────────────────────────────────────────────────────
HEAD_HIDDEN_1 = 512
HEAD_HIDDEN_2 = 128
HEAD_DROPOUT = 0.2
