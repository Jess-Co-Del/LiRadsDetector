"""
Shared constants for the AMPLIFAI LI-RADS classifier
"""

import math
import os

# ── Labels ───────────────────────────────────────────────────────────────────
ORDINAL_LABELS = ["No lesion", "LR-1", "LR-2", "LR-3", "LR-4", "LR-5"]
SPECIAL_LABELS = ["LR-M", "LR-TIV"]
VALID_LABELS = ORDINAL_LABELS + SPECIAL_LABELS

# 3-way super-category head: ordinal vs. the two special classes.
CAT_NAMES = ["ordinal", "LR-M", "LR-TIV"]

# Used by run.py if a case fails preprocessing/inference entirely.
FALLBACK_LABEL = "LR-4"

# ── CT phases ────────────────────────────────────────────────────────────────
PHASE_NAMES = ["ART", "VEN", "DEL"]  # , "DRY"

# Axis in the NIfTI array (as returned by nibabel's get_fdata()) that indexes
# axial slices. AMPLIFAI volumes are harmonized/resampled to a consistent
# orientation, so this is fixed rather than inferred from the affine.
SLICE_AXIS = 2

# ── Slice sampling ───────────────────────────────────────────────────────────
# Cap on how many axial slices (evenly spread across the lesion's z-extent)
# are fed through the backbone per case, per phase.
MAX_SLICES_PER_CASE = 8

# ── CT windowing ─────────────────────────────────────────────────────────────
# Generic abdominal soft-tissue window (HU), applied identically to all four
# phases before normalization. Wide enough to keep both hypervascular lesion
# enhancement and liver parenchyma contrast within range.
WINDOW_CENTER = 0
WINDOW_WIDTH = 250
WINDOW_LOW = WINDOW_CENTER - WINDOW_WIDTH / 2
WINDOW_HIGH = WINDOW_CENTER + WINDOW_WIDTH / 2

# ── DINOv2 slice encoder ─────────────────────────────────────────────────────
# Loaded via transformers.AutoModel (HuggingFace Hub). "hub" (needs internet)
# downloads weights and is used for training/weight export. "local" loads
# from a vendored local snapshot (see scripts/vendor_dinov2.sh) with
# local_files_only=True — used at inference, where the real weights come from
# our own checkpoint's state_dict instead, so no network call is needed.
DINOV2_MODEL_ID = "facebook/dinov2-large"

DINOV2_LOCAL_DIR = os.path.join(os.path.dirname(__file__), "vendor", "dinov2-large")

PATCH_SIZE = 14
IMG_SIZE = 224            # lesion crop is resized to this before patch-alignment padding
# DINOv2's patch_embed requires H and W to be exact multiples of PATCH_SIZE.
PADDED_SIZE = math.ceil(IMG_SIZE / PATCH_SIZE) * PATCH_SIZE  # 518
GRID_SIZE = PADDED_SIZE // PATCH_SIZE   # 37
EMBED_DIM = 1024          # dinov2 ViT-L/14 hidden size (register tokens don't change this)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Fraction of the lesion's in-plane bounding-box extent added as margin on
# each side before the square crop, so the model sees some surrounding
# parenchyma rather than a tight lesion-only crop.
CROP_MARGIN_FRAC = 0.5
MIN_CROP_SIZE_PX = 32  # floor for tiny lesions, in original-slice pixels

# ── Classification head ──────────────────────────────────────────────────────
HEAD_HIDDEN_1 = 512
HEAD_HIDDEN_2 = 128
HEAD_DROPOUT = 0.2

# ── Clinical/tabular features ────────────────────────────────────────────────
# Major LI-RADS imaging features recorded per-lesion in train_metadata.csv
# (aphe, washout_venous, washout_delayed, capsule_venous, capsule_delayed).
# The challenge's own submission input is just a case_id (see
# amplifai-codabench/SUBMISSION_GUIDE.md), so this branch is optional per
# case: LiRadsNet falls back to a learned placeholder embedding when it's
# absent, the same pattern used for a missing CT phase.
APHE_CATEGORIES = ["Absent", "Non-rim APHE", "Rim APHE", "Unknown"]
CLINICAL_BINARY_FEATURES = ["washout_venous", "washout_delayed", "capsule_venous", "capsule_delayed"]
CLINICAL_FEATURE_DIM = len(APHE_CATEGORIES) + len(CLINICAL_BINARY_FEATURES)  # 8
CLINICAL_EMBED_DIM = 64

# ── Per-phase 3D-CNN volume encoder ──────────────────────────────────────────
# Alongside the 2D DINOv2 slice encoder, each phase's stack of lesion-cropped
# slices is also treated as a single (1, S, IMG_SIZE, IMG_SIZE) volume and run
# through a small 3D CNN (one per phase, since contrast behavior differs by
# phase), giving the model genuine cross-slice 3D context that per-slice 2D
# processing can't see. An AdaptiveAvgPool3d collapses depth to 1 regardless
# of how many slices S were sampled, producing a fixed-size single-channel
# CNN_FEATURE_MAP_SIZE x CNN_FEATURE_MAP_SIZE feature map per phase, flattened
# and concatenated onto that phase's DINOv2 feature vector.
CNN_FEATURE_MAP_SIZE = 16
CNN_HIDDEN_CHANNELS = 32
