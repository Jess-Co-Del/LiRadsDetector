"""
Shared constants for the AMPLIFAI LI-RADS classifier
"""

import math
import os
from time import time
from datetime import datetime


# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
LOG_FILE_PATH = os.path.join(LOG_DIR, "lirads.log")
os.makedirs(LOG_DIR, exist_ok=True)


def print_to_log(a, LOG_FILE_PATH=LOG_FILE_PATH):
    timestamp = time()
    dt_object = datetime.fromtimestamp(timestamp)
    line = f"{dt_object}: {a}"
    print(line)
    if LOG_FILE_PATH:
        with open(LOG_FILE_PATH, "a") as f:
            f.write(line + "\n")

# ── Labels ───────────────────────────────────────────────────────────────────
ORDINAL_LABELS = ["LR-1", "LR-2", "LR-3", "LR-4", "LR-5"]
SPECIAL_LABELS = ["LR-M", "LR-TIV", "No lesion"]
VALID_LABELS = ORDINAL_LABELS + SPECIAL_LABELS

# "No lesion" cases have no mask file by design (there's no target lesion to
# segment) -- preprocessing.build_case_tensors uses this to decide when an
# all-zero mask is expected rather than a load failure.
NO_LESION_LABEL = "No lesion"

# The real challenge never scores NO_LESION_LABEL (see CAT_NAMES below), so a
# final submission must never emit it literally. predict.predict_case /
# predict_case_ensemble (used by submission/run.py) remap it to this instead.
# Training and internal fold evaluation (predict.run_inference) never apply
# this remap, so "No lesion" predictions stay visible there for diagnostics.
NO_LESION_SUBMIT_LABEL = "LR-1"

# 4-way super-category head: ordinal vs. the three special classes. Note
# "No lesion" isn't a label the actual AMPLIFAI challenge ever scores (its
# own evaluate.py only recognizes LR-1..LR-5/LR-M/LR-TIV) -- it's trained
# here purely so the model learns to recognize "no real target lesion"
# imagery as its own category instead of that signal corrupting the ordinal
# head (see preprocessing.build_case_tensors's `label` argument). A
# submission pipeline must never emit "No lesion" as a final prediction.
CAT_NAMES = ["ordinal", "LR-M", "LR-TIV", "No lesion"]

# Used by run.py if a case fails preprocessing/inference entirely.
FALLBACK_LABEL = "LR-3"

# ── CT phases ────────────────────────────────────────────────────────────────
PHASE_NAMES = ["ART", "VEN", "DEL"]  # , "DRY"

# Axis in the NIfTI array (as returned by nibabel's get_fdata()) that indexes
# axial slices. AMPLIFAI volumes are harmonized/resampled to a consistent
# orientation, so this is fixed rather than inferred from the affine.
SLICE_AXIS = 2

# ── Slice sampling ───────────────────────────────────────────────────────────
# Cap on how many axial slices (evenly spread across the lesion's z-extent)
# are fed through the backbone per case, per phase.
MAX_SLICES_PER_CASE = 16

# ── CT windowing ─────────────────────────────────────────────────────────────
# Generic abdominal soft-tissue window (HU), applied identically to all four
# phases before normalization. Wide enough to keep both hypervascular lesion
# enhancement and liver parenchyma contrast within range.
WINDOW_CENTER = 0
WINDOW_WIDTH = 500
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
# max_diameter_mm ranges roughly 0-280 in train_metadata.csv (see its
# per-label breakdown) -- clinical_encoder's first layer is a per-sample
# LayerNorm over the whole clinical vector, so a raw mm value would dominate
# that normalization next to the 0/1 one-hot/binary features. Dividing by
# this scale first brings it into a comparable ~0-3 range.
CLINICAL_DIAMETER_SCALE_MM = 290  # MAX at 281.9
CLINICAL_FEATURE_DIM = len(APHE_CATEGORIES) + len(CLINICAL_BINARY_FEATURES) + 1  # 9 (+1 for scaled max_diameter_mm)
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
CNN_HIDDEN_CHANNELS = 256

# ── Data augmentation (training only) ────────────────────────────────────────
# One set of parameters is sampled per case and reused identically for every
# slice of every phase (see augmentation.py), so the 3D-CNN volume branch
# still sees a spatially coherent volume and phases stay mutually aligned.
# Each transform is independently applied with its own probability; a
# disabled transform is a no-op (rotate 0deg / zoom 1.0 / no flip / no
# intensity jitter).
AUGMENT_ROTATION_DEG = 180.0                  # max +/- rotation
AUGMENT_ROTATION_PROB = 0.5
AUGMENT_ZOOM_RANGE = (0.85, 1.15)            # scale factor range
AUGMENT_ZOOM_PROB = 0.5
AUGMENT_FLIP_PROB = 0.5                      # independent prob. for horizontal and vertical flip
AUGMENT_INTENSITY_SHIFT_HU = 15.0            # max +/- additive HU shift (image only, never the mask)
AUGMENT_INTENSITY_SCALE_RANGE = (0.9, 1.1)   # multiplicative HU jitter range

# ── Test-time augmentation (inference only) ──────────────────────────────────
# predict.predict_case/predict_case_ensemble always decide the category gate
# (ordinal vs. LR-M/LR-TIV/No lesion) from a single deterministic pass. When
# that pass says "ordinal", TTA_VIEWS additional forward passes -- each on a
# fresh augmentation.sample_augment_params() view of the same case, reusing
# the same transforms/probabilities as training -- are averaged in with it to
# pick the final LR-1..LR-5 index, trading inference cost for a steadier
# ordinal decision. Never applied to the category gate itself, and never
# applied at training time. 0 disables TTA (single deterministic pass, the
# original behavior).
TTA_VIEWS = 4
AUGMENT_INTENSITY_PROB = 0.5

# ── Lesion transplantation (training only) ───────────────────────────────────
# LR-1/LR-2/LR-3 have very few real cases (see train_metadata.csv). Rather than
# train on the same handful of static examples every epoch, lesion_transplant.py
# extracts a donor case's real, correctly-labeled lesion (all phases, full 3D
# patch) and pastes it into a different recipient case's liver at a random
# plausible location, alpha-feathering the seam. This multiplies background
# diversity (surrounding parenchyma, vasculature, noise) per rare lesion while
# keeping the lesion's own true appearance -- the synthetic case is labeled
# with the donor's real label, never the recipient's.
#
# Placement is constrained to the recipient's own liver, segmented by a
# pretrained nnUNetv2 model (see scripts/segment_livers.py) rather than
# approximated -- the segmenter's output is expected at
# <case_dir>/annotations/liver.nii.gz (find_case_liver_path()), the same
# per-case layout as the existing lesion mask.
TRANSPLANT_DONOR_LABELS = ["LR-1", "LR-2", "LR-3", "LR-4"]
TRANSPLANT_PROB = 0.5              # per __getitem__ call on an eligible donor case
TRANSPLANT_MARGIN_FRAC = 0.15      # patch margin around the lesion bbox, each side
TRANSPLANT_FEATHER_VOX = 4         # gaussian-blur radius (voxels) for the paste alpha
TRANSPLANT_LIVER_ERODE_MARGIN_VOX = 2  # extra shrink of the recipient liver mask, beyond the patch's own half-extent, so the pasted patch doesn't touch the liver boundary
TRANSPLANT_MAX_PLACEMENT_ATTEMPTS = 25  # random center draws tried before giving up on a recipient

# Phase the liver segmenter was trained/run on (portal venous, the standard
# phase for liver segmentation datasets like LiTS); segment_livers.py reads
# this phase's volume per case.
LIVER_SEGMENTATION_PHASE = "DEL"

# ── Anatomy-informed augmentation (training only) ────────────────────────────
# Locally warps around the case's own lesion segmentation to give the lesion
# a plausible new shape/size -- capsule-like bulging/indentation of the
# lesion boundary from breathing or adjacent-tissue distension -- instead of
# a generic global affine warp. Adapted from batchgenerators'
# AnatomyInformedTransform ("Anatomy-informed Data Augmentation for Enhanced
# Prostate Cancer Detection", MICCAI 2023:
# https://github.com/MIC-DKFZ/anatomy_informed_DA), which deforms around an
# organ boundary; here that "organ" is the lesion mask itself. Applied to the
# full 3D volume+mask before z-index slicing (see
# preprocessing.build_case_tensors_from_volumes) rather than the 2D per-slice
# stack the other geometric transforms use, since it needs the lesion's real
# 3D shape to compute the deformation field. Needs only the lesion.nii.gz
# every case already has -- no liver segmentation required.
ANATOMY_AUGMENT_PROB = 0.25                # per-case probability the deformation is applied at all
ANATOMY_DILATION_RANGE_VOX = (-15.0, 15.0)  # signed warp magnitude in voxels; negative compresses the lesion inward, positive distends it outward
# In-plane / slice-thickness voxel spacing ratio, needed to scale the warp's
# blur/gradient along the slice axis correctly. Volumes here aren't
# affine-tracked past preprocessing.load_volume (which drops nibabel's
# affine), so this is a fixed approximation rather than computed per case --
# tune it to this dataset's typical CT protocol if slices are markedly
# thicker/thinner than in-plane pixels.
ANATOMY_SPACING_RATIO = 1.0
ANATOMY_BLUR = 16                          # gaussian kernel (voxels) smoothing the lesion gradient field
