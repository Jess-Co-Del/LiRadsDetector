# LiRadsDetector

A LI-RADS classifier for the [AMPLIFAI challenge](https://um-ihc-ca2i.github.io/amplifai-challenge/index.html): given a multi-phase CT scan (ART/VEN/DEL/DRY) and a target lesion segmentation, predicts one of `LR-1, LR-2, LR-3, LR-4, LR-5, LR-M, LR-TIV`.

See [`amplifai-codabench/`](https://github.com/UM-IHC-CA2i/amplifai-codabench.git) for the challenge's own submission spec (`README.md`, `SUBMISSION_GUIDE.md`, `evaluate.py`) — that folder is reference material, not part of this codebase.

## Approach

The model is a hybrid of two image encoders per CT phase — a frozen 2D DINOv2 ViT-L/14 (register-token variant, [`facebook/dinov2-with-registers-large`](https://huggingface.co/facebook/dinov2-with-registers-large), loaded via `transformers.AutoModel`) reading each slice independently, plus a small trainable 3D CNN reading the same slices as one volume:

1. For each of the four CT phases, up to **32 axial slices** are sampled evenly across the lesion's z-extent (all of them if the lesion spans fewer than 32).
2. Each slice is cropped square around the lesion (with margin), resized to 224×224, HU-windowed and replicated to pseudo-RGB.
3. Every slice is run through the frozen backbone, producing a 16×16 grid of patch tokens + a CLS token.
4. The patch-token grid is pooled using the lesion mask (downsampled to the same 16×16 grid), so only lesion-covering patches contribute — a mask-guided pooling head, not a predicted segmentation.
5. Slices are combined per phase, weighted by lesion area in that slice; missing phases get a learned placeholder embedding instead of breaking the pipeline.
6. In parallel, that same phase's slice stack is also treated as one single-channel `(1, S, 224, 224)` volume and run through a per-phase 3D CNN (see below), giving the model 3D context DINOv2's per-slice view can't see.
7. Per phase, `[masked-pooled patch feature | CLS feature | 3D-CNN feature map]` is concatenated; all four phases' vectors plus a clinical-feature embedding (see below) are concatenated again and fed to a small MLP with two heads:
   - a 3-way head (ordinal / LR-M / LR-TIV), which drives the **Special Category Recognition** metric,
   - a 5-way ordinal head (LR-1..LR-5), used only when the case is ordinal, which drives **Adjusted QWK**.

   This split mirrors exactly how `amplifai-codabench/evaluate.py` scores submissions.

Only the DINOv2 backbone is frozen — the 3D CNNs, clinical encoder, and heads are all trained.

Both extra branches below are independently optional, toggled at training time with `train.py --use_cnn`/`--no-use_cnn` and `--use_clinical`/`--no-use_clinical` (both default on). The choice is saved into the checkpoint itself, so `predict.py`/`submission/run.py` always reconstruct the matching architecture automatically — no flag needs to be passed again at inference time. Checkpoints trained before this option existed load as if both had been on (their actual shape).

### 3D-CNN volume branch

Alongside DINOv2, each phase gets its own `PhaseVolumeCNN` (`lirads_model/model.py`) — one 3D CNN per phase, since contrast behavior differs by phase (e.g. washout only shows up on venous/delayed). It takes that phase's `(1, S, 224, 224)` slice stack (the same windowed lesion crop DINOv2 sees, but single-channel and without the patch-alignment padding — `preprocessing.prepare_phase_tensors`'s 4th return value) through a few `Conv3d`/`InstanceNorm3d`/`ReLU` layers, then an `AdaptiveAvgPool3d` that collapses depth to 1 regardless of how many slices `S` were sampled, producing a fixed `16×16` single-channel feature map (`config.CNN_FEATURE_MAP_SIZE`) that's flattened to 256-dim and concatenated onto that phase's DINOv2 feature vector. Pass `--no-use_cnn` to `train.py` to fall back to DINOv2-only.

### Clinical/tabular features

`train_metadata.csv` also records the major LI-RADS imaging features a radiologist annotated per lesion: `aphe`, `washout_venous`, `washout_delayed`, `capsule_venous`, `capsule_delayed`. These are one-hot/binary-encoded into an 8-dim vector (`aphe` one-hot over `Absent`/`Non-rim APHE`/`Rim APHE`/`Unknown`, plus the four binary flags — see `dataset.encode_clinical_features`), run through a small `Linear` projection (`LiRadsNet.clinical_encoder`), and scaled by a learned weight (`clinical_scale`) before being concatenated onto the image encoding. Pass `--no-use_clinical` to `train.py` to train on images alone.

The challenge's own submission input is just a `case_id` — no clinical metadata — so even when this branch is enabled it's optional per case: when `clinical_features` isn't passed to `LiRadsNet.forward` (as in `predict.predict_case`, used by `submission/run.py`), a learned placeholder embedding (`missing_clinical_embed`) stands in, the same pattern already used for a missing CT phase. Training (`train.py`, via `LiRadsCaseDataset`) always supplies the real per-case vector when the branch is enabled.

### Data augmentation

Random rotation, zoom, horizontal/vertical flip, and HU intensity jitter (`lirads_model/augmentation.py`) are applied to the training split only (`LiRadsCaseDataset(..., augment=True)`, wired up by default in `train.py`; `--no-augment` disables it). Each transform is independently enabled with its own probability (see the `AUGMENT_*` constants in `config.py`), and one set of parameters is sampled per case and reused identically for every slice of every phase — so the 3D-CNN branch still sees a spatially coherent volume, the phases stay mutually aligned, and the lesion mask is transformed in lockstep with the image so mask-guided pooling stays correct. Validation/test splits and inference (`predict.py`, `submission/run.py`) never augment.

### Lesion transplantation

`LR-1`/`LR-2`/`LR-3` have very few real cases in `train_metadata.csv`. `lirads_model/lesion_transplant.py` (`train.py --transplant`, off by default) extracts a donor case's real, correctly-labeled lesion — the full 3D patch, across every CT phase, so its true multi-phase enhancement pattern is preserved — and pastes it into a different recipient case's liver at a random plausible location, blending the seam with a distance-transform alpha feather (exact donor signal deep inside the lesion, smooth taper to the recipient's own tissue at the boundary — not a global blur, which would wash out lesions smaller than the feather radius). The synthesized case keeps the donor's real label; only the surrounding parenchyma, vasculature, and noise texture vary each time it's drawn, multiplying background diversity per rare lesion instance without inventing a fake one.

Placement is constrained to the recipient's own liver rather than approximated, using a liver segmentation mask produced ahead of time by a pretrained nnUNetv2 model — run `scripts/segment_livers.py` once (offline, wherever the model + case data live) before training with `--transplant`; it saves one `<case_dir>/annotations/liver.nii.gz` per case, the same per-case layout as the lesion mask. Recipients are always drawn from `config.NO_LESION_LABEL` cases when available (real liver background, no competing real lesion to accidentally leave unlabeled in the synthesized mask), and a case without a liver mask on disk simply isn't used as a recipient — `--transplant` degrades gracefully to no-op if `segment_livers.py` hasn't been run yet, rather than failing.

## Layout

```
lirads_model/
├── config.py         # labels, slice cap, image/patch sizes, CT windowing, hub names
├── preprocessing.py  # NIfTI loading, lesion slice sampling, crop/resize/window, mask->patch-grid
├── augmentation.py    # train-only rotation/zoom/flip/intensity augmentation, shared across a case's phases
├── lesion_transplant.py # train-only lesion copy-paste augmentation for rare ordinal classes (needs scripts/segment_livers.py output)
├── backbone.py        # frozen DINOv2 wrapper (transformers.AutoModel; local/offline or hub/pretrained)
├── model.py           # LiRadsNet: mask-guided DINOv2 pooling + per-phase 3D-CNN + dual head, decode_prediction()
├── dataset.py          # PyTorch Dataset over a metadata CSV (optionally filtered to a case_id list) + case folders
├── splits.py            # builds N stratified train/val/test folds over a metadata CSV, saved as JSON
├── train.py            # training loop for one fold, validates each epoch with the real challenge metric, tests on the held-out fold at the end
├── predict.py          # load a checkpoint + run inference (one case, or CLI over a fold's test split)
└── vendor/dinov2-with-registers-large/ # populated by scripts/vendor_dinov2.sh, not checked in

scripts/
├── vendor_dinov2.sh    # one-time: downloads a local HF Hub snapshot for offline backbone reconstruction
├── segment_livers.py   # one-time (per dataset): runs a pretrained nnUNetv2 model to produce per-case liver.nii.gz, needed by --transplant
└── diagnose_zextent_crop.py # diagnostic: measures how much of a lesion mask lesion_slice_indices's fixed-size window trims, by class

submission/
├── run.py              # challenge entry point (SUBMISSION_GUIDE.md contract)
├── metadata             # `command: ...` file required by the challenge harness
└── build.sh             # bundles nibabel + copies lirads_model/ into the submission dir

tests/
└── test_smoke.py        # CPU-only, no-internet, no-GPU pipeline shape/sanity check
```

## Setup

```bash
pip install -r requirements-dev.txt
```

This is the **training-time** environment (needs a GPU and internet). The challenge's own inference container already ships torch/numpy/pandas/scipy/scikit-learn — see `amplifai-codabench/SUBMISSION_GUIDE.md` for that image's exact contents and the compiled-package ABI warning before bundling anything extra.

## 1. Vendor the DINOv2 weights (once, needs internet)

Submission containers have no network access, so the backbone must be loadable offline. This downloads a local HF Hub snapshot (architecture + pretrained weights) into `lirads_model/vendor/dinov2-with-registers-large/`:

```bash
./scripts/vendor_dinov2.sh
```

## 2. Get the training data

Download and extract the AMPLIFAI batches from Hugging Face ([`UM-IHC-CA2i/AMPLIFAI`](https://huggingface.co/datasets/UM-IHC-CA2i/AMPLIFAI)):

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="UM-IHC-CA2i/AMPLIFAI", repo_type="dataset", filename="batches/batch_001.zip")
# ...repeat per batch, then unzip each into one shared directory, e.g. ./data/cases/
```

Also grab a metadata CSV covering the whole dataset, with `case_id`, `lirads_score`, `aphe`, `washout_venous`, `washout_delayed`, `capsule_venous`, `capsule_delayed` columns (among others) — `LiRadsCaseDataset` requires all of these. After extraction, cases should sit as `<data_root>/**/<case_id>/{ct,annotations}/...` — `lirads_model/dataset.py` searches recursively so batch subfolders are fine.

Note: not every case has all four phases — `preprocessing.py`/`model.py` already handle a phase being absent.

## 3. Sanity-check the pipeline (no GPU, no data, no internet)

```bash
python3 tests/test_smoke.py
```

Builds a synthetic case on the fly and a tiny random backbone stub, and runs the full preprocessing → pooling → dual-head pipeline, checking shapes and that the decoded label is valid. This is what to re-run after any pipeline change, before spending GPU time.

## 4. Build train/val/test splits

```bash
python -m lirads_model.splits \
  --metadata_csv ./data/metadata.csv \
  --n_folds 5 \
  --out ./data/splits.json
```

Generates a genuine `n_folds`-way `StratifiedKFold` (by `lirads_score`) partition: each case's `test`-fold membership is fixed, and the `n_folds` test sets are disjoint and together cover every case exactly once (unlike independent random draws). Within each fold's non-test remainder, `train`/`val` is a further stratified random split at `--val_frac` (default 0.15). All folds are written to one JSON keyed by fold index (`{"0": {"train": [...], "val": [...], "test": [...]}, "1": {...}, ...}`) — pick one fold index at training time via `--fold`.

## 4b. (optional) Segment livers, for lesion transplantation

Only needed if you plan to train with `--transplant` (see "Lesion transplantation" above):

```bash
python -m scripts.segment_livers \
  --metadata_csv ./data/metadata.csv \
  --data_root ./data/cases \
  --model_dir /path/to/nnUNet_results/Dataset003_Liver/nnUNetTrainer__nnUNetPlans__2d \
  --folds 0 1 2 3 4
```

Runs a pretrained nnUNetv2 model over every case's `VEN`-phase volume (`--phase` to change), saving `<case_dir>/annotations/liver.nii.gz`. `--model_dir` is an `nnUNet_results/<Dataset>/<Trainer>__<Plans>__<configuration>` folder from an nnUNetv2 training run; needs the `nnunetv2` package (not one of this repo's own dependencies — install separately). Cases that already have a `liver.nii.gz` are skipped unless `--overwrite` is passed.

## 5. Train

```bash
python -m lirads_model.train \
  --data_root ./data/cases \
  --metadata_csv ./data/metadata.csv \
  --splits_json ./data/splits.json \
  --fold 0 \
  --epochs 30 \
  --batch_size 4 \
  --out checkpoints/lirads_model.pt
```

- Downloads the pretrained backbone from the HuggingFace Hub on first run (needs internet).
- Everything except the DINOv2 backbone is optimized (head, 3D-CNN branch, clinical branch, missing-phase embedding); the backbone is always kept in eval mode. Add `--no-use_cnn` and/or `--no-use_clinical` to drop those branches (see "Approach" above).
- Category/ordinal losses are class-weighted by inverse frequency in the fold's `train` split.
- The train split is drawn via a `WeightedRandomSampler` (inverse-frequency over the full `lirads_score` label, not just the 4-way category) rather than plain random shuffling, so a rare special class like LR-TIV gets oversampled to roughly the same per-epoch exposure as a more common one (e.g. LR-M) that happens to share its category bucket — loss weighting alone can't fix a class that a short, randomly-shuffled epoch never happens to draw. `--no-balanced_sampling` reverts to plain `shuffle=True`.
- After each epoch, predictions on the fold's `val` split are scored with the actual `amplifai-codabench/evaluate.py` metric (QWK + SCR composite) plus a per-class precision/recall/F1/support breakdown, logged line by line; the checkpoint is overwritten whenever `final_score` improves.
- `--tta_views N` (default 0, off) enables test-time augmentation for both the per-epoch `val` scoring and the final `test`-split evaluation: whenever a case's deterministic pass gates to the ordinal category, `N` more forward passes on freshly augmented views of that case (same transforms as training, see `augmentation.py`) are averaged into the LR-1..LR-5 decision, aiming to steady predictions across LR-1 through LR-5 specifically. The category gate itself is always decided from the single deterministic pass. Non-zero `N` multiplies inference cost for ordinal-gated cases by `N+1`, every epoch if enabled during training, so it's off by default; `predict.py`/`submission/run.py` use `config.TTA_VIEWS` (default 4) at actual inference time regardless of this flag.
- `--transplant` (default off) enables lesion copy-paste augmentation for `config.TRANSPLANT_DONOR_LABELS` (LR-1/LR-2/LR-3) — see "Lesion transplantation" above. Requires `scripts/segment_livers.py` to have already produced a `liver.nii.gz` for at least some `config.NO_LESION_LABEL` cases in the train split; otherwise it's a no-op.
- The saved checkpoint (`model_state_dict`) contains the full model including backbone weights, so it's self-contained — no separate weights-export step needed.
- `--out`'s filename always gets `_fold{N}` inserted before the extension (e.g. `lirads_model.pt` → `lirads_model_fold0.pt`), whether left at its default or set explicitly, so checkpoints from different folds never collide.
- Once training finishes, the best checkpoint is reloaded and scored once more on the fold's held-out `test` split (never touched during training); predictions are saved alongside the checkpoint (`--test_predictions_out` to override the path), plus a confusion matrix image (`..._foldN_confusion_matrix.png`) and a per-class precision/recall/F1/support CSV (`..._foldN_per_class_metrics.csv`) saved next to it.

Useful flags: `--max_slices`, `--lr`, `--weight_decay`, `--num_workers`, `--device` (auto-detects CUDA).

Once you've looked at your training label distribution, update `FALLBACK_LABEL` in `lirads_model/config.py` (currently a placeholder `"LR-4"`) to whatever class is actually most common — it's what `run.py` predicts if a case throws during inference.

### Re-predicting on a fold's test split later

`train.py` already does this once at the end of training, but `predict.py` exposes it as a standalone CLI too — useful for re-scoring a saved checkpoint without retraining, or checking a checkpoint against a different fold's test split:

```bash
python -m lirads_model.predict \
  --checkpoint checkpoints/lirads_model_fold0.pt \
  --data_root ./data/cases \
  --metadata_csv ./data/metadata.csv \
  --splits_json ./data/splits.json \
  --fold 0 \
  --score
```

`--score` additionally prints the challenge metric against ground truth and saves a confusion matrix image plus a per-class precision/recall/F1/support CSV next to the predictions CSV (filenames always tagged with the fold, e.g. `..._fold0_confusion_matrix.png` / `..._fold0_per_class_metrics.csv`); omit it to just dump predictions. Defaults to `--backbone_source local` (the vendored offline snapshot); pass `--backbone_source hub` if you haven't vendored it yet.

`--checkpoint` accepts more than one path (`--checkpoint ckpt_a.pt ckpt_b.pt ckpt_c.pt`); with more than one, each model's decoded prediction is majority-voted per case (ties broken by whichever tied label the earliest-listed model predicted).

## 6. Evaluating predictions locally

```bash
python amplifai-codabench/evaluate.py \
  --ground_truth path/to/gt.csv \
  --predictions path/to/pred.csv \
  --no-bootstrap
```
