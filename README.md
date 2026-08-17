# LiRadsDetector

A LI-RADS classifier for the [AMPLIFAI challenge](https://um-ihc-ca2i.github.io/amplifai-challenge/index.html): given a multi-phase CT scan (ART/VEN/DEL/DRY) and a target lesion segmentation, predicts one of `LR-1, LR-2, LR-3, LR-4, LR-5, LR-M, LR-TIV`.

See [`amplifai-codabench/`](https://github.com/UM-IHC-CA2i/amplifai-codabench.git) for the challenge's own submission spec (`README.md`, `SUBMISSION_GUIDE.md`, `evaluate.py`) — that folder is reference material, not part of this codebase.

## Approach

A frozen DINOv2 ViT-L/14 (register-token variant, [`facebook/dinov2-with-registers-large`](https://huggingface.co/facebook/dinov2-with-registers-large), loaded via `transformers.AutoModel`) is used as a slice-wise feature extractor:

1. For each of the four CT phases, up to **32 axial slices** are sampled evenly across the lesion's z-extent (all of them if the lesion spans fewer than 32).
2. Each slice is cropped square around the lesion (with margin), resized to 224×224, HU-windowed and replicated to pseudo-RGB.
3. Every slice is run through the frozen backbone, producing a 16×16 grid of patch tokens + a CLS token.
4. The patch-token grid is pooled using the lesion mask (downsampled to the same 16×16 grid), so only lesion-covering patches contribute — a mask-guided pooling head, not a predicted segmentation.
5. Slices are combined per phase, weighted by lesion area in that slice; missing phases get a learned placeholder embedding instead of breaking the pipeline.
6. The four phases' `[masked-pooled patch feature | CLS feature]` vectors are concatenated and fed to a small MLP with two heads:
   - a 3-way head (ordinal / LR-M / LR-TIV), which drives the **Special Category Recognition** metric,
   - a 5-way ordinal head (LR-1..LR-5), used only when the case is ordinal, which drives **Adjusted QWK**.

   This split mirrors exactly how `amplifai-codabench/evaluate.py` scores submissions.

Only the head is trained — the DINOv2 backbone stays frozen throughout.

## Layout

```
lirads_model/
├── config.py         # labels, slice cap, image/patch sizes, CT windowing, hub names
├── preprocessing.py  # NIfTI loading, lesion slice sampling, crop/resize/window, mask->patch-grid
├── backbone.py        # frozen DINOv2 wrapper (transformers.AutoModel; local/offline or hub/pretrained)
├── model.py           # LiRadsNet: mask-guided pooling + dual head, decode_prediction()
├── dataset.py          # PyTorch Dataset over a metadata CSV (optionally filtered to a case_id list) + case folders
├── splits.py            # builds N stratified train/val/test folds over a metadata CSV, saved as JSON
├── train.py            # training loop for one fold, validates each epoch with the real challenge metric, tests on the held-out fold at the end
├── predict.py          # load a checkpoint + run inference (one case, or CLI over a fold's test split)
└── vendor/dinov2-with-registers-large/ # populated by scripts/vendor_dinov2.sh, not checked in

scripts/
└── vendor_dinov2.sh    # one-time: downloads a local HF Hub snapshot for offline backbone reconstruction

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

Also grab a metadata CSV (case_id + `lirads_score` columns, among others) covering the whole dataset. After extraction, cases should sit as `<data_root>/**/<case_id>/{ct,annotations}/...` — `lirads_model/dataset.py` searches recursively so batch subfolders are fine.

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

Generates `n_folds` independent stratified-by-`lirads_score` random splits at 70/15/15 train/val/test (fractions configurable via `--train_frac`/`--val_frac`/`--test_frac`), and writes them all to one JSON keyed by fold index (`{"0": {"train": [...], "val": [...], "test": [...]}, "1": {...}, ...}`). Folds are independent draws, not a non-overlapping k-fold partition, so a case's test-set membership can repeat or vary across folds — pick one fold index at training time via `--fold`.

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
- Only head parameters (+ the missing-phase embedding) are optimized; the backbone is always kept in eval mode.
- Category/ordinal losses are class-weighted by inverse frequency in the fold's `train` split.
- After each epoch, predictions on the fold's `val` split are scored with the actual `amplifai-codabench/evaluate.py` metric (QWK + SCR composite); the checkpoint is overwritten whenever `final_score` improves.
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

## 6. Build the submission zip

```bash
cp checkpoints/lirads_model.pt submission/model/lirads_model.pt   # mkdir -p submission/model first
cd submission
./build.sh
zip -r submission.zip run.py metadata lirads_model/ model/ packages/
```

`build.sh` copies the top-level `lirads_model/` package (including the vendored DINOv2 snapshot) into `submission/` and bundles `nibabel` and `transformers` (+ their light deps) into `packages/`, warning if the vendored snapshot or the checkpoint are missing.

## 7. Test offline before uploading

Network is disabled in the real environment, so test the exact zip inside the same Docker image first — see the full recipe in `amplifai-codabench/SUBMISSION_GUIDE.md` ("Test your submission locally before uploading"):

```bash
mkdir -p /tmp/test_input /tmp/test_output
echo "case_id" > /tmp/test_input/sample_cases.csv
echo "CASE00001" >> /tmp/test_input/sample_cases.csv

unzip -o submission.zip -d /tmp/test_submission

docker run --rm --network none --gpus all \
  -v /tmp/test_submission:/app/ingested_program:ro \
  -v /path/to/data:/app/data:ro \
  -v /tmp/test_input:/app/input_data:ro \
  -v /tmp/test_output:/app/output \
  codalab/codalab-legacy:gpu310 \
  python3 /app/ingested_program/run.py /app/input_data /app/output

cat /tmp/test_output/predictions.csv
```

Then upload `submission.zip` on the [AMPLIFAI Codabench page](https://www.codabench.org/competitions/14290/).

## Evaluating predictions locally

```bash
python amplifai-codabench/evaluate.py \
  --ground_truth path/to/gt.csv \
  --predictions path/to/pred.csv \
  --no-bootstrap
```
