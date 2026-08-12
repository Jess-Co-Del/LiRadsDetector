# LiRadsDetector

A LI-RADS classifier for the [AMPLIFAI challenge](https://um-ihc-ca2i.github.io/amplifai-challenge/index.html): given a multi-phase CT scan (ART/VEN/DEL/DRY) and a target lesion segmentation, predicts one of `LR-1, LR-2, LR-3, LR-4, LR-5, LR-M, LR-TIV`.

See [`amplifai-codabench/`](amplifai-codabench/) for the challenge's own submission spec (`README.md`, `SUBMISSION_GUIDE.md`, `evaluate.py`) — that folder is reference material, not part of this codebase.

## Approach

A frozen DINOv2 ViT-L/14 (register-token variant, `dinov2_vitl14_reg`) is used as a slice-wise feature extractor:

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
├── backbone.py        # frozen DINOv2 wrapper (torch.hub; local/offline or github/pretrained)
├── model.py           # LiRadsNet: mask-guided pooling + dual head, decode_prediction()
├── dataset.py          # PyTorch Dataset over train/val metadata CSVs + case folders
├── train.py            # training loop, validates each epoch with the real challenge metric
├── predict.py          # load a checkpoint + run inference on one case
└── vendor/dinov2_repo/ # populated by scripts/vendor_dinov2.sh, not checked in

scripts/
└── vendor_dinov2.sh    # one-time: clones dinov2 source for offline backbone reconstruction

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

## 1. Vendor the DINOv2 source (once, needs internet)

Submission containers have no network access, so the backbone architecture must be reconstructible offline. This clones the model-definition source (no weights) into `lirads_model/vendor/dinov2_repo/`:

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

Also grab `train_metadata.csv` and `val_metadata.csv` (case_id + `lirads_score` columns, among others). After extraction, cases should sit as `<data_root>/**/<case_id>/{ct,annotations}/...` — `lirads_model/dataset.py` searches recursively so batch subfolders are fine.

Note: not every case has all four phases — `preprocessing.py`/`model.py` already handle a phase being absent.

## 3. Sanity-check the pipeline (no GPU, no data, no internet)

```bash
python3 tests/test_smoke.py
```

Builds a synthetic case on the fly and a tiny random backbone stub, and runs the full preprocessing → pooling → dual-head pipeline, checking shapes and that the decoded label is valid. This is what to re-run after any pipeline change, before spending GPU time.

## 4. Train

```bash
python -m lirads_model.train \
  --data_root ./data/cases \
  --train_csv ./data/train_metadata.csv \
  --val_csv   ./data/val_metadata.csv \
  --epochs 30 \
  --batch_size 4 \
  --out checkpoints/lirads_model.pt
```

- Downloads the pretrained backbone via `torch.hub` on first run (needs internet).
- Only head parameters (+ the missing-phase embedding) are optimized; the backbone is always kept in eval mode.
- Category/ordinal losses are class-weighted by inverse frequency in `train_csv`.
- After each epoch, predictions on `val_csv` are scored with the actual `amplifai-codabench/evaluate.py` metric (QWK + SCR composite); the checkpoint is overwritten whenever `final_score` improves.
- The saved checkpoint (`model_state_dict`) contains the full model including backbone weights, so it's self-contained — no separate weights-export step needed.

Useful flags: `--max_slices`, `--lr`, `--weight_decay`, `--num_workers`, `--device` (auto-detects CUDA).

Once you've looked at your training label distribution, update `FALLBACK_LABEL` in `lirads_model/config.py` (currently a placeholder `"LR-4"`) to whatever class is actually most common — it's what `run.py` predicts if a case throws during inference.

## 5. Build the submission zip

```bash
cp checkpoints/lirads_model.pt submission/model/lirads_model.pt   # mkdir -p submission/model first
cd submission
./build.sh
zip -r submission.zip run.py metadata lirads_model/ model/ packages/
```

`build.sh` copies the top-level `lirads_model/` package (including the vendored DINOv2 source) into `submission/` and bundles `nibabel` into `packages/`, warning if the vendored repo or the checkpoint are missing.

## 6. Test offline before uploading

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
