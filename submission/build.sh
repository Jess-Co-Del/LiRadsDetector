#!/usr/bin/env bash
# Assembles the submission directory:
#   - copies lirads_model/ (including the vendored dinov2 snapshot) in from
#     the project root
#   - bundles nibabel + transformers (+ light deps), not present in the
#     challenge's base image (codalab/codalab-legacy:gpu310) — torch/numpy/
#     pandas/scipy are already there, don't rebundle them (see
#     SUBMISSION_GUIDE.md's ABI warning)
#   - warns if the vendored dinov2 snapshot or trained checkpoint(s) are missing
#
# Run once before zipping. For ABI safety, run inside (or matching) the
# target image.
#
# Usage:
#   ./build.sh
#   zip -r submission.zip run.py metadata lirads_model/ model/ packages/
#
# model/ may hold one or more .pt checkpoints -- run.py majority-votes across
# all of them when there's more than one.

set -e
cd "$(dirname "$0")"

rm -rf lirads_model
cp -r ../lirads_model .
find lirads_model -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

pip install --target=packages --no-deps -q transformers==4.48 tokenizers==0.21 huggingface_hub==0.24.0
pip install --target=packages --no-deps -q \
    nibabel packaging importlib-resources typing-extensions filelock pyyaml regex safetensors tqdm typer-slim
pip install --target=packages --no-deps -q  batchgeneratorsv2==0.3.2 batchgenerators==0.25

if [ ! -d "lirads_model/vendor/dinov2-with-registers-large" ]; then
    echo "WARNING: lirads_model/vendor/dinov2-with-registers-large is missing."
    echo "  Run ../scripts/vendor_dinov2.sh once (with internet) before building."
fi

n_checkpoints=$(find model -maxdepth 1 -name '*1.pt' 2>/dev/null | wc -l)
if [ "$n_checkpoints" -eq 0 ]; then
    echo "WARNING: no .pt checkpoints found in model/."
    echo "  Copy at least one trained checkpoint there (see lirads_model/train.py --out)."
else
    echo "Found $n_checkpoints checkpoint(s) in model/."
fi

echo "packages/ and lirads_model/ ready."
echo "Zip with: zip -r submission.zip run.py metadata lirads_model/ model/ packages/"
