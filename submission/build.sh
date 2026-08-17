#!/usr/bin/env bash
# Assembles the submission directory:
#   - copies lirads_model/ (including the vendored dinov2 snapshot) in from
#     the project root
#   - bundles nibabel + transformers (+ light deps), not present in the
#     challenge's base image (codalab/codalab-legacy:gpu310) — torch/numpy/
#     pandas/scipy are already there, don't rebundle them (see
#     SUBMISSION_GUIDE.md's ABI warning)
#   - warns if the vendored dinov2 snapshot or trained checkpoint are missing
#
# Run once before zipping. For ABI safety, run inside (or matching) the
# target image.
#
# Usage:
#   ./build.sh
#   zip -r submission.zip run.py metadata lirads_model/ model/ packages/

set -e
cd "$(dirname "$0")"

rm -rf lirads_model
cp -r ../lirads_model .
find lirads_model -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

pip install --target=packages --no-deps -q transformers==4.48 tokenizers==0.21 huggingface_hub==0.24.0
pip install --target=packages --no-deps -q nibabel packaging importlib-resources typing-extensions
pip install --target=packages --no-deps -q \
    filelock pyyaml regex safetensors tqdm typer-slim

if [ ! -d "lirads_model/vendor/dinov2-with-registers-large" ]; then
    echo "WARNING: lirads_model/vendor/dinov2-with-registers-large is missing."
    echo "  Run ../scripts/vendor_dinov2.sh once (with internet) before building."
fi

if [ ! -f "checkpoints/lirads_model.pt" ]; then
    echo "WARNING: model/lirads_model.pt is missing."
    echo "  Copy your trained checkpoint there (see lirads_model/train.py --out)."
fi

echo "packages/ and lirads_model/ ready."
echo "Zip with: zip -r submission.zip run.py metadata lirads_model/ model/ packages/"
