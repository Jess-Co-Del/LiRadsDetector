#!/usr/bin/env bash
# Vendors a local snapshot of the facebook/dinov2-with-registers-large model
# (architecture + pretrained weights) from the HuggingFace Hub, so the
# backbone can be reconstructed offline via
# transformers.AutoModel.from_pretrained(..., local_files_only=True) at
# inference time, when the challenge container has no network access. Run
# this once, with internet, before training or building the submission zip.
#
# Usage: ./scripts/vendor_dinov2.sh

set -e
cd "$(dirname "$0")/.."

DEST=lirads_model/vendor/dinov2-with-registers-large
mkdir -p lirads_model/vendor
rm -rf "$DEST"

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/dinov2-with-registers-large', local_dir='$DEST')
"

echo "Vendored dinov2-with-registers-large at $DEST"
echo "Verify with: python3 -c \"from lirads_model.backbone import build_dinov2_backbone; build_dinov2_backbone(source='local')\""
