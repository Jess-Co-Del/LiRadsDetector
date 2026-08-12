#!/usr/bin/env bash
# Vendors the facebookresearch/dinov2 repo source (architecture code only —
# no weights) so the backbone can be reconstructed offline via
# torch.hub.load(..., source="local") at inference time, when the challenge
# container has no network access. Run this once, with internet, before
# training or building the submission zip.
#
# Usage: ./scripts/vendor_dinov2.sh

set -e
cd "$(dirname "$0")/.."

DEST=lirads_model/vendor/dinov2_repo
mkdir -p lirads_model/vendor
rm -rf "$DEST"
git clone --depth 1 https://github.com/facebookresearch/dinov2.git "$DEST"
rm -rf "$DEST/.git"

echo "Vendored dinov2 repo at $DEST"
echo "Verify with: python3 -c \"from lirads_model.backbone import build_dinov2_backbone; build_dinov2_backbone(source='local')\""
