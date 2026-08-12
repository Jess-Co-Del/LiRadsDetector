"""CPU-only, no-internet smoke test for the full LiRadsNet pipeline.

Builds a synthetic case (random NIfTI volumes + a lesion mask, one phase
deliberately missing) on disk, runs it through preprocessing and a tiny
random backbone stub (same forward_features interface as the real torch.hub
DINOv2 model, but small enough to run instantly on CPU with no download),
and checks that shapes and the final decoded label are sane end-to-end.
"""

import os
import shutil
import sys
import tempfile

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lirads_model import config, preprocessing
from lirads_model.backbone import Dinov2SliceEncoder
from lirads_model.model import LiRadsNet, decode_prediction


class TinyBackboneStub(nn.Module):
    """Mimics the real torch.hub DINOv2 model's forward_features() interface
    (x_norm_clstoken / x_norm_patchtokens) with a single conv layer, so the
    smoke test doesn't need internet or minutes of CPU time for ViT-L."""

    def __init__(self, embed_dim=config.EMBED_DIM, patch_size=config.PATCH_SIZE):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_param = nn.Parameter(torch.zeros(1, embed_dim))

    def forward_features(self, x):
        feat = self.patch_embed(x)
        patch_tokens = feat.flatten(2).transpose(1, 2)
        cls_token = self.cls_param.expand(x.shape[0], -1)
        return {"x_norm_clstoken": cls_token, "x_norm_patchtokens": patch_tokens}


def _make_synthetic_case(case_dir: str, case_id: str, shape=(64, 64, 40)) -> None:
    ct_dir = os.path.join(case_dir, "ct")
    ann_dir = os.path.join(case_dir, "annotations")
    os.makedirs(ct_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    rng = np.random.default_rng(0)
    affine = np.eye(4)

    mask = np.zeros(shape, dtype=np.float32)
    mask[20:36, 20:36, 12:26] = 1.0  # synthetic lesion spanning 14 slices
    nib.save(nib.Nifti1Image(mask, affine), os.path.join(ann_dir, "lesion.nii.gz"))

    for phase in config.PHASE_NAMES:
        if phase == "DRY":
            continue  # deliberately missing, to exercise that code path
        vol = rng.normal(50, 100, size=shape).astype(np.float32)
        nib.save(nib.Nifti1Image(vol, affine), os.path.join(ct_dir, f"{case_id}_{phase}.nii.gz"))


def test_pipeline_smoke() -> None:
    tmp = tempfile.mkdtemp()
    try:
        case_id = "CASE00001"
        case_dir = os.path.join(tmp, case_id)
        _make_synthetic_case(case_dir, case_id)

        phase_paths = preprocessing.find_case_phase_paths(case_dir, case_id)
        mask_path = preprocessing.find_case_mask_path(case_dir)
        phase_data = preprocessing.build_case_tensors(phase_paths, mask_path, max_slices=8)

        assert phase_data["DRY"] is None
        for phase in ["ART", "VEN", "DEL"]:
            pixel_values, mask_grids, slice_weights = phase_data[phase]
            assert pixel_values.shape[1:] == (3, config.IMG_SIZE, config.IMG_SIZE)
            assert pixel_values.shape[0] <= 8
            assert mask_grids.shape[1:] == (config.GRID_SIZE, config.GRID_SIZE)
            assert slice_weights.shape[0] == pixel_values.shape[0]

        backbone = Dinov2SliceEncoder(TinyBackboneStub())
        model = LiRadsNet(backbone)
        model.eval()

        with torch.no_grad():
            logits_cat, logits_ord = model([phase_data])

        assert logits_cat.shape == (1, len(config.CAT_NAMES))
        assert logits_ord.shape == (1, len(config.ORDINAL_LABELS))

        label = decode_prediction(logits_cat[0], logits_ord[0])
        assert label in config.VALID_LABELS

        print(f"smoke test OK -- predicted {label!r} from a synthetic 3-phase case")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_pipeline_smoke()
