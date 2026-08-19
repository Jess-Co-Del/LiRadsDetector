"""CPU-only, no-internet smoke test for the full LiRadsNet pipeline.

Builds a synthetic case (random NIfTI volumes + a lesion mask) on disk, runs
it through preprocessing and a tiny random backbone stub (same
callable/output interface as the real transformers AutoModel-loaded DINOv2
model, but small enough to run instantly on CPU with no download), and
checks that shapes and the final decoded label are sane end-to-end.
"""

import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lirads_model import config, preprocessing
from lirads_model.backbone import Dinov2SliceEncoder
from lirads_model.model import LiRadsNet, decode_prediction


class TinyBackboneStub(nn.Module):
    """Mimics the real transformers Dinov2WithRegistersModel's callable
    interface -- forward(pixel_values=...) returning an object with
    .last_hidden_state, plus a .config.num_register_tokens -- used by
    Dinov2SliceEncoder.forward(). Small enough to run instantly on CPU with
    no download. Uses a nonzero register-token count so the CLS/register/
    patch token-splitting logic is actually exercised."""

    def __init__(self, embed_dim=config.EMBED_DIM, patch_size=config.PATCH_SIZE, num_register_tokens=2):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.config = SimpleNamespace(num_register_tokens=num_register_tokens)

    def forward(self, pixel_values):
        patch_tokens = self.proj(pixel_values).flatten(2).transpose(1, 2)  # (B, n_patches, D)
        cls_tokens = self.cls_token.expand(pixel_values.shape[0], -1, -1)
        reg_tokens = self.register_tokens.expand(pixel_values.shape[0], -1, -1)
        tokens = torch.cat([cls_tokens, reg_tokens, patch_tokens], dim=1)
        return SimpleNamespace(last_hidden_state=self.norm(tokens))


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

        for phase in config.PHASE_NAMES:
            pixel_values, mask_grids, slice_weights, volume = phase_data[phase]
            assert pixel_values.shape[1:] == (3, config.PADDED_SIZE, config.PADDED_SIZE)
            assert pixel_values.shape[0] == 8  # lesion_slice_indices always returns exactly max_slices
            assert mask_grids.shape[1:] == (config.GRID_SIZE, config.GRID_SIZE)
            assert slice_weights.shape[0] == pixel_values.shape[0]
            assert volume.shape == (pixel_values.shape[0], config.IMG_SIZE, config.IMG_SIZE)

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
