"""Shared inference logic used by both local evaluation and submission/run.py."""

import torch

from . import config, preprocessing
from .backbone import Dinov2SliceEncoder
from .model import LiRadsNet, decode_prediction


def load_model(checkpoint_path: str, device: torch.device, backbone_source: str = "local") -> LiRadsNet:
    """backbone_source="local": no network (submission container). "hub":
    re-downloads the pretrained backbone from the HuggingFace Hub before
    loading our trained weights on top (useful for local dev without a
    vendored snapshot)."""
    if backbone_source == "local":
        backbone = Dinov2SliceEncoder.from_local()
    else:
        backbone = Dinov2SliceEncoder.from_pretrained()

    model = LiRadsNet(backbone).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_case(
    model: LiRadsNet,
    case_dir: str,
    case_id: str,
    device: torch.device,
    max_slices: int = config.MAX_SLICES_PER_CASE,
) -> str:
    phase_paths = preprocessing.find_case_phase_paths(case_dir, case_id)
    mask_path = preprocessing.find_case_mask_path(case_dir)
    phase_data = preprocessing.build_case_tensors(phase_paths, mask_path, max_slices)

    logits_cat, logits_ord = model([phase_data])
    return decode_prediction(logits_cat[0].cpu(), logits_ord[0].cpu())
