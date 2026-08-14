"""Frozen DINOv2 ViT-L/14 (with-registers variant) slice encoder, loaded via
HuggingFace `transformers`.

`source="hub"` downloads weights from the HuggingFace Hub (needs internet)
and is used for training / weight export. `source="local"` loads from a
vendored local snapshot (see scripts/vendor_dinov2.sh) via
`local_files_only=True` — the real weights are loaded afterwards from our own
checkpoint via LiRadsNet.load_state_dict(), so no network call is needed.
This is what submission/run.py uses at inference, since the challenge
container has no outbound network access.
"""
import torch
import torch.nn as nn
from transformers import AutoModel

from . import config


def build_dinov2_backbone(source: str = "hub") -> nn.Module:
    if source == "hub":
        return AutoModel.from_pretrained(config.DINOV2_MODEL_ID)
    if source == "local":
        return AutoModel.from_pretrained(config.DINOV2_LOCAL_DIR, local_files_only=True)
    raise ValueError(f"unknown source: {source!r}")


class Dinov2SliceEncoder(nn.Module):
    """Wraps a frozen DINOv2 backbone. Forward takes a batch of 2D pseudo-RGB
    slices and returns (patch_tokens, cls_token)."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    @classmethod
    def from_pretrained(cls) -> "Dinov2SliceEncoder":
        return cls(build_dinov2_backbone(source="hub"))

    @classmethod
    def from_local(cls) -> "Dinov2SliceEncoder":
        return cls(build_dinov2_backbone(source="local"))

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor):
        # transformers' Dinov2WithRegistersModel interpolates the position
        # embeddings to the input resolution internally, so no custom
        # resolution-aware interpolation is needed here.
        last_hidden_state = self.backbone(pixel_values=pixel_values).last_hidden_state
        num_register_tokens = self.backbone.config.num_register_tokens
        # token layout is [CLS, reg_1..reg_R, patch_1..patch_N]
        return last_hidden_state[:, 1 + num_register_tokens :], last_hidden_state[:, 0]
