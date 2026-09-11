"""
Frozen DINOv2 ViT-L/14 (with-registers variant) slice encoder, loaded via
HuggingFace `transformers`
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
        #
        # fp16 autocast is safe here: frozen backbone, no_grad, inference
        # only. Cast back to float32 immediately after so every downstream
        # consumer (masked pooling, the 3D-CNN branch, the heads) keeps
        # seeing the same dtype as before this was added.
        with torch.autocast(device_type=pixel_values.device.type, dtype=torch.float16, enabled=pixel_values.is_cuda):
            last_hidden_state = self.backbone(pixel_values=pixel_values).last_hidden_state
        last_hidden_state = last_hidden_state.float()
        num_register_tokens = self.backbone.config.num_register_tokens if hasattr(self.backbone.config, 'num_register_tokens') else 0
        # token layout is [CLS, reg_1..reg_R, patch_1..patch_N]
        return last_hidden_state[:, 1 + num_register_tokens :], last_hidden_state[:, 0]
