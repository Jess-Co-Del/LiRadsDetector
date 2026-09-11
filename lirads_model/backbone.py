"""
Frozen DINOv2 ViT-L/14 (with-registers variant) slice encoder, loaded via
HuggingFace `transformers`
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from . import config


def build_dinov2_backbone(source: str = "hub") -> nn.Module:
    if source == "hub":
        return AutoModel.from_pretrained(config.DINOV2_MODEL_ID)
    if source == "local":
        return AutoModel.from_pretrained(config.DINOV2_LOCAL_DIR, local_files_only=True)
    raise ValueError(f"unknown source: {source!r}")


def _pad_to_patch_multiple(volume: torch.Tensor, out_size: int) -> torch.Tensor:
    """Zero-ish (constant 0.0 -- "at or below WINDOW_LOW", the same
    background convention preprocessing.py's own windowed-normalization
    uses) pad a batch of square images (S, H, W) up to out_size, split
    evenly on both sides (extra pixel on the bottom/right if odd). Skips
    the pad (and its tensor copy) entirely when H/W already equal out_size
    -- true today (config.PADDED_SIZE == config.IMG_SIZE), but this stays
    correct if IMG_SIZE/PATCH_SIZE ever stop being an exact multiple."""
    pad_h = out_size - volume.shape[-2]
    pad_w = out_size - volume.shape[-1]
    if pad_h == 0 and pad_w == 0:
        return volume
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    return F.pad(volume, (left, right, top, bottom), mode="constant", value=0.0)


class Dinov2SliceEncoder(nn.Module):
    """Wraps a frozen DINOv2 backbone. Forward takes a batch of single-
    channel, windowed-normalized (to [0,1]) lesion-crop slices -- the same
    `volume` tensor preprocessing.prepare_phase_tensors also hands directly
    to the per-phase 3D CNN (model.PhaseVolumeCNN) -- and returns
    (patch_tokens, cls_token). Padding to DINOv2's patch-multiple input
    size, replicating to pseudo-RGB, and ImageNet-normalizing all happen
    here rather than in preprocessing, since they're specific to what this
    particular pretrained backbone expects, not a property of the case data
    itself; the 3D-CNN branch needs none of it. This is a relocation, not a
    behavior change: the actual values reaching the wrapped HF model's
    patch_embed are identical to before, so it doesn't affect any existing
    or future checkpoint (the backbone is frozen and never touched by the
    optimizer either way -- see train.py's trainable_params)."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        # persistent=False: these are a fixed constant, not learned state --
        # excluding them from state_dict() keeps them out of
        # load_state_dict()'s strict key-matching entirely, so adding them
        # here can't break loading any checkpoint saved before this existed.
        self.register_buffer(
            "_imagenet_mean", torch.tensor(config.IMAGENET_MEAN, dtype=torch.float32)[:, None, None], persistent=False,
        )
        self.register_buffer(
            "_imagenet_std", torch.tensor(config.IMAGENET_STD, dtype=torch.float32)[:, None, None], persistent=False,
        )

    @classmethod
    def from_pretrained(cls) -> "Dinov2SliceEncoder":
        return cls(build_dinov2_backbone(source="hub"))

    @classmethod
    def from_local(cls) -> "Dinov2SliceEncoder":
        return cls(build_dinov2_backbone(source="local"))

    @torch.no_grad()
    def forward(self, volume: torch.Tensor):
        padded = _pad_to_patch_multiple(volume, config.PADDED_SIZE)  # (S,H,W)
        pixel_values = padded.unsqueeze(1).repeat(1, 3, 1, 1)  # (S,3,H,W) pseudo-RGB
        pixel_values = (pixel_values - self._imagenet_mean) / self._imagenet_std

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
