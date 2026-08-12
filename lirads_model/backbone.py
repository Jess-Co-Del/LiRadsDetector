"""Frozen DINOv2 ViT-L/14 (register-token variant) slice encoder, via torch.hub.

`source="github"` downloads code+pretrained weights (needs internet) and is
used for training / weight export. `source="local"` reconstructs the same
architecture from a vendored copy of the repo (see
scripts/vendor_dinov2.sh) with random init — the real weights are loaded
afterwards from our own checkpoint via LiRadsNet.load_state_dict(), so no
network call is needed. This is what submission/run.py uses at inference,
since the challenge container has no outbound network access.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


def build_dinov2_backbone(source: str = "github") -> nn.Module:
    pretrained = source == "github"
    repo = config.DINOV2_HUB_REPO if source == "github" else config.DINOV2_LOCAL_REPO
    backbone = torch.hub.load(repo, config.DINOV2_HUB_MODEL, source=source, pretrained=pretrained)
    backbone.head = nn.Identity()
    return backbone


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
        return cls(build_dinov2_backbone(source="github"))

    @classmethod
    def from_local(cls) -> "Dinov2SliceEncoder":
        return cls(build_dinov2_backbone(source="local"))

    def interpolate_pos_encoding(
        self,
        pos_embed: torch.Tensor,   # (1, 1 + N, D)  -- includes CLS slot at index 0
        x: torch.Tensor,           # (B, 1 + n, D)  -- current tokens, used only for shape check
        w: int,                    # input image width in pixels
        h: int,                    # input image height in pixels
        patch_size: int,
        num_register_tokens: int = 0,
    ) -> torch.Tensor:
        n_patches_now = x.shape[1] - 1 - num_register_tokens
        N = pos_embed.shape[1] - 1
        dim = pos_embed.shape[-1]

        if n_patches_now == N and w == h:
            return pos_embed

        class_pos_embed = pos_embed[:, :1]
        patch_pos_embed = pos_embed[:, 1:]

        sqrt_N = int(math.sqrt(N))
        assert sqrt_N * sqrt_N == N, "pretrained pos_embed grid must be square"

        new_h, new_w = h // patch_size, w // patch_size

        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_N, sqrt_N, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed, size=(new_h, new_w),
            mode="bicubic", align_corners=False, antialias=True,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, new_h * new_w, dim)

        return torch.cat([class_pos_embed, patch_pos_embed], dim=1)

    def prepare_tokens(self, x):
        B, _, H, W = x.shape
        x = self.backbone.patch_embed(x)                    # (B, n_patches, D)
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)       # (B, 1+n_patches, D)

        pos = self.interpolate_pos_encoding(self.backbone.pos_embed, x, W, H, self.backbone.patch_size)
        x = x + pos

        if self.backbone.num_register_tokens:
            reg = self.backbone.register_tokens.expand(B, -1, -1)
            x = torch.cat([x[:, :1], reg, x[:, 1:]], dim=1)  # [CLS] + [reg] + [patches]

        return x

    @torch.no_grad()
    def forward(self, inputs):
        # prepare_tokens() applies our own resolution-aware pos-embed
        # interpolation (needed since target_size != the 224x224 the
        # checkpoint was pretrained at), so we run the transformer blocks
        # and final norm by hand here instead of calling the backbone's
        # forward_features(), which would re-tokenize from raw pixels
        # using its own interpolation and discard ours.
        x = self.prepare_tokens(inputs)
        for blk in self.backbone.blocks:
            x = blk(x)
        x_norm = self.backbone.norm(x)
        return x_norm[:, 1:], x_norm[:, self.backbone.num_register_tokens:]

    @torch.no_grad()
    def nointerp_forward(self, pixel_values: torch.Tensor):
        """pixel_values: (S, 3, H, W) -> patch_tokens (S, N, D), cls_token (S, D)."""
        feats = self.backbone.forward_features(pixel_values)
        cls_token = feats["x_norm_clstoken"]
        patch_tokens = feats["x_norm_patchtokens"]
        return patch_tokens, cls_token
