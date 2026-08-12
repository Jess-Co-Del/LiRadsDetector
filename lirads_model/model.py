"""LiRadsNet: mask-guided-pooled DINOv2 features -> dual classification head.

Per phase: each of up to MAX_SLICES_PER_CASE slices is embedded by the frozen
DINOv2 backbone into a (grid, grid, embed_dim) patch-token map + a CLS token.
The patch-token map is average-pooled using the lesion's downsampled mask
(so only lesion-covering patches contribute), then slices are combined with
weights proportional to lesion area in that slice. The four phases'
[masked-pooled patch feature | CLS feature] vectors are concatenated and fed
to a small MLP with two heads:
  - a 3-way head (ordinal / LR-M / LR-TIV) that drives Special Category
    Recognition,
  - a 5-way ordinal head (LR-1..LR-5) used only when the case is ordinal,
    that drives Adjusted QWK.
This mirrors the exact split evaluate.py uses to score submissions.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from . import config
from .backbone import Dinov2SliceEncoder

PhaseData = Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class LiRadsNet(nn.Module):
    def __init__(
        self,
        backbone: Dinov2SliceEncoder,
        embed_dim: int = config.EMBED_DIM,
        grid_size: int = config.GRID_SIZE,
        hidden1: int = config.HEAD_HIDDEN_1,
        hidden2: int = config.HEAD_HIDDEN_2,
        dropout: float = config.HEAD_DROPOUT,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.phase_names = config.PHASE_NAMES

        phase_feat_dim = embed_dim * 2  # masked-pooled patch feat + CLS feat
        self.missing_phase_embed = nn.Parameter(torch.randn(len(self.phase_names), phase_feat_dim) * 0.02)

        in_dim = phase_feat_dim * len(self.phase_names)
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
        )
        self.cat_head = nn.Linear(hidden2, len(config.CAT_NAMES))
        self.ord_head = nn.Linear(hidden2, len(config.ORDINAL_LABELS))

    def encode_phase(self, pixel_values: torch.Tensor, mask_grids: torch.Tensor, slice_weights: torch.Tensor) -> torch.Tensor:
        patch_tokens, cls_token = self.backbone(pixel_values)  # (S,N,D), (S,D)
        S = patch_tokens.shape[0]
        patch_tokens = patch_tokens.view(S, self.grid_size, self.grid_size, self.embed_dim)

        mask_sum = mask_grids.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        mask_w = mask_grids / mask_sum
        masked_pooled = (patch_tokens * mask_w.unsqueeze(-1)).sum(dim=(1, 2))  # (S,D)

        slice_w = slice_weights / slice_weights.sum().clamp_min(1e-6)  # (S,)
        phase_masked = (masked_pooled * slice_w.unsqueeze(-1)).sum(dim=0)  # (D,)
        phase_cls = (cls_token * slice_w.unsqueeze(-1)).sum(dim=0)  # (D,)
        return torch.cat([phase_masked, phase_cls], dim=0)  # (2D,)

    def encode_case(self, phase_data: Dict[str, PhaseData]) -> torch.Tensor:
        device = self.missing_phase_embed.device
        feats = []
        for i, phase in enumerate(self.phase_names):
            data = phase_data.get(phase)
            if data is None:
                feats.append(self.missing_phase_embed[i])
            else:
                pixel_values, mask_grids, slice_weights = data
                feats.append(
                    self.encode_phase(
                        pixel_values.to(device), mask_grids.to(device), slice_weights.to(device)
                    )
                )
        return torch.cat(feats, dim=0)  # (phase_feat_dim * n_phases,)

    def forward(self, batch_phase_data: List[Dict[str, PhaseData]]) -> Tuple[torch.Tensor, torch.Tensor]:
        case_feats = torch.stack([self.encode_case(pd) for pd in batch_phase_data], dim=0)
        h = self.head(case_feats)
        return self.cat_head(h), self.ord_head(h)


def decode_prediction(logits_cat: torch.Tensor, logits_ord: torch.Tensor) -> str:
    """logits_cat: (3,), logits_ord: (5,) -> a VALID_LABELS string."""
    cat_idx = int(torch.argmax(logits_cat).item())
    cat_name = config.CAT_NAMES[cat_idx]
    if cat_name == "ordinal":
        ord_idx = int(torch.argmax(logits_ord).item())
        return config.ORDINAL_LABELS[ord_idx]
    return cat_name
