"""
LiRadsNet: mask-guided-pooled DINOv2 features -> dual classification head
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from . import config
from .backbone import Dinov2SliceEncoder
from .losses import corn_label_from_logits

PhaseData = Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]

# The backbone-already-applied counterpart of PhaseData: (patch_tokens,
# cls_token, mask_grids, slice_weights, volume) for a phase that was present,
# or None for a missing one. Produced once per case/view by
# compute_backbone_feats() and consumed by LiRadsNet/ClinicalPredictorNet's
# *_from_backbone_feats methods below -- see compute_backbone_feats's
# docstring for why this exists (sharing one backbone forward pass across an
# ensemble of checkpoints that all carry the same frozen backbone).
BackboneFeats = Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]


def compute_backbone_feats(
    backbone: Dinov2SliceEncoder, phase_data: Dict[str, PhaseData], device: torch.device,
) -> Dict[str, BackboneFeats]:
    """Runs `backbone` once per phase present in `phase_data` and packages
    its (patch_tokens, cls_token) output alongside that phase's
    mask_grids/slice_weights/volume (moved to `device`), keyed by phase name
    -- a missing phase stays None. Every LiRadsNet/ClinicalPredictorNet
    checkpoint carries its own backbone instance, but all of them are frozen
    (requires_grad=False, never in the optimizer -- see train.py's
    trainable_params) and loaded from the same vendored DINOv2 snapshot, so
    their backbone weights are always numerically identical to each other
    and to the original pretrained snapshot; only the per-model pooling/head
    work downstream of the backbone actually differs between checkpoints.
    That makes it safe and exact (not an approximation) for an ensemble of
    such checkpoints to call this once, with any one of their own `backbone`
    attributes, and feed the result to every model's *_from_backbone_feats
    method instead of each one separately re-running its own backbone
    forward on the same pixel_values -- see predict.predict_case_ensemble
    and predict_clinical.predict_case_metadata. This invariant breaks (and
    this sharing would silently become wrong) if a future training run ever
    unfreezes or otherwise diverges one checkpoint's backbone from another's."""
    feats = {}
    for phase, data in phase_data.items():
        if data is None:
            feats[phase] = None
            continue
        pixel_values, mask_grids, slice_weights, volume = data
        patch_tokens, cls_token = backbone(pixel_values.to(device))
        feats[phase] = (patch_tokens, cls_token, mask_grids.to(device), slice_weights.to(device), volume.to(device))
    return feats


class PhaseVolumeCNN(nn.Module):
    """
    Small 3D CNN that encodes one phase's lesion-cropped slice stack,
    given as a single-channel (1, S, H, W) volume, into a fixed-size
    (1, out_size, out_size) feature map. AdaptiveAvgPool3d collapses the
    depth dimension to 1 regardless of S (the sampled slice count varies per
    case), so this works for any number of slices, including S=1."""

    def __init__(self, out_size: int = config.CNN_FEATURE_MAP_SIZE, hidden_channels: int = config.CNN_HIDDEN_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1, stride=(1, 2, 2)),
            nn.InstanceNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1, stride=(1, 2, 2)),
            nn.InstanceNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, out_size, out_size))

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """volume: (1, 1, S, H, W) -> (out_size, out_size)"""
        feat_map = self.pool(self.net(volume))  # (1, 1, 1, out_size, out_size)
        return feat_map.reshape(-1)  # (out_size * out_size,)


class LiRadsNet(nn.Module):
    def __init__(
        self,
        backbone: Dinov2SliceEncoder,
        embed_dim: int = config.EMBED_DIM,
        grid_size: int = config.GRID_SIZE,
        hidden1: int = config.HEAD_HIDDEN_1,
        hidden2: int = config.HEAD_HIDDEN_2,
        dropout: float = config.HEAD_DROPOUT,
        clinical_dim: int = config.CLINICAL_FEATURE_DIM,
        clinical_embed_dim: int = config.CLINICAL_EMBED_DIM,
        use_cnn: bool = True,
        use_clinical: bool = True,
        use_cat_head: bool = True,
        ordinal_head_type: str = "softmax",
        cat_names: Sequence[str] = config.CAT_NAMES,
    ):
        super().__init__()
        if ordinal_head_type not in ("softmax", "corn"):
            raise ValueError(f"unrecognized ordinal_head_type: {ordinal_head_type!r}")
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.phase_names = config.PHASE_NAMES
        self.use_cnn = use_cnn
        self.use_clinical = use_clinical
        self.use_cat_head = use_cat_head
        self.ordinal_head_type = ordinal_head_type
        # Normally config.CAT_NAMES (4-way: ordinal/LR-M/LR-TIV/No lesion),
        # but a caller training on a narrower category set (e.g. train.py's
        # --no-include_no_lesion, which drops "No lesion") passes a shorter
        # list here so cat_head is sized to match and decode_prediction()
        # can map cat_head's argmax back to the right name,see
        # dataset.LiRadsCaseDataset's cat_names/label_to_targets, which must
        # agree with whatever's passed here for a given training run.
        self.cat_names = list(cat_names)

        for p in self.backbone.parameters():
            p.requires_grad = False

        # One 3D CNN per phase (contrast behavior differs by phase), each
        # encoding that phase's slice stack as a genuine 3D volume rather
        # than S independent 2D images,complementary to DINOv2's per-slice
        # view. See PhaseVolumeCNN and config.CNN_FEATURE_MAP_SIZE. Optional:
        # use_cnn=False runs DINOv2-only.
        self.cnn_out_size = config.CNN_FEATURE_MAP_SIZE
        if self.use_cnn:
            self.cnn_encoders = nn.ModuleList([PhaseVolumeCNN(self.cnn_out_size) for _ in self.phase_names])
            cnn_feat_dim = self.cnn_out_size * self.cnn_out_size
        else:
            self.cnn_encoders = None
            cnn_feat_dim = 0

        phase_feat_dim = embed_dim * 2 + cnn_feat_dim  # masked-pooled patch feat + CLS feat + (optional) 3D-CNN feature map
        self.missing_phase_embed = nn.Parameter(torch.randn(len(self.phase_names), phase_feat_dim) * 0.02)

        # Tabular LI-RADS major features (aphe/washout/capsule): a small
        # Linear projection, scaled by a learned weight before being
        # concatenated onto the image encoding. Not every case has this
        # metadata (the challenge submission input never does), so a learned
        # placeholder embedding stands in when it's absent,the same
        # pattern as missing_phase_embed above. Optional: use_clinical=False
        # drops this branch (images only).
        if self.use_clinical:
            self.clinical_encoder = nn.Sequential(
                nn.LayerNorm(clinical_dim),
                nn.Linear(clinical_dim, clinical_embed_dim),
                nn.GELU(),
            )
            self.clinical_scale = nn.Parameter(torch.tensor(1.0))
            self.missing_clinical_embed = nn.Parameter(torch.randn(clinical_embed_dim) * 0.02)
            clinical_out_dim = clinical_embed_dim
        else:
            self.clinical_encoder = None
            self.clinical_scale = None
            self.missing_clinical_embed = None
            clinical_out_dim = 0

        in_dim = phase_feat_dim * len(self.phase_names) + clinical_out_dim
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
        )
        # Optional: use_cat_head=False drops the 4-way category head
        # entirely, for single-head training on the ordinal target alone
        # (see config.ORDINAL_LABELS and LiRadsCaseDataset's ordinal_only),
        # forward() then returns None in its place, and decode_prediction()
        # skips the category gate and reads the ordinal head directly.
        self.cat_head = nn.Linear(hidden2, len(self.cat_names)) if self.use_cat_head else None
        # "corn" sizes the ordinal head to num_classes-1 conditional-threshold
        # logits instead of a plain num_classes-way softmax,see
        # losses.CornSoftQWKLoss/corn_loss and decode_prediction() below.
        ord_out_dim = len(config.ORDINAL_LABELS) - 1 if self.ordinal_head_type == "corn" else len(config.ORDINAL_LABELS)
        self.ord_head = nn.Linear(hidden2, ord_out_dim)

    def _pool_phase(
        self, patch_tokens: torch.Tensor, cls_token: torch.Tensor, mask_grids: torch.Tensor, slice_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Mask-guided pooling of one phase's already-backbone-encoded
        tokens into a single (2D,) feature -- the part of encode_phase that
        doesn't touch the backbone, split out so it can run either straight
        off a fresh backbone forward (encode_phase) or off a backbone
        forward computed once and shared across an ensemble
        (encode_case_from_backbone_feats/compute_backbone_feats)."""
        S = patch_tokens.shape[0]
        patch_tokens = patch_tokens.view(S, self.grid_size, self.grid_size, self.embed_dim)

        mask_sum = mask_grids.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        mask_w = mask_grids / mask_sum
        masked_pooled = (patch_tokens * mask_w.unsqueeze(-1)).sum(dim=(1, 2))  # (S,D)

        slice_w = slice_weights / slice_weights.sum().clamp_min(1e-6)  # (S,)
        phase_masked = (masked_pooled * slice_w.unsqueeze(-1)).sum(dim=0)  # (D,)
        phase_cls = (cls_token * slice_w.unsqueeze(-1)).sum(dim=0)  # (D,)
        return torch.cat([phase_masked, phase_cls], dim=0)  # (2D,)

    def encode_phase(self, pixel_values: torch.Tensor, mask_grids: torch.Tensor, slice_weights: torch.Tensor) -> torch.Tensor:
        patch_tokens, cls_token = self.backbone(pixel_values)  # (S,N,D), (S,D)
        return self._pool_phase(patch_tokens, cls_token, mask_grids, slice_weights)

    def encode_phase_cnn(self, volume: torch.Tensor, phase_idx: int) -> torch.Tensor:
        """volume: (S, H, W) single-channel slice stack -> (cnn_out_size**2,)"""
        x = volume.unsqueeze(0).unsqueeze(0)  # (1, 1, S, H, W)
        return self.cnn_encoders[phase_idx](x)

    def encode_case(self, phase_data: Dict[str, PhaseData]) -> torch.Tensor:
        device = self.missing_phase_embed.device
        feats = []
        for i, phase in enumerate(self.phase_names):
            data = phase_data.get(phase)
            if data is None:
                feats.append(self.missing_phase_embed[i])
            else:
                pixel_values, mask_grids, slice_weights, volume = data
                dinov2_feat = self.encode_phase(
                    pixel_values.to(device), mask_grids.to(device), slice_weights.to(device)
                )
                if self.use_cnn:
                    cnn_feat = self.encode_phase_cnn(volume.to(device), i)
                    feats.append(torch.cat([dinov2_feat, cnn_feat], dim=0))
                else:
                    feats.append(dinov2_feat)
        return torch.cat(feats, dim=0)  # (phase_feat_dim * n_phases,)

    def encode_case_from_backbone_feats(self, backbone_feats: Dict[str, "BackboneFeats"]) -> torch.Tensor:
        """Same as encode_case, but `backbone_feats` (see
        compute_backbone_feats) already carries each phase's backbone
        output instead of raw pixel_values, so this never touches
        self.backbone -- lets an ensemble of checkpoints share one backbone
        forward pass per case/view (see predict.predict_case_ensemble)."""
        device = self.missing_phase_embed.device
        feats = []
        for i, phase in enumerate(self.phase_names):
            data = backbone_feats.get(phase)
            if data is None:
                feats.append(self.missing_phase_embed[i])
            else:
                patch_tokens, cls_token, mask_grids, slice_weights, volume = data
                dinov2_feat = self._pool_phase(
                    patch_tokens.to(device), cls_token.to(device), mask_grids.to(device), slice_weights.to(device),
                )
                if self.use_cnn:
                    cnn_feat = self.encode_phase_cnn(volume.to(device), i)
                    feats.append(torch.cat([dinov2_feat, cnn_feat], dim=0))
                else:
                    feats.append(dinov2_feat)
        return torch.cat(feats, dim=0)  # (phase_feat_dim * n_phases,)

    def encode_clinical(self, clinical_features: Optional[torch.Tensor], batch_size: int) -> torch.Tensor:
        device = self.missing_clinical_embed.device
        if clinical_features is None:
            return self.missing_clinical_embed.unsqueeze(0).expand(batch_size, -1)
        return self.clinical_encoder(clinical_features.to(device)) * self.clinical_scale

    def forward(
        self,
        batch_phase_data: List[Dict[str, PhaseData]],
        clinical_features: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        case_feats = torch.stack([self.encode_case(pd) for pd in batch_phase_data], dim=0)
        if self.use_clinical:
            clinical_feats = self.encode_clinical(clinical_features, len(batch_phase_data))
            case_feats = torch.cat([case_feats, clinical_feats], dim=1)
        h = self.head(case_feats)
        logits_cat = self.cat_head(h) if self.use_cat_head else None
        return logits_cat, self.ord_head(h)

    def forward_from_backbone_feats(
        self,
        batch_backbone_feats: List[Dict[str, "BackboneFeats"]],
        clinical_features: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Same as forward(), but each batch element's images have already
        been run through the (shared, frozen) backbone -- see
        encode_case_from_backbone_feats/compute_backbone_feats."""
        case_feats = torch.stack([self.encode_case_from_backbone_feats(bf) for bf in batch_backbone_feats], dim=0)
        if self.use_clinical:
            clinical_feats = self.encode_clinical(clinical_features, len(batch_backbone_feats))
            case_feats = torch.cat([case_feats, clinical_feats], dim=1)
        h = self.head(case_feats)
        logits_cat = self.cat_head(h) if self.use_cat_head else None
        return logits_cat, self.ord_head(h)


class ClinicalPredictorNet(nn.Module):
    """
    Predicts encode_clinical_features's 8 non-diameter dims (aphe's one-hot
    + the 4 washout/capsule binary flags) straight from the CT images:
    max_diameter_mm, the 9th, is deterministic geometry instead
    (preprocessing.compute_max_diameter_mm), not something this model
    touches. See config's "Clinical feature prediction" section for why:
    the real submission input never includes clinical metadata, so
    predict_clinical.py runs this model per case to synthesize the row
    LiRadsNet's clinical branch expects, trained by train_clinical.py on
    train_metadata.csv's own aphe/washout/capsule columns as targets.

    Same image-encoding shape as LiRadsNet's trunk (per-phase masked-pooled
    DINOv2 features + optional PhaseVolumeCNN,see encode_phase/
    encode_phase_cnn/encode_case below, deliberately duplicated from
    LiRadsNet rather than shared: this model must never take clinical
    features as *input*,there'd be nothing left to predict,so it
    carries its own backbone instance and is trained/checkpointed entirely
    independently of the main LiRadsNet, and duplicating this trunk means
    neither model's checkpoint depends on the other's module layout).
    """

    def __init__(
        self,
        backbone: Dinov2SliceEncoder,
        embed_dim: int = config.EMBED_DIM,
        grid_size: int = config.GRID_SIZE,
        hidden1: int = config.HEAD_HIDDEN_1,
        hidden2: int = config.HEAD_HIDDEN_2,
        dropout: float = config.HEAD_DROPOUT,
        use_cnn: bool = True,
        aphe_categories: Sequence[str] = config.APHE_PREDICTABLE_CATEGORIES,
        binary_features: Sequence[str] = config.CLINICAL_BINARY_FEATURES,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.phase_names = config.PHASE_NAMES
        self.use_cnn = use_cnn
        self.aphe_categories = list(aphe_categories)
        self.binary_features = list(binary_features)

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.cnn_out_size = config.CNN_FEATURE_MAP_SIZE
        if self.use_cnn:
            self.cnn_encoders = nn.ModuleList([PhaseVolumeCNN(self.cnn_out_size) for _ in self.phase_names])
            cnn_feat_dim = self.cnn_out_size * self.cnn_out_size
        else:
            self.cnn_encoders = None
            cnn_feat_dim = 0

        phase_feat_dim = embed_dim * 2 + cnn_feat_dim
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
        # aphe_head: single-label softmax over aphe_categories (mutually
        # exclusive). binary_head: one independent sigmoid logit per
        # binary_features entry (not mutually exclusive with each other).
        self.aphe_head = nn.Linear(hidden2, len(self.aphe_categories))
        self.binary_head = nn.Linear(hidden2, len(self.binary_features))

    # _pool_phase/encode_phase/encode_phase_cnn/encode_case/
    # encode_case_from_backbone_feats: identical to LiRadsNet's own (see
    # there for the "why" of each step) -- duplicated, see this class's
    # docstring.
    def _pool_phase(
        self, patch_tokens: torch.Tensor, cls_token: torch.Tensor, mask_grids: torch.Tensor, slice_weights: torch.Tensor,
    ) -> torch.Tensor:
        S = patch_tokens.shape[0]
        patch_tokens = patch_tokens.view(S, self.grid_size, self.grid_size, self.embed_dim)

        mask_sum = mask_grids.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        mask_w = mask_grids / mask_sum
        masked_pooled = (patch_tokens * mask_w.unsqueeze(-1)).sum(dim=(1, 2))  # (S,D)

        slice_w = slice_weights / slice_weights.sum().clamp_min(1e-6)  # (S,)
        phase_masked = (masked_pooled * slice_w.unsqueeze(-1)).sum(dim=0)  # (D,)
        phase_cls = (cls_token * slice_w.unsqueeze(-1)).sum(dim=0)  # (D,)
        return torch.cat([phase_masked, phase_cls], dim=0)  # (2D,)

    def encode_phase(self, pixel_values: torch.Tensor, mask_grids: torch.Tensor, slice_weights: torch.Tensor) -> torch.Tensor:
        patch_tokens, cls_token = self.backbone(pixel_values)  # (S,N,D), (S,D)
        return self._pool_phase(patch_tokens, cls_token, mask_grids, slice_weights)

    def encode_phase_cnn(self, volume: torch.Tensor, phase_idx: int) -> torch.Tensor:
        """volume: (S, H, W) single-channel slice stack -> (cnn_out_size**2,)"""
        x = volume.unsqueeze(0).unsqueeze(0)  # (1, 1, S, H, W)
        return self.cnn_encoders[phase_idx](x)

    def encode_case(self, phase_data: Dict[str, PhaseData]) -> torch.Tensor:
        device = self.missing_phase_embed.device
        feats = []
        for i, phase in enumerate(self.phase_names):
            data = phase_data.get(phase)
            if data is None:
                feats.append(self.missing_phase_embed[i])
            else:
                pixel_values, mask_grids, slice_weights, volume = data
                dinov2_feat = self.encode_phase(
                    pixel_values.to(device), mask_grids.to(device), slice_weights.to(device)
                )
                if self.use_cnn:
                    cnn_feat = self.encode_phase_cnn(volume.to(device), i)
                    feats.append(torch.cat([dinov2_feat, cnn_feat], dim=0))
                else:
                    feats.append(dinov2_feat)
        return torch.cat(feats, dim=0)  # (phase_feat_dim * n_phases,)

    def encode_case_from_backbone_feats(self, backbone_feats: Dict[str, "BackboneFeats"]) -> torch.Tensor:
        device = self.missing_phase_embed.device
        feats = []
        for i, phase in enumerate(self.phase_names):
            data = backbone_feats.get(phase)
            if data is None:
                feats.append(self.missing_phase_embed[i])
            else:
                patch_tokens, cls_token, mask_grids, slice_weights, volume = data
                dinov2_feat = self._pool_phase(
                    patch_tokens.to(device), cls_token.to(device), mask_grids.to(device), slice_weights.to(device),
                )
                if self.use_cnn:
                    cnn_feat = self.encode_phase_cnn(volume.to(device), i)
                    feats.append(torch.cat([dinov2_feat, cnn_feat], dim=0))
                else:
                    feats.append(dinov2_feat)
        return torch.cat(feats, dim=0)  # (phase_feat_dim * n_phases,)

    def forward(self, batch_phase_data: List[Dict[str, PhaseData]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (aphe_logits, binary_logits): (B, len(aphe_categories))
        and (B, len(binary_features))."""
        case_feats = torch.stack([self.encode_case(pd) for pd in batch_phase_data], dim=0)
        h = self.head(case_feats)
        return self.aphe_head(h), self.binary_head(h)

    def forward_from_backbone_feats(
        self, batch_backbone_feats: List[Dict[str, "BackboneFeats"]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same as forward(), but off backbone output already computed once
        and shared across the clinical-predictor ensemble -- see
        encode_case_from_backbone_feats/compute_backbone_feats and
        predict_clinical.predict_case_metadata."""
        case_feats = torch.stack([self.encode_case_from_backbone_feats(bf) for bf in batch_backbone_feats], dim=0)
        h = self.head(case_feats)
        return self.aphe_head(h), self.binary_head(h)


def decode_clinical_prediction(
    aphe_logits: torch.Tensor, binary_logits: torch.Tensor,
    aphe_categories: Sequence[str] = config.APHE_PREDICTABLE_CATEGORIES,
    binary_features: Sequence[str] = config.CLINICAL_BINARY_FEATURES,
) -> dict:
    """
    One case's ClinicalPredictorNet output -> {"aphe": str,
    <binary_features[i]>: 0/1, ...},the same column names
    dataset.encode_clinical_features expects, aside from max_diameter_mm
    (see preprocessing.compute_max_diameter_mm). aphe_logits:
    (len(aphe_categories),), binary_logits: (len(binary_features),).
    """
    aphe = aphe_categories[int(torch.argmax(aphe_logits).item())]
    result = {"aphe": aphe}
    for name, p in zip(binary_features, torch.sigmoid(binary_logits).tolist()):
        result[name] = int(p >= 0.5)
    return result


def decode_prediction(
    logits_cat: Optional[torch.Tensor], logits_ord: torch.Tensor, ordinal_head_type: str = "softmax",
    cat_names: Sequence[str] = config.CAT_NAMES,
) -> str:
    """logits_cat: (len(cat_names),) or None, logits_ord: (5,) for
    ordinal_head_type="softmax" or (4,) for "corn" (must match the LiRadsNet
    that produced it,see its ordinal_head_type) -> one of
    config.VALID_LABELS or config.NO_LESION_LABEL (the latter must be
    remapped before it's ever submitted to the actual challenge, which
    doesn't score it). logits_cat is None for a single-head, ordinal-only
    model (LiRadsNet(use_cat_head=False)),there's no category gate to
    consult, so the ordinal head's decoded rank is returned directly.
    `cat_names` must be the same list the producing LiRadsNet was built with
    (its `.cat_names` attribute),pass config.CAT_NAMES only for a model
    trained with the default 4-way category head."""
    if logits_cat is not None:
        cat_idx = int(torch.argmax(logits_cat).item())
        cat_name = cat_names[cat_idx]
        if cat_name != "ordinal":
            return cat_name
    if ordinal_head_type == "corn":
        ord_idx = int(corn_label_from_logits(logits_ord.unsqueeze(0))[0].item())
    else:
        ord_idx = int(torch.argmax(logits_ord).item())
    return config.ORDINAL_LABELS[ord_idx]
