"""PyTorch Dataset for AMPLIFAI metadata + case folders.

Expects a metadata CSV with at least `case_id` and `lirads_score` columns,
and a `data_root` containing extracted batch zips, i.e.
`<data_root>/**/<case_id>/` directories laid out as:
    <case_id>/ct/<case_id>_{ART,VEN,DEL,DRY}.nii.gz
    <case_id>/annotations/lesion.nii.gz
"""

import glob
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config, lesion_transplant, preprocessing


def label_to_targets(label: str):
    """Returns (cat_idx, ord_idx). ord_idx is -1 when the label isn't ordinal.
    cat_idx follows config.CAT_NAMES's order: ordinal=0, LR-M=1, LR-TIV=2,
    No lesion=3."""
    label = label.strip()
    if label in config.ORDINAL_LABELS:
        return 0, config.ORDINAL_LABELS.index(label)
    if label == "LR-M":
        return 1, 0
    if label == "LR-TIV":
        return 2, 0
    if label == config.NO_LESION_LABEL:
        return 3, 0
    raise ValueError(f"unrecognized LI-RADS label: {label!r}")


def encode_clinical_features(row: pd.Series) -> torch.Tensor:
    """One-hots `aphe` (missing/unrecognized values fall into an explicit
    "Unknown" category), appends the four binary washout/capsule flags, and
    appends max_diameter_mm scaled by config.CLINICAL_DIAMETER_SCALE_MM,
    giving a fixed-length config.CLINICAL_FEATURE_DIM vector."""
    aphe = row["aphe"]
    aphe = str(aphe).strip() if pd.notna(aphe) else "Unknown"
    if aphe not in config.APHE_CATEGORIES:
        aphe = "Unknown"
    aphe_onehot = [1.0 if aphe == cat else 0.0 for cat in config.APHE_CATEGORIES]
    binary_feats = [float(row[col]) for col in config.CLINICAL_BINARY_FEATURES]
    diameter_feat = [float(row["max_diameter_mm"]) / config.CLINICAL_DIAMETER_SCALE_MM]
    return torch.tensor(aphe_onehot + binary_feats + diameter_feat, dtype=torch.float32)


def _find_case_dir(data_root: str, case_id: str) -> str:
    direct = os.path.join(data_root, case_id)
    if os.path.isdir(direct):
        return direct
    else:
        raise FileNotFoundError(f"case directory for {case_id!r} not found under {data_root!r}")


class LiRadsCaseDataset(Dataset):
    def __init__(
        self,
        metadata_csv: str,
        data_root: str,
        max_slices: int = config.MAX_SLICES_PER_CASE,
        case_ids: Optional[Sequence[str]] = None,
        augment: bool = False,
        transplant: bool = False,
    ):
        df = pd.read_csv(metadata_csv)
        df.columns = df.columns.str.strip().str.lower()
        if "lirads_score" not in df.columns:
            raise ValueError(f"{metadata_csv} is missing a 'lirads_score' column")
        if "case_id" not in df.columns:
            raise ValueError(f"{metadata_csv} is missing a 'case_id' column")
        clinical_cols = ["aphe", "max_diameter_mm"] + config.CLINICAL_BINARY_FEATURES
        missing_cols = [c for c in clinical_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{metadata_csv} is missing clinical feature column(s): {missing_cols}")
        if case_ids is not None:
            wanted = set(str(c) for c in case_ids)
            df = df[df["case_id"].astype(str).isin(wanted)]
            missing = wanted - set(df["case_id"].astype(str))
            if missing:
                raise ValueError(f"{len(missing)} case_id(s) from the split not found in {metadata_csv}: {sorted(missing)[:5]}...")
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.max_slices = max_slices
        self.augment = augment
        self.transplant = transplant

    def __len__(self) -> int:
        return len(self.df)

    def _build_phase_data(self, row: pd.Series, case_id: str, case_dir: str, label: str) -> dict:
        """Normally just preprocessing.build_case_tensors() on this case's
        own files. When self.transplant is on and `label` is one of
        config.TRANSPLANT_DONOR_LABELS (LR-1/LR-2/LR-3 by default), with
        probability config.TRANSPLANT_PROB this case instead becomes a
        lesion *donor*: its real lesion is pasted into a different random
        recipient case's liver (lesion_transplant.transplant_case()), and
        tensors are built from that synthesized volume instead -- still
        labeled `label`, since the transplanted lesion is the donor's real,
        correctly-labeled one. Falls back to this case's own real data if no
        recipient has a liver mask yet, or none has room for this lesion.
        """
        if self.transplant and label in config.TRANSPLANT_DONOR_LABELS:
            rng = np.random.default_rng()
            if rng.random() < config.TRANSPLANT_PROB:
                recipient_case_id = lesion_transplant.find_recipient_case_id(self.df, case_id, rng)
                if recipient_case_id is not None:
                    try:
                        recipient_dir = _find_case_dir(self.data_root, recipient_case_id)
                        phase_vols, mask_vol = lesion_transplant.transplant_case(
                            case_dir, case_id, recipient_dir, recipient_case_id, rng,
                        )
                        return preprocessing.build_case_tensors_from_volumes(
                            phase_vols, mask_vol, self.max_slices, augment=self.augment, rng=rng,
                        )
                    except (FileNotFoundError, ValueError):
                        pass  # no liver mask yet, or no room for this lesion -- fall back below

        phase_paths = preprocessing.find_case_phase_paths(case_dir, case_id)
        mask_path = preprocessing.find_case_mask_path(case_dir)
        return preprocessing.build_case_tensors(
            phase_paths, mask_path, self.max_slices, augment=self.augment, label=label,
        )

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        case_id = str(row["case_id"])
        case_dir = _find_case_dir(self.data_root, case_id)

        label = str(row["lirads_score"]).strip()
        phase_data = self._build_phase_data(row, case_id, case_dir, label)

        cat_idx, ord_idx = label_to_targets(label)
        clinical_features = encode_clinical_features(row)
        return {
            "case_id": case_id, "phase_data": phase_data, "cat_idx": cat_idx, "ord_idx": ord_idx,
            "clinical_features": clinical_features,
        }


def collate_cases(batch: list) -> dict:
    return {
        "case_ids": [b["case_id"] for b in batch],
        "phase_data": [b["phase_data"] for b in batch],
        "cat_idx": torch.tensor([b["cat_idx"] for b in batch], dtype=torch.long),
        "ord_idx": torch.tensor([b["ord_idx"] for b in batch], dtype=torch.long),
        "clinical_features": torch.stack([b["clinical_features"] for b in batch], dim=0),
    }
