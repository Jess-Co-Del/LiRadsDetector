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

import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config, preprocessing


def label_to_targets(label: str):
    """Returns (cat_idx, ord_idx). ord_idx is -1 when the label isn't ordinal."""
    label = label.strip()
    if label in config.ORDINAL_LABELS:
        return 0, config.ORDINAL_LABELS.index(label)
    if label == "LR-M":
        return 1, -1
    if label == "LR-TIV":
        return 2, -1
    raise ValueError(f"unrecognized LI-RADS label: {label!r}")


def _find_case_dir(data_root: str, case_id: str) -> str:
    direct = os.path.join(data_root, case_id)
    if os.path.isdir(direct):
        return direct
    matches = glob.glob(os.path.join(data_root, "**", case_id), recursive=True)
    if not matches:
        raise FileNotFoundError(f"case directory for {case_id!r} not found under {data_root!r}")
    return matches[0]


class LiRadsCaseDataset(Dataset):
    def __init__(
        self,
        metadata_csv: str,
        data_root: str,
        max_slices: int = config.MAX_SLICES_PER_CASE,
        case_ids: Optional[Sequence[str]] = None,
    ):
        df = pd.read_csv(metadata_csv)
        df.columns = df.columns.str.strip().str.lower()
        if "lirads_score" not in df.columns:
            raise ValueError(f"{metadata_csv} is missing a 'lirads_score' column")
        if "case_id" not in df.columns:
            raise ValueError(f"{metadata_csv} is missing a 'case_id' column")
        if case_ids is not None:
            wanted = set(str(c) for c in case_ids)
            df = df[df["case_id"].astype(str).isin(wanted)]
            missing = wanted - set(df["case_id"].astype(str))
            if missing:
                raise ValueError(f"{len(missing)} case_id(s) from the split not found in {metadata_csv}: {sorted(missing)[:5]}...")
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.max_slices = max_slices

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        case_id = str(row["case_id"])
        case_dir = _find_case_dir(self.data_root, case_id)

        phase_paths = preprocessing.find_case_phase_paths(case_dir, case_id)
        mask_path = preprocessing.find_case_mask_path(case_dir)
        phase_data = preprocessing.build_case_tensors(phase_paths, mask_path, self.max_slices)

        cat_idx, ord_idx = label_to_targets(str(row["lirads_score"]))
        return {"case_id": case_id, "phase_data": phase_data, "cat_idx": cat_idx, "ord_idx": ord_idx}


def collate_cases(batch: list) -> dict:
    return {
        "case_ids": [b["case_id"] for b in batch],
        "phase_data": [b["phase_data"] for b in batch],
        "cat_idx": torch.tensor([b["cat_idx"] for b in batch], dtype=torch.long),
        "ord_idx": torch.tensor([b["ord_idx"] for b in batch], dtype=torch.long),
    }
