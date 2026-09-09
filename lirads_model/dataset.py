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


def label_to_targets(label: str, cat_names: Sequence[str] = config.CAT_NAMES):
    """Returns (cat_idx, ord_idx). ord_idx is -1 when the label isn't ordinal.
    cat_idx is `label`'s (or, for an ordinal label, "ordinal"'s) position in
    `cat_names` -- normally config.CAT_NAMES's order (ordinal=0, LR-M=1,
    LR-TIV=2, No lesion=3), but a caller training a narrower category head
    (e.g. train.py's --no-include_no_lesion, see LiRadsCaseDataset's
    cat_names) passes that same narrower list here so cat_idx lines up with
    the model's actual cat_head width."""
    label = label.strip()
    if label in config.ORDINAL_LABELS:
        return cat_names.index("ordinal"), config.ORDINAL_LABELS.index(label)
    if label in cat_names:
        return cat_names.index(label), 0
    raise ValueError(f"unrecognized LI-RADS label for this model's category head {list(cat_names)!r}: {label!r}")


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


# A single top-level knob (train.py's/scripts/profile_train.py's
# --augment_mode) for which of LiRadsCaseDataset's three independent
# augmentation strategies are on, instead of three separate flags whose
# combinations aren't all meaningful: lesion transplant and anatomy-informed
# augmentation are each layered *on top of* the geometric (spatial) warp --
# see lesion_transplant.py/config.ANATOMY_* -- so neither is offered with
# spatial augmentation off. "none" is the eval/test/inference default in all
# but name; it's spelled out here so it can be requested explicitly too, e.g.
# for an augmentation ablation run.
AUGMENT_MODES = {
    "none": {"augment": False, "transplant": False, "anatomy": False},
    "spatial": {"augment": True, "transplant": False, "anatomy": False},
    "spatial+transplant": {"augment": True, "transplant": True, "anatomy": False},
    "spatial+anatomy": {"augment": True, "transplant": False, "anatomy": True},
    "all": {"augment": True, "transplant": True, "anatomy": True},
}


def resolve_augment_mode(mode: str) -> dict:
    """Maps an --augment_mode string to the {augment, transplant, anatomy}
    kwargs LiRadsCaseDataset expects, e.g.
    `LiRadsCaseDataset(..., **resolve_augment_mode(args.augment_mode))`."""
    if mode not in AUGMENT_MODES:
        raise ValueError(f"unrecognized augment_mode {mode!r} (must be one of {sorted(AUGMENT_MODES)})")
    return AUGMENT_MODES[mode]


class LiRadsCaseDataset(Dataset):
    def __init__(
        self,
        metadata_csv: str,
        data_root: str,
        max_slices: int = config.MAX_SLICES_PER_CASE,
        case_ids: Optional[Sequence[str]] = None,
        augment: bool = False,
        transplant: bool = False,
        anatomy: bool = False,
        ordinal_only: bool = False,
        cat_names: Sequence[str] = config.CAT_NAMES,
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
        if ordinal_only:
            # Single-head ordinal-only training (see model.LiRadsNet's
            # use_cat_head=False): there's no category head/target to make
            # sense of LR-M/LR-TIV/No lesion cases here, so they're dropped
            # from the split (after case_ids selection, so this is a silent
            # narrowing of an existing split rather than a "requested case
            # missing" error).
            df = df[df["lirads_score"].str.strip().isin(config.ORDINAL_LABELS)]
        elif config.NO_LESION_LABEL not in cat_names:
            # cat_names is narrower than the full category set (train.py's
            # --no-include_no_lesion): config.NO_LESION_LABEL cases have no
            # category left to route them through, so they're dropped from
            # the split -- the same silent-narrowing-of-an-existing-split
            # behavior as the ordinal_only filter above.
            df = df[df["lirads_score"].str.strip() != config.NO_LESION_LABEL]
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.max_slices = max_slices
        self.augment = augment
        self.transplant = transplant
        self.anatomy = anatomy
        self.ordinal_only = ordinal_only
        self.cat_names = list(cat_names)

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

        self.anatomy independently controls whether either path below may
        trigger the anatomy-informed deform (a random local warp of the
        case's own lesion, see config.ANATOMY_AUGMENT_PROB and
        augmentation.apply_anatomy_informed_deform), regardless of
        self.transplant. It no longer needs a liver mask -- the deformation
        field is computed from the lesion segmentation itself.
        """
        if self.transplant and label in config.TRANSPLANT_DONOR_LABELS:
            rng = np.random.default_rng()
            if rng.random() < config.TRANSPLANT_PROB:
                recipient_case_id = lesion_transplant.find_recipient_case_id(self.df, case_id, rng)
                if recipient_case_id is not None:
                    try:
                        recipient_dir = _find_case_dir(self.data_root, recipient_case_id)
                        phase_vols, mask_vol, _ = lesion_transplant.transplant_case(
                            case_dir, case_id, recipient_dir, recipient_case_id, rng,
                        )
                        return preprocessing.build_case_tensors_from_volumes(
                            phase_vols, mask_vol, self.max_slices, augment=self.augment, rng=rng,
                            anatomy=self.anatomy,
                        )
                    except (FileNotFoundError, ValueError):
                        pass  # no liver mask yet, or no room for this lesion -- fall back below

        phase_paths = preprocessing.find_case_phase_paths(case_dir, case_id)
        mask_path = preprocessing.find_case_mask_path(case_dir)
        return preprocessing.build_case_tensors(
            phase_paths, mask_path, self.max_slices, augment=self.augment, label=label, anatomy=self.anatomy,
        )

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        case_id = str(row["case_id"])
        case_dir = _find_case_dir(self.data_root, case_id)

        label = str(row["lirads_score"]).strip()
        phase_data = self._build_phase_data(row, case_id, case_dir, label)

        cat_idx, ord_idx = label_to_targets(label, self.cat_names)
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


class ClinicalMetadataDataset(Dataset):
    """
    Training data for model.ClinicalPredictorNet (see train_clinical.py):
    the same per-case CT images LiRadsCaseDataset uses, but targets are
    train_metadata.csv's own aphe/washout/capsule columns -- what a real
    submission's input never supplies (see config's "Clinical feature
    prediction" section) -- instead of lirads_score.

    config.NO_LESION_LABEL rows are never iterated over as training
    examples: there's no target lesion for aphe/washout/capsule to
    describe, so their clinical columns are meaningless training targets,
    not just missing ones. They're still kept around as candidate
    lesion-transplant *recipients* (see `transplant` below) -- clean liver
    background with no lesion of their own is exactly what makes them safe
    paste targets in the first place.

    `transplant`: same lesion-transplant augmentation as LiRadsCaseDataset,
    for the same config.TRANSPLANT_DONOR_LABELS-eligible rows (LR-1/2/3/4 by
    default) -- see lesion_transplant.py. When a row is picked as a donor,
    its own real lesion gets pasted into a different, randomly chosen
    recipient case's liver; the *donor's* own aphe/washout/capsule values
    are still the right training target for the resulting synthetic case,
    since the transplanted lesion tissue is genuinely the donor's, just
    against a different background -- exactly the same reasoning
    LiRadsCaseDataset already relies on for the donor's lirads_score.
    """

    def __init__(
        self,
        metadata_csv: str,
        data_root: str,
        max_slices: int = config.MAX_SLICES_PER_CASE,
        case_ids: Optional[Sequence[str]] = None,
        augment: bool = False,
        transplant: bool = False,
        aphe_categories: Sequence[str] = config.APHE_PREDICTABLE_CATEGORIES,
    ):
        df = pd.read_csv(metadata_csv)
        df.columns = df.columns.str.strip().str.lower()
        required_cols = ["case_id", "lirads_score", "aphe"] + config.CLINICAL_BINARY_FEATURES
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{metadata_csv} is missing column(s): {missing_cols}")
        if case_ids is not None:
            wanted = set(str(c) for c in case_ids)
            df = df[df["case_id"].astype(str).isin(wanted)]
            missing = wanted - set(df["case_id"].astype(str))
            if missing:
                raise ValueError(f"{len(missing)} case_id(s) from the split not found in {metadata_csv}: {sorted(missing)[:5]}...")
        # This split's full case pool (config.NO_LESION_LABEL rows included)
        # -- lesion_transplant.find_recipient_case_id draws from this, since
        # it prefers exactly those clean-liver rows as recipients. self.df
        # below (the actual per-__getitem__ training rows) then drops them.
        self._recipient_pool = df.reset_index(drop=True)
        df = df[df["lirads_score"].str.strip() != config.NO_LESION_LABEL]
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.max_slices = max_slices
        self.augment = augment
        self.transplant = transplant
        self.aphe_categories = list(aphe_categories)

    def __len__(self) -> int:
        return len(self.df)

    def _build_phase_data(self, case_id: str, case_dir: str, label: str) -> dict:
        """Normally just preprocessing.build_case_tensors() on this case's
        own files. When self.transplant is on and `label` is one of
        config.TRANSPLANT_DONOR_LABELS, with probability
        config.TRANSPLANT_PROB this case instead becomes a lesion *donor*:
        see this class's docstring. Falls back to this case's own real data
        if no recipient has a liver mask yet, or none has room for this
        lesion -- same fallback conditions as LiRadsCaseDataset."""
        if self.transplant and label in config.TRANSPLANT_DONOR_LABELS:
            rng = np.random.default_rng()
            if rng.random() < config.TRANSPLANT_PROB:
                recipient_case_id = lesion_transplant.find_recipient_case_id(self._recipient_pool, case_id, rng)
                if recipient_case_id is not None:
                    try:
                        recipient_dir = _find_case_dir(self.data_root, recipient_case_id)
                        phase_vols, mask_vol, _ = lesion_transplant.transplant_case(
                            case_dir, case_id, recipient_dir, recipient_case_id, rng,
                        )
                        return preprocessing.build_case_tensors_from_volumes(
                            phase_vols, mask_vol, self.max_slices, augment=self.augment, rng=rng,
                        )
                    except (FileNotFoundError, ValueError):
                        pass  # no liver mask yet, or no room for this lesion -- fall back below

        phase_paths = preprocessing.find_case_phase_paths(case_dir, case_id)
        mask_path = preprocessing.find_case_mask_path(case_dir)
        return preprocessing.build_case_tensors(phase_paths, mask_path, self.max_slices, augment=self.augment)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        case_id = str(row["case_id"])
        case_dir = _find_case_dir(self.data_root, case_id)
        label = str(row["lirads_score"]).strip()
        phase_data = self._build_phase_data(case_id, case_dir, label)

        # -1 for a missing/unrecognized aphe label -- masked out of the
        # aphe loss term (see train_clinical.py) rather than trained
        # against, since there's no real target for those rows. Always this
        # row's own aphe/washout/capsule, whether or not transplant swapped
        # in a different background -- see this class's docstring.
        aphe = str(row["aphe"]).strip() if pd.notna(row["aphe"]) else None
        aphe_idx = self.aphe_categories.index(aphe) if aphe in self.aphe_categories else -1
        binary_targets = torch.tensor([float(row[c]) for c in config.CLINICAL_BINARY_FEATURES], dtype=torch.float32)

        return {"case_id": case_id, "phase_data": phase_data, "aphe_idx": aphe_idx, "binary_targets": binary_targets}


def collate_clinical_cases(batch: list) -> dict:
    return {
        "case_ids": [b["case_id"] for b in batch],
        "phase_data": [b["phase_data"] for b in batch],
        "aphe_idx": torch.tensor([b["aphe_idx"] for b in batch], dtype=torch.long),
        "binary_targets": torch.stack([b["binary_targets"] for b in batch], dim=0),
    }
