"""Generates a genuine N-way stratified k-fold partition over a metadata CSV
and saves it to a single JSON file, keyed by fold index.

The outer split (test) is a `StratifiedKFold` (by lirads_score) over the
whole dataset: each case's `test`-fold membership is fixed and the N test
sets are disjoint and cover every case exactly once, unlike independent
random draws. Within each fold's non-test remainder, train/val is a further
stratified random split at `val_frac`. Use `--fold` in train.py to pick one.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def _make_test_folds(case_ids: np.ndarray, labels: np.ndarray, n_folds: int, seed: int):
    """Yields (trainval_idx, test_idx) index arrays, one per fold. Falls back
    to a plain (unstratified) KFold if some class has fewer members than
    n_folds, which StratifiedKFold can't handle."""
    try:
        return list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(case_ids, labels))
    except ValueError:
        return list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(case_ids))


def _split_case_ids(case_ids: np.ndarray, labels: np.ndarray, val_frac: float, seed: int) -> dict:
    """Splits one fold's (already test-held-out) case_ids/labels into
    train/val."""
    try:
        train_ids, val_ids = train_test_split(case_ids, test_size=val_frac, stratify=labels, random_state=seed)
    except ValueError:
        # A class too small to stratify at this fraction (e.g. only 1-2
        # cases) -- fall back to a plain random split.
        train_ids, val_ids = train_test_split(case_ids, test_size=val_frac, random_state=seed)
    return sorted(train_ids.tolist()), sorted(val_ids.tolist())


def fold_tagged_path(path: str, fold) -> str:
    """Inserts `_fold{fold}` before the extension, unless it's already there
    -- so output files (checkpoints, predictions, plots, ...) for different
    folds never collide, whether the base path was left at its default or
    set explicitly."""
    root, ext = os.path.splitext(path)
    tag = f"_fold_{fold}"
    if tag in root:
        return path
    return f"{root}{tag}{ext}"


def load_fold(splits_json: str, fold) -> dict:
    """Loads one {"train": [...], "val": [...], "test": [...]} fold out of a
    JSON file written by make_folds()/this module's CLI."""
    with open(splits_json) as f:
        all_folds = json.load(f)
    if str(fold) not in all_folds:
        raise ValueError(f"fold {fold!r} not found in {splits_json} (have: {sorted(all_folds)})")
    return all_folds[str(fold)]


def make_folds(
    metadata_csv: str,
    n_folds: int,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")

    df = pd.read_csv(metadata_csv)
    df.columns = df.columns.str.strip().str.lower()
    if "case_id" not in df.columns or "lirads_score" not in df.columns:
        raise ValueError(f"{metadata_csv} must have 'case_id' and 'lirads_score' columns")

    case_ids = df["case_id"].astype(str).to_numpy()
    labels = df["lirads_score"].to_numpy()

    folds = {}
    for fold_idx, (trainval_idx, test_idx) in enumerate(_make_test_folds(case_ids, labels, n_folds, seed)):
        train_ids, val_ids = _split_case_ids(
            case_ids[trainval_idx], labels[trainval_idx], val_frac, seed=seed + fold_idx,
        )
        folds[str(fold_idx)] = {"train": train_ids, "val": val_ids, "test": sorted(case_ids[test_idx].tolist())}
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Build N-fold stratified train/val/test splits, saved as JSON")
    parser.add_argument("--metadata_csv", required=True, help="CSV with case_id + lirads_score columns")
    parser.add_argument("--n_folds", type=int, required=True, help="number of stratified k-folds (test partitions)")
    parser.add_argument(
        "--val_frac", type=float, default=0.15,
        help="fraction of each fold's non-test remainder held out for validation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="output JSON path")
    args = parser.parse_args()

    folds = make_folds(args.metadata_csv, args.n_folds, args.val_frac, args.seed)
    with open(args.out, "w") as f:
        json.dump(folds, f, indent=2)

    for fold_idx, split in folds.items():
        print(f"fold {fold_idx}: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")
    print(f"wrote {len(folds)} folds to {args.out}")


if __name__ == "__main__":
    main()
