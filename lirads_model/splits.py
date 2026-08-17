"""Generates N repeated stratified train/val/test splits over a metadata CSV
and saves them to a single JSON file, keyed by fold index.

Each fold is an independent stratified (by lirads_score) random split at the
given train/val/test fractions -- not a non-overlapping k-fold partition, so
folds' test sets may overlap. Use `--fold` in train.py to pick one.
"""

import argparse
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def _split_case_ids(df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float, seed: int) -> dict:
    case_ids = df["case_id"].astype(str).to_numpy()
    labels = df["lirads_score"].to_numpy()

    try:
        train_ids, rest_ids, train_labels, rest_labels = train_test_split(
            case_ids, labels, train_size=train_frac, stratify=labels, random_state=seed,
        )
        val_rel_frac = val_frac / (val_frac + test_frac)
        val_ids, test_ids = train_test_split(
            rest_ids, train_size=val_rel_frac, stratify=rest_labels, random_state=seed,
        )
    except ValueError:
        # A class too small to stratify at this fraction (e.g. only 1-2
        # cases) -- fall back to a plain random split for this fold.
        train_ids, rest_ids = train_test_split(case_ids, train_size=train_frac, random_state=seed)
        val_rel_frac = val_frac / (val_frac + test_frac)
        val_ids, test_ids = train_test_split(rest_ids, train_size=val_rel_frac, random_state=seed)

    return {"train": sorted(train_ids.tolist()), "val": sorted(val_ids.tolist()), "test": sorted(test_ids.tolist())}


def fold_tagged_path(path: str, fold) -> str:
    """Inserts `_fold{fold}` before the extension, unless it's already there
    -- so output files (checkpoints, predictions, plots, ...) for different
    folds never collide, whether the base path was left at its default or
    set explicitly."""
    root, ext = os.path.splitext(path)
    tag = f"_fold{fold}"
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
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError(f"train/val/test fractions must sum to 1.0, got {train_frac + val_frac + test_frac}")

    df = pd.read_csv(metadata_csv)
    df.columns = df.columns.str.strip().str.lower()
    if "case_id" not in df.columns or "lirads_score" not in df.columns:
        raise ValueError(f"{metadata_csv} must have 'case_id' and 'lirads_score' columns")

    folds = {}
    for fold_idx in range(n_folds):
        folds[str(fold_idx)] = _split_case_ids(df, train_frac, val_frac, test_frac, seed=seed + fold_idx)
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Build N-fold stratified train/val/test splits, saved as JSON")
    parser.add_argument("--metadata_csv", required=True, help="CSV with case_id + lirads_score columns")
    parser.add_argument("--n_folds", type=int, required=True, help="number of independent splits to generate")
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="output JSON path")
    args = parser.parse_args()

    folds = make_folds(
        args.metadata_csv, args.n_folds, args.train_frac, args.val_frac, args.test_frac, args.seed,
    )
    with open(args.out, "w") as f:
        json.dump(folds, f, indent=2)

    for fold_idx, split in folds.items():
        print(f"fold {fold_idx}: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")
    print(f"wrote {len(folds)} folds to {args.out}")


if __name__ == "__main__":
    main()
