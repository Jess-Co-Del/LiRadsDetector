"""
Generates a genuine N-way stratified k-fold partition over a metadata CSV
and saves it to a single JSON file, keyed by fold index.

The outer split (test) is a `StratifiedKFold` (by lirads_score) over cases
outside `full_inclusion_labels`: each such case's `test`-fold membership is
fixed and the N test sets are disjoint and cover every one of these cases
exactly once, unlike independent random draws. Within each fold's non-test
remainder, train/val is a further stratified random split at `val_frac`.

Cases whose label is in `full_inclusion_labels` (LR-1/2/3 by default,too
few per class to hold out a disjoint test slice from) are instead added to
*every* fold's train and test sets in full, with a proportional-by-class
subset (at least one case per class) also added to val. Use `--fold` in
train.py to pick one fold.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

DEFAULT_FULL_INCLUSION_LABELS = ("LR-1", "LR-2", "LR-3")


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
        # cases),fall back to a plain random split.
        train_ids, val_ids = train_test_split(case_ids, test_size=val_frac, random_state=seed)
    return sorted(train_ids.tolist()), sorted(val_ids.tolist())


def _sample_val_ids(case_ids: np.ndarray, labels: np.ndarray, val_frac: float, seed: int) -> list:
    """Picks a proportional-by-class subset of case_ids for val, keeping at
    least one case per class. Used for full_inclusion_labels classes, which
    are otherwise added to train/test in full rather than split."""
    rng = np.random.RandomState(seed)
    val_ids = []
    for label in np.unique(labels):
        class_ids = case_ids[labels == label]
        k = min(len(class_ids), max(1, round(val_frac * len(class_ids))))
        val_ids.extend(rng.choice(class_ids, size=k, replace=False).tolist())
    return val_ids


def fold_tagged_path(path: str, fold) -> str:
    """Inserts `_fold{fold}` before the extension, unless it's already there
   ,so output files (checkpoints, predictions, plots, ...) for different
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
    full_inclusion_labels=DEFAULT_FULL_INCLUSION_LABELS,
) -> dict:
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")

    df = pd.read_csv(metadata_csv)
    df.columns = df.columns.str.strip().str.lower()
    if "case_id" not in df.columns or "lirads_score" not in df.columns:
        raise ValueError(f"{metadata_csv} must have 'case_id' and 'lirads_score' columns")

    is_full_inclusion = df["lirads_score"].isin(set(full_inclusion_labels))

    stratified_df = df[~is_full_inclusion]
    case_ids = stratified_df["case_id"].astype(str).to_numpy()
    labels = stratified_df["lirads_score"].to_numpy()

    full_case_ids = df.loc[is_full_inclusion, "case_id"].astype(str).to_numpy()
    full_labels = df.loc[is_full_inclusion, "lirads_score"].to_numpy()

    folds = {}
    for fold_idx, (trainval_idx, test_idx) in enumerate(_make_test_folds(case_ids, labels, n_folds, seed)):
        train_ids, val_ids = _split_case_ids(
            case_ids[trainval_idx], labels[trainval_idx], val_frac, seed=seed + fold_idx,
        )
        full_val_ids = (
            _sample_val_ids(full_case_ids, full_labels, val_frac, seed=seed + fold_idx)
            if len(full_case_ids) else []
        )
        folds[str(fold_idx)] = {
            "train": sorted(set(train_ids) | set(full_case_ids.tolist())),
            "val": sorted(set(val_ids) | set(full_val_ids)),
            "test": sorted(set(case_ids[test_idx].tolist()) | set(full_case_ids.tolist())),
        }
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
    parser.add_argument(
        "--full_inclusion_labels", default=",".join(DEFAULT_FULL_INCLUSION_LABELS),
        help=(
            "comma-separated lirads_score labels added in full to every fold's "
            "train and test (too rare to hold out a disjoint test slice from), "
            "with a proportional-by-class, at-least-one-per-class subset also "
            "added to val"
        ),
    )
    parser.add_argument("--out", required=True, help="output JSON path")
    args = parser.parse_args()

    full_inclusion_labels = tuple(l.strip() for l in args.full_inclusion_labels.split(",") if l.strip())
    folds = make_folds(args.metadata_csv, args.n_folds, args.val_frac, args.seed, full_inclusion_labels)
    with open(args.out, "w") as f:
        json.dump(folds, f, indent=2)

    for fold_idx, split in folds.items():
        print(f"fold {fold_idx}: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")
    print(f"wrote {len(folds)} folds to {args.out}")


if __name__ == "__main__":
    main()
