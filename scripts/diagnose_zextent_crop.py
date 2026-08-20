"""
Checks whether preprocessing.lesion_slice_indices's fixed-size, center-trimmed
slice window (config.MAX_SLICES_PER_CASE) is cutting off part of the lesion
mask, broken down by lirads_score.

Motivation: LR-TIV (tumor extending into a vein) is structurally the class
most likely to have an elongated, asymmetric z-footprint -- primary mass at
one end, thrombus trailing toward the vessel at the other -- unlike a
roughly-spherical, z-compact LR-M/LR-5 nodule. Center-trimming a z-range
longer than max_slices keeps the middle and discards the periphery, which
for LR-TIV is disproportionately likely to be the vein-invasion segment
itself. This script measures, per case, the mask's full z-extent vs. what
lesion_slice_indices actually retains, and what fraction of mask voxels fall
outside the retained window -- then aggregates by label so you can see
whether LR-TIV is hit harder than other classes.

Usage:
    python -m scripts.diagnose_zextent_crop \
        --metadata_csv ./train_metadata.csv \
        --data_root ./data/cases
"""

import argparse, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lirads_model import config, preprocessing
from lirads_model.dataset import _find_case_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--max_slices", type=int, default=config.MAX_SLICES_PER_CASE)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    df.columns = df.columns.str.strip().str.lower()

    rows = []
    for _, row in df.iterrows():
        case_id = str(row["case_id"])
        label = str(row["lirads_score"]).strip()
        if label == config.NO_LESION_LABEL:
            continue

        try:
            case_dir = _find_case_dir(args.data_root, case_id)
            mask_path = preprocessing.find_case_mask_path(case_dir)
            mask = preprocessing.load_volume(mask_path) > 0.5
        except Exception as e:
            print(f"skip {case_id} ({label}): {e}")
            continue

        other_axes = tuple(a for a in range(mask.ndim) if a != config.SLICE_AXIS)
        z_with_lesion = np.where(mask.sum(axis=other_axes) > 0)[0]
        if len(z_with_lesion) == 0:
            continue
        z_min, z_max = int(z_with_lesion.min()), int(z_with_lesion.max())
        full_extent = z_max - z_min + 1

        kept_z = preprocessing.lesion_slice_indices(mask, args.max_slices)
        total_voxels = int(mask.sum())
        kept_voxels = sum(int(np.take(mask, int(z), axis=config.SLICE_AXIS).sum()) for z in kept_z)
        dropped_frac = 1.0 - (kept_voxels / total_voxels if total_voxels else 1.0)

        rows.append({
            "case_id": case_id,
            "label": label,
            "full_z_extent": full_extent,
            "truncated": full_extent > args.max_slices,
            "mask_voxels_dropped_frac": dropped_frac,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        print("no cases with a loadable mask were found -- check --data_root")
        return

    summary = out.groupby("label").agg(
        n_cases=("case_id", "count"),
        mean_full_z_extent=("full_z_extent", "mean"),
        pct_truncated=("truncated", "mean"),
        mean_voxels_dropped_frac=("mask_voxels_dropped_frac", "mean"),
    ).sort_values("mean_voxels_dropped_frac", ascending=False)
    summary["pct_truncated"] = (summary["pct_truncated"] * 100).round(1)
    summary["mean_voxels_dropped_frac"] = (summary["mean_voxels_dropped_frac"] * 100).round(2)
    summary["mean_full_z_extent"] = summary["mean_full_z_extent"].round(1)
    print(summary.rename(columns={
        "pct_truncated": "pct_truncated_%",
        "mean_voxels_dropped_frac": "mean_voxels_dropped_%",
    }))


if __name__ == "__main__":
    main()
