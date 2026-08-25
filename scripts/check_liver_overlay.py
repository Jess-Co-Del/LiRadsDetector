"""Quick visual sanity-check: CT slices with the liver segmentation mask
(scripts/segment_livers.py's output) overlaid as a red contour, saved to one
PNG grid (rows = cases, columns = slices spread evenly across the liver's
z-extent).

Meant to be a cheap first check before trusting the liver mask to drive
lesion_transplant.py's paste placement -- run this on a handful of cases
right after segment_livers.py, before running it over the whole dataset. Also
prints each case's liver voxel count so a totally-empty or wildly-off mask is
obvious even without opening the image.

Usage:
    python -m scripts.check_liver_overlay \
        --data_root ./data/cases \
        --case_id CASE00001 CASE00002 CASE00003 \
        --out liver_overlay.png
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lirads_model import config, preprocessing  # noqa: E402
from lirads_model.dataset import _find_case_dir  # noqa: E402


def _pick_slice_indices(mask: np.ndarray, n_slices: int) -> list:
    """Evenly spaced z-indices across the liver mask's own z-extent -- not
    preprocessing.lesion_slice_indices, which is sized/centered for a lesion
    crop, not for surveying an entire liver."""
    other_axes = tuple(a for a in range(mask.ndim) if a != config.SLICE_AXIS)
    z_with_liver = np.where(mask.sum(axis=other_axes) > 0)[0]
    if len(z_with_liver) == 0:
        raise ValueError("liver mask has no positive voxels")
    picks = np.linspace(z_with_liver.min(), z_with_liver.max(), n_slices)
    return sorted(set(int(round(p)) for p in picks))


def plot_case_overlay(case_dir: str, case_id: str, phase: str, ax_row) -> None:
    phase_paths = preprocessing.find_case_phase_paths(case_dir, case_id)
    ct_path = phase_paths.get(phase)
    if not ct_path or not os.path.exists(ct_path):
        raise FileNotFoundError(f"{case_id}: no {phase} phase volume at {ct_path}")
    liver_path = preprocessing.find_case_liver_path(case_dir)
    if not os.path.exists(liver_path):
        raise FileNotFoundError(f"{case_id}: no liver mask at {liver_path} -- run scripts/segment_livers.py first")

    ct_vol = preprocessing.load_volume(ct_path)
    liver_vol = preprocessing.load_volume(liver_path) > 0.5
    liver_vol = preprocessing._resample_to_shape(liver_vol, ct_vol.shape) > 0.5
    print(f"{case_id}: liver voxels = {int(liver_vol.sum())} / {liver_vol.numel()} ({100 * liver_vol.float().mean():.1f}%)")

    z_indices = _pick_slice_indices(liver_vol, len(ax_row))
    for i, ax in enumerate(ax_row):
        if i >= len(z_indices):
            ax.axis("off")
            continue
        z = z_indices[i]
        ct_slice = np.clip(ct_vol[:, :, z], config.WINDOW_LOW, config.WINDOW_HIGH)
        ax.imshow(ct_slice.T, cmap="gray", origin="lower")
        ax.contour(liver_vol[:, :, z].T, levels=[0.5], colors="red", linewidths=1)
        ax.set_title(f"{case_id} z={z}", fontsize=8)
        ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", required=True, help="root dir containing extracted case folders")
    parser.add_argument("--case_id", nargs="+", required=True, help="one or more case_ids to check")
    parser.add_argument("--phase", default=config.LIVER_SEGMENTATION_PHASE, choices=config.PHASE_NAMES)
    parser.add_argument("--n_slices", type=int, default=6, help="slices per case, evenly spread across the liver's z-extent")
    parser.add_argument("--out", default="liver_overlay.png")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(args.case_id), args.n_slices,
        figsize=(2.2 * args.n_slices, 2.4 * len(args.case_id)), squeeze=False,
    )

    for case_id, ax_row in zip(args.case_id, axes):
        case_dir = _find_case_dir(args.data_root, case_id)
        plot_case_overlay(case_dir, case_id, args.phase, ax_row)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"saved overlay to {args.out}")


if __name__ == "__main__":
    main()
