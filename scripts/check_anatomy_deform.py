"""Quick visual sanity-check for augmentation.apply_anatomy_informed_deform:
runs the real anatomy-informed deform on a case's own volumes and saves a
before/after PNG grid so it's easy to see that the local warp is distending/
compressing the *lesion* (not the liver, and not the whole image globally).

For each case the grid has two rows,the original volume and the deformed
one,sharing the same slice indices and the same tight crop around the
lesion, with the lesion mask overlaid as a red contour. The "after" row also
draws the original lesion outline in dashed cyan, so the shape change is
visible at a glance. Each case's sampled warp magnitude and its lesion voxel
count before/after are printed.

Usage:
    python -m scripts.check_anatomy_deform \
        --data_root ./data/cases \
        --case_id CASE00001 CASE00002 \
        --out anatomy_deform_check.png

    # force a specific warp magnitude (voxels; positive distends, negative
    # compresses) instead of sampling from config.ANATOMY_DILATION_RANGE_VOX:
    python -m scripts.check_anatomy_deform \
        --data_root ./data/cases --case_id CASE00001 --dilation 12 \
        --out anatomy_deform_check.png

    # or let it sample lesion cases itself from the metadata csv:
    python -m scripts.check_anatomy_deform \
        --data_root ./data/cases --metadata_csv ./train_metadata.csv \
        --n_cases 4 --out anatomy_deform_check.png
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lirads_model import augmentation, config, preprocessing  # noqa: E402
from lirads_model.dataset import _find_case_dir  # noqa: E402


def _lesion_z_indices(mask: np.ndarray, n_slices: int) -> list:
    """Evenly spaced z-indices across the lesion mask's own z-extent,same
    idea as scripts/check_transplant.py's helper."""
    other_axes = tuple(a for a in range(mask.ndim) if a != config.SLICE_AXIS)
    z_with_lesion = np.where(mask.sum(axis=other_axes) > 0)[0]
    if len(z_with_lesion) == 0:
        raise ValueError("lesion mask has no positive voxels")
    picks = np.linspace(z_with_lesion.min(), z_with_lesion.max(), n_slices)
    return sorted(set(int(round(p)) for p in picks))


def _shared_inplane_bbox(masks: list, margin_frac: float = 0.6) -> tuple:
    """In-plane (row, col) bounding box covering every mask in `masks`
    (each a full 3D bool array, SLICE_AXIS last), padded by margin_frac of
    the larger side. Shared by the before/after panels so the lesion is
    framed identically and the deformation,not a shifting crop,is what
    moves on screen."""
    combined = np.zeros(
        tuple(s for a, s in enumerate(masks[0].shape) if a != config.SLICE_AXIS), dtype=bool
    )
    for m in masks:
        combined |= m.any(axis=config.SLICE_AXIS)
    rows = np.where(combined.any(axis=1))[0]
    cols = np.where(combined.any(axis=0))[0]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    pad = int(round(max(r1 - r0, c1 - c0) * margin_frac)) + 4
    H, W = combined.shape
    return (
        max(r0 - pad, 0), min(r1 + pad + 1, H),
        max(c0 - pad, 0), min(c1 + pad + 1, W),
    )


def _show(ax, ct_slice, mask_slice, bbox, title, ref_mask_slice=None) -> None:
    r0, r1, c0, c1 = bbox
    ct_crop = np.clip(ct_slice[r0:r1, c0:c1], config.WINDOW_LOW, config.WINDOW_HIGH)
    ax.imshow(ct_crop.T, cmap="gray", origin="lower")
    if ref_mask_slice is not None and ref_mask_slice[r0:r1, c0:c1].any():
        ax.contour(ref_mask_slice[r0:r1, c0:c1].T, levels=[0.5], colors="cyan", linewidths=0.8, linestyles="dashed")
    if mask_slice[r0:r1, c0:c1].any():
        ax.contour(mask_slice[r0:r1, c0:c1].T, levels=[0.5], colors="red", linewidths=1.0)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def plot_case(case_dir: str, case_id: str, phase: str, n_slices: int, rng: np.random.Generator, ax_rows) -> None:
    phase_paths = preprocessing.find_case_phase_paths(case_dir, case_id)
    mask_path = preprocessing.find_case_mask_path(case_dir)
    phase_vols, mask_vol, _, _ = preprocessing.load_case_volumes(phase_paths, mask_path)
    if phase not in phase_vols:
        raise ValueError(f"{case_id}: phase {phase!r} not among {list(phase_vols)}")

    # apply_anatomy_informed_deform's only rng draw is the warp magnitude
    # (rng.uniform(*ANATOMY_DILATION_RANGE_VOX)); seed a dedicated generator
    # so we can report the exact magnitude it will use.
    lo, hi = config.ANATOMY_DILATION_RANGE_VOX
    deform_seed = int(rng.integers(2**31))
    dil = float(np.random.default_rng(deform_seed).uniform(lo, hi))

    new_phase_vols, new_mask_vol = augmentation.apply_anatomy_informed_deform(
        {p: v.clone() for p, v in phase_vols.items()}, mask_vol.clone(), np.random.default_rng(deform_seed),
    )

    before_m = mask_vol.numpy().astype(bool)
    after_m = new_mask_vol.numpy().astype(bool)
    print(
        f"{case_id}: warp magnitude = {dil:+.1f} vox | lesion voxels {int(before_m.sum())} -> {int(after_m.sum())} "
        f"({100 * (after_m.sum() / max(before_m.sum(), 1) - 1):+.0f}%)"
    )

    z_indices = _lesion_z_indices(before_m, n_slices)
    bbox = _shared_inplane_bbox([before_m, after_m])
    before_ct, after_ct = phase_vols[phase].numpy(), new_phase_vols[phase].numpy()

    for i, ax in enumerate(ax_rows[0]):
        if i >= len(z_indices):
            ax.axis("off")
            continue
        z = z_indices[i]
        _show(ax, before_ct[:, :, z], before_m[:, :, z], bbox, f"{case_id} z={z}\noriginal")
    for i, ax in enumerate(ax_rows[1]):
        if i >= len(z_indices):
            ax.axis("off")
            continue
        z = z_indices[i]
        _show(
            ax, after_ct[:, :, z], after_m[:, :, z], bbox,
            f"z={z}  deform {dil:+.1f}vox", ref_mask_slice=before_m[:, :, z],
        )


def pick_case_ids(metadata_csv: str, n_cases: int, rng: np.random.Generator) -> list:
    import pandas as pd

    df = pd.read_csv(metadata_csv)
    df.columns = df.columns.str.strip().str.lower()
    lesion = df[df["lirads_score"].astype(str).str.strip() != config.NO_LESION_LABEL]
    if len(lesion) == 0:
        raise ValueError("no lesion cases in metadata")
    n = min(n_cases, len(lesion))
    return lesion["case_id"].astype(str).sample(n=n, random_state=int(rng.integers(2**31))).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", required=True, help="root dir containing extracted case folders")
    parser.add_argument("--case_id", nargs="+", help="one or more case_ids to check")
    parser.add_argument("--metadata_csv", help="pick lesion cases from here when --case_id is omitted")
    parser.add_argument("--n_cases", type=int, default=4, help="how many cases to sample when using --metadata_csv")
    parser.add_argument("--phase", default="ART", choices=config.PHASE_NAMES)
    parser.add_argument("--n_slices", type=int, default=5, help="slices per case, evenly spread across the lesion z-extent")
    parser.add_argument("--dilation", type=float, help="force this warp magnitude (voxels) instead of sampling")
    parser.add_argument("--blur", type=float, help="override config.ANATOMY_BLUR (gaussian kernel smoothing the lesion gradient field)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="anatomy_deform_check.png")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    case_ids = args.case_id
    if not case_ids:
        if not args.metadata_csv:
            parser.error("pass --case_id ... or --metadata_csv to sample from")
        case_ids = pick_case_ids(args.metadata_csv, args.n_cases, rng)
        print(f"sampled cases: {case_ids}")

    if args.dilation is not None:
        config.ANATOMY_DILATION_RANGE_VOX = (args.dilation, args.dilation)
    if args.blur is not None:
        config.ANATOMY_BLUR = args.blur

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2 * len(case_ids), args.n_slices,
        figsize=(2.4 * args.n_slices, 2.6 * 2 * len(case_ids)), squeeze=False,
    )

    for k, case_id in enumerate(case_ids):
        case_dir = _find_case_dir(args.data_root, case_id)
        plot_case(case_dir, case_id, args.phase, args.n_slices, rng, axes[2 * k:2 * k + 2])

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"saved before/after grid to {args.out}")


if __name__ == "__main__":
    main()
