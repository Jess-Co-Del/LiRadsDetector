"""Quick visual sanity-check for lesion_transplant.py: for each donor case,
runs the real transplant_case() pipeline (donor lesion pasted into a random
recipient's liver) and saves a PNG grid so a botched paste,a visible seam,
a lesion clipped at the liver boundary, an obviously wrong location,is
easy to spot before trusting the augmentation in training.

Each row is one donor: the first column is the donor's own lesion as it
really looks (before extraction), the remaining columns are slices through
the synthesized (pasted) case, evenly spread across the pasted lesion's
z-extent, cropped tight around it like the model's own input crop, with the
lesion mask overlaid as a red contour.

Usage:
    python -m scripts.check_transplant \
        --data_root ./data/cases \
        --metadata_csv ./train_metadata.csv \
        --donor_case_id CASE00003 CASE00017 \
        --out transplant_check.png

    # or let it sample donors itself (LR-1/LR-2/LR-3 cases, see
    # config.TRANSPLANT_DONOR_LABELS) and auto-pick recipients:
    python -m scripts.check_transplant \
        --data_root ./data/cases --metadata_csv ./train_metadata.csv \
        --n_donors 4 --out transplant_check.png
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lirads_model import config, lesion_transplant, preprocessing  # noqa: E402
from lirads_model.dataset import _find_case_dir  # noqa: E402


def _pick_slice_indices(mask: np.ndarray, n_slices: int) -> list:
    """Evenly spaced z-indices across mask's own z-extent,like
    scripts/check_liver_overlay.py's helper, but for the pasted lesion mask
    rather than the liver mask, so the columns actually walk through the
    transplanted lesion instead of the recipient's whole liver."""
    other_axes = tuple(a for a in range(mask.ndim) if a != config.SLICE_AXIS)
    z_with_lesion = np.where(mask.sum(axis=other_axes) > 0)[0]
    if len(z_with_lesion) == 0:
        raise ValueError("pasted mask has no positive voxels")
    picks = np.linspace(z_with_lesion.min(), z_with_lesion.max(), n_slices)
    return sorted(set(int(round(p)) for p in picks))


def _crop_and_show(ax, vol: np.ndarray, mask: np.ndarray, z: int, title: str) -> None:
    ct_slice = preprocessing._get_slice(vol, z)
    mask_slice = preprocessing._get_slice(mask, z)
    r0, r1, c0, c1 = preprocessing._crop_bbox_from_mask(mask_slice)
    ct_crop = np.clip(ct_slice[r0:r1, c0:c1], config.WINDOW_LOW, config.WINDOW_HIGH)
    mask_crop = mask_slice[r0:r1, c0:c1]
    ax.imshow(ct_crop.T, cmap="gray", origin="lower")
    if mask_crop.any():
        ax.contour(mask_crop.T, levels=[0.5], colors="red", linewidths=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def pick_donor_case_ids(df: pd.DataFrame, n_donors: int, rng: np.random.Generator) -> list:
    donors = df[df["lirads_score"].str.strip().isin(config.TRANSPLANT_DONOR_LABELS)]
    if len(donors) == 0:
        raise ValueError(f"no cases in metadata with a label in {config.TRANSPLANT_DONOR_LABELS}")
    n = min(n_donors, len(donors))
    return donors["case_id"].astype(str).sample(n=n, random_state=int(rng.integers(2**31))).tolist()


def plot_transplant_row(
    data_root: str,
    df: pd.DataFrame,
    donor_case_id: str,
    recipient_case_id: str,
    phase: str,
    n_slices: int,
    rng: np.random.Generator,
    ax_row,
    max_recipient_attempts: int,
) -> None:
    donor_dir = _find_case_dir(data_root, donor_case_id)
    donor_label = str(df.loc[df["case_id"].astype(str) == donor_case_id, "lirads_score"].iloc[0]).strip()

    donor_phase_paths = preprocessing.find_case_phase_paths(donor_dir, donor_case_id)
    donor_mask_path = preprocessing.find_case_mask_path(donor_dir)
    donor_phase_vols, donor_mask_vol, _ = preprocessing.load_case_volumes(donor_phase_paths, donor_mask_path)
    donor_z = int(_pick_slice_indices(donor_mask_vol, 1)[0])
    _crop_and_show(ax_row[0], donor_phase_vols[phase], donor_mask_vol, donor_z, f"{donor_case_id} ({donor_label})\ndonor original")

    tried_recipients = [recipient_case_id] if recipient_case_id else []
    last_error = None
    phase_vols = mask_vol = rid = None
    attempt = -1
    for attempt in range(max_recipient_attempts):
        rid = tried_recipients[attempt] if attempt < len(tried_recipients) else lesion_transplant.find_recipient_case_id(
            df, donor_case_id, rng,
        )
        if rid is None:
            last_error = ValueError("no eligible recipient case in metadata")
            break
        try:
            recipient_dir = _find_case_dir(data_root, rid)
            phase_vols, mask_vol, _ = lesion_transplant.transplant_case(donor_dir, donor_case_id, recipient_dir, rid, rng)
            break
        except (FileNotFoundError, ValueError) as e:
            last_error = e
            phase_vols = mask_vol = None

    if phase_vols is None:
        for ax in ax_row[1:]:
            ax.axis("off")
        ax_row[1].set_title(f"transplant failed:\n{last_error}", fontsize=8, color="red")
        print(f"{donor_case_id}: FAILED after {attempt + 1} recipient attempt(s),{last_error}")
        return

    print(f"{donor_case_id} -> {rid}: pasted lesion voxels = {int(mask_vol.sum())}")
    z_indices = _pick_slice_indices(mask_vol, len(ax_row) - 1)
    for i, ax in enumerate(ax_row[1:]):
        if i >= len(z_indices):
            ax.axis("off")
            continue
        z = z_indices[i]
        _crop_and_show(ax, phase_vols[phase], mask_vol, z, f"-> {rid} z={z}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", required=True, help="root dir containing extracted case folders")
    parser.add_argument("--metadata_csv", required=True, help="CSV with case_id + lirads_score for the whole dataset")
    parser.add_argument("--donor_case_id", nargs="+", default=None, help="donor case_ids to transplant from; if omitted, sampled from LR-1/LR-2/LR-3 cases")
    parser.add_argument("--recipient_case_id", nargs="+", default=None, help="one recipient case_id per donor_case_id; if omitted, auto-picked (with retries) like training does")
    parser.add_argument("--n_donors", type=int, default=4, help="how many donors to sample when --donor_case_id is omitted")
    parser.add_argument("--phase", default=config.PHASE_NAMES[0], choices=config.PHASE_NAMES)
    parser.add_argument("--n_slices", type=int, default=5, help="slices through the pasted lesion per row, plus one more for the donor original")
    parser.add_argument("--max_recipient_attempts", type=int, default=5, help="recipient candidates to try per donor before giving up")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="transplant_check.png")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.metadata_csv)
    df.columns = df.columns.str.strip().str.lower()

    donor_case_ids = args.donor_case_id or pick_donor_case_ids(df, args.n_donors, rng)
    if args.recipient_case_id and len(args.recipient_case_id) != len(donor_case_ids):
        raise ValueError(f"--recipient_case_id has {len(args.recipient_case_id)} entries but there are {len(donor_case_ids)} donors")
    recipient_case_ids = args.recipient_case_id or [None] * len(donor_case_ids)

    n_cols = args.n_slices + 1
    fig, axes = plt.subplots(
        len(donor_case_ids), n_cols,
        figsize=(2.2 * n_cols, 2.4 * len(donor_case_ids)), squeeze=False,
    )

    for donor_id, recipient_id, ax_row in zip(donor_case_ids, recipient_case_ids, axes):
        plot_transplant_row(
            args.data_root, df, donor_id, recipient_id, args.phase, args.n_slices,
            rng, ax_row, args.max_recipient_attempts,
        )

    fig.tight_layout(h_pad=2.5)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"saved transplant check to {args.out}")


if __name__ == "__main__":
    main()
