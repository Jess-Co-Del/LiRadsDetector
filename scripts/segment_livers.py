"""Batch-runs a pretrained nnUNetv2 liver-segmentation model over every case
in a metadata CSV, saving one liver mask per case at the layout
preprocessing.find_case_liver_path() (and lesion_transplant.py) expect:

    <case_dir>/annotations/liver.nii.gz

Run this once, offline, wherever the trained model and case data actually
live (this repo's dev sandbox has access to neither),before training with
train.py --transplant, since a recipient case needs a liver mask to
constrain paste placement to real liver tissue; a case without one on disk
is simply skipped as a candidate recipient (see
lesion_transplant.find_recipient_case_id()/transplant_case()).

Uses nnunetv2.inference.predict_from_raw_data.nnUNetPredictor
(https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/inference) directly
against an already-trained model folder,no retraining/preprocessing step,
just inference.

The output is saved verbatim (thresholded >0.5 by whatever reads it back,
see preprocessing.find_case_liver_path()'s callers): if the model is a
binary liver-vs-background segmenter this is exactly "is this voxel liver",
but if it's a LiTS-style multi-class model (0=background, 1=liver, 2=tumor)
every non-background voxel,tumor included,currently counts as
liver-plausible for paste placement. Recipients are chosen to be
config.NO_LESION_LABEL cases (see find_recipient_case_id()) so they
shouldn't have a real tumor region to begin with, but if you know this model
is multi-class and want placement restricted to label 1 specifically,
threshold on that label here before saving instead.

Usage:
    python -m scripts.segment_livers \
        --metadata_csv ./train_metadata.csv \
        --data_root /leonardo_scratch/fast/EUHPC_D35_139/nnunet_base/nnunet_format/amplifai/batch_001/cases \
        --model_dir /leonardo_scratch/fast/EUHPC_D35_139/nnunet_base/nnunet_format/nnUNet_results/Dataset003_Liver/nnUNetTrainer__nnUNetPlans__2d \
        --folds 0 1 2 3 4
"""

import argparse
import os, sys
import shutil
import tempfile
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lirads_model import config, preprocessing
from lirads_model.dataset import _find_case_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata_csv", required=True, help="CSV with a case_id column for the whole dataset")
    parser.add_argument("--data_root", required=True, help="root dir containing extracted case folders")
    parser.add_argument(
        "--model_dir", required=True,
        help="nnUNet_results/<Dataset>/<Trainer>__<Plans>__<configuration> folder for the trained liver model",
    )
    parser.add_argument("--folds", type=str, nargs="+", default="all", help="cross-val folds to ensemble")
    parser.add_argument("--checkpoint_name", default="checkpoint_final.pth")
    parser.add_argument(
        "--phase", default=config.LIVER_SEGMENTATION_PHASE, choices=config.PHASE_NAMES,
        help="which CT phase to segment (portal venous is standard for liver segmentation, matching LiTS)",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--overwrite", action="store_true", help="re-segment cases that already have a liver.nii.gz")
    parser.add_argument("--num_processes_preprocessing", type=int, default=2)
    parser.add_argument("--num_processes_export", type=int, default=2)
    args = parser.parse_args()

    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    print(f"Starting predictor")
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=(args.device == "cuda"), device=torch.device(args.device), verbose=False,
    )
    print(f"Loading predictor from {args.model_dir}")

    predictor.initialize_from_trained_model_folder(
        args.model_dir, use_folds=tuple(args.folds), checkpoint_name=args.checkpoint_name,
    )
    
    print(f"Loading cases from {args.metadata_csv}")

    df = pd.read_csv(args.metadata_csv)
    df.columns = df.columns.str.strip().str.lower()
    if "case_id" not in df.columns:
        raise ValueError(f"{args.metadata_csv} is missing a 'case_id' column")

    tmp_in = tempfile.mkdtemp(prefix="segment_livers_in_")
    tmp_out = tempfile.mkdtemp(prefix="segment_livers_out_")
    try:
        input_lists, case_ids, out_files = [], [], []
        for case_id in df["case_id"].astype(str):
            try:
                case_dir = _find_case_dir(args.data_root, case_id)
            except FileNotFoundError:
                print(f"skip {case_id}: case directory not found under {args.data_root}")
                continue

            liver_path = preprocessing.find_case_liver_path(case_dir)
            if os.path.exists(liver_path) and not args.overwrite:
                continue

            phase_paths = preprocessing.find_case_phase_paths(case_dir, case_id)
            src = phase_paths.get(args.phase)
            if not src or not os.path.exists(src):
                print(f"skip {case_id}: no {args.phase} phase volume found")
                continue

            # nnU-Net's file-based inference expects a per-case
            # modality-tagged copy (<case_id>_0000.nii.gz) in one input
            # folder; symlinking avoids duplicating the (large) CT volumes.
            linked = os.path.join(tmp_in, f"{case_id}_0000.nii.gz")
            if not os.path.exists(linked):
                os.symlink(os.path.abspath(src), linked)
            input_lists.append([linked])
            case_ids.append(case_id)
            out_files.append(os.path.join(tmp_out, case_id))

        if not input_lists:
            print("nothing to segment (every case already has a liver mask, or none were found)")
            return

        print(f"segmenting {len(input_lists)} case(s) on the {args.phase} phase using {args.model_dir} ...")
        predictor.predict_from_files(
            input_lists, out_files, save_probabilities=False, overwrite=True,
            num_processes_preprocessing=args.num_processes_preprocessing,
            num_processes_segmentation_export=args.num_processes_export,
        )

        saved = 0
        for case_id in case_ids:
            produced = os.path.join(tmp_out, f"{case_id}.nii.gz")
            if not os.path.exists(produced):
                print(f"  WARNING: {case_id}: nnUNet produced no output, skipping")
                continue
            case_dir = _find_case_dir(args.data_root, case_id)
            liver_path = preprocessing.find_case_liver_path(case_dir)
            os.makedirs(os.path.dirname(liver_path), exist_ok=True)
            shutil.move(produced, liver_path)
            saved += 1
            print(f"  {case_id}: saved {liver_path}")

        print(f"done. {saved}/{len(case_ids)} liver mask(s) saved.")
    finally:
        shutil.rmtree(tmp_in, ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)


if __name__ == "__main__":
    main()
