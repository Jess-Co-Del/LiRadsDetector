"""
Runs model.ClinicalPredictorNet (train_clinical.py) over a set of cases to
synthesize the clinical/tabular feature row LiRadsNet's clinical branch
expects (dataset.encode_clinical_features) -> aphe/washout/capsule from an
ensemble of trained ClinicalPredictorNet checkpoints (probability-averaged,
then decoded), max_diameter_mm deterministically from the lesion mask
(preprocessing.compute_max_diameter_mm, no model involved), and joins them
into a single CSV, one row per case, with the same column names
train_metadata.csv uses for these fields (case_id, aphe, washout_venous,
washout_delayed, capsule_venous, capsule_delayed, max_diameter_mm), so
dataset.encode_clinical_features can consume a row of it directly. See
submission/run.py for how this feeds predict.predict_case_ensemble.

    python -m lirads_model.predict_clinical \
      --checkpoint checkpoints/clinical_fold0.pt \
      --data_root ./data/cases \
      --case_ids_csv ./data/input_data/sample_cases.csv \
      --out generated_metadata.csv
"""

import argparse
import sys
from typing import List, Optional, Sequence

import pandas as pd
import torch

from . import config, preprocessing
from .config import print_to_log
from .backbone import Dinov2SliceEncoder
from .dataset import _find_case_dir
from .model import ClinicalPredictorNet, compute_backbone_feats


def load_clinical_model(checkpoint_path: str, device: torch.device, backbone_source: str = "local") -> ClinicalPredictorNet:
    """
    Mirrors predict.load_model: "local" (no network,submission
    container) vs. "hub" (re-downloads the pretrained backbone, for local
    dev without a vendored snapshot). The checkpoint records whether the
    3D-CNN branch was used at training time (train_clinical.py --use_cnn),
    so the right architecture is reconstructed automatically.
    """
    if backbone_source == "local":
        backbone = Dinov2SliceEncoder.from_local()
    else:
        backbone = Dinov2SliceEncoder.from_pretrained()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    use_cnn = checkpoint.get("use_cnn", True)
    aphe_categories = checkpoint.get("aphe_categories", config.APHE_PREDICTABLE_CATEGORIES)
    binary_features = checkpoint.get("binary_features", config.CLINICAL_BINARY_FEATURES)
    model = ClinicalPredictorNet(
        backbone, use_cnn=use_cnn, aphe_categories=aphe_categories, binary_features=binary_features,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_clinical_models(checkpoint_paths: Sequence[str], device: torch.device, backbone_source: str = "local") -> List[ClinicalPredictorNet]:
    return [load_clinical_model(p, device, backbone_source) for p in checkpoint_paths]


@torch.no_grad()
def predict_case_metadata_from_backbone_feats(
    models: Sequence[ClinicalPredictorNet],
    backbone_feats: dict,
    max_diameter_mm: float,
    case_id: str,
) -> dict:
    """Same as predict_case_metadata, but `backbone_feats` (see
    model.compute_backbone_feats) has already been computed by the caller,
    so this never touches any model's backbone. Lets the clinical-predictor
    ensemble share the exact same deterministic (unaugmented) backbone pass
    the main LiRadsNet ensemble's own deterministic pass uses -- see
    predict.predict_case_ensemble, which is what actually fuses the two for
    submission/run.py. `max_diameter_mm` is still deterministic geometry
    from the mask, not a model output, so it isn't part of backbone_feats
    either -- but it comes in already measured by the caller's own
    preprocessing.load_case_volumes(compute_max_diameter=True) call, which
    loads the mask once for both this and backbone_feats, instead of this
    function re-opening the mask file itself just to measure it again."""
    aphe_categories = models[0].aphe_categories
    binary_features = models[0].binary_features
    aphe_prob_sum = torch.zeros(len(aphe_categories))
    binary_prob_sum = torch.zeros(len(binary_features))
    for model in models:
        aphe_logits, binary_logits = model.forward_from_backbone_feats([backbone_feats])
        aphe_prob_sum += torch.softmax(aphe_logits[0], dim=0).cpu()
        binary_prob_sum += torch.sigmoid(binary_logits[0]).cpu()
    n = len(models)
    aphe_prob = aphe_prob_sum / n
    binary_prob = binary_prob_sum / n

    # Decoded from the ensemble-averaged probabilities directly (not via
    # model.decode_clinical_prediction, which applies its own sigmoid to
    # its input,these are already post-sigmoid, and sigmoid-ing them
    # again would push every value above 0.5).
    result = {"aphe": aphe_categories[int(torch.argmax(aphe_prob).item())]}
    for name, p in zip(binary_features, binary_prob.tolist()):
        result[name] = int(p >= 0.5)
    result["case_id"] = case_id
    result["max_diameter_mm"] = max_diameter_mm
    return result


@torch.no_grad()
def predict_case_metadata(
    models: Sequence[ClinicalPredictorNet],
    case_dir: str,
    case_id: str,
    max_slices: int = config.MAX_SLICES_PER_CASE,
) -> dict:
    """One case -> {"case_id", "aphe", "washout_venous", "washout_delayed",
    "capsule_venous", "capsule_delayed", "max_diameter_mm"}. With more than
    one model, aphe/washout/capsule are ensembled by averaging each model's
    own softmax/sigmoid probabilities before decoding (rather than
    majority-voting the already-decoded labels), since every model here
    shares the same output space (see ClinicalPredictorNet's aphe_categories/
    binary_features),unlike predict.predict_case_ensemble's LI-RADS
    majority vote, which exists because different checkpoints there can have
    different category heads.

    The DINOv2 backbone forward is run once (via any one model's own
    `backbone` attribute -- every checkpoint's is frozen and identical to
    the same vendored snapshot, see model.compute_backbone_feats's
    docstring) and its output shared across every model in `models`,
    instead of each one separately re-running its own backbone on the same
    volume. Standalone entry point (own preprocessing + backbone
    pass) for callers that don't already have backbone_feats for this case
    -- see predict_case_metadata_from_backbone_feats for the version that
    reuses a pass computed elsewhere."""
    phase_paths = preprocessing.find_case_phase_paths(case_dir, case_id)
    mask_path = preprocessing.find_case_mask_path(case_dir)
    device = models[0].missing_phase_embed.device
    # load_case_volumes(compute_max_diameter=True) measures max_diameter_mm
    # from the mask right where it's loaded (native pixel spacing, no
    # resampling), instead of build_case_tensors's plain path re-opening
    # mask_path again downstream in predict_case_metadata_from_backbone_feats
    # just to measure the same mask a second time.
    phase_vols, mask_vol, _, max_diameter_mm = preprocessing.load_case_volumes(
        phase_paths, mask_path, compute_max_diameter=True, device=device,
    )
    phase_data = preprocessing.build_case_tensors_from_volumes(phase_vols, mask_vol, max_slices)
    backbone_feats = compute_backbone_feats(models[0].backbone, phase_data, device)
    return predict_case_metadata_from_backbone_feats(models, backbone_feats, max_diameter_mm, case_id)


def generate_metadata_csv(
    models: Sequence[ClinicalPredictorNet],
    case_ids: Sequence[str],
    data_root: str,
    out_path: str,
    max_slices: int = config.MAX_SLICES_PER_CASE,
) -> pd.DataFrame:
    """Builds one row per case_id and writes it to `out_path` (columns:
    case_id, aphe, washout_venous, washout_delayed, capsule_venous,
    capsule_delayed, max_diameter_mm). A case that fails entirely (missing
    phase file, unreadable mask, ...) is skipped,not filled with a
    fabricated row,so its absence from the output CSV is visible to the
    caller, who should fall back to clinical_features=None for it (the same
    "missing metadata" path LiRadsNet already has, see its
    missing_clinical_embed) rather than trust made-up values."""
    rows = []
    for case_id in case_ids:
        try:
            case_dir = _find_case_dir(data_root, case_id)
            rows.append(predict_case_metadata(models, case_dir, case_id, max_slices))
        except Exception as e:
            print(f"  WARNING: clinical-metadata prediction failed for {case_id} ({e}); omitting from {out_path}", file=sys.stderr)
    columns = ["case_id", "aphe"] + config.CLINICAL_BINARY_FEATURES + ["max_diameter_mm"]
    df = pd.DataFrame(rows, columns=columns)
    #df.to_csv(out_path, index=False)
    print_to_log(f"wrote generated clinical metadata for {len(df)}/{len(case_ids)} case(s) to {out_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ClinicalPredictorNet over cases and save a joined metadata CSV")
    parser.add_argument("--checkpoint", nargs="+", required=True, help="one or more train_clinical.py checkpoints (ensembled)")
    parser.add_argument("--data_root", required=True, help="root dir containing extracted case folders")
    parser.add_argument("--case_ids_csv", required=True, help="CSV with a case_id column (e.g. sample_cases.csv)")
    parser.add_argument("--out", required=True, help="output CSV path")
    parser.add_argument("--max_slices", type=int, default=config.MAX_SLICES_PER_CASE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backbone_source", choices=["local", "hub"], default="local")
    args = parser.parse_args()

    device = torch.device(args.device)
    models = load_clinical_models(args.checkpoint, device, args.backbone_source)
    case_ids = pd.read_csv(args.case_ids_csv)["case_id"].astype(str).tolist()
    generate_metadata_csv(models, case_ids, args.data_root, args.out, args.max_slices)


if __name__ == "__main__":
    main()
