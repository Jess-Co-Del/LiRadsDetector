"""Shared inference logic used by both local evaluation and submission/run.py,
plus a CLI to predict one or more checkpoints on one fold's held-out test
split (majority-voted across checkpoints when more than one is given):

    python -m lirads_model.predict \
      --checkpoint checkpoints/lirads_model_fold0.pt checkpoints/lirads_model_fold0_seed1.pt \
      --data_root ./data/cases \
      --metadata_csv ./data/metadata.csv \
      --splits_json ./data/splits.json \
      --fold 0 \
      --score

By default the test split's clinical/tabular features (aphe/washout/capsule/
max_diameter_mm) come straight from --metadata_csv's own ground-truth
columns, same as training -- the "factual" run. Add --clinical_checkpoint
(one or more lirads_model.train_clinical.py checkpoints) to instead run
model.ClinicalPredictorNet over the same images and substitute *its*
predictions for those columns before predicting -- the "inferred" run,
matching what submission/run.py actually sees (the real challenge input
never supplies clinical metadata; see config's "Clinical feature
prediction" section). Run the CLI twice, with and without
--clinical_checkpoint, to compare the two directly; the inferred run's
output files get a distinct name (see --out) so neither overwrites the
other.
"""

import argparse
import os
import sys
import tempfile
import time
from collections import Counter
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from . import config, preprocessing
from .config import print_to_log
from .backbone import Dinov2SliceEncoder
from .dataset import LiRadsCaseDataset, _find_case_dir, collate_cases, encode_clinical_features
from .model import LiRadsNet, decode_prediction
from .predict_clinical import generate_metadata_csv, load_clinical_models
from .splits import fold_tagged_path, load_fold

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "amplifai-codabench"))
from evaluate import evaluate as compute_challenge_score  # noqa: E402


def load_model(checkpoint_path: str, device: torch.device, backbone_source: str = "local") -> LiRadsNet:
    """backbone_source="local": no network (submission container). "hub":
    re-downloads the pretrained backbone from the HuggingFace Hub before
    loading our trained weights on top (useful for local dev without a
    vendored snapshot).

    The checkpoint records whether the 3D-CNN, clinical, and category-head
    branches were used at training time (train.py --use_cnn/--use_clinical/
    --head_mode), so the right architecture is reconstructed automatically;
    checkpoints saved before these existed default to all branches on,
    matching their actual shape."""
    if backbone_source == "local":
        backbone = Dinov2SliceEncoder.from_local()
    else:
        backbone = Dinov2SliceEncoder.from_pretrained()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    use_cnn = checkpoint.get("use_cnn", True)
    use_clinical = checkpoint.get("use_clinical", True)
    use_cat_head = checkpoint.get("use_cat_head", True)
    ordinal_head_type = checkpoint.get("ordinal_head_type", "softmax")
    cat_names = checkpoint.get("cat_names", config.CAT_NAMES)
    model = LiRadsNet(
        backbone, use_cnn=use_cnn, use_clinical=use_clinical, use_cat_head=use_cat_head,
        ordinal_head_type=ordinal_head_type, cat_names=cat_names,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_models(checkpoint_paths: Sequence[str], device: torch.device, backbone_source: str = "local") -> List[LiRadsNet]:
    return [load_model(p, device, backbone_source) for p in checkpoint_paths]


def majority_vote(labels: Sequence[str]) -> str:
    """Most-common label; ties broken by whichever tied label appears first
    in `labels` (so, in an ensemble, by the earliest-listed model)."""
    return Counter(labels).most_common(1)[0][0]


def _preprocess_case(case_dir: str, case_id: str, max_slices: int):
    phase_paths = preprocessing.find_case_phase_paths(case_dir, case_id)
    mask_path = preprocessing.find_case_mask_path(case_dir)
    return preprocessing.build_case_tensors(phase_paths, mask_path, max_slices)


def _remap_for_submission(label: str) -> str:
    """The real challenge never scores config.NO_LESION_LABEL -- any final
    submission prediction must remap it to a valid label instead of emitting
    it literally. See config.NO_LESION_SUBMIT_LABEL."""
    return config.NO_LESION_SUBMIT_LABEL if label == config.NO_LESION_LABEL else label


@torch.no_grad()
def _forward_with_tta(
    model: LiRadsNet,
    case_dir: str,
    case_id: str,
    max_slices: int,
    tta_views: int,
    clinical_features: Optional[torch.Tensor] = None,
    rng: Optional[np.random.Generator] = None,
):
    """
    Runs one deterministic (unaugmented) forward pass to decide the
    category gate, then, only when that pass says "ordinal" (always true
    for a single-head, ordinal-only model, where logits_cat is None, see
    model.LiRadsNet's use_cat_head) and tta_views > 0, runs `tta_views`
    more forward passes on freshly augmented views of the same case
    (config.TTA_VIEWS / see augmentation.py) and averages their ordinal-head
    logits in with the deterministic pass's, before the caller decodes a
    final label. The category-gate logits are always the single
    deterministic pass's, never averaged, TTA here only steadies which of
    LR-1..LR-5 gets picked. Returns (logits_cat, logits_ord): logits_cat is
    a (4,) CPU tensor, or None for an ordinal-only model; logits_ord is a
    (5,) CPU tensor for a "softmax" ordinal head or (4,) for a "corn" one
    (see model.LiRadsNet's ordinal_head_type), the caller must decode it
    with that same model's ordinal_head_type (decode_prediction's third arg).

    `clinical_features`: this case's (1, config.CLINICAL_FEATURE_DIM) tabular
    feature row (dataset.encode_clinical_features), reused unchanged for
    every view, augmentation only perturbs the images. Must be supplied
    whenever the model was trained with the clinical branch
    (model.LiRadsNet's use_clinical) and the features are available, or the
    model falls back to its `missing_clinical_embed`, which train.py never
    exercises and so never trains. None only when they genuinely aren't
    available (e.g. submission/run.py, where the challenge supplies images
    and a mask but no metadata).
    """
    phase_paths = preprocessing.find_case_phase_paths(case_dir, case_id)
    mask_path = preprocessing.find_case_mask_path(case_dir)

    phase_data = preprocessing.build_case_tensors(phase_paths, mask_path, max_slices)
    logits_cat, logits_ord = model([phase_data], clinical_features)
    print("Lever pred:", case_id, logits_cat, logits_ord)
    logits_cat = logits_cat[0].cpu() if logits_cat is not None else None
    logits_ord = logits_ord[0].cpu()

    is_ordinal = logits_cat is None or model.cat_names[int(torch.argmax(logits_cat).item())] == "ordinal"
    if tta_views > 0 and is_ordinal:
        rng = rng if rng is not None else np.random.default_rng()
        ord_logits_sum = logits_ord.clone()
        for _ in range(tta_views):
            aug_phase_data = preprocessing.build_case_tensors(
                phase_paths, mask_path, max_slices, augment=True, rng=rng,
            )
            aug_logits_cat, aug_logits_ord = model([aug_phase_data], clinical_features)
            print("augmented pred:", aug_logits_cat, aug_logits_ord)

            ord_logits_sum += aug_logits_ord[0].cpu()
        logits_ord = ord_logits_sum / (tta_views + 1)

    return logits_cat, logits_ord


@torch.no_grad()
def predict_case(
    model: LiRadsNet,
    case_dir: str,
    case_id: str,
    device: torch.device,
    max_slices: int = config.MAX_SLICES_PER_CASE,
    tta_views: int = config.TTA_VIEWS,
    clinical_features: Optional[torch.Tensor] = None,
) -> str:
    logits_cat, logits_ord = _forward_with_tta(model, case_dir, case_id, max_slices, tta_views, clinical_features)
    label = decode_prediction(logits_cat, logits_ord, model.ordinal_head_type, model.cat_names)
    return _remap_for_submission(label)


@torch.no_grad()
def predict_case_ensemble(
    models: Sequence[LiRadsNet],
    case_dir: str,
    case_id: str,
    device: torch.device,
    max_slices: int = config.MAX_SLICES_PER_CASE,
    tta_views: int = config.TTA_VIEWS,
    clinical_features: Optional[torch.Tensor] = None,
) -> str:
    """Runs every model in `models` on the same case (each with its own TTA
    pass, see _forward_with_tta) and majority-votes over their decoded
    predictions. With a single model this is equivalent to predict_case()."""
    labels = []
    for model in models:
        logits_cat, logits_ord = _forward_with_tta(model, case_dir, case_id, max_slices, tta_views, clinical_features)
        labels.append(decode_prediction(logits_cat, logits_ord, model.ordinal_head_type, model.cat_names))
    return _remap_for_submission(majority_vote(labels))


@torch.no_grad()
def run_inference(model: LiRadsNet, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    """
    Batched inference over a LiRadsCaseDataset DataLoader. Returns a
    (case_id, prediction) DataFrame in amplifai-codabench/evaluate.py's format
    """
    model.eval()
    rows = []
    total_elapsed = 0.0
    for batch in loader:
        start = time.perf_counter()
        logits_cat, logits_ord = model(batch["phase_data"], batch["clinical_features"])
        total_elapsed += time.perf_counter() - start
        for i, case_id in enumerate(batch["case_ids"]):
            cat_i = logits_cat[i].cpu() if logits_cat is not None else None
            label = decode_prediction(cat_i, logits_ord[i].cpu(), model.ordinal_head_type, model.cat_names)
            rows.append({"case_id": case_id, "prediction": label})
    if rows:
        print_to_log(f"mean inference time per case: {total_elapsed / len(rows):.3f}s ({len(rows)} cases)")
    return pd.DataFrame(rows)


@torch.no_grad()
def run_inference_tta(model: LiRadsNet, dataset: LiRadsCaseDataset, device: torch.device, tta_views: int) -> pd.DataFrame:
    """
    Per-case (not batched) equivalent of run_inference() that applies TTA to
    the ordinal decision -- see _forward_with_tta. Not batched because TTA
    needs to rebuild each case's tensors several times with independent
    random augmentations, which a single collated DataLoader batch can't
    express. Fine for the case counts a fold's val/test split or a
    submission run involve; use plain run_inference() for tta_views=0.
    """
    model.eval()
    rows = []
    total_elapsed = 0.0
    for _, row in dataset.df.iterrows():
        case_id = str(row["case_id"])
        case_dir = _find_case_dir(dataset.data_root, case_id)
        # Same clinical row run_inference() gets through collate_cases, without
        # it this path would silently fall back to LiRadsNet's untrained
        # missing_clinical_embed and score far worse than the batched path.
        clinical_features = encode_clinical_features(row).unsqueeze(0)
        start = time.perf_counter()
        logits_cat, logits_ord = _forward_with_tta(
            model, case_dir, case_id, dataset.max_slices, tta_views, clinical_features,
        )
        total_elapsed += time.perf_counter() - start
        rows.append({"case_id": case_id, "prediction": decode_prediction(logits_cat, logits_ord, model.ordinal_head_type, model.cat_names)})
    if rows:
        print_to_log(f"mean inference time per case: {total_elapsed / len(rows):.3f}s ({len(rows)} cases)")
    return pd.DataFrame(rows)


def save_confusion_matrix(gt_labels, pred_labels, out_path: str) -> None:
    """
    Plots a (ground truth rows x predicted columns) confusion matrix over
    config.VALID_LABELS and saves it as an image. matplotlib is imported
    lazily here rather than at module level, since predict.py's load_model/
    predict_case are also imported by submission/run.py, whose container
    doesn't bundle matplotlib
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(gt_labels, pred_labels, labels=config.VALID_LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.VALID_LABELS)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False, values_format="d")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_per_class_metrics(gt_labels, pred_labels) -> pd.DataFrame:
    """
    Per-class precision/recall/F1/support over config.VALID_LABELS.
    Classes absent from both gt and pred still get a (zero-valued) row
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        gt_labels, pred_labels, labels=config.VALID_LABELS, zero_division=0,
    )
    return pd.DataFrame({
        "label": config.VALID_LABELS,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })


_GENERATED_CLINICAL_COLS = ["aphe"] + config.CLINICAL_BINARY_FEATURES + ["max_diameter_mm"]


def override_clinical_columns(df: pd.DataFrame, data_root: str, clinical_checkpoint: Sequence[str], device: torch.device, backbone_source: str = "local") -> pd.DataFrame:
    """Returns a copy of `df` (a LiRadsCaseDataset.df-shaped frame: needs a
    case_id column, plus whatever else the caller already has) with its
    aphe/washout/capsule/max_diameter_mm columns replaced by
    model.ClinicalPredictorNet's own predictions for each row's case_id --
    the same image-only path submission/run.py takes, run here instead over
    an evaluation split so its effect on scored predictions can be compared
    directly against the ground-truth-clinical run (see this module's
    docstring). Every other column (lirads_score included) is left alone.

    A case_id the generator couldn't produce a row for (see
    predict_clinical.generate_metadata_csv) falls back to the same "no
    clinical info available" representation LiRadsNet's clinical branch
    already has for a missing case -- NaN aphe (-> encode_clinical_features's
    "Unknown"), 0 for every binary flag, 0mm diameter -- never the case's
    real ground-truth values, which would defeat the point of this
    comparison."""
    clinical_models = load_clinical_models(clinical_checkpoint, device, backbone_source)
    case_ids = df["case_id"].astype(str).tolist()
    with tempfile.TemporaryDirectory() as tmp:
        generated = generate_metadata_csv(clinical_models, case_ids, data_root, os.path.join(tmp, "generated_metadata.csv"))
    del clinical_models

    generated = generated.set_index(generated["case_id"].astype(str))
    missing = [c for c in case_ids if c not in generated.index]
    if missing:
        print_to_log(f"  --clinical_checkpoint: {len(missing)}/{len(case_ids)} case(s) fell back to 'no clinical info' (generation failed): {missing[:5]}...")

    df = df.copy()
    for col in _GENERATED_CLINICAL_COLS:
        default = 0.0 if col != "aphe" else np.nan
        df[col] = [generated[col].get(cid, default) for cid in case_ids]
    return df


def predict_fold_test_set(
    checkpoint_paths: Sequence[str],
    data_root: str,
    metadata_csv: str,
    splits_json: str,
    fold,
    device: torch.device,
    max_slices: int = config.MAX_SLICES_PER_CASE,
    batch_size: int = 4,
    num_workers: int = 4,
    backbone_source: str = "local",
    tta_views: int = 0,
    clinical_checkpoint: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Runs every model in `checkpoint_paths` over the `test` split of `fold`
    (looked up in `splits.json`), returning a (case_id, prediction) DataFrame.
    With more than one checkpoint, each model's decoded predictions are
    majority-voted per case. `tta_views > 0` switches to the per-case
    run_inference_tta() path (see its docstring for why it can't batch).

    `clinical_checkpoint`: when given, one or more train_clinical.py
    checkpoints whose image-only predictions replace the test split's
    ground-truth clinical columns before prediction (see
    override_clinical_columns) -- the "inferred" run described in this
    module's docstring. None (the default) uses --metadata_csv's own
    ground-truth clinical columns unchanged, same as training.
    """
    if isinstance(checkpoint_paths, str):
        checkpoint_paths = [checkpoint_paths]

    test_case_ids = load_fold(splits_json, fold)["test"]

    test_ds = LiRadsCaseDataset(metadata_csv, data_root, max_slices, case_ids=test_case_ids)

    if clinical_checkpoint:
        test_ds.df = override_clinical_columns(
            test_ds.df, data_root, clinical_checkpoint,
            device, backbone_source
        )

    per_model_preds = []
    if tta_views > 0:
        for checkpoint_path in checkpoint_paths:
            model = load_model(checkpoint_path, device, backbone_source)
            per_model_preds.append(run_inference_tta(model, test_ds, device, tta_views))
    else:
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_cases,
        )
        for checkpoint_path in checkpoint_paths:
            model = load_model(checkpoint_path, device, backbone_source)
            per_model_preds.append(run_inference(model, test_loader, device))

    combined = pd.concat(per_model_preds, ignore_index=True)
    return combined.groupby("case_id", as_index=False)["prediction"].agg(lambda preds: majority_vote(list(preds)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict (and optionally score) one or more checkpoints on one fold's held-out test split"
    )
    parser.add_argument(
        "--checkpoint", required=True, nargs="+",
        help="path to one or more lirads_model checkpoints (.pt); with more than one, predictions are majority-voted",
    )
    parser.add_argument("--data_root", required=True, help="root dir containing extracted case folders")
    parser.add_argument("--metadata_csv", required=True, help="CSV with case_id + lirads_score for the whole dataset")
    parser.add_argument("--splits_json", required=True, help="output of `python -m lirads_model.splits`")
    parser.add_argument("--fold", type=int, required=True, help="fold index into splits_json whose test split to predict on")
    parser.add_argument("--out", default=None, help="where to save predictions CSV (default: next to the first --checkpoint)")
    parser.add_argument("--backbone_source", choices=["local", "hub"], default="local")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_slices", type=int, default=config.MAX_SLICES_PER_CASE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score", action="store_true", help="also score predictions against ground truth with the challenge metric")
    parser.add_argument(
        "--tta_views", type=int, default=0,
        help=(
            "test-time augmentation: average this many extra augmented forward passes into the "
            "LR-1..LR-5 ordinal decision (0 disables TTA; see config.TTA_VIEWS for the default used "
            "by predict_case/predict_case_ensemble at submission time)"
        ),
    )
    parser.add_argument(
        "--clinical_checkpoint", nargs="+", default=None,
        help=(
            "one or more train_clinical.py checkpoints; when given, the test split's aphe/washout/"
            "capsule/max_diameter_mm columns are replaced by ClinicalPredictorNet's own image-only "
            "predictions before predicting (the 'inferred' run -- see this module's docstring), "
            "instead of --metadata_csv's ground-truth clinical columns (the default 'factual' run). "
            "Output filenames get a distinct '_clinical_inferred' tag so the two runs don't overwrite "
            "each other."
        ),
    )
    args = parser.parse_args()
    clinical_tag = "_clinical_inferred" if args.clinical_checkpoint else ""
    if len(args.checkpoint) == 1:
        default_stem = os.path.splitext(args.checkpoint[0])[0] + f"_test_predictions{clinical_tag}.csv"
    else:
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint[0]))
        default_stem = os.path.join(ckpt_dir, f"ensemble_of_{len(args.checkpoint)}_test_predictions{clinical_tag}.csv")
    out_path = fold_tagged_path(args.out or default_stem, args.fold)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.splitext(args.checkpoint[0])[0] + ".log"
    print_to_log("=" * 70, log_path)
    print_to_log(f"Starting prediction. Model paths = {out_path}.", log_path)
    print_to_log("=" * 70, log_path)

    device = torch.device(args.device)
    preds = predict_fold_test_set(
        args.checkpoint, args.data_root, args.metadata_csv, args.splits_json, args.fold, device,
        max_slices=args.max_slices, batch_size=args.batch_size, num_workers=args.num_workers,
        tta_views=args.tta_views,
        backbone_source=args.backbone_source,
        clinical_checkpoint=args.clinical_checkpoint,
    )

    preds.to_csv(default_stem, index=False)
    print_to_log(f"saved {len(preds)} predictions to {default_stem}", log_path)

    if args.score:
        test_case_ids = load_fold(args.splits_json, args.fold)["test"]
        gt_ds = LiRadsCaseDataset(args.metadata_csv, args.data_root, args.max_slices, case_ids=test_case_ids)
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = os.path.join(tmp, "gt.csv")
            pred_path = os.path.join(tmp, "pred.csv")
            gt_ds.df.rename(columns={"lirads_score": "label"})[["case_id", "label"]].to_csv(gt_path, index=False)
            preds.to_csv(pred_path, index=False)
            result = compute_challenge_score(gt_path, pred_path, bootstrap=False)
        print_to_log(
            f"fold {args.fold} test set: final_score={result['final_score']:.4f} "
            f"qwk={result['adjusted_qwk']:.4f} scr={result['special_category_recognition']:.4f}",
            log_path
        )

        cm_path = fold_tagged_path(os.path.splitext(out_path)[0] + "_confusion_matrix.png", args.fold)
        merged = gt_ds.df[["case_id", "lirads_score"]].astype({"case_id": str}).merge(
            preds.astype({"case_id": str}), on="case_id", how="inner",
        )
        save_confusion_matrix(merged["lirads_score"], merged["prediction"], cm_path)
        print_to_log(f"saved confusion matrix to {cm_path}", log_path)

        metrics_df = compute_per_class_metrics(merged["lirads_score"], merged["prediction"])
        metrics_path = fold_tagged_path(os.path.splitext(out_path)[0] + "_per_class_metrics.csv", args.fold)
        metrics_df.to_csv(metrics_path, index=False)
        print_to_log(f"saved per-class precision/recall to {metrics_path}", log_path)
        print_to_log(metrics_df.to_string(index=False), log_path)


if __name__ == "__main__":
    main()
