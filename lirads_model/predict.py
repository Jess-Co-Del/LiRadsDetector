"""Shared inference logic used by both local evaluation and submission/run.py,
plus a CLI to predict a checkpoint on one fold's held-out test split:

    python -m lirads_model.predict \
      --checkpoint checkpoints/lirads_model_fold0.pt \
      --data_root ./data/cases \
      --metadata_csv ./data/metadata.csv \
      --splits_json ./data/splits.json \
      --fold 0 \
      --score
"""

import argparse
import os
import sys
import tempfile

import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from . import config, preprocessing
from .backbone import Dinov2SliceEncoder
from .dataset import LiRadsCaseDataset, collate_cases
from .model import LiRadsNet, decode_prediction
from .splits import fold_tagged_path, load_fold

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "amplifai-codabench"))
from evaluate import evaluate as compute_challenge_score  # noqa: E402


def load_model(checkpoint_path: str, device: torch.device, backbone_source: str = "local") -> LiRadsNet:
    """backbone_source="local": no network (submission container). "hub":
    re-downloads the pretrained backbone from the HuggingFace Hub before
    loading our trained weights on top (useful for local dev without a
    vendored snapshot)."""
    if backbone_source == "local":
        backbone = Dinov2SliceEncoder.from_local()
    else:
        backbone = Dinov2SliceEncoder.from_pretrained()

    model = LiRadsNet(backbone).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_case(
    model: LiRadsNet,
    case_dir: str,
    case_id: str,
    device: torch.device,
    max_slices: int = config.MAX_SLICES_PER_CASE,
) -> str:
    phase_paths = preprocessing.find_case_phase_paths(case_dir, case_id)
    mask_path = preprocessing.find_case_mask_path(case_dir)
    phase_data = preprocessing.build_case_tensors(phase_paths, mask_path, max_slices)

    logits_cat, logits_ord = model([phase_data])
    return decode_prediction(logits_cat[0].cpu(), logits_ord[0].cpu())


@torch.no_grad()
def run_inference(model: LiRadsNet, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    """
    Batched inference over a LiRadsCaseDataset DataLoader. Returns a
    (case_id, prediction) DataFrame in amplifai-codabench/evaluate.py's format
    """
    model.eval()
    rows = []
    for batch in loader:
        logits_cat, logits_ord = model(batch["phase_data"], batch["clinical_features"])
        for i, case_id in enumerate(batch["case_ids"]):
            label = decode_prediction(logits_cat[i].cpu(), logits_ord[i].cpu())
            rows.append({"case_id": case_id, "prediction": label})
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


def predict_fold_test_set(
    checkpoint_path: str,
    data_root: str,
    metadata_csv: str,
    splits_json: str,
    fold,
    device: torch.device,
    max_slices: int = config.MAX_SLICES_PER_CASE,
    batch_size: int = 4,
    num_workers: int = 4,
    backbone_source: str = "local",
) -> pd.DataFrame:
    """
    Runs the model in `checkpoint_path` over the `test` split of `fold`
    (looked up in `splits.json`), returning a (case_id, prediction) DataFrame
    """
    test_case_ids = load_fold(splits_json, fold)["test"]
    test_ds = LiRadsCaseDataset(metadata_csv, data_root, max_slices, case_ids=test_case_ids)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_cases,
    )
    model = load_model(checkpoint_path, device, backbone_source)
    return run_inference(model, test_loader, device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict (and optionally score) a checkpoint on one fold's held-out test split"
    )
    parser.add_argument("--checkpoint", required=True, help="path to a lirads_model checkpoint (.pt)")
    parser.add_argument("--data_root", required=True, help="root dir containing extracted case folders")
    parser.add_argument("--metadata_csv", required=True, help="CSV with case_id + lirads_score for the whole dataset")
    parser.add_argument("--splits_json", required=True, help="output of `python -m lirads_model.splits`")
    parser.add_argument("--fold", type=int, required=True, help="fold index into splits_json whose test split to predict on")
    parser.add_argument("--out", default=None, help="where to save predictions CSV (default: next to --checkpoint)")
    parser.add_argument("--backbone_source", choices=["local", "hub"], default="local")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_slices", type=int, default=config.MAX_SLICES_PER_CASE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score", action="store_true", help="also score predictions against ground truth with the challenge metric")
    args = parser.parse_args()

    device = torch.device(args.device)
    preds = predict_fold_test_set(
        args.checkpoint, args.data_root, args.metadata_csv, args.splits_json, args.fold, device,
        max_slices=args.max_slices, batch_size=args.batch_size, num_workers=args.num_workers,
        backbone_source=args.backbone_source,
    )

    checkpoint_stem = os.path.splitext(args.checkpoint)[0]
    out_path = fold_tagged_path(args.out or f"{checkpoint_stem}_test_predictions.csv", args.fold)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    preds.to_csv(out_path, index=False)
    print(f"saved {len(preds)} predictions to {out_path}")

    if args.score:
        test_case_ids = load_fold(args.splits_json, args.fold)["test"]
        gt_ds = LiRadsCaseDataset(args.metadata_csv, args.data_root, args.max_slices, case_ids=test_case_ids)
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = os.path.join(tmp, "gt.csv")
            pred_path = os.path.join(tmp, "pred.csv")
            gt_ds.df.rename(columns={"lirads_score": "label"})[["case_id", "label"]].to_csv(gt_path, index=False)
            preds.to_csv(pred_path, index=False)
            result = compute_challenge_score(gt_path, pred_path, bootstrap=False)
        print(
            f"fold {args.fold} test set: final_score={result['final_score']:.4f} "
            f"qwk={result['adjusted_qwk']:.4f} scr={result['special_category_recognition']:.4f}"
        )

        cm_path = fold_tagged_path(os.path.splitext(out_path)[0] + "_confusion_matrix.png", args.fold)
        merged = gt_ds.df[["case_id", "lirads_score"]].astype({"case_id": str}).merge(
            preds.astype({"case_id": str}), on="case_id", how="inner",
        )
        save_confusion_matrix(merged["lirads_score"], merged["prediction"], cm_path)
        print(f"saved confusion matrix to {cm_path}")

        metrics_df = compute_per_class_metrics(merged["lirads_score"], merged["prediction"])
        metrics_path = fold_tagged_path(os.path.splitext(out_path)[0] + "_per_class_metrics.csv", args.fold)
        metrics_df.to_csv(metrics_path, index=False)
        print(f"saved per-class precision/recall to {metrics_path}")
        print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
