"""
Trainer of model.ClinicalPredictorNet,the image-only side model that lets
submission/run.py synthesize a clinical/tabular feature row for LiRadsNet's
clinical branch, even though the real submission input never supplies one
(see config's "Clinical feature prediction" section). Reuses the same
splits.json fold structure as train.py; see predict_clinical.py for running
a trained checkpoint over a set of cases and joining the result into a CSV.

    python -m lirads_model.train_clinical \
      --data_root ./data/cases \
      --metadata_csv ./data/metadata.csv \
      --splits_json ./data/splits.json \
      --fold 0 \
      --out checkpoints/clinical_model.pt
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from . import config
from .config import print_to_log
from .backbone import Dinov2SliceEncoder
from .dataset import ClinicalMetadataDataset, collate_clinical_cases
from .model import ClinicalPredictorNet
from .splits import fold_tagged_path, load_fold


def compute_class_weights(counts: dict, num_classes: int) -> torch.Tensor:
    freqs = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float64)
    freqs = np.clip(freqs, 1, None)  # avoid div-by-zero for unseen classes
    weights = freqs.sum() / (num_classes * freqs)
    return torch.tensor(weights, dtype=torch.float32)


def compute_pos_weight(binary_targets: np.ndarray) -> torch.Tensor:
    """Per-feature BCEWithLogitsLoss pos_weight = negatives/positives
    (clamped to at least 1 positive), so an imbalanced flag like
    capsule_delayed (~11% positive in train_metadata.csv) doesn't just get
    predicted 0 for every case."""
    pos = binary_targets.sum(axis=0)
    neg = binary_targets.shape[0] - pos
    return torch.tensor(neg / np.clip(pos, 1, None), dtype=torch.float32)


@torch.no_grad()
def evaluate(model: ClinicalPredictorNet, loader: DataLoader, device: torch.device) -> dict:
    """Returns per-feature precision/recall/f1 (aphe: macro over its 3
    classes; each binary feature: for its positive class) plus a combined
    `score` (mean of aphe macro-f1 and the 4 binary features' f1s) used to
    pick the best checkpoint,unweighted across features since none is
    intrinsically more important than another for the downstream clinical
    vector."""
    model.eval()
    aphe_true, aphe_pred = [], []
    binary_true, binary_pred = [], []
    for batch in loader:
        aphe_logits, binary_logits = model(batch["phase_data"])
        aphe_idx = batch["aphe_idx"]
        known = aphe_idx >= 0
        if known.any():
            aphe_true.extend(aphe_idx[known].tolist())
            aphe_pred.extend(torch.argmax(aphe_logits[known], dim=1).cpu().tolist())
        binary_true.append(batch["binary_targets"].cpu().numpy())
        binary_pred.append((torch.sigmoid(binary_logits).cpu().numpy() >= 0.5).astype(float))

    result = {}
    if aphe_true:
        _, _, aphe_f1, _ = precision_recall_fscore_support(
            aphe_true, aphe_pred, labels=list(range(len(config.APHE_PREDICTABLE_CATEGORIES))),
            average="macro", zero_division=0,
        )
        result["aphe_f1"] = float(aphe_f1)
    else:
        result["aphe_f1"] = 0.0

    binary_true = np.concatenate(binary_true, axis=0)
    binary_pred = np.concatenate(binary_pred, axis=0)
    binary_f1s = {}
    for i, name in enumerate(config.CLINICAL_BINARY_FEATURES):
        _, _, f1, _ = precision_recall_fscore_support(
            binary_true[:, i], binary_pred[:, i], labels=[0, 1], average=None, zero_division=0,
        )
        binary_f1s[name] = float(f1[1])  # positive-class f1
    result["binary_f1"] = binary_f1s
    result["score"] = (result["aphe_f1"] + sum(binary_f1s.values())) / (1 + len(binary_f1s))
    return result


def log_eval(result: dict, log_path: str) -> None:
    print_to_log(f"    aphe macro-f1={result['aphe_f1']:.4f}", log_path)
    for name, f1 in result["binary_f1"].items():
        print_to_log(f"    {name} f1={f1:.4f}", log_path)


def train(args: argparse.Namespace) -> None:
    args.out = fold_tagged_path(args.out, args.fold)
    log_path = os.path.splitext(args.out)[0] + ".log"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print_to_log("=" * 70, log_path)
    print_to_log(f"Starting clinical-predictor training. Fold = {args.fold}.", log_path)
    print_to_log("=" * 70, log_path)
    device = torch.device(args.device)

    fold = load_fold(args.splits_json, args.fold)

    train_ds = ClinicalMetadataDataset(
        args.metadata_csv, args.data_root, args.max_slices, case_ids=fold["train"],
        augment=args.augment, transplant=args.transplant,
    )
    val_ds = ClinicalMetadataDataset(
        args.metadata_csv, args.data_root, args.max_slices, case_ids=fold["val"],
    )
    print_to_log(
        f"Datasets loaded: train={len(train_ds)} val={len(val_ds)} "
        f"(config.NO_LESION_LABEL cases excluded,see ClinicalMetadataDataset)",
        log_path,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_clinical_cases,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_clinical_cases,
    )

    backbone = Dinov2SliceEncoder.from_pretrained()
    model = ClinicalPredictorNet(backbone, use_cnn=args.use_cnn).to(device)

    aphe_counts = pd.Series(train_ds.df["aphe"].dropna().str.strip()).value_counts().to_dict()
    aphe_class_counts = {i: aphe_counts.get(cat, 0) for i, cat in enumerate(config.APHE_PREDICTABLE_CATEGORIES)}
    aphe_criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(aphe_class_counts, len(config.APHE_PREDICTABLE_CATEGORIES)).to(device),
        ignore_index=-1,
    )
    binary_targets_all = np.stack([
        [float(train_ds.df.iloc[i][c]) for c in config.CLINICAL_BINARY_FEATURES] for i in range(len(train_ds))
    ])
    binary_criterion = nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight(binary_targets_all).to(device))
    print_to_log(f"aphe class counts: {aphe_class_counts}", log_path)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
        if args.lr_scheduler == "cosine" else None
    )

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        print_to_log("", log_path)
        print_to_log(f"Epoch {epoch}.", log_path)
        print_to_log(f"Current learning rate: {np.round(optimizer.param_groups[0]['lr'], decimals=6)}", log_path)
        model.train()
        model.backbone.eval()  # frozen backbone: never let dropout/drop-path move it

        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            aphe_idx = batch["aphe_idx"].to(device)
            binary_targets = batch["binary_targets"].to(device)
            aphe_logits, binary_logits = model(batch["phase_data"])

            loss = binary_criterion(binary_logits, binary_targets)
            if (aphe_idx >= 0).any():
                loss = loss + aphe_criterion(aphe_logits, aphe_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if scheduler is not None:
            scheduler.step()

        result = evaluate(model, val_loader, device)
        print_to_log(f"Epoch {epoch}: train_loss={avg_loss:.4f} val_score={result['score']:.4f}", log_path)
        log_eval(result, log_path)

        if result["score"] > best_score:
            best_score = result["score"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_score": best_score,
                    "use_cnn": args.use_cnn,
                    "aphe_categories": list(config.APHE_PREDICTABLE_CATEGORIES),
                    "binary_features": list(config.CLINICAL_BINARY_FEATURES),
                },
                args.out,
            )
            print_to_log(f"  saved new best checkpoint to {args.out} (score={best_score:.4f})", log_path)

    print_to_log(f"Training complete. best val score={best_score:.4f}", log_path)

    test_ds = ClinicalMetadataDataset(args.metadata_csv, args.data_root, args.max_slices, case_ids=fold["test"])
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_clinical_cases,
    )
    if os.path.exists(args.out):
        checkpoint = torch.load(args.out, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print_to_log(f"  no checkpoint was ever saved to {args.out}; testing with the last epoch's in-memory weights", log_path)

    test_result = evaluate(model, test_loader, device)
    print_to_log(f"Test (fold {args.fold}): score={test_result['score']:.4f}", log_path)
    log_eval(test_result, log_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AMPLIFAI clinical-feature predictor (image-only)")
    parser.add_argument("--data_root", required=True, help="Root dir containing extracted case folders")
    parser.add_argument("--metadata_csv", required=True, help="CSV with case_id + aphe/washout/capsule columns")
    parser.add_argument("--splits_json", required=True, help="output of `python -m lirads_model.splits`")
    parser.add_argument("--fold", type=int, default=0, help="fold index into splits_json to train/test on")
    parser.add_argument(
        "--use_cnn", action=argparse.BooleanOptionalAction, default=True,
        help="use the per-phase 3D-CNN volume branch alongside DINOv2 (see model.PhaseVolumeCNN)",
    )
    parser.add_argument(
        "--augment", action=argparse.BooleanOptionalAction, default=True,
        help="geometric/intensity augmentation on the training split (see preprocessing.build_case_tensors)",
    )
    parser.add_argument(
        "--transplant", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "lesion-transplant augmentation on the training split, for config.TRANSPLANT_DONOR_LABELS "
            "rows (LR-1/2/3/4 by default) -- see dataset.ClinicalMetadataDataset and lesion_transplant.py. "
            "Needs scripts/segment_livers.py to have run on the data (recipient placement is constrained "
            "to the recipient's own segmented liver)."
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--lr_min", type=float, default=0.0)
    parser.add_argument("--max_slices", type=int, default=config.MAX_SLICES_PER_CASE)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="checkpoints/clinical_model.pt")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
