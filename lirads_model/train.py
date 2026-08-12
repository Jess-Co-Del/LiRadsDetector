"""Trains the head (+ missing-phase embedding) on top of a frozen DINOv2
backbone, and validates each epoch using the real challenge metric.

Requires internet the first time it runs (torch.hub download of the
pretrained backbone). Saves a single self-contained checkpoint whose
architecture can later be reconstructed fully offline via
Dinov2SliceEncoder.from_local() (see scripts/vendor_dinov2.sh).

Usage:
    python -m lirads_model.train \
        --data_root /path/to/extracted/cases \
        --train_csv /path/to/train_metadata.csv \
        --val_csv   /path/to/val_metadata.csv \
        --out checkpoints/lirads_model.pt
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from .backbone import Dinov2SliceEncoder
from .dataset import LiRadsCaseDataset, collate_cases, label_to_targets
from .model import LiRadsNet, decode_prediction

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "amplifai-codabench"))
from evaluate import evaluate as compute_challenge_score  # noqa: E402


def compute_class_weights(counts: dict, num_classes: int) -> torch.Tensor:
    freqs = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float64)
    freqs = np.clip(freqs, 1, None)  # avoid div-by-zero for unseen classes
    weights = freqs.sum() / (num_classes * freqs)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def run_inference(model: LiRadsNet, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        logits_cat, logits_ord = model(batch["phase_data"])
        for i, case_id in enumerate(batch["case_ids"]):
            label = decode_prediction(logits_cat[i].cpu(), logits_ord[i].cpu())
            rows.append({"case_id": case_id, "prediction": label})
    return pd.DataFrame(rows)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    train_ds = LiRadsCaseDataset(args.train_csv, args.data_root, args.max_slices)
    val_ds = LiRadsCaseDataset(args.val_csv, args.data_root, args.max_slices)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_cases,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_cases,
    )

    backbone = Dinov2SliceEncoder.from_pretrained()
    model = LiRadsNet(backbone).to(device)

    cat_idxs, ord_idxs = zip(*(label_to_targets(str(l)) for l in train_ds.df["lirads_score"]))
    cat_counts = {i: cat_idxs.count(i) for i in range(len(config.CAT_NAMES))}
    ord_only = [o for o in ord_idxs if o >= 0]
    ord_counts = {i: ord_only.count(i) for i in range(len(config.ORDINAL_LABELS))}

    cat_criterion = nn.CrossEntropyLoss(weight=compute_class_weights(cat_counts, len(config.CAT_NAMES)).to(device))
    ord_criterion = nn.CrossEntropyLoss(weight=compute_class_weights(ord_counts, len(config.ORDINAL_LABELS)).to(device))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.backbone.eval()  # frozen backbone: never let dropout/drop-path move it

        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            cat_idx = batch["cat_idx"].to(device)
            ord_idx = batch["ord_idx"].to(device)

            logits_cat, logits_ord = model(batch["phase_data"])
            loss = cat_criterion(logits_cat, cat_idx)

            ord_mask = cat_idx == 0
            if ord_mask.any():
                loss = loss + ord_criterion(logits_ord[ord_mask], ord_idx[ord_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        val_preds = run_inference(model, val_loader, device)
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = os.path.join(tmp, "gt.csv")
            pred_path = os.path.join(tmp, "pred.csv")
            val_ds.df.rename(columns={"lirads_score": "label"})[["case_id", "label"]].to_csv(gt_path, index=False)
            val_preds.to_csv(pred_path, index=False)
            result = compute_challenge_score(gt_path, pred_path, bootstrap=False)

        print(
            f"epoch {epoch}: train_loss={avg_loss:.4f} "
            f"final_score={result['final_score']:.4f} "
            f"qwk={result['adjusted_qwk']:.4f} scr={result['special_category_recognition']:.4f}"
        )

        if result["final_score"] > best_score:
            best_score = result["final_score"]
            torch.save({"model_state_dict": model.state_dict()}, args.out)
            print(f"  saved new best checkpoint to {args.out} (score={best_score:.4f})")

    print(f"training complete. best val final_score={best_score:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AMPLIFAI LI-RADS classifier")
    parser.add_argument("--data_root", required=True, help="Root dir containing extracted case folders")
    parser.add_argument("--train_csv", required=True, help="train_metadata.csv path")
    parser.add_argument("--val_csv", required=True, help="val_metadata.csv path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_slices", type=int, default=config.MAX_SLICES_PER_CASE)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="checkpoints/lirads_model.pt")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
