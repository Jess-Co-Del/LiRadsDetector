"""
Trainer of LiRadsNet
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from time import time

from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter

from . import config
from .backbone import Dinov2SliceEncoder
from .dataset import LiRadsCaseDataset, collate_cases, label_to_targets
from .model import LiRadsNet
from .predict import compute_per_class_metrics, run_inference, save_confusion_matrix
from .splits import fold_tagged_path, load_fold

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "amplifai-codabench"))
from evaluate import evaluate as compute_challenge_score  # noqa: E402


def print_to_log(a):
    timestamp = time()
    dt_object = datetime.fromtimestamp(timestamp)
    args = (f"{dt_object}:", a)
    print(*args)


def compute_class_weights(counts: dict, num_classes: int) -> torch.Tensor:
    freqs = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float64)
    freqs = np.clip(freqs, 1, None)  # avoid div-by-zero for unseen classes
    weights = freqs.sum() / (num_classes * freqs)
    return torch.tensor(weights, dtype=torch.float32)


class InfiniteDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
        
    def __iter__(self):
        return self
        
    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    args.out = fold_tagged_path(args.out, args.fold)

    fold = load_fold(args.splits_json, args.fold)

    train_ds = LiRadsCaseDataset(args.metadata_csv, args.data_root, args.max_slices, case_ids=fold["train"])
    val_ds = LiRadsCaseDataset(args.metadata_csv, args.data_root, args.max_slices, case_ids=fold["val"])

    train_loader = InfiniteDataLoader(
        DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_cases,
        )
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
    print_to_log("=" * 70)
    print_to_log(f"Starting training.")
    print_to_log("=" * 70)

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        print_to_log('')
        print_to_log(f"Epoch {epoch}.")
        print_to_log(f"Current learning rate: {np.round(optimizer.param_groups[0]['lr'], decimals=5)}")
        model.train()
        model.backbone.eval()  # frozen backbone: never let dropout/drop-path move it

        total_loss, n_batches = 0.0, 0
        for _ in range(args.num_iterations_per_epoch):
            batch = next(train_loader)
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

        print_to_log(
            f"Epoch {epoch}: train_loss={avg_loss:.4f} "
            f"final_score={result['final_score']:.4f} "
            f"qwk={result['adjusted_qwk']:.4f} scr={result['special_category_recognition']:.4f}"
        )

        if result["final_score"] > best_score:
            best_score = result["final_score"]
            torch.save({"model_state_dict": model.state_dict()}, args.out)
            print_to_log(f"  saved new best checkpoint to {args.out} (score={best_score:.4f})")

    print_to_log(f"Training complete. best val final_score={best_score:.4f}")

    test_ds = LiRadsCaseDataset(args.metadata_csv, args.data_root, args.max_slices, case_ids=fold["test"])
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_cases,
    )
    if os.path.exists(args.out):
        checkpoint = torch.load(args.out, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print_to_log(f"  no checkpoint was ever saved to {args.out}; testing with the last epoch's in-memory weights")

    test_preds = run_inference(model, test_loader, device)
    with tempfile.TemporaryDirectory() as tmp:
        gt_path = os.path.join(tmp, "gt.csv")
        pred_path = os.path.join(tmp, "pred.csv")
        test_ds.df.rename(columns={"lirads_score": "label"})[["case_id", "label"]].to_csv(gt_path, index=False)
        test_preds.to_csv(pred_path, index=False)
        test_result = compute_challenge_score(gt_path, pred_path, bootstrap=False)

    print_to_log(
        f"Test (fold {args.fold}): final_score={test_result['final_score']:.4f} "
        f"qwk={test_result['adjusted_qwk']:.4f} scr={test_result['special_category_recognition']:.4f}"
    )

    test_pred_path = args.test_predictions_out or os.path.splitext(args.out)[0] + "_test_predictions.csv"
    test_preds.to_csv(test_pred_path, index=False)
    print_to_log(f"  saved test predictions to {test_pred_path}")

    cm_path = fold_tagged_path(os.path.splitext(test_pred_path)[0] + "_confusion_matrix.png", args.fold)
    merged = test_ds.df[["case_id", "lirads_score"]].astype({"case_id": str}).merge(
        test_preds.astype({"case_id": str}), on="case_id", how="inner",
    )
    save_confusion_matrix(merged["lirads_score"], merged["prediction"], cm_path)
    print_to_log(f"  saved test confusion matrix to {cm_path}")

    metrics_df = compute_per_class_metrics(merged["lirads_score"], merged["prediction"])
    metrics_path = fold_tagged_path(os.path.splitext(test_pred_path)[0] + "_per_class_metrics.csv", args.fold)
    metrics_df.to_csv(metrics_path, index=False)
    print_to_log(f"  saved per-class precision/recall to {metrics_path}")
    for _, row in metrics_df.iterrows():
        print_to_log(
            f"    {row['label']:<10} precision={row['precision']:.3f} recall={row['recall']:.3f} "
            f"f1={row['f1']:.3f} support={int(row['support'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AMPLIFAI LI-RADS classifier")
    parser.add_argument("--data_root", required=True, help="Root dir containing extracted case folders")
    parser.add_argument("--metadata_csv", required=True, help="CSV with case_id + lirads_score for the whole dataset")
    parser.add_argument("--splits_json", required=True, help="output of `python -m lirads_model.splits`")
    parser.add_argument("--fold", type=int, default=0, help="fold index into splits_json to train/test on")
    parser.add_argument("--test_predictions_out", default=None, help="where to save test-split predictions CSV")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_iterations_per_epoch", type=int, default=40)
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
