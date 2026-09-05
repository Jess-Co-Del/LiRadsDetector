"""
Trainer of LiRadsNet
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter

from . import config
from .config import print_to_log
from .backbone import Dinov2SliceEncoder
from .dataset import AUGMENT_MODES, LiRadsCaseDataset, collate_cases, label_to_targets, resolve_augment_mode
from .losses import CornSoftQWKLoss, SORDLoss
from .model import LiRadsNet
from .predict import compute_per_class_metrics, run_inference, run_inference_tta, save_confusion_matrix
from .splits import fold_tagged_path, load_fold

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "amplifai-codabench"))
from evaluate import evaluate as compute_challenge_score  # noqa: E402


def make_lr_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    """Builds the epoch-level LR schedule selected by --lr_scheduler. Called
    once per epoch (see train()'s epoch loop, right after that epoch's
    validation) rather than per-iteration, since validation only runs once
    per epoch and 'plateau' needs its metric. "none" (the default) returns
    None and keeps --lr constant for the whole run, matching the previous
    (pre-scheduler) behavior."""
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    if args.lr_scheduler == "plateau":
        # mode="max": final_score (the challenge metric) is better when
        # higher, unlike a loss.
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.lr_gamma, patience=args.lr_patience,
        )
    raise ValueError(f"unrecognized --lr_scheduler: {args.lr_scheduler!r}")


def compute_class_weights(counts: dict, num_classes: int) -> torch.Tensor:
    freqs = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float64)
    freqs = np.clip(freqs, 1, None)  # avoid div-by-zero for unseen classes
    weights = freqs.sum() / (num_classes * freqs)
    return torch.tensor(weights, dtype=torch.float32)


def build_ordinal_criterion(args: argparse.Namespace, ord_counts: dict, device: torch.device) -> nn.Module:
    """Builds the loss for the ordinal head (LR-1..LR-5) selected by
    --ordinal_loss. All options use the same class weighting (inverse
    frequency over ord_counts). 'ce' and 'sord' both train a plain 5-way
    softmax head (LiRadsNet(ordinal_head_type="softmax")) -- the only
    difference is whether the target is one-hot (ce) or a rank-distance-
    softened distribution (sord, see losses.SORDLoss). 'corn_qwk' instead
    trains a 4-output CORN head (LiRadsNet(ordinal_head_type="corn"), see
    build_ordinal_head_type() below) with L = CORN_loss + --qwk_lambda *
    (1 - soft_QWK) -- see losses.CornSoftQWKLoss."""
    weight = compute_class_weights(ord_counts, len(config.ORDINAL_LABELS)).to(device)
    if args.ordinal_loss == "ce":
        return nn.CrossEntropyLoss(weight=weight)
    if args.ordinal_loss == "sord":
        return SORDLoss(len(config.ORDINAL_LABELS), weight=weight).to(device)
    if args.ordinal_loss == "corn_qwk":
        return CornSoftQWKLoss(len(config.ORDINAL_LABELS), lambda_qwk=args.qwk_lambda, weight=weight).to(device)
    raise ValueError(f"unrecognized --ordinal_loss: {args.ordinal_loss!r}")


def build_ordinal_head_type(args: argparse.Namespace) -> str:
    """The ordinal head's output shape depends on --ordinal_loss: 'corn_qwk'
    needs LiRadsNet(ordinal_head_type="corn") (num_classes-1 conditional-
    threshold logits), while 'ce'/'sord' both use the plain "softmax" head
    (num_classes logits) -- see model.LiRadsNet."""
    return "corn" if args.ordinal_loss == "corn_qwk" else "softmax"


def make_balanced_sampler(labels) -> WeightedRandomSampler:
    """
    Per-example inverse-frequency weight over the full lirads_score label
    (not just the 4-way cat_idx), so a rare special class like LR-TIV gets
    oversampled relative to a more common one (e.g. LR-M) that happens to
    share its category bucket -- loss-level class weighting alone can't fix
    this, since with few train iterations per epoch a rare class can simply
    never get drawn
    """
    counts = pd.Series([str(l).strip() for l in labels]).value_counts()
    weights = [1.0 / counts[str(l).strip()] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def log_per_class_metrics(metrics_df: pd.DataFrame, log_path: str) -> None:
    for _, row in metrics_df.iterrows():
        print_to_log(
            f"    {row['label']:<10} precision={row['precision']:.3f} recall={row['recall']:.3f} "
            f"f1={row['f1']:.3f} support={int(row['support'])}",
            log_path,
        )


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
    args.out = fold_tagged_path(args.out, args.fold)
    log_path = os.path.splitext(args.out)[0] + ".log"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print_to_log("=" * 70, log_path)
    print_to_log(f"Starting training. Fold = {args.fold}.", log_path)
    print_to_log("=" * 70, log_path)
    device = torch.device(args.device)

    fold = load_fold(args.splits_json, args.fold)

    ordinal_only = args.head_mode == "ordinal"

    # cat_names is the category head's actual label set, in cat_head's
    # output order -- normally config.CAT_NAMES (4-way), narrowed to drop
    # "No lesion" when --no-include_no_lesion. Threaded through
    # LiRadsCaseDataset (label_to_targets) and LiRadsNet (cat_head's width)
    # below so both agree on what each cat_idx means; see model.LiRadsNet's
    # cat_names docstring.
    cat_names = (
        list(config.CAT_NAMES) if args.include_no_lesion
        else [c for c in config.CAT_NAMES if c != config.NO_LESION_LABEL]
    )

    # --resume: peek at an existing checkpoint at args.out *before*
    # building the dataset/model below, since its architecture flags
    # (use_cnn/use_clinical/use_cat_head/cat_names) are frozen into its
    # saved weights' shapes and must override whatever
    # --use_cnn/--use_clinical/--head_mode/--include_no_lesion were passed
    # this time -- resuming with a different architecture would make
    # model.load_state_dict() below fail (or silently mismatch).
    resume_checkpoint = None
    if args.resume:
        if os.path.exists(args.out):
            resume_checkpoint = torch.load(args.out, map_location=device)
            print_to_log(f"--resume: resuming from existing checkpoint at {args.out}.", log_path)
            ckpt_use_cnn = resume_checkpoint.get("use_cnn", args.use_cnn)
            ckpt_use_clinical = resume_checkpoint.get("use_clinical", args.use_clinical)
            ckpt_use_cat_head = resume_checkpoint.get("use_cat_head", not ordinal_only)
            ckpt_cat_names = resume_checkpoint.get("cat_names", list(config.CAT_NAMES))
            if (ckpt_use_cnn, ckpt_use_clinical, ckpt_use_cat_head, ckpt_cat_names) != (args.use_cnn, args.use_clinical, not ordinal_only, cat_names):
                print_to_log(
                    "  --resume: overriding --use_cnn/--use_clinical/--head_mode/--include_no_lesion with the "
                    f"checkpoint's own architecture (use_cnn={ckpt_use_cnn}, use_clinical={ckpt_use_clinical}, "
                    f"use_cat_head={ckpt_use_cat_head}, cat_names={ckpt_cat_names}) -- a model's architecture "
                    "can't change mid-training.",
                    log_path,
                )
            args.use_cnn, args.use_clinical = ckpt_use_cnn, ckpt_use_clinical
            ordinal_only = not ckpt_use_cat_head
            cat_names = ckpt_cat_names
        else:
            print_to_log(f"--resume: no checkpoint found at {args.out}; starting fresh.", log_path)

    # Unlike use_cnn/use_clinical/use_cat_head above, ordinal_head_type is
    # *not* forced to match the checkpoint -- --ordinal_loss is meant to be
    # freely switchable across a --resume (e.g. a soft-QWK/CORN fine-tuning
    # stage on top of a checkpoint trained with plain ce/sord). 'ce' and
    # 'sord' share the same "softmax" head shape, so switching between those
    # two resumes cleanly; switching to/from 'corn_qwk' changes ord_head's
    # output shape, so that case gets a partial (trunk-only) load below
    # instead of a full state_dict load.
    ordinal_head_type = build_ordinal_head_type(args)
    ordinal_head_changed = (
        resume_checkpoint is not None
        and resume_checkpoint.get("ordinal_head_type", "softmax") != ordinal_head_type
    )

    # train_ds = LiRadsCaseDataset(
    #     args.metadata_csv,
    #     args.data_root,
    #     args.max_slices,
    #     case_ids=fold["train"],
    #     **resolve_augment_mode(args.augment_mode),
    #     ordinal_only=ordinal_only,
    # )
    # val_ds = LiRadsCaseDataset(
    #     args.metadata_csv,
    #     args.data_root,
    #     args.max_slices,
    #     case_ids=fold["val"],
    #     ordinal_only=ordinal_only,
    # )

    train_ds = LiRadsCaseDataset(
        args.metadata_csv,
        args.data_root,
        args.max_slices,
        case_ids=pd.read_csv('/leonardo/home/userexternal/jcondess/LiRadsDetector/train_metadata.csv').case_id.to_list(),
        **resolve_augment_mode(args.augment_mode),
        ordinal_only=ordinal_only,
        cat_names=cat_names,
    )
    val_ds = LiRadsCaseDataset(
        '/leonardo/home/userexternal/jcondess/LiRadsDetector/train_metadata.csv',
        args.data_root,
        args.max_slices,
        case_ids=pd.read_csv('/leonardo/home/userexternal/jcondess/LiRadsDetector/val_metadata.csv').case_id.to_list(),
        ordinal_only=ordinal_only,
        cat_names=cat_names,
    )

    train_loader = InfiniteDataLoader(
        DataLoader(
            train_ds, batch_size=args.batch_size,
            sampler=make_balanced_sampler(train_ds.df["lirads_score"]) if args.balanced_sampling else None,
            shuffle=False if args.balanced_sampling else True,
            num_workers=args.num_workers, collate_fn=collate_cases,
        )
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_cases,
    )

    print_to_log(f"Datasets loaded.", log_path)

    backbone = Dinov2SliceEncoder.from_pretrained()
    model = LiRadsNet(
        backbone, use_cnn=args.use_cnn, use_clinical=args.use_clinical, use_cat_head=not ordinal_only,
        ordinal_head_type=ordinal_head_type, cat_names=cat_names,
    ).to(device)

    cat_idxs, ord_idxs = zip(*(label_to_targets(str(l), cat_names) for l in train_ds.df["lirads_score"]))
    ord_only = [o for o in ord_idxs if o >= 0]
    ord_counts = {i: ord_only.count(i) for i in range(len(config.ORDINAL_LABELS))}
    ord_criterion = build_ordinal_criterion(args, ord_counts, device)
    print_to_log(f"Ordinal head loss: {args.ordinal_loss}", log_path)
    if not ordinal_only:
        print_to_log(f"Category head labels ({len(cat_names)}-way): {cat_names}", log_path)

    if ordinal_only:
        cat_criterion = None
    else:
        cat_counts = {i: cat_idxs.count(i) for i in range(len(cat_names))}
        cat_criterion = nn.CrossEntropyLoss(weight=compute_class_weights(cat_counts, len(cat_names)).to(device))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_lr_scheduler(optimizer, args)

    start_epoch = 0
    best_score = -1.0
    if resume_checkpoint is not None and ordinal_head_changed:
        old_type = resume_checkpoint.get("ordinal_head_type", "softmax")
        print_to_log(
            f"  --resume: checkpoint's ordinal head is '{old_type}', this run's --ordinal_loss="
            f"{args.ordinal_loss!r} needs '{ordinal_head_type}' -- ord_head's output shape differs "
            "between the two, so it can't be resumed directly. Every other weight (backbone-adjacent "
            "trunk, cat_head, clinical/CNN branches) is still loaded from the checkpoint; only ord_head "
            "starts fresh, and the optimizer/scheduler/epoch counter reset to a fresh run since they're "
            "tied to ord_head's old parameters/shape. best_score still carries over as the bar this run "
            "needs to beat, since it's the same architecture-independent challenge metric.",
            log_path,
        )
        trunk_state = {k: v for k, v in resume_checkpoint["model_state_dict"].items() if not k.startswith("ord_head.")}
        model.load_state_dict(trunk_state, strict=False)
        best_score = resume_checkpoint.get("best_score", -1.0)
    elif resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        else:
            print_to_log("  --resume: checkpoint predates optimizer-state saving -- optimizer starts fresh.", log_path)
        if scheduler is not None and resume_checkpoint.get("scheduler_state_dict") is not None:
            try:
                scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            except Exception as e:
                print_to_log(f"  --resume: couldn't restore scheduler state ({e}) -- scheduler starts fresh.", log_path)
        start_epoch = resume_checkpoint.get("epoch", 0)
        best_score = resume_checkpoint.get("best_score", -1.0)
        print_to_log(
            f"  resuming after epoch {start_epoch} (best val final_score so far={best_score:.4f}); "
            f"running {args.epochs} more epoch(s).",
            log_path,
        )

    print_to_log(f"Model loaded.", log_path)

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        print_to_log('', log_path)
        print_to_log(f"Epoch {epoch}.", log_path)
        print_to_log(f"Current learning rate: {np.round(optimizer.param_groups[0]['lr'], decimals=5)}", log_path)
        model.train()
        model.backbone.eval()  # frozen backbone: never let dropout/drop-path move it

        total_loss, n_batches = 0.0, 0
        for _ in range(args.num_iterations_per_epoch):
            batch = next(train_loader)
            cat_idx = batch["cat_idx"].to(device)
            ord_idx = batch["ord_idx"].to(device)
            logits_cat, logits_ord = model(batch["phase_data"], batch["clinical_features"])
            if ordinal_only:
                # every case in the batch is ordinal-labeled already (see
                # LiRadsCaseDataset's ordinal_only filtering), so ord_idx is
                # always valid -- no masking needed, and there's no cat_head
                # / cat_criterion to contribute a loss term.
                loss = ord_criterion(logits_ord, ord_idx)
            else:
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

        if args.tta_views > 0:
            val_preds = run_inference_tta(model, val_ds, device, args.tta_views)
        else:
            val_preds = run_inference(model, val_loader, device)
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = os.path.join(tmp, "gt.csv")
            pred_path = os.path.join(tmp, "pred.csv")
            val_ds.df.rename(columns={"lirads_score": "label"})[["case_id", "label"]].to_csv(gt_path, index=False)
            val_preds.to_csv(pred_path, index=False)
            result = compute_challenge_score(gt_path, pred_path, bootstrap=False)

        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(result["final_score"])
            else:
                scheduler.step()

        print_to_log(
            f"Epoch {epoch}: train_loss={avg_loss:.4f} "
            f"final_score={result['final_score']:.4f} "
            f"qwk={result['adjusted_qwk']:.4f} scr={result['special_category_recognition']:.4f}",
            log_path
        )
        val_merged = val_ds.df[["case_id", "lirads_score"]].astype({"case_id": str}).merge(
            val_preds.astype({"case_id": str}), on="case_id", how="inner",
        )
        log_per_class_metrics(compute_per_class_metrics(val_merged["lirads_score"], val_merged["prediction"]), log_path)

        if result["final_score"] > best_score:
            best_score = result["final_score"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": epoch,
                    "best_score": best_score,
                    "use_cnn": args.use_cnn,
                    "use_clinical": args.use_clinical,
                    "use_cat_head": not ordinal_only,
                    "ordinal_head_type": ordinal_head_type,
                    "cat_names": cat_names,
                },
                args.out,
            )
            print_to_log(f"  saved new best checkpoint to {args.out} (score={best_score:.4f})", log_path)

    print_to_log(f"Training complete. best val final_score={best_score:.4f}", log_path)

    test_ds = LiRadsCaseDataset(
        args.metadata_csv, args.data_root, args.max_slices, case_ids=fold["test"], ordinal_only=ordinal_only,
        cat_names=cat_names,
    )
    if os.path.exists(args.out):
        checkpoint = torch.load(args.out, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print_to_log(f"  no checkpoint was ever saved to {args.out}; testing with the last epoch's in-memory weights", log_path)

    if args.tta_views > 0:
        print_to_log(f"  running test-time augmentation ({args.tta_views} views) on the ordinal decision", log_path)
        test_preds = run_inference_tta(model, test_ds, device, args.tta_views)
    else:
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_cases,
        )
        test_preds = run_inference(model, test_loader, device)
    with tempfile.TemporaryDirectory() as tmp:
        gt_path = os.path.join(tmp, "gt.csv")
        pred_path = os.path.join(tmp, "pred.csv")
        test_ds.df.rename(columns={"lirads_score": "label"})[["case_id", "label"]].to_csv(gt_path, index=False)
        test_preds.to_csv(pred_path, index=False)
        test_result = compute_challenge_score(gt_path, pred_path, bootstrap=False)

    print_to_log(
        f"Test (fold {args.fold}): final_score={test_result['final_score']:.4f} "
        f"qwk={test_result['adjusted_qwk']:.4f} scr={test_result['special_category_recognition']:.4f}",
        log_path
    )

    test_pred_path = args.test_predictions_out or os.path.splitext(args.out)[0] + "_test_predictions.csv"
    test_preds.to_csv(test_pred_path, index=False)
    print_to_log(f"  saved test predictions to {test_pred_path}", log_path)

    cm_path = fold_tagged_path(os.path.splitext(test_pred_path)[0] + "_confusion_matrix.png", args.fold)
    merged = test_ds.df[["case_id", "lirads_score"]].astype({"case_id": str}).merge(
        test_preds.astype({"case_id": str}), on="case_id", how="inner",
    )
    save_confusion_matrix(merged["lirads_score"], merged["prediction"], cm_path)
    print_to_log(f"  saved test confusion matrix to {cm_path}", log_path)

    metrics_df = compute_per_class_metrics(merged["lirads_score"], merged["prediction"])
    metrics_path = fold_tagged_path(os.path.splitext(test_pred_path)[0] + "_per_class_metrics.csv", args.fold)
    metrics_df.to_csv(metrics_path, index=False)
    print_to_log(f"  saved per-class precision/recall to {metrics_path}", log_path)
    log_per_class_metrics(metrics_df, log_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AMPLIFAI LI-RADS classifier")
    parser.add_argument("--data_root", required=True, help="Root dir containing extracted case folders")
    parser.add_argument("--metadata_csv", required=True, help="CSV with case_id + lirads_score for the whole dataset")
    parser.add_argument("--splits_json", required=True, help="output of `python -m lirads_model.splits`")
    parser.add_argument("--fold", type=int, default=0, help="fold index into splits_json to train/test on")
    parser.add_argument("--test_predictions_out", default="checkpoints/test_pred.csv", help="where to save test-split predictions CSV")
    parser.add_argument(
        "--use_cnn", action=argparse.BooleanOptionalAction, default=True,
        help="use the per-phase 3D-CNN volume branch alongside DINOv2 (--no-use_cnn for DINOv2-only)",
    )
    parser.add_argument(
        "--use_clinical", action=argparse.BooleanOptionalAction, default=True,
        help="use the clinical/tabular feature branch (aphe/washout/capsule) (--no-use_clinical for images-only)",
    )
    parser.add_argument(
        "--head_mode", choices=["dual", "ordinal"], default="dual",
        help=(
            "'dual' (default): the current 4-way category head + 5-way ordinal head, jointly trained on the "
            "full label set (LR-1..5, LR-M, LR-TIV, No lesion). 'ordinal': single-head training on the "
            "ordinal target alone -- no category head at all, and LR-M/LR-TIV/No lesion cases are dropped "
            "from train/val/test (see LiRadsCaseDataset's ordinal_only), since they have no meaningful "
            "ordinal target and there's no category head left to route them through."
        ),
    )
    parser.add_argument(
        "--include_no_lesion", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "include config.NO_LESION_LABEL ('No lesion') cases in train/val/test and its own slot in the "
            "category head (default). --no-include_no_lesion drops those cases from every split and shrinks "
            "the category head from 4-way (ordinal/LR-M/LR-TIV/No lesion) to 3-way (ordinal/LR-M/LR-TIV) -- "
            "see dataset.LiRadsCaseDataset's cat_names. Only meaningful with --head_mode dual: --head_mode "
            "ordinal already drops every special class (LR-M/LR-TIV/No lesion), this one included, since "
            "there's no category head left for any of them to route through."
        ),
    )
    parser.add_argument(
        "--ordinal_loss", choices=["ce", "sord", "corn_qwk"], default="ce",
        help=(
            "loss for the ordinal head (LR-1..LR-5). 'ce' (default): plain weighted "
            "CrossEntropyLoss over a 5-way softmax head, treating the 5 ranks as unrelated "
            "categories. 'sord': SORD (Soft Ordinal Regression) over that same 5-way softmax "
            "head -- the one-hot target is replaced with a rank-distance-softened distribution, "
            "so a near-miss prediction (e.g. LR-3 on a true LR-4) is penalized less than a far "
            "one (e.g. LR-1 on a true LR-4). 'corn_qwk': switches the ordinal head itself to a "
            "4-output CORN (rank-consistent conditional) parameterization and trains it with "
            "L = CORN_loss + --qwk_lambda * (1 - soft_QWK), a differentiable quadratic-weighted-"
            "kappa term added on top of CORN's per-threshold supervision -- meant as a "
            "fine-tuning stage on top of a checkpoint already stable under 'ce'/'sord' (see "
            "--resume), since it directly targets the challenge's own QWK-based metric. Only "
            "affects the ordinal head; the category head (LR-M/LR-TIV/No lesion vs. ordinal) "
            "always uses plain CE, since those classes have no ordinal relationship. Independent "
            "of --resume's architecture lock: 'ce'/'sord' share one head shape and resume freely "
            "between each other, while switching to/from 'corn_qwk' reuses every other weight "
            "from the checkpoint but reinitializes ord_head fresh (see build_ordinal_head_type() "
            "and the --resume handling in train())."
        ),
    )
    parser.add_argument(
        "--qwk_lambda", type=float, default=1.0,
        help="[corn_qwk] weight of the soft-QWK term in L = CORN_loss + qwk_lambda * (1 - soft_QWK)",
    )
    parser.add_argument(
        "--augment_mode", choices=sorted(AUGMENT_MODES), default="spatial",
        help=(
            "which train-split augmentation strategies to apply, layered in order: 'none' disables all of them; "
            "'spatial' applies random rotation/zoom/flip/intensity augmentation (augmentation.py) only; "
            "'spatial+transplant' additionally pastes a donor case's real lesion into a different recipient "
            "case's liver for config.TRANSPLANT_DONOR_LABELS (LR-1/LR-2/LR-3 by default, see lesion_transplant.py); "
            "'spatial+anatomy' additionally warps each case around a random local deformation of its own lesion "
            "segmentation (augmentation.apply_anatomy_informed_deform, see config.ANATOMY_*); 'all' applies every strategy. "
            "transplant requires scripts/segment_livers.py to have already produced a liver.nii.gz for the "
            "relevant cases; cases missing one simply fall back to their own real data."
        ),
    )
    parser.add_argument(
        "--balanced_sampling", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "oversample rare lirads_score classes (e.g. LR-TIV) via a WeightedRandomSampler on the train split, "
            "instead of plain random shuffling (--no-balanced_sampling to disable)"
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "resume from the checkpoint already at --out (fold-tagged), if one exists: restores model, "
            "optimizer, and LR scheduler state, plus the best val final_score seen so far, and runs --epochs "
            "more epochs on top of it. The checkpoint's own use_cnn/use_clinical/head_mode (whatever it was "
            "originally trained with) override this run's --use_cnn/--use_clinical/--head_mode, since a "
            "model's architecture can't change mid-training. If --out doesn't exist yet, starts fresh instead."
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_iterations_per_epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler", choices=["none", "cosine", "step", "plateau"], default="none",
        help=(
            "epoch-level learning rate schedule, stepped once per epoch after that epoch's validation "
            "(see make_lr_scheduler()). 'none' (default): --lr stays constant for the whole run. "
            "'cosine': cosine decay from --lr down to --lr_min over --epochs. 'step': multiply by --lr_gamma "
            "every --lr_step_size epochs. 'plateau': multiply by --lr_gamma when val final_score hasn't "
            "improved for --lr_patience epochs."
        ),
    )
    parser.add_argument("--lr_min", type=float, default=0.0, help="[cosine] learning rate at the end of the schedule")
    parser.add_argument("--lr_step_size", type=int, default=10, help="[step] epochs between each decay")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="[step/plateau] multiplicative decay factor")
    parser.add_argument(
        "--lr_patience", type=int, default=3,
        help="[plateau] epochs with no val final_score improvement before decaying",
    )
    parser.add_argument("--max_slices", type=int, default=config.MAX_SLICES_PER_CASE)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="checkpoints/lirads_model.pt")
    parser.add_argument(
        "--tta_views", type=int, default=0,
        help=(
            "test-time augmentation, applied to both the per-epoch val-split scoring and the final "
            "held-out test-split evaluation: average this many extra augmented forward passes into the "
            "LR-1..LR-5 ordinal decision (0 disables TTA). Non-zero multiplies per-epoch validation cost, "
            "since it reruns every ordinal-gated val case tta_views+1 times."
        ),
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
