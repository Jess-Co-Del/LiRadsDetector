"""
Profiles per-function timing of the actual training iteration loop (fetch a
batch -> forward -> backward -> optimizer step) that lirads_model.train runs
every epoch, using cProfile, and writes a readable, sorted summary to a .log
file -- plus the raw pstats dump for deeper digging with snakeviz/tuna.

Reuses train.py's own dataset/model/loss-construction code (LiRadsCaseDataset,
InfiniteDataLoader, LiRadsNet, build_ordinal_criterion, ...) so it profiles
the real training path, but deliberately skips the per-epoch val/test
evaluation train.py also runs -- those touch the whole val/test split and
would dwarf and obscure the training-loop timing this script exists to
isolate. Point this at whatever's causing a slow training launch (e.g. "2
hours for 10 batches") to see which function is actually eating the time --
preprocessing.load_case_volumes's per-sample, per-phase full-volume
scipy.ndimage.zoom resample is a likely suspect (see its docstring).

IMPORTANT: run with --num_workers 0 (the default here, unlike train.py's
default of 2). With num_workers > 0, DataLoader batches are built in worker
*subprocesses* that cProfile -- which only instruments the current process --
can't see into, so all that time collapses into a single opaque "waiting for
worker" frame instead of the actual functions responsible.

Also note: on CUDA, individual forward/backward timings can be misleading,
since GPU kernels queue asynchronously and cProfile only sees when a Python
call returns, not when the GPU actually finishes -- per-iteration total time
is still accurate (loss.item() below forces a sync every iteration, same as
train.py's own loop), but time attributed to particular sub-calls inside the
forward/backward can be off. For a precise CUDA kernel-level breakdown, use
torch.profiler instead. cProfile itself also adds real overhead (commonly
20-50%), so treat the absolute wall-clock numbers as approximate and the
relative proportions between functions as the actionable signal.

Usage:
    python -m scripts.profile_train \\
        --data_root ./data/cases --metadata_csv ./train_metadata.csv \\
        --splits_json ./data/splits.json --fold 0 \\
        --iterations 5 --batch_size 4 --num_workers 0
"""

import argparse
import cProfile
import io
import os
import pstats
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from lirads_model import config  # noqa: E402
from lirads_model.backbone import Dinov2SliceEncoder  # noqa: E402
from lirads_model.config import print_to_log  # noqa: E402
from lirads_model.dataset import AUGMENT_MODES, LiRadsCaseDataset, collate_cases, label_to_targets, resolve_augment_mode  # noqa: E402
from lirads_model.model import LiRadsNet  # noqa: E402
from lirads_model.splits import load_fold  # noqa: E402
from lirads_model.train import (  # noqa: E402
    InfiniteDataLoader,
    build_ordinal_criterion,
    build_ordinal_head_type,
    compute_class_weights,
    make_balanced_sampler,
)


def _write_stats(profiler: cProfile.Profile, log_path: str, prof_path: str, sort_keys, top_n: int) -> None:
    """Appends a readable pstats summary (one section per sort key) to
    log_path, and dumps the raw profile to prof_path for later inspection
    (e.g. `snakeviz prof_path` for a flame-graph view)."""
    profiler.dump_stats(prof_path)
    for sort_key in sort_keys:
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs().sort_stats(sort_key).print_stats(top_n)
        print_to_log(f"--- top {top_n} functions sorted by {sort_key} ---\n{stream.getvalue()}", log_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the lirads_model training iteration loop with cProfile")
    parser.add_argument("--data_root", required=True, help="root dir containing extracted case folders")
    parser.add_argument("--metadata_csv", required=True, help="CSV with case_id + lirads_score for the whole dataset")
    parser.add_argument("--splits_json", required=True, help="output of `python -m lirads_model.splits`")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--iterations", type=int, default=5,
        help="number of training iterations to profile (train.py's equivalent is --num_iterations_per_epoch); "
        "kept small on purpose -- this is meant to characterize where one pass spends its time, not to train",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_slices", type=int, default=config.MAX_SLICES_PER_CASE)
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="0 (default here, unlike train.py's default of 2) so cProfile can see inside __getitem__ -- see module docstring",
    )
    parser.add_argument("--use_cnn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_clinical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--head_mode", choices=["dual", "ordinal"], default="dual")
    parser.add_argument("--ordinal_loss", choices=["ce", "sord", "corn_qwk"], default="ce")
    parser.add_argument("--qwk_lambda", type=float, default=1.0)
    parser.add_argument("--augment_mode", choices=sorted(AUGMENT_MODES), default="spatial")
    parser.add_argument("--balanced_sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--profile_out", default=os.path.join(config.LOG_DIR, "profile_train.log"))
    parser.add_argument(
        "--profile_sort", nargs="+", default=["cumulative", "tottime"], choices=["cumulative", "tottime", "calls"],
        help="pstats sort key(s), each reported as its own section: cumulative = time in the function plus "
        "everything it calls; tottime = time in the function itself only",
    )
    parser.add_argument("--profile_top_n", type=int, default=40)
    args = parser.parse_args()

    log_path = args.profile_out
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    print_to_log("=" * 70, log_path)
    print_to_log(
        f"Profiling {args.iterations} training iteration(s): batch_size={args.batch_size}, "
        f"num_workers={args.num_workers}, device={args.device}, ordinal_loss={args.ordinal_loss}.",
        log_path,
    )
    if args.num_workers != 0:
        print_to_log(
            "  WARNING: --num_workers != 0 -- DataLoader batches build in worker subprocesses cProfile "
            "can't see into, so preprocessing time will collapse into an opaque wait instead of showing "
            "the actual functions responsible. Re-run with --num_workers 0 for meaningful results.",
            log_path,
        )
    print_to_log("=" * 70, log_path)

    device = torch.device(args.device)
    ordinal_only = args.head_mode == "ordinal"
    ordinal_head_type = build_ordinal_head_type(args)
    fold = load_fold(args.splits_json, args.fold)

    train_ds = LiRadsCaseDataset(
        args.metadata_csv, args.data_root, args.max_slices, case_ids=fold["train"],
        **resolve_augment_mode(args.augment_mode), ordinal_only=ordinal_only,
    )
    train_loader = InfiniteDataLoader(
        DataLoader(
            train_ds, batch_size=args.batch_size,
            sampler=make_balanced_sampler(train_ds.df["lirads_score"]) if args.balanced_sampling else None,
            shuffle=False if args.balanced_sampling else True,
            num_workers=args.num_workers, collate_fn=collate_cases,
        )
    )
    print_to_log(f"Dataset loaded ({len(train_ds)} cases in the train split).", log_path)

    backbone = Dinov2SliceEncoder.from_pretrained()
    model = LiRadsNet(
        backbone, use_cnn=args.use_cnn, use_clinical=args.use_clinical, use_cat_head=not ordinal_only,
        ordinal_head_type=ordinal_head_type,
    ).to(device)

    cat_idxs, ord_idxs = zip(*(label_to_targets(str(l)) for l in train_ds.df["lirads_score"]))
    ord_only = [o for o in ord_idxs if o >= 0]
    ord_counts = {i: ord_only.count(i) for i in range(len(config.ORDINAL_LABELS))}
    ord_criterion = build_ordinal_criterion(args, ord_counts, device)
    if ordinal_only:
        cat_criterion = None
    else:
        cat_counts = {i: cat_idxs.count(i) for i in range(len(config.CAT_NAMES))}
        cat_criterion = nn.CrossEntropyLoss(weight=compute_class_weights(cat_counts, len(config.CAT_NAMES)).to(device))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay,
    )
    print_to_log("Model loaded. Starting profiled iterations.", log_path)

    def run_iterations() -> float:
        """Mirrors train.py's train()'s per-iteration loop body exactly
        (fetch batch -> forward -> loss -> backward -> step), minus the
        per-epoch val/test evaluation around it -- see module docstring for
        why that's excluded from the profile."""
        model.train()
        model.backbone.eval()
        total_loss = 0.0
        for _ in range(args.iterations):
            batch = next(train_loader)
            cat_idx = batch["cat_idx"].to(device)
            ord_idx = batch["ord_idx"].to(device)
            logits_cat, logits_ord = model(batch["phase_data"], batch["clinical_features"])
            if ordinal_only:
                loss = ord_criterion(logits_ord, ord_idx)
            else:
                loss = cat_criterion(logits_cat, cat_idx)
                ord_mask = cat_idx == 0
                if ord_mask.any():
                    loss = loss + ord_criterion(logits_ord[ord_mask], ord_idx[ord_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()  # forces a CUDA sync, same as train.py's own loop
        return total_loss / max(args.iterations, 1)

    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    try:
        avg_loss = run_iterations()
    finally:
        profiler.disable()
    elapsed = time.perf_counter() - start

    print_to_log(
        f"Profiled {args.iterations} iteration(s) in {elapsed:.2f}s ({elapsed / max(args.iterations, 1):.2f}s/iteration; "
        f"avg_loss={avg_loss:.4f}). Note: cProfile overhead inflates this vs. an unprofiled run.",
        log_path,
    )

    prof_path = os.path.splitext(log_path)[0] + ".prof"
    _write_stats(profiler, log_path, prof_path, args.profile_sort, args.profile_top_n)
    print_to_log(f"Done. Full profiling log at {log_path}, raw pstats at {prof_path} (inspect with `snakeviz {prof_path}`).", log_path)


if __name__ == "__main__":
    main()
