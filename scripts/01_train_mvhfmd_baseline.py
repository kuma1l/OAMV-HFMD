"""Train MV-HFMD baseline (E1).

Usage:
    python scripts/01_train_mvhfmd_baseline.py \\
        --config configs/E1_mvhfmd_baseline.yaml \\
        --seed 42 \\
        --n-views 4 \\
        [--epochs 1] [--batch-size 16] [--out-dir results/smoke]

Reads YAML config, instantiates model/data/loss/trainer per PLAN.md §10, runs
training, evaluates exhaustively, dumps results to ``results/<run>/``.

PLAN.md §6.E1.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import timm
import torch
from torch.utils.data import DataLoader

from oamv_hfmd.data import HotelsDataset
from oamv_hfmd.eval import Evaluator
from oamv_hfmd.losses import baseline_md_loss
from oamv_hfmd.model import MultiImageHybrid
from oamv_hfmd.trainer import TrainConfig, Trainer
from oamv_hfmd.utils import dump_json, get_logger, load_yaml, set_seed, worker_init_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--n-views", type=str, default=None,
                   help="Override n_views from config. Accepts int or 'single'.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epochs from config (use 1 for smoke runs).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch size from config. RTX 3050: 16, fallback 8.")
    p.add_argument("--num-workers", type=int, default=None,
                   help="Override num_workers (CPU loaders). Lower on Windows.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Override paths.out_dir from config.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate config + instantiate model, do not train")
    return p.parse_args()


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge — override wins; nested dicts merge."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_config_with_inherits(path: str | Path) -> dict:
    """load_yaml + recursively resolve top-level ``inherits:`` keys (path relative
    to the importing file's dir). Inheriting config wins on conflicts."""
    path = Path(path)
    cfg = load_yaml(path)
    inherits = cfg.pop("inherits", None)
    if inherits:
        parent_path = (path.parent / inherits).resolve()
        parent_cfg = _load_config_with_inherits(parent_path)
        cfg = _deep_merge(parent_cfg, cfg)
    return cfg


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _resolve_n_views(arg: str | None, cfg: dict) -> int:
    raw = arg if arg is not None else cfg["model"]["n_views"]
    if isinstance(raw, str):
        if raw.lower() == "single":
            return 1
        return int(raw)
    return int(raw)


def _dir_name_for(n_views: int, seed: int) -> str:
    return f"single_seed{seed}" if n_views == 1 else f"N{n_views}_seed{seed}"


def main():
    args = parse_args()
    cfg = _load_config_with_inherits(args.config)
    set_seed(args.seed)

    n_views = _resolve_n_views(args.n_views, cfg)
    epochs = args.epochs if args.epochs is not None else cfg["train"]["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["train"]["batch_size"]
    val_batch_size = cfg["train"].get("val_batch_size", 128)
    num_workers = args.num_workers if args.num_workers is not None else cfg["train"]["num_workers"]

    out_root = Path(args.out_dir or cfg["paths"]["out_dir"])
    # When --out-dir is given by the user, honor the spec's seedXX layout
    # (e.g. results/smoke/seed42/). Otherwise use the per-experiment
    # {N<k>,single}_seed<s>/ convention.
    if args.out_dir is not None:
        out_dir = out_root / f"seed{args.seed}"
    else:
        out_dir = out_root / _dir_name_for(n_views, args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(f"01_mvhfmd.{out_dir.name}", out_dir / "train.log")
    logger.info(f"config={args.config} seed={args.seed} n_views={n_views} "
                f"epochs={epochs} batch_size={batch_size} out_dir={out_dir}")

    # --- snapshot config.json
    data_dir = cfg["paths"]["data_dir"]
    arch = cfg["model"]["arch"]
    config_snapshot = {
        "method": cfg.get("method", "mvhfmd_baseline"),
        "arch": arch,
        "n_views": n_views,
        "seed": args.seed,
        "epochs": epochs,
        "lr": cfg["train"]["lr"],
        "weight_decay": cfg["train"]["weight_decay"],
        "momentum": cfg["train"]["momentum"],
        "batch_size": batch_size,
        "val_batch_size": val_batch_size,
        "lambda_md": cfg["loss"]["lambda_md"],
        "md_temp": cfg["loss"]["md_temp"],
        "tau_overlap": cfg["loss"].get("tau_overlap", 4.0),
        "grad_clip": cfg["train"]["grad_clip"],
        "amp": cfg["train"]["amp"],
        "label_smoothing": cfg["train"]["label_smoothing"],
        "ce_ignore_index": cfg["train"]["ce_ignore_index"],
        "split_source": "fallback_stream3_md5_seed42",
        "data_dir": data_dir,
        "git_commit": _git_commit(),
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
    }

    # --- datasets / loaders
    logger.info("building datasets")
    ds_train = HotelsDataset(data_dir, split="train", n=n_views, train=True)
    ds_val   = HotelsDataset(data_dir, split="val",   n=n_views, train=False, classes=ds_train.classes)
    ds_test  = HotelsDataset(data_dir, split="test",  n=n_views, train=False, classes=ds_train.classes)
    logger.info(f"train={len(ds_train)} val={len(ds_val)} test={len(ds_test)} "
                f"num_classes={ds_train.num_classes}")
    config_snapshot["num_classes"] = int(ds_train.num_classes)
    dump_json(config_snapshot, out_dir / "config.json")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=False, pin_memory=pin,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        ds_val, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=pin,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        ds_test, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=pin,
        worker_init_fn=worker_init_fn,
    )

    # --- model
    logger.info(f"building model arch={arch} n={n_views}")
    model = MultiImageHybrid(
        arch=arch, num_classes=ds_train.num_classes, n=n_views,
        pretrained_weights=cfg["model"].get("pretrained", True),
    )

    if args.dry_run:
        logger.info("dry-run: model instantiated, exiting before training")
        return

    # --- trainer
    train_cfg = TrainConfig(
        optimizer=cfg["train"]["optimizer"],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        momentum=cfg["train"]["momentum"],
        scheduler=cfg["train"]["scheduler"],
        div_factor=cfg["train"]["onecycle"]["div_factor"],
        final_div_factor=cfg["train"]["onecycle"]["final_div_factor"],
        pct_start_epochs=cfg["train"]["onecycle"]["pct_start_epochs"],
        anneal_strategy=cfg["train"]["onecycle"]["anneal_strategy"],
        epochs=epochs,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        grad_clip=cfg["train"]["grad_clip"],
        amp=cfg["train"]["amp"],
        seed=args.seed,
        n_views=n_views,
        num_workers=num_workers,
        label_smoothing=cfg["train"]["label_smoothing"],
        ce_ignore_index=cfg["train"]["ce_ignore_index"],
        lambda_md=cfg["loss"]["lambda_md"],
        md_temp=cfg["loss"]["md_temp"],
        tau_overlap=cfg["loss"].get("tau_overlap", 4.0),
    )

    # MD loss only when N > 1 (single-view runs skip it).
    md_loss_fn = baseline_md_loss if n_views > 1 else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        config=train_cfg, md_loss_fn=md_loss_fn, device=device, out_dir=out_dir,
    )

    t_train_start = time.time()
    try:
        conv_report = trainer.fit()
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA OOM at batch_size={batch_size}. Retry with "
                     f"--batch-size {max(1, batch_size // 2)} (then 8 if still failing).")
        raise SystemExit(2) from e
    elapsed_train = time.time() - t_train_start

    # --- final exhaustive eval on test set with the best weights (trainer.fit
    # restores them before returning).
    logger.info("running exhaustive test-set evaluation")
    evaluator = Evaluator(model=model, n=n_views, device=device)
    score_dict = evaluator.evaluate(test_loader)
    for view_type, metrics in score_dict.items():
        for metric, value in metrics.items():
            logger.info(f"test {view_type} {metric}: {value:.6f}")

    # --- per-image full-test single-view eval (no combo expansion).
    # The exhaustive eval above only covers C(M, N)-eligible hotels (e.g. for
    # N=4 on the fallback test split, only 633 of 7,754 hotels qualify). This
    # second pass forwards EVERY test image individually so we get an unbiased
    # single-view top-1 over the full test set, for comparison with the
    # `single_view_top1` field (which stays as the biased-subset number that
    # matches MV-HFMD's reported metric).
    logger.info("running per-image full-test single-view evaluation (n=1 pass)")
    t_pi_start = time.time()
    ds_per_image = HotelsDataset(data_dir, split="test", n=1, train=False, classes=ds_train.classes)
    per_image_loader = DataLoader(
        ds_per_image, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=pin,
        worker_init_fn=worker_init_fn,
    )
    # The trained model has n=n_views. Load its weights into a fresh n=1 copy
    # (img_embed_matrix shape differs; strict=False skips that one parameter,
    # which is only used by the mv_collection branch — irrelevant for single).
    per_image_model = MultiImageHybrid(
        arch=arch, num_classes=ds_train.num_classes, n=1, pretrained_weights=False,
    ).to(device)
    missing, unexpected = per_image_model.load_state_dict(model.state_dict(), strict=False)
    logger.info(f"per-image model load: missing={len(missing)} unexpected={len(unexpected)} "
                f"(img_embed_matrix shape mismatch is expected)")
    per_image_model.eval()

    pi_correct = 0
    pi_total = 0
    with torch.no_grad():
        for images, targets, _ in per_image_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).flatten()
            with torch.cuda.amp.autocast(enabled=cfg["train"]["amp"] and torch.cuda.is_available()):
                out = per_image_model(images)
            preds = out["single"]["logits"].argmax(dim=-1)
            pi_correct += (preds == targets).sum().item()
            pi_total += int(targets.numel())
    per_image_top1 = pi_correct / pi_total if pi_total > 0 else 0.0
    elapsed_per_image = time.time() - t_pi_start
    logger.info(f"per-image single-view top1={per_image_top1:.6f} over {pi_total} images "
                f"({elapsed_per_image:.1f}s)")

    # --- flatten score_dict -> results/.schema/eval_results.json
    if "mv_collection" in score_dict:
        test_top1 = float(score_dict["mv_collection"]["top1_acc"])
        test_top5 = float(score_dict["mv_collection"]["top5_acc"])
        single_view_top1 = float(score_dict["single"]["top1_acc"])
    else:
        # n=1 case
        test_top1 = float(score_dict["single"]["top1_acc"])
        test_top5 = float(score_dict["single"]["top5_acc"])
        single_view_top1 = test_top1

    eval_results = {
        "test_top1": test_top1,
        "test_top5": test_top5,
        "single_view_top1": single_view_top1,
        "single_view_per_image_top1_full_test": float(per_image_top1),
        "single_view_per_image_n_evaluated": int(pi_total),
        "n_test_hotels": int(ds_train.num_classes),
        "n_test_samples": int(len(ds_test)),
        "n_views": int(n_views),
        "eval_mode": "exhaustive_combinations",
        "best_checkpoint_epoch": int(conv_report.get("best_qualifying_ckpt_epoch", 0)),
        "best_val_top1": float(conv_report.get("best_val_top1", 0.0)),
        "elapsed_train_seconds": float(elapsed_train),
        "elapsed_per_image_eval_seconds": float(elapsed_per_image),
        "per_view_metrics": {
            vt: {k: float(v) for k, v in metrics.items()}
            for vt, metrics in score_dict.items()
        },
    }
    dump_json(eval_results, out_dir / "eval_results.json")
    logger.info(f"wrote {out_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
