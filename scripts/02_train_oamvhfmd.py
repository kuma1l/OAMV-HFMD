"""Train OAMV-HFMD variant (E2 combined; E3 embed_only / E4 loss_only flagged TODO).

Usage:
    python scripts/02_train_oamvhfmd.py \\
        --config configs/E2_oamvhfmd_upstream.yaml \\
        --seed 42 \\
        --n-views 4 \\
        [--variant combined] [--epochs 1] [--batch-size 40] [--out-dir results/smoke]

Reads YAML config, instantiates OverlapAwareHybrid + overlap_md_loss per PLAN.md
§4.2/§4.3, runs training, evaluates exhaustively, dumps results to ``results/<run>/``.

PLAN.md §6.E2.
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
from oamv_hfmd.losses import overlap_md_loss
from oamv_hfmd.model import OverlapAwareHybrid
from oamv_hfmd.trainer import TrainConfig, Trainer
from oamv_hfmd.utils import dump_json, get_logger, load_yaml, set_seed, worker_init_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--n-views", type=str, default=None,
                   help="Override n_views from config. Accepts int or 'single'.")
    p.add_argument("--variant", type=str, default="combined",
                   choices=["combined", "embed_only", "loss_only"],
                   help="OAMV ablation. Only 'combined' is implemented; the isolation "
                        "variants are flagged follow-ups (see NotImplementedError below).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epochs from config (use 1 for smoke runs).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch size from config. OAMV adds DINOv2-S forward; "
                        "expect higher VRAM than baseline at same batch size.")
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

    if args.variant != "combined":
        raise NotImplementedError(
            "embed_only/loss_only need the trainer's similarity-based routing "
            "(trainer.py L192) decoupled from the loss choice, since the "
            "OverlapAwareHybrid always returns 'similarity' at n>1 — handle as a "
            "follow-up (E3/E4)."
        )

    cfg = _load_config_with_inherits(args.config)
    set_seed(args.seed)

    n_views = _resolve_n_views(args.n_views, cfg)
    epochs = args.epochs if args.epochs is not None else cfg["train"]["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["train"]["batch_size"]
    val_batch_size = cfg["train"].get("val_batch_size", 128)
    num_workers = args.num_workers if args.num_workers is not None else cfg["train"]["num_workers"]
    dl_extras = dict(persistent_workers=True, prefetch_factor=4) if num_workers > 0 else {}

    out_root = Path(args.out_dir or cfg["paths"]["out_dir"])
    if args.out_dir is not None:
        out_dir = out_root / f"seed{args.seed}"
    else:
        out_dir = out_root / _dir_name_for(n_views, args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(f"02_oamvhfmd.{out_dir.name}", out_dir / "train.log")
    logger.info(f"config={args.config} variant={args.variant} seed={args.seed} "
                f"n_views={n_views} epochs={epochs} batch_size={batch_size} out_dir={out_dir}")

    # --- snapshot config.json
    data_dir = cfg["paths"]["data_dir"]
    arch = cfg["model"]["arch"]
    oracle_kind = cfg["oracle"]["kind"]
    mlp_hidden = cfg["oracle"]["mlp_hidden"]
    config_snapshot = {
        "method": cfg.get("method", "oamvhfmd_combined_upstream"),
        "variant": args.variant,
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
        "oracle_kind": oracle_kind,
        "mlp_hidden": mlp_hidden,
        "grad_clip": cfg["train"]["grad_clip"],
        "amp": cfg["train"]["amp"],
        "label_smoothing": cfg["train"]["label_smoothing"],
        "ce_ignore_index": cfg["train"]["ce_ignore_index"],
        "split_source": cfg.get("split_source", "unknown"),
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
        worker_init_fn=worker_init_fn, **dl_extras,
    )
    val_loader = DataLoader(
        ds_val, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=pin,
        worker_init_fn=worker_init_fn, **dl_extras,
    )
    test_loader = DataLoader(
        ds_test, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=pin,
        worker_init_fn=worker_init_fn, **dl_extras,
    )

    # --- model
    logger.info(f"building OverlapAwareHybrid arch={arch} n={n_views} "
                f"oracle={oracle_kind} mlp_hidden={mlp_hidden}")
    model = OverlapAwareHybrid(
        arch=arch,
        num_classes=ds_train.num_classes,
        n=n_views,
        pretrained_weights=cfg["model"].get("pretrained", True),
        oracle_kind=oracle_kind,
        mlp_hidden=mlp_hidden,
        enable_overlap_embed=cfg["model"].get("enable_overlap_embed", True),
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

    # Trainer auto-routes via output["similarity"] — pass overlap_md_loss directly.
    md_loss_fn = overlap_md_loss if n_views > 1 else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        config=train_cfg, md_loss_fn=md_loss_fn, device=device, out_dir=out_dir,
    )

    t_train_start = time.time()
    try:
        conv_report = trainer.fit()
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA OOM at batch_size={batch_size}. OAMV adds a frozen DINOv2-S "
                     f"forward per view, so VRAM is higher than the baseline. Retry with "
                     f"--batch-size {max(1, batch_size // 2)} (and re-run the baseline at "
                     f"the same batch size to keep the A/B fair — batch size changes the "
                     f"OneCycle schedule).")
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

    # --- flatten score_dict -> eval_results.json
    # NOTE: per-image full-test single-view eval (the n=1 second pass that script
    # 01 runs) is skipped for OAMV. An n=1 OverlapAwareHybrid has the oracle+MLP
    # params and no similarity path, so the comparison would be confounded;
    # treat as follow-up if needed. single_view_per_image_top1_full_test is null.
    if "mv_collection" in score_dict:
        test_top1 = float(score_dict["mv_collection"]["top1_acc"])
        test_top5 = float(score_dict["mv_collection"]["top5_acc"])
        single_view_top1 = float(score_dict["single"]["top1_acc"])
    else:
        test_top1 = float(score_dict["single"]["top1_acc"])
        test_top5 = float(score_dict["single"]["top5_acc"])
        single_view_top1 = test_top1

    eval_results = {
        "test_top1": test_top1,
        "test_top5": test_top5,
        "single_view_top1": single_view_top1,
        "single_view_per_image_top1_full_test": None,  # see note above
        "single_view_per_image_n_evaluated": 0,
        "n_test_hotels": int(ds_train.num_classes),
        "n_test_samples": int(len(ds_test)),
        "n_views": int(n_views),
        "eval_mode": "exhaustive_combinations",
        "best_checkpoint_epoch": int(conv_report.get("best_qualifying_ckpt_epoch", 0)),
        "best_val_top1": float(conv_report.get("best_val_top1", 0.0)),
        "elapsed_train_seconds": float(elapsed_train),
        "per_view_metrics": {
            vt: {k: float(v) for k, v in metrics.items()}
            for vt, metrics in score_dict.items()
        },
    }
    dump_json(eval_results, out_dir / "eval_results.json")
    logger.info(f"wrote {out_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
