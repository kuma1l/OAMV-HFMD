"""CheXpert MV cross-domain (E11).

Usage:
    python scripts/09_chexpert_mv.py --config configs/E11_chexpert.yaml --seed 42 [--n-views 4]

Reads YAML config, instantiates model/data/loss/trainer per PLAN.md §10, runs
training, evaluates exhaustively, dumps results to ``results/<run>/``.

PLAN.md §6.E11.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from oamv_hfmd.utils import set_seed, get_logger, load_yaml, dump_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--n-views", type=int, default=None,
                   help="Override n_views from config")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Override paths.out_dir from config")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate config + instantiate model, do not train")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(args.seed)

    out_dir = Path(args.out_dir or cfg["paths"]["out_dir"]) / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("09_chx", out_dir / "train.log")
    logger.info(f"Config: {args.config}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Out dir: {out_dir}")

    if args.dry_run:
        logger.info("Dry run — exit before training")
        return

    raise NotImplementedError("Wire trainer per PLAN.md §10 / §E11")


if __name__ == "__main__":
    main()
