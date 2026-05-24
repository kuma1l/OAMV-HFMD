"""
Production version of Sanity Stream 3 — build fallback train/val/test .npy
splits for MV-HFMD's loader.

Usage:
    python scripts/build_splits.py
        --data-root D:\\Research-WS\\PIVOT\\mvhfmd_data
        --min-images 2
        --seed 42

Per-hotel 80/10/10 split with md5(hotel_id)-derived seed. PLAN.md §3.2.
"""
from __future__ import annotations
import argparse, hashlib, random
from pathlib import Path
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",  type=str, required=True)
    p.add_argument("--min-images", type=int, default=2)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    hotels = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))

    by_hotel = {}
    for h in hotels:
        imgs = sorted(f.name for f in (root / h).iterdir()
                      if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if len(imgs) >= args.min_images:
            by_hotel[h] = imgs

    train, val, test = [], [], []
    for h in sorted(by_hotel):
        imgs = list(by_hotel[h])
        seed_int = int(hashlib.md5(h.encode()).hexdigest(), 16) % (2 ** 32)
        rng = random.Random(args.seed + seed_int)
        rng.shuffle(imgs)
        n = len(imgs)
        n_train = max(1, int(0.8 * n))
        n_val   = max(1, int(0.1 * n))
        n_test  = n - n_train - n_val
        while n_test < 1 and n_train > 1:
            n_train -= 1
            n_test = n - n_train - n_val
        if n_test < 1 or n_val < 1 or n_train < 1:
            continue
        for img in imgs[:n_train]:                       train.append(f"{h}/{img}")
        for img in imgs[n_train:n_train + n_val]:        val.append(f"{h}/{img}")
        for img in imgs[n_train + n_val:]:               test.append(f"{h}/{img}")

    np.save(root / "train.npy", np.array(train))
    np.save(root / "val.npy",   np.array(val))
    np.save(root / "test.npy",  np.array(test))
    print(f"hotels={len(by_hotel)} train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
