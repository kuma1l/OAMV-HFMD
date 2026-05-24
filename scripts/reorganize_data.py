"""
Production version of Sanity Stream 2 — reorganize Kaggle Hotels-8k into the
per-hotel layout MV-HFMD's loader expects.

Usage:
    python scripts/reorganize_data.py
        --csv      D:\\Research-WS\\PIVOT\\hotels-8k\\train.csv
        --src-root D:\\Research-WS\\PIVOT\\hotels-8k\\train_images
        --dst-root D:\\Research-WS\\PIVOT\\mvhfmd_data

Hardlinks to avoid duplicating ~24 GB of images. PLAN.md §2.2.
"""
from __future__ import annotations
import argparse, csv, os, shutil
from pathlib import Path
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv",      type=str, default=r"D:\Research-WS\PIVOT\hotels-8k\train.csv")
    p.add_argument("--src-root", type=str, default=r"D:\Research-WS\PIVOT\hotels-8k\train_images")
    p.add_argument("--dst-root", type=str, default=r"D:\Research-WS\PIVOT\mvhfmd_data")
    return p.parse_args()


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    by_hotel = defaultdict(list)
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            by_hotel[row["hotel_id"]].append((row["image"], row["chain"]))

    n_linked = n_copied = n_missing = 0
    for h, items in by_hotel.items():
        (dst_root / h).mkdir(parents=True, exist_ok=True)
        for img, chain in items:
            src = src_root / chain / img
            dst = dst_root / h / img
            if dst.exists():
                continue
            if not src.exists():
                n_missing += 1
                continue
            try:
                os.link(src, dst)
                n_linked += 1
            except OSError:
                shutil.copy2(src, dst)
                n_copied += 1
    print(f"linked={n_linked} copied={n_copied} missing={n_missing}")


if __name__ == "__main__":
    main()
