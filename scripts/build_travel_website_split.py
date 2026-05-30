"""Build the travel_website test split for the Stage-1 Go/No-Go diagnostic.

Walks images/train/<chain>/<hotel_id>/travel_website/, keeps hotels that
(a) are in our 7,754-hotel Stream-3 class set AND (b) have >= 4 travel_website
images. Hardlinks each image to mvhfmd_data/<hotel_id>/tw_<original_filename>
(falls back to copy on OSError). Writes mvhfmd_data/test_travel_website.npy
as an array of "<hotel_id>/tw_<filename>" strings (same convention as
{train,val,test}.npy).

Diagnostic-only. Does not modify the existing splits.
"""
from __future__ import annotations
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MV   = REPO / "mvhfmd_data"
IMG  = REPO / "images" / "train"
MIN_VIEWS = 4
CAP_K       = 10
CAP_SEED    = 42

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    hotel_classes_path = REPO / "hotel_classes.txt"
    classes = [ln.strip() for ln in open(hotel_classes_path) if ln.strip()]
    class_set = set(classes)
    print(f"hotel_classes: {len(classes)} ids")
    rng = random.Random(CAP_SEED)

    # Walk images/train/<chain>/<hotel>/travel_website/*
    n_seen_hotels = 0
    rel_entries: list[str] = []
    n_link = 0
    n_copy = 0
    n_skip_exists = 0
    n_capped_hotels = 0
    selected_hotels: list[str] = []

    for chain in sorted(IMG.iterdir()):
        if not chain.is_dir():
            continue
        for hotel_dir in sorted(chain.iterdir()):
            if not hotel_dir.is_dir():
                continue
            hid = hotel_dir.name
            if hid not in class_set:
                continue
            tw = hotel_dir / "travel_website"
            if not tw.exists() or not tw.is_dir():
                continue
            n_seen_hotels += 1
            imgs = sorted([p for p in tw.iterdir()
                           if p.is_file() and p.suffix.lower() in IMG_EXTS])
            if len(imgs) < MIN_VIEWS:
                continue
            dest_dir = MV / hid
            dest_dir.mkdir(parents=True, exist_ok=True)
            selected_hotels.append(hid)
            # Materialize ALL images on disk (cheap, hardlinks). The cap only
            # affects which entries land in test_travel_website.npy.
            for src in imgs:
                dest_name = f"tw_{src.name}"
                dest = dest_dir / dest_name
                if dest.exists():
                    n_skip_exists += 1
                else:
                    try:
                        os.link(src, dest)
                        n_link += 1
                    except OSError:
                        shutil.copy2(src, dest)
                        n_copy += 1

            # Seeded cap for the .npy entries (compute-budget control). Single
            # shared rng over hotels iterated in sorted order — reproducible.
            entries_for_hotel = sorted(f"tw_{src.name}" for src in imgs)
            if len(entries_for_hotel) > CAP_K:
                entries_for_hotel = sorted(rng.sample(entries_for_hotel, CAP_K))
                n_capped_hotels += 1
            for name in entries_for_hotel:
                rel_entries.append(f"{hid}/{name}")

    n_hotels = len(selected_hotels)
    print(f"hotels with travel_website dir (in 7754 set): {n_seen_hotels}")
    print(f"hotels selected (>= {MIN_VIEWS} travel_website images): {n_hotels}")
    print(f"images materialized: linked={n_link} copied={n_copy} preexisting={n_skip_exists}")
    print(f"per-hotel cap: K={CAP_K} seed={CAP_SEED}  hotels capped: {n_capped_hotels}")
    print(f"total test entries (after cap): {len(rel_entries)}")

    if n_hotels == 0:
        sys.exit("no hotels selected — aborting")

    # Sanity-check 3 random entries are on disk where we say they are.
    rng = random.Random(0)
    samples = rng.sample(rel_entries, min(3, len(rel_entries)))
    print("sample paths:")
    for s in samples:
        full = MV / s
        print(f"  {s}  exists={full.exists()}")

    # Write the split file.
    out = MV / "test_travel_website.npy"
    # Save as fixed-width unicode (matches existing {train,val,test}.npy dtype,
    # avoiding allow_pickle=False load failure).
    np.save(out, np.array(rel_entries))
    print(f"wrote {out}  (n={len(rel_entries)})")

    # Soft sanity flag vs CURRENT_STATUS's ~354 expectation.
    if abs(n_hotels - 354) / 354.0 > 0.20:
        print(f"WARNING: n_hotels={n_hotels} is >20% off from expected ~354")


if __name__ == "__main__":
    main()
