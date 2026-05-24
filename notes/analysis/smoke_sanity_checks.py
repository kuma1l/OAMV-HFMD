"""
Sanity checks for the 1-epoch smoke run at results/smoke/seed42/.
Investigates whether the headline numbers (single 49.5% / mv 93.1%) are
real or inflated by eval subset bias / leakage / dedup bugs.

ASCII-only output for cp1252 compatibility on Windows consoles.
"""
from __future__ import annotations
import sys
import json
from collections import Counter
from pathlib import Path

# Belt-and-braces: force UTF-8 stdout even on cp1252 consoles
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

REPO     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path("D:/Research-WS/PIVOT/mvhfmd_data")
SMOKE    = REPO / "results" / "smoke" / "seed42"

print("=" * 70)
print("  SMOKE SANITY CHECKS -- OAMV-HFMD 1-epoch run, seed 42")
print("=" * 70)


# ----------------------------------------------------------------------
# CHECK 1: Eval subset bias -- how many hotels contribute to each N?
# ----------------------------------------------------------------------
print("\n[CHECK 1] Eval subset bias by N")
print("-" * 70)

test_paths = np.load(DATA_DIR / "test.npy")
hotel_counts = Counter(p.split("/")[0] for p in test_paths)
total_hotels = len(hotel_counts)
total_imgs   = sum(hotel_counts.values())

print(f"  Total test images:  {total_imgs}")
print(f"  Total test hotels:  {total_hotels}")
print(f"  Mean images/hotel:  {total_imgs/total_hotels:.2f}")
print(f"  Median:             {int(np.median(list(hotel_counts.values())))}")
print(f"  Max:                {max(hotel_counts.values())}")
print()
print(f"  {'N':>4} | {'eligible_hotels':>16} | {'pct hotels':>12} | "
      f"{'images covered':>16} | {'pct images':>12}")
print(f"  {'-'*4}-+-{'-'*16}-+-{'-'*12}-+-{'-'*16}-+-{'-'*12}")
for n in [2, 4, 6, 8]:
    eligible = sum(1 for c in hotel_counts.values() if c >= n)
    covered  = sum(c for c in hotel_counts.values() if c >= n)
    print(f"  {n:>4} | {eligible:>16} | {eligible/total_hotels*100:>11.1f}% "
          f"| {covered:>16} | {covered/total_imgs*100:>11.1f}%")


# ----------------------------------------------------------------------
# CHECK 2: Train/test path overlap (must be 0)
# ----------------------------------------------------------------------
print("\n[CHECK 2] Train/test path overlap")
print("-" * 70)

train = set(np.load(DATA_DIR / "train.npy"))
val   = set(np.load(DATA_DIR / "val.npy"))
test  = set(np.load(DATA_DIR / "test.npy"))
print(f"  |train|            = {len(train)}")
print(f"  |val|              = {len(val)}")
print(f"  |test|             = {len(test)}")
print(f"  |train cap val|    = {len(train & val):>4}  (must be 0)")
print(f"  |train cap test|   = {len(train & test):>4}  (must be 0)")
print(f"  |val cap test|     = {len(val & test):>4}  (must be 0)")
ov = train & test
if ov:
    print(f"  First 5 train/test overlap examples:")
    for p in list(ov)[:5]:
        print(f"    {p}")


# ----------------------------------------------------------------------
# CHECK 3: Verify single-view dedup count from evaluator output
# ----------------------------------------------------------------------
print("\n[CHECK 3] Single-view dedup sanity")
print("-" * 70)

N = 4
expected_unique = sum(c for c in hotel_counts.values() if c >= N)
results_path = SMOKE / "eval_results.json"

if not results_path.exists():
    print(f"  [SKIP] {results_path} does not exist")
else:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    print(f"  Expected unique single-view images after dedup (N={N}): {expected_unique}")
    print(f"  eval_results.json contents:")
    print(f"    {json.dumps(results, indent=4, default=str)}")
    print(f"  If 'n_test_images' or similar field appears above, it should equal {expected_unique}.")
    print(f"  If it equals {N*expected_unique} = {N}x{expected_unique}, dedup is BROKEN.")


# ----------------------------------------------------------------------
# CHECK 4: Per-image single-view accuracy on FULL test set (no combos)
# ----------------------------------------------------------------------
print("\n[CHECK 4] Per-image single-view accuracy -- full test set, no combos")
print("-" * 70)

ckpt_path = SMOKE / "checkpoint_best.pt"
if not ckpt_path.exists():
    for alt in ["best.pth", "checkpoint_last.pt", "last.pth"]:
        if (SMOKE / alt).exists():
            ckpt_path = SMOKE / alt
            break

if not ckpt_path.exists():
    print(f"  [SKIP] No checkpoint found in {SMOKE}")
else:
    print(f"  Loading {ckpt_path} ...")

    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(REPO))
    from oamv_hfmd.model import MultiImageHybrid
    from oamv_hfmd.data  import HotelsDataset

    n_classes = len({p.split("/")[0] for p in train})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MultiImageHybrid("vit_small_r26_s32_224",
                              num_classes=n_classes, n=1).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()

    ds = HotelsDataset(str(DATA_DIR), split="test", n=1, train=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=True)

    correct = total = 0
    with torch.no_grad():
        for images, targets, _ in loader:
            images  = images.to(device)
            targets = targets.to(device).flatten()
            out    = model(images)
            logits = out["single"]["logits"]
            preds  = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total   += targets.numel()

    acc = correct / total if total > 0 else 0.0
    print(f"  Per-image test images evaluated: {total}")
    print(f"  Per-image single-view top-1:     {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Smoke evaluator reported:        0.4953  (49.53%)")
    print(f"  Delta:                           {(acc - 0.4953)*100:+.2f} pp")
    print()
    print(f"  If delta is within +/- 2 pp, the reported single-view number is")
    print(f"  real and the dedup is working correctly.")
    print(f"  If per-image is meaningfully lower, the evaluator's single-view")
    print(f"  dedup is broken and the reported number is inflated.")


print("\n" + "=" * 70)
print("  Sanity checks complete.")
print("=" * 70)
