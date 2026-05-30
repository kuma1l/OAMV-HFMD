"""Travel-website Go/No-Go diagnostic.

Loads the 10-epoch MV-HFMD checkpoint (results/E1_mvhfmd_baseline/seed42/
checkpoint_best.pt, n=4) and evaluates it on the travel_website test split
(mvhfmd_data/test_travel_website.npy). Reports:

  - per-image single-view top-1 over the full test set
  - exhaustive N=4 mv_collection top-1 (micro and macro)
  - per-hotel accuracy deciles

Then prints a verdict (VIABLE / SATURATED / TOO_HARD / AMBIGUOUS) based on
mv_N4 micro. Writes results/diagnostic_travel_website/seed42/eval_results.json.

N=2 mv is intentionally NOT evaluated — only the N=4 checkpoint exists and
the model's img_embed_matrix shape is tied to N at construction. See task
spec.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from oamv_hfmd.data import HotelsDataset           # noqa: E402
from oamv_hfmd.eval import Evaluator               # noqa: E402
from oamv_hfmd.model import MultiImageHybrid       # noqa: E402
from oamv_hfmd.utils import load_yaml, worker_init_fn  # noqa: E402


CFG_PATH   = REPO / "configs" / "E1_mvhfmd_baseline.yaml"
BASE_PATH  = REPO / "configs" / "base.yaml"
CKPT_PATH  = REPO / "results" / "E1_mvhfmd_baseline" / "seed42" / "checkpoint_best.pt"
DATA_DIR   = REPO / "mvhfmd_data"
SPLIT_NAME = "test_travel_website"
OUT_DIR    = REPO / "results" / "diagnostic_travel_website" / "seed42"
N_VIEWS    = 4
SEED       = 42
# Per-hotel cap applied by scripts/build_travel_website_split.py before the .npy
# was written. We just reflect those values in the eval_results.json metadata.
PER_HOTEL_CAP_K    = 10
PER_HOTEL_CAP_SEED = 42
SANITY_MAX_COMBOS  = 150_000


def _load_cfg() -> dict:
    base = load_yaml(BASE_PATH)
    over = load_yaml(CFG_PATH)
    # E1 inherits from base — base.yaml has the model/train fields we need
    cfg = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


@torch.no_grad()
def _single_view_per_image_top1(model_n1, loader, device, amp):
    correct = 0
    total = 0
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).flatten()
        with torch.cuda.amp.autocast(enabled=amp and torch.cuda.is_available()):
            out = model_n1(images)
        preds = out["single"]["logits"].argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += int(targets.numel())
    return (correct / total if total > 0 else 0.0), total


def main():
    if not CKPT_PATH.exists():
        sys.exit(f"missing checkpoint: {CKPT_PATH}")
    if not (DATA_DIR / f"{SPLIT_NAME}.npy").exists():
        sys.exit(f"missing split file: {DATA_DIR / (SPLIT_NAME + '.npy')}. "
                 f"Run scripts/build_travel_website_split.py first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg()
    arch = cfg["model"]["arch"]
    amp = bool(cfg["train"]["amp"])
    # Original config val_batch_size=128 caused 97% VRAM and a system crash on
    # this RTX 3050 8 GB (batch=128 x N=4 = 512 images per forward through
    # ViT-S). Drop to 32 collections per batch (= 128 images per forward) for
    # the diagnostic — comfortable headroom, ~4x more iterations but same total
    # work.
    val_bs = 32
    # Windows + spawn workers can hang on this repo; use 0 to be safe.
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} arch={arch} amp={amp} val_batch_size={val_bs}")

    # Use the training-class set so model indices match what was learned.
    classes = [ln.strip() for ln in open(REPO / "hotel_classes.txt") if ln.strip()]
    assert len(classes) == 7754, f"hotel_classes.txt has {len(classes)} lines, expected 7754"

    # --- N=4 exhaustive mv eval --------------------------------------------------
    print(f"building dataset split={SPLIT_NAME} n={N_VIEWS} (exhaustive)")
    ds_n4 = HotelsDataset(str(DATA_DIR), split=SPLIT_NAME, n=N_VIEWS, train=False, classes=classes)
    n_hotels_eligible = len({t for t in ds_n4.targets2indices.keys()
                             if len(ds_n4.targets2indices[t]) >= N_VIEWS})
    total_n4_combos = len(ds_n4)
    n_hotels_capped = sum(1 for t in ds_n4.targets2indices.values() if len(t) >= PER_HOTEL_CAP_K)
    print(f"  combos enumerated: {total_n4_combos}  hotels with >={N_VIEWS} imgs: {n_hotels_eligible}")
    print(f"  per-hotel cap K={PER_HOTEL_CAP_K} (seed={PER_HOTEL_CAP_SEED})  hotels at cap: {n_hotels_capped}")
    if total_n4_combos > SANITY_MAX_COMBOS:
        sys.exit(f"total_n4_combos={total_n4_combos} > {SANITY_MAX_COMBOS}; "
                 f"refusing to launch eval. Re-check the cap.")

    loader_n4 = DataLoader(
        ds_n4, batch_size=val_bs, shuffle=False, num_workers=num_workers,
        drop_last=False, pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
    )

    print(f"building model arch={arch} num_classes={len(classes)} n={N_VIEWS}")
    model = MultiImageHybrid(arch=arch, num_classes=len(classes), n=N_VIEWS,
                             pretrained_weights=False).to(device)

    print(f"loading checkpoint {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  missing[:5]={missing[:5]}")
    if unexpected:
        print(f"  unexpected[:5]={unexpected[:5]}")
    model.eval()

    # Use Evaluator.extract so we can compute macro + deciles ourselves.
    evaluator = Evaluator(model=model, n=N_VIEWS, device=device)
    t0 = time.time()
    extracted = evaluator.extract(loader_n4)
    elapsed_mv = time.time() - t0
    print(f"  extract done in {elapsed_mv:.1f}s")

    mv_logits = extracted["mv_collection"]["logits"]   # (M, K) torch tensor
    mv_classes = extracted["mv_collection"]["classes"] # (M,) ndarray
    preds = torch.argmax(mv_logits, dim=1).cpu().numpy()
    gt = np.asarray(mv_classes).reshape(-1)
    correct_mask = (preds == gt)

    mv_micro = float(correct_mask.mean()) if correct_mask.size > 0 else 0.0

    # Macro: mean of per-hotel accuracies (each hotel weighted equally).
    per_hotel_acc: dict[int, float] = {}
    for h in np.unique(gt):
        mask = (gt == h)
        per_hotel_acc[int(h)] = float(correct_mask[mask].mean())
    mv_macro = float(np.mean(list(per_hotel_acc.values()))) if per_hotel_acc else 0.0
    n_hotels_evaluated = len(per_hotel_acc)
    n_combos = int(correct_mask.size)

    # Per-hotel-accuracy deciles: 10 bins [0,0.1), [0.1,0.2), ..., [0.9,1.0].
    acc_values = np.array(list(per_hotel_acc.values()), dtype=np.float64)
    edges = np.linspace(0.0, 1.0, 11)
    bin_idx = np.minimum(np.digitize(acc_values, edges) - 1, 9)
    deciles_counts = [int((bin_idx == i).sum()) for i in range(10)]

    # --- per-image single-view eval ---------------------------------------------
    print("building per-image single-view dataset n=1")
    ds_n1 = HotelsDataset(str(DATA_DIR), split=SPLIT_NAME, n=1, train=False, classes=classes)
    loader_n1 = DataLoader(
        ds_n1, batch_size=val_bs, shuffle=False, num_workers=num_workers,
        drop_last=False, pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
    )

    # Mirror scripts/01_train_mvhfmd_baseline.py:270 — drop img_embed_matrix when
    # loading into the n=1 model.
    model_n1 = MultiImageHybrid(arch=arch, num_classes=len(classes), n=1,
                                pretrained_weights=False).to(device)
    sd_n1 = {k: v for k, v in sd.items() if k != "img_embed_matrix"}
    miss1, unex1 = model_n1.load_state_dict(sd_n1, strict=False)
    print(f"  per-image model load: missing={len(miss1)} unexpected={len(unex1)} "
          f"(img_embed_matrix shape mismatch is expected)")
    model_n1.eval()

    t0 = time.time()
    sv_top1, sv_total = _single_view_per_image_top1(model_n1, loader_n1, device, amp)
    elapsed_sv = time.time() - t0
    print(f"single-view per-image top1={sv_top1:.6f} over {sv_total} images "
          f"({elapsed_sv:.1f}s)")

    # --- assemble results --------------------------------------------------------
    n_images_total = int(len(ds_n1))
    results = {
        "n_hotels": n_hotels_evaluated,
        "n_images": n_images_total,
        "single_view_top1": float(sv_top1),
        "mv_N4_top1_micro": float(mv_micro),
        "mv_N4_top1_macro": float(mv_macro),
        "mv_N4_n_hotels":   int(n_hotels_evaluated),
        "mv_N4_n_samples":  int(n_combos),
        "per_hotel_acc_deciles": deciles_counts,
        "per_hotel_cap_k":      int(PER_HOTEL_CAP_K),
        "per_hotel_cap_seed":   int(PER_HOTEL_CAP_SEED),
        "n_hotels_capped":      int(n_hotels_capped),
        "total_n4_combos":      int(total_n4_combos),
        "_meta": {
            "split":      SPLIT_NAME,
            "checkpoint": str(CKPT_PATH.relative_to(REPO)),
            "arch":       arch,
            "num_classes": len(classes),
            "n_views":    N_VIEWS,
            "seed":       SEED,
            "device":     device,
            "amp":        amp,
            "elapsed_mv_extract_seconds": float(elapsed_mv),
            "elapsed_single_view_seconds": float(elapsed_sv),
            "per_hotel_acc_decile_edges": [round(float(e), 2) for e in edges.tolist()],
            "n_hotels_with_4plus_views_in_split": int(n_hotels_eligible),
        },
    }
    out_path = OUT_DIR / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")

    # --- verdict -----------------------------------------------------------------
    m = float(mv_micro)
    if m > 0.85:
        verdict = "SATURATED"
    elif m < 0.30:
        verdict = "TOO_HARD"
    elif 0.40 <= m <= 0.75:
        verdict = "VIABLE"
    else:
        verdict = "AMBIGUOUS"

    print()
    print("=" * 60)
    print(f"VERDICT: {verdict}")
    print(f"  mv_N4_top1_micro = {mv_micro:.6f}")
    print(f"  mv_N4_top1_macro = {mv_macro:.6f}")
    print(f"  mv_N4_n_hotels   = {n_hotels_evaluated}")
    print(f"  mv_N4_n_samples  = {n_combos}")
    print(f"  single_view_top1 = {sv_top1:.6f}  (n={sv_total})")
    print(f"  n_hotels         = {n_hotels_evaluated}")
    print(f"  n_images         = {n_images_total}")
    print(f"  per_hotel_acc_deciles (10 bins [0,0.1)..[0.9,1.0]):")
    print(f"    {deciles_counts}")
    print("=" * 60)


if __name__ == "__main__":
    main()
