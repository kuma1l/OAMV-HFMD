# scripts/verify_zero.py
# Three checks to rule out a masking/index bug behind the 0.0 result on the
# travel_website diagnostic. Mirrors scripts/diagnose_travel_website.py for
# model load + single-view forward + eval transform.

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path("D:/Research-WS/PIVOT/OAMV-HFMD")
sys.path.insert(0, str(ROOT))

CKPT        = ROOT / "results/E1_mvhfmd_baseline/seed42/checkpoint_best.pt"
CLASSES_TXT = ROOT / "hotel_classes.txt"
TW_NPY      = ROOT / "mvhfmd_data/test_travel_website.npy"
DATA_ROOT   = ROOT / "mvhfmd_data"

# --- Load class mapping ------------------------------------------------------
classes    = [l.strip() for l in CLASSES_TXT.read_text().splitlines() if l.strip()]
cls_to_idx = {c: i for i, c in enumerate(classes)}
assert len(classes) == 7754, f"expected 7754 classes, got {len(classes)}"

tw_paths   = np.load(TW_NPY)
tw_hotels  = sorted({str(p).split("/")[0] for p in tw_paths})
print(f"tw hotels in split: {len(tw_hotels)} | tw images: {len(tw_paths)}")

# === CHECK 1: are tw hotels even in the class index space? ===================
missing = [h for h in tw_hotels if h not in cls_to_idx]
print(f"CHECK 1 — tw hotels missing from class set: {len(missing)}")
if missing:
    print(f"  examples: {missing[:5]}")
    print("  >>> BUG: hotels not in head.weight index space. Fix this first.")
    sys.exit(1)
tw_idx_set = {cls_to_idx[h] for h in tw_hotels}

# --- Load model exactly as diagnose_travel_website.py does -------------------
# diagnose_travel_website.py uses MultiImageHybrid(arch=..., num_classes=...,
# n=1, pretrained_weights=False) for the single-view pass, then loads the
# checkpoint state-dict with the img_embed_matrix filter (matching scripts/01).
from oamv_hfmd.model import MultiImageHybrid
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultiImageHybrid(
    arch="vit_small_r26_s32_224",
    num_classes=len(classes),
    n=1,
    pretrained_weights=False,
).to(device)

ckpt = torch.load(CKPT, map_location=device, weights_only=False)
sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
# Same filter as scripts/01 line 270 and diagnose_travel_website.py:
sd = {k: v for k, v in sd.items() if k != "img_embed_matrix"}
missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
print(f"  load: missing={len(missing_keys)} unexpected={len(unexpected_keys)} "
      f"(img_embed_matrix in missing is expected)")
model.eval()

# === CHECK 2: head.weight norms — are tw classes silently zeroed/masked? =====
# In this codebase the final classifier is `self.model.head` (timm head),
# so the parameter name is "model.head.weight".
W = None
W_name = None
for name, p in model.named_parameters():
    if name.endswith("head.weight"):
        W = p.detach().cpu()
        W_name = name
        break
assert W is not None and W.shape[0] == 7754, (
    f"could not find a (7754, d) head.weight (found {W_name}={None if W is None else tuple(W.shape)})"
)
print(f"  using {W_name} shape={tuple(W.shape)}")
tw_indices  = sorted(tw_idx_set)
oth_indices = [i for i in range(7754) if i not in tw_idx_set]
tw_norm  = W[tw_indices].norm(dim=1)
oth_norm = W[oth_indices].norm(dim=1)
print(f"CHECK 2 — head.weight row-norm:")
print(f"  tw classes ({len(tw_indices)}):  mean={tw_norm.mean():.3f}  min={tw_norm.min():.3f}  max={tw_norm.max():.3f}")
print(f"  other     ({len(oth_indices)}): mean={oth_norm.mean():.3f}  min={oth_norm.min():.3f}  max={oth_norm.max():.3f}")
if tw_norm.mean() < 0.5 * oth_norm.mean():
    print("  >>> BUG SUSPECTED: tw class weights look systematically suppressed.")

# === CHECK 3: rank of true class on 100 random tw images =====================
# diagnose_travel_website.py uses oamv_hfmd.data.build_eval_transform for the
# single-view eval (Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize).
# The model's forward expects shape (B, N, C, H, W); for n=1 that's
# (1, 1, 3, 224, 224). The single-view logits are at out["single"]["logits"],
# shape (B*N, K) = (1, 7754).
from oamv_hfmd.data import build_eval_transform
tfm = build_eval_transform()

import random; random.seed(0)
sample = random.sample(list(tw_paths), 100)
ranks, top10_hits = [], 0
with torch.no_grad():
    for p in sample:
        hotel    = str(p).split("/")[0]
        true_idx = cls_to_idx[hotel]
        img      = tfm(Image.open(DATA_ROOT / p).convert("RGB"))         # (3,224,224)
        img      = img.unsqueeze(0).unsqueeze(0).to(device)              # (1,1,3,224,224)
        logits   = model(img)["single"]["logits"][0].cpu()               # (7754,)
        rank     = int((logits > logits[true_idx]).sum().item()) + 1     # 1 = best
        ranks.append(rank)
        top10    = set(torch.topk(logits, 10).indices.tolist())
        if top10 & tw_idx_set:
            top10_hits += 1

ranks = np.array(ranks)
print(f"CHECK 3 — rank of true class on 100 tw images (1 = best, 7754 = worst):")
print(f"  median={np.median(ranks):.0f}  p25={np.percentile(ranks,25):.0f}  p75={np.percentile(ranks,75):.0f}  min={ranks.min()}  max={ranks.max()}")
print(f"  top-10 contains ANY of the 354 tw hotels: {top10_hits}/100 images")

# === Verdict =================================================================
print()
med = np.median(ranks)
if med < 500:
    print("VERDICT: rank-of-true is suspiciously LOW — signal exists but never #1. BUG LIKELY.")
elif med > 6000:
    print("VERDICT: rank skewed HIGH — model actively steered AWAY from tw classes. Anti-knowledge confirmed; 0.0 is real.")
elif 3000 <= med <= 5000:
    print(f"VERDICT: rank ~ uniform random (~{7754//2}). Model has no signal on tw images. 0.0 is real, shift is severe.")
else:
    print(f"VERDICT: rank median={med:.0f}. Inconclusive — inspect ranks histogram before pivoting.")
