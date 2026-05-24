# OAMV-HFMD — Overlap-Aware Multi-View Hybrid Fusion with Mutual Distillation

**Status**: project setup phase. See `PLAN.md` for full design.

This repository implements an extension to MV-HFMD (Black & Souvenir, WACV 2024)
that uses a frozen foundation-model encoder (DINOv2-S) as a free, pose-free
overlap oracle. Two mechanisms consume the oracle's `N×N` cosine-similarity
matrix `S`:

1. **Content-conditional view embedding** — `MLP(S[i, :])` replaces MV-HFMD's
   static `E_img` slot-indexed embedding.
2. **Similarity-weighted symmetric distillation** — per-pair softmax-of-`S`
   weights replace MV-HFMD's uniform `L_md` aggregation.

Target venue: WACV 2027 Main Track Round 2 (deadline 2026-08-28).

## Quickstart

```bash
# 1. Environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Build the data layout and splits (one-time)
python scripts/reorganize_data.py    # hardlinks Hotels-8k into per-hotel layout
python scripts/build_splits.py       # fallback train/val/test .npy files

# 3. Run the MV-HFMD reproduction (E1)
python scripts/01_train_mvhfmd_baseline.py --config configs/E1_mvhfmd_baseline.yaml --seed 42
```

See `PLAN.md` §10 for the full implementation roadmap and §6 for the experiment
list (E1–E11) tagged by priority (MUST / STRETCH-main / STRETCH-appendix).

## Directory map

```
.
├── PLAN.md               # The single source of truth — read this first
├── oamv_hfmd/            # Importable Python package (model, data, losses, trainer)
├── configs/              # Per-experiment YAML hyperparameter files
├── scripts/              # Numbered driver scripts (one per experiment)
├── results/              # Per-experiment outputs — checkpoints, logs, eval JSON
├── paper/                # LaTeX sources + figures + tables
├── tests/                # Unit tests for non-trivial logic
├── docs/                 # decisions.md, upstream_status.md, reproduction_log.md
├── notes/                # Ad-hoc analysis (do not commit to results)
└── requirements.txt      # Pinned dependencies (timm==0.9.10 is non-negotiable)
```

## Critical environment pin

`timm==0.9.10` is required. Later timm versions (1.0.x) lack the
`block.attn.fused_attn` attribute that MV-HFMD's `model.py` mutates at init —
the model will silently fail to instantiate. See `PLAN.md` §4.1.

## Upstream blockers

- Open GitHub issue on `vidarlab/multi-viewhybrid` for the broken dataset link.
- Same on `vidarlab/multi-view-hybrid` (HMDMV, same lab) for the .npy splits.
- The trainer's `md_loss` implementation and the entry-point hyperparameters
  are not yet in our hands. Tracked in `docs/upstream_status.md`.

If upstream silence by 2026-06-05, we commit to the synthetic fallback split
(produced by `scripts/build_splits.py`) and the `md_loss` reimplementation
from the paper §3.2.
