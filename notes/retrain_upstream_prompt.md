# Claude Code prompt — MV-HFMD reproduction on the upstream (author) split

Paste everything inside the fenced block below into Claude Code, running from the
repo root `D:\Research-WS\PIVOT\OAMV-HFMD`. It is self-contained (no chat history
needed).

---

```text
You are working in the OAMV-HFMD repo (D:\Research-WS\PIVOT\OAMV-HFMD). Read
docs/CURRENT_STATUS.md §1-§6 and docs/PLAN.md §3.3, §6 E1, §18 row 1 before acting.
Obey CLAUDE.md: surgical changes only, think before coding, simplicity first, and do
NOT modify docs/PLAN.md or docs/CURRENT_STATUS.md.

## Context (what changed)
The corresponding author (Sam Black) delivered the canonical Hotels-8k split. It is
staged at hotel_8k_images/ :
  - train.npy  91,513 images / 7,774 hotels   (paths: train/<chain>/<subchain>/<hotel>/<img>.jpg)
  - val.npy     8,000 images / 2,000 hotels
  - test.npy   12,026 images / 3,298 hotels    (paths: test/<chain>/filler/<hotel>/<img>.jpg)
    (test_old.npy is byte-identical to test.npy — ignore it)
This is the paper's real data AND label space. It uses Hotels-50K-native hotel IDs,
which are a DIFFERENT id space than our old Stream 3 split (built from Kaggle FGVC8).
Only 197 of the 3,298 test hotels exist in the old checkpoint's class space, so the
existing results/E1_mvhfmd_baseline checkpoint CANNOT be evaluated on this test set.
We must retrain on the upstream split. Per PLAN §3.3 the upstream split is now primary;
Stream 3 drops to appendix.

The data loader is already compatible: oamv_hfmd/data.py:_extract_hotel_id uses the
second-to-last path component, which is the hotel_id in both the nested train layout
and the test "filler" layout. No change to data.py is needed.

Already prepared (verify they exist, do not recreate):
  - configs/E1_mvhfmd_upstream.yaml   (data_dir: hotel_8k_images, out_dir: results/E1_mvhfmd_upstream)
  - hotel_classes_upstream.txt        (7,774 sorted hotel IDs; equals what the loader derives)

## Non-negotiables (from CURRENT_STATUS §4 / PLAN §5.1) — do not deviate
  - timm==0.9.10 for training. Backbone arch "vit_small_r26_s32_224" (no augreg tag).
  - SGD momentum 0.9, lr 0.01, OneCycleLR, weight_decay 5e-4, 50 epochs, NO label
    smoothing, grad_clip 80.0, lambda_md 0.1, md_temp 4.0. All already in configs/base.yaml.
  - MD loss applies tau^2 * lambda Hinton scaling internally — never double-multiply.
  - Closed-set classification; head dim = ds_train.num_classes (the driver derives 7,774
    from train.npy automatically — do not hardcode).
  - No ColorJitter in train transforms (enforced by tests/test_data.py).
  - Batch size 16 on RTX 3050 (OOMs higher); 64 on A100 80GB.
  - Cloud VM disk >= 60 GB.

## Task

### Step 0 — surgical provenance fix (one line + one config line)
In scripts/01_train_mvhfmd_baseline.py the config snapshot hardcodes
  "split_source": "fallback_stream3_md5_seed42",
Change ONLY that value to read from config with the old string as fallback:
  "split_source": cfg.get("split_source", "fallback_stream3_md5_seed42"),
Then add this top-level key to configs/E1_mvhfmd_upstream.yaml:
  split_source: "upstream_author_split_2026-05-29"
This keeps the old E1 config's behavior identical (it has no split_source key) while
recording the correct provenance for the upstream run. Make no other edits.

### Step 1 — local wiring verification (cheap, no training)
Run, from repo root:
  python scripts/01_train_mvhfmd_baseline.py --config configs/E1_mvhfmd_upstream.yaml --seed 42 --n-views 4 --num-workers 2 --dry-run
Confirm in the log: a line "train=... val=... test=... num_classes=7774". The
num_classes MUST be 7774. If it is not, STOP and report — the label space is wrong.
Do not proceed past a failed dry-run.

### Step 2 — cloud 50-epoch reproduction (A100)
This is the real C1 and runs on a Vast.ai A100 (the RTX 3050 is too slow for the full
sweep). Use the existing recipe in scripts/bootstrap_vast.sh and CURRENT_STATUS §11
(fresh instance -> git clone -> bootstrap_vast.sh). Ensure hotel_8k_images/ is present
on the VM (re-download / rsync cloud-to-cloud per CURRENT_STATUS §12 note 5 — do NOT
upload from the home machine).

Run the full E1 sweep: 3 seeds {42,1337,2024} x {single, N=2, N=4}, batch 64, 50 epochs,
in a tmux session. One invocation per (n_views, seed), e.g.:
  for S in 42 1337 2024; do for N in single 2 4; do
    python scripts/01_train_mvhfmd_baseline.py --config configs/E1_mvhfmd_upstream.yaml \
      --seed $S --n-views $N --batch-size 64 --num-workers 8
  done; done
Outputs land in results/E1_mvhfmd_upstream/{single,N2,N4}_seed{42,1337,2024}/.
Budget guide: ~18 A100-hours total (~$27 at $1.50/h) per PLAN §14.1.

### Step 3 — aggregate and check the C1 criterion
Run scripts/aggregate_results.py to populate results/tables/ (add an
mvhfmd_baseline_upstream.csv if the aggregator is split-keyed). Then evaluate the C1
pass criterion (PLAN §18 row 1):
  - mean test top-1 over 3 seeds, per view count, within +/-2 pp of paper:
    single 46.3 %, N=2 65.1 %, N=4 ~70 %.
  - fusion delta (N=2 mean - single mean) within <= 1 pp of paper's ~18.8 pp,
    95% CI lower bound on the fusion delta > 0.
Report PASS or FAIL per the criterion. If a mean is >2 pp off but the fusion delta
holds, that is acceptable (ship with a split caveat). If the fusion delta degrades
by >2 pp, flag Risk #2 (PLAN §13) and stop for a human decision.

## Downstream note (do not act on yet)
The test set has 3,298 hotels at N=2 and 2,217 at N=4 (solid), but only 103 at N=6 and
66 at N=8. The later N-scaling experiment (E5 / C4 / Figure 1) will rest on a thin pool
at N>=6 — note it when you get there; it does not affect this E1 reproduction.

## Reporting
After Step 1: report the dry-run num_classes and dataset sizes, then pause for go-ahead
before spending cloud compute. After Step 3: report the Table 1 numbers and the
PASS/FAIL verdict. Do not edit PLAN.md / CURRENT_STATUS.md — leave a consolidated update
for the human.
```

---

## Notes for Kumail (not part of the prompt)

- I baked in the `split_source` provenance fix as Step 0 since you were moving forward;
  it's the one-line change I flagged. If you'd rather leave the driver untouched, delete
  Step 0 from the prompt and the `config.json` will just carry the stale label.
- The prompt pauses after the local dry-run so you approve before the ~$27 cloud spend.
- It deliberately does not run the full 1-epoch local smoke (~7h on the 3050); the
  dry-run covers the new wiring and the A100's first epoch is the real-data check.
