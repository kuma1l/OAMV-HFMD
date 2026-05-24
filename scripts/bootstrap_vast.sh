#!/usr/bin/env bash
# bootstrap_vast.sh — set up a fresh Vast.ai pytorch instance for E1.
#
# Usage:
#     # From the repo root after `git clone`:
#     bash scripts/bootstrap_vast.sh
#
# Prerequisites — at least one of:
#   (a) export KAGGLE_API_TOKEN=KGAT_...
#   (b) ~/.kaggle/access_token  containing the KGAT_... token
#   (c) ~/.kaggle/kaggle.json   (old format: {"username":"...","key":"..."})
#
# Idempotent — safe to re-run; stages skip if their output already exists.
# Total walltime on 1x A100 with fast Kaggle download: ~10-20 min.

set -euo pipefail

# --- Config ----------------------------------------------------------------
WORKSPACE="${WORKSPACE:-/workspace}"
RAW_DIR="${RAW_DIR:-${WORKSPACE}/raw_kaggle}"
DATA_DIR="${DATA_DIR:-${WORKSPACE}/mvhfmd_data}"
KAGGLE_COMP="${KAGGLE_COMP:-hotel-id-2021-fgvc8}"

# Expected split counts from PLAN.md §3.2 (md5-seeded; deterministic)
EXPECTED_TRAIN=74281
EXPECTED_VAL=10151
EXPECTED_TEST=13091

# Repo dir = parent of this script's dir
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- Helpers ---------------------------------------------------------------
banner() { printf '\n%s\n  %s\n%s\n' "============================================================" "$1" "============================================================"; }
die()    { printf 'ERROR: %s\n' "$1" >&2; exit 1; }

# --- Stage 1: env sanity ---------------------------------------------------
banner "1/8  Env sanity"
python --version
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — is this a GPU instance?"
nvidia-smi | head -20
command -v unzip >/dev/null || { echo "Installing unzip..."; apt-get install -y -qq unzip; }

# --- Stage 2: pip install --------------------------------------------------
banner "2/8  Installing Python dependencies"
pip install --quiet --upgrade pip
pip install --quiet -r "${REPO_DIR}/requirements.txt"
echo "OK"

# --- Stage 3: verify critical pins -----------------------------------------
banner "3/8  Verifying critical pins"
python - <<'PYEOF'
import timm, torch, sys
errs = []
if timm.__version__ != "0.9.10":
    errs.append(f"timm pinned wrong: {timm.__version__} (expected 0.9.10) — see PLAN.md §4.1")
if not torch.__version__.startswith("2."):
    errs.append(f"torch must be 2.x, got {torch.__version__}")
if errs:
    for e in errs: print("FAIL:", e)
    sys.exit(1)
print(f"  timm:  {timm.__version__}")
print(f"  torch: {torch.__version__}")
PYEOF

# --- Stage 4: kaggle creds check -------------------------------------------
banner "4/8  Verifying Kaggle credentials"
if [[ -z "${KAGGLE_API_TOKEN:-}" ]] && [[ ! -f "${HOME}/.kaggle/access_token" ]] && [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
  die "No Kaggle credentials found. See header of this script for setup options."
fi
kaggle competitions list --search hotel-id-2021 | head -3 \
  || die "Kaggle API call failed — check token validity, expiry, and rules-accepted status."

# --- Stage 5: download from Kaggle -----------------------------------------
banner "5/8  Downloading Hotels-8k from Kaggle (~24 GB)"
mkdir -p "${RAW_DIR}"
cd "${RAW_DIR}"
if [[ -f "${KAGGLE_COMP}.zip" ]]; then
  echo "  zip already present at ${RAW_DIR}/${KAGGLE_COMP}.zip, skipping download"
else
  kaggle competitions download -c "${KAGGLE_COMP}"
  ls -lh "${KAGGLE_COMP}.zip"
fi

# --- Stage 6: extract + free the zip immediately to save 24 GB -------------
banner "6/8  Extracting (and removing zip after success to reclaim 24 GB)"
if [[ -d "${RAW_DIR}/train_images" ]] && [[ -f "${RAW_DIR}/train.csv" ]]; then
  echo "  already extracted, skipping"
  [[ -f "${KAGGLE_COMP}.zip" ]] && { echo "  removing leftover zip..."; rm "${KAGGLE_COMP}.zip"; }
else
  # Pre-flight: need ~30 GB free for zip + extracted tree to coexist briefly.
  free_kb=$(df -P "${RAW_DIR}" | awk 'NR==2 {print $4}')
  free_gb=$((free_kb / 1024 / 1024))
  if [[ "${free_gb}" -lt 30 ]]; then
    die "Only ${free_gb} GB free in ${RAW_DIR}; need ~30 GB headroom to extract safely. Free space or resize instance."
  fi
  unzip -q "${KAGGLE_COMP}.zip"
  if [[ -d "${RAW_DIR}/train_images" ]] && [[ -f "${RAW_DIR}/train.csv" ]]; then
    echo "  extraction OK, removing zip to reclaim 24 GB..."
    rm "${KAGGLE_COMP}.zip"
  else
    die "Extraction completed but train_images/ or train.csv missing — not deleting zip. Investigate."
  fi
fi
ls "${RAW_DIR}" | head -10
df -h "${RAW_DIR}" | tail -1

# --- Stage 7: per-hotel reorganization + splits ----------------------------
banner "7/8  Reorganizing into per-hotel layout + building splits"
if [[ -d "${DATA_DIR}" ]] && [[ "$(find "${DATA_DIR}" -maxdepth 1 -type d | wc -l)" -gt 1000 ]]; then
  echo "  ${DATA_DIR} already populated, skipping reorganize"
else
  python "${REPO_DIR}/scripts/reorganize_data.py" \
      --csv      "${RAW_DIR}/train.csv" \
      --src-root "${RAW_DIR}/train_images" \
      --dst-root "${DATA_DIR}"
fi

python "${REPO_DIR}/scripts/build_splits.py" --data-root "${DATA_DIR}"

# --- Stage 8: verify split counts ------------------------------------------
banner "8/8  Verifying split counts vs PLAN.md §3.2"
python - <<PYEOF
import numpy as np, sys
data = "${DATA_DIR}"
got = {s: int(len(np.load(f"{data}/{s}.npy"))) for s in ("train", "val", "test")}
exp = {"train": ${EXPECTED_TRAIN}, "val": ${EXPECTED_VAL}, "test": ${EXPECTED_TEST}}
total_got = sum(got.values()); total_exp = sum(exp.values())
print(f"          got      expected  diff")
for s in ("train", "val", "test"):
    print(f"  {s:6s} {got[s]:>7d}  {exp[s]:>7d}  {got[s]-exp[s]:+d}")
# Strict equality is ideal but +/- 0.1% is acceptable (a few image files in
# the Kaggle archive may differ from the snapshot used to compute PLAN's numbers).
if abs(total_got - total_exp) / total_exp > 0.001:
    print(f"FAIL: total off by {total_got-total_exp} (>0.1%) -- investigate before E1.")
    sys.exit(1)
print("OK (within tolerance).")
PYEOF

# --- Done ------------------------------------------------------------------
banner "Bootstrap complete"
cat <<MSG
Data ready at:  ${DATA_DIR}
Code ready at:  ${REPO_DIR}

Next: time one epoch of E1 at A100 batch size 64 to validate compute budget.

  cd ${REPO_DIR}
  python scripts/01_train_mvhfmd_baseline.py \\
      --config configs/E1_mvhfmd_baseline.yaml \\
      --seed 42 --n-views 4 --epochs 1 --batch-size 64 \\
      --out-dir results/cloud_smoke

If first epoch takes ~10 min  -> PLAN budget honest, launch the full 50-epoch run.
If first epoch takes >>30 min -> stop, debug; do not scale up on a stale estimate.
MSG
