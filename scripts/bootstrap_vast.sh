#!/usr/bin/env bash
# bootstrap_vast.sh — set up a fresh Vast.ai instance for E1 on the UPSTREAM split.
#
# This is the author-provided canonical Hotels-8k split (delivered by Sam Black
# 2026-05-29), staged as a single tarball on S3. It is the paper's real data +
# label space and is now the PRIMARY split (PLAN.md §3.3). The old Kaggle/Stream-3
# flow is preserved at scripts/bootstrap_vast_kaggle.sh for the appendix repro.
#
# Usage:
#     # From the repo root after `git clone`:
#     bash scripts/bootstrap_vast.sh
#
# No Kaggle credentials needed — the dataset is pulled from a public S3 object.
# Override the source if you re-host it:  DATA_URL=https://... bash scripts/bootstrap_vast.sh
#
# Idempotent — safe to re-run; stages skip if their output already exists.
# Dataset is ~1.6 GB; total bootstrap walltime on a fresh instance: ~5-10 min.
# Any 24 GB+ GPU (RTX 3090/4090, A5000/A6000, A100) runs E1 at batch 64.

set -euo pipefail

# --- Config ----------------------------------------------------------------
DATA_URL="${DATA_URL:-https://mvfmd.s3.us-east-2.amazonaws.com/hotel_8k_images.tar}"
WORKSPACE="${WORKSPACE:-/workspace}"
TAR_PATH="${TAR_PATH:-${WORKSPACE}/hotel_8k_images.tar}"

# Repo dir = parent of this script's dir. The training config's data_dir is the
# repo-relative "hotel_8k_images", so the dataset must extract to REPO_DIR.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_DIR}/hotel_8k_images"

# Expected counts for the upstream split (from the delivered .npy files).
EXPECTED_TRAIN=91513
EXPECTED_VAL=8000
EXPECTED_TEST=12026
EXPECTED_CLASSES=7774

# --- Helpers ---------------------------------------------------------------
banner() { printf '\n%s\n  %s\n%s\n' "============================================================" "$1" "============================================================"; }
die()    { printf 'ERROR: %s\n' "$1" >&2; exit 1; }

# --- Stage 1: env sanity ---------------------------------------------------
banner "1/6  Env sanity"
python --version
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — is this a GPU instance?"
nvidia-smi | head -20
command -v curl >/dev/null || { echo "Installing curl..."; apt-get update -qq && apt-get install -y -qq curl; }
command -v tar  >/dev/null || die "tar not found — unexpected on an Ubuntu image."

# --- Stage 2: pip install --------------------------------------------------
banner "2/6  Installing Python dependencies"
pip install --quiet --upgrade pip
pip install --quiet -r "${REPO_DIR}/requirements.txt"
echo "OK"

# --- Stage 3: verify critical pins -----------------------------------------
banner "3/6  Verifying critical pins"
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

# --- Stage 4: download dataset tarball from S3 -----------------------------
banner "4/6  Downloading dataset (~1.6 GB) from ${DATA_URL}"
if [[ -f "${DATA_DIR}/train.npy" ]]; then
  echo "  ${DATA_DIR}/train.npy already present — dataset extracted, skipping download"
elif [[ -f "${TAR_PATH}" ]]; then
  echo "  tarball already at ${TAR_PATH}, skipping download"
else
  free_kb=$(df -P "${WORKSPACE}" | awk 'NR==2 {print $4}')
  free_gb=$((free_kb / 1024 / 1024))
  [[ "${free_gb}" -lt 5 ]] && die "Only ${free_gb} GB free in ${WORKSPACE}; need ~5 GB for tar + extract."
  # -f: fail loudly on HTTP 4xx/5xx (this is the link verification gate).
  curl -fL --retry 3 --retry-delay 5 -o "${TAR_PATH}" "${DATA_URL}" \
    || die "Download failed (bad URL, 403/404, or network). Check DATA_URL and S3 object ACL."
  ls -lh "${TAR_PATH}"
fi

# --- Stage 5: extract ------------------------------------------------------
banner "5/6  Extracting to ${DATA_DIR}"
if [[ -f "${DATA_DIR}/train.npy" ]]; then
  echo "  already extracted, skipping"
else
  # Expect the tar to contain a top-level hotel_8k_images/ directory.
  tar xf "${TAR_PATH}" -C "${REPO_DIR}"
  [[ -f "${DATA_DIR}/train.npy" ]] || die "Extracted but ${DATA_DIR}/train.npy not found. The tarball layout may differ from the expected top-level hotel_8k_images/ folder — inspect with: tar tf ${TAR_PATH} | head"
  echo "  extraction OK, removing tar to reclaim space..."
  rm -f "${TAR_PATH}"
fi
ls "${DATA_DIR}" | head

# --- Stage 6: verify counts + label space ----------------------------------
banner "6/6  Verifying upstream split counts + label space"
python - <<PYEOF
import numpy as np, sys
d = "${DATA_DIR}"
got = {s: int(len(np.load(f"{d}/{s}.npy", allow_pickle=True))) for s in ("train","val","test")}
exp = {"train": ${EXPECTED_TRAIN}, "val": ${EXPECTED_VAL}, "test": ${EXPECTED_TEST}}
print("          got      expected")
ok = True
for s in ("train","val","test"):
    flag = "" if got[s]==exp[s] else "  <-- MISMATCH"
    if got[s]!=exp[s]: ok = False
    print(f"  {s:6s} {got[s]:>7d}  {exp[s]:>7d}{flag}")
tr = np.load(f"{d}/train.npy", allow_pickle=True)
nclass = len({p.split("/")[-2] for p in tr})
flag = "" if nclass==${EXPECTED_CLASSES} else "  <-- MISMATCH"
if nclass!=${EXPECTED_CLASSES}: ok = False
print(f"  classes {nclass:>5d}  {${EXPECTED_CLASSES}:>7d}{flag}")
if not ok:
    print("FAIL: dataset does not match the expected upstream split — do not train."); sys.exit(1)
print("OK — upstream split verified.")
PYEOF

# --- Done ------------------------------------------------------------------
banner "Bootstrap complete"
cat <<MSG
Data ready at:  ${DATA_DIR}
Code ready at:  ${REPO_DIR}

Next: time one epoch of E1 at batch 64 to validate the compute budget.

  cd ${REPO_DIR}
  python scripts/01_train_mvhfmd_baseline.py --config configs/E1_mvhfmd_upstream.yaml --seed 42 --n-views 4 --epochs 1 --batch-size 64 --out-dir results/cloud_smoke_upstream

If first epoch is sane, launch the full 50-epoch sweep (3 seeds x {single,2,4}) in tmux:

  for S in 42 1337 2024; do for N in single 2 4; do
    python scripts/01_train_mvhfmd_baseline.py --config configs/E1_mvhfmd_upstream.yaml --seed \$S --n-views \$N --batch-size 64 --num-workers 8
  done; done

Outputs land in results/E1_mvhfmd_upstream/{single,N2,N4}_seed{42,1337,2024}/.
MSG
