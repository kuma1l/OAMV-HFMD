#!/usr/bin/env bash
# bootstrap_vast.sh — set up a fresh Vast.ai instance for E1/E2 on the UPSTREAM split.
#
# Canonical author-provided Hotels-8k split (delivered by Sam Black 2026-05-29),
# staged as a single tarball on S3. Primary split (PLAN.md §3.3). The old
# Kaggle/Stream-3 flow is preserved at scripts/bootstrap_vast_kaggle.sh.
#
# Usage (from repo root after `git clone`):
#     DATA_URL='<fresh-presigned-s3-url>' bash scripts/bootstrap_vast.sh
#
# The S3 object is private; pass a presigned URL via DATA_URL (console -> object ->
# Share with a presigned URL). No Kaggle credentials needed.
#
# Idempotent + robust: validates the tarball, auto re-downloads a bad/partial file,
# locates train.npy wherever it lands, and pre-pulls the DINOv2 oracle for OAMV.
# Dataset ~1.6 GB; bootstrap walltime ~5-10 min. Any 24 GB+ GPU runs E1 at batch 40-64.

set -euo pipefail

# --- Config ----------------------------------------------------------------
DATA_URL="${DATA_URL:-https://mvfmd.s3.us-east-2.amazonaws.com/hotel_8k_images.tar}"
WORKSPACE="${WORKSPACE:-/workspace}"
TAR_PATH="${TAR_PATH:-${WORKSPACE}/hotel_8k_images.tar}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_DIR}/hotel_8k_images"

EXPECTED_TRAIN=91513
EXPECTED_VAL=8000
EXPECTED_TEST=12026
EXPECTED_CLASSES=7774

# --- Helpers ---------------------------------------------------------------
banner()    { printf '\n%s\n  %s\n%s\n' "============================================================" "$1" "============================================================"; }
die()       { printf 'ERROR: %s\n' "$1" >&2; exit 1; }
valid_tar() { tar -tf "$1" >/dev/null 2>&1; }   # true iff $1 is a readable tar

# --- Stage 1: env sanity ---------------------------------------------------
banner "1/7  Env sanity"
python --version
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — is this a GPU instance?"
nvidia-smi | head -20
command -v curl >/dev/null || { echo "Installing curl..."; apt-get update -qq && apt-get install -y -qq curl; }
command -v tar  >/dev/null || die "tar not found — unexpected on an Ubuntu image."

# --- Stage 2: pip install --------------------------------------------------
banner "2/7  Installing Python dependencies"
pip install --quiet --upgrade pip
pip install --quiet -r "${REPO_DIR}/requirements.txt"
echo "OK"

# --- Stage 3: verify critical pins -----------------------------------------
banner "3/7  Verifying critical pins"
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

# --- Stage 4: download dataset tarball (robust) ----------------------------
banner "4/7  Downloading dataset (~1.6 GB)"
if [[ -f "${DATA_DIR}/train.npy" ]]; then
  echo "  ${DATA_DIR}/train.npy already present — skipping download"
elif [[ -f "${TAR_PATH}" ]] && valid_tar "${TAR_PATH}"; then
  echo "  valid tarball already at ${TAR_PATH}, skipping download"
else
  if [[ -f "${TAR_PATH}" ]]; then
    echo "  existing ${TAR_PATH} is partial/corrupt — removing and re-downloading"
    rm -f "${TAR_PATH}"
  fi
  free_kb=$(df -P "${WORKSPACE}" | awk 'NR==2 {print $4}')
  free_gb=$((free_kb / 1024 / 1024))
  [[ "${free_gb}" -lt 5 ]] && die "Only ${free_gb} GB free in ${WORKSPACE}; need ~5 GB for tar + extract."
  echo "  source: ${DATA_URL%%\?*}${DATA_URL:+ (presigned)}"
  # -f fails on HTTP 4xx/5xx (link/expiry gate); retries cover transient network.
  curl -fL --connect-timeout 30 --retry 5 --retry-delay 5 --retry-connrefused \
       -o "${TAR_PATH}" "${DATA_URL}" \
    || die "Download failed (bad URL, 403/404, expired presign, or network). Regenerate the presigned URL and pass it as DATA_URL."
  # Integrity gate: a truncated download or an S3 XML error body is NOT a valid tar.
  valid_tar "${TAR_PATH}" || { rm -f "${TAR_PATH}"; die "Downloaded file is not a valid tar (truncated or error body). Check DATA_URL / expiry and retry."; }
  ls -lh "${TAR_PATH}"
fi

# --- Stage 5: extract (layout-robust) --------------------------------------
banner "5/7  Extracting to ${DATA_DIR}"
if [[ -f "${DATA_DIR}/train.npy" ]]; then
  echo "  already extracted, skipping"
else
  tar xf "${TAR_PATH}" -C "${REPO_DIR}"
  if [[ ! -f "${DATA_DIR}/train.npy" ]]; then
    # Tarball didn't yield the expected top-level hotel_8k_images/ — locate train.npy.
    found="$(find "${REPO_DIR}" -maxdepth 4 -name train.npy 2>/dev/null | head -1)"
    [[ -z "${found}" ]] && die "train.npy not found after extract. Inspect: tar tf ${TAR_PATH} | head"
    src_dir="$(cd "$(dirname "${found}")" && pwd)"
    if [[ "${src_dir}" != "${DATA_DIR}" ]]; then
      echo "  tar layout differs; linking ${DATA_DIR} -> ${src_dir}"
      ln -sfn "${src_dir}" "${DATA_DIR}"
    fi
  fi
  echo "  extraction OK, removing tar to reclaim space..."
  rm -f "${TAR_PATH}"
fi
ls "${DATA_DIR}" | head

# --- Stage 6: verify counts + label space ----------------------------------
banner "6/7  Verifying upstream split counts + label space"
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

# --- Stage 7: pre-pull DINOv2-S oracle (for OAMV; non-fatal) ----------------
banner "7/7  Pre-pulling DINOv2-S oracle (OAMV only; non-fatal)"
python - <<'PYEOF' || echo "  WARN: oracle pre-pull failed (network?). OAMV's first run will retry; baseline (E1) runs are unaffected."
import torch
torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
print("  DINOv2-S cached OK")
PYEOF

# --- Done ------------------------------------------------------------------
banner "Bootstrap complete"
cat <<MSG
Data ready at:  ${DATA_DIR}
Code ready at:  ${REPO_DIR}

Use --num-workers 8 (NOT 32 — that OOM-killed the eval). Keep --batch-size CONSTANT
across baseline and OAMV for a fair A/B (40 if matching the existing E1 seed-42 runs).

A) Sanity-check the OAMV wiring (no training):
   python scripts/02_train_oamvhfmd.py --config configs/E2_oamvhfmd_upstream.yaml \\
       --seed 42 --n-views 4 --num-workers 8 --dry-run

B) Decisive A/B — OAMV combined N=4 vs baseline N=4 (68.89%):
   nohup python scripts/02_train_oamvhfmd.py --config configs/E2_oamvhfmd_upstream.yaml \\
       --seed 42 --n-views 4 --batch-size 40 --num-workers 8 > oamv_n4.log 2>&1 &
   tail -f oamv_n4.log

Result -> results/E2_oamvhfmd_upstream/N4_seed42/eval_results.json  (compare test_top1 to 0.6889).
MSG
