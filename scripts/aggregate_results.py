"""
Aggregate per-seed eval_results.json across a results directory into a single
CSV table.

Usage:
    python scripts/aggregate_results.py
        --exp results/E1_mvhfmd_baseline
        --out results/tables/mvhfmd_baseline.csv

For each per-seed subdir, reads:
  - config.json           (hyperparameters)
  - eval_results.json     (test_top1, test_top5, etc.)
  - convergence_report.json (status, effective_epochs, etc.)

Emits one CSV row per run, with mean ± std across seeds in the last rows.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from statistics import mean, stdev


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", type=str, required=True, help="Experiment dir under results/")
    p.add_argument("--out", type=str, required=True, help="Output CSV path")
    return p.parse_args()


def main():
    args = parse_args()
    exp = Path(args.exp)
    rows = []
    for run in sorted(exp.iterdir()):
        if not run.is_dir():
            continue
        cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
        ev  = json.loads((run / "eval_results.json").read_text()) if (run / "eval_results.json").exists() else {}
        cv  = json.loads((run / "convergence_report.json").read_text()) if (run / "convergence_report.json").exists() else {}
        ci = ev.get("test_top1_ci_hotelcluster") or {}
        rows.append({
            "run": run.name,
            "n_views": cfg.get("n_views"),
            "seed":    cfg.get("seed"),
            "tau_overlap": cfg.get("tau_overlap"),
            "test_top1": ev.get("test_top1"),
            "test_top5": ev.get("test_top5"),
            "test_top1_ci_lo_hotelcluster": ci.get("lo"),
            "test_top1_ci_hi_hotelcluster": ci.get("hi"),
            "convergence_status": cv.get("status"),
        })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
