"""
Convergence classifier. Single source of truth for whether a training run is
acceptable for headline comparison.

Carried over from spike_hotels50k_v2/convergence.py with slightly tightened
thresholds suited to a trainable backbone.

PLAN.md §5.5, §6.0.
"""
from __future__ import annotations


def classify(report: dict) -> tuple[str, str]:
    """Returns (status, diagnosis).

    Status: "CONVERGED" | "UNDERTRAINED" | "OVERFIT".

    UNDERTRAINED checks fire first: a run that is both undertrained and overfit
    is reported as UNDERTRAINED (the more fundamental fix).
    """
    if report["effective_epochs"] < 5:
        return "UNDERTRAINED", "fewer than 5 effective epochs of full coverage"
    if report["loss_reduction_frac"] < 0.20:
        return "UNDERTRAINED", "loss reduced by < 20% — model barely learning"
    if report["final_train_val_gap_pp"] > 20:
        return "OVERFIT", f"train-val gap = {report['final_train_val_gap_pp']:.1f} pp > 20 pp"
    # Field is named "val_points" because validation runs every 5 epochs in
    # Trainer.fit, so the last 3 val points span ~10 training epochs, not 3.
    if report.get("val_top1_delta_last_3_val_points_pp", 0.0) > 0.5:
        return "UNDERTRAINED", "val_top1 still improving in last 3 val points (> 0.5 pp delta)"
    return "CONVERGED", "all four §5.5 criteria satisfied"
