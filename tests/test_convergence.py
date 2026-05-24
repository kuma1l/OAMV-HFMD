"""Tests for oamv_hfmd.convergence."""
from oamv_hfmd.convergence import classify


def _report(**kwargs) -> dict:
    base = {
        "effective_epochs": 100,
        "loss_reduction_frac": 0.5,
        "final_train_val_gap_pp": 5.0,
        "val_top1_delta_last_3_epochs_pp": 0.1,
    }
    base.update(kwargs)
    return base


def test_converged_when_all_criteria_met():
    status, _ = classify(_report())
    assert status == "CONVERGED"


def test_undertrained_few_effective_epochs():
    status, _ = classify(_report(effective_epochs=3))
    assert status == "UNDERTRAINED"


def test_undertrained_low_loss_reduction():
    status, _ = classify(_report(loss_reduction_frac=0.1))
    assert status == "UNDERTRAINED"


def test_overfit_large_gap():
    status, _ = classify(_report(final_train_val_gap_pp=25.0))
    assert status == "OVERFIT"


def test_undertrained_val_still_improving():
    status, _ = classify(_report(val_top1_delta_last_3_epochs_pp=1.0))
    assert status == "UNDERTRAINED"
