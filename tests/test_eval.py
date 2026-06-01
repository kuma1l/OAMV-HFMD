"""Tests for oamv_hfmd.eval."""
import numpy as np


def _naive_combo_bootstrap_ci(correct, n_boot=1000, seed=0, ci=0.95):
    """Naive per-combo bootstrap (treats combos as independent)."""
    correct = np.asarray(correct, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = correct.shape[0]
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = correct[idx].mean()
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def test_cluster_bootstrap_wider_than_naive_two_hotel():
    """A 2-hotel toy set in which one hotel is all-correct and the other is
    all-wrong should yield a wider 95% CI under hotel clustering than under
    a naive per-combo bootstrap: clustering captures the between-hotel
    variance that the combo-level resample hides."""
    from oamv_hfmd.eval import cluster_bootstrap_ci

    # Hotel A (id=0): 50 combos, all correct. Hotel B (id=1): 50 combos, all wrong.
    correct = np.array([1] * 50 + [0] * 50, dtype=np.int64)
    hotel_ids = np.array([0] * 50 + [1] * 50, dtype=np.int64)

    out = cluster_bootstrap_ci(correct, hotel_ids, n_boot=2000, seed=0)
    cluster_width = out["hi"] - out["lo"]

    naive_lo, naive_hi = _naive_combo_bootstrap_ci(correct, n_boot=2000, seed=0)
    naive_width = naive_hi - naive_lo

    assert cluster_width > naive_width, (
        f"cluster CI ({out['lo']:.3f},{out['hi']:.3f}) width {cluster_width:.3f} "
        f"is not wider than naive ({naive_lo:.3f},{naive_hi:.3f}) width {naive_width:.3f}"
    )
    # And specifically: cluster bootstrap should span almost the full [0,1]
    # because hotel-level resampling sometimes draws (A,A) and sometimes (B,B).
    assert cluster_width >= 0.5
    # Mean is reported (not bootstrapped), so it stays 0.5.
    assert abs(out["mean"] - 0.5) < 1e-9


def test_cluster_bootstrap_basic_shape():
    from oamv_hfmd.eval import cluster_bootstrap_ci
    correct = np.array([1, 0, 1, 1, 0, 1], dtype=np.int64)
    hotel_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    out = cluster_bootstrap_ci(correct, hotel_ids, n_boot=100, seed=42)
    assert set(out.keys()) == {"mean", "lo", "hi", "n_boot"}
    assert out["n_boot"] == 100
    assert 0.0 <= out["lo"] <= out["mean"] <= out["hi"] <= 1.0
