"""Tests for oamv_hfmd.losses."""
import pytest
import torch


def test_symmetric_kl_zero_on_identical_inputs():
    """sym_KL(p, p) == 0."""
    from oamv_hfmd.losses import symmetric_kl
    p = torch.randn(4, 10)
    loss = symmetric_kl(p, p.clone(), tau=4.0)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


def test_symmetric_kl_positive_on_disjoint_inputs():
    """sym_KL on clearly different distributions is > 0."""
    from oamv_hfmd.losses import symmetric_kl
    p = torch.zeros(2, 10); p[:, 0] = 10.0   # one-hot on class 0
    q = torch.zeros(2, 10); q[:, 1] = 10.0   # one-hot on class 1
    loss = symmetric_kl(p, q, tau=1.0)
    assert loss.item() > 0


def test_overlap_md_loss_shape_and_finite():
    """overlap_md_loss returns finite scalar."""
    from oamv_hfmd.losses import overlap_md_loss
    B, N, K = 2, 4, 10
    z_mv = torch.randn(B, K)
    z_single = torch.randn(B, N, K)
    S = torch.eye(N).expand(B, N, N).float()
    loss = overlap_md_loss(z_mv, z_single, S)
    assert torch.isfinite(loss)


def test_baseline_md_loss_shape_and_finite():
    """baseline_md_loss returns finite scalar."""
    from oamv_hfmd.losses import baseline_md_loss
    B, N, K = 2, 4, 10
    z_mv = torch.randn(B, K)
    z_single = torch.randn(B, N, K)
    loss = baseline_md_loss(z_mv, z_single)
    assert torch.isfinite(loss)


def test_baseline_md_loss_matches_upstream():
    """Numerical equality against multi-view-hybrid/loss/loss.py:MutualDistillationLoss."""
    import sys
    sys.path.insert(0, r"D:\Research-WS\PIVOT\multi-view-hybrid")
    try:
        from loss.loss import MutualDistillationLoss
    except ImportError:
        pytest.skip("upstream multi-view-hybrid not available")

    from oamv_hfmd.losses import baseline_md_loss

    torch.manual_seed(0)
    B, N, K = 4, 4, 100
    z_mv = torch.randn(B, K)
    z_single = torch.randn(B, N, K)
    targets = torch.randint(0, K, (B,))

    upstream = MutualDistillationLoss(temp=4.0, lambda_hyperparam=0.1)
    loss_up = upstream(z_mv, z_single, targets)
    loss_ours = baseline_md_loss(z_mv, z_single, tau=4.0, lambda_hyperparam=0.1)
    assert torch.allclose(loss_up, loss_ours, atol=1e-5), (
        f"upstream={float(loss_up)} ours={float(loss_ours)} diff={float(loss_up - loss_ours)}"
    )
