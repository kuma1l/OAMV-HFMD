"""Tests for oamv_hfmd.model."""
import pytest
import torch
import torch.nn as nn

ARCH = "vit_small_r26_s32_224"


def test_mvhfmd_baseline_forward_shape():
    """MultiImageHybrid forward returns expected dict and shapes."""
    from oamv_hfmd.model import MultiImageHybrid
    m = MultiImageHybrid(ARCH, num_classes=10, n=4, pretrained_weights=False)
    out = m(torch.zeros(2, 4, 3, 224, 224))
    assert set(out.keys()) == {"single", "mv_collection"}
    assert out["single"]["logits"].shape == (2 * 4, 10)
    assert out["mv_collection"]["logits"].shape == (2, 10)


def test_overlap_aware_hybrid_oracle_grad_free():
    """The frozen oracle must have grad_free parameters after instantiation."""
    from oamv_hfmd.model import OverlapAwareHybrid
    m = OverlapAwareHybrid(ARCH, num_classes=10, n=4, pretrained_weights=False)
    assert len(list(m.oracle.parameters())) > 0
    for p in m.oracle.parameters():
        assert not p.requires_grad


def test_overlap_aware_hybrid_forward_shape():
    """OverlapAwareHybrid forward returns the dict + similarity matrix."""
    from oamv_hfmd.model import OverlapAwareHybrid
    m = OverlapAwareHybrid(ARCH, num_classes=10, n=4, pretrained_weights=False)
    out = m(torch.zeros(2, 4, 3, 224, 224))
    assert out["single"]["logits"].shape == (2 * 4, 10)
    assert out["mv_collection"]["logits"].shape == (2, 10)
    assert out["similarity"].shape == (2, 4, 4)


def test_ccce_perturbation_guard():
    """Zeroing any slot must change the output (carries from sanity Stream 6)."""
    from oamv_hfmd.model import OverlapAwareHybrid
    torch.manual_seed(0)
    m = OverlapAwareHybrid(ARCH, num_classes=10, n=4, pretrained_weights=False)
    # Re-init head so gradients/logits aren't all zero (parent zeroes the head).
    with torch.no_grad():
        nn.init.normal_(m.model.head.weight, std=0.02)
    m.eval()
    x = torch.randn(2, 4, 3, 224, 224)
    with torch.no_grad():
        ref = m(x)["mv_collection"]["logits"]
        for k in range(4):
            x_pert = x.clone()
            x_pert[:, k] = 0.0
            out_k = m(x_pert)["mv_collection"]["logits"]
            assert not torch.allclose(ref, out_k, atol=1e-4), (
                f"CCCE guard failed: zeroing slot {k} did not change mv_collection logits"
            )
