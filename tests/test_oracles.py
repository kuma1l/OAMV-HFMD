"""Tests for oamv_hfmd.oracles."""
import pytest
import torch


def test_dinov2_oracle_grad_free():
    """All DINOv2 params must be frozen."""
    from oamv_hfmd.oracles import build_oracle
    o = build_oracle("dinov2_s")
    assert len(list(o.parameters())) > 0
    for p in o.parameters():
        assert not p.requires_grad


def test_oracle_factory_unknown_kind():
    """build_oracle raises on unknown kind."""
    from oamv_hfmd.oracles import build_oracle
    with pytest.raises(ValueError):
        build_oracle("not_a_real_oracle")


def test_random_cnn_oracle_deterministic_init():
    """Two RandomCNNOracle(seed=0) produce the same forward output."""
    pytest.skip("Implement once RandomCNNOracle is fully wired")
