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


def test_loss_only_embedding_is_trainable():
    """E4 / loss-only path: parent's img_embed_matrix must be trainable and
    actually receive gradients on a forward+backward."""
    from oamv_hfmd.model import OverlapAwareHybrid
    torch.manual_seed(0)
    m = OverlapAwareHybrid(
        ARCH, num_classes=10, n=4, pretrained_weights=False,
        enable_overlap_embed=False,
    )
    assert m.img_embed_matrix.requires_grad is True
    with torch.no_grad():
        nn.init.normal_(m.model.head.weight, std=0.02)
    m.train()
    x = torch.randn(2, 4, 3, 224, 224)
    out = m(x)
    # mv_collection path is what consumes img_embed_matrix.
    loss = out["mv_collection"]["logits"].sum()
    loss.backward()
    assert m.img_embed_matrix.grad is not None
    assert m.img_embed_matrix.grad.abs().sum().item() > 0.0

    # Embed-enabled model still forwards and returns similarity.
    m2 = OverlapAwareHybrid(
        ARCH, num_classes=10, n=4, pretrained_weights=False,
        enable_overlap_embed=True,
    )
    out2 = m2(torch.zeros(2, 4, 3, 224, 224))
    assert "similarity" in out2


def test_overlap_mlp_input_is_column_permutation_invariant():
    """Per D-1(a): sorting S[i, :] before the MLP makes the per-view embedding
    invariant to column permutations of S (the random within-collection view
    ordering). Same view → same embedding regardless of how columns are
    shuffled. Tested at the MLP-input level: feed two column-permuted versions
    of S through the sort+MLP path and assert equality.
    """
    import torch.nn.functional as F
    from oamv_hfmd.model import OverlapAwareHybrid
    torch.manual_seed(0)
    m = OverlapAwareHybrid(ARCH, num_classes=10, n=4, pretrained_weights=False)
    m.eval()

    # Synthesise an arbitrary similarity matrix with 1.0 on diagonal.
    B, N = 2, 4
    S = torch.rand(B, N, N) * 0.7 + 0.3
    S = 0.5 * (S + S.transpose(1, 2))
    for b in range(B):
        S[b].fill_diagonal_(1.0)

    # Permute columns; for the per-view embedding to remain meaningful we
    # ALSO need to permute rows (a view-permutation permutes both). We test
    # both the column-only invariance (sufficient for the sort fix) and the
    # combined row+column permutation (sufficient at the mechanism level).
    perm = torch.tensor([2, 0, 3, 1])

    def _emb(s):
        s_sorted, _ = torch.sort(s, dim=-1, descending=True)
        out = m.overlap_mlp(s_sorted)
        return F.normalize(out, dim=-1)

    emb_orig = _emb(S)
    emb_col_perm = _emb(S[:, :, perm])  # columns shuffled only
    assert torch.allclose(emb_orig, emb_col_perm, atol=1e-6), (
        "MLP output should be invariant to column permutation of S after sort"
    )

    # Full view-permutation: rows and columns permuted together. The set of
    # per-view embeddings should match (the i-th original view's embedding
    # should appear at the permuted position).
    S_view_perm = S[:, perm][:, :, perm]
    emb_view_perm = _emb(S_view_perm)
    assert torch.allclose(emb_orig[:, perm], emb_view_perm, atol=1e-6), (
        "Under view permutation, per-view embeddings should permute correspondingly"
    )


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
