"""
Model definitions for OAMV-HFMD.

Two model classes:

- ``MultiImageHybrid`` — the MV-HFMD baseline (Black & Souvenir, WACV 2024).
  Productionized from ``multi-view-hybrid/model/multi_image_model.py``.

- ``OverlapAwareHybrid`` — our method. Subclass of MultiImageHybrid that
  swaps the static embedding for ``MLP(S[i, :])`` and adds a side-channel
  forward through a frozen oracle to produce S.

The frozen oracle has gradients disabled at __init__; this is asserted at
runtime at the end of ``OverlapAwareHybrid.__init__``.

PLAN.md §4.1, §4.2, §4.3.
"""
from __future__ import annotations

import einops
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .oracles import build_oracle


class MultiImageHybrid(nn.Module):
    """MV-HFMD baseline. Reproduces ``multi-view-hybrid/model/multi_image_model.py``.

    Forward returns dict:
        ``{"single": {"logits": (B*N, K)}, "mv_collection": {"logits": (B, K)}}``
    (``mv_collection`` is omitted when ``n == 1``.)

    Args:
        arch: timm model name. Use ``vit_small_r26_s32_224``.
        num_classes: number of hotel classes.
        n: number of input views.
        pretrained_weights: load timm pretrained weights.
    """

    def __init__(self, arch: str, num_classes: int, n: int, pretrained_weights: bool = True):
        super().__init__()
        self.n = n
        self.num_classes = num_classes
        self.pretrained_weights = pretrained_weights

        drop_rate = 0.0 if "tiny" in arch else 0.1
        self.model = timm.create_model(
            arch,
            pretrained=pretrained_weights,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
        for block in self.model.blocks:
            block.attn.fused_attn = False
        for block in self.model.blocks:
            block.attn.proj_drop = nn.Dropout(p=0.0)

        self.embed_dim = self.model.embed_dim

        self.img_embed_matrix = nn.Parameter(
            torch.zeros(1, n, self.embed_dim), requires_grad=True
        )
        nn.init.xavier_uniform_(self.img_embed_matrix)

        nn.init.zeros_(self.model.head.weight)
        nn.init.zeros_(self.model.head.bias)

    def format_multi_image_tokens(self, x, batch_size, tokens_per_image, **kwargs):
        x = einops.rearrange(x, "(b n) s c -> b (n s) c", b=batch_size, n=self.n)
        first_img_token_idx = 0
        if self.model.cls_token is not None:
            for i in range(1, self.n):
                excess_cls_index = i * tokens_per_image + 1
                x = torch.cat((x[:, :excess_cls_index], x[:, excess_cls_index + 1:]), dim=1)
            first_img_token_idx = 1

        image_embeddings = F.normalize(self.img_embed_matrix, dim=-1)
        x[:, first_img_token_idx:] += torch.repeat_interleave(
            image_embeddings, tokens_per_image, dim=1
        )
        return x

    def forward(self, x: torch.Tensor) -> dict:
        batch_size = len(x)
        output_dict: dict = {"single": {}}
        if self.n > 1:
            output_dict["mv_collection"] = {}

        x = einops.rearrange(x, "b n c h w -> (b n) c h w")
        x = self.model.patch_embed(x)

        tokens_per_image = x.shape[1]
        x = self.model._pos_embed(x)

        for view_type in output_dict:
            tokens = x.clone()
            if view_type == "mv_collection":
                tokens = self.format_multi_image_tokens(tokens, batch_size, tokens_per_image)
            tokens = self.model.blocks(tokens)
            tokens = self.model.norm(tokens)
            output_dict[view_type]["logits"] = self.model.forward_head(tokens)

        return output_dict


class FrozenDinoOracle(nn.Module):
    """Frozen DINOv2-S used only for computing similarity matrix S.

    Loads via torch.hub. ``feature_dim = 384``. All parameters frozen at __init__.
    Kept here for backwards compatibility with sanity Stream 6; the canonical
    factory is :func:`oamv_hfmd.oracles.build_oracle`.
    """

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.model.eval().float()
        for p in self.model.parameters():
            p.requires_grad = False
        self.feature_dim = 384

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls = self.model(x).float()
        return F.normalize(cls, dim=-1)


class OverlapAwareHybrid(MultiImageHybrid):
    """Our method. Subclass of MV-HFMD with:

    1. A frozen oracle producing per-view CLS tokens, yielding ``S = cos(CLS_i, CLS_j)``.
    2. ``MLP(S[i, :])`` replacing the static ``img_embed_matrix``
       (when ``enable_overlap_embed=True``).

    The original ``img_embed_matrix`` is kept as a (frozen) parameter so that
    state-dict loading from an MV-HFMD checkpoint still works.

    Forward returns dict with ``'similarity'`` added when ``n > 1``:
        ``{"single": ..., "mv_collection": ..., "similarity": (B, N, N)}``
    so the trainer can feed S into ``overlap_md_loss``.

    Args (in addition to parent):
        oracle_kind: which foundation oracle to use ("dinov2_s", "clip_b16", ...).
        mlp_hidden: hidden width of the embedding MLP. Default 64.
        enable_overlap_embed: if True (default), use MLP(S[i,:]) for per-view
            embedding (E3 / E2). If False (E4, loss-only), fall back to the
            parent's static ``img_embed_matrix``.
    """

    def __init__(
        self,
        arch: str,
        num_classes: int,
        n: int,
        pretrained_weights: bool = True,
        oracle_kind: str = "dinov2_s",
        mlp_hidden: int = 64,
        enable_overlap_embed: bool = True,
    ):
        super().__init__(arch, num_classes, n, pretrained_weights)
        self.oracle_kind = oracle_kind
        self.enable_overlap_embed = enable_overlap_embed

        # Freeze the static embedding — we replace it with MLP(S) when enabled.
        self.img_embed_matrix.requires_grad_(False)

        self.oracle = build_oracle(oracle_kind)
        self.overlap_mlp = nn.Sequential(
            nn.Linear(n, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, self.embed_dim),
        )

        # Critical invariant: oracle must be grad-free after __init__.
        assert all(not p.requires_grad for p in self.oracle.parameters()), (
            "OverlapAwareHybrid: oracle parameters must be frozen after __init__"
        )

    def compute_similarity(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, 3, H, W) -> S: (B, N, N) cosine similarities (L2-normalized)."""
        B, N, C, H, W = x.shape
        flat = einops.rearrange(x, "b n c h w -> (b n) c h w")
        cls = self.oracle(flat)  # (B*N, feature_dim), L2-normalized
        cls = einops.rearrange(cls, "(b n) d -> b n d", b=B, n=N)
        return torch.bmm(cls, cls.transpose(1, 2))

    def format_multi_image_tokens(self, x, batch_size, tokens_per_image, S=None, **kwargs):
        """Override: optionally use MLP(S[i, :]) for the per-view embedding."""
        if not self.enable_overlap_embed:
            return super().format_multi_image_tokens(x, batch_size, tokens_per_image)

        assert S is not None, (
            "OverlapAwareHybrid.format_multi_image_tokens requires S when "
            "enable_overlap_embed=True"
        )
        x = einops.rearrange(x, "(b n) s c -> b (n s) c", b=batch_size, n=self.n)
        first_img_token_idx = 0
        if self.model.cls_token is not None:
            for i in range(1, self.n):
                excess = i * tokens_per_image + 1
                x = torch.cat((x[:, :excess], x[:, excess + 1:]), dim=1)
            first_img_token_idx = 1

        emb = self.overlap_mlp(S)  # (B, N, embed_dim)
        emb = F.normalize(emb, dim=-1)
        x[:, first_img_token_idx:] += torch.repeat_interleave(emb, tokens_per_image, dim=1)
        return x

    def forward(self, x: torch.Tensor) -> dict:
        batch_size = len(x)
        output_dict: dict = {"single": {}}
        if self.n > 1:
            output_dict["mv_collection"] = {}

        S = self.compute_similarity(x) if self.n > 1 else None

        x_flat = einops.rearrange(x, "b n c h w -> (b n) c h w")
        feats = self.model.patch_embed(x_flat)
        tokens_per_image = feats.shape[1]
        feats = self.model._pos_embed(feats)

        for view_type in list(output_dict.keys()):
            tokens = feats.clone()
            if view_type == "mv_collection":
                tokens = self.format_multi_image_tokens(
                    tokens, batch_size, tokens_per_image, S=S
                )
            tokens = self.model.blocks(tokens)
            tokens = self.model.norm(tokens)
            output_dict[view_type]["logits"] = self.model.forward_head(tokens)

        if S is not None:
            output_dict["similarity"] = S
        return output_dict
