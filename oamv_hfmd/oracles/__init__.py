"""Foundation-model oracles for the similarity matrix S.

All oracles share the interface:

    class Oracle(nn.Module):
        feature_dim: int
        @torch.no_grad()
        def forward(self, x):  # (B*N, 3, H, W) -> (B*N, feature_dim) L2-normalized

The `build_oracle(kind)` factory dispatches to the right class. PLAN.md §1.4 C2, §6.E7.
"""
from __future__ import annotations

import torch.nn as nn


def build_oracle(kind: str) -> nn.Module:
    """Factory for oracles. ``kind`` ∈ {dinov2_s, clip_b16, siglip_b16, resnet50_in1k, random_cnn}."""
    if kind == "dinov2_s":
        from .dinov2 import DinoV2SmallOracle
        return DinoV2SmallOracle()
    if kind == "clip_b16":
        from .clip import ClipB16Oracle
        return ClipB16Oracle()
    if kind == "siglip_b16":
        from .siglip import SigLipB16Oracle
        return SigLipB16Oracle()
    if kind == "resnet50_in1k":
        from .resnet import ResNet50In1kOracle
        return ResNet50In1kOracle()
    if kind == "random_cnn":
        from .random_cnn import RandomCNNOracle
        return RandomCNNOracle()
    raise ValueError(f"Unknown oracle kind: {kind}")
