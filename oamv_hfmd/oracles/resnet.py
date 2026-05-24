"""ImageNet-pretrained ResNet50 oracle. PLAN.md §6.E7 ablation (non-foundation control)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import FrozenOracleBase


class ResNet50In1kOracle(FrozenOracleBase):
    feature_dim = 2048

    def __init__(self):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Replace classifier with identity to expose pre-fc features
        m.fc = nn.Identity()
        self.model = m
        self._freeze()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalize(self.model(x))
