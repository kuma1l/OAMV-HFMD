"""Random-init untrained CNN. THE LOAD-BEARING CONTROL in E7 (PLAN.md §1.4 C7 / §18 row 7).

If this oracle works as well as DINOv2, the paper's "foundation features as
oracle" claim narrows substantially. We commit to running this comparison.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import FrozenOracleBase


class RandomCNNOracle(FrozenOracleBase):
    """ResNet50 architecture, random weights (no pretraining)."""

    feature_dim = 2048

    def __init__(self, seed: int = 0):
        super().__init__()
        from torchvision.models import resnet50
        torch.manual_seed(seed)
        m = resnet50(weights=None)  # random init
        m.fc = nn.Identity()
        self.model = m
        self._freeze()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalize(self.model(x))
