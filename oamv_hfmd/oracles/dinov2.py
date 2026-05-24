"""DINOv2-S oracle via torch.hub. The default choice. PLAN.md §1.4 C2."""
from __future__ import annotations

import torch

from .base import FrozenOracleBase


class DinoV2SmallOracle(FrozenOracleBase):
    feature_dim = 384

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self._freeze()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*N, 3, 224, 224)  ->  (B*N, 384) L2-normalized
        return self._normalize(self.model(x))
