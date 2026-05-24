"""SigLIP-B/16 oracle (google/siglip-base-patch16-224). PLAN.md §6.E7 ablation."""
from __future__ import annotations

import torch

from .base import FrozenOracleBase


class SigLipB16Oracle(FrozenOracleBase):
    feature_dim = 768

    def __init__(self):
        super().__init__()
        # TODO: transformers.SiglipVisionModel
        self.model = None
        self._freeze()
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
