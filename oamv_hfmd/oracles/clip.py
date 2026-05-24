"""CLIP-B/16 oracle (openai/clip-vit-base-patch16). PLAN.md §6.E7 ablation."""
from __future__ import annotations

import torch

from .base import FrozenOracleBase


class ClipB16Oracle(FrozenOracleBase):
    feature_dim = 512

    def __init__(self):
        super().__init__()
        # TODO: load via transformers.CLIPVisionModel
        # from transformers import CLIPVisionModel
        # self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        self.model = None
        self._freeze()
        raise NotImplementedError("Wire transformers CLIPVisionModel; pool to pooler_output")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CLIP expects different normalization than ImageNet; document at call site.
        raise NotImplementedError
