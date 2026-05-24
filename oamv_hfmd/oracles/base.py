"""Base class for oracles. Enforces frozen parameters and L2-normalized output."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenOracleBase(nn.Module):
    """Common scaffolding: freeze params, return L2-normalized CLS-equivalent."""

    feature_dim: int

    def _freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.float(), dim=-1)
