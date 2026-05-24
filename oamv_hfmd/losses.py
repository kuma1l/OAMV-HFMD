"""
Loss functions for OAMV-HFMD.

- ``baseline_md_loss`` — MV-HFMD's uniform mutual-distillation loss (asymmetric
  KL pair). Reimplemented from the paper §3.2 until upstream code is available.

- ``overlap_md_loss`` — our overlap-weighted symmetric KL replacement. Uses
  similarity matrix S to weight per-pair contributions.

PLAN.md §1.4 C3, §4.3.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_kl(p: torch.Tensor, q: torch.Tensor, tau: float = 4.0) -> torch.Tensor:
    """Jensen-Shannon-style symmetric KL between two logit tensors.

    Args:
        p, q: logits, same shape (..., K).
        tau: temperature.

    Returns:
        scalar loss.
    """
    p_soft = F.softmax(p / tau, dim=-1)
    q_soft = F.softmax(q / tau, dim=-1)
    m = 0.5 * (p_soft + q_soft)
    log_m = m.log()
    return 0.5 * F.kl_div(log_m, p_soft, reduction="batchmean") \
         + 0.5 * F.kl_div(log_m, q_soft, reduction="batchmean")


def baseline_md_loss(
    z_mv: torch.Tensor,
    z_single: torch.Tensor,
    tau: float = 4.0,
    lambda_hyperparam: float = 0.1,
) -> torch.Tensor:
    """MV-HFMD's uniform mutual-distillation loss.

    Reproduces ``MutualDistillationLoss.forward`` from upstream
    ``multi-view-hybrid/loss/loss.py`` exactly:
      1. Score-fused teacher = mean over views of (logit / tau)
      2. Two KL terms with gradient-detached teachers (asymmetric pair)
      3. Hinton-style scaling: final loss = mean_KL * tau**2 * lambda_hyperparam

    Args:
        z_mv:   (B, K) multi-view logits.
        z_single: (B, N, K) per-view single-view logits.
        tau:    distillation temperature (main.py md_temp default 4.0).
        lambda_hyperparam: loss weight (main.py md_lambda default 0.1).

    Returns:
        scalar loss already scaled by tau**2 * lambda — add directly to total loss.
    """
    avg_single = torch.mean(z_single, dim=1)                  # (B, K)

    p = torch.softmax(z_mv / tau, dim=1)
    q = torch.softmax(avg_single / tau, dim=1)

    log_p = torch.log_softmax(z_mv / tau, dim=1)
    log_q = torch.log_softmax(avg_single / tau, dim=1)

    # nn.KLDivLoss(reduction='none').sum(dim=1).mean()  — per-sample sum, then batch mean
    kl_pq = F.kl_div(log_p, q.detach(), reduction="none").sum(dim=1).mean()
    kl_qp = F.kl_div(log_q, p.detach(), reduction="none").sum(dim=1).mean()
    loss = 0.5 * (kl_pq + kl_qp)

    return loss * (tau ** 2) * lambda_hyperparam


def overlap_md_loss(
    z_mv: torch.Tensor,
    z_single: torch.Tensor,
    S: torch.Tensor,
    tau_overlap: float = 4.0,
    tau_kl: float = 4.0,
    lambda_hyperparam: float = 0.1,
) -> torch.Tensor:
    """Overlap-weighted symmetric mutual distillation. PLAN.md §1.4 C3.

    Drop-in replacement for ``baseline_md_loss`` with two changes:
      1. The teacher z_bar[i] for view i is the softmax(S[i,:]/tau_overlap)-weighted
         aggregation over per-view logits (instead of the uniform mean).
      2. The per-pair KL is symmetric (Jensen-Shannon-style) instead of MV-HFMD's
         asymmetric pair.

    Same Hinton scaling as ``baseline_md_loss`` for fair comparison: final loss is
    scaled by ``tau_kl ** 2 * lambda_hyperparam``.

    Args:
        z_mv: (B, K) multi-view logits.
        z_single: (B, N, K) per-view single-view logits.
        S: (B, N, N) cosine similarity matrix from the frozen oracle.
        tau_overlap: temperature for the softmax over S rows.
        tau_kl: KL temperature (matches md_temp in upstream).
        lambda_hyperparam: loss weight (matches md_lambda in upstream).

    Returns:
        scalar loss already scaled by tau_kl**2 * lambda — add directly to total loss.
    """
    weights = torch.softmax(S / tau_overlap, dim=-1)   # (B, N, N)
    z_bar = torch.bmm(weights, z_single)                # (B, N, K)
    B, N, _ = z_bar.shape

    loss = 0.0
    for i in range(N):
        loss = loss + symmetric_kl(z_mv, z_bar[:, i], tau=tau_kl)
    loss = loss / N

    return loss * (tau_kl ** 2) * lambda_hyperparam
