"""Listwise cross-entropy over the candidates of one support set."""

from __future__ import annotations

import torch


def ranking_loss(scores: torch.Tensor, gold: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Gold is a mask of every candidate whose language matches the latent one.

    The loss is ``log Z - log Z_gold``. Picking any latent-perfect candidate
    is a hit; the others share the remaining mass.
    """
    filled = scores.masked_fill(~mask, -1e9)
    gold_filled = scores.masked_fill(~(mask & gold), -1e9)
    return (torch.logsumexp(filled, dim=-1) - torch.logsumexp(gold_filled, dim=-1)).mean()
