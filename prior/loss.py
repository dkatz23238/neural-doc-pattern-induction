"""Listwise cross-entropy over the candidates of one support set."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ranking_loss(scores: torch.Tensor, gold: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Gold is a mask of every candidate whose language matches the latent one.

    The loss is ``log Z - log Z_gold``. Picking any latent-perfect candidate
    is a hit; the others share the remaining mass.
    """
    filled = scores.masked_fill(~mask, -1e9)
    gold_filled = scores.masked_fill(~(mask & gold), -1e9)
    return (torch.logsumexp(filled, dim=-1) - torch.logsumexp(gold_filled, dim=-1)).mean()


def factor_loss(
    scores: torch.Tensor,
    aux_logits: torch.Tensor,
    gold: torch.Tensor,
    mask: torch.Tensor,
    targets: torch.Tensor,
    weight: float = 0.5,
) -> torch.Tensor:
    """Rank the sum of local scores, and train each piece on its own target.

    The auxiliary term asks whether the field, the width, the repeat, the
    optional, and the affix match a probe-perfect candidate. A negative
    candidate that got one of those right still teaches that piece.
    """
    rank = ranking_loss(scores, gold, mask)
    valid = mask.unsqueeze(-1).to(dtype=aux_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(aux_logits, targets, reduction="none")
    extra = (bce * valid).sum() / (valid.sum().clamp_min(1.0) * bce.shape[-1])
    return rank + weight * extra
