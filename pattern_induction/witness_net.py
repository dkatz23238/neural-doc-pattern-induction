"""A small head on the frozen relational encoder.

The question is narrow: given the supports and one string where candidate
languages disagree, is that string in the language? The encoder stays frozen.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pattern_induction.membership import Episode
from pattern_induction.program import MAX_LEN, MAX_SUPPORT
from pattern_induction.renaming import featurize
from pattern_induction.span_net import _attend, _mean_max


def geometry(supports: list[str], query: str) -> list[float]:
    """Class-and-count features. Renaming inside a class leaves them unchanged."""

    def run(text: str, pred) -> int:
        best = current = 0
        for ch in text:
            if pred(ch):
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    def puncts(text: str) -> int:
        return sum(not ch.isalnum() for ch in text)

    longest = max((len(text) for text in supports), default=1)
    widest = max((run(text, str.isdigit) for text in supports), default=0)
    support_puncts = {ch for text in supports for ch in text if not ch.isalnum()}
    query_puncts = {ch for ch in query if not ch.isalnum()}
    return [
        len(query) / 16.0,
        longest / 16.0,
        (len(query) - longest) / 16.0,
        puncts(query) / 4.0,
        max((puncts(text) for text in supports), default=0) / 4.0,
        run(query, str.isdigit) / 4.0,
        widest / 4.0,
        float(query_puncts <= support_puncts),
    ]


def collate_witnesses(pairs: list[tuple[list[str], str]]) -> dict[str, torch.Tensor]:
    episodes = [
        Episode(supports[:MAX_SUPPORT], query[:MAX_LEN], 1, "witness", "s1", "witness") for supports, query in pairs
    ]
    local, classes, rel, pad = featurize(episodes, max_len=MAX_LEN, max_support=MAX_SUPPORT)
    geom = torch.tensor([geometry(supports, query[:MAX_LEN]) for supports, query in pairs], dtype=torch.float32)
    return {"local": local, "classes": classes, "rel": rel, "pad": pad, "geom": geom}


class WitnessNet(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        dim = encoder.d_model
        self.head = nn.Sequential(nn.Linear(dim * 4 + 8, dim), nn.GELU(), nn.Linear(dim, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.__dict__["encoder"] = encoder

    def latent(self, batch) -> torch.Tensor:
        encoder = self.encoder
        encoder.eval()
        local, classes, rel, pad = batch["local"], batch["classes"], batch["rel"], batch["pad"]
        _batch, slots, length = local.shape
        roles = torch.zeros(slots, dtype=torch.long)
        roles[-1] = 1
        which = torch.arange(slots)
        hidden = encoder.class_emb(classes) + encoder.local_emb(local) + encoder.rel_proj(rel)
        hidden = hidden + encoder.char_pos[:length][None, None, :, :]
        hidden = hidden + encoder.role(roles)[None, :, None, :] + encoder.which(which)[None, :, None, :]
        hidden = hidden * (~pad).unsqueeze(-1).float()
        with torch.no_grad():
            encoded = _attend(encoder.char_encoder, hidden, pad)
            support_pad = pad[:, :-1].reshape(pad.shape[0], -1)
            support = encoded[:, :-1].reshape(encoded.shape[0], -1, encoded.shape[-1])
            # _mean_max expects pad aligned with the flattened support characters.
            support_mean, support_max = _mean_max(support, support_pad)
            query_mean, query_max = _mean_max(encoded[:, -1], pad[:, -1])
        return torch.cat([support_mean, support_max, query_mean, query_max, batch["geom"]], dim=-1)

    def forward(self, batch) -> torch.Tensor:
        return self.head(self.latent(batch)).squeeze(-1)


def witness_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weight = torch.where(target > 0.5, torch.full_like(target, 2.0), torch.ones_like(target))
    loss = F.binary_cross_entropy_with_logits(logit, target, weight=weight)
    positive = logit[target > 0.5]
    negative = logit[target < 0.5]
    if positive.numel() and negative.numel():
        loss = loss + 0.5 * F.softplus(negative[:, None] - positive[None, :]).mean()
    return loss
