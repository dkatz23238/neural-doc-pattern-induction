"""One support vector from the frozen 1.226M recognizer.

The encoder is not part of the prior. Its weights stay frozen, and the same
vector is stored in the shard so training does not reload it.
"""

from __future__ import annotations

from pathlib import Path

import torch

from pattern_induction.membership import Episode
from pattern_induction.program import MAX_LEN, MAX_SUPPORT
from pattern_induction.renaming import featurize
from pattern_induction.select_train import load_selector
from pattern_induction.span_net import _attend, _mean_max

SUPPORT_DIM = 160


def load_frozen_encoder(root: Path):
    scope, _witness, _prior = load_selector(root)
    encoder = scope.local
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


@torch.no_grad()
def embed_supports(encoder, supports: list[list[str]]) -> torch.Tensor:
    """Mean of the frozen character states over the support slots. Shape [B, 160]."""
    encoder.eval()
    episodes = [
        Episode(list(group[:MAX_SUPPORT]), group[0] if group else "", 1, "prior", "prior", "prior")
        for group in supports
    ]
    local, classes, rel, pad = featurize(episodes, max_len=MAX_LEN, max_support=MAX_SUPPORT)
    _batch, slots, length = local.shape
    roles = torch.zeros(slots, dtype=torch.long)
    roles[-1] = 1
    which = torch.arange(slots)
    hidden = encoder.class_emb(classes) + encoder.local_emb(local) + encoder.rel_proj(rel)
    hidden = hidden + encoder.char_pos[:length][None, None, :, :]
    hidden = hidden + encoder.role(roles)[None, :, None, :] + encoder.which(which)[None, :, None, :]
    hidden = hidden * (~pad).unsqueeze(-1).float()
    encoded = _attend(encoder.char_encoder, hidden, pad)
    # The last slot is the query copy. The support set was encoded once, ahead of it.
    flat = encoded[:, :-1].reshape(encoded.shape[0], -1, encoded.shape[-1])
    flat_pad = pad[:, :-1].reshape(pad.shape[0], -1)
    mean, _maxed = _mean_max(flat, flat_pad)
    return mean
