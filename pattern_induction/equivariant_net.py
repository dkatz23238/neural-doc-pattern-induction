"""Membership encoder that never sees a raw codepoint.

Each character is a structural class, a task-local symbol id, and its
relations to the support: same aligned symbol, same class, constant column,
or new to the task. A within-class rename does not change those inputs, so
the latent language state cannot treat `§` as suspicious merely because that
byte was absent from training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pattern_induction.membership import MAX_SUPPORT, Episode
from pattern_induction.renaming import MAX_SYMBOLS, N_CLASS, N_REL, featurize


class EquivariantNet(nn.Module):
    def __init__(
        self,
        d_model: int = 160,
        layers: int = 3,
        heads: int = 4,
        ff: int = 640,
        max_len: int = 48,
        max_support: int = MAX_SUPPORT,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.max_len = max_len
        self.max_support = max_support
        self.d_model = d_model
        self.class_emb = nn.Embedding(N_CLASS, d_model)
        self.local_emb = nn.Embedding(MAX_SYMBOLS + 1, d_model, padding_idx=0)
        self.rel_proj = nn.Linear(N_REL, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.role = nn.Embedding(2, d_model)
        self.which = nn.Embedding(max_support + 1, d_model)
        nn.init.normal_(self.pos.weight, std=0.02)
        nn.init.normal_(self.role.weight, std=0.02)
        nn.init.normal_(self.which.weight, std=0.02)
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(
        self,
        local: torch.Tensor,
        classes: torch.Tensor,
        rel: torch.Tensor,
        pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, slots, length = local.shape
        positions = torch.arange(length, device=local.device)
        roles = torch.zeros(slots, dtype=torch.long, device=local.device)
        roles[-1] = 1
        which = torch.arange(slots, device=local.device)
        hidden = self.class_emb(classes) + self.local_emb(local) + self.rel_proj(rel)
        hidden = hidden + self.pos(positions)[None, None, :, :]
        hidden = hidden + self.role(roles)[None, :, None, :] + self.which(which)[None, :, None, :]
        hidden = hidden * (~pad).unsqueeze(-1).float()
        flat = hidden.reshape(batch, slots * length, self.d_model)
        key_pad = pad.reshape(batch, slots * length)
        if bool(key_pad.all(dim=1).any()):
            key_pad = key_pad.clone()
            key_pad[key_pad.all(dim=1), 0] = False
        encoded = self.encoder(flat, src_key_padding_mask=key_pad)
        encoded = encoded.reshape(batch, slots, length, self.d_model)
        query = encoded[:, -1]
        live = (~pad[:, -1]).float().unsqueeze(-1)
        latent = (query * live).sum(dim=1) / live.sum(dim=1).clamp(min=1.0)
        return self.head(latent).squeeze(-1), latent


def collate_episodes(
    episodes: list[Episode],
    max_len: int = 48,
    max_support: int = MAX_SUPPORT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    local, classes, rel, pad = featurize(episodes, max_len=max_len, max_support=max_support)
    labels = torch.tensor([float(episode.label) for episode in episodes])
    return local, classes, rel, pad, labels


def equivariant_loss(
    logit: torch.Tensor,
    latent: torch.Tensor,
    logit_renamed: torch.Tensor,
    latent_renamed: torch.Tensor,
    labels: torch.Tensor,
    consistency: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = F.binary_cross_entropy_with_logits(logit, labels)
    bce = bce + F.binary_cross_entropy_with_logits(logit_renamed, labels)
    first = F.normalize(latent, dim=-1)
    second = F.normalize(latent_renamed, dim=-1)
    agree = (1.0 - (first * second).sum(dim=-1)).mean()
    return bce + consistency * agree, bce.detach(), agree.detach()


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
