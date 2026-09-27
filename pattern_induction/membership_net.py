"""Token-level set encoder for P(x in L_S).

Support strings and the query keep a state per byte. A frozen identity code
makes two copies of the same byte match even when that byte never occurred in
training. A pointer then copies support states onto query positions that share
that identity. The membership head reads the query, and an auxiliary head
marks which query bytes were literals or delimiters.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pattern_induction.membership import MAX_LEN, MAX_SUPPORT, PAD_ID, Episode, byte_flags

# Unit-vector dots are about 1 for the same byte and about 0 otherwise.
# The support is hundreds of bytes long, so the pointer needs a wide gap
# before softmax or it stays almost uniform.
POINTER_SCALE = 12.0


class MembershipNet(nn.Module):
    def __init__(
        self,
        d_model: int = 192,
        layers: int = 4,
        heads: int = 4,
        ff: int = 768,
        max_len: int = MAX_LEN,
        max_support: int = MAX_SUPPORT,
        id_dim: int = 48,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.max_len = max_len
        self.max_support = max_support
        self.d_model = d_model
        self.byte = nn.Embedding(PAD_ID + 1, d_model, padding_idx=PAD_ID)
        nn.init.normal_(self.byte.weight, std=0.02)
        with torch.no_grad():
            self.byte.weight[PAD_ID].zero_()
        identity = F.normalize(torch.randn(PAD_ID + 1, id_dim), dim=-1)
        identity[PAD_ID].zero_()
        self.register_buffer("identity", identity)
        self.register_buffer("id_proj", torch.randn(id_dim, d_model) / (id_dim**0.5))
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
        self.inv_head = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor, pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, slots, length = tokens.shape
        learned = self.byte(tokens)
        identity = F.embedding(tokens, self.identity) @ self.id_proj
        positions = torch.arange(length, device=tokens.device)
        roles = torch.zeros(slots, dtype=torch.long, device=tokens.device)
        roles[-1] = 1
        which = torch.arange(slots, device=tokens.device)
        hidden = learned + identity + self.pos(positions)[None, None, :, :]
        hidden = hidden + self.role(roles)[None, :, None, :] + self.which(which)[None, :, None, :]
        flat = hidden.reshape(batch, slots * length, self.d_model)
        key_pad = pad.reshape(batch, slots * length)
        if bool(key_pad.all(dim=1).any()):
            key_pad = key_pad.clone()
            key_pad[key_pad.all(dim=1), 0] = False
        encoded = self.encoder(flat, src_key_padding_mask=key_pad).reshape(batch, slots, length, self.d_model)
        query = encoded[:, -1]
        support = encoded[:, :-1]
        query_id = F.embedding(tokens[:, -1], self.identity)
        support_id = F.embedding(tokens[:, :-1], self.identity).reshape(batch, (slots - 1) * length, -1)
        support_pad = pad[:, :-1].reshape(batch, (slots - 1) * length)
        cosine = torch.matmul(query_id, support_id.transpose(-1, -2))
        cosine = cosine.masked_fill(support_pad[:, None, :], -2.0)
        pointed = torch.matmul(
            torch.softmax(cosine * POINTER_SCALE, dim=-1),
            support.reshape(batch, (slots - 1) * length, self.d_model),
        )
        has_support = (~support_pad).any(dim=1).float()
        query = query + pointed * has_support[:, None, None]
        live = (~pad[:, -1]).float().unsqueeze(-1)
        pooled = (query * live).sum(dim=1) / live.sum(dim=1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1), self.inv_head(query).squeeze(-1)


def collate(episodes: list[Episode], max_len: int = MAX_LEN, max_support: int = MAX_SUPPORT) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(episodes)
    slots = max_support + 1
    tokens = torch.full((batch, slots, max_len), PAD_ID, dtype=torch.long)
    pad = torch.ones((batch, slots, max_len), dtype=torch.bool)
    labels = torch.zeros(batch)
    invariant = torch.full((batch, max_len), -1.0)
    for row, episode in enumerate(episodes):
        strings = list(episode.support[:max_support])
        while len(strings) < max_support:
            strings.append("")
        strings.append(episode.query)
        for slot, text in enumerate(strings):
            raw = text.encode("utf-8")[:max_len]
            if not raw:
                continue
            tokens[row, slot, : len(raw)] = torch.tensor(list(raw), dtype=torch.long)
            pad[row, slot, : len(raw)] = False
        labels[row] = float(episode.label)
        if episode.label == 1 and episode.invariant:
            flags = byte_flags(episode.query, episode.invariant)[:max_len]
            if len(flags) == min(len(episode.query.encode("utf-8")), max_len):
                for index, flag in enumerate(flags):
                    if not pad[row, -1, index]:
                        invariant[row, index] = float(flag)
    return tokens, pad, labels, invariant


def membership_loss(
    logit: torch.Tensor,
    inv_logit: torch.Tensor,
    labels: torch.Tensor,
    invariant: torch.Tensor,
    aux_weight: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = F.binary_cross_entropy_with_logits(logit, labels)
    known = invariant >= 0
    if bool(known.any()):
        aux = F.binary_cross_entropy_with_logits(inv_logit[known], invariant[known])
    else:
        aux = logit.new_zeros(())
    # Training batches are (positive, negative) pairs that share a support set.
    if logit.numel() >= 2 and logit.numel() % 2 == 0:
        rank = F.softplus(logit[1::2] - logit[0::2]).mean()
    else:
        rank = logit.new_zeros(())
    return bce + aux_weight * aux + 0.5 * rank, bce.detach(), aux.detach()


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
