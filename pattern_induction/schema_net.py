"""Tiny set encoder that predicts a latent schema from a handful of strings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pattern_induction.schema import (
    CHAR_SLOTS,
    CHARSETS,
    MAX_HI,
    MAX_LEN,
    MAX_LO,
    MAX_PARTS,
    MAX_REP,
    MAX_STRINGS,
    N_KIND,
    PAD,
    VOCAB,
    Part,
    arrays_to_parts,
)

D_MODEL = 96


class SchemaNet(nn.Module):
    def __init__(self, d: int = D_MODEL, heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.d = d
        self.embed = nn.Embedding(VOCAB, d, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_LEN, d)
        self.seg = nn.Embedding(MAX_STRINGS, d)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=heads,
                dim_feedforward=2 * d,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            ),
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.queries = nn.Parameter(torch.randn(MAX_PARTS, d) * 0.02)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.kind = nn.Linear(d, N_KIND)
        self.optional = nn.Linear(d, 2)
        self.charset = nn.Linear(d, len(CHARSETS))
        self.lo = nn.Linear(d, MAX_LO + 1)
        self.hi = nn.Linear(d, MAX_HI + 1)
        self.rep_lo = nn.Linear(d, MAX_REP + 1)
        self.rep_hi = nn.Linear(d, MAX_REP + 1)
        self.delim = nn.Linear(d, VOCAB)
        self.char_in = nn.Embedding(VOCAB, d, padding_idx=PAD)
        self.char_step = nn.GRUCell(d, d)
        self.char_out = nn.Linear(d, VOCAB)

    def encode(
        self,
        tokens: torch.Tensor,
        token_pad: torch.Tensor,
        slot_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return memory (B, S*L, d) and a key padding mask that is True on pads."""
        batch, n_strings, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        segments = torch.arange(n_strings, device=tokens.device)
        hidden = self.embed(tokens) + self.pos(positions)[None, None, :, :] + self.seg(segments)[None, :, None, :]
        flat = hidden.reshape(batch, n_strings * length, self.d)
        char_pad = token_pad.reshape(batch, n_strings * length)
        # An unused string slot is padding even when its first position was unmasked.
        string_pad = slot_pad[:, :, None].expand(batch, n_strings, length).reshape(batch, n_strings * length)
        key_pad = char_pad | string_pad
        all_pad = key_pad.all(dim=1)
        safe = key_pad.clone()
        safe[all_pad, 0] = False
        memory = self.encoder(flat, src_key_padding_mask=safe)
        memory = memory.masked_fill(all_pad[:, None, None], 0.0)
        return memory, safe

    def forward(
        self,
        tokens: torch.Tensor,
        token_pad: torch.Tensor,
        slot_pad: torch.Tensor,
        char_teacher: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        memory, key_pad = self.encode(tokens, token_pad, slot_pad)
        batch = tokens.shape[0]
        query = self.queries[None, :, :].expand(batch, MAX_PARTS, self.d)
        parts, _weights = self.cross(query, memory, memory, key_padding_mask=key_pad, need_weights=False)
        char_logits = self._chars(parts, char_teacher)
        return {
            "kind": self.kind(parts),
            "optional": self.optional(parts),
            "charset": self.charset(parts),
            "lo": self.lo(parts),
            "hi": self.hi(parts),
            "rep_lo": self.rep_lo(parts),
            "rep_hi": self.rep_hi(parts),
            "delim": self.delim(parts),
            "chars": char_logits,
        }

    def _chars(self, parts: torch.Tensor, teacher: torch.Tensor | None) -> torch.Tensor:
        batch, n_parts, _d = parts.shape
        hidden = parts.reshape(batch * n_parts, self.d)
        prev = torch.zeros(batch * n_parts, dtype=torch.long, device=parts.device)
        logits = []
        for slot in range(CHAR_SLOTS):
            hidden = self.char_step(self.char_in(prev), hidden)
            step = self.char_out(hidden)
            logits.append(step)
            if teacher is None:
                prev = step.argmax(dim=-1)
            else:
                prev = teacher[:, :, slot].reshape(batch * n_parts)
        return torch.stack(logits, dim=1).reshape(batch, n_parts, CHAR_SLOTS, VOCAB)


def _active_mask(kinds: torch.Tensor) -> torch.Tensor:
    """True through the first END part. Trailing padding is not a training signal."""
    ended = torch.cumsum((kinds == 0).int(), dim=1) - (kinds == 0).int()
    return ended == 0


def _masked_ce(pred: torch.Tensor, gold: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), gold.reshape(-1), reduction="none")
    weights = valid.reshape(-1).to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp(min=1.0)


def schema_loss(logits: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> torch.Tensor:
    valid = _active_mask(target["kind"])
    char_valid = valid[:, :, None].expand_as(target["chars"])

    def ce(name: str) -> torch.Tensor:
        return _masked_ce(logits[name], target[name], valid)

    return (
        ce("kind")
        + ce("optional")
        + 0.5 * (ce("charset") + ce("lo") + ce("hi") + ce("rep_lo") + ce("rep_hi") + ce("delim"))
        + _masked_ce(logits["chars"], target["chars"], char_valid)
    )


def logits_to_parts(logits: dict[str, torch.Tensor]) -> list[list[Part]]:
    """Greedy schema for each row of a batch."""
    kind = logits["kind"].argmax(dim=-1)
    optional = logits["optional"].argmax(dim=-1)
    charset = logits["charset"].argmax(dim=-1)
    lo = logits["lo"].argmax(dim=-1)
    hi = logits["hi"].argmax(dim=-1)
    rep_lo = logits["rep_lo"].argmax(dim=-1)
    rep_hi = logits["rep_hi"].argmax(dim=-1)
    delim = logits["delim"].argmax(dim=-1)
    chars = logits["chars"].argmax(dim=-1)
    batch = kind.shape[0]
    decoded = []
    for row in range(batch):
        arrays = {
            "kind": kind[row].tolist(),
            "optional": optional[row].tolist(),
            "charset": charset[row].tolist(),
            "lo": lo[row].tolist(),
            "hi": hi[row].tolist(),
            "rep_lo": rep_lo[row].tolist(),
            "rep_hi": rep_hi[row].tolist(),
            "delim": delim[row].tolist(),
            "chars": chars[row].tolist(),
        }
        decoded.append(arrays_to_parts(arrays))
    return decoded


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())
