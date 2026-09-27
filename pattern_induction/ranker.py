"""Tiny candidate ranker from the learnability plan, plus a pooled ablation.

Both models see the visible strings and the candidate regexes. Neither sees
hidden strings. The cross-attention model lets each candidate query the
examples. The pooled model replaces that attention with an MLP of about the
same size applied to the mean of the example vectors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

PAD_ID = 256
VOCAB = 258
MAX_LEN = 128
TAU = 0.05


def encode_utf8(text: str, max_len: int = MAX_LEN) -> tuple[list[int], list[bool], bool]:
    """Return byte ids, a True-where-pad mask, and whether the text was truncated."""
    raw = text.encode("utf-8", errors="replace")
    truncated = len(raw) > max_len
    raw = raw[:max_len]
    pad_len = max_len - len(raw)
    ids = list(raw) + [PAD_ID] * pad_len
    pad = [False] * len(raw) + [True] * pad_len
    if not raw:
        pad = [True] * max_len
    return ids, pad, truncated


class TinyRegexRanker(nn.Module):
    def __init__(
        self,
        cross_attention: bool = True,
        d: int = 96,
        heads: int = 4,
        string_layers: int = 2,
        regex_layers: int = 1,
        vocab: int = VOCAB,
        max_len: int = MAX_LEN,
    ) -> None:
        super().__init__()
        self.cross_attention = cross_attention
        self.d = d
        self.max_len = max_len
        self.embed = nn.Embedding(vocab, d, padding_idx=PAD_ID)
        self.pos = nn.Embedding(max_len, d)
        self.string_encoder = nn.TransformerEncoder(
            _layer(d, heads),
            num_layers=string_layers,
            enable_nested_tensor=False,
        )
        self.regex_encoder = nn.TransformerEncoder(
            _layer(d, heads),
            num_layers=regex_layers,
            enable_nested_tensor=False,
        )
        if cross_attention:
            self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.pool_mix = None
        else:
            self.cross_attn = None
            target = sum(p.numel() for p in nn.MultiheadAttention(d, heads, batch_first=True).parameters())
            hidden = max(d, round((target - d) / (2 * d + 1)))
            self.pool_mix = nn.Sequential(
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.Linear(hidden, d),
            )
        self.score = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def encode(self, tokens: torch.Tensor, token_pad: torch.Tensor, encoder: nn.TransformerEncoder) -> torch.Tensor:
        """Mean-pool a batch of byte sequences. token_pad is True on padding."""
        _batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        hidden = self.embed(tokens) + self.pos(positions)[None, :, :]
        all_pad = token_pad.all(dim=1)
        safe_pad = token_pad.clone()
        safe_pad[all_pad, 0] = False
        hidden = encoder(hidden, src_key_padding_mask=safe_pad)
        valid = (~token_pad).unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return pooled.masked_fill(all_pad.unsqueeze(-1), 0.0)

    def forward(
        self,
        strings: torch.Tensor,
        string_pad: torch.Tensor,
        string_slot_pad: torch.Tensor,
        candidates: torch.Tensor,
        cand_pad: torch.Tensor,
        cand_slot_pad: torch.Tensor,
    ) -> torch.Tensor:
        """Score each candidate. Unused candidate slots are filled with a large negative."""
        batch, n_strings, length = strings.shape
        n_cands = candidates.shape[1]
        string_vec = self.encode(
            strings.reshape(batch * n_strings, length),
            string_pad.reshape(batch * n_strings, length),
            self.string_encoder,
        ).reshape(batch, n_strings, self.d)
        regex_vec = self.encode(
            candidates.reshape(batch * n_cands, length),
            cand_pad.reshape(batch * n_cands, length),
            self.regex_encoder,
        ).reshape(batch, n_cands, self.d)

        if self.cross_attention:
            query = regex_vec.reshape(batch * n_cands, 1, self.d)
            examples = string_vec[:, None, :, :].expand(batch, n_cands, n_strings, self.d)
            examples = examples.reshape(batch * n_cands, n_strings, self.d)
            slot_pad = string_slot_pad[:, None, :].expand(batch, n_cands, n_strings)
            slot_pad = slot_pad.reshape(batch * n_cands, n_strings).contiguous()
            all_pad = slot_pad.all(dim=1)
            safe_pad = slot_pad.clone()
            safe_pad[all_pad, 0] = False
            context, _weights = self.cross_attn(
                query,
                examples,
                examples,
                key_padding_mask=safe_pad,
                need_weights=False,
            )
            context = context[:, 0, :]
            context = context.masked_fill(all_pad.unsqueeze(-1), 0.0)
            candidate = query[:, 0, :]
        else:
            valid = (~string_slot_pad).unsqueeze(-1).to(string_vec.dtype)
            pooled = (string_vec * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
            mixed = self.pool_mix(pooled)
            context = mixed[:, None, :].expand(batch, n_cands, self.d).reshape(batch * n_cands, self.d)
            candidate = regex_vec.reshape(batch * n_cands, self.d)

        features = torch.cat([candidate, context, candidate * context], dim=-1)
        scores = self.score(features).reshape(batch, n_cands)
        return scores.masked_fill(cand_slot_pad, -1.0e9)


def _layer(d: int, heads: int) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=d,
        nhead=heads,
        dim_feedforward=2 * d,
        dropout=0.0,
        batch_first=True,
        norm_first=True,
        activation="gelu",
    )


def soft_cross_entropy(logits: torch.Tensor, accuracy: torch.Tensor, valid: torch.Tensor, tau: float = TAU) -> torch.Tensor:
    """Cross-entropy against a softmax of hidden semantic accuracy."""
    filled = accuracy.masked_fill(~valid, -1.0e4)
    target = torch.softmax(filled / tau, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


def collate_tasks(rows: list[dict], max_len: int = MAX_LEN) -> dict[str, torch.Tensor]:
    batch = len(rows)
    n_strings = max(len(row["string_ids"]) for row in rows)
    n_cands = max(len(row["cand_ids"]) for row in rows)
    strings = torch.full((batch, n_strings, max_len), PAD_ID, dtype=torch.long)
    string_pad = torch.ones((batch, n_strings, max_len), dtype=torch.bool)
    string_slot_pad = torch.ones((batch, n_strings), dtype=torch.bool)
    candidates = torch.full((batch, n_cands, max_len), PAD_ID, dtype=torch.long)
    cand_pad = torch.ones((batch, n_cands, max_len), dtype=torch.bool)
    cand_slot_pad = torch.ones((batch, n_cands), dtype=torch.bool)
    accuracy = torch.zeros((batch, n_cands), dtype=torch.float32)
    for index, row in enumerate(rows):
        for slot, (ids, pad) in enumerate(zip(row["string_ids"], row["string_pad"])):
            strings[index, slot] = torch.tensor(ids, dtype=torch.long)
            string_pad[index, slot] = torch.tensor(pad, dtype=torch.bool)
            string_slot_pad[index, slot] = False
        for slot, (ids, pad) in enumerate(zip(row["cand_ids"], row["cand_pad"])):
            candidates[index, slot] = torch.tensor(ids, dtype=torch.long)
            cand_pad[index, slot] = torch.tensor(pad, dtype=torch.bool)
            cand_slot_pad[index, slot] = False
            accuracy[index, slot] = row["accuracies"][slot]
    return {
        "strings": strings,
        "string_pad": string_pad,
        "string_slot_pad": string_slot_pad,
        "candidates": candidates,
        "cand_pad": cand_pad,
        "cand_slot_pad": cand_slot_pad,
        "accuracy": accuracy,
    }


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())
