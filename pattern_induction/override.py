"""Residual override model.

The old code-length rule picks R0. The model scores each alternative by the
hidden-accuracy gap Δ = A(Ri) - A(R0) and may replace R0 only when the best
predicted gap clears a threshold. A false override is penalized more than a
missed improvement.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pattern_induction.ranker import MAX_LEN, PAD_ID, VOCAB, encode_utf8
from pattern_induction.selector import BASE_FEATURES, raw_features

IMPROVE_EPS = 0.01
# Optimistic mistakes on pairs that do not improve R0 are weighted by this factor.
DEFAULT_LAMBDA = 5.0


def pair_feature_names() -> tuple[str, ...]:
    left = tuple(f"alt_{name}" for name in BASE_FEATURES)
    right = tuple(f"rule_{name}" for name in BASE_FEATURES)
    diff = tuple(f"diff_{name}" for name in BASE_FEATURES)
    return left + right + diff


PAIR_FEATURES = pair_feature_names()


def _clip(value: float) -> float:
    if value > 1.0e4:
        return 1.0e4
    if value < -1.0e4:
        return -1.0e4
    return value


def pair_rows(alt_patterns: list[str], rule: str, heuristic: str, strings: list[str]) -> list[list[float]]:
    """Visible features of each alternative relative to the old rule's choice."""
    needed = set(alt_patterns)
    needed.add(rule)
    raw = {pattern: raw_features(pattern, heuristic, strings) for pattern in needed}
    rule_raw = raw[rule]
    rows = []
    for pattern in alt_patterns:
        alt = raw[pattern]
        vector = [_clip(alt[name]) for name in BASE_FEATURES]
        vector.extend(_clip(rule_raw[name]) for name in BASE_FEATURES)
        vector.extend(_clip(alt[name] - rule_raw[name]) for name in BASE_FEATURES)
        rows.append(vector)
    return rows


class ResidualOverride(nn.Module):
    """Tiny encoder over the visible strings, R0, and one alternative."""

    def __init__(
        self,
        cross_attention: bool = True,
        d: int = 128,
        heads: int = 4,
        string_layers: int = 2,
        regex_layers: int = 1,
        n_features: int = len(PAIR_FEATURES),
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
            target = sum(parameter.numel() for parameter in nn.MultiheadAttention(d, heads, batch_first=True).parameters())
            hidden = max(d, round((target - d) / (2 * d + 1)))
            self.pool_mix = nn.Sequential(
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.Linear(hidden, d),
            )
        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, d),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(6 * d, d),
            nn.GELU(),
            nn.Linear(d, 2),
        )

    def encode(self, tokens: torch.Tensor, token_pad: torch.Tensor, encoder: nn.TransformerEncoder) -> torch.Tensor:
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

    def _context(
        self,
        query: torch.Tensor,
        strings: torch.Tensor,
        string_slot_pad: torch.Tensor,
    ) -> torch.Tensor:
        if self.cross_attention:
            all_pad = string_slot_pad.all(dim=1)
            safe_pad = string_slot_pad.clone()
            safe_pad[all_pad, 0] = False
            context, _weights = self.cross_attn(
                query,
                strings,
                strings,
                key_padding_mask=safe_pad,
                need_weights=False,
            )
            return context.masked_fill(all_pad[:, None, None], 0.0)
        valid = (~string_slot_pad).unsqueeze(-1).to(strings.dtype)
        pooled = (strings * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.pool_mix(pooled)

    def forward(
        self,
        strings: torch.Tensor,
        string_pad: torch.Tensor,
        string_slot_pad: torch.Tensor,
        rule: torch.Tensor,
        rule_pad: torch.Tensor,
        alts: torch.Tensor,
        alt_pad: torch.Tensor,
        alt_slot_pad: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predicted Δ and an improvement logit for each alternative."""
        batch, n_strings, length = strings.shape
        n_alts = alts.shape[1]
        string_vec = self.encode(
            strings.reshape(batch * n_strings, length),
            string_pad.reshape(batch * n_strings, length),
            self.string_encoder,
        ).reshape(batch, n_strings, self.d)
        rule_vec = self.encode(rule, rule_pad, self.regex_encoder)
        alt_vec = self.encode(
            alts.reshape(batch * n_alts, length),
            alt_pad.reshape(batch * n_alts, length),
            self.regex_encoder,
        ).reshape(batch, n_alts, self.d)
        if self.cross_attention:
            ctx_rule = self._context(rule_vec[:, None, :], string_vec, string_slot_pad)
            ctx_alt = self._context(alt_vec, string_vec, string_slot_pad)
            ctx_rule = ctx_rule.expand(batch, n_alts, self.d)
        else:
            mixed = self._context(rule_vec[:, None, :], string_vec, string_slot_pad)
            ctx_rule = mixed[:, None, :].expand(batch, n_alts, self.d)
            ctx_alt = ctx_rule
        feat = self.feature_proj(features)
        rule_exp = rule_vec[:, None, :].expand(batch, n_alts, self.d)
        packed = torch.cat(
            [alt_vec, rule_exp, alt_vec - rule_exp, ctx_alt, ctx_rule, feat],
            dim=-1,
        )
        out = self.head(packed)
        delta = out[..., 0].masked_fill(alt_slot_pad, -1.0e9)
        logit = out[..., 1].masked_fill(alt_slot_pad, 0.0)
        return delta, logit


def override_loss(
    delta: torch.Tensor,
    logit: torch.Tensor,
    actual: torch.Tensor,
    valid: torch.Tensor,
    lam: float,
    tau: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L = L_ranking + λ L_false_override.

    Ranking fits Δ and orders the alternatives. The false-override term is the
    same regression on pairs that do not beat R0, plus the negative class of
    the improvement probability, both weighted by λ.
    """
    mask = valid.to(delta.dtype)
    denom = mask.sum().clamp(min=1.0)
    negative = valid & (actual <= 0)
    positive = valid & (actual > IMPROVE_EPS)
    huber = F.smooth_l1_loss(delta, actual, reduction="none")
    # Pairs that do not beat R0 contribute λ times as much as pairs that do.
    huber_weight = torch.where(negative, torch.full_like(delta, lam), torch.ones_like(delta))
    weighted = (huber * huber_weight * mask).sum() / denom
    ranking_huber = (huber * mask).sum() / denom

    filled_pred = delta.masked_fill(~valid, -1.0e4)
    filled_actual = actual.masked_fill(~valid, -1.0e4)
    row_ok = valid.any(dim=-1)
    log_probs = F.log_softmax(filled_pred, dim=-1)
    target = torch.softmax(filled_actual / tau, dim=-1)
    listwise_rows = -(target * log_probs).sum(dim=-1)
    listwise = listwise_rows[row_ok].mean() if bool(row_ok.any()) else delta.new_zeros(())

    improve = (actual > IMPROVE_EPS).to(delta.dtype)
    bce = F.binary_cross_entropy_with_logits(logit, improve, reduction="none")
    bce_weight = torch.where(negative, torch.full_like(delta, lam), torch.ones_like(delta))
    bce_loss = (bce * bce_weight * mask).sum() / denom
    bce_rank = (bce * mask).sum() / denom

    false_extra = (weighted - ranking_huber) + (bce_loss - bce_rank)
    ranking = ranking_huber + listwise + bce_rank
    loss = ranking + false_extra
    parts = {
        "loss": float(loss.detach()),
        "ranking": float(ranking.detach()),
        "false_override": float(false_extra.detach()),
    }
    return loss, parts


def choose_alternative(deltas: list[float], threshold: float) -> int | None:
    """Index of the override, or None when R0 stays."""
    if not deltas:
        return None
    best = max(range(len(deltas)), key=lambda index: deltas[index])
    if deltas[best] > threshold:
        return best
    return None


def collate_pairs(
    rows: list[dict],
    mean: list[float],
    scale: list[float],
    max_len: int = MAX_LEN,
) -> dict[str, torch.Tensor]:
    batch = len(rows)
    n_strings = max(len(row["string_ids"]) for row in rows)
    n_alts = max(max(len(row["alt_ids"]), 1) for row in rows)
    strings = torch.full((batch, n_strings, max_len), PAD_ID, dtype=torch.long)
    string_pad = torch.ones((batch, n_strings, max_len), dtype=torch.bool)
    string_slot_pad = torch.ones((batch, n_strings), dtype=torch.bool)
    rule = torch.full((batch, max_len), PAD_ID, dtype=torch.long)
    rule_pad = torch.ones((batch, max_len), dtype=torch.bool)
    alts = torch.full((batch, n_alts, max_len), PAD_ID, dtype=torch.long)
    alt_pad = torch.ones((batch, n_alts, max_len), dtype=torch.bool)
    alt_slot_pad = torch.ones((batch, n_alts), dtype=torch.bool)
    features = torch.zeros((batch, n_alts, len(mean)), dtype=torch.float32)
    actual = torch.zeros((batch, n_alts), dtype=torch.float32)
    for index, row in enumerate(rows):
        for slot, (ids, pad) in enumerate(zip(row["string_ids"], row["string_pad"])):
            strings[index, slot] = torch.tensor(ids, dtype=torch.long)
            string_pad[index, slot] = torch.tensor(pad, dtype=torch.bool)
            string_slot_pad[index, slot] = False
        rule[index] = torch.tensor(row["rule_ids"], dtype=torch.long)
        rule_pad[index] = torch.tensor(row["rule_pad"], dtype=torch.bool)
        for slot, (ids, pad, vector, delta) in enumerate(
            zip(row["alt_ids"], row["alt_pad"], row["features"], row["alt_delta"])
        ):
            alts[index, slot] = torch.tensor(ids, dtype=torch.long)
            alt_pad[index, slot] = torch.tensor(pad, dtype=torch.bool)
            alt_slot_pad[index, slot] = False
            scaled = [(value - center) / width for value, center, width in zip(vector, mean, scale)]
            features[index, slot] = torch.tensor(scaled, dtype=torch.float32)
            actual[index, slot] = delta
    return {
        "strings": strings,
        "string_pad": string_pad,
        "string_slot_pad": string_slot_pad,
        "rule": rule,
        "rule_pad": rule_pad,
        "alts": alts,
        "alt_pad": alt_pad,
        "alt_slot_pad": alt_slot_pad,
        "features": features,
        "actual": actual,
    }


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


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
