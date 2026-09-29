"""Span hierarchy on top of frozen relational character features.

Characters are still a structural class, a task-local symbol id, and the
alignment relations from the renaming model. No codepoint embedding is added.
Class runs, and the adjacent pair of two runs, become spans. A second
transformer relates those spans. Membership reads the query. Auxiliary heads
read the generator's span id, kind, boundaries, and operator roles.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from pattern_induction.compose import MAX_GOLD_SPANS, N_KIND, AnnotatedEpisode
from pattern_induction.renaming import MAX_SYMBOLS, N_CLASS, N_REL, featurize

N_SPAN_FEAT = 8
SPAN_CLASS_L = 0
SPAN_CLASS_R = 1
SPAN_LENGTH = 2
SPAN_IS_PAIR = 3
SPAN_EQ_PREV_CLASS = 4
SPAN_EQ_PREV_LEN = 5
SPAN_EQ_BACK_CLASS = 6
SPAN_EQ_BACK_LEN = 7

MAX_LEN = 32
MAX_SUPPORT = 4
MAX_SPANS = 24


def _sinusoid(length: int, d_model: int) -> torch.Tensor:
    position = torch.arange(length).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    angles = position * div
    table = torch.zeros(length, d_model)
    table[:, 0::2] = torch.sin(angles)
    table[:, 1::2] = torch.cos(angles)
    return table


def _encoder(d_model: int, heads: int, ff: int, dropout: float, layers: int) -> nn.TransformerEncoder:
    block = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=heads,
        dim_feedforward=ff,
        dropout=dropout,
        batch_first=True,
        activation="gelu",
        norm_first=True,
    )
    return nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)


def propose_spans(
    classes: torch.Tensor,
    pad: torch.Tensor,
    max_spans: int = MAX_SPANS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Class runs, then each adjacent pair of runs.

    Positions restart at zero for pairs, so a longer repeat does not push the
    first pair onto a new absolute index merely because there are more runs.
    """
    batch, slots, length = classes.shape
    device = classes.device
    start = torch.zeros(batch, slots, max_spans, dtype=torch.long, device=device)
    end = torch.zeros(batch, slots, max_spans, dtype=torch.long, device=device)
    feat = torch.zeros(batch, slots, max_spans, N_SPAN_FEAT, device=device)
    span_pad = torch.ones(batch, slots, max_spans, dtype=torch.bool, device=device)
    pos_index = torch.zeros(batch, slots, max_spans, dtype=torch.long, device=device)
    cls_list = classes.detach().tolist()
    pad_list = pad.detach().tolist()
    scale = float(max(length, 1))
    for row in range(batch):
        for slot in range(slots):
            runs: list[tuple[int, int, int]] = []
            index = 0
            row_cls = cls_list[row][slot]
            row_pad = pad_list[row][slot]
            while index < length and not row_pad[index]:
                stop = index + 1
                while stop < length and not row_pad[stop] and row_cls[stop] == row_cls[index]:
                    stop += 1
                runs.append((index, stop, int(row_cls[index])))
                index = stop
            run_sig = [(kind, kind, right - left) for left, right, kind in runs]
            records: list[tuple] = []
            for cursor, (left, right, kind) in enumerate(runs):
                records.append(
                    (
                        left,
                        right,
                        kind,
                        kind,
                        right - left,
                        0,
                        cursor,
                        _match(run_sig[cursor - 1] if cursor else None, run_sig[cursor]),
                        _match(run_sig[cursor - 2] if cursor >= 2 else None, run_sig[cursor]),
                    )
                )
            pair_sig: list[tuple[int, int, int]] = []
            for cursor in range(len(runs) - 1):
                left, _mid, kind_l = runs[cursor]
                _mid2, right, kind_r = runs[cursor + 1]
                pair_sig.append((kind_l, kind_r, right - left))
                signature = pair_sig[-1]
                records.append(
                    (
                        left,
                        right,
                        kind_l,
                        kind_r,
                        right - left,
                        1,
                        cursor,
                        _match(pair_sig[cursor - 1] if cursor else None, signature),
                        _match(pair_sig[cursor - 2] if cursor >= 2 else None, signature),
                    )
                )
            for span_i, record in enumerate(records[:max_spans]):
                left, right, kind_l, kind_r, span_len, is_pair, cursor, prev_eq, back_eq = record
                start[row, slot, span_i] = left
                end[row, slot, span_i] = right
                span_pad[row, slot, span_i] = False
                pos_index[row, slot, span_i] = min(cursor, max_spans - 1)
                feat[row, slot, span_i, SPAN_CLASS_L] = kind_l / 4.0
                feat[row, slot, span_i, SPAN_CLASS_R] = kind_r / 4.0
                feat[row, slot, span_i, SPAN_LENGTH] = span_len / scale
                feat[row, slot, span_i, SPAN_IS_PAIR] = float(is_pair)
                feat[row, slot, span_i, SPAN_EQ_PREV_CLASS] = prev_eq[0]
                feat[row, slot, span_i, SPAN_EQ_PREV_LEN] = prev_eq[1]
                feat[row, slot, span_i, SPAN_EQ_BACK_CLASS] = back_eq[0]
                feat[row, slot, span_i, SPAN_EQ_BACK_LEN] = back_eq[1]
    return start, end, feat, span_pad, pos_index


def _match(other: tuple[int, int, int] | None, current: tuple[int, int, int]) -> tuple[float, float]:
    if other is None:
        return 0.0, 0.0
    return float(other[0] == current[0] and other[1] == current[1]), float(other[2] == current[2])


def _pool_spans(hidden: torch.Tensor, start: torch.Tensor, end: torch.Tensor, span_pad: torch.Tensor) -> torch.Tensor:
    _batch, _slots, length, _dim = hidden.shape
    index = torch.arange(length, device=hidden.device)
    mask = (index[None, None, None, :] >= start[:, :, :, None]) & (index[None, None, None, :] < end[:, :, :, None])
    mask = mask & ~span_pad[:, :, :, None]
    weights = mask.float()
    denom = weights.sum(dim=-1).clamp(min=1.0)
    pooled = torch.einsum("bsld,bsml->bsmd", hidden, weights) / denom.unsqueeze(-1)
    return pooled * (~span_pad).unsqueeze(-1).float()


def _mean_max(hidden: torch.Tensor, pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    live = (~pad).unsqueeze(-1).float()
    mean = (hidden * live).sum(dim=1) / live.sum(dim=1).clamp(min=1.0)
    filled = hidden.masked_fill(pad.unsqueeze(-1), torch.finfo(hidden.dtype).min)
    maxed = filled.max(dim=1).values
    maxed = torch.where(live.sum(dim=1) > 0, maxed, torch.zeros_like(maxed))
    return mean, maxed


def _attend(encoder: nn.TransformerEncoder, hidden: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    batch, slots, count, dim = hidden.shape
    flat = hidden.reshape(batch, slots * count, dim)
    key_pad = pad.reshape(batch, slots * count)
    if bool(key_pad.all(dim=1).any()):
        key_pad = key_pad.clone()
        key_pad[key_pad.all(dim=1), 0] = False
    encoded = encoder(flat, src_key_padding_mask=key_pad)
    return encoded.reshape(batch, slots, count, dim)


class SpanNet(nn.Module):
    def __init__(
        self,
        d_model: int = 160,
        char_layers: int = 2,
        span_layers: int = 2,
        heads: int = 4,
        ff: int = 320,
        max_len: int = MAX_LEN,
        max_support: int = MAX_SUPPORT,
        max_spans: int = MAX_SPANS,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.max_len = max_len
        self.max_support = max_support
        self.max_spans = max_spans
        self.d_model = d_model
        self.class_emb = nn.Embedding(N_CLASS, d_model)
        self.local_emb = nn.Embedding(MAX_SYMBOLS + 1, d_model, padding_idx=0)
        self.rel_proj = nn.Linear(N_REL, d_model)
        self.role = nn.Embedding(2, d_model)
        self.which = nn.Embedding(max_support + 1, d_model)
        self.span_proj = nn.Linear(N_SPAN_FEAT, d_model)
        self.register_buffer("char_pos", _sinusoid(max_len, d_model), persistent=False)
        self.register_buffer("span_pos", _sinusoid(max_spans, d_model), persistent=False)
        self.char_encoder = _encoder(d_model, heads, ff, dropout, char_layers)
        self.span_encoder = _encoder(d_model, heads, ff, dropout, span_layers)
        self.mem = nn.Sequential(nn.Linear(d_model * 4, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.kind_head = nn.Linear(d_model, N_KIND)
        self.span_head = nn.Linear(d_model, MAX_GOLD_SPANS)
        self.bound_head = nn.Linear(d_model, 2)
        self.role_head = nn.Linear(d_model, 3)

    def forward(
        self,
        local: torch.Tensor,
        classes: torch.Tensor,
        rel: torch.Tensor,
        pad: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _batch, slots, length = local.shape
        roles = torch.zeros(slots, dtype=torch.long, device=local.device)
        roles[-1] = 1
        which = torch.arange(slots, device=local.device)
        hidden = self.class_emb(classes) + self.local_emb(local) + self.rel_proj(rel)
        hidden = hidden + self.char_pos[:length][None, None, :, :]
        hidden = hidden + self.role(roles)[None, :, None, :] + self.which(which)[None, :, None, :]
        hidden = hidden * (~pad).unsqueeze(-1).float()
        encoded = _attend(self.char_encoder, hidden, pad)

        start, end, feat, span_pad, pos_index = propose_spans(classes, pad, self.max_spans)
        pooled = _pool_spans(encoded, start, end, span_pad)
        span_h = pooled + self.span_proj(feat) + self.span_pos[pos_index]
        span_h = span_h * (~span_pad).unsqueeze(-1).float()
        span_h = _attend(self.span_encoder, span_h, span_pad)

        char_mean, char_max = _mean_max(encoded[:, -1], pad[:, -1])
        span_mean, span_max = _mean_max(span_h[:, -1], span_pad[:, -1])
        latent = torch.cat([char_mean, char_max, span_mean, span_max], dim=-1)
        aux = {
            "kind": self.kind_head(encoded),
            "span": self.span_head(encoded),
            "bound": self.bound_head(encoded),
            "role": self.role_head(encoded),
        }
        return self.mem(latent).squeeze(-1), aux


def collate(
    episodes: list[AnnotatedEpisode],
    max_len: int = MAX_LEN,
    max_support: int = MAX_SUPPORT,
) -> tuple[torch.Tensor, ...]:
    local, classes, rel, pad = featurize(
        [episode.as_episode() for episode in episodes],
        max_len=max_len,
        max_support=max_support,
    )
    batch = len(episodes)
    slots = max_support + 1
    labels = torch.tensor([float(episode.label) for episode in episodes])
    span_id = torch.full((batch, slots, max_len), -1, dtype=torch.long)
    kind = torch.full((batch, slots, max_len), -1, dtype=torch.long)
    bound = torch.zeros(batch, slots, max_len, 2)
    role = torch.zeros(batch, slots, max_len, 3)
    mask = torch.zeros(batch, slots, max_len, dtype=torch.bool)

    def write(row: int, slot: int, text: str, ann) -> None:
        if ann is None:
            return
        for index, _ch in enumerate(text[:max_len]):
            if ann.span[index] < 0:
                continue
            span_id[row, slot, index] = ann.span[index]
            kind[row, slot, index] = ann.kind[index]
            bound[row, slot, index, 0] = float(ann.start[index])
            bound[row, slot, index, 1] = float(ann.end[index])
            role[row, slot, index, 0] = float(ann.role[index][0])
            role[row, slot, index, 1] = float(ann.role[index][1])
            role[row, slot, index, 2] = float(ann.role[index][2])
            mask[row, slot, index] = True

    for row, episode in enumerate(episodes):
        support = episode.support[:max_support]
        for slot, text in enumerate(support):
            write(row, slot, text, episode.support_ann[slot])
        write(row, slots - 1, episode.query, episode.query_ann)
    return local, classes, rel, pad, labels, span_id, kind, bound, role, mask


def composition_loss(
    logit: torch.Tensor,
    aux: dict[str, torch.Tensor],
    labels: torch.Tensor,
    span_id: torch.Tensor,
    kind: torch.Tensor,
    bound: torch.Tensor,
    role: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Membership stays the main term. Span labels are a denser hint, not a decoder."""
    membership = F.binary_cross_entropy_with_logits(logit, labels)
    positive = logit[labels > 0.5]
    negative = logit[labels < 0.5]
    if positive.numel() and negative.numel():
        rank = F.softplus(negative[:, None] - positive[None, :]).mean()
    else:
        rank = logit.new_zeros(())
    membership = membership + 0.5 * rank
    kind_loss = F.cross_entropy(aux["kind"].reshape(-1, N_KIND), kind.reshape(-1), ignore_index=-1)
    span_loss = F.cross_entropy(aux["span"].reshape(-1, MAX_GOLD_SPANS), span_id.reshape(-1), ignore_index=-1)
    if bool(mask.any()):
        bound_weight = torch.full((2,), 3.0, device=logit.device)
        role_weight = torch.full((3,), 3.0, device=logit.device)
        bound_loss = F.binary_cross_entropy_with_logits(aux["bound"][mask], bound[mask], pos_weight=bound_weight)
        role_loss = F.binary_cross_entropy_with_logits(aux["role"][mask], role[mask], pos_weight=role_weight)
    else:
        bound_loss = logit.new_zeros(())
        role_loss = logit.new_zeros(())
    total = membership + 0.3 * kind_loss + 0.3 * span_loss + 0.2 * bound_loss + 0.2 * role_loss
    return total, membership.detach(), (kind_loss + span_loss).detach()


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
