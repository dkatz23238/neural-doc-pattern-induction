"""Scope relations over a frozen local recognizer.

The chart stays as it is. These heads name relations the generator already
commits to: the same repeated unit, containment, the start and end of a group,
an optional child, an alternation branch, and a shared prefix or suffix.
Recall at 8 is the objective. A second bound rides along with each strong
class or repeat span, and the alternation score is the frozen bilinear plus a
residual on those relations.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from pattern_induction.chart import LOCAL_K, Hyp, gold_facts
from pattern_induction.local_net import (
    LocalNet,
    _bce,
    _bilinear,
    _ids,
    _span_tensors,
    collate_local,
    hypotheses_from_batch,
)
from pattern_induction.program import MAX_LEN, MAX_SUPPORT, Task

MAX_REL = 48
N_GEOM = 16
PAIR_NAMES = ("same_unit", "contains", "alt_branch", "shared_prefix", "shared_suffix")
UNARY_NAMES = ("starts_group", "ends_group", "optional_child")
ALT_CHANNEL = PAIR_NAMES.index("alt_branch")
SAME_UNIT = PAIR_NAMES.index("same_unit")
CONTAINS = PAIR_NAMES.index("contains")
SHARED_PREFIX = PAIR_NAMES.index("shared_prefix")
SHARED_SUFFIX = PAIR_NAMES.index("shared_suffix")


def total_parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def _supports(task: Task) -> list[str]:
    return [text[:MAX_LEN] for text in task.supports[:MAX_SUPPORT]]


def _signature(ids: list[int], start: int, end: int) -> int | None:
    chunk = ids[start:end]
    if not chunk or any(item != chunk[0] for item in chunk):
        return None
    return chunk[0]


def geometry(
    ids: list[list[int]],
    lengths: list[int],
    slot_l: int,
    start_l: int,
    end_l: int,
    slot_r: int,
    start_r: int,
    end_r: int,
) -> list[float]:
    """Sixteen features from class ids and span indices."""
    width_l = end_l - start_l
    width_r = end_r - start_r
    left_sig = _signature(ids[slot_l], start_l, end_l) if slot_l < len(ids) else None
    right_sig = _signature(ids[slot_r], start_r, end_r) if slot_r < len(ids) else None
    shared = 0.0
    if start_l == start_r and start_l > 0 and slot_l < len(ids) and slot_r < len(ids):
        if ids[slot_l][:start_l] == ids[slot_r][:start_r] and ids[slot_l][:start_l]:
            shared = 1.0
    return [
        float(slot_l == slot_r),
        float(end_l == start_r or end_r == start_l),
        float(start_l <= start_r and end_r <= end_l and (start_l, end_l) != (start_r, end_r) and width_r > 0),
        float(start_r <= start_l and end_l <= end_r and (start_l, end_l) != (start_r, end_r) and width_l > 0),
        float(start_l == start_r),
        float(end_l == end_r),
        float(width_l == width_r and width_l > 0),
        float(start_l == 0 and width_l > 0),
        float(start_r == 0 and width_r > 0),
        float(slot_l < len(lengths) and end_l == lengths[slot_l] and width_l > 0),
        float(slot_r < len(lengths) and end_r == lengths[slot_r] and width_r > 0),
        float(left_sig is not None and left_sig == right_sig),
        float(left_sig is not None and right_sig is not None and left_sig != right_sig),
        shared,
        width_l / 8.0,
        width_r / 8.0,
    ]


def _span_index(spans: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {span: cursor for cursor, span in enumerate(spans)}


def _add_label(labels: dict[tuple[int, int, int, int], set[int]], slot_l: int, left: int, slot_r: int, right: int, channel: int) -> None:
    if left < 0 or right < 0:
        return
    labels.setdefault((slot_l, left, slot_r, right), set()).add(channel)


def _gold_labels(facts: list[Hyp], spans: list[list[tuple[int, int]]], lengths: list[int]) -> tuple[dict[tuple[int, int, int, int], set[int]], list[list[list[int]]]]:
    indexes = [_span_index(slot_spans) for slot_spans in spans]
    labels: dict[tuple[int, int, int, int], set[int]] = {}
    unary = [[[0, 0, 0] for _ in slot_spans] for slot_spans in spans]

    def at(slot: int, start: int, end: int) -> int:
        if slot < 0 or slot >= len(indexes):
            return -1
        return indexes[slot].get((start, end), -1)

    for fact in facts:
        if fact.kind == "pair":
            _add_label(labels, fact.slot, at(fact.slot, fact.start, fact.end), fact.slot, at(fact.slot, fact.start2, fact.end2), SAME_UNIT)
    repeats = [fact for fact in facts if fact.kind == "repeat"]
    for left_fact in repeats:
        for right_fact in repeats:
            if left_fact.slot == right_fact.slot:
                continue
            left = at(left_fact.slot, left_fact.start, left_fact.end)
            right = at(right_fact.slot, right_fact.start, right_fact.end)
            _add_label(labels, left_fact.slot, left, right_fact.slot, right, SAME_UNIT)
    outers = [fact for fact in facts if fact.kind in {"repeat", "optional"}]
    inners = [fact for fact in facts if fact.kind in {"cls", "copy"}]
    for outer in outers:
        for inner in inners:
            if inner.slot != outer.slot:
                continue
            if outer.start <= inner.start and inner.end <= outer.end and (outer.start, outer.end) != (inner.start, inner.end):
                _add_label(labels, outer.slot, at(outer.slot, outer.start, outer.end), inner.slot, at(inner.slot, inner.start, inner.end), CONTAINS)
    for fact in facts:
        if fact.kind == "alt":
            _add_label(labels, fact.slot, at(fact.slot, fact.start, fact.end), fact.slot2, at(fact.slot2, fact.start2, fact.end2), ALT_CHANNEL)
    prefixes = [fact for fact in facts if fact.kind == "copy" and fact.start == 0 and fact.end > fact.start]
    for left_fact in prefixes:
        for right_fact in prefixes:
            if left_fact.slot == right_fact.slot or left_fact.end != right_fact.end:
                continue
            _add_label(
                labels,
                left_fact.slot,
                at(left_fact.slot, left_fact.start, left_fact.end),
                right_fact.slot,
                at(right_fact.slot, right_fact.start, right_fact.end),
                SHARED_PREFIX,
            )
    suffixes = [
        fact
        for fact in facts
        if fact.kind == "copy" and fact.slot < len(lengths) and fact.end == lengths[fact.slot] and fact.end > fact.start
    ]
    for left_fact in suffixes:
        for right_fact in suffixes:
            if left_fact.slot == right_fact.slot or (left_fact.end - left_fact.start) != (right_fact.end - right_fact.start):
                continue
            _add_label(
                labels,
                left_fact.slot,
                at(left_fact.slot, left_fact.start, left_fact.end),
                right_fact.slot,
                at(right_fact.slot, right_fact.start, right_fact.end),
                SHARED_SUFFIX,
            )
    for fact in facts:
        slots = []
        if fact.kind in {"repeat", "optional"}:
            slots.append((fact.slot, fact.start, fact.end, fact.kind == "optional"))
        elif fact.kind == "alt":
            slots.append((fact.slot, fact.start, fact.end, False))
            slots.append((fact.slot2, fact.start2, fact.end2, False))
        for slot, start, end, optional in slots:
            cursor = at(slot, start, end)
            if cursor < 0 or slot >= len(unary) or cursor >= len(unary[slot]):
                continue
            unary[slot][cursor][0] = 1
            unary[slot][cursor][1] = 1
            if optional:
                unary[slot][cursor][2] = 1
    return labels, unary


def _consider(order: list[tuple[int, int, int, int]], seen: set[tuple[int, int, int, int]], key: tuple[int, int, int, int]) -> None:
    if len(order) >= MAX_REL or key in seen:
        return
    if min(key) < 0:
        return
    seen.add(key)
    order.append(key)


def _candidate_order(labels: dict[tuple[int, int, int, int], set[int]], spans: list[list[tuple[int, int]]], lengths: list[int]) -> list[tuple[int, int, int, int]]:
    order: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for key in labels:
        _consider(order, seen, key)
    slots = len(spans)
    for slot in range(slots):
        for other in range(slots):
            if other == slot:
                continue
            by_start = {}
            for cursor, span in enumerate(spans[other]):
                by_start.setdefault(span[0], cursor)
            for cursor, span in enumerate(spans[slot]):
                match = by_start.get(span[0])
                if match is not None:
                    _consider(order, seen, (slot, cursor, other, match))
    for slot, slot_spans in enumerate(spans):
        by_start: dict[int, list[int]] = {}
        for cursor, span in enumerate(slot_spans):
            by_start.setdefault(span[0], []).append(cursor)
        for cursor, span in enumerate(slot_spans):
            for right in by_start.get(span[1], []):
                _consider(order, seen, (slot, cursor, slot, right))
            for other, other_span in enumerate(slot_spans):
                if other == cursor:
                    continue
                if span[0] <= other_span[0] and other_span[1] <= span[1]:
                    _consider(order, seen, (slot, cursor, slot, other))
    for slot, slot_spans in enumerate(spans):
        for other, other_spans in enumerate(spans):
            if other == slot:
                continue
            for cursor, span in enumerate(slot_spans):
                for match, other_span in enumerate(other_spans):
                    if span[0] == 0 and other_span[0] == 0 and span[1] == other_span[1]:
                        _consider(order, seen, (slot, cursor, other, match))
                    left_len = lengths[slot] if slot < len(lengths) else 0
                    right_len = lengths[other] if other < len(lengths) else 0
                    if span[1] == left_len and other_span[1] == right_len and (span[1] - span[0]) == (other_span[1] - other_span[0]) and span[1] > span[0]:
                        _consider(order, seen, (slot, cursor, other, match))
    return order


def collate_scope(tasks: list[Task]) -> dict[str, torch.Tensor]:
    batch = collate_local(tasks)
    _start, _end, _span_pad, catalog = _span_tensors(batch["classes"], batch["pad"])
    support_slots = batch["start"].shape[1]
    rel_slot_l = torch.zeros(len(tasks), MAX_REL, dtype=torch.long)
    rel_slot_r = torch.zeros(len(tasks), MAX_REL, dtype=torch.long)
    rel_left = torch.zeros(len(tasks), MAX_REL, dtype=torch.long)
    rel_right = torch.zeros(len(tasks), MAX_REL, dtype=torch.long)
    rel_y = torch.zeros(len(tasks), MAX_REL, len(PAIR_NAMES))
    rel_mask = torch.zeros(len(tasks), MAX_REL, dtype=torch.bool)
    rel_geom = torch.zeros(len(tasks), MAX_REL, N_GEOM)
    unary_y = torch.zeros(len(tasks), support_slots, batch["start"].shape[-1], len(UNARY_NAMES))
    cls_list = batch["classes"].detach().tolist()
    pad_list = batch["pad"].detach().tolist()
    for row, task in enumerate(tasks):
        supports = _supports(task)
        lengths = [int(sum(not flag for flag in pad_list[row][slot])) for slot in range(support_slots)]
        ids = [_ids(cls_list[row][slot], pad_list[row][slot]) for slot in range(support_slots)]
        facts = gold_facts(task.program, supports)
        labels, unary = _gold_labels(facts, catalog[row], lengths)
        for slot, rows in enumerate(unary):
            for cursor, values in enumerate(rows):
                if cursor >= unary_y.shape[2]:
                    break
                for channel, value in enumerate(values):
                    unary_y[row, slot, cursor, channel] = float(value)
        for cursor, key in enumerate(_candidate_order(labels, catalog[row], lengths)):
            slot_l, left, slot_r, right = key
            rel_slot_l[row, cursor] = slot_l
            rel_slot_r[row, cursor] = slot_r
            rel_left[row, cursor] = left
            rel_right[row, cursor] = right
            rel_mask[row, cursor] = True
            span_l = catalog[row][slot_l][left]
            span_r = catalog[row][slot_r][right]
            rel_geom[row, cursor] = torch.tensor(geometry(ids, lengths, slot_l, span_l[0], span_l[1], slot_r, span_r[0], span_r[1]))
            for channel in labels.get(key, ()):
                rel_y[row, cursor, channel] = 1.0
    batch.update(
        {
            "rel_slot_l": rel_slot_l,
            "rel_slot_r": rel_slot_r,
            "rel_left": rel_left,
            "rel_right": rel_right,
            "rel_y": rel_y,
            "rel_mask": rel_mask,
            "rel_geom": rel_geom,
            "unary_y": unary_y,
        }
    )
    return batch


class ScopeNet(nn.Module):
    """Frozen local recognizer plus a residual scope head."""

    def __init__(self):
        super().__init__()
        self.local = LocalNet()
        for param in self.local.parameters():
            param.requires_grad_(False)
        dim = self.local.d_model
        self.pair_in = nn.Linear(dim * 2 + N_GEOM, dim)
        self.pair_out = nn.Linear(dim, len(PAIR_NAMES))
        self.unary = nn.Linear(dim, len(UNARY_NAMES))
        nn.init.zeros_(self.pair_out.weight)
        nn.init.zeros_(self.pair_out.bias)
        nn.init.zeros_(self.unary.weight)
        nn.init.zeros_(self.unary.bias)

    def forward(self, batch) -> dict[str, torch.Tensor]:
        self.local.eval()
        with torch.no_grad():
            hidden = self.local.encode(
                batch["local"], batch["classes"], batch["rel"], batch["pad"], batch["start"], batch["end"], batch["span_pad"]
            )
            l1_alt = _bilinear(
                hidden,
                self.local.alt_q,
                self.local.alt_k,
                batch["rel_slot_l"],
                batch["rel_slot_r"],
                batch["rel_left"],
                batch["rel_right"],
            )
        rows = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        left_h = hidden[rows, batch["rel_slot_l"].clamp(0, hidden.shape[1] - 1), batch["rel_left"].clamp(0, hidden.shape[2] - 1)]
        right_h = hidden[rows, batch["rel_slot_r"].clamp(0, hidden.shape[1] - 1), batch["rel_right"].clamp(0, hidden.shape[2] - 1)]
        feat = torch.cat([left_h, right_h, batch["rel_geom"]], dim=-1)
        pair = self.pair_out(F.gelu(self.pair_in(feat)))
        return {"pair": pair, "unary": self.unary(hidden), "l1_alt": l1_alt, "hidden": hidden}


def _rank_term(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive = logit[target > 0.5]
    negative = logit[target < 0.5]
    if positive.numel() == 0 or negative.numel() == 0:
        return logit.new_zeros(())
    return F.softplus(negative[:, None] - positive[None, :]).mean()


def scope_loss(logits, batch) -> tuple[torch.Tensor, torch.Tensor]:
    alt = (logits["pair"][..., ALT_CHANNEL] + logits["l1_alt"]).unsqueeze(-1)
    combined = torch.cat([logits["pair"][..., :ALT_CHANNEL], alt, logits["pair"][..., ALT_CHANNEL + 1 :]], dim=-1)
    mask = batch["rel_mask"]
    if bool(mask.any()):
        pair = _bce(combined[mask], batch["rel_y"][mask], 4.0)
        rank = combined.new_zeros(())
        channels = 0
        for row in range(combined.shape[0]):
            live = mask[row]
            if not bool(live.any()):
                continue
            for channel in range(combined.shape[-1]):
                term = _rank_term(combined[row, live, channel], batch["rel_y"][row, live, channel])
                if term.grad_fn is None and float(term) == 0.0:
                    continue
                rank = rank + term
                channels += 1
        if channels:
            pair = pair + 0.5 * rank / channels
    else:
        pair = combined.new_zeros(())
    live = ~batch["span_pad"]
    unary_logit = logits["unary"][live]
    unary_target = batch["unary_y"][live]
    unary = _bce(unary_logit, unary_target, 4.0)
    return pair + unary, pair.detach()


def _class_bounds(width: int) -> tuple[tuple[int, int], tuple[int, int]]:
    exact = (min(4, width), min(4, width))
    opened = (1, 2) if width <= 2 else (1, min(8, max(2, width)))
    return exact, opened


def _bound_variant(hyp: Hyp, exact_spans: set[tuple[int, int, int]]) -> tuple[int, int] | None:
    """A second bound for the same span.

    Repeat digits are a 1–2 class, while an optional body is exactly as wide
    as the span. The strongest optional span on a slot prefers the exact
    width; every other class span prefers 1–2. Repeat heads collapse to a
    closed maximum, so the open maximum stays beside that prediction.
    """
    width = max(0, hyp.end - hyp.start)
    if width < 1:
        return None
    if hyp.kind == "cls":
        exact, opened = _class_bounds(width)
        primary, fallback = (exact, opened) if (hyp.slot, hyp.start, hyp.end) in exact_spans else (opened, exact)
        bound = primary if (hyp.lo, hyp.hi) != primary else fallback
    elif (hyp.lo, hyp.hi) == (1, 8):
        bound = (1, 3)
    else:
        bound = (1, 8)
    if bound[0] < 1 or bound[1] < bound[0] or (hyp.lo, hyp.hi) == bound:
        return None
    return bound


def _exact_width_spans(hyps: list[Hyp]) -> set[tuple[int, int, int]]:
    best: dict[int, Hyp] = {}
    for hyp in hyps:
        if hyp.kind != "optional":
            continue
        current = best.get(hyp.slot)
        if current is None or hyp.score > current.score:
            best[hyp.slot] = hyp
    return {(hyp.slot, hyp.start, hyp.end) for hyp in best.values()}


def _diversify_bounds(hyps: list[Hyp]) -> list[Hyp]:
    exact_spans = _exact_width_spans(hyps)
    groups: dict[tuple, list[Hyp]] = {}
    for hyp in hyps:
        if hyp.kind == "cls":
            groups.setdefault((hyp.slot, hyp.cls), []).append(hyp)
        elif hyp.kind == "repeat":
            groups.setdefault((hyp.slot,), []).append(hyp)
    extra: list[Hyp] = []
    for rows in groups.values():
        rows.sort(key=lambda item: item.score, reverse=True)
        for hyp in rows[: LOCAL_K // 2]:
            bound = _bound_variant(hyp, exact_spans)
            if bound is None:
                continue
            extra.append(replace(hyp, lo=bound[0], hi=bound[1], score=hyp.score - 0.01))
    return hyps + extra


def _alt_candidates(live, start, end) -> list[tuple[int, int, int, int]]:
    found = []
    slots = live.shape[0]
    for slot in range(slots):
        for other in range(slots):
            if other == slot:
                continue
            rights: dict[int, int] = {}
            for index in range(live.shape[1]):
                if bool(live[other, index]):
                    rights.setdefault(int(start[other, index]), index)
            for index in range(live.shape[1]):
                if not bool(live[slot, index]):
                    continue
                match = rights.get(int(start[slot, index]))
                if match is None:
                    continue
                found.append((slot, index, other, match))
    return found


def _rescore_alts(model: ScopeNet, hidden: torch.Tensor, batch, row: int, hyps: list[Hyp]) -> list[Hyp]:
    live = ~batch["span_pad"][row]
    start = batch["start"][row]
    end = batch["end"][row]
    pairs = _alt_candidates(live, start, end)
    kept = [hyp for hyp in hyps if hyp.kind != "alt"]
    if not pairs:
        return kept
    slot_l = torch.tensor([item[0] for item in pairs], dtype=torch.long)
    left = torch.tensor([item[1] for item in pairs], dtype=torch.long)
    slot_r = torch.tensor([item[2] for item in pairs], dtype=torch.long)
    right = torch.tensor([item[3] for item in pairs], dtype=torch.long)
    query = model.local.alt_q(hidden)
    key = model.local.alt_k(hidden)
    l1 = (query[slot_l, left] * key[slot_r, right]).sum(-1)
    cls_list = batch["classes"][row].detach().tolist()
    pad_list = batch["pad"][row].detach().tolist()
    ids = [_ids(cls_list[slot], pad_list[slot]) for slot in range(live.shape[0])]
    lengths = [int(sum(not flag for flag in pad_list[slot])) for slot in range(live.shape[0])]
    geom = []
    for slot, index, other, match in pairs:
        geom.append(geometry(ids, lengths, slot, int(start[slot, index]), int(end[slot, index]), other, int(start[other, match]), int(end[other, match])))
    geom_t = torch.tensor(geom, dtype=hidden.dtype)
    left_h = hidden[slot_l, left]
    right_h = hidden[slot_r, right]
    delta = model.pair_out(F.gelu(model.pair_in(torch.cat([left_h, right_h, geom_t], dim=-1))))[:, ALT_CHANNEL]
    scores = (l1 + delta).tolist()
    order = sorted(range(len(pairs)), key=lambda cursor: scores[cursor], reverse=True)
    per_slot: dict[int, int] = {}
    for cursor in order:
        slot, index, other, match = pairs[cursor]
        used = per_slot.get(slot, 0)
        if used >= LOCAL_K:
            continue
        per_slot[slot] = used + 1
        kept.append(
            Hyp(
                "alt",
                slot,
                int(start[slot, index]),
                int(end[slot, index]),
                score=float(scores[cursor]),
                slot2=other,
                start2=int(start[other, match]),
                end2=int(end[other, match]),
            )
        )
    return kept


def lock_hypotheses(hyps: list[Hyp], k: int = LOCAL_K) -> list[Hyp]:
    """Keep the top k per kind and slot, with ranks that a second truncate preserves."""
    grouped: dict[tuple[str, int], list[Hyp]] = {}
    for hyp in hyps:
        grouped.setdefault((hyp.kind, hyp.slot), []).append(hyp)
    kept: list[Hyp] = []
    for rows in grouped.values():
        rows.sort(key=lambda item: (item.score, item.end, item.start, item.lo, item.hi, item.slot2, item.start2, item.end2), reverse=True)
        for rank, item in enumerate(rows[:k]):
            kept.append(replace(item, score=float(item.score) - rank * 1e-3))
    return kept


def hypotheses_from_scope(model: ScopeNet, batch, supports: list[list[str]], k: int = LOCAL_K) -> list[list[Hyp]]:
    model.local.eval()
    base = hypotheses_from_batch(model.local, batch, supports, k)
    with torch.no_grad():
        hidden = model.local.encode(batch["local"], batch["classes"], batch["rel"], batch["pad"], batch["start"], batch["end"], batch["span_pad"])
        rows = []
        for row, _texts in enumerate(supports):
            diversified = _diversify_bounds(base[row])
            rescored = _rescore_alts(model, hidden[row], batch, row, diversified)
            rows.append(lock_hypotheses(rescored, k))
    return rows
