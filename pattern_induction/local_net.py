"""Independent local scores on the relational span trunk.

Each head keeps its own top hypotheses. Nothing in this module commits to a
single abstract syntax tree.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pattern_induction.chart import MAX_CAND, LOCAL_K, Hyp, gold_facts, spans_from_ids
from pattern_induction.program import MAX_LEN, MAX_SUPPORT, OPEN, Task, feature_batch
from pattern_induction.span_net import SpanNet, _attend, _pool_spans

ATOM = ("copy", "digit", "upper", "lower")
MAX_PAIRS = 48


def _ids(classes: list[int], pad: list[bool]) -> list[int]:
    ids = []
    for kind, is_pad in zip(classes, pad):
        if is_pad:
            break
        ids.append(int(kind))
    return ids


def _span_tensors(classes: torch.Tensor, pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[tuple[int, int]]]]]:
    batch, slots, _length = classes.shape
    support_slots = slots - 1
    start = torch.zeros(batch, support_slots, MAX_CAND, dtype=torch.long)
    end = torch.zeros(batch, support_slots, MAX_CAND, dtype=torch.long)
    span_pad = torch.ones(batch, support_slots, MAX_CAND, dtype=torch.bool)
    catalog: list[list[list[tuple[int, int]]]] = []
    cls_list = classes.detach().tolist()
    pad_list = pad.detach().tolist()
    for row in range(batch):
        row_spans = []
        for slot in range(support_slots):
            spans = spans_from_ids(_ids(cls_list[row][slot], pad_list[row][slot]))
            row_spans.append(spans)
            for index, (left, right) in enumerate(spans):
                start[row, slot, index] = left
                end[row, slot, index] = right
                span_pad[row, slot, index] = False
        catalog.append(row_spans)
    return start, end, span_pad, catalog


def collate_local(tasks: list[Task]) -> dict[str, torch.Tensor]:
    local, classes, rel, pad = feature_batch(tasks)
    start, end, span_pad, catalog = _span_tensors(classes, pad)
    batch, support_slots, _cands = start.shape
    atom = torch.zeros(batch, support_slots, MAX_CAND, 4)
    repeat = torch.zeros(batch, support_slots, MAX_CAND)
    optional = torch.zeros(batch, support_slots, MAX_CAND)
    cut = torch.zeros(batch, support_slots, MAX_CAND)
    lo = torch.full((batch, support_slots, MAX_CAND), -1)
    hi = torch.full((batch, support_slots, MAX_CAND), -1)
    pair_slot = torch.zeros(batch, MAX_PAIRS, dtype=torch.long)
    pair_left = torch.zeros(batch, MAX_PAIRS, dtype=torch.long)
    pair_right = torch.zeros(batch, MAX_PAIRS, dtype=torch.long)
    pair_y = torch.zeros(batch, MAX_PAIRS)
    pair_mask = torch.zeros(batch, MAX_PAIRS, dtype=torch.bool)
    alt_slot_l = torch.zeros(batch, MAX_PAIRS, dtype=torch.long)
    alt_slot_r = torch.zeros(batch, MAX_PAIRS, dtype=torch.long)
    alt_left = torch.zeros(batch, MAX_PAIRS, dtype=torch.long)
    alt_right = torch.zeros(batch, MAX_PAIRS, dtype=torch.long)
    alt_y = torch.zeros(batch, MAX_PAIRS)
    alt_mask = torch.zeros(batch, MAX_PAIRS, dtype=torch.bool)
    for row, task in enumerate(tasks):
        facts = gold_facts(task.program, [text[:MAX_LEN] for text in task.supports[:MAX_SUPPORT]])
        for slot, spans in enumerate(catalog[row]):
            index = {span: cursor for cursor, span in enumerate(spans)}
            for fact in facts:
                if fact.kind == "cut" and fact.slot == slot:
                    for span, cursor in index.items():
                        if span[0] == fact.index:
                            cut[row, slot, cursor] = 1.0
                    continue
                if fact.slot != slot or fact.kind in {"alt", "pair"}:
                    continue
                cursor = index.get((fact.start, fact.end))
                if cursor is None:
                    continue
                if fact.kind == "copy":
                    atom[row, slot, cursor, 0] = 1.0
                elif fact.kind == "cls":
                    name = {"digit": 1, "upper": 2, "lower": 3}.get(fact.cls, 1)
                    atom[row, slot, cursor, name] = 1.0
                    lo[row, slot, cursor] = max(0, min(fact.lo, 4) - 1)
                    hi[row, slot, cursor] = max(0, min(fact.hi, OPEN) - 1)
                elif fact.kind == "repeat":
                    repeat[row, slot, cursor] = 1.0
                    lo[row, slot, cursor] = max(0, min(fact.lo, 4) - 1)
                    hi[row, slot, cursor] = max(0, min(fact.hi, OPEN) - 1)
                elif fact.kind == "optional":
                    optional[row, slot, cursor] = 1.0
            _fill_pairs(facts, slot, index, row, pair_slot, pair_left, pair_right, pair_y, pair_mask, positives=True)
        for slot, spans in enumerate(catalog[row]):
            index = {span: cursor for cursor, span in enumerate(spans)}
            _fill_pairs(facts, slot, index, row, pair_slot, pair_left, pair_right, pair_y, pair_mask, positives=False)
        _fill_alts(facts, catalog[row], row, alt_slot_l, alt_slot_r, alt_left, alt_right, alt_y, alt_mask)
    return {
        "local": local,
        "classes": classes,
        "rel": rel,
        "pad": pad,
        "start": start,
        "end": end,
        "span_pad": span_pad,
        "atom": atom,
        "repeat": repeat,
        "optional": optional,
        "cut": cut,
        "lo": lo,
        "hi": hi,
        "pair_slot": pair_slot,
        "pair_left": pair_left,
        "pair_right": pair_right,
        "pair_y": pair_y,
        "pair_mask": pair_mask,
        "alt_slot_l": alt_slot_l,
        "alt_slot_r": alt_slot_r,
        "alt_left": alt_left,
        "alt_right": alt_right,
        "alt_y": alt_y,
        "alt_mask": alt_mask,
    }


def _fill_pairs(facts, slot, index, row, pair_slot, pair_left, pair_right, pair_y, pair_mask, positives: bool) -> None:
    used = int(pair_mask[row].sum())
    if positives:
        for fact in facts:
            if fact.kind != "pair" or fact.slot != slot or used >= MAX_PAIRS:
                continue
            left = index.get((fact.start, fact.end))
            right = index.get((fact.start2, fact.end2))
            if left is None or right is None:
                continue
            pair_slot[row, used] = slot
            pair_left[row, used] = left
            pair_right[row, used] = right
            pair_y[row, used] = 1.0
            pair_mask[row, used] = True
            used += 1
        return
    positive = {(int(pair_left[row, cursor]), int(pair_right[row, cursor])) for cursor in range(used) if int(pair_slot[row, cursor]) == slot and pair_y[row, cursor] > 0.5}
    by_start: dict[int, list[int]] = {}
    for span, cursor in index.items():
        by_start.setdefault(span[0], []).append(cursor)
    for span, left in index.items():
        if used >= MAX_PAIRS:
            break
        for right in by_start.get(span[1], []):
            if (left, right) in positive:
                continue
            pair_slot[row, used] = slot
            pair_left[row, used] = left
            pair_right[row, used] = right
            pair_mask[row, used] = True
            used += 1
            if used >= MAX_PAIRS:
                break


def _fill_alts(facts, spans, row, alt_slot_l, alt_slot_r, alt_left, alt_right, alt_y, alt_mask) -> None:
    used = 0
    indexes = [{span: cursor for cursor, span in enumerate(slot_spans)} for slot_spans in spans]
    positive = set()
    for fact in facts:
        if fact.kind != "alt" or used >= MAX_PAIRS:
            continue
        if fact.slot >= len(indexes) or fact.slot2 >= len(indexes) or fact.slot2 < 0:
            continue
        left = indexes[fact.slot].get((fact.start, fact.end))
        right = indexes[fact.slot2].get((fact.start2, fact.end2))
        if left is None or right is None:
            continue
        alt_slot_l[row, used] = fact.slot
        alt_slot_r[row, used] = fact.slot2
        alt_left[row, used] = left
        alt_right[row, used] = right
        alt_y[row, used] = 1.0
        alt_mask[row, used] = True
        positive.add((fact.slot, left, fact.slot2, right))
        used += 1
    for slot, slot_spans in enumerate(spans):
        for other in range(len(spans)):
            if other == slot:
                continue
            other_index = {span[0]: cursor for cursor, span in enumerate(spans[other])}
            for cursor, span in enumerate(slot_spans):
                match = other_index.get(span[0])
                if match is None or (slot, cursor, other, match) in positive:
                    continue
                if used >= MAX_PAIRS:
                    return
                alt_slot_l[row, used] = slot
                alt_slot_r[row, used] = other
                alt_left[row, used] = cursor
                alt_right[row, used] = match
                alt_mask[row, used] = True
                used += 1


class LocalNet(SpanNet):
    """Relational characters in, independent span hypotheses out."""

    def __init__(self):
        super().__init__()
        dim = self.d_model
        self.mem = nn.Identity()
        self.kind_head = nn.Identity()
        self.span_head = nn.Identity()
        self.bound_head = nn.Identity()
        self.role_head = nn.Identity()
        self.mix = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.atom_head = nn.Linear(dim, 4)
        self.repeat_head = nn.Linear(dim, 1)
        self.opt_head = nn.Linear(dim, 1)
        self.cut_head = nn.Linear(dim, 1)
        self.lo_head = nn.Linear(dim, 4)
        self.hi_head = nn.Linear(dim, 8)
        self.pair_q = nn.Linear(dim, 32, bias=False)
        self.pair_k = nn.Linear(dim, 32, bias=False)
        self.alt_q = nn.Linear(dim, 32, bias=False)
        self.alt_k = nn.Linear(dim, 32, bias=False)

    def encode(self, local, classes, rel, pad, start, end, span_pad):
        _batch, slots, length = local.shape
        roles = torch.zeros(slots, dtype=torch.long, device=local.device)
        roles[-1] = 1
        which = torch.arange(slots, device=local.device)
        hidden = self.class_emb(classes) + self.local_emb(local) + self.rel_proj(rel)
        hidden = hidden + self.char_pos[:length][None, None, :, :]
        hidden = hidden + self.role(roles)[None, :, None, :] + self.which(which)[None, :, None, :]
        hidden = hidden * (~pad).unsqueeze(-1).float()
        encoded = _attend(self.char_encoder, hidden, pad)
        pooled = _pool_spans(encoded[:, :-1], start, end, span_pad)
        span_h = pooled + self.span_pos[torch.arange(start.shape[-1]).clamp(max=self.max_spans - 1)]
        span_h = span_h * (~span_pad).unsqueeze(-1).float()
        span_h = _attend(self.span_encoder, span_h, span_pad)
        return self.mix(span_h)

    def forward(self, batch) -> dict[str, torch.Tensor]:
        hidden = self.encode(batch["local"], batch["classes"], batch["rel"], batch["pad"], batch["start"], batch["end"], batch["span_pad"])
        return {
            "hidden": hidden,
            "atom": self.atom_head(hidden),
            "repeat": self.repeat_head(hidden).squeeze(-1),
            "optional": self.opt_head(hidden).squeeze(-1),
            "cut": self.cut_head(hidden).squeeze(-1),
            "lo": self.lo_head(hidden),
            "hi": self.hi_head(hidden),
            "pair": _bilinear(hidden, self.pair_q, self.pair_k, batch["pair_slot"], batch["pair_slot"], batch["pair_left"], batch["pair_right"]),
            "alt": _bilinear(hidden, self.alt_q, self.alt_k, batch["alt_slot_l"], batch["alt_slot_r"], batch["alt_left"], batch["alt_right"]),
        }


def _bilinear(hidden, query, key, slot_l, slot_r, left, right) -> torch.Tensor:
    batch, pairs = left.shape
    rows = torch.arange(batch, device=hidden.device)[:, None]
    left_h = hidden[rows, slot_l.clamp(0, hidden.shape[1] - 1), left.clamp(0, hidden.shape[2] - 1)]
    right_h = hidden[rows, slot_r.clamp(0, hidden.shape[1] - 1), right.clamp(0, hidden.shape[2] - 1)]
    return (query(left_h) * key(right_h)).sum(-1)


def local_loss(logits, batch) -> tuple[torch.Tensor, torch.Tensor]:
    mask = ~batch["span_pad"]
    atom = _bce(logits["atom"][mask], batch["atom"][mask], 2.0)
    repeat = _bce(logits["repeat"][mask], batch["repeat"][mask], 4.0)
    optional = _bce(logits["optional"][mask], batch["optional"][mask], 4.0)
    cut = _bce(logits["cut"][mask], batch["cut"][mask], 2.0)
    lo_mask = batch["lo"] >= 0
    hi_mask = batch["hi"] >= 0
    if bool(lo_mask.any()):
        bounds = F.cross_entropy(logits["lo"][lo_mask], batch["lo"][lo_mask]) + F.cross_entropy(logits["hi"][hi_mask], batch["hi"][hi_mask])
    else:
        bounds = logits["atom"].new_zeros(())
    pair = _pair_bce(logits["pair"], batch["pair_y"], batch["pair_mask"])
    alt = _pair_bce(logits["alt"], batch["alt_y"], batch["alt_mask"])
    total = atom + repeat + optional + cut + 0.5 * bounds + pair + alt
    return total, atom.detach()


def _bce(logit, target, pos_weight: float) -> torch.Tensor:
    if logit.numel() == 0:
        return logit.new_zeros(())
    weight = torch.where(target > 0.5, logit.new_full(target.shape, pos_weight), torch.ones_like(target))
    return F.binary_cross_entropy_with_logits(logit, target, weight=weight)


def _pair_bce(logit, target, mask) -> torch.Tensor:
    if not bool(mask.any()):
        return logit.new_zeros(())
    return _bce(logit[mask], target[mask], 4.0)


def _topk_index(scores: torch.Tensor, live: torch.Tensor, k: int) -> list[int]:
    masked = scores.masked_fill(~live, -1e9)
    count = int(live.sum())
    if count <= 0:
        return []
    chosen = torch.topk(masked, k=min(k, count)).indices.tolist()
    return [int(index) for index in chosen if bool(live[index])]


def hypotheses_from_batch(model: LocalNet, batch, supports: list[list[str]], k: int = LOCAL_K) -> list[list[Hyp]]:
    model.eval()
    with torch.no_grad():
        hidden = model.encode(batch["local"], batch["classes"], batch["rel"], batch["pad"], batch["start"], batch["end"], batch["span_pad"])
        atom = model.atom_head(hidden)
        repeat = model.repeat_head(hidden).squeeze(-1)
        optional = model.opt_head(hidden).squeeze(-1)
        cut = model.cut_head(hidden).squeeze(-1)
        lo = model.lo_head(hidden)
        hi = model.hi_head(hidden)
        rows = []
        for row, texts in enumerate(supports):
            rows.append(_row_hypotheses(hidden[row], atom[row], repeat[row], optional[row], cut[row], lo[row], hi[row], batch, row, texts, k, model))
    return rows


def _bounds(lo_row, hi_row, index: int) -> tuple[int, int]:
    lo_index = int(lo_row[index].argmax())
    hi_scores = hi_row[index].clone()
    hi_scores[:lo_index] = -1e9
    hi_index = int(hi_scores.argmax())
    if hi_index < lo_index:
        hi_index = lo_index
    return lo_index + 1, min(hi_index, 7) + 1


def _row_hypotheses(hidden, atom, repeat, optional, cut, lo, hi, batch, row, texts, k, model: LocalNet) -> list[Hyp]:
    hyps: list[Hyp] = []
    live = ~batch["span_pad"][row]
    start = batch["start"][row]
    end = batch["end"][row]
    names = ("digit", "upper", "lower")
    for slot in range(live.shape[0]):
        slot_live = live[slot]
        for index in _topk_index(atom[slot, :, 0], slot_live, k):
            hyps.append(Hyp("copy", slot, int(start[slot, index]), int(end[slot, index]), score=float(atom[slot, index, 0])))
        for name_index, name in enumerate(names, start=1):
            for index in _topk_index(atom[slot, :, name_index], slot_live, k):
                lower, upper = _bounds(lo[slot], hi[slot], index)
                hyps.append(Hyp("cls", slot, int(start[slot, index]), int(end[slot, index]), score=float(atom[slot, index, name_index]), cls=name, lo=lower, hi=upper))
        for index in _topk_index(repeat[slot], slot_live, k):
            lower, upper = _bounds(lo[slot], hi[slot], index)
            hyps.append(Hyp("repeat", slot, int(start[slot, index]), int(end[slot, index]), score=float(repeat[slot, index]), lo=lower, hi=upper))
        for index in _topk_index(optional[slot], slot_live, k):
            hyps.append(Hyp("optional", slot, int(start[slot, index]), int(end[slot, index]), score=float(optional[slot, index])))
        best_cut: dict[int, float] = {}
        for index in range(slot_live.shape[0]):
            if not bool(slot_live[index]):
                continue
            position = int(start[slot, index])
            score = float(cut[slot, index])
            if position not in best_cut or score > best_cut[position]:
                best_cut[position] = score
        for position, score in sorted(best_cut.items(), key=lambda item: item[1], reverse=True)[:k]:
            hyps.append(Hyp("cut", slot, position, position, score=score, index=position))
        hyps.extend(_pair_hyps(hidden, model, slot, slot_live, start[slot], end[slot], k))
    hyps.extend(_alt_hyps(hidden, model, live, start, end, k))
    return hyps


def _pair_hyps(hidden, model, slot, live, start, end, k) -> list[Hyp]:
    indexes = [index for index in range(live.shape[0]) if bool(live[index])]
    by_start: dict[int, list[int]] = {}
    for index in indexes:
        by_start.setdefault(int(start[index]), []).append(index)
    scored = []
    query = model.pair_q(hidden[slot])
    key = model.pair_k(hidden[slot])
    for left in indexes:
        for right in by_start.get(int(end[left]), []):
            score = float((query[left] * key[right]).sum())
            scored.append((score, left, right))
    scored.sort(reverse=True)
    hyps = []
    for score, left, right in scored[:k]:
        hyps.append(
            Hyp(
                "pair",
                slot,
                int(start[left]),
                int(end[left]),
                score=score,
                slot2=slot,
                start2=int(start[right]),
                end2=int(end[right]),
            )
        )
    return hyps


def _alt_hyps(hidden, model, live, start, end, k) -> list[Hyp]:
    query = model.alt_q(hidden)
    key = model.alt_k(hidden)
    scored = []
    slots = live.shape[0]
    for slot in range(slots):
        for other in range(slots):
            if other == slot:
                continue
            rights = {}
            for index in range(live.shape[1]):
                if bool(live[other, index]):
                    rights.setdefault(int(start[other, index]), index)
            for index in range(live.shape[1]):
                if not bool(live[slot, index]):
                    continue
                match = rights.get(int(start[slot, index]))
                if match is None:
                    continue
                score = float((query[slot, index] * key[other, match]).sum())
                scored.append((score, slot, index, other, match))
    scored.sort(reverse=True)
    hyps = []
    for score, slot, index, other, match in scored[:k]:
        hyps.append(
            Hyp(
                "alt",
                slot,
                int(start[slot, index]),
                int(end[slot, index]),
                score=score,
                slot2=other,
                start2=int(start[other, match]),
                end2=int(end[other, match]),
            )
        )
    return hyps
