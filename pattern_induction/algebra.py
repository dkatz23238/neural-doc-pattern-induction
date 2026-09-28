"""Shared operator modules over a gold schema.

H3-oracle builds the operator tree from the support derivations. The same
SEQ, REPEAT, OPTIONAL, and ALT functions run on every language. REPEAT sees
the schema's min and max, not how many copies this string happened to use, so
a longer query repeat is the same operator.

The query is aligned to that schema from classes and local ids. Its gold
annotation is not an input, so a positive label cannot arrive as "a tree exists."
"""

from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F
from torch import nn

from pattern_induction.groups import OP_ALT, OP_LEAF, OP_OPTIONAL, OP_REPEAT, OP_SEQ, _leaves, lift_tree
from pattern_induction.renaming import DIGIT, LOWER, PUNCT, SPACE, UPPER, structural_class

_CLASS = {"upper": UPPER, "lower": LOWER, "digit": DIGIT, "space": SPACE, "punct": PUNCT}
from pattern_induction.span_net import SpanNet, _attend, _pool_spans, propose_spans


def _runs(values: list[int]) -> list[tuple[int, int, int]]:
    runs = []
    index = 0
    while index < len(values):
        stop = index + 1
        while stop < len(values) and values[stop] == values[index]:
            stop += 1
        runs.append((index, stop, values[index]))
        index = stop
    return runs


def _nest(ann, text: str, slot: int) -> dict | None:
    tree = lift_tree(ann)
    if not tree:
        return None
    kids: dict[int, list[int]] = {}
    for index, node in enumerate(tree):
        kids.setdefault(node.parent, []).append(index)

    def build(index: int) -> dict:
        node = tree[index]
        children = [build(child) for child in kids.get(index, [])]
        payload = {"op": node.op, "children": children, "exemplars": [], "cls": 0, "min_n": 1, "max_n": 1}
        if node.op == OP_LEAF and node.end > node.start:
            payload["cls"] = structural_class(text[node.start])
            payload["exemplars"] = [(slot, node.start, node.end)]
        if node.op == OP_REPEAT:
            payload["min_n"] = max(len(children), 1)
            payload["max_n"] = max(len(children), 1)
        return payload

    return build(0)


def _key(node: dict) -> tuple:
    if node["op"] == OP_LEAF:
        return ("L", node["cls"])
    if node["op"] == OP_REPEAT:
        return ("R",)
    if node["op"] == OP_OPTIONAL:
        # Absent and present copies of one optional are the same operator.
        return ("O",)
    if node["op"] == OP_ALT:
        return ("A",)
    return ("S", len(node["children"]))


def _size(node: dict) -> int:
    return 1 + sum(_size(child) for child in node["children"])


def _subsequence(needle: list, hay: list) -> bool:
    cursor = 0
    for item in hay:
        if cursor < len(needle) and item == needle[cursor]:
            cursor += 1
    return cursor == len(needle)


def _leaf_cls(node: dict) -> int:
    current = node
    while current["op"] != OP_LEAF and current["children"]:
        current = current["children"][0]
    return current.get("cls", 0)


def merge_nodes(nodes: list[dict], depth: int = 0) -> dict:
    nodes = [node for node in nodes if node]
    if not nodes:
        return {"op": OP_LEAF, "children": [], "exemplars": [(0, 0, 1)], "cls": 0, "min_n": 1, "max_n": 1}
    if len(nodes) == 1:
        return nodes[0]
    if depth > 24:
        return max(nodes, key=_size)
    ops = {node["op"] for node in nodes}
    if ops == {OP_LEAF}:
        exemplars = [item for node in nodes for item in node["exemplars"]]
        donor = next((node for node in nodes if node.get("exemplars")), nodes[0])
        return {**donor, "exemplars": exemplars or donor["exemplars"]}
    if ops == {OP_REPEAT}:
        mins = []
        maxs = []
        units = []
        for node in nodes:
            kids = node["children"]
            if not kids:
                continue
            if node.get("bounds"):
                mins.append(int(node["min_n"]))
                maxs.append(int(node["max_n"]))
                units.append(kids[0])
            else:
                count = max(len(kids), 1)
                mins.append(count)
                maxs.append(count)
                units.append(kids[-1])
        child = merge_nodes(units, depth + 1) if units else merge_nodes([], depth + 1)
        return {
            "op": OP_REPEAT,
            "children": [child],
            "exemplars": [],
            "cls": 0,
            "min_n": min(mins) if mins else 1,
            "max_n": max(maxs) if maxs else 1,
            "bounds": True,
        }
    if ops == {OP_OPTIONAL}:
        kids = [node["children"][0] for node in nodes if node["children"]]
        child = merge_nodes(kids, depth + 1) if kids else merge_nodes([], depth + 1)
        return {"op": OP_OPTIONAL, "children": [child], "exemplars": [], "cls": 0, "min_n": 1, "max_n": 1}
    if ops == {OP_ALT}:
        grouped: dict[int, list[dict]] = {}
        for node in nodes:
            if not node["children"]:
                continue
            grouped.setdefault(_leaf_cls(node["children"][0]), []).append(node["children"][0])
        kids = [merge_nodes(group, depth + 1) for group in grouped.values()]
        if len(kids) == 1:
            kids.append(kids[0])
        return {"op": OP_ALT, "children": kids[:2], "exemplars": [], "cls": 0, "min_n": 1, "max_n": 1}
    if ops <= {OP_SEQ, OP_OPTIONAL}:
        return _merge_sequences(nodes, depth + 1)
    return max(nodes, key=_size)


def _merge_sequences(nodes: list[dict], depth: int = 0) -> dict:
    sequences = []
    for node in nodes:
        if node["op"] == OP_SEQ:
            sequences.append(node["children"])
        else:
            sequences.append([node])
    keyed = [[_key(child) for child in children] for children in sequences]
    shortest = min(range(len(sequences)), key=lambda index: (len(sequences[index]), index))
    backbone = keyed[shortest]
    if not backbone or not all(_subsequence(backbone, keys) for keys in keyed):
        richest = max(sequences, key=len)
        return {"op": OP_SEQ, "children": richest, "exemplars": [], "cls": 0, "min_n": 1, "max_n": 1}
    merged = []
    pointers = [0] * len(sequences)
    for key in backbone:
        extras = []
        matched = []
        for index, children in enumerate(sequences):
            cursor = pointers[index]
            while cursor < len(children) and keyed[index][cursor] != key:
                extras.append(children[cursor])
                cursor += 1
            if cursor < len(children) and keyed[index][cursor] == key:
                matched.append(children[cursor])
                cursor += 1
            pointers[index] = cursor
        if extras:
            extra = merge_nodes(extras, depth + 1)
            if extra["op"] != OP_OPTIONAL:
                extra = {"op": OP_OPTIONAL, "children": [extra], "exemplars": [], "cls": 0, "min_n": 1, "max_n": 1}
            merged.append(extra)
        if matched:
            merged.append(merge_nodes(matched, depth + 1))
    trailing = []
    for index, children in enumerate(sequences):
        trailing.extend(children[pointers[index] :])
    if trailing:
        extra = merge_nodes(trailing, depth + 1)
        if extra["op"] != OP_OPTIONAL:
            extra = {"op": OP_OPTIONAL, "children": [extra], "exemplars": [], "cls": 0, "min_n": 1, "max_n": 1}
        merged.append(extra)
    if len(merged) == 1:
        return merged[0]
    return {"op": OP_SEQ, "children": merged, "exemplars": [], "cls": 0, "min_n": 1, "max_n": 1}


def _blank(op: int = OP_LEAF) -> dict:
    return {"op": op, "children": [], "exemplars": [], "cls": 0, "min_n": 1, "max_n": 1}


def _dsl_payload(episode) -> dict | None:
    pattern = getattr(episode, "pattern", "") or ""
    if not pattern.startswith("{"):
        return None
    try:
        payload = json.loads(pattern)
    except json.JSONDecodeError:
        return None
    schema = payload.get("schema") if isinstance(payload, dict) else None
    if isinstance(schema, dict) and schema.get("type"):
        return schema
    return None


def _schema_child(node: dict) -> dict:
    if isinstance(node.get("child"), dict):
        return node["child"]
    kids = node.get("children") or []
    return kids[0] if kids else {}


def _schema_class(node: dict) -> int | None:
    if not isinstance(node, dict):
        return None
    kind = node.get("type")
    if kind in ("class", "literal_slot"):
        return _CLASS.get(str(node.get("class") or ""))
    if kind == "seq":
        kids = node.get("children") or []
        return _schema_class(kids[0]) if kids else None
    if kind in ("optional", "repeat"):
        return _schema_class(_schema_child(node))
    return None


def _leaf_from_span(leaf: dict, text: str, slot: int) -> dict:
    start = int(leaf["start"])
    end = int(leaf["end"])
    cls = structural_class(text[start]) if start < len(text) else 0
    return {"op": OP_LEAF, "children": [], "exemplars": [(slot, start, end)], "cls": cls, "min_n": 1, "max_n": 1}


def _bind_dsl(schema: dict, ann, text: str, slot: int) -> dict:
    """One support derivation, shaped by the schema rather than by how many copies it drew."""
    leaves = _leaves(ann) if ann is not None else []
    index = 0

    def walk(node: dict) -> dict:
        nonlocal index
        kind = node.get("type")
        if kind in ("class", "literal_slot"):
            if index >= len(leaves):
                return _blank()
            current = leaves[index]
            index += 1
            return _leaf_from_span(current, text, slot)
        if kind == "seq":
            children = [walk(child) for child in node.get("children") or []]
            return {"op": OP_SEQ, "children": children, "exemplars": [], "cls": 0, "min_n": 1, "max_n": 1}
        if kind == "optional":
            child_schema = _schema_child(node)
            if index < len(leaves) and leaves[index]["role"][1] == 1:
                child = walk(child_schema)
                children = [child]
            else:
                children = []
            return {"op": OP_OPTIONAL, "children": children, "exemplars": [], "cls": 0, "min_n": 1, "max_n": 1}
        if kind == "alt":
            branches = list(node.get("children") or [])
            chosen = 0
            if index < len(leaves) and branches:
                actual = structural_class(text[leaves[index]["start"]]) if leaves[index]["start"] < len(text) else None
                for branch_index, branch in enumerate(branches):
                    if _schema_class(branch) == actual:
                        chosen = branch_index
                        break
            child = walk(branches[chosen]) if branches else _blank()
            return {"op": OP_ALT, "children": [child], "exemplars": [], "cls": 0, "min_n": 1, "max_n": 1}
        if kind == "repeat":
            child_schema = _schema_child(node)
            copies = []
            while index < len(leaves):
                if copies and leaves[index]["role"][0] != 1:
                    break
                expected = _schema_class(child_schema)
                if expected is not None and leaves[index]["start"] < len(text):
                    if structural_class(text[leaves[index]["start"]]) != expected:
                        break
                before = index
                copies.append(walk(child_schema))
                if index == before:
                    break
            unit = merge_nodes(copies) if copies else _blank()
            return {
                "op": OP_REPEAT,
                "children": [unit],
                "exemplars": [],
                "cls": 0,
                "min_n": int(node.get("min", 1)),
                "max_n": int(node.get("max", node.get("min", 1))),
                "bounds": True,
            }
        return _blank()

    return walk(schema)


def schema_from_episode(episode) -> dict:
    """Operator tree of the language. Built from support derivations only.

    A compiled schema is the generator's tree: repeat bounds come from that
    schema, so a one-copy support still carries the repeat operator. Atomic
    episodes have no schema object; their tree is lifted from the annotations.
    """
    dsl = _dsl_payload(episode)
    nested = []
    for slot, (text, ann) in enumerate(zip(episode.support, episode.support_ann)):
        if dsl is not None:
            nested.append(_bind_dsl(dsl, ann, text, slot))
            continue
        node = _nest(ann, text, slot)
        if node is not None:
            nested.append(node)
    return merge_nodes(nested)


def describe(node: dict) -> str:
    if node["op"] == OP_LEAF:
        return "L"
    if node["op"] == OP_REPEAT and node["children"]:
        return f"REP{node['min_n']}-{node['max_n']}({describe(node['children'][0])})"
    if node["op"] == OP_OPTIONAL and node["children"]:
        return f"OPT({describe(node['children'][0])})"
    if node["op"] == OP_ALT:
        return "ALT(" + ",".join(describe(child) for child in node["children"]) + ")"
    if node["op"] == OP_SEQ:
        return "SEQ(" + ",".join(describe(child) for child in node["children"]) + ")"
    return "?"


def _slot_values(tensor: torch.Tensor, slot: int, start: int, end: int) -> list[int]:
    limit = int(tensor.shape[-1])
    start = max(0, min(int(start), limit))
    end = max(start, min(int(end), limit))
    if end <= start:
        return []
    return [int(tensor[slot, index]) for index in range(start, end)]


def _const_offsets(node: dict, local: torch.Tensor, classes: torch.Tensor) -> list[int | None]:
    exemplars = node.get("exemplars") or []
    if not exemplars:
        return []
    width = min(end - start for _slot, start, end in exemplars)
    if width <= 0:
        return []
    fully = True
    lengths = {end - start for _slot, start, end in exemplars}
    if len(lengths) != 1:
        fully = False
    else:
        tuples = [tuple(_slot_values(local, slot, start, end)) for slot, start, end in exemplars]
        fully = len(set(tuples)) == 1
    offsets = []
    for offset in range(width):
        vals = []
        kinds = []
        for slot, start, end in exemplars:
            if start + offset >= end:
                continue
            vals.append(int(local[slot, start + offset]))
            kinds.append(int(classes[slot, start + offset]))
        if len(vals) != len(exemplars) or len(set(vals)) != 1:
            offsets.append(None)
            continue
        if fully or kinds[0] != DIGIT:
            offsets.append(vals[0])
        else:
            offsets.append(None)
    return offsets


class SeqOp(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.norm(left + right + self.ff(torch.cat([left, right], dim=-1)))


class RepeatOp(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.bound = nn.Linear(2, dim)

    def forward(self, unit: torch.Tensor, min_n: int, max_n: int) -> torch.Tensor:
        scale = unit.new_tensor([math.log1p(min_n) / math.log1p(8.0), math.log1p(max_n) / math.log1p(8.0)])
        bounds = self.bound(scale)
        return self.norm(unit + bounds + self.ff(torch.cat([unit, bounds], dim=-1)))


class OptionalOp(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.absent = nn.Parameter(torch.zeros(dim))

    def forward(self, child: torch.Tensor, present: bool) -> torch.Tensor:
        src = child if present else self.absent
        return self.norm(src + self.ff(src))


class AltOp(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        mixed = torch.logsumexp(torch.stack([left, right], dim=0), dim=0)
        return self.norm(mixed + self.ff(torch.cat([left, right], dim=-1)))


class OperatorOracle(SpanNet):
    """Gold schema, shared operator modules, relational leaf features."""

    def __init__(self):
        super().__init__()
        dim = self.d_model
        self.arch = "h3o"
        self.mem = nn.Identity()
        self.kind_head = nn.Identity()
        self.span_head = nn.Identity()
        self.bound_head = nn.Identity()
        self.role_head = nn.Identity()
        self.leaf = nn.Sequential(nn.Linear(dim * 4 + 4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.op_seq = SeqOp(dim)
        self.op_repeat = RepeatOp(dim)
        self.op_optional = OptionalOp(dim)
        self.op_alt = AltOp(dim)
        self.eos = nn.Parameter(torch.zeros(dim))
        self.head = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1))
        self.last_repeat_max = 0
        self.last_cursors: list[tuple[int, int]] = []

    def _pool_chars(self, encoded: torch.Tensor, start: int, end: int) -> torch.Tensor:
        if end <= start:
            return encoded.new_zeros(encoded.shape[-1])
        return encoded[start:end].mean(dim=0)

    def _pool_span(
        self,
        span_h: torch.Tensor,
        span_start: torch.Tensor,
        span_end: torch.Tensor,
        span_pad: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        chosen = []
        for index in range(span_h.shape[0]):
            if bool(span_pad[index]):
                continue
            left = int(span_start[index])
            right = int(span_end[index])
            if right > left and left >= start and right <= end:
                chosen.append(span_h[index])
        if not chosen:
            return span_h.new_zeros(span_h.shape[-1])
        return torch.stack(chosen).mean(dim=0)

    def _leaf_vector(
        self,
        query_h: torch.Tensor,
        query_span: torch.Tensor,
        bind_h: torch.Tensor,
        bind_span: torch.Tensor,
        id_agree: float,
        class_agree: float,
        len_ratio: float,
        present: float,
    ) -> torch.Tensor:
        feat = query_h.new_tensor([id_agree, class_agree, len_ratio, present])
        return self.leaf(torch.cat([query_h, query_span, bind_h, bind_span, feat], dim=-1))

    def _binding_views(self, node: dict, encoded, span_h, span_start, span_end, span_pad, local, classes, row: int):
        exemplars = node.get("exemplars") or [(0, 0, 1)]
        slot, start, end = exemplars[0]
        slot = min(slot, encoded.shape[1] - 1)
        end = min(end, encoded.shape[2])
        start = min(start, end)
        bind_h = self._pool_chars(encoded[row, slot], start, end)
        bind_span = self._pool_span(span_h[row, slot], span_start[row, slot], span_end[row, slot], span_pad[row, slot], start, end)
        exemplar_cls = _slot_values(classes[row], slot, start, end)
        exemplar_ids = _slot_values(local[row], slot, start, end)
        return bind_h, bind_span, exemplar_cls, exemplar_ids, _const_offsets(node, local[row], classes[row])

    def _score_span(self, exemplar_cls: list[int], exemplar_ids: list[int], const_offsets: list[int | None], q_cls: list[int], q_ids: list[int]):
        if not exemplar_cls or not q_cls:
            return 0.0, 0.0, 0.0
        e_runs = [run[2] for run in _runs(exemplar_cls)]
        q_runs = [run[2] for run in _runs(q_cls)]
        class_agree = sum(left == right for left, right in zip(e_runs, q_runs)) / max(len(e_runs), len(q_runs))
        checked = 0
        matched = 0
        for offset, ident in enumerate(const_offsets):
            if ident is None or offset >= len(q_ids):
                continue
            checked += 1
            matched += int(q_ids[offset] == ident)
        id_agree = 1.0 if checked == 0 else matched / checked
        len_ratio = 1.0 - min(1.0, abs(len(q_cls) - len(exemplar_cls)) / max(len(exemplar_cls), 1))
        return id_agree, class_agree, len_ratio

    def _consume_leaf(self, node, q_cls: list[int], cursor: int, local_row, classes_row):
        exemplars = node.get("exemplars") or []
        if not exemplars:
            return cursor, False
        _slot, start, end = exemplars[0]
        exemplar_cls = _slot_values(classes_row, _slot, start, end)
        runs = _runs(exemplar_cls)
        if not runs:
            return cursor, False
        widths = {b - a for _slot_i, a, b in exemplars}
        id_tuples = {tuple(_slot_values(local_row, slot_i, a, b)) for slot_i, a, b in exemplars}
        if len(id_tuples) == 1 and len(widths) == 1:
            width = end - start
            if cursor + width > len(q_cls):
                return cursor, False
            return cursor + width, True
        index = cursor
        for _begin, _stop, kind in runs:
            if index >= len(q_cls) or q_cls[index] != kind:
                return cursor, False
            nxt = index + 1
            while nxt < len(q_cls) and q_cls[nxt] == kind:
                nxt += 1
            index = nxt
        return index, index > cursor

    def _leaf_match(self, node, encoded, span_pack, local, classes, row, slot, cursor, length, bind):
        bind_h, bind_span, exemplar_cls, exemplar_ids, const_offsets = bind
        q_cls_all = _slot_values(classes[row], slot, 0, length)
        end, ok = self._consume_leaf(node, q_cls_all, cursor, local[row], classes[row])
        if not ok:
            zero_h = encoded.new_zeros(encoded.shape[-1])
            zero_s = span_pack[0].new_zeros(span_pack[0].shape[-1])
            return self._leaf_vector(zero_h, zero_s, bind_h, bind_span, 0.0, 0.0, 0.0, 0.0), cursor, False, 0.0
        q_cls = _slot_values(classes[row], slot, cursor, end)
        q_ids = _slot_values(local[row], slot, cursor, end)
        id_agree, class_agree, len_ratio = self._score_span(exemplar_cls, exemplar_ids, const_offsets, q_cls, q_ids)
        q_h = self._pool_chars(encoded[row, slot], cursor, end)
        q_span = self._pool_span(span_pack[0][row, slot], span_pack[1][row, slot], span_pack[2][row, slot], span_pack[3][row, slot], cursor, end)
        vector = self._leaf_vector(q_h, q_span, bind_h, bind_span, id_agree, class_agree, len_ratio, 1.0)
        return vector, end, True, id_agree + class_agree

    def _bind_leaf(self, node, encoded, span_pack, local, classes, row):
        bind_h, bind_span, _cls, _ids, _offs = self._binding_views(node, encoded, span_pack[0], span_pack[1], span_pack[2], span_pack[3], local, classes, row)
        return self._leaf_vector(bind_h, bind_span, bind_h, bind_span, 1.0, 1.0, 1.0, 1.0)

    def _content_class(self, node, classes, row) -> int:
        current = node
        while current["op"] != OP_LEAF and current["children"]:
            current = current["children"][-1]
        exemplars = current.get("exemplars") or []
        if not exemplars:
            return DIGIT
        slot, start, end = exemplars[0]
        values = _slot_values(classes[row], slot, start, end)
        return values[-1] if values else DIGIT

    def _starts_with_delim(self, node, classes, row) -> bool:
        current = node
        while current["op"] in (OP_SEQ, OP_REPEAT, OP_OPTIONAL) and current["children"]:
            current = current["children"][0]
        if current["op"] != OP_LEAF:
            return False
        exemplars = current.get("exemplars") or []
        if not exemplars:
            return False
        slot, start, end = exemplars[0]
        values = _slot_values(classes[row], slot, start, end)
        return bool(values) and values[0] == PUNCT

    def compose(self, node, mode, cursor, encoded, span_pack, local, classes, row, slot, length):
        if node["op"] == OP_LEAF:
            if mode == "bind":
                return self._bind_leaf(node, encoded, span_pack, local, classes, row), cursor, True, 2.0
            bind = self._binding_views(node, encoded, span_pack[0], span_pack[1], span_pack[2], span_pack[3], local, classes, row)
            vector, end, ok, score = self._leaf_match(node, encoded, span_pack, local, classes, row, slot, cursor, length, bind)
            return vector, end, ok, score
        if node["op"] == OP_SEQ:
            hidden = None
            score = 0.0
            ok_all = True
            for child in node["children"]:
                vector, cursor, ok, child_score = self.compose(child, mode, cursor, encoded, span_pack, local, classes, row, slot, length)
                score += child_score
                hidden = vector if hidden is None else self.op_seq(hidden, vector)
                if mode == "query" and not ok:
                    ok_all = False
                    break
            if hidden is None:
                hidden = encoded.new_zeros(self.d_model)
            return hidden, cursor, ok_all if mode == "query" else True, score
        if node["op"] == OP_OPTIONAL and node["children"]:
            child = node["children"][0]
            if mode == "bind":
                vector, _, _, _ = self.compose(child, mode, cursor, encoded, span_pack, local, classes, row, slot, length)
                return self.op_optional(vector, True), cursor, True, 2.0
            probe, _probe_end, ok, probe_score = self.compose(child, mode, cursor, encoded, span_pack, local, classes, row, slot, length)
            if ok and probe_score > 0:
                return self.op_optional(probe, True), _probe_end, True, probe_score
            return self.op_optional(self.op_optional.absent, False), cursor, True, 0.0
        if node["op"] == OP_ALT and node["children"]:
            if mode == "bind":
                parts = []
                for child in node["children"][:2]:
                    vector, _, _, _ = self.compose(child, mode, cursor, encoded, span_pack, local, classes, row, slot, length)
                    parts.append(vector)
                while len(parts) < 2:
                    parts.append(parts[0])
                return self.op_alt(parts[0], parts[1]), cursor, True, 2.0
            best = None
            parts = []
            for child in node["children"][:2]:
                vector, end, ok, score = self.compose(child, mode, cursor, encoded, span_pack, local, classes, row, slot, length)
                parts.append(vector)
                if ok and (best is None or score > best[0]):
                    best = (score, end)
            while len(parts) < 2:
                parts.append(parts[0])
            advance = best[1] if best else cursor
            return self.op_alt(parts[0], parts[1]), advance, best is not None, best[0] if best else 0.0
        if node["op"] == OP_REPEAT and node["children"]:
            self.last_repeat_max = max(self.last_repeat_max, int(node["max_n"]))
            unit = node["children"][0]
            if mode == "bind":
                vector, _, _, _ = self.compose(unit, mode, cursor, encoded, span_pack, local, classes, row, slot, length)
                return self.op_repeat(vector, int(node["min_n"]), int(node["max_n"])), cursor, True, 2.0
            copies = []
            q_cls = _slot_values(classes[row], slot, 0, length)
            if self._starts_with_delim(unit, classes, row) and cursor < length and q_cls[cursor] == self._content_class(unit, classes, row):
                nxt = cursor + 1
                while nxt < length and q_cls[nxt] == q_cls[cursor]:
                    nxt += 1
                bind = self._binding_views(unit if unit["op"] == OP_LEAF else _leaf_descendant(unit), encoded, span_pack[0], span_pack[1], span_pack[2], span_pack[3], local, classes, row)
                q_h = self._pool_chars(encoded[row, slot], cursor, nxt)
                q_span = self._pool_span(span_pack[0][row, slot], span_pack[1][row, slot], span_pack[2][row, slot], span_pack[3][row, slot], cursor, nxt)
                copies.append(self._leaf_vector(q_h, q_span, bind[0], bind[1], 1.0, 1.0, 1.0, 1.0))
                cursor = nxt
            for _ in range(8):
                vector, nxt, ok, _score = self.compose(unit, mode, cursor, encoded, span_pack, local, classes, row, slot, length)
                if not ok or nxt <= cursor:
                    break
                copies.append(vector)
                cursor = nxt
            if not copies:
                zero = encoded.new_zeros(self.d_model)
                return self.op_repeat(zero, int(node["min_n"]), int(node["max_n"])), cursor, False, 0.0
            unit_vec = torch.stack(copies).mean(dim=0)
            return self.op_repeat(unit_vec, int(node["min_n"]), int(node["max_n"])), cursor, True, 1.0
        zero = encoded.new_zeros(self.d_model)
        return zero, cursor, False, 0.0

    def forward(self, local, classes, rel, pad, episodes) -> torch.Tensor:
        self.last_repeat_max = 0
        self.last_cursors = []
        _batch, slots, length = local.shape
        roles = torch.zeros(slots, dtype=torch.long, device=local.device)
        roles[-1] = 1
        which = torch.arange(slots, device=local.device)
        hidden = self.class_emb(classes) + self.local_emb(local) + self.rel_proj(rel)
        hidden = hidden + self.char_pos[:length][None, None, :, :]
        hidden = hidden + self.role(roles)[None, :, None, :] + self.which(which)[None, :, None, :]
        hidden = hidden * (~pad).unsqueeze(-1).float()
        encoded = _attend(self.char_encoder, hidden, pad)
        span_start, span_end, span_feat, span_pad, pos_index = propose_spans(classes, pad, self.max_spans)
        pooled = _pool_spans(encoded, span_start, span_end, span_pad)
        span_h = pooled + self.span_proj(span_feat) + self.span_pos[pos_index]
        span_h = span_h * (~span_pad).unsqueeze(-1).float()
        span_h = _attend(self.span_encoder, span_h, span_pad)
        span_pack = (span_h, span_start, span_end, span_pad)
        logits = []
        for row, episode in enumerate(episodes):
            schema = schema_from_episode(episode)
            query_slot = slots - 1
            z_lang, _, _, _ = self.compose(schema, "bind", 0, encoded, span_pack, local, classes, row, 0, length)
            q_len = int((~pad[row, query_slot]).sum())
            z_query, cursor, _ok, _score = self.compose(schema, "query", 0, encoded, span_pack, local, classes, row, query_slot, q_len)
            self.last_cursors.append((int(cursor), int(q_len)))
            if cursor >= q_len:
                end_vec = self.eos
            else:
                end_vec = encoded.new_zeros(self.d_model)
            z_lang = self.op_seq(z_lang, self.eos)
            z_query = self.op_seq(z_query, end_vec)
            delta = (z_lang - z_query).abs()
            logits.append(self.head(torch.cat([z_lang, z_query, delta], dim=-1)).squeeze(-1))
        return torch.stack(logits)


def _leaf_descendant(node: dict) -> dict:
    current = node
    while current["op"] != OP_LEAF and current["children"]:
        current = current["children"][-1]
    return current


def operator_loss(logit: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    membership = F.binary_cross_entropy_with_logits(logit, labels)
    positive = logit[labels > 0.5]
    negative = logit[labels < 0.5]
    if positive.numel() and negative.numel():
        rank = F.softplus(negative[:, None] - positive[None, :]).mean()
    else:
        rank = logit.new_zeros(())
    total = membership + 0.5 * rank
    return total, membership.detach()
