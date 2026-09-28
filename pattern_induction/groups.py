"""Gold operator trees and the groups the hierarchical models read.

The generator's character spans already mark repetition, optionality, and
alternation. This module lifts those spans into a small tree: a sequence of
groups, a repeat node over copies of one unit, and optional or alt wrappers.
The model never sees that tree at scoring time. It proposes groups from
structural class and the frozen alignment relations, then the loss asks those
groups to match the tree.

A within-class rename leaves classes and relations unchanged, so the proposed
graph is the same graph.
"""

from __future__ import annotations

import math

import torch

from pattern_induction.compose import (
    KIND_CLS,
    KIND_DELIM,
    KIND_OPT,
    KIND_REPEAT,
    CharAnn,
    AnnotatedEpisode,
)
from pattern_induction.renaming import LOWER, PUNCT, REL_LEFT_CONSTANT, REL_RIGHT_CONSTANT, UPPER

OP_LEAF = 0
OP_SEQ = 1
OP_REPEAT = 2
OP_OPTIONAL = 3
OP_ALT = 4
N_OP = 5

EDGE_NONE = 0
EDGE_CHILD = 1
EDGE_NEXT = 2
EDGE_SAME = 3
N_EDGE = 4

REL_CHILD = 0
REL_PARENT = 1
REL_NEXT = 2
REL_SAME = 3
REL_ALIGN = 4
N_REL_G = 5

MAX_GROUPS = 16
MAX_EDGES = 192
MAX_PAIRS = 96

G_CLASS_L = 0
G_CLASS_R = 1
G_LOG_LEN = 2
G_LOG_UNITS = 3
G_IS_ROOT = 4
G_IS_REPEAT = 5
G_IS_LEAF = 6
G_IS_PREFIX = 7
G_OPTIONAL = 8
G_ALT = 9
G_SIB = 10
G_ENDPOINT = 11
N_GFEAT = 12


class GoldNode:
    def __init__(self, start: int, end: int, op: int, parent: int, sibling: int, unit: int):
        self.start = start
        self.end = end
        self.op = op
        self.parent = parent
        self.sibling = sibling
        self.unit = unit


def _leaves(ann: CharAnn) -> list[dict]:
    leaves = []
    index = 0
    size = len(ann.span)
    while index < size:
        if ann.span[index] < 0:
            index += 1
            continue
        stop = index + 1
        span_id = ann.span[index]
        while stop < size and ann.span[stop] == span_id:
            stop += 1
        leaves.append(
            {
                "start": index,
                "end": stop,
                "kinds": tuple(ann.kind[cursor] for cursor in range(index, stop)),
                "role": tuple(ann.role[index]),
            }
        )
        index = stop
    return leaves


def _pattern(leaf: dict) -> tuple[int, ...]:
    """Kind pattern with repeated content collapsed, so a 1-digit copy matches a 2-digit copy."""
    mapped = []
    for kind in leaf["kinds"]:
        if kind == KIND_REPEAT:
            norm = KIND_CLS
        elif kind == KIND_OPT:
            norm = KIND_DELIM
        else:
            norm = kind
        if mapped and mapped[-1] == norm:
            continue
        mapped.append(norm)
    return tuple(mapped)


def _shape_a(leaves: list[dict], index: int) -> tuple[int, int] | None:
    """Equal copies. The first copy is unmarked and every later copy is marked repeat."""
    remain = len(leaves) - index
    best: tuple[int, int] | None = None
    for width in range(1, remain + 1):
        copies = remain // width
        if copies < 2:
            continue
        pattern = [_pattern(leaves[index + cursor]) for cursor in range(width)]
        if any(leaves[index + cursor]["role"][0] == 1 for cursor in range(width)):
            continue
        taken = 1
        while taken < copies:
            base = index + taken * width
            if any(leaves[base + cursor]["role"][0] != 1 for cursor in range(width)):
                break
            if [_pattern(leaves[base + cursor]) for cursor in range(width)] != pattern:
                break
            taken += 1
        if taken < 2:
            continue
        covered = taken * width
        if best is None or covered > best[0] or (covered == best[0] and width < best[1]):
            best = (covered, width)
    return best


def _shape_b(leaves: list[dict], index: int) -> int | None:
    """One unmarked first piece, then copies that each carry the repeat role.

    The numeric repeat language paints the first number without a separator and
    every later number with its separator, so the kind pattern does not match.
    """
    if leaves[index]["role"][0] == 1:
        return None
    if index + 1 >= len(leaves) or leaves[index + 1]["role"][0] != 1:
        return None
    pattern = _pattern(leaves[index + 1])
    stop = index + 1
    while stop < len(leaves) and leaves[stop]["role"][0] == 1 and _pattern(leaves[stop]) == pattern:
        stop += 1
    if stop == index + 1:
        return None
    return stop - index


def lift_tree(ann: CharAnn | None) -> list[GoldNode]:
    """Sequence root over leaves, with repeat / optional / alt nodes where the roles require them."""
    if ann is None or not any(span >= 0 for span in ann.span):
        return []
    leaves = _leaves(ann)
    if not leaves:
        return []
    pieces: list[tuple] = []
    index = 0
    while index < len(leaves):
        role = leaves[index]["role"]
        if role[1] == 1 and role[0] == 0:
            stop = index + 1
            while stop < len(leaves) and leaves[stop]["role"][1] == 1 and leaves[stop]["role"][0] == 0:
                stop += 1
            pieces.append(("opt", leaves[index:stop]))
            index = stop
            continue
        if role[2] == 1 and role[0] == 0:
            stop = index + 1
            while stop < len(leaves) and leaves[stop]["role"][2] == 1 and leaves[stop]["role"][0] == 0:
                stop += 1
            pieces.append(("alt", leaves[index:stop]))
            index = stop
            continue
        shaped = _shape_a(leaves, index)
        marked = _shape_b(leaves, index)
        if shaped is not None and (marked is None or shaped[0] >= marked):
            covered, width = shaped
            units = [leaves[index + copy * width : index + (copy + 1) * width] for copy in range(covered // width)]
            pieces.append(("repeat", units))
            index += covered
            continue
        if marked is not None:
            units = [[leaves[index]]] + [[leaves[index + 1 + copy]] for copy in range(marked - 1)]
            pieces.append(("repeat", units))
            index += marked
            continue
        pieces.append(("leaf", [leaves[index]]))
        index += 1

    nodes: list[GoldNode] = []

    def add(start: int, end: int, op: int, parent: int, sibling: int, unit: int) -> int:
        nodes.append(GoldNode(start, end, op, parent, sibling, unit))
        return len(nodes) - 1

    root = add(leaves[0]["start"], leaves[-1]["end"], OP_SEQ, -1, 0, -1)

    def add_leaves(group: list[dict], parent: int, unit: int) -> None:
        if len(group) == 1:
            leaf = group[0]
            add(leaf["start"], leaf["end"], OP_LEAF, parent, 0, unit)
            return
        seq = add(group[0]["start"], group[-1]["end"], OP_SEQ, parent, 0, unit)
        for sibling, leaf in enumerate(group):
            add(leaf["start"], leaf["end"], OP_LEAF, seq, sibling, -1)

    for sibling, piece in enumerate(pieces):
        kind, payload = piece
        if kind == "leaf":
            leaf = payload[0]
            add(leaf["start"], leaf["end"], OP_LEAF, root, sibling, -1)
        elif kind == "opt":
            node = add(payload[0]["start"], payload[-1]["end"], OP_OPTIONAL, root, sibling, -1)
            add_leaves(payload, node, -1)
        elif kind == "alt":
            node = add(payload[0]["start"], payload[-1]["end"], OP_ALT, root, sibling, -1)
            add_leaves(payload, node, -1)
        else:
            units = payload
            node = add(units[0][0]["start"], units[-1][-1]["end"], OP_REPEAT, root, sibling, -1)
            for unit_index, unit_leaves in enumerate(units):
                if len(unit_leaves) == 1:
                    leaf = unit_leaves[0]
                    add(leaf["start"], leaf["end"], OP_LEAF, node, unit_index, node)
                else:
                    seq = add(unit_leaves[0]["start"], unit_leaves[-1]["end"], OP_SEQ, node, unit_index, node)
                    for child_index, leaf in enumerate(unit_leaves):
                        add(leaf["start"], leaf["end"], OP_LEAF, seq, child_index, -1)
    return nodes


def scope_labels(ann: CharAnn | None, length: int) -> list[int]:
    """Deepest repeat, optional, or alt operator covering each character, else the sequence."""
    tree = lift_tree(ann)
    labels = [-1] * length
    if ann is None:
        return labels
    limit = min(length, len(ann.span))
    for index in range(limit):
        if ann.span[index] < 0:
            continue
        covering = [node for node in tree if node.op != OP_LEAF and node.start <= index < node.end]
        if not covering:
            labels[index] = OP_LEAF
            continue
        covering.sort(key=lambda node: (0 if node.op != OP_SEQ else 1, node.end - node.start))
        labels[index] = covering[0].op
    return labels


def _runs(classes: list[int], cuts: set[int]) -> list[tuple[int, int, int]]:
    runs = []
    index = 0
    size = len(classes)
    while index < size:
        stop = index + 1
        while stop < size and classes[stop] == classes[index] and stop not in cuts:
            stop += 1
        runs.append((index, stop, classes[index]))
        index = stop
    return runs


def _literal_span(classes: list[int], flags: list[float], forward: bool) -> int:
    if not classes:
        return 0
    order = range(len(classes)) if forward else range(len(classes) - 1, -1, -1)
    start = 0 if forward else len(classes) - 1
    if flags[start] < 0.5 or classes[start] not in (UPPER, LOWER):
        return 0
    kind = classes[start]
    taken = 0
    for index in order:
        if flags[index] < 0.5 or classes[index] != kind:
            break
        taken += 1
    if taken >= len(classes):
        return 0
    return taken


def _delimited(runs: list[tuple[int, int, int]], index: int) -> int | None:
    if index >= len(runs):
        return None
    content = runs[index][2]
    if index + 2 >= len(runs):
        return None
    if runs[index + 1][2] != PUNCT or content == PUNCT:
        return None
    stop = index + 1
    units = 1
    while stop + 1 < len(runs) and runs[stop][2] == PUNCT and runs[stop + 1][2] == content:
        stop += 2
        units += 1
    if units < 2:
        return None
    return stop


def _endpoint(index: int, count: int) -> float:
    if count <= 1:
        return 0.0
    if index == 0:
        return -1.0
    if index == count - 1:
        return 1.0
    return 0.0


def _sib(index: int, count: int) -> float:
    if count <= 1:
        return 0.0
    return index / (count - 1)


def propose_slot(classes: list[int], left_constant: list[float], right_constant: list[float]) -> list[dict]:
    """One string's groups. Root is first. Repeat copies are children, not a longer flat chain."""
    if not classes:
        return []
    prefix = _literal_span(classes, left_constant, True)
    suffix = _literal_span(classes, right_constant, False)
    if prefix + suffix >= len(classes):
        suffix = 0
    cuts: set[int] = set()
    if prefix:
        cuts.add(prefix)
    if suffix:
        cuts.add(len(classes) - suffix)
    runs = _runs(classes, cuts)
    pieces: list[tuple[str, list[tuple[int, int, int]]]] = []
    index = 0
    while index < len(runs):
        run = runs[index]
        if prefix and run[1] <= prefix:
            pieces.append(("prefix", [run]))
            index += 1
            continue
        if suffix and run[0] >= len(classes) - suffix:
            pieces.append(("suffix", [run]))
            index += 1
            continue
        stop = _delimited(runs, index)
        if stop is not None and suffix:
            while stop > index and runs[stop - 1][0] >= len(classes) - suffix:
                stop -= 1
            if stop <= index + 2:
                stop = None
        if stop is not None:
            pieces.append(("repeat", runs[index:stop]))
            index = stop
            continue
        pieces.append(("leaf", [run]))
        index += 1

    nodes: list[dict] = [
        {
            "start": 0,
            "end": len(classes),
            "parent": -1,
            "kind": "root",
            "units": 0,
            "prefix": 0.0,
            "optional": 0.0,
            "alt": 0.0,
            "sib": 0.0,
            "endpoint": 0.0,
            "class_l": runs[0][2] if runs else 0,
            "class_r": runs[-1][2] if runs else 0,
        }
    ]
    top_ids: list[int] = []
    for kind, group in pieces:
        if kind == "repeat":
            content_runs = [run for run in group if run[2] != PUNCT]
            if len(content_runs) < 2:
                kind = "leaf"
                group = [group[0]]
        if kind == "repeat":
            parent = len(nodes)
            units = [group[0]]
            cursor = 1
            while cursor + 1 < len(group):
                units.append((group[cursor][0], group[cursor + 1][1], group[cursor + 1][2]))
                cursor += 2
            nodes.append(
                {
                    "start": group[0][0],
                    "end": group[-1][1],
                    "parent": 0,
                    "kind": "repeat",
                    "units": len(units),
                    "prefix": 0.0,
                    "optional": 0.0,
                    "alt": 0.0,
                    "sib": 0.0,
                    "endpoint": 0.0,
                    "class_l": group[0][2],
                    "class_r": group[-1][2],
                }
            )
            top_ids.append(parent)
            for unit_index, unit in enumerate(units):
                if len(nodes) >= MAX_GROUPS:
                    break
                nodes.append(
                    {
                        "start": unit[0],
                        "end": unit[1],
                        "parent": parent,
                        "kind": "leaf",
                        "units": 0,
                        "prefix": 0.0,
                        "optional": 0.0,
                        "alt": 0.0,
                        "sib": _sib(unit_index, len(units)),
                        "endpoint": _endpoint(unit_index, len(units)),
                        "class_l": classes[unit[0]],
                        "class_r": classes[unit[1] - 1],
                    }
                )
            continue
        if len(nodes) >= MAX_GROUPS:
            continue
        run = group[0]
        optional = 1.0 if run[2] == PUNCT and kind != "prefix" else 0.0
        nodes.append(
            {
                "start": run[0],
                "end": run[1],
                "parent": 0,
                "kind": "leaf",
                "units": 0,
                "prefix": 1.0 if kind == "prefix" else 0.0,
                "optional": optional,
                "alt": 0.0,
                "sib": 0.0,
                "endpoint": 0.0,
                "class_l": run[2],
                "class_r": run[2],
            }
        )
        top_ids.append(len(nodes) - 1)
    if len(top_ids) == 1 and nodes[top_ids[0]]["kind"] == "leaf" and nodes[top_ids[0]]["end"] - nodes[top_ids[0]]["start"] == len(classes):
        nodes[top_ids[0]]["alt"] = 1.0
    count = len(top_ids)
    for index, node_id in enumerate(top_ids):
        nodes[node_id]["sib"] = _sib(index, count)
        nodes[node_id]["endpoint"] = _endpoint(index, count)
    nodes[0]["units"] = count
    return nodes[:MAX_GROUPS]


def _live_chars(classes: torch.Tensor, pad: torch.Tensor, left: torch.Tensor, right: torch.Tensor, row: int, slot: int):
    length = classes.shape[-1]
    chars = []
    left_flags = []
    right_flags = []
    for index in range(length):
        if bool(pad[row, slot, index]):
            break
        chars.append(int(classes[row, slot, index]))
        left_flags.append(float(left[row, slot, index]))
        right_flags.append(float(right[row, slot, index]))
    return chars, left_flags, right_flags


def _featurize_node(node: dict, scale: float) -> list[float]:
    return [
        node["class_l"] / 4.0,
        node["class_r"] / 4.0,
        math.log1p(max(node["end"] - node["start"], 0)) / math.log1p(scale),
        math.log1p(node["units"]) / math.log1p(8.0),
        1.0 if node["kind"] == "root" else 0.0,
        1.0 if node["kind"] == "repeat" else 0.0,
        1.0 if node["kind"] == "leaf" else 0.0,
        node["prefix"],
        node["optional"],
        node["alt"],
        node["sib"],
        node["endpoint"],
    ]


def _signature(feat: list[float]) -> tuple:
    return (
        round(feat[G_CLASS_L] * 4),
        round(feat[G_CLASS_R] * 4),
        int(feat[G_IS_REPEAT] > 0.5),
        int(feat[G_IS_LEAF] > 0.5),
        int(feat[G_IS_ROOT] > 0.5),
        int(feat[G_IS_PREFIX] > 0.5),
    )


def propose_groups(
    classes: torch.Tensor,
    rel: torch.Tensor,
    pad: torch.Tensor,
    max_groups: int = MAX_GROUPS,
) -> dict[str, torch.Tensor]:
    """Batch of group nodes and typed edges. Edges are a function of class and alignment only."""
    batch, slots, length = classes.shape
    device = classes.device
    start = torch.zeros(batch, slots, max_groups, dtype=torch.long, device=device)
    end = torch.zeros(batch, slots, max_groups, dtype=torch.long, device=device)
    feat = torch.zeros(batch, slots, max_groups, N_GFEAT, device=device)
    gpad = torch.ones(batch, slots, max_groups, dtype=torch.bool, device=device)
    edge_src = torch.zeros(batch, MAX_EDGES, dtype=torch.long, device=device)
    edge_dst = torch.zeros(batch, MAX_EDGES, dtype=torch.long, device=device)
    edge_rel = torch.zeros(batch, MAX_EDGES, dtype=torch.long, device=device)
    edge_pad = torch.ones(batch, MAX_EDGES, dtype=torch.bool, device=device)
    left = rel[:, :, :, REL_LEFT_CONSTANT]
    right = rel[:, :, :, REL_RIGHT_CONSTANT]
    for row in range(batch):
        row_sigs: list[tuple] = []
        edges: list[tuple[int, int, int]] = []
        for slot in range(slots):
            chars, left_flags, right_flags = _live_chars(classes, pad, left, right, row, slot)
            nodes = propose_slot(chars, left_flags, right_flags) if chars else []
            for index, node in enumerate(nodes):
                if index >= max_groups:
                    break
                start[row, slot, index] = node["start"]
                end[row, slot, index] = node["end"]
                gpad[row, slot, index] = False
                values = _featurize_node(node, float(length))
                feat[row, slot, index] = torch.tensor(values, device=device)
                row_sigs.append(_signature(values))
            for index, node in enumerate(nodes):
                if index >= max_groups or node["parent"] < 0 or node["parent"] >= max_groups:
                    continue
                src = slot * max_groups + index
                dst = slot * max_groups + node["parent"]
                edges.append((src, dst, REL_CHILD))
                edges.append((dst, src, REL_PARENT))
            children: dict[int, list[int]] = {}
            for index, node in enumerate(nodes):
                if index >= max_groups or node["parent"] < 0:
                    continue
                children.setdefault(node["parent"], []).append(index)
            for parent, kids in children.items():
                ordered = sorted(kids, key=lambda item: int(start[row, slot, item]))
                for left_i, right_i in zip(ordered, ordered[1:]):
                    edges.append((slot * max_groups + left_i, slot * max_groups + right_i, REL_NEXT))
                if parent < len(nodes) and nodes[parent]["kind"] == "repeat" and ordered:
                    first = ordered[0]
                    for other in ordered[1:]:
                        edges.append((slot * max_groups + other, slot * max_groups + first, REL_SAME))
        query = slots - 1
        sig_at = [[None for _ in range(max_groups)] for _ in range(slots)]
        cursor = 0
        for slot in range(slots):
            for index in range(max_groups):
                if bool(gpad[row, slot, index]):
                    continue
                sig_at[slot][index] = row_sigs[cursor]
                cursor += 1
        for index in range(max_groups):
            if sig_at[query][index] is None:
                continue
            matched = 0
            for slot in range(slots - 1):
                for other in range(max_groups):
                    if sig_at[slot][other] is None or sig_at[slot][other] != sig_at[query][index]:
                        continue
                    edges.append((query * max_groups + index, slot * max_groups + other, REL_ALIGN))
                    edges.append((slot * max_groups + other, query * max_groups + index, REL_ALIGN))
                    matched += 1
                    break
                if matched >= 2:
                    break
        for edge_index, (src, dst, rel_id) in enumerate(edges[:MAX_EDGES]):
            edge_src[row, edge_index] = src
            edge_dst[row, edge_index] = dst
            edge_rel[row, edge_index] = rel_id
            edge_pad[row, edge_index] = False
    return {
        "start": start,
        "end": end,
        "feat": feat,
        "pad": gpad,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_rel": edge_rel,
        "edge_pad": edge_pad,
    }


def _align(start: int, end: int, tree: list[GoldNode]) -> int:
    if end <= start or not tree:
        return -1
    best_index = -1
    best_rank = None
    span = end - start
    for index, node in enumerate(tree):
        overlap = max(0, min(end, node.end) - max(start, node.start))
        if overlap <= 0:
            continue
        score = overlap / span + overlap / max(node.end - node.start, 1)
        specific = 1 if node.op in (OP_REPEAT, OP_OPTIONAL, OP_ALT) else 0
        rank = (score, -(node.end - node.start), specific)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_index = index
    if best_rank is None or best_rank[0] < 1.0:
        return -1
    return best_index


def _relation(tree: list[GoldNode], src: int, dst: int) -> int:
    if src < 0 or dst < 0 or not tree:
        return -1
    left = tree[src]
    right = tree[dst]
    if left.parent == dst:
        return EDGE_CHILD
    if left.parent >= 0 and left.parent == right.parent and right.sibling == left.sibling + 1:
        return EDGE_NEXT
    if left.unit >= 0 and left.unit == right.unit and src != dst:
        return EDGE_SAME
    return EDGE_NONE


def group_targets(
    episodes: list[AnnotatedEpisode],
    classes: torch.Tensor,
    rel: torch.Tensor,
    pad: torch.Tensor,
    max_len: int,
    max_groups: int = MAX_GROUPS,
) -> dict[str, torch.Tensor]:
    """Align proposed groups to the generator tree. Unaligned pairs are ignored by the loss."""
    proposed = propose_groups(classes, rel, pad, max_groups)
    batch, slots, _length = classes.shape
    op = torch.full((batch, slots, max_groups), -1, dtype=torch.long)
    scope = torch.full((batch, slots, max_len), -1, dtype=torch.long)
    pair_src = torch.zeros(batch, MAX_PAIRS, dtype=torch.long)
    pair_dst = torch.zeros(batch, MAX_PAIRS, dtype=torch.long)
    pair_label = torch.full((batch, MAX_PAIRS), -1, dtype=torch.long)
    anns = []
    for episode in episodes:
        row_ann = list(episode.support_ann[: slots - 1])
        while len(row_ann) < slots - 1:
            row_ann.append(None)
        row_ann.append(episode.query_ann)
        anns.append(row_ann)
    for row in range(batch):
        trees = []
        aligned = []
        for slot in range(slots):
            ann = anns[row][slot]
            labels = scope_labels(ann, max_len)
            for index, label in enumerate(labels):
                scope[row, slot, index] = label
            tree = lift_tree(ann)
            trees.append(tree)
            slot_align = []
            for index in range(max_groups):
                if bool(proposed["pad"][row, slot, index]):
                    slot_align.append(-1)
                    continue
                gold = _align(int(proposed["start"][row, slot, index]), int(proposed["end"][row, slot, index]), tree)
                slot_align.append(gold)
                if gold >= 0:
                    op[row, slot, index] = tree[gold].op
            aligned.append(slot_align)
        pairs: list[tuple[int, int, int]] = []
        for edge_index in range(MAX_EDGES):
            if bool(proposed["edge_pad"][row, edge_index]):
                continue
            if int(proposed["edge_rel"][row, edge_index]) == REL_ALIGN:
                continue
            src = int(proposed["edge_src"][row, edge_index])
            dst = int(proposed["edge_dst"][row, edge_index])
            src_slot, src_i = divmod(src, max_groups)
            dst_slot, dst_i = divmod(dst, max_groups)
            if src_slot != dst_slot or src_slot >= slots:
                continue
            label = _relation(trees[src_slot], aligned[src_slot][src_i], aligned[dst_slot][dst_i])
            pairs.append((src, dst, label))
        for slot in range(slots):
            live = [index for index in range(max_groups) if not bool(proposed["pad"][row, slot, index])]
            for left_i, right_i in zip(live, live[2:]):
                src = slot * max_groups + left_i
                dst = slot * max_groups + right_i
                if any(pair[0] == src and pair[1] == dst for pair in pairs):
                    continue
                label = _relation(trees[slot], aligned[slot][left_i], aligned[slot][right_i])
                pairs.append((src, dst, label))
                if len(pairs) >= MAX_PAIRS:
                    break
        for pair_index, (src, dst, label) in enumerate(pairs[:MAX_PAIRS]):
            pair_src[row, pair_index] = src
            pair_dst[row, pair_index] = dst
            pair_label[row, pair_index] = label
    return {
        "op": op,
        "scope": scope,
        "pair_src": pair_src,
        "pair_dst": pair_dst,
        "pair_label": pair_label,
        "proposed": proposed,
    }
