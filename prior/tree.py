"""Pack a candidate program as a tree the prior can compose.

Execution stays in the symbolic interpreter. This module only records the
nodes, their semantic parameters, and a few relations to the supports.
Letter identity is not a parameter: a within-class rename of a prefix leaves
the pack unchanged. Digit endpoints and lexicon names stay semantic.
"""

from __future__ import annotations

import json
import math

import torch

from pattern_induction.charset import charset_spec, spec_chars
from pattern_induction.latent import _CLASS_CHARS
from pattern_induction.program import Prog
from prior.features import _CONVENTION_SET, _FAMILY_CHARS, _LEX_GROUP, _literal, _round_interval, _support_view

PAD = 0
LITERAL = 1
CHARSET = 2
RANGE = 3
LEXICON = 4
OPTIONAL = 5
REPEAT = 6
SEQ = 7
ALT = 8

PARAM_DIM = 8
MAX_NODES = 20
REL_DIM = 8
LEX_NAMES = ("",) + tuple(sorted(_LEX_GROUP))
LEX_INDEX = {name: index for index, name in enumerate(LEX_NAMES)}


def dump_prog(prog: Prog) -> list:
    return [
        prog.op,
        prog.slot,
        prog.start,
        prog.end,
        prog.cls,
        prog.lo,
        prog.hi,
        prog.spec,
        [dump_prog(child) for child in prog.children],
    ]


def load_prog(row: list) -> Prog:
    op, slot, start, end, cls, lo, hi, spec, children = row
    return Prog(
        op,
        slot=int(slot),
        start=int(start),
        end=int(end),
        cls=cls,
        lo=int(lo),
        hi=int(hi),
        spec=spec,
        children=[load_prog(child) for child in children],
    )


def programs_json(programs: list[Prog]) -> str:
    return json.dumps([dump_prog(prog) for prog in programs], separators=(",", ":"))


def programs_from_json(text: str) -> list[Prog]:
    return [load_prog(row) for row in json.loads(text)]


def _binarize(node: Prog) -> Prog:
    children = [_binarize(child) for child in node.children]
    if node.op in {"SEQ", "ALT"} and len(children) > 2:
        acc = children[0]
        for child in children[1:]:
            acc = Prog(node.op, children=[acc, child])
        return acc
    if node.op in {"SEQ", "ALT"} and len(children) == 1:
        return children[0]
    return Prog(
        node.op,
        slot=node.slot,
        start=node.start,
        end=node.end,
        cls=node.cls,
        lo=node.lo,
        hi=node.hi,
        spec=node.spec,
        children=children,
    )


def _log(value: float) -> float:
    return math.log10(value + 1.0) / 5.0


def _leaf_params(node: Prog, supports: list[str]) -> tuple[int, list[float]]:
    if node.op == "COPY":
        text = _literal(node, supports)
        digit = float(text.isdigit()) if text else 0.0
        value = _log(float(int(text))) if digit and text else 0.0
        params = [
            len(text) / 8.0,
            digit,
            float(bool(text) and text.isupper()),
            float(bool(text) and text.islower()),
            float(bool(text) and not text.isalnum()),
            value,
            float(bool(digit) and len(text) > 1 and text.startswith("0")),
            node.slot / 4.0,
        ]
        return LITERAL, params
    if node.op in {"CLS", "SET"}:
        if node.op == "CLS":
            chars = _CLASS_CHARS.get(node.cls, set())
            family = 1.0 if node.cls in {"digit", "upper", "lower"} else 0.0
        else:
            chars = spec_chars(node.spec)
            family = float(bool(chars) and (charset_spec(chars) in _FAMILY_CHARS or node.spec in _FAMILY_CHARS))
        kind_digit = float(bool(chars) and chars <= set("0123456789"))
        kind_upper = float(bool(chars) and chars <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        kind_lower = float(bool(chars) and chars <= set("abcdefghijklmnopqrstuvwxyz"))
        params = [
            node.lo / 8.0,
            min(node.hi, 128) / 32.0,
            float(node.lo == node.hi),
            float(node.hi >= 64),
            _log(float(len(chars))),
            family,
            kind_digit + 0.5 * kind_upper + 0.25 * kind_lower,
            node.slot / 4.0,
        ]
        return CHARSET, params
    if node.op == "INT" and ":" in node.spec:
        lo, hi = (int(part) for part in node.spec.split(":"))
        params = [
            _log(float(lo)),
            _log(float(hi)),
            _log(float(max(hi - lo, 0))),
            node.lo / 8.0,
            min(node.hi, 128) / 32.0,
            float(node.lo == node.hi),
            float((lo, hi) in _CONVENTION_SET),
            float(_round_interval(lo, hi)),
        ]
        return RANGE, params
    if node.op == "LEX":
        params = [float(LEX_INDEX.get(node.spec, 0))] + [0.0] * (PARAM_DIM - 1)
        return LEXICON, params
    return PAD, [0.0] * PARAM_DIM


def _op_params(node: Prog) -> list[float]:
    if node.op == "REPEAT":
        return [
            node.lo / 8.0,
            min(node.hi, 8) / 8.0,
            float(node.lo == node.hi),
            float(node.lo <= 1 and node.hi >= 8),
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    return [0.0] * PARAM_DIM


def relation_features(prog: Prog, supports: list[str]) -> list[float]:
    """Relations between one program and the strings it was proposed from."""
    view = _support_view(supports)
    covers = hull = convention = 0.0
    set_cover = set_family = lex = opened = 0.0

    def walk(node: Prog) -> None:
        nonlocal covers, hull, convention, set_cover, set_family, lex, opened
        if node.op == "INT" and ":" in node.spec:
            lo, hi = (int(part) for part in node.spec.split(":"))
            if view["has_num"] and lo <= view["num_min"] and hi >= view["num_max"]:
                covers = 1.0
                if lo == view["num_min"] and hi == view["num_max"]:
                    hull = 1.0
            if (lo, hi) in _CONVENTION_SET:
                convention = 1.0
            if node.hi >= 64:
                opened = 1.0
        elif node.op == "SET":
            chars = spec_chars(node.spec)
            if view["chars"] and view["chars"] <= chars:
                set_cover = 1.0
            if chars and chars in _FAMILY_CHARS.values():
                set_family = 1.0
            if node.hi >= 64:
                opened = 1.0
        elif node.op == "CLS" and node.hi >= 64:
            opened = 1.0
        elif node.op == "LEX":
            lex = 1.0
        elif node.op == "REPEAT" and node.lo <= 1 and node.hi >= 8:
            opened = 1.0
        for child in node.children:
            walk(child)

    walk(prog)
    return [
        covers,
        hull,
        convention,
        set_cover,
        set_family,
        lex,
        opened,
        view["len_span"] / 8.0,
    ]


def pack_program(prog: Prog, supports: list[str]) -> dict[str, torch.Tensor]:
    """One program as fixed arrays. The root is ``root``; children sit at higher indexes."""
    node = _binarize(prog)
    flat: list[tuple[Prog, list[int]]] = []

    def walk(current: Prog) -> int:
        child_ids = [walk(child) for child in current.children[:2]]
        flat.append((current, child_ids))
        return len(flat) - 1

    root = walk(node)
    if len(flat) > MAX_NODES:
        raise ValueError(f"program has {len(flat)} nodes, above {MAX_NODES}")
    kind = torch.zeros(MAX_NODES, dtype=torch.long)
    params = torch.zeros(MAX_NODES, PARAM_DIM)
    left = torch.full((MAX_NODES,), -1, dtype=torch.long)
    right = torch.full((MAX_NODES,), -1, dtype=torch.long)
    depth = torch.full((MAX_NODES,), -1, dtype=torch.long)
    depths: dict[int, int] = {}

    def depth_of(index: int) -> int:
        cached = depths.get(index)
        if cached is not None:
            return cached
        _node, kids = flat[index]
        value = 0 if not kids else 1 + max(depth_of(kid) for kid in kids)
        depths[index] = value
        return value

    for index, (current, kids) in enumerate(flat):
        if current.op in {"COPY", "CLS", "SET", "INT", "LEX"}:
            code, values = _leaf_params(current, supports)
        elif current.op == "OPT":
            code, values = OPTIONAL, _op_params(current)
        elif current.op == "REPEAT":
            code, values = REPEAT, _op_params(current)
        elif current.op == "SEQ":
            code, values = SEQ, _op_params(current)
        elif current.op == "ALT":
            code, values = ALT, _op_params(current)
        else:
            code, values = PAD, [0.0] * PARAM_DIM
        kind[index] = code
        params[index] = torch.tensor(values)
        depth[index] = depth_of(index)
        if kids:
            left[index] = kids[0]
        if len(kids) > 1:
            right[index] = kids[1]
    return {
        "kind": kind,
        "params": params,
        "left": left,
        "right": right,
        "depth": depth,
        "root": torch.tensor(root, dtype=torch.long),
        "relation": torch.tensor(relation_features(prog, supports), dtype=torch.float32),
    }


def pack_candidates(programs: list[Prog], supports: list[str], width: int) -> dict[str, torch.Tensor]:
    """``width`` candidate slots. Missing slots stay padding."""
    packed = [pack_program(prog, supports) for prog in programs[:width]]
    blank = {
        "kind": torch.zeros(MAX_NODES, dtype=torch.long),
        "params": torch.zeros(MAX_NODES, PARAM_DIM),
        "left": torch.full((MAX_NODES,), -1, dtype=torch.long),
        "right": torch.full((MAX_NODES,), -1, dtype=torch.long),
        "depth": torch.full((MAX_NODES,), -1, dtype=torch.long),
        "root": torch.zeros((), dtype=torch.long),
        "relation": torch.zeros(REL_DIM),
    }
    packed.extend(blank for _ in range(width - len(packed)))
    return {key: torch.stack([item[key] for item in packed]) for key in blank}
