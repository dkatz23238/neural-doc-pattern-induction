"""Local factors of a candidate, scored separately and then added.

A recursive encoder can still give ``REPEAT(width)`` and ``REPEAT(interval)``
two unrelated vectors. Here the repeat slot never sees the child, and the
field slot never sees the quantifier. The candidate score is the sum of those
pieces plus a bounded interaction. Letter identity is not a feature.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from pattern_induction.charset import charset_spec, spec_chars
from pattern_induction.program import Prog
from prior.features import _CONVENTION_SET, _FAMILY_CHARS, _literal, _round_interval, _support_view
from prior.tree import LEX_INDEX, programs_from_json, programs_json

FIELD = 0
WIDTH = 1
REPEAT = 2
OPTIONAL = 3
AFFIX = 4
SEQ = 5
ALT = 6
N_FACTORS = 7
FEAT_DIM = 16
N_AUX = 5
AUX_NAMES = ("field", "width", "repeat", "optional", "affix")


def _log(value: float) -> float:
    return math.log10(value + 1.0) / 5.0


def _walk(node: Prog):
    yield node
    for child in node.children:
        yield from _walk(child)


def _depth(node: Prog) -> int:
    if not node.children:
        return 0
    return 1 + max(_depth(child) for child in node.children)


def _content_kind(node: Prog) -> tuple:
    if node.op == "CLS":
        return ("cls", node.cls)
    if node.op == "SET":
        return ("set", node.spec)
    if node.op == "INT" and ":" in node.spec:
        return ("int", node.spec)
    if node.op == "LEX":
        return ("lex", node.spec)
    return ()


def _affix_kind(text: str) -> tuple:
    if text.isdigit():
        return ("digit", int(text))
    if text.isupper():
        return ("upper", len(text))
    if text.islower():
        return ("lower", len(text))
    if text and not text.isalnum():
        return ("sep", len(text))
    return ("other", len(text))


def _leaves(node: Prog) -> list[Prog]:
    if node.op in {"COPY", "CLS", "SET", "INT", "LEX"} or not node.children:
        return [node]
    found: list[Prog] = []
    for child in node.children:
        found.extend(_leaves(child))
    return found


def describe(prog: Prog, supports: list[str]) -> tuple[list[list[float]], list[bool], dict]:
    """Feature rows, presence, and the signatures used as auxiliary targets."""
    view = _support_view(supports)
    feat = [[0.0] * FEAT_DIM for _ in range(N_FACTORS)]
    classes: list[Prog] = []
    sets: list[Prog] = []
    ints: list[tuple[int, int, Prog]] = []
    lexes: list[str] = []
    repeats: list[tuple[int, int]] = []
    optional_roles: list[str] = []
    copies: list[tuple[str, str]] = []

    def walk(node: Prog, index: int) -> None:
        if node.op == "CLS":
            classes.append(node)
        elif node.op == "SET":
            sets.append(node)
        elif node.op == "INT" and ":" in node.spec:
            lo, hi = (int(part) for part in node.spec.split(":"))
            ints.append((lo, hi, node))
        elif node.op == "LEX":
            lexes.append(node.spec)
        elif node.op == "REPEAT":
            repeats.append((node.lo, node.hi))
        elif node.op == "OPT":
            optional_roles.append("prefix" if index == 0 else "other")
        for child_index, child in enumerate(node.children):
            walk(child, child_index)

    walk(prog, 0)
    leaves = _leaves(prog)
    content_at = [i for i, leaf in enumerate(leaves) if leaf.op in {"CLS", "SET", "INT", "LEX"}]
    first = content_at[0] if content_at else len(leaves)
    last = content_at[-1] if content_at else -1
    for index, leaf in enumerate(leaves):
        if leaf.op != "COPY":
            continue
        if not content_at:
            role = "span"
        elif index < first:
            role = "prefix"
        elif index > last:
            role = "suffix"
        else:
            role = "infix"
        copies.append((role, _literal(leaf, supports)))

    field = feat[FIELD]
    for node in classes:
        if node.cls == "digit":
            field[0] = 1.0
        elif node.cls == "upper":
            field[1] = 1.0
        elif node.cls == "lower":
            field[2] = 1.0
    if sets:
        node = max(sets, key=lambda item: len(spec_chars(item.spec)))
        chars = spec_chars(node.spec)
        field[3] = 1.0
        field[4] = float(charset_spec(chars) in _FAMILY_CHARS or node.spec in _FAMILY_CHARS)
        field[5] = float(bool(view["chars"]) and view["chars"] <= chars)
        field[6] = float(bool(view["chars"]) and chars == view["chars"])
    if ints:
        lo, hi, _node = min(ints, key=lambda item: item[1] - item[0])
        covers = bool(view["has_num"]) and lo <= view["num_min"] and hi >= view["num_max"]
        field[7] = 1.0
        field[8] = _log(float(lo))
        field[9] = _log(float(hi))
        field[10] = _log(float(max(hi - lo, 0)))
        field[11] = float((lo, hi) in _CONVENTION_SET)
        field[12] = float(_round_interval(lo, hi))
        field[13] = float(covers)
        field[14] = float(covers and lo == view["num_min"] and hi == view["num_max"])
    if lexes:
        field[15] = LEX_INDEX.get(lexes[0], 0) / 32.0

    widths = [(node.op, node.lo, node.hi) for node in (*classes, *sets)]
    widths.extend(("INT", node.lo, node.hi) for _lo, _hi, node in ints)
    if widths:
        los = [item[1] for item in widths]
        his = [item[2] for item in widths]
        width = feat[WIDTH]
        width[0] = len(widths) / 4.0
        width[1] = min(los) / 8.0
        width[2] = min(max(his), 128) / 32.0
        width[3] = float(all(lo == hi for _op, lo, hi in widths))
        width[4] = float(any(hi >= 64 for _op, _lo, hi in widths))
        width[5] = max(los) / 8.0
        width[6] = min(max(his), 128) / 32.0

    if repeats:
        los = [lo for lo, _hi in repeats]
        his = [hi for _lo, hi in repeats]
        lengths = {len(text) for text in supports}
        row = feat[REPEAT]
        row[0] = len(repeats) / 2.0
        row[1] = min(los) / 8.0
        row[2] = min(max(his), 8) / 8.0
        row[3] = float(all(lo == hi for lo, hi in repeats))
        row[4] = float(any(lo <= 1 and hi >= 8 for lo, hi in repeats))
        row[5] = view["len_span"] / 8.0
        row[6] = len(lengths) / 4.0

    if optional_roles:
        row = feat[OPTIONAL]
        row[0] = len(optional_roles) / 2.0
        row[1] = float("prefix" in optional_roles)
        row[2] = float(any(role != "prefix" for role in optional_roles))
        row[3] = float(view["len_span"] > 0)
        row[4] = optional_roles.count("prefix") / 2.0

    if copies:
        lengths = [len(text) for _role, text in copies]
        prefix_lengths = [len(text) for role, text in copies if role == "prefix"]
        row = feat[AFFIX]
        row[0] = len(copies) / 4.0
        row[1] = (sum(lengths) / len(lengths)) / 8.0
        row[2] = max(lengths) / 8.0
        row[3] = sum(bool(text) and not text.isalnum() for _role, text in copies) / len(copies)
        row[4] = sum(text.isdigit() for _role, text in copies) / len(copies)
        row[5] = sum(bool(text) and text.isupper() for _role, text in copies) / len(copies)
        row[6] = sum(bool(text) and text.islower() for _role, text in copies) / len(copies)
        row[7] = float(any(role == "prefix" for role, _text in copies))
        row[8] = float(any(role == "suffix" for role, _text in copies))
        row[9] = float(any(role == "infix" for role, _text in copies))
        if prefix_lengths:
            row[10] = (sum(prefix_lengths) / len(prefix_lengths)) / 8.0

    sequences = [node for node in _walk(prog) if node.op == "SEQ"]
    if sequences:
        row = feat[SEQ]
        row[0] = len(sequences) / 4.0
        row[1] = len(prog.children) / 4.0
        row[2] = _depth(prog) / 8.0
        row[3] = float(prog.op == "SEQ")

    alternations = [node for node in _walk(prog) if node.op == "ALT"]
    if alternations:
        feat[ALT][0] = len(alternations) / 2.0
        feat[ALT][1] = 1.0

    present = [
        bool(classes or sets or ints or lexes),
        bool(widths),
        bool(repeats),
        bool(optional_roles),
        bool(copies),
        bool(sequences),
        bool(alternations),
    ]
    signature = {
        "field": tuple(sorted(kind for node in _walk(prog) if (kind := _content_kind(node)))),
        "width": tuple(sorted(widths)),
        "repeat": tuple(sorted(repeats)),
        "optional": tuple(sorted(optional_roles)),
        "affix": tuple(sorted((role, _affix_kind(text)) for role, text in copies)),
    }
    return feat, present, signature


def _targets(signature: dict, gold: list[dict]) -> list[float]:
    if not gold:
        return [0.0] * N_AUX
    return [float(any(signature[name] == item[name] for item in gold)) for name in AUX_NAMES]


def pack_rows(
    programs: list[str],
    supports: list[list[str]],
    gold: list[list[bool]],
    width: int,
) -> dict[str, torch.Tensor]:
    """Pack a split. Absent candidate slots stay zero and carry no factors."""
    count = len(programs)
    feat = np.zeros((count, width, N_FACTORS, FEAT_DIM), np.float32)
    present = np.zeros((count, width, N_FACTORS), np.bool_)
    aux = np.zeros((count, width, N_AUX), np.float32)
    for index, (text, support, flags) in enumerate(zip(programs, supports, gold)):
        decoded = programs_from_json(text)
        described = [describe(prog, support) for prog in decoded[:width]]
        gold_sigs = [item[2] for item, flag in zip(described, flags) if flag]
        for slot, (rows, mask, signature) in enumerate(described):
            feat[index, slot] = rows
            present[index, slot] = mask
            aux[index, slot] = _targets(signature, gold_sigs)
    return {
        "feat": torch.from_numpy(feat),
        "present": torch.from_numpy(present),
        "aux": torch.from_numpy(aux),
    }


def pack_candidates(programs: list[Prog], supports: list[str], gold: list[bool], width: int) -> dict[str, torch.Tensor]:
    """One row, used by the renaming check and the unit tests."""
    return pack_rows([programs_json(programs)], [supports], [gold], width)
