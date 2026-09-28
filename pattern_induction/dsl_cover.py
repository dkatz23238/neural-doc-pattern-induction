"""Whether a gold regex is a program this executor can run.

The fragment is literals, the three full classes (digit, A–Z, a–z),
concatenation, optional, alternation, and repeat. A repeat of a compound
unit stops at 8, which is the executor cap. A class width is a single node
and may be wider than 8. A finite set of specific characters is an
alternation of literals. A partial range such as ``[4-9]`` or ``[0-5]`` is
not expanded into those literals. Trees larger than 16 nodes are out.

``exact`` means the gold language is that fragment. ``open`` means the only
gap is an unbounded quantifier, which the executor approximates with its
cap. ``no`` means some other construct is outside the fragment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from re._constants import (
    ANY,
    AT,
    AT_BEGINNING,
    AT_END,
    BRANCH,
    CATEGORY,
    CATEGORY_DIGIT,
    IN,
    LITERAL,
    MAX_REPEAT,
    MAXREPEAT,
    RANGE,
    SUBPATTERN,
)
from re._parser import parse as parse_regex

REPEAT_CAP = 8
NODE_CAP = 16


@dataclass(frozen=True)
class Cover:
    status: str
    nodes: int
    reason: str


class _Outside(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason


@dataclass(frozen=True)
class _Piece:
    nodes: int
    open: bool


def _strip_anchors(tokens: list) -> list:
    tokens = list(tokens)
    if tokens and tokens[0][0] is AT and tokens[0][1] is AT_BEGINNING:
        tokens = tokens[1:]
    if tokens and tokens[-1][0] is AT and tokens[-1][1] is AT_END:
        tokens = tokens[:-1]
    if any(op is AT for op, _arg in tokens):
        raise _Outside("anchor")
    return tokens


def _class_shape(items: list) -> tuple[str, list[str]]:
    kinds: set[str] = set()
    literals: list[str] = []
    for op, arg in items:
        if op is CATEGORY and arg is CATEGORY_DIGIT:
            kinds.add("digit")
        elif op is RANGE:
            span = tuple(arg)
            if span == (48, 57):
                kinds.add("digit")
            elif span == (65, 90):
                kinds.add("upper")
            elif span == (97, 122):
                kinds.add("lower")
            else:
                raise _Outside("partial_range")
        elif op is LITERAL:
            literals.append(chr(arg))
        elif op is CATEGORY:
            name = str(arg)
            if "SPACE" in name:
                raise _Outside("space_class")
            if "WORD" in name:
                raise _Outside("word_class")
            raise _Outside("mixed_class")
        else:
            raise _Outside("mixed_class")
    if kinds and literals:
        raise _Outside("mixed_class")
    if len(kinds) > 1:
        raise _Outside("mixed_class")
    if len(kinds) == 1:
        return "class", []
    if literals:
        return "literals", list(dict.fromkeys(literals))
    raise _Outside("empty_class")


def _literals_piece(chars: list[str]) -> _Piece:
    count = len(chars)
    if count == 1:
        return _Piece(1, False)
    return _Piece(count + (count - 1), False)


def _concat(pieces: list[_Piece]) -> _Piece:
    if not pieces:
        raise _Outside("empty")
    if len(pieces) == 1:
        return pieces[0]
    return _Piece(1 + sum(piece.nodes for piece in pieces), any(piece.open for piece in pieces))


def _is_pure_class(tokens: list) -> bool:
    tokens = _strip_anchors(tokens)
    if len(tokens) != 1 or tokens[0][0] is not IN:
        return False
    shape, _chars = _class_shape(tokens[0][1])
    return shape == "class"


def _unbounded(hi: int) -> bool:
    return hi == MAXREPEAT


def _quantify(lo: int, hi: int, body: _Piece, pure_class: bool) -> _Piece:
    if lo < 0 or (not _unbounded(hi) and hi < lo):
        raise _Outside("quantifier")
    if pure_class and lo >= 1 and not _unbounded(hi):
        return _Piece(1, False)
    if pure_class and lo >= 1 and _unbounded(hi):
        return _Piece(1, True)
    if pure_class and lo == 0 and not _unbounded(hi) and hi >= 1:
        return _Piece(2, False)
    if pure_class and lo == 0 and _unbounded(hi):
        return _Piece(2, True)
    if lo == 0 and hi == 1:
        return _Piece(1 + body.nodes, body.open)
    if lo > REPEAT_CAP or (not _unbounded(hi) and hi > REPEAT_CAP):
        raise _Outside("repeat_cap")
    opened = body.open or _unbounded(hi)
    if lo == 0:
        return _Piece(2 + body.nodes, opened)
    return _Piece(1 + body.nodes, opened)


def _compile(tokens: list) -> _Piece:
    tokens = _strip_anchors(tokens)
    pieces: list[_Piece] = []
    index = 0
    while index < len(tokens):
        op, arg = tokens[index]
        if op is LITERAL:
            while index < len(tokens) and tokens[index][0] is LITERAL:
                index += 1
            pieces.append(_Piece(1, False))
            continue
        if op is IN:
            shape, chars = _class_shape(arg)
            pieces.append(_Piece(1, False) if shape == "class" else _literals_piece(chars))
            index += 1
            continue
        if op is MAX_REPEAT:
            lo, hi, sub = arg
            pieces.append(_quantify(lo, hi, _compile(sub), _is_pure_class(sub)))
            index += 1
            continue
        if op is BRANCH:
            _flags, arms = arg
            compiled = [_compile(arm) for arm in arms]
            if len(compiled) == 1:
                pieces.append(compiled[0])
            else:
                nodes = sum(piece.nodes for piece in compiled) + (len(compiled) - 1)
                pieces.append(_Piece(nodes, any(piece.open for piece in compiled)))
            index += 1
            continue
        if op is SUBPATTERN:
            pieces.append(_compile(arg[-1]))
            index += 1
            continue
        if op is ANY:
            raise _Outside("dot")
        raise _Outside("other")
    return _concat(pieces)


def cover_regex(pattern: str) -> Cover:
    """Classify one anchored gold regex."""
    try:
        piece = _compile(parse_regex(pattern))
    except _Outside as exc:
        return Cover("no", 0, exc.reason)
    except re.error:
        return Cover("no", 0, "parse")
    if piece.nodes > NODE_CAP:
        return Cover("no", piece.nodes, "nodes")
    if piece.open:
        return Cover("open", piece.nodes, "unbounded")
    return Cover("exact", piece.nodes, "")

