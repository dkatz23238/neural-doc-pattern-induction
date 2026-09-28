"""Whether a gold regex is a program this executor can run.

The fragment is literals, character sets, fixed-width and canonical
integer ranges, concatenation, optional, alternation, and repeat. A full
digit, upper, or lower class stays a class. A partial range or a mixture
of those alphabets is one character set. A contiguous integer interval is
one integer range, compiled to regex by the executor rather than induced
as an alternation. A repeat of a compound unit stops at 8. A class or a
character set is one node and may be wider than 8. Word and space classes
stay outside. Trees larger than 16 nodes are out.

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
    partial = False
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
                partial = True
        elif op is LITERAL:
            literals.append(chr(arg))
        elif op is CATEGORY:
            name = str(arg)
            if "SPACE" in name:
                raise _Outside("space_class")
            if "WORD" in name:
                raise _Outside("word_class")
            raise _Outside("other_class")
        else:
            raise _Outside("other_class")
    if not kinds and not literals and not partial:
        raise _Outside("empty_class")
    if not partial and not literals and len(kinds) == 1:
        return "class", []
    return "set", literals


def _concat(pieces: list[_Piece]) -> _Piece:
    if not pieces:
        raise _Outside("empty")
    if len(pieces) == 1:
        return pieces[0]
    return _Piece(1 + sum(piece.nodes for piece in pieces), any(piece.open for piece in pieces))


def _is_atomic_charset(tokens: list) -> bool:
    """One character drawn from a full class or from a character set."""
    tokens = _strip_anchors(tokens)
    if len(tokens) != 1 or tokens[0][0] is not IN:
        return False
    try:
        _class_shape(tokens[0][1])
    except _Outside:
        return False
    return True


def _unbounded(hi: int) -> bool:
    return hi == MAXREPEAT


def _quantify(lo: int, hi: int, body: _Piece, atomic: bool) -> _Piece:
    if lo < 0 or (not _unbounded(hi) and hi < lo):
        raise _Outside("quantifier")
    if atomic and lo >= 1 and not _unbounded(hi):
        return _Piece(1, False)
    if atomic and lo >= 1 and _unbounded(hi):
        return _Piece(1, True)
    if atomic and lo == 0 and not _unbounded(hi) and hi >= 1:
        return _Piece(2, False)
    if atomic and lo == 0 and _unbounded(hi):
        return _Piece(2, True)
    if lo == 0 and hi == 1:
        return _Piece(1 + body.nodes, body.open)
    if lo > REPEAT_CAP or (not _unbounded(hi) and hi > REPEAT_CAP):
        raise _Outside("repeat_cap")
    opened = body.open or _unbounded(hi)
    if lo == 0:
        return _Piece(2 + body.nodes, opened)
    return _Piece(1 + body.nodes, opened)


_INT_CAP = 800


def _class_chars(items: list) -> set[str] | None:
    chars: set[str] = set()
    for op, arg in items:
        if op is CATEGORY and arg is CATEGORY_DIGIT:
            chars.update(chr(code) for code in range(48, 58))
        elif op is RANGE:
            start, end = arg
            if end < start or end - start > 120:
                return None
            chars.update(chr(code) for code in range(start, end + 1))
        elif op is LITERAL:
            chars.add(chr(arg))
        else:
            return None
    return chars


def _finite_token(token: tuple, cap: int) -> set[str] | None:
    op, arg = token
    if op is LITERAL:
        return {chr(arg)}
    if op is IN:
        chars = _class_chars(arg)
        return chars if chars is not None else None
    if op is MAX_REPEAT:
        lo, hi, sub = arg
        if _unbounded(hi) or hi > 4 or lo > hi:
            return None
        body = _finite(sub, cap)
        if body is None:
            return None
        current = {""}
        found: set[str] = {""} if lo == 0 else set()
        for count in range(1, hi + 1):
            nxt: set[str] = set()
            for left in current:
                for piece in body:
                    nxt.add(left + piece)
                    if len(nxt) > cap:
                        return None
            current = nxt
            if count >= lo:
                found |= current
                if len(found) > cap:
                    return None
        return found
    if op is BRANCH:
        _flags, arms = arg
        found = set()
        for arm in arms:
            part = _finite(arm, cap)
            if part is None:
                return None
            found |= part
            if len(found) > cap:
                return None
        return found
    if op is SUBPATTERN:
        return _finite(arg[-1], cap)
    return None


def _finite(tokens: list, cap: int = _INT_CAP) -> set[str] | None:
    acc = {""}
    for token in tokens:
        step = _finite_token(token, cap)
        if step is None:
            return None
        nxt: set[str] = set()
        for left in acc:
            for right in step:
                nxt.add(left + right)
                if len(nxt) > cap:
                    return None
        acc = nxt
    return acc


def _proper_integer(language: set[str] | None) -> bool:
    """A contiguous digit interval that is not a full character class."""
    if not language or any(not text.isdigit() for text in language):
        return False
    widths = {len(text) for text in language}
    values = {int(text) for text in language}
    lo, hi = min(values), max(values)
    if len(values) != hi - lo + 1:
        return False
    if len(widths) == 1:
        width = next(iter(widths))
        if width < 2:
            return False
        if lo == 0 and hi == 10**width - 1:
            return False
        return len(language) == hi - lo + 1
    if any(len(text) > 1 and text.startswith("0") for text in language):
        return False
    return language == {str(value) for value in range(lo, hi + 1)}


def _integer_prefix(tokens: list, index: int) -> int:
    best = 0
    for length in range(1, len(tokens) - index + 1):
        language = _finite(tokens[index : index + length])
        if language is None:
            break
        if _proper_integer(language):
            best = length
    return best


def _compile(tokens: list) -> _Piece:
    tokens = _strip_anchors(tokens)
    pieces: list[_Piece] = []
    index = 0
    while index < len(tokens):
        taken = _integer_prefix(tokens, index)
        if taken:
            pieces.append(_Piece(1, False))
            index += taken
            continue
        op, arg = tokens[index]
        if op is LITERAL:
            while index < len(tokens) and tokens[index][0] is LITERAL:
                index += 1
            pieces.append(_Piece(1, False))
            continue
        if op is IN:
            _class_shape(arg)
            pieces.append(_Piece(1, False))
            index += 1
            continue
        if op is MAX_REPEAT:
            lo, hi, sub = arg
            pieces.append(_quantify(lo, hi, _compile(sub), _is_atomic_charset(sub)))
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

