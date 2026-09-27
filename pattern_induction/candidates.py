"""Symbolic regex neighborhood used by the candidate-oracle test.

The heuristic proposes one pattern. This module proposes a handful of nearby
patterns by mutating classes, widths, literals, and optionality, and by
re-reading the visible strings as delimiter-separated slots. Every candidate
except a broken heuristic has to match the visible strings. Hidden examples
are never consulted here.
"""

from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass

from pattern_induction.evaluate import compile_regex, is_catastrophic
from pattern_induction.heuristic import _char_class, escape_literal, infer_regex

DIGITS = set(string.digits)
LOWER = set(string.ascii_lowercase)
UPPER = set(string.ascii_uppercase)
HEX_LOWER = set("0123456789abcdef")
HEX_UPPER = set("0123456789ABCDEF")
MONTHS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)
EXTENSIONS = (
    "pdf", "docx", "xlsx", "csv", "txt", "png",
    "jpg", "jpeg", "json", "xml", "html", "zip",
)
CATALOGS = (("month", MONTHS), ("ampm", ("AM", "PM")), ("extension", EXTENSIONS))

# Classes rewritten wholesale by the hex-collapse candidate.
HEX_CONVERT = {
    r"\d",
    r"[0-9]",
    r"[A-Za-z0-9]",
    r"[A-Za-z0-9_]",
    r"\w",
    r"[a-z0-9]",
    r"[A-Z0-9]",
    r"[0-9a-f]",
    r"[0-9A-F]",
    r"[0-9A-Fa-f]",
}

CLASS_ALTS = {
    r"\d": (r"[0-9]", r"[0-9a-f]", r"\w"),
    r"[0-9]": (r"\d", r"[0-9a-f]"),
    r"[A-Za-z0-9]": (
        r"[0-9a-f]",
        r"[0-9A-F]",
        r"[0-9A-Fa-f]",
        r"[a-z0-9]",
        r"[A-Z0-9]",
        r"\w",
        r"[A-Za-z]",
        r"\d",
    ),
    r"[a-z]": (
        r"[a-z0-9]",
        r"[a-z0-9_]",
        r"[a-z0-9._%+-]",
        r"[a-z0-9.-]",
        r"[A-Za-z]",
        r"[0-9a-f]",
    ),
    r"[A-Z]": (r"[A-Za-z]", r"[A-Z0-9]", r"[0-9A-F]"),
    r"[A-Za-z]": (r"[A-Za-z0-9]", r"[a-z]", r"[A-Z]"),
    r"\w": (r"[A-Za-z0-9_]", r"[A-Za-z0-9]", r"[0-9a-f]", r"\d"),
    r".": (r"\d", r"\w", r"\S", r"[A-Za-z0-9]"),
    r"[0-9a-f]": (r"[0-9A-F]", r"[A-Za-z0-9]", r"\d"),
    r"[0-9A-F]": (r"[0-9a-f]", r"[A-Za-z0-9]", r"\d"),
    r"[a-z0-9]": (r"[0-9a-f]", r"[a-z0-9_]", r"[A-Za-z0-9]"),
    r"[a-z0-9_]": (r"[a-z0-9]", r"\w", r"[a-z0-9.-]"),
}

TAG_PRIORITY = (
    "heuristic",
    "hex_collapse",
    "slot_digit_general",
    "slot_tight",
    "literal_digit",
    "width",
    "width_aggressive",
    "class_tighten",
    "class_widen",
    "affix",
    "catalog",
    "repeat",
    "optional",
    "slot_widen",
    "delimiter",
)

FEATURE_NAMES = (
    "data_bits",
    "desc_bits",
    "is_heuristic",
    "has_hex",
    "has_broad_alnum",
    "has_dot",
    "n_unbounded",
    "n_ranges",
    "range_slack",
    "n_optional",
    "n_alternations",
    "n_short_digit_lits",
    "pattern_len",
    "n_nodes",
)


@dataclass(frozen=True)
class Node:
    op: str
    text: str = ""
    lo: int = 1
    hi: int | None = 1
    kids: tuple["Node", ...] = ()

    def replace(self, **kwargs: object) -> "Node":
        data = {
            "op": self.op,
            "text": self.text,
            "lo": self.lo,
            "hi": self.hi,
            "kids": self.kids,
        }
        data.update(kwargs)
        return Node(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Candidate:
    pattern: str
    tag: str


class ParseError(ValueError):
    pass


class _Parser:
    def __init__(self, source: str) -> None:
        self.source = source
        self.i = 0
        self.n = len(source)

    def peek(self) -> str:
        if self.i >= self.n:
            return ""
        return self.source[self.i]

    def parse(self) -> Node:
        node = self._parse_alt()
        if self.i != self.n:
            raise ParseError(f"unparsed tail {self.source[self.i:]!r}")
        return compress(node)

    def _parse_alt(self) -> Node:
        nodes = [self._parse_seq()]
        while self.peek() == "|":
            self.i += 1
            nodes.append(self._parse_seq())
        if len(nodes) == 1:
            return nodes[0]
        return Node("ALT", kids=tuple(nodes))

    def _parse_seq(self) -> Node:
        kids: list[Node] = []
        while self.i < self.n and self.peek() not in ")|":
            kids.append(self._parse_piece())
        if not kids:
            return Node("EMPTY")
        if len(kids) == 1:
            return kids[0]
        return Node("SEQ", kids=tuple(kids))

    def _parse_piece(self) -> Node:
        return self._apply_quant(self._parse_atom())

    def _parse_atom(self) -> Node:
        ch = self.peek()
        if ch == "":
            raise ParseError("unexpected end")
        if ch == "(":
            return self._parse_group()
        if ch == "[":
            return self._parse_class()
        if ch == "\\":
            self.i += 1
            if self.i >= self.n:
                raise ParseError("trailing escape")
            esc = self.source[self.i]
            self.i += 1
            if esc in "dDwWsS":
                return Node("CLS", text="\\" + esc)
            return Node("LIT", text=esc)
        if ch == ".":
            self.i += 1
            return Node("DOT")
        if ch in "*+?{)":
            raise ParseError(f"bare quantifier {ch}")
        self.i += 1
        return Node("LIT", text=ch)

    def _parse_group(self) -> Node:
        self.i += 1
        if self.source.startswith("?:", self.i):
            self.i += 2
        elif self.peek() == "?":
            raise ParseError("unsupported group")
        inner = self._parse_alt()
        if self.peek() != ")":
            raise ParseError("unclosed group")
        self.i += 1
        return inner

    def _parse_class(self) -> Node:
        start = self.i
        self.i += 1
        if self.peek() == "^":
            self.i += 1
        while self.i < self.n:
            if self.peek() == "\\":
                self.i += 2
                continue
            if self.peek() == "]":
                self.i += 1
                return Node("CLS", text=self.source[start:self.i])
            self.i += 1
        raise ParseError("unclosed class")

    def _apply_quant(self, node: Node) -> Node:
        ch = self.peek()
        if ch == "?":
            self.i += 1
            return _wrap_quant(node, 0, 1)
        if ch == "+":
            self.i += 1
            return _wrap_quant(node, 1, None)
        if ch == "*":
            self.i += 1
            return _wrap_quant(node, 0, None)
        if ch != "{":
            return node
        self.i += 1
        lo_s, hi_s, unbounded = self._parse_brace()
        lo = int(lo_s)
        hi = None if unbounded else int(hi_s if hi_s is not None else lo_s)
        return _wrap_quant(node, lo, hi)

    def _parse_brace(self) -> tuple[str, str | None, bool]:
        start = self.i
        while self.peek().isdigit():
            self.i += 1
        lo_s = self.source[start:self.i]
        if not lo_s:
            raise ParseError("empty quantifier")
        if self.peek() == "}":
            self.i += 1
            return lo_s, None, False
        if self.peek() != ",":
            raise ParseError("bad quantifier")
        self.i += 1
        if self.peek() == "}":
            self.i += 1
            return lo_s, None, True
        start = self.i
        while self.peek().isdigit():
            self.i += 1
        hi_s = self.source[start:self.i]
        if self.peek() != "}" or not hi_s:
            raise ParseError("bad range")
        self.i += 1
        return lo_s, hi_s, False


def _wrap_quant(node: Node, lo: int, hi: int | None) -> Node:
    if lo == 1 and hi == 1:
        return node
    if node.op in {"LIT", "CLS", "DOT", "EMPTY"} and node.lo == 1 and node.hi == 1:
        return node.replace(lo=lo, hi=hi)
    return Node("REPEAT", lo=lo, hi=hi, kids=(node,))


def _lit_kind(text: str) -> str:
    if text.isdigit():
        return "d"
    if text.isalpha():
        return "a"
    return "p"


def merge_kids(kids: list[Node]) -> list[Node]:
    merged: list[Node] = []
    for kid in kids:
        if (
            merged
            and kid.op == "LIT"
            and merged[-1].op == "LIT"
            and merged[-1].lo == 1
            and merged[-1].hi == 1
            and kid.lo == 1
            and kid.hi == 1
            and _lit_kind(merged[-1].text) == _lit_kind(kid.text)
        ):
            merged[-1] = merged[-1].replace(text=merged[-1].text + kid.text)
            continue
        if (
            merged
            and kid.op == "CLS"
            and merged[-1].op == "CLS"
            and kid.text == merged[-1].text
            and merged[-1].hi is not None
            and kid.hi is not None
        ):
            previous = merged[-1]
            merged[-1] = previous.replace(lo=previous.lo + kid.lo, hi=previous.hi + kid.hi)
            continue
        merged.append(kid)
    return merged


def compress(node: Node) -> Node:
    if node.op == "REPEAT":
        return node.replace(kids=(compress(node.kids[0]),))
    if node.op in {"SEQ", "ALT"}:
        kids = merge_kids([compress(kid) for kid in node.kids]) if node.op == "SEQ" else [
            compress(kid) for kid in node.kids
        ]
        node = node.replace(kids=tuple(kids))
        if node.op == "SEQ" and len(node.kids) == 1 and node.lo == 1 and node.hi == 1:
            return node.kids[0]
        return node
    return node


def parse_pattern(pattern: str) -> Node | None:
    body = pattern.strip()
    if body.startswith("^"):
        body = body[1:]
    if body.endswith("$") and not body.endswith(r"\$"):
        body = body[:-1]
    if body == "":
        return Node("EMPTY")
    try:
        return _Parser(body).parse()
    except ParseError:
        return None


def _simple_atom(body: str) -> bool:
    if body in {".", r"\d", r"\D", r"\w", r"\W", r"\s", r"\S"}:
        return True
    if len(body) == 1:
        return True
    if len(body) == 2 and body.startswith("\\"):
        return True
    return body.startswith("[") and body.endswith("]") and body.count("[") == 1


def _quantify(body: str, lo: int, hi: int | None) -> str:
    if lo == 1 and hi == 1:
        return body
    if not _simple_atom(body):
        body = f"(?:{body})"
    if lo == 0 and hi == 1:
        return f"{body}?"
    if lo == 1 and hi is None:
        return f"{body}+"
    if lo == 0 and hi is None:
        return f"{body}*"
    if hi is None:
        return f"{body}{{{lo},}}"
    if lo == hi:
        return body if lo == 1 else f"{body}{{{lo}}}"
    return f"{body}{{{lo},{hi}}}"


def render_node(node: Node) -> str:
    if node.op == "EMPTY":
        return ""
    if node.op == "LIT":
        return _quantify(escape_literal(node.text), node.lo, node.hi)
    if node.op == "CLS":
        return _quantify(node.text, node.lo, node.hi)
    if node.op == "DOT":
        return _quantify(".", node.lo, node.hi)
    if node.op == "SEQ":
        return "".join(render_node(kid) for kid in node.kids)
    if node.op == "ALT":
        return "(?:" + "|".join(render_node(kid) for kid in node.kids) + ")"
    if node.op == "REPEAT":
        return _quantify(render_node(node.kids[0]), node.lo, node.hi)
    raise ValueError(node.op)


def anchor(node: Node) -> str:
    return "^" + render_node(node) + "$"


def class_chars(text: str) -> set[str] | None:
    if text in {r"\d", r"[0-9]"}:
        return set(DIGITS)
    if text == r"\w":
        return set(string.ascii_letters + string.digits + "_")
    if text == r"\s":
        return set(" \t\r\n\f\v")
    if not (text.startswith("[") and text.endswith("]")):
        return None
    body = text[1:-1]
    if body.startswith("^"):
        return None
    chars: set[str] = set()
    i = 0
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body):
            chars.add(body[i + 1])
            i += 2
            continue
        if i + 2 < len(body) and body[i + 1] == "-" and body[i] != "\\":
            start, end = ord(body[i]), ord(body[i + 2])
            if end < start:
                return None
            chars.update(chr(code) for code in range(start, end + 1))
            i += 3
            continue
        chars.add(body[i])
        i += 1
    return chars


def class_size(text: str) -> int:
    if text == r"\D" or text == r"\W" or text == r"\S":
        return 90
    chars = class_chars(text)
    if chars is None:
        return 32
    return max(len(chars), 1)


def _lit_data_bits(text: str, digit_prior: bool) -> float:
    if not digit_prior or not text.isdigit() or len(text) >= 4:
        return 0.0
    return len(text) * math.log2(10) + 1.0


def _quant_bits(lo: int, hi: int | None, unit_bits: float) -> float:
    if hi is None:
        return 16.0 + lo * unit_bits
    length_bits = 0.0 if lo == hi else math.log2(hi - lo + 1)
    midpoint = (lo + hi) / 2
    return length_bits + midpoint * unit_bits


def data_bits(node: Node, digit_prior: bool) -> float:
    if node.op == "EMPTY":
        return 0.0
    if node.op == "LIT":
        return _quant_bits(node.lo, node.hi, _lit_data_bits(node.text, digit_prior))
    if node.op == "CLS":
        return _quant_bits(node.lo, node.hi, math.log2(class_size(node.text)))
    if node.op == "DOT":
        return _quant_bits(node.lo, node.hi, math.log2(95))
    if node.op == "SEQ":
        return sum(data_bits(kid, digit_prior) for kid in node.kids)
    if node.op == "ALT":
        if not node.kids:
            return 8.0
        mean = sum(data_bits(kid, digit_prior) for kid in node.kids) / len(node.kids)
        return math.log2(len(node.kids)) + mean
    if node.op == "REPEAT":
        return _quant_bits(node.lo, node.hi, data_bits(node.kids[0], digit_prior))
    return 32.0


def description_nodes(node: Node) -> int:
    here = 0 if node.op in {"SEQ", "EMPTY"} else 1
    return here + sum(description_nodes(kid) for kid in node.kids)


def _char_in_class(text: str, ch: str) -> bool:
    if text == r"\d":
        return ch.isdigit()
    if text == r"\D":
        return not ch.isdigit()
    if text == r"\w":
        return ch.isalnum() or ch == "_"
    if text == r"\W":
        return not (ch.isalnum() or ch == "_")
    if text == r"\s":
        return ch.isspace()
    if text == r"\S":
        return not ch.isspace()
    chars = class_chars(text)
    return chars is not None and ch in chars


def _length_bits(lo: int, hi: int | None, count: int) -> float:
    if hi is None:
        return math.log2(count + 1) + 1.0
    if lo == hi:
        return 0.0
    return math.log2(hi - lo + 1)


def _ends(
    node: Node,
    text: str,
    pos: int,
    digit_prior: bool,
    cache: dict,
) -> dict[int, float]:
    """Min bits to match a prefix of text[pos:] with node, keyed by end index.

    Length choices are charged for the length that was actually used, so a
    loose range is not cheaper than an exact width just because its midpoint
    is short.
    """
    key = (id(node), pos, digit_prior)
    cached = cache.get(key)
    if cached is not None:
        return cached
    ends = _ends_uncached(node, text, pos, digit_prior, cache)
    cache[key] = ends
    return ends


def _ends_uncached(
    node: Node,
    text: str,
    pos: int,
    digit_prior: bool,
    cache: dict,
) -> dict[int, float]:
    if node.op == "EMPTY":
        return {pos: 0.0}
    if node.op == "SEQ":
        states = {pos: 0.0}
        for kid in node.kids:
            nxt: dict[int, float] = {}
            for start, base in states.items():
                for end, extra in _ends(kid, text, start, digit_prior, cache).items():
                    cost = base + extra
                    if end not in nxt or cost < nxt[end]:
                        nxt[end] = cost
            states = nxt
            if not states:
                break
        return states
    if node.op == "ALT":
        if not node.kids:
            return {}
        choice = math.log2(len(node.kids))
        out: dict[int, float] = {}
        for kid in node.kids:
            for end, extra in _ends(kid, text, pos, digit_prior, cache).items():
                cost = choice + extra
                if end not in out or cost < out[end]:
                    out[end] = cost
        return out
    if node.op == "REPEAT":
        return _repeat_ends(node, text, pos, digit_prior, cache)
    return _quantified_atom_ends(node, text, pos, digit_prior)


def _quantified_atom_ends(node: Node, text: str, pos: int, digit_prior: bool) -> dict[int, float]:
    if node.op == "LIT":
        unit = node.text
        unit_len = len(unit)
        unit_bits = _lit_data_bits(unit, digit_prior)

        def ok(count: int) -> bool:
            chunk = unit * count
            return text.startswith(chunk, pos)

    elif node.op == "CLS":
        unit_len = 1
        unit_bits = math.log2(class_size(node.text))

        def ok(count: int) -> bool:
            stop = pos + count
            return stop <= len(text) and all(_char_in_class(node.text, ch) for ch in text[pos:stop])

    elif node.op == "DOT":
        unit_len = 1
        unit_bits = math.log2(95)

        def ok(count: int) -> bool:
            stop = pos + count
            return stop <= len(text) and all(ch != "\n" for ch in text[pos:stop])

    else:
        return {}
    if unit_len == 0:
        return {pos: 0.0}
    room = (len(text) - pos) // unit_len
    upper = room if node.hi is None else min(node.hi, room)
    out: dict[int, float] = {}
    for count in range(node.lo, upper + 1):
        if not ok(count):
            break
        out[pos + count * unit_len] = _length_bits(node.lo, node.hi, count) + count * unit_bits
    return out


def _repeat_ends(node: Node, text: str, pos: int, digit_prior: bool, cache: dict) -> dict[int, float]:
    inner = node.kids[0]
    upper = len(text) - pos + 1 if node.hi is None else node.hi
    frontier = {pos: 0.0}
    reached = {0: {pos: 0.0}}
    for count in range(1, upper + 1):
        nxt: dict[int, float] = {}
        for start, base in frontier.items():
            for end, extra in _ends(inner, text, start, digit_prior, cache).items():
                if end == start:
                    continue
                cost = base + extra
                if end not in nxt or cost < nxt[end]:
                    nxt[end] = cost
        if not nxt:
            break
        reached[count] = nxt
        frontier = nxt
    out: dict[int, float] = {}
    for count, ends in reached.items():
        if count < node.lo or (node.hi is not None and count > node.hi):
            continue
        length_bits = _length_bits(node.lo, node.hi, count)
        for end, base in ends.items():
            cost = base + length_bits
            if end not in out or cost < out[end]:
                out[end] = cost
    return out


def observed_data_bits(pattern: str, strings: list[str], digit_prior: bool) -> float | None:
    """Bits to encode these exact strings. None if the pattern misses one."""
    node = parse_pattern(pattern)
    if node is None:
        return None
    total = 0.0
    for text in strings:
        ends = _ends(node, text, 0, digit_prior, {})
        bits = ends.get(len(text))
        if bits is None:
            return None
        total += bits
    return total


def pattern_cost(pattern: str, strings: list[str] | int, digit_prior: bool) -> float:
    """Two-part code length. Lower is a better visible-only explanation.

    `digit_prior` charges short digit literals as if they were unstable, which
    is the hand-written rule's only departure from pure compression. An int
    `strings` value keeps the old length-free approximation for callers that
    have no examples.
    """
    node = parse_pattern(pattern)
    if node is None:
        return 1_000_000.0 + len(pattern)
    desc = 1.5 * description_nodes(node)
    if isinstance(strings, int):
        return desc + strings * data_bits(node, digit_prior)
    data = observed_data_bits(pattern, strings, digit_prior)
    if data is None:
        return 1_000_000.0 + len(pattern)
    return desc + data


def _walk_stats(node: Node, stats: dict) -> None:
    if node.op == "CLS":
        if "0-9a-f" in node.text or "0-9A-F" in node.text:
            stats["has_hex"] = 1.0
        if node.text in {r"[A-Za-z0-9]", r"[A-Za-z0-9_]", r"\w"}:
            stats["has_broad_alnum"] = 1.0
    if node.op == "DOT":
        stats["has_dot"] = 1.0
    if node.op == "LIT" and node.text.isdigit() and len(node.text) <= 2:
        stats["n_short_digit_lits"] += 1.0
    if node.op == "ALT":
        stats["n_alternations"] += 1.0
    if node.op in {"LIT", "CLS", "DOT", "REPEAT"}:
        if node.hi is None:
            stats["n_unbounded"] += 1.0
        elif node.lo != node.hi:
            stats["n_ranges"] += 1.0
            stats["range_slack"] += float(node.hi - node.lo)
        if node.lo == 0:
            stats["n_optional"] += 1.0
    for kid in node.kids:
        _walk_stats(kid, stats)


def feature_vector(pattern: str, heuristic: str, strings: list[str]) -> list[float]:
    """Visible-only features. Hidden labels never enter this vector."""
    node = parse_pattern(pattern)
    stats = {
        "has_hex": 0.0,
        "has_broad_alnum": 0.0,
        "has_dot": 0.0,
        "n_unbounded": 0.0,
        "n_ranges": 0.0,
        "range_slack": 0.0,
        "n_optional": 0.0,
        "n_alternations": 0.0,
        "n_short_digit_lits": 0.0,
    }
    if node is not None:
        _walk_stats(node, stats)
        data = observed_data_bits(pattern, strings, False)
        if data is None:
            data = 1_000_000.0
        desc = 1.5 * description_nodes(node)
        nodes = float(description_nodes(node))
    else:
        data = 1_000_000.0
        desc = float(len(pattern))
        nodes = 0.0
    values = {
        "data_bits": data,
        "desc_bits": desc,
        "is_heuristic": 1.0 if pattern == heuristic else 0.0,
        "pattern_len": float(len(pattern)),
        "n_nodes": nodes,
        **stats,
    }
    return [values[name] for name in FEATURE_NAMES]


def _single_site(root: Node, mutate) -> list[Node]:
    found: list[Node] = []

    def walk(node: Node) -> list[Node]:
        local: list[Node] = [compress(item) for item in mutate(node)]
        kids = list(node.kids)
        for index, kid in enumerate(kids):
            for new_kid in walk(kid):
                updated = kids.copy()
                updated[index] = new_kid
                local.append(compress(node.replace(kids=tuple(updated))))
        return local

    found.extend(walk(root))
    return found


def _quant_options(lo: int, hi: int | None, aggressive: bool) -> list[tuple[int, int | None]]:
    if hi is None:
        return []
    options: list[tuple[int, int | None]] = []
    if lo == hi:
        n = lo
        if aggressive:
            options.append((max(1, n), None))
            options.append((1, max(n + 4, n * 2)))
        else:
            options.append((max(1, n - 1), n + 1))
            options.append((max(1, n - 2), n + 2))
            if n > 1:
                options.append((n, n))  # dropped below
            options.append((lo, lo))
    elif aggressive:
        options.append((lo, None))
        options.append((max(1, lo), max(hi + 8, hi * 2)))
    else:
        options.append((lo, hi + 2))
        options.append((max(0, lo - 1), hi))
        options.append((lo, lo))
        options.append((hi, hi))
    unique: list[tuple[int, int | None]] = []
    seen = {(lo, hi)}
    for item in options:
        nlo, nhi = item
        if nhi is not None and nlo > nhi:
            continue
        if nlo < 0 or item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _width_mutator(aggressive: bool):
    def mutate(node: Node) -> list[Node]:
        if node.op not in {"LIT", "CLS", "DOT", "REPEAT"}:
            return []
        return [node.replace(lo=lo, hi=hi) for lo, hi in _quant_options(node.lo, node.hi, aggressive)]

    return mutate


def _optional_mutator(node: Node) -> list[Node]:
    if node.op not in {"LIT", "CLS", "DOT", "REPEAT"}:
        return []
    if node.lo == 0:
        if node.hi == 1:
            return [node.replace(lo=1, hi=1)]
        return [node.replace(lo=1)]
    hi = node.hi if node.hi is not None else node.lo
    return [node.replace(lo=0, hi=hi)]


def _class_tag(old: str, new: str) -> str:
    if "0-9a-f" in new or "0-9A-F" in new:
        return "hex_collapse"
    if class_size(new) > class_size(old) + 0.5:
        return "class_widen"
    return "class_tighten"


def _class_mutator(node: Node) -> list[tuple[str, Node]]:
    if node.op == "DOT":
        alts = CLASS_ALTS.get(".", ())
        return [(_class_tag(".", alt), Node("CLS", text=alt, lo=node.lo, hi=node.hi)) for alt in alts]
    if node.op != "CLS":
        return []
    return [
        (_class_tag(node.text, alt), node.replace(text=alt))
        for alt in CLASS_ALTS.get(node.text, ())
        if alt != node.text
    ]


def _literal_digit_mutator(node: Node) -> list[Node]:
    if node.op != "LIT" or not node.text.isdigit() or node.lo != 1 or node.hi != 1:
        return []
    width = len(node.text)
    return [Node("CLS", text=r"\d", lo=width, hi=width)]


def _map_nodes(node: Node, fn) -> Node:
    kids = tuple(_map_nodes(kid, fn) for kid in node.kids)
    updated = node.replace(kids=kids) if kids else node
    return fn(updated)


def _hex_collapse(root: Node, alphabet: str) -> Node:
    def convert(node: Node) -> Node:
        if node.op == "CLS" and node.text in HEX_CONVERT:
            return node.replace(text=alphabet)
        if (
            node.op == "LIT"
            and node.text.isdigit()
            and len(node.text) <= 2
            and node.lo == 1
            and node.hi == 1
        ):
            return Node("CLS", text=alphabet, lo=len(node.text), hi=len(node.text))
        return node

    return compress(_map_nodes(root, convert))


def _render_added(root: Node, mutate, tag: str, sink: dict[str, str], limit: int) -> None:
    for variant in _single_site(root, mutate):
        if len(sink) >= limit:
            return
        _add_pattern(sink, anchor(variant), tag, limit)


def _family_tag(pattern: str, tag: str) -> str:
    """Hex alphabets are their own failure family even when a slot rebuild found them."""
    if "[0-9a-f]" in pattern or "[0-9A-F]" in pattern or "[0-9A-Fa-f]" in pattern:
        return "hex_collapse"
    return tag


def _add_pattern(sink: dict[str, str], pattern: str, tag: str, limit: int) -> None:
    if not pattern or pattern in sink or len(sink) >= limit or len(pattern) > 300:
        return
    sink[pattern] = tag


def named_class(chars: set[str]) -> str:
    if not chars:
        return "."
    if len(chars) == 1:
        return escape_literal(next(iter(chars)))
    word = set(string.ascii_letters + string.digits + "_")
    if chars <= DIGITS:
        return r"\d"
    if chars <= HEX_LOWER:
        return "[0-9a-f]"
    if chars <= HEX_UPPER:
        return "[0-9A-F]"
    if chars <= HEX_LOWER | HEX_UPPER:
        return "[0-9A-Fa-f]"
    if chars <= LOWER:
        return "[a-z]"
    if chars <= UPPER:
        return "[A-Z]"
    if chars <= LOWER | UPPER:
        return "[A-Za-z]"
    if chars <= LOWER | DIGITS | {"_"}:
        return "[a-z0-9_]"
    if chars <= UPPER | DIGITS:
        return "[A-Z0-9]"
    if chars <= LOWER | UPPER | DIGITS:
        return "[A-Za-z0-9]"
    if chars <= word:
        return r"\w"
    if chars <= set(string.ascii_letters + string.digits + "+/=_"):
        return "[A-Za-z0-9+/=]"
    if all(ch.isspace() for ch in chars):
        return r"\s"
    if len(chars) <= 8:
        return _char_class(chars)
    return "."


def _split_parts(text: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    i = 0
    while i < len(text):
        if text[i].isalnum():
            j = i + 1
            while j < len(text) and text[j].isalnum():
                j += 1
            parts.append(("slot", text[i:j]))
        else:
            j = i + 1
            while j < len(text) and not text[j].isalnum():
                j += 1
            parts.append(("delim", text[i:j]))
        i = j
    return parts


def _aligned_parts(strings: list[str]) -> list[list[tuple[str, str]]] | None:
    parts = [_split_parts(text) for text in strings]
    width = len(parts[0])
    if width == 0 or any(len(row) != width for row in parts):
        return None
    for index in range(width):
        kind = parts[0][index][0]
        if any(row[index][0] != kind for row in parts):
            return None
    return parts


def _piece_for_values(values: list[str], generalize_short_digits: bool) -> str:
    unique = set(values)
    lengths = [len(value) for value in values]
    lo, hi = min(lengths), max(lengths)
    if len(unique) == 1:
        token = values[0]
        if generalize_short_digits and token.isdigit() and len(token) <= 3:
            return _quantify(r"\d", len(token), len(token))
        return escape_literal(token)
    chars = set("".join(values))
    if len(chars) == 1 and lo == hi:
        return _quantify(escape_literal(next(iter(chars))), lo, hi)
    return _quantify(named_class(chars), lo, hi)


def _slot_pattern(rows: list[list[tuple[str, str]]], generalize_short_digits: bool) -> str:
    pieces = []
    for index in range(len(rows[0])):
        values = [row[index][1] for row in rows]
        pieces.append(_piece_for_values(values, generalize_short_digits))
    return "^" + "".join(pieces) + "$"


def _repeat_pattern(rows: list[list[tuple[str, str]]]) -> str | None:
    slots = [index for index in range(len(rows[0])) if rows[0][index][0] == "slot"]
    delims = [index for index in range(len(rows[0])) if rows[0][index][0] == "delim"]
    if len(slots) < 3 or len(delims) != len(slots) - 1:
        return None
    if any(slots[i] > delims[i] or delims[i] > slots[i + 1] for i in range(len(delims))):
        return None
    delim_values = {rows[0][index][1] for index in delims}
    if len(delim_values) != 1 or any(len({row[index][1] for row in rows}) != 1 for index in delims):
        return None
    lengths = [len(row[index][1]) for row in rows for index in slots]
    chars = {ch for row in rows for index in slots for ch in row[index][1]}
    if not chars <= DIGITS:
        return None
    lo, hi = min(lengths), max(lengths)
    delim = escape_literal(next(iter(delim_values)))
    unit = _quantify(r"\d", lo, hi)
    return f"^(?:{unit}{delim}){{{len(slots) - 1}}}{unit}$"


def _catalog_patterns(rows: list[list[tuple[str, str]]]) -> list[str]:
    patterns = []
    base = []
    for index in range(len(rows[0])):
        base.append([row[index][1] for row in rows])
    for index, values in enumerate(base):
        if rows[0][index][0] != "slot":
            continue
        observed = set(values)
        if len(observed) < 2:
            continue
        for _name, catalog in CATALOGS:
            if observed <= set(catalog) and len(observed) < len(catalog):
                pieces = []
                for piece_index, piece_values in enumerate(base):
                    if piece_index == index:
                        body = "|".join(escape_literal(item) for item in catalog)
                        pieces.append(f"(?:{body})")
                    else:
                        pieces.append(_piece_for_values(piece_values, False))
                patterns.append("^" + "".join(pieces) + "$")
                break
        if len(patterns) >= 3:
            break
    return patterns


def _affixes(strings: list[str]) -> tuple[str, str]:
    prefix = strings[0]
    suffix = strings[0]
    for text in strings[1:]:
        while prefix and not text.startswith(prefix):
            prefix = prefix[:-1]
        while suffix and not text.endswith(suffix):
            suffix = suffix[:-1]
    if prefix and suffix and len(prefix) + len(suffix) > min(len(text) for text in strings):
        overlap = len(prefix) + len(suffix) - min(len(text) for text in strings)
        suffix = suffix[overlap:]
    return prefix, suffix


def _affix_pattern(strings: list[str]) -> str | None:
    prefix, suffix = _affixes(strings)
    if not prefix and not suffix:
        return None
    middles = []
    for text in strings:
        body = text[len(prefix):] if prefix else text
        if suffix:
            body = body[: len(body) - len(suffix)]
        middles.append(body)
    nonempty = [middle for middle in middles if middle]
    if not nonempty:
        inner = ""
    else:
        chars = set("".join(nonempty))
        lo, hi = min(len(middle) for middle in nonempty), max(len(middle) for middle in nonempty)
        atom = named_class(chars) if len(chars) != 1 else escape_literal(next(iter(chars)))
        inner = _quantify(atom, lo, hi)
        if any(middle == "" for middle in middles):
            inner = f"(?:{inner})?"
    return "^" + escape_literal(prefix) + inner + escape_literal(suffix) + "$"


def _widen_classes(node: Node) -> Node:
    ladder = {
        r"\d": "[0-9a-f]",
        r"[0-9]": "[0-9a-f]",
        r"[0-9a-f]": "[a-z0-9]",
        r"[0-9A-F]": "[A-Z0-9]",
        r"[a-z]": "[a-z0-9_]",
        r"[A-Z]": "[A-Z0-9]",
        r"[A-Za-z]": "[A-Za-z0-9]",
        r"[A-Za-z0-9]": r"\w",
        r"\w": r"\S",
    }

    def convert(current: Node) -> Node:
        text = ladder.get(current.text, current.text) if current.op == "CLS" else current.text
        lo, hi = current.lo, current.hi
        if current.op in {"CLS", "DOT"} and hi is not None:
            lo = max(0, lo - 1)
            hi = hi + 2
        return current.replace(text=text, lo=lo, hi=hi)

    return compress(_map_nodes(node, convert))


def _delimiter_mutator(node: Node) -> list[Node]:
    if node.op != "LIT" or node.lo != 1 or node.hi != 1 or len(node.text) != 1:
        return []
    if node.text not in "-_./: ":
        return []
    return [Node("CLS", text="[-_./: ]", lo=1, hi=1)]


def propose_patterns(strings: list[str], pool_limit: int = 80) -> list[Candidate]:
    """Return visible-consistent candidates, heuristic first, then round-robin tags.

    Structural rebuilds are added before the larger mutation lists so a raw cap
    cannot crowd them out. Patterns that reject a visible string are dropped
    before the pool is truncated.
    """
    heuristic = infer_regex(strings)
    raw_limit = max(pool_limit * 3, pool_limit)
    sink: dict[str, str] = {heuristic: "heuristic"}
    rows = _aligned_parts(strings)
    if rows is not None:
        tight_pattern = _slot_pattern(rows, False)
        _add_pattern(sink, tight_pattern, _family_tag(tight_pattern, "slot_tight"), raw_limit)
        general_pattern = _slot_pattern(rows, True)
        _add_pattern(sink, general_pattern, _family_tag(general_pattern, "slot_digit_general"), raw_limit)
        repeated = _repeat_pattern(rows)
        if repeated:
            _add_pattern(sink, repeated, "repeat", raw_limit)
        for pattern in _catalog_patterns(rows):
            _add_pattern(sink, pattern, "catalog", raw_limit)
        tight = parse_pattern(_slot_pattern(rows, False))
        if tight is not None:
            _add_pattern(sink, anchor(_widen_classes(tight)), "slot_widen", raw_limit)
    affix = _affix_pattern(strings)
    if affix:
        _add_pattern(sink, affix, "affix", raw_limit)
    root = parse_pattern(heuristic)
    if root is not None:
        for alphabet in ("[0-9a-f]", "[0-9A-F]"):
            _add_pattern(sink, anchor(_hex_collapse(root, alphabet)), "hex_collapse", raw_limit)
        _render_added(root, _literal_digit_mutator, "literal_digit", sink, raw_limit)
        _render_added(root, _width_mutator(False), "width", sink, raw_limit)
        _render_added(root, _width_mutator(True), "width_aggressive", sink, raw_limit)
        _render_added(root, _optional_mutator, "optional", sink, raw_limit)
        _render_added(root, _delimiter_mutator, "delimiter", sink, raw_limit)
        for variant_tag, variant in _class_sites(root):
            _add_pattern(sink, anchor(variant), variant_tag, raw_limit)

    kept = [Candidate(pattern, tag) for pattern, tag in sink.items() if _keep(pattern, strings, heuristic)]
    ordered = _order(kept, heuristic)
    return ordered[:pool_limit]


def _class_sites(root: Node) -> list[tuple[str, Node]]:
    paired: list[tuple[str, Node]] = []

    def walk(node: Node) -> list[tuple[str, Node]]:
        local = [(tag, compress(variant)) for tag, variant in _class_mutator(node)]
        kids = list(node.kids)
        for index, kid in enumerate(kids):
            for tag, new_kid in walk(kid):
                updated = kids.copy()
                updated[index] = new_kid
                local.append((tag, compress(node.replace(kids=tuple(updated)))))
        return local

    paired.extend(walk(root))
    return paired


def _keep(pattern: str, strings: list[str], heuristic: str) -> bool:
    if pattern == heuristic:
        return True
    try:
        compiled = compile_regex(pattern)
    except re.error:
        return False
    try:
        if not all(compiled.fullmatch(text) for text in strings):
            return False
    except re.error:
        return False
    return not is_catastrophic(pattern, compiled)


def _bucket_key(item: Candidate, heuristic: str, tag: str) -> tuple:
    """Put the tightest hex explanation first. Other families stay near the heuristic."""
    if tag == "hex_collapse":
        node = parse_pattern(item.pattern)
        if node is None:
            return (1e9, 99, len(item.pattern), item.pattern)
        return (data_bits(node, False), description_nodes(node), len(item.pattern), item.pattern)
    return (abs(len(item.pattern) - len(heuristic)), len(item.pattern), item.pattern)


def _order(items: list[Candidate], heuristic: str, priority: tuple[str, ...] = TAG_PRIORITY) -> list[Candidate]:
    buckets: dict[str, list[Candidate]] = {}
    first: Candidate | None = None
    for item in items:
        if item.pattern == heuristic:
            first = item
            continue
        buckets.setdefault(item.tag, []).append(item)
    for tag, bucket in buckets.items():
        bucket.sort(key=lambda item, tag=tag: _bucket_key(item, heuristic, tag))
        buckets[tag] = bucket
    tags = [tag for tag in priority if tag in buckets]
    tags.extend(sorted(tag for tag in buckets if tag not in tags))
    ordered: list[Candidate] = []
    if first is not None:
        ordered.append(first)
    else:
        ordered.append(Candidate(heuristic, "heuristic"))
    cursors = {tag: 0 for tag in tags}
    while True:
        progressed = False
        for tag in tags:
            cursor = cursors[tag]
            bucket = buckets[tag]
            if cursor < len(bucket):
                ordered.append(bucket[cursor])
                cursors[tag] = cursor + 1
                progressed = True
        if not progressed:
            break
    return ordered


def presented_candidates(strings: list[str], k: int = 8, pool_limit: int = 80) -> tuple[list[Candidate], list[Candidate]]:
    """Return (the K candidates a ranker would see, the full filtered pool)."""
    pool = propose_patterns(strings, pool_limit=pool_limit)
    if len(pool) > pool_limit:
        pool = pool[:pool_limit]
    shown = pool[:k]
    if not shown:
        heuristic = infer_regex(strings)
        shown = [Candidate(heuristic, "heuristic")]
        pool = shown
    return shown, pool


# New structural families are interleaved with the families that already win,
# so Oracle@8 still sees hex collapse and slot rebuilds.
EXPANDED_PRIORITY = (
    "heuristic",
    "mixture",
    "repeat_motif",
    "grouping",
    "hex_collapse",
    "slot_digit_general",
    "slot_tight",
    "optional_span",
    "delimiter_split",
    "segment",
    "span_merge",
    "primitive",
    "align",
    "ast_search",
    "literal_digit",
    "width",
    "width_aggressive",
    "class_tighten",
    "class_widen",
    "affix",
    "catalog",
    "repeat",
    "optional",
    "slot_widen",
    "delimiter",
)


def expanded_candidates(
    strings: list[str],
    k: int = 8,
    pool_limit: int = 160,
) -> tuple[list[Candidate], list[Candidate]]:
    """Union the local neighborhood with structural generators.

    The heuristic stays first. Later slots round-robin across tags so a new
    family is not buried behind dozens of single-site edits.
    """
    from pattern_induction.structural import structural_candidates

    heuristic = infer_regex(strings)
    merged: dict[str, str] = {}
    for item in propose_patterns(strings, pool_limit=max(pool_limit, 80)):
        merged.setdefault(item.pattern, item.tag)
    for item in structural_candidates(strings):
        merged.setdefault(item.pattern, item.tag)
    kept = [Candidate(pattern, tag) for pattern, tag in merged.items() if _keep(pattern, strings, heuristic)]
    ordered = _order(kept, heuristic, EXPANDED_PRIORITY)
    pool = ordered[:pool_limit]
    shown = pool[:k] if pool else [Candidate(heuristic, "heuristic")]
    if not pool:
        pool = shown
    return shown, pool
