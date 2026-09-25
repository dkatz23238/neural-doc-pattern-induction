"""B1 structural heuristic.

Infers a regex from positive strings using common literals, character classes,
fixed and variable spans, and optional regions. It does not memorize the
examples as an alternation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def _kind(ch: str) -> str:
    if ch.isdigit():
        return "d"
    if ch.isupper():
        return "u"
    if ch.islower():
        return "l"
    if ch.isspace():
        return "s"
    return "p"


CLASS_ATOM = {
    "d": r"\d",
    "u": r"[A-Z]",
    "l": r"[a-z]",
    "a": r"[A-Za-z]",
    "n": r"[A-Za-z0-9]",
    "s": r"\s",
    "x": r".",
}


_META = set(r".^$*+?{}[]|()\\")


def escape_literal(text: str) -> str:
    return "".join("\\" + ch if ch in _META else ch for ch in text)


def _char_class(chars: set[str]) -> str:
    ordered = sorted(chars)
    body = []
    for ch in ordered:
        if ch == "\\":
            body.append("\\\\")
        elif ch == "]":
            body.append("\\]")
        elif ch == "^":
            body.append("\\^")
        elif ch == "-":
            body.append("\\-")
        else:
            body.append(re.escape(ch) if not ch.isalnum() and ch not in "-]" else ch)
    # Keep '-' at the end for readability when present.
    text = "".join(body).replace("\\-", "")
    if "-" in chars:
        text += "-"
    return f"[{text}]"


def _atom_for_chars(chars: set[str]) -> tuple[str, str]:
    """Return ('lit', text) or ('atom', atom)."""
    if len(chars) == 1:
        return ("lit", next(iter(chars)))
    if all(c.isdigit() for c in chars):
        return ("atom", r"\d")
    if all(c.isupper() for c in chars):
        return ("atom", r"[A-Z]")
    if all(c.islower() for c in chars):
        return ("atom", r"[a-z]")
    if all(c.isalpha() for c in chars):
        return ("atom", r"[A-Za-z]")
    if all(c.isalnum() for c in chars):
        return ("atom", r"[A-Za-z0-9]")
    if all(c.isspace() for c in chars):
        return ("atom", r"\s")
    if len(chars) <= 8:
        return ("atom", _char_class(chars))
    return ("atom", r".")


def _quantify(atom: str, count: int) -> str:
    if count == 1:
        return atom
    return f"{atom}{{{count}}}"


def _quantify_range(atom: str, lo: int, hi: int, optional: bool = False) -> str:
    if optional and lo == hi == 0:
        return ""
    if lo == hi:
        piece = _quantify(atom, lo) if not optional else f"(?:{_quantify(atom, lo)})?"
        return piece
    if lo == 0 and hi == 1:
        return f"{atom}?"
    if lo == 0:
        inner = f"{atom}{{{hi}}}" if False else f"{atom}{{0,{hi}}}"
        return inner
    piece = f"{atom}{{{lo},{hi}}}"
    if optional:
        return f"(?:{piece})?"
    return piece


def column_regex(strings: list[str]) -> str | None:
    lengths = {len(s) for s in strings}
    if len(lengths) != 1 or not strings or lengths.pop() == 0:
        return None
    width = len(strings[0])
    parts: list[tuple] = []
    for i in range(width):
        chars = {s[i] for s in strings}
        mode, value = _atom_for_chars(chars)
        if mode == "lit":
            if parts and parts[-1][0] == "lit":
                parts[-1] = ("lit", parts[-1][1] + value)
            else:
                parts.append(("lit", value))
        else:
            if parts and parts[-1][0] == "atom" and parts[-1][1] == value:
                parts[-1] = ("atom", value, parts[-1][2] + 1)
            else:
                parts.append(("atom", value, 1))
    rendered = []
    for part in parts:
        if part[0] == "lit":
            rendered.append(escape_literal(part[1]))
        else:
            rendered.append(_quantify(part[1], part[2]))
    return "^" + "".join(rendered) + "$"


@dataclass
class Tok:
    kind: str
    text: str


def tokenize(text: str) -> list[Tok]:
    tokens: list[Tok] = []
    for ch in text:
        kind = _kind(ch)
        if tokens and tokens[-1].kind == kind:
            tokens[-1] = Tok(kind, tokens[-1].text + ch)
        else:
            tokens.append(Tok(kind, ch))
    return tokens


@dataclass
class Slot:
    kind: str
    literal: str | None
    min_len: int
    max_len: int
    optional: bool = False
    chars: set[str] = field(default_factory=set)

    @staticmethod
    def from_tok(tok: Tok) -> "Slot":
        return Slot(tok.kind, tok.text, len(tok.text), len(tok.text), False, set(tok.text))


def _kinds_compatible(a: str, b: str) -> str | None:
    if a == b:
        return a
    letters = {"u", "l", "a"}
    if a in letters and b in letters:
        return "a"
    if {a, b} <= {"d", "u", "l", "a", "n"} and "d" in (a, b):
        return None
    return None


def _can_match(slot: Slot, tok: Tok) -> bool:
    if slot.literal is not None and slot.kind == "p" and tok.kind == "p":
        return slot.literal == tok.text
    merged = _kinds_compatible(slot.kind, tok.kind)
    return merged is not None


def _merge_slot(slot: Slot, tok: Tok) -> Slot:
    kind = _kinds_compatible(slot.kind, tok.kind) or slot.kind
    literal = slot.literal
    if literal != tok.text:
        literal = None
    return Slot(
        kind=kind,
        literal=literal,
        min_len=min(slot.min_len, len(tok.text)),
        max_len=max(slot.max_len, len(tok.text)),
        optional=slot.optional,
        chars=set(slot.chars) | set(tok.text),
    )


def _align(slots: list[Slot], toks: list[Tok]) -> list[tuple[Slot | None, Tok | None]] | None:
    n, m = len(slots), len(toks)
    neg = -10**6
    dp = [[neg] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] - 2
        back[i][0] = "slot"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] - 2
        back[0][j] = "tok"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = dp[i - 1][j] - 2
            how = "slot"
            alt = dp[i][j - 1] - 2
            if alt > best:
                best, how = alt, "tok"
            if _can_match(slots[i - 1], toks[j - 1]):
                bonus = 4
                if slots[i - 1].kind == "p":
                    bonus = 8
                # Prefer matching the token that sits against the same separator.
                if (
                    i < n
                    and j < m
                    and slots[i].kind == "p"
                    and toks[j].kind == "p"
                    and slots[i].literal == toks[j].text
                ):
                    bonus += 3
                score = dp[i - 1][j - 1] + bonus
                if score >= best:
                    best, how = score, "match"
            dp[i][j] = best
            back[i][j] = how
    if dp[n][m] < 0:
        return None
    out: list[tuple[Slot | None, Tok | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        how = back[i][j]
        if how == "match":
            out.append((slots[i - 1], toks[j - 1]))
            i -= 1
            j -= 1
        elif how == "slot":
            out.append((slots[i - 1], None))
            i -= 1
        elif how == "tok":
            out.append((None, toks[j - 1]))
            j -= 1
        else:
            return None
    out.reverse()
    return out


def _render_slot(slot: Slot) -> str:
    if slot.literal is not None and slot.min_len == slot.max_len == len(slot.literal):
        piece = escape_literal(slot.literal)
        if slot.optional:
            return f"(?:{piece})?"
        return piece
    if slot.kind == "p" and slot.literal is None and slot.min_len == slot.max_len == 1:
        atom = _char_class(slot.chars) if slot.chars else r"\W"
    else:
        atom = CLASS_ATOM.get(slot.kind, r".")
    return _quantify_range(atom, slot.min_len, slot.max_len, slot.optional)


def _bind_optional_groups(slots: list[Slot]) -> str:
    """Render slots, binding an optional separator plus the following optional run."""
    parts: list[str] = []
    i = 0
    while i < len(slots):
        slot = slots[i]
        if (
            slot.optional
            and slot.kind == "p"
            and i + 1 < len(slots)
            and slots[i + 1].optional
            and slots[i + 1].kind in {"d", "u", "l", "a", "n"}
        ):
            left = _render_slot(Slot(slot.kind, slot.literal, slot.min_len, slot.max_len, False, slot.chars))
            right = _render_slot(
                Slot(
                    slots[i + 1].kind,
                    slots[i + 1].literal,
                    slots[i + 1].min_len,
                    slots[i + 1].max_len,
                    False,
                    slots[i + 1].chars,
                )
            )
            parts.append(f"(?:{left}{right})?")
            i += 2
            continue
        parts.append(_render_slot(slot))
        i += 1
    return "^" + "".join(parts) + "$"


def token_regex(strings: list[str]) -> str | None:
    if not strings:
        return None
    slots = [Slot.from_tok(tok) for tok in tokenize(strings[0])]
    for text in strings[1:]:
        aligned = _align(slots, tokenize(text))
        if aligned is None:
            return None
        rebuilt: list[Slot] = []
        for slot, tok in aligned:
            if slot is not None and tok is not None:
                rebuilt.append(_merge_slot(slot, tok))
            elif slot is not None:
                slot.optional = True
                rebuilt.append(slot)
            elif tok is not None:
                created = Slot.from_tok(tok)
                created.optional = True
                rebuilt.append(created)
        slots = rebuilt
    if sum(1 for slot in slots if slot.optional) > 4:
        return None
    return _bind_optional_groups(slots)


def _affix_regex(strings: list[str]) -> str | None:
    if not strings:
        return None
    prefix = strings[0]
    suffix = strings[0]
    for text in strings[1:]:
        while prefix and not text.startswith(prefix):
            prefix = prefix[:-1]
        while suffix and not text.endswith(suffix):
            suffix = suffix[:-1]
        if prefix and suffix and prefix.endswith(suffix) and len(prefix) + len(suffix) > min(map(len, strings)):
            # Overlapping affixes on short strings: keep the prefix only.
            suffix = ""
    if suffix and prefix and len(prefix) + len(suffix) > min(len(s) for s in strings):
        overlap = len(prefix) + len(suffix) - min(len(s) for s in strings)
        suffix = suffix[overlap:]
    middles = []
    for text in strings:
        body = text
        if prefix:
            body = body[len(prefix) :]
        if suffix:
            body = body[: len(body) - len(suffix)] if suffix else body
        middles.append(body)
    if any(m == "" for m in middles) and any(m != "" for m in middles):
        # Optional middle. Still require a class if non-empty middles agree.
        nonempty = [m for m in middles if m]
        inner = _uniform_body(nonempty)
        if inner is None:
            return None
        piece = f"(?:{inner})?"
    else:
        piece = _uniform_body(middles)
        if piece is None:
            return None
    return "^" + escape_literal(prefix) + piece + escape_literal(suffix) + "$"


def _uniform_body(parts: list[str]) -> str | None:
    if not parts:
        return ""
    if all(p == "" for p in parts):
        return ""
    kinds = []
    for part in parts:
        if not part:
            return None
        ks = {_kind(ch) for ch in part}
        if len(ks) != 1:
            return None
        kinds.append(next(iter(ks)))
    if len(set(kinds)) != 1:
        return None
    kind = kinds[0]
    lo, hi = min(map(len, parts)), max(map(len, parts))
    if kind == "p":
        if len(set(parts)) == 1:
            return escape_literal(parts[0])
        return None
    atom = CLASS_ATOM[kind if kind != "u" else "u"]
    if kind == "l":
        atom = CLASS_ATOM["l"]
    return _quantify_range(atom, lo, hi)


def _accepts_all(pattern: str, strings: list[str]) -> bool:
    try:
        compiled = re.compile(pattern)
    except re.error:
        return False
    return all(compiled.fullmatch(s) for s in strings)


def _specificity(pattern: str) -> tuple:
    unbounded = len(re.findall(r"(?<!\\)[*+]", pattern)) + len(
        re.findall(r"\{0,", pattern)
    )
    literals = len(re.findall(r"[A-Za-z0-9]", pattern))
    return (unbounded, pattern.count("|"), len(pattern), -literals)


def infer_regex(strings: list[str]) -> str:
    """Return one anchored Python regex that matches every input string."""
    cleaned = [s for s in strings if s is not None]
    if not cleaned:
        return "^$"
    if len(set(cleaned)) == 1:
        return "^" + escape_literal(cleaned[0]) + "$"
    candidates = []
    for builder in (column_regex, token_regex, _affix_regex):
        pattern = builder(cleaned)
        if pattern and _accepts_all(pattern, cleaned):
            candidates.append(pattern)
    if not candidates:
        lo, hi = min(map(len, cleaned)), max(map(len, cleaned))
        return "^" + _quantify_range(r".", lo, hi) + "$"
    return min(candidates, key=_specificity)
