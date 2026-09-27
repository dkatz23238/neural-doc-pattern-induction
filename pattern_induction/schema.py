"""Latent document schema: a small structural language that compiles to a regex.

A schema is a short sequence of parts. Literals, classes, repeated groups, and
alternations are explicit fields. The compiler is deterministic, so a model
that predicts those fields is predicting the language rather than ranking a
candidate list.
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass

from pattern_induction.candidates import parse_pattern
from pattern_induction.heuristic import escape_literal

MAX_PARTS = 8
CHAR_SLOTS = 16
ALT_LEN = 4
MAX_LO = 16
MAX_HI = 16
MAX_REP = 6
MAX_STRINGS = 8
MAX_LEN = 64

KIND_END = 0
KIND_LIT = 1
KIND_CLS = 2
KIND_REP = 3
KIND_ALT = 4
N_KIND = 5

CHARSETS = (
    "digit",
    "upper",
    "lower",
    "alpha",
    "alnum",
    "upper_alnum",
    "lower_alnum",
    "hex",
    "hex_upper",
    "word",
    "lower_word",
)
CHARSET_REGEX = {
    "digit": r"\d",
    "upper": r"[A-Z]",
    "lower": r"[a-z]",
    "alpha": r"[A-Za-z]",
    "alnum": r"[A-Za-z0-9]",
    "upper_alnum": r"[A-Z0-9]",
    "lower_alnum": r"[a-z0-9]",
    "hex": r"[0-9a-f]",
    "hex_upper": r"[0-9A-F]",
    "word": r"[A-Za-z0-9_]",
    "lower_word": r"[a-z0-9_]",
}
CHARSET_POOL = {
    "digit": string.digits,
    "upper": string.ascii_uppercase,
    "lower": string.ascii_lowercase,
    "alpha": string.ascii_letters,
    "alnum": string.ascii_letters + string.digits,
    "upper_alnum": string.ascii_uppercase + string.digits,
    "lower_alnum": string.ascii_lowercase + string.digits,
    "hex": "0123456789abcdef",
    "hex_upper": "0123456789ABCDEF",
    "word": string.ascii_letters + string.digits + "_",
    "lower_word": string.ascii_lowercase + string.digits + "_",
}
CLASS_TO_CHARSET = {body: name for name, body in CHARSET_REGEX.items()}

# Index 0 is padding. Index 1 is unknown. Everything else is a character the
# schema can copy out of the visible strings.
CHARS = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    " -_./:#$%+@'\"()[]=,~*"
    "§¥€–副本"
)
PAD = 0
UNK = 1
STOI = {ch: index + 2 for index, ch in enumerate(CHARS)}
ITOS = {index: ch for ch, index in STOI.items()}
VOCAB = 2 + len(CHARS)

TRAIN_DELIMS = list("-_./: ")
VAL_DELIMS = list("#~*=")
PREFIX_ALPHABET = string.ascii_letters + string.digits


@dataclass
class Part:
    kind: int = KIND_END
    optional: int = 0
    charset: int = 0
    lo: int = 0
    hi: int = 0
    rep_lo: int = 0
    rep_hi: int = 0
    delim: int = 0
    chars: tuple[int, ...] = ()

    def text(self) -> str:
        return "".join(ITOS[index] for index in self.chars if index > UNK)


def empty_parts() -> list[Part]:
    return [Part() for _ in range(MAX_PARTS)]


def _quant(lo: int, hi: int) -> str:
    lo = max(0, min(lo, MAX_HI))
    hi = max(lo, min(hi, MAX_HI))
    if lo == 1 and hi == 1:
        return ""
    if lo == hi:
        return "{" + str(lo) + "}"
    return "{" + str(lo) + "," + str(hi) + "}"


def _class_body(charset: int, lo: int, hi: int) -> str:
    name = CHARSETS[max(0, min(charset, len(CHARSETS) - 1))]
    return CHARSET_REGEX[name] + _quant(max(lo, 1), max(hi, max(lo, 1)))


def compile_parts(parts: list[Part]) -> str | None:
    """Compile parts to an anchored regex. None when the parts do not describe one."""
    pieces: list[str] = []
    for part in parts:
        if part.kind == KIND_END:
            break
        if part.kind == KIND_LIT:
            text = part.text()
            if not text:
                continue
            piece = escape_literal(text)
        elif part.kind == KIND_CLS:
            piece = _class_body(part.charset, part.lo, part.hi)
        elif part.kind == KIND_REP:
            delim = ITOS.get(part.delim, "")
            if not delim:
                continue
            atom = _class_body(part.charset, part.lo, part.hi)
            rep_lo = max(0, min(part.rep_lo, MAX_REP))
            rep_hi = max(rep_lo, min(part.rep_hi, MAX_REP))
            piece = "(?:" + escape_literal(delim) + atom + ")" + _quant(rep_lo, rep_hi)
        elif part.kind == KIND_ALT:
            count = max(2, min(part.lo, CHAR_SLOTS // ALT_LEN))
            options = []
            for index in range(count):
                chunk = part.chars[index * ALT_LEN : (index + 1) * ALT_LEN]
                text = "".join(ITOS[item] for item in chunk if item > UNK)
                if text:
                    options.append(escape_literal(text))
            if len(options) < 2:
                continue
            piece = "(?:" + "|".join(options) + ")"
        else:
            return None
        if part.optional:
            piece = "(?:" + piece + ")?"
        pieces.append(piece)
    if not pieces:
        return None
    return "^" + "".join(pieces) + "$"


def encode_text(text: str, width: int) -> tuple[int, ...]:
    values = [STOI.get(ch, UNK) for ch in text[:width]]
    values.extend([PAD] * (width - len(values)))
    return tuple(values)


def encode_strings(strings: list[str]) -> tuple[list[list[int]], list[list[bool]]]:
    """Character ids and a True-on-pad mask, truncated to the model window."""
    ids: list[list[int]] = []
    pad: list[list[bool]] = []
    for text in strings[:MAX_STRINGS]:
        row = [STOI.get(ch, UNK) for ch in text[:MAX_LEN]]
        mask = [False] * len(row)
        row.extend([PAD] * (MAX_LEN - len(row)))
        mask.extend([True] * (MAX_LEN - len(mask)))
        ids.append(row)
        pad.append(mask)
    return ids, pad


def _lit(text: str, optional: int = 0) -> Part:
    return Part(KIND_LIT, optional, lo=len(text), hi=len(text), chars=encode_text(text, CHAR_SLOTS))


def _cls(name: str, lo: int, hi: int, optional: int = 0) -> Part:
    return Part(KIND_CLS, optional, CHARSETS.index(name), lo, hi)


def _rep(name: str, lo: int, hi: int, delim: str, rep_lo: int, rep_hi: int) -> Part:
    return Part(
        KIND_REP,
        0,
        CHARSETS.index(name),
        lo,
        hi,
        rep_lo,
        rep_hi,
        STOI[delim],
    )


def _alt(options: list[str]) -> Part:
    chars: list[int] = []
    for option in options[: CHAR_SLOTS // ALT_LEN]:
        chars.extend(encode_text(option, ALT_LEN))
    chars.extend([PAD] * (CHAR_SLOTS - len(chars)))
    return Part(KIND_ALT, 0, lo=min(len(options), CHAR_SLOTS // ALT_LEN), chars=tuple(chars))


def _finish(parts: list[Part]) -> list[Part]:
    parts = parts[:MAX_PARTS]
    parts.extend(Part() for _ in range(MAX_PARTS - len(parts)))
    return parts


def _draw_class(rng: random.Random, shift: bool) -> tuple[str, int, int]:
    name = rng.choice(CHARSETS)
    if rng.random() < 0.55:
        width = rng.randint(5, 8) if shift else rng.randint(1, 4)
        return name, width, width
    lo = rng.randint(1, 3)
    hi = rng.randint(lo + 1, min(MAX_HI, lo + (6 if shift else 3)))
    return name, lo, hi


def _draw_literal(rng: random.Random, alphabet: str, lo: int, hi: int) -> str:
    width = rng.randint(lo, hi)
    return "".join(rng.choice(alphabet) for _ in range(width))


def sample_parts(rng: random.Random, shift: bool = False) -> list[Part]:
    """Draw one identifiable schema. `shift` holds out train delimiters and stretches lengths."""
    delims = VAL_DELIMS if shift else TRAIN_DELIMS
    prefix_lo, prefix_hi = (6, 8) if shift else (1, 5)
    mode = rng.randrange(6)
    if mode == 0:
        prefix = _draw_literal(rng, PREFIX_ALPHABET, prefix_lo, prefix_hi)
        name, lo, hi = _draw_class(rng, shift)
        return _finish([_lit(prefix), _cls(name, lo, hi)])
    if mode == 1:
        name, lo, hi = _draw_class(rng, shift)
        return _finish([_cls(name, lo, hi)])
    if mode == 2:
        delim = rng.choice(delims)
        left_name, left_lo, left_hi = _draw_class(rng, shift)
        right_name, right_lo, right_hi = _draw_class(rng, shift)
        if rng.random() < 0.65:
            prefix = _draw_literal(rng, PREFIX_ALPHABET, 2, 4)
            return _finish([_lit(prefix + delim), _cls(left_name, left_lo, left_hi), _lit(delim), _cls(right_name, right_lo, right_hi)])
        mid_name, mid_lo, mid_hi = _draw_class(rng, shift)
        return _finish([
            _cls(left_name, left_lo, left_hi),
            _lit(delim),
            _cls(mid_name, mid_lo, mid_hi),
            _lit(delim),
            _cls(right_name, right_lo, right_hi),
        ])
    if mode == 3:
        name, lo, hi = _draw_class(rng, shift)
        delim = rng.choice(delims).strip() or "."
        if delim not in STOI:
            delim = "#" if shift else "-"
        rep_lo = rng.randint(1, 2 if not shift else 3)
        rep_hi = rng.randint(rep_lo, min(MAX_REP, rep_lo + (3 if shift else 2)))
        parts = []
        if rng.random() < 0.6:
            prefix_chars = "§" if (shift and rng.random() < 0.4) else PREFIX_ALPHABET
            parts.append(_lit(_draw_literal(rng, prefix_chars, 1, 2 if shift else 3)))
            if rng.random() < 0.5:
                parts.append(_lit(" ", optional=1))
        parts.append(_cls(name, lo, hi))
        parts.append(_rep(name, lo, hi, delim, rep_lo, rep_hi))
        return _finish(parts)
    if mode == 4:
        n_alts = rng.randint(2, 4)
        length = rng.randint(1, ALT_LEN)
        alphabet = string.ascii_uppercase if rng.random() < 0.5 else string.ascii_lowercase
        options = []
        while len(options) < n_alts:
            token = _draw_literal(rng, alphabet, length, length)
            if token not in options:
                options.append(token)
        return _finish([_alt(options)])
    name, lo, hi = _draw_class(rng, shift)
    tail = rng.choice([".pdf", ".txt", ".log"] if not shift else [".docx", ".csv"])
    return _finish([_cls(name, max(lo, 1), max(hi, lo)), _lit(tail, optional=1)])


def _sample_piece(part: Part, rng: random.Random, omit: bool) -> str:
    if omit and part.optional:
        return ""
    if part.kind == KIND_LIT:
        return part.text()
    if part.kind == KIND_CLS:
        pool = CHARSET_POOL[CHARSETS[part.charset]]
        width = rng.randint(max(part.lo, 1), max(part.hi, max(part.lo, 1)))
        return "".join(rng.choice(pool) for _ in range(width))
    if part.kind == KIND_REP:
        pool = CHARSET_POOL[CHARSETS[part.charset]]
        delim = ITOS[part.delim]
        count = rng.randint(part.rep_lo, max(part.rep_hi, part.rep_lo))
        chunks = []
        for _ in range(count):
            width = rng.randint(max(part.lo, 1), max(part.hi, max(part.lo, 1)))
            chunks.append(delim + "".join(rng.choice(pool) for _ in range(width)))
        return "".join(chunks)
    if part.kind == KIND_ALT:
        count = part.lo
        options = []
        for index in range(count):
            chunk = part.chars[index * ALT_LEN : (index + 1) * ALT_LEN]
            options.append("".join(ITOS[item] for item in chunk if item > UNK))
        return rng.choice([option for option in options if option])
    return ""


def sample_string(parts: list[Part], rng: random.Random, omit_optional: bool | None = None) -> str:
    pieces = []
    for part in parts:
        if part.kind == KIND_END:
            break
        omit = (rng.random() < 0.5) if omit_optional is None else omit_optional
        pieces.append(_sample_piece(part, rng, omit))
    return "".join(pieces)


def _identifiable(parts: list[Part], strings: list[str]) -> bool:
    if any(not text or len(text) > MAX_LEN for text in strings):
        return False
    pattern = compile_parts(parts)
    if pattern is None:
        return False
    try:
        compiled = re.compile(pattern)
    except re.error:
        return False
    if not all(compiled.fullmatch(text) for text in strings):
        return False
    active = [part for part in parts if part.kind != KIND_END]
    for part in active:
        if part.optional and part.kind == KIND_LIT:
            token = part.text()
            present = sum(token in text for text in strings)
            if present == 0 or present == len(strings):
                return False
    for part in active:
        if part.kind == KIND_ALT:
            for index in range(part.lo):
                chunk = part.chars[index * ALT_LEN : (index + 1) * ALT_LEN]
                token = "".join(ITOS[item] for item in chunk if item > UNK)
                if token and not any(token in text for text in strings):
                    return False
        if part.kind == KIND_CLS and part.hi > part.lo:
            widths = set()
            # Width is not read back here; variety of whole-string lengths is a proxy.
            widths.update(len(text) for text in strings)
            if len(widths) < 2:
                return False
    return True


def sample_task(rng: random.Random, n_visible: int, shift: bool = False) -> tuple[list[Part], list[str], str]:
    for _ in range(50):
        parts = sample_parts(rng, shift=shift)
        strings = [sample_string(parts, rng) for _ in range(n_visible)]
        if _identifiable(parts, strings):
            pattern = compile_parts(parts)
            if pattern is not None:
                return parts, strings, pattern
    parts = _finish([_lit("AB"), _cls("digit", 4, 4)])
    strings = [sample_string(parts, rng) for _ in range(n_visible)]
    return parts, strings, compile_parts(parts) or "^AB\\d{4}$"


def parts_to_arrays(parts: list[Part]) -> dict[str, list]:
    return {
        "kind": [part.kind for part in parts],
        "optional": [part.optional for part in parts],
        "charset": [part.charset for part in parts],
        "lo": [part.lo for part in parts],
        "hi": [part.hi for part in parts],
        "rep_lo": [part.rep_lo for part in parts],
        "rep_hi": [part.rep_hi for part in parts],
        "delim": [part.delim for part in parts],
        "chars": [list(part.chars or (PAD,) * CHAR_SLOTS) for part in parts],
    }


def arrays_to_parts(rows: dict[str, list]) -> list[Part]:
    parts = []
    for index in range(MAX_PARTS):
        chars = tuple(int(value) for value in rows["chars"][index])
        parts.append(
            Part(
                int(rows["kind"][index]),
                int(rows["optional"][index]),
                int(rows["charset"][index]),
                int(rows["lo"][index]),
                int(rows["hi"][index]),
                int(rows["rep_lo"][index]),
                int(rows["rep_hi"][index]),
                int(rows["delim"][index]),
                chars,
            )
        )
    return parts


def _literal_run(node) -> str | None:
    if node.op != "LIT" or node.hi != node.lo:
        return None
    return node.text * node.lo


def from_node(node) -> list[Part] | None:
    """Map a parsed gold regex into the schema when it is in the fragment."""
    if node is None:
        return None
    optional = 1 if node.lo == 0 and node.hi == 1 else 0
    if node.op == "SEQ":
        parts: list[Part] = []
        index = 0
        kids = list(node.kids)
        while index < len(kids):
            kid = kids[index]
            mapped = from_node(kid)
            if mapped is None:
                return None
            if (
                len(mapped) == 1
                and mapped[0].kind == KIND_CLS
                and index + 1 < len(kids)
                and kids[index + 1].op == "REPEAT"
            ):
                repeated = _repeat_part(kids[index + 1])
                if repeated is not None:
                    parts.extend(mapped)
                    parts.append(repeated)
                    index += 2
                    continue
            parts.extend(mapped)
            index += 1
        return parts or None
    if node.op == "REPEAT" and node.lo == 0 and node.hi == 1 and node.kids:
        inner = from_node(node.kids[0])
        if inner is None or len([part for part in inner if part.kind != KIND_END]) != 1:
            return None
        inner[0].optional = 1
        return inner
    if node.op == "LIT":
        text = _literal_run(node)
        if text is None or any(ch not in STOI for ch in text) or len(text) > CHAR_SLOTS:
            return None
        return [_lit(text, optional)]
    if node.op == "CLS":
        name = CLASS_TO_CHARSET.get(node.text)
        if name is None or node.hi is None:
            return None
        return [_cls(name, node.lo, node.hi, optional)]
    if node.op == "ALT":
        options = []
        for kid in node.kids:
            text = _literal_run(kid)
            if text is None or len(text) > ALT_LEN or any(ch not in STOI for ch in text):
                return None
            options.append(text)
        if not 2 <= len(options) <= CHAR_SLOTS // ALT_LEN:
            return None
        return [_alt(options)]
    return None


def _repeat_part(node) -> Part | None:
    if node.op != "REPEAT" or not node.kids:
        return None
    hi = MAX_REP if node.hi is None else node.hi
    if hi > MAX_REP or node.lo > MAX_REP:
        return None
    inner = node.kids[0]
    if inner.op != "SEQ" or len(inner.kids) != 2:
        return None
    delim, cls = inner.kids
    delim_text = _literal_run(delim)
    if delim_text is None or len(delim_text) != 1 or delim_text not in STOI:
        return None
    mapped = from_node(cls)
    if mapped is None or len([part for part in mapped if part.kind != KIND_END]) != 1:
        return None
    atom = mapped[0]
    if atom.kind != KIND_CLS:
        return None
    return _rep(CHARSETS[atom.charset], atom.lo, atom.hi, delim_text, node.lo, hi)


def from_regex(pattern: str) -> list[Part] | None:
    node = parse_pattern(pattern)
    if node is None:
        return None
    try:
        parts = from_node(node)
    except (KeyError, IndexError, ValueError):
        return None
    if parts is None:
        return None
    active = [part for part in parts if part.kind != KIND_END]
    if not active or len(active) > MAX_PARTS:
        return None
    return _finish(active)


def properties(pattern: str) -> dict[str, str] | None:
    """Structural facts shared by a gold regex and a compiled schema."""
    node = parse_pattern(pattern)
    if node is None:
        return None
    facts = {
        "prefix": "",
        "suffix": "",
        "alternation": "0",
        "optional": "0",
        "repeat": "0",
        "digit": "0",
        "upper": "0",
        "lower": "0",
        "hex": "0",
        "delim": "",
    }
    lits: list[str] = []

    def walk(item) -> None:
        if item.lo == 0:
            facts["optional"] = "1"
        if item.op == "REPEAT" or (item.hi is not None and item.lo != item.hi):
            facts["repeat"] = "1"
        if item.op == "ALT":
            facts["alternation"] = "1"
        if item.op == "LIT":
            lits.append(item.text)
        if item.op == "CLS":
            name = CLASS_TO_CHARSET.get(item.text, "")
            if name in {"digit", "alnum", "upper_alnum", "lower_alnum", "word", "lower_word", "hex", "hex_upper"}:
                facts["digit"] = "1"
            if name in {"upper", "alpha", "alnum", "upper_alnum", "word", "hex_upper"}:
                facts["upper"] = "1"
            if name in {"lower", "alpha", "alnum", "lower_alnum", "word", "lower_word", "hex"}:
                facts["lower"] = "1"
            if "hex" in name:
                facts["hex"] = "1"
        for kid in item.kids:
            walk(kid)

    walk(node)
    if node.op == "SEQ" and node.kids and node.kids[0].op == "LIT" and node.kids[0].lo == node.kids[0].hi:
        facts["prefix"] = node.kids[0].text * node.kids[0].lo
    if node.op == "LIT" and node.lo == node.hi:
        facts["prefix"] = node.text * node.lo
        facts["suffix"] = facts["prefix"]
    if node.op == "SEQ" and node.kids and node.kids[-1].op == "LIT" and node.kids[-1].lo == node.kids[-1].hi:
        facts["suffix"] = node.kids[-1].text * node.kids[-1].lo
    delims = [text for text in lits if len(text) == 1 and not text.isalnum()]
    facts["delim"] = delims[0] if delims else ""
    return facts


PROPERTY_NAMES = (
    "prefix",
    "suffix",
    "alternation",
    "optional",
    "repeat",
    "digit",
    "upper",
    "lower",
    "hex",
    "delim",
)
