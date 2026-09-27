"""Synthetic episodes for set-conditioned membership.

Each episode is a support set S and a query x. The label is whether x belongs
to the language that generated S. Training never sees the held-out delimiter
bytes. The shifted split uses those bytes, longer spans, and harder schemas.
Negatives are near misses: edits, wrong classes, wrong repeats, and strings a
non-gold structural candidate would accept.
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass, field, replace

from pattern_induction.candidates import expanded_candidates, pattern_cost
from pattern_induction.heuristic import infer_regex
from pattern_induction.schema import (
    ALT_LEN,
    CHAR_SLOTS,
    CHARSET_POOL,
    CHARSETS,
    ITOS,
    KIND_ALT,
    KIND_CLS,
    KIND_END,
    KIND_LIT,
    KIND_REP,
    MAX_HI,
    MAX_REP,
    STOI,
    TRAIN_DELIMS,
    UNK,
    Part,
    compile_parts,
    encode_text,
    sample_string,
    sample_task,
)

PAD_ID = 256
MAX_LEN = 48
MAX_SUPPORT = 6
HELD_OUT = set("#~*=§")
_SAFE = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_. /:")


@dataclass
class Episode:
    support: list[str]
    query: str
    label: int
    pattern: str
    split: str
    kind: str
    invariant: list[int] = field(default_factory=list)

    def held_out(self) -> bool:
        blob = "".join(self.support) + self.query
        return any(ch in HELD_OUT for ch in blob)


def fits(text: str) -> bool:
    return bool(text) and len(text) <= MAX_LEN and len(text.encode("utf-8")) <= MAX_LEN


def _active(parts: list[Part]) -> list[Part]:
    return [part for part in parts if part.kind != KIND_END]


def _hard(parts: list[Part]) -> bool:
    active = _active(parts)
    return any(part.kind in (KIND_REP, KIND_ALT) or part.optional for part in active) or len(active) >= 4


def _clean(text: str, shift: bool) -> bool:
    if shift:
        return fits(text)
    return fits(text) and not any(ch in HELD_OUT for ch in text)


def _piece(part: Part, rng: random.Random) -> tuple[str, list[int]]:
    if part.kind == KIND_LIT:
        text = part.text()
        return text, [1] * len(text)
    if part.kind == KIND_CLS:
        pool = CHARSET_POOL[CHARSETS[part.charset]]
        width = rng.randint(max(part.lo, 1), max(part.hi, max(part.lo, 1)))
        text = "".join(rng.choice(pool) for _ in range(width))
        return text, [0] * len(text)
    if part.kind == KIND_REP:
        pool = CHARSET_POOL[CHARSETS[part.charset]]
        delim = ITOS.get(part.delim, "")
        count = rng.randint(part.rep_lo, max(part.rep_hi, part.rep_lo))
        chars: list[str] = []
        flags: list[int] = []
        for _ in range(count):
            width = rng.randint(max(part.lo, 1), max(part.hi, max(part.lo, 1)))
            chars.append(delim)
            flags.extend([1] * len(delim))
            body = "".join(rng.choice(pool) for _ in range(width))
            chars.append(body)
            flags.extend([0] * len(body))
        return "".join(chars), flags
    if part.kind == KIND_ALT:
        options = []
        for index in range(part.lo):
            chunk = part.chars[index * ALT_LEN : (index + 1) * ALT_LEN]
            option = "".join(ITOS[item] for item in chunk if item > UNK)
            if option:
                options.append(option)
        text = rng.choice(options) if options else ""
        return text, [0] * len(text)
    return "", []


def sample_annotated(parts: list[Part], rng: random.Random) -> tuple[str, list[int]]:
    chars: list[str] = []
    flags: list[int] = []
    for part in parts:
        if part.kind == KIND_END:
            break
        if part.optional and rng.random() < 0.5:
            continue
        piece, piece_flags = _piece(part, rng)
        chars.append(piece)
        flags.extend(piece_flags)
    return "".join(chars), flags


def byte_flags(text: str, flags: list[int]) -> list[int]:
    if len(flags) != len(text):
        return []
    out: list[int] = []
    for ch, flag in zip(text, flags):
        out.extend([flag] * len(ch.encode("utf-8")))
    return out


def _language(
    rng: random.Random,
    shift: bool,
    n_support: int,
    prefer_hard: bool,
) -> tuple[list[Part], list[str], str, re.Pattern[str]] | None:
    for _ in range(24):
        parts, strings, pattern = sample_task(rng, n_support, shift=shift)
        if any(not _clean(text, shift) for text in strings):
            continue
        if prefer_hard and not _hard(parts):
            continue
        try:
            compiled = re.compile(pattern)
        except re.error:
            continue
        if not all(compiled.fullmatch(text) for text in strings):
            continue
        return parts, strings, pattern, compiled
    return None


def _positive(parts: list[Part], compiled: re.Pattern[str], rng: random.Random, shift: bool) -> tuple[str, list[int]] | None:
    for _ in range(16):
        text, flags = sample_annotated(parts, rng)
        if _clean(text, shift) and compiled.fullmatch(text):
            return text, flags
    return None


def _alphabet(support: list[str], shift: bool) -> list[str]:
    chars = set(_SAFE)
    chars.update("".join(support))
    if not shift:
        chars -= HELD_OUT
    return sorted(chars) or ["0"]


def _edit_negative(
    text: str,
    compiled: re.Pattern[str],
    rng: random.Random,
    shift: bool,
    support: list[str],
) -> tuple[str, str] | None:
    alphabet = _alphabet(support, shift)
    for _ in range(16):
        chars = list(text)
        if len(chars) < 2:
            return None
        op = rng.randrange(3)
        index = rng.randrange(len(chars))
        kind = "edit"
        if op == 0 and len(chars) > 2:
            del chars[index]
            kind = "repeat"
        elif op == 1:
            replacement = rng.choice(alphabet)
            kind = "class" if replacement.isalnum() != chars[index].isalnum() else "edit"
            chars[index] = replacement
        else:
            chars.insert(index, rng.choice(alphabet))
            kind = "repeat"
        cand = "".join(chars)
        if _clean(cand, shift) and compiled.fullmatch(cand) is None:
            return cand, kind
    return None


def _schema_negative(
    parts: list[Part],
    compiled: re.Pattern[str],
    rng: random.Random,
    shift: bool,
) -> str | None:
    active = [index for index, part in enumerate(parts) if part.kind != KIND_END]
    if not active:
        return None
    for _ in range(8):
        index = rng.choice(active)
        part = parts[index]
        if part.kind == KIND_CLS:
            if rng.random() < 0.5:
                width = min(MAX_HI, max(part.hi + 1, part.lo + 1))
                updated = replace(part, lo=width, hi=width)
            else:
                updated = replace(part, charset=(part.charset + rng.randint(1, len(CHARSETS) - 1)) % len(CHARSETS))
        elif part.kind == KIND_REP:
            if part.rep_hi >= MAX_REP and rng.random() < 0.5:
                updated = replace(part, charset=(part.charset + 1) % len(CHARSETS))
            else:
                count = min(MAX_REP, part.rep_hi + 1)
                updated = replace(part, rep_lo=count, rep_hi=count)
        elif part.kind == KIND_LIT and part.text():
            chars = list(part.text())
            chars[rng.randrange(len(chars))] = rng.choice("XYZ9")
            updated = Part(
                KIND_LIT,
                part.optional,
                lo=len(chars),
                hi=len(chars),
                chars=_encode_literal(chars),
            )
        else:
            continue
        trial = list(parts)
        trial[index] = updated
        text = sample_string(trial, rng)
        if _clean(text, shift) and compiled.fullmatch(text) is None:
            return text
    return None


def _encode_literal(chars: list[str]) -> tuple[int, ...]:
    from pattern_induction.schema import CHAR_SLOTS, encode_text

    return encode_text("".join(chars), CHAR_SLOTS)


def _easy_negative(
    support: list[str],
    compiled: re.Pattern[str],
    rng: random.Random,
    shift: bool,
) -> str | None:
    """A gross mismatch: shuffled or heavily rewritten, still rejected by the gold regex."""
    text = rng.choice(support)
    chars = list(text)
    if len(chars) < 2:
        return None
    rng.shuffle(chars)
    alphabet = _alphabet(support, shift)
    if compiled.fullmatch("".join(chars)):
        for index in range(0, len(chars), 2):
            chars[index] = rng.choice(alphabet)
    cand = "".join(chars)
    if _clean(cand, shift) and compiled.fullmatch(cand) is None:
        return cand
    return None


def structural_negative(
    support: list[str],
    gold: str,
    compiled: re.Pattern[str],
    rng: random.Random,
    shift: bool,
) -> str | None:
    """A string some non-gold structural candidate accepts and the gold rejects."""
    try:
        shown, _pool = expanded_candidates(support, k=8, pool_limit=60)
    except Exception:
        return None
    wrong: list[re.Pattern[str]] = []
    for cand in shown:
        if cand.pattern == gold:
            continue
        try:
            wrong.append(re.compile(cand.pattern))
        except re.error:
            continue
    if not wrong:
        return None
    alphabet = _alphabet(support, shift)
    base = rng.choice(support)
    for _ in range(24):
        chars = list(base)
        if not chars:
            continue
        index = rng.randrange(len(chars))
        op = rng.randrange(3)
        if op == 0 and len(chars) > 2:
            del chars[index]
        elif op == 1:
            chars[index] = rng.choice(alphabet)
        else:
            chars.insert(index, rng.choice(alphabet))
        text = "".join(chars)
        if not _clean(text, shift) or compiled.fullmatch(text):
            continue
        if any(item.fullmatch(text) for item in wrong):
            return text
    return None


def _negative_episode(
    parts: list[Part],
    support: list[str],
    pattern: str,
    compiled: re.Pattern[str],
    rng: random.Random,
    shift: bool,
    structural_rate: float,
    split: str,
    easy_rate: float = 0.0,
) -> Episode | None:
    """Draw a near miss.

    Edits of one support string overlap the examples more than a fresh member
    does, so they are not the whole mixture. Schema samples are fresh strings
    from a wrong language and do not reward that shortcut.
    """
    roll = rng.random()
    if easy_rate and roll < easy_rate:
        order = ("easy", "schema", "structural", "edit")
    elif roll < easy_rate + 0.4:
        order = ("schema", "structural", "edit", "easy")
    elif structural_rate and roll < easy_rate + 0.4 + structural_rate:
        order = ("structural", "schema", "edit", "easy")
    else:
        order = ("edit", "schema", "structural", "easy")
    for kind in order:
        if kind == "easy":
            text = _easy_negative(support, compiled, rng, shift)
            if text is not None:
                return Episode(support, text, 0, pattern, split, "easy")
        elif kind == "structural":
            text = structural_negative(support, pattern, compiled, rng, shift)
            if text is not None:
                return Episode(support, text, 0, pattern, split, "structural")
        elif kind == "schema":
            text = _schema_negative(parts, compiled, rng, shift)
            if text is not None:
                return Episode(support, text, 0, pattern, split, "schema")
        else:
            edited = _edit_negative(rng.choice(support), compiled, rng, shift, support)
            if edited is not None:
                query, name = edited
                return Episode(support, query, 0, pattern, split, name)
    return None


def make_episode(
    rng: random.Random,
    shift: bool,
    label: int,
    n_support: int,
    structural_rate: float,
    prefer_hard: bool,
    split: str,
    easy_rate: float = 0.0,
) -> Episode | None:
    drawn = _language(rng, shift, n_support, prefer_hard)
    if drawn is None:
        return None
    parts, support, pattern, compiled = drawn
    if label == 1:
        positive = _positive(parts, compiled, rng, shift)
        if positive is None:
            return None
        query, flags = positive
        return Episode(support, query, 1, pattern, split, "positive", flags)
    return _negative_episode(parts, support, pattern, compiled, rng, shift, structural_rate, split, easy_rate)


def sample_pair(
    rng: random.Random,
    shift: bool,
    n_support: int,
    structural_rate: float,
    prefer_hard: bool,
    split: str,
    easy_rate: float,
) -> tuple[Episode, Episode] | None:
    """One positive and one negative drawn from the same support set."""
    drawn = _language(rng, shift, n_support, prefer_hard)
    if drawn is None:
        return None
    parts, support, pattern, compiled = drawn
    # An exact support string is a positive with total overlap. Without these,
    # fresh members overlap the set less than one-character edits do, and the
    # label is anti-correlated with surface copying.
    if rng.random() < 0.35:
        pos = Episode(support, rng.choice(support), 1, pattern, split, "positive", [])
    else:
        positive = _positive(parts, compiled, rng, shift)
        if positive is None:
            return None
        query, flags = positive
        pos = Episode(support, query, 1, pattern, split, "positive", flags)
    neg = _negative_episode(parts, support, pattern, compiled, rng, shift, structural_rate, split, easy_rate)
    if neg is None:
        return None
    return pos, neg


def _lit_part(text: str, optional: int = 0) -> Part:
    return Part(KIND_LIT, optional, lo=len(text), hi=len(text), chars=encode_text(text, CHAR_SLOTS))


def _cls_part(name: str, lo: int, hi: int) -> Part:
    return Part(KIND_CLS, 0, CHARSETS.index(name), lo, hi)


def _rep_part(name: str, lo: int, hi: int, delim: str, rep_lo: int, rep_hi: int) -> Part:
    return Part(KIND_REP, 0, CHARSETS.index(name), lo, hi, rep_lo, rep_hi, STOI[delim])


def _alt_part(options: list[str]) -> Part:
    chars: list[int] = []
    for option in options[: CHAR_SLOTS // ALT_LEN]:
        chars.extend(encode_text(option, ALT_LEN))
    if len(chars) < CHAR_SLOTS:
        chars.extend([0] * (CHAR_SLOTS - len(chars)))
    return Part(KIND_ALT, 0, lo=min(len(options), CHAR_SLOTS // ALT_LEN), chars=tuple(chars))


def _vary_class(rng: random.Random) -> tuple[str, int, int]:
    name = rng.choice(("digit", "upper", "lower", "alnum", "hex", "word"))
    if rng.random() < 0.6:
        width = rng.randint(2, 6)
        return name, width, width
    lo = rng.randint(1, 3)
    hi = rng.randint(lo + 1, min(8, lo + 3))
    return name, lo, hi


def _prefix(rng: random.Random, lo: int = 1, hi: int = 4) -> str:
    alphabet = string.ascii_letters
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(lo, hi)))


def _draw_parts(rng: random.Random, level: int) -> list[Part]:
    """Easier languages first. Level 3 is the full schema sampler, handled elsewhere."""
    if level <= 0:
        name, lo, hi = _vary_class(rng)
        parts = [_cls_part(name, lo, hi)]
        if rng.random() < 0.8:
            parts.insert(0, _lit_part(_prefix(rng)))
        return parts
    if level == 1:
        delim = rng.choice(TRAIN_DELIMS)
        n_fields = 2 if rng.random() < 0.75 else 3
        parts = []
        if rng.random() < 0.5:
            parts.append(_lit_part(_prefix(rng, 1, 3)))
        for _index in range(n_fields):
            if parts:
                parts.append(_lit_part(delim))
            name, lo, hi = _vary_class(rng)
            parts.append(_cls_part(name, lo, hi))
        return parts
    roll = rng.random()
    if roll < 0.45:
        delim = rng.choice(TRAIN_DELIMS)
        name, lo, hi = _vary_class(rng)
        rep_lo = rng.randint(1, 2)
        rep_hi = rng.randint(rep_lo, min(MAX_REP, rep_lo + 2))
        parts = []
        if rng.random() < 0.7:
            parts.append(_lit_part(_prefix(rng, 1, 3)))
        parts.append(_cls_part(name, lo, hi))
        parts.append(_rep_part(name, lo, hi, delim, rep_lo, rep_hi))
        return parts
    if roll < 0.75:
        name, lo, hi = _vary_class(rng)
        tail = rng.choice((".pdf", ".txt", ".log"))
        return [_cls_part(name, max(lo, 2), max(hi, lo)), _lit_part(tail, optional=1)]
    options = []
    length = rng.randint(2, ALT_LEN)
    alphabet = string.ascii_uppercase if rng.random() < 0.5 else string.ascii_lowercase
    while len(options) < rng.randint(2, 4):
        token = "".join(rng.choice(alphabet) for _ in range(length))
        if token not in options:
            options.append(token)
    return [_alt_part(options)]


def _outside_char(name: str) -> str:
    if name in ("digit", "hex", "hex_upper"):
        return "Z"
    if name in ("upper", "upper_alnum"):
        return "a"
    if name in ("lower", "lower_alnum", "lower_word"):
        return "1"
    return "!"


def _break_positive(parts: list[Part], compiled: re.Pattern[str], rng: random.Random) -> tuple[str, str] | None:
    """Change one literal, class, or repeat so the gold regex rejects the string."""
    for _ in range(10):
        pieces: list[str] = []
        metas: list[Part] = []
        for part in parts:
            if part.kind == KIND_END:
                break
            if part.optional and rng.random() < 0.5:
                continue
            text, _flags = _piece(part, rng)
            if text:
                pieces.append(text)
                metas.append(part)
        if not pieces:
            continue
        index = rng.randrange(len(pieces))
        part = metas[index]
        text = pieces[index]
        kind = "edit"
        if part.kind == KIND_LIT and text:
            chars = list(text)
            slot = rng.randrange(len(chars))
            chars[slot] = "Q" if chars[slot] != "Q" else "7"
            pieces[index] = "".join(chars)
            kind = "edit"
        elif part.kind == KIND_CLS:
            name = CHARSETS[part.charset]
            if rng.random() < 0.5:
                width = part.hi + 1 if part.hi < MAX_HI else max(0, part.lo - 1)
                pool = CHARSET_POOL[name]
                pieces[index] = "".join(rng.choice(pool) for _ in range(width))
                kind = "repeat"
            else:
                chars = list(text) or ["0"]
                chars[rng.randrange(len(chars))] = _outside_char(name)
                pieces[index] = "".join(chars)
                kind = "class"
        elif part.kind == KIND_REP:
            name = CHARSETS[part.charset]
            if rng.random() < 0.5:
                pieces[index] = ""
            else:
                extra, _flags = _piece(part, rng)
                pieces[index] = text + extra
            kind = "repeat"
        else:
            chars = list(text)
            if chars:
                chars[0] = "Q" if chars[0] != "Q" else "7"
                pieces[index] = "".join(chars)
        cand = "".join(pieces)
        if _clean(cand, False) and compiled.fullmatch(cand) is None:
            return cand, kind
    return None


def _curriculum_language(rng: random.Random, level: int, n_support: int):
    for _ in range(30):
        parts = _draw_parts(rng, level)
        pattern = compile_parts(parts)
        if pattern is None:
            continue
        try:
            compiled = re.compile(pattern)
        except re.error:
            continue
        support: list[str] = []
        for _ in range(n_support * 6):
            if len(support) >= n_support:
                break
            text, _flags = sample_annotated(parts, rng)
            if _clean(text, False) and compiled.fullmatch(text):
                support.append(text)
        if len(support) < n_support:
            continue
        if not all(compiled.fullmatch(text) for text in support):
            continue
        optional = [part.text() for part in parts if part.optional and part.kind == KIND_LIT and part.text()]
        if optional and any(sum(token in text for text in support) in (0, len(support)) for token in optional):
            continue
        return parts, support, pattern, compiled
    return None


def sample_curriculum_pair(
    rng: random.Random,
    level: int,
    n_support: int,
    structural_rate: float,
    easy_rate: float,
    split: str,
) -> tuple[Episode, Episode] | None:
    if level >= 3:
        return sample_pair(rng, False, n_support, structural_rate, True, split, easy_rate)
    drawn = _curriculum_language(rng, level, n_support)
    if drawn is None:
        return None
    parts, support, pattern, compiled = drawn
    if rng.random() < 0.25:
        pos = Episode(support, rng.choice(support), 1, pattern, split, "positive", [])
    else:
        positive = _positive(parts, compiled, rng, False)
        if positive is None:
            return None
        query, flags = positive
        pos = Episode(support, query, 1, pattern, split, "positive", flags)
    broken = _break_positive(parts, compiled, rng)
    if broken is not None and rng.random() < 0.75:
        query, kind = broken
        neg = Episode(support, query, 0, pattern, split, kind)
    else:
        neg = _negative_episode(
            parts,
            support,
            pattern,
            compiled,
            rng,
            False,
            structural_rate if level >= 1 else 0.0,
            split,
            easy_rate,
        )
    if neg is None:
        return None
    return pos, neg


def sample_batch(
    rng: random.Random,
    batch: int,
    shift: bool = False,
    structural_rate: float = 0.3,
    prefer_hard: bool = False,
    easy_rate: float = 0.35,
    level: int = 3,
) -> list[Episode]:
    rows: list[Episode] = []
    guard = 0
    split = "train" if not shift else "ood"
    while len(rows) < batch and guard < batch * 40:
        guard += 1
        n_support = rng.randint(2, MAX_SUPPORT)
        if level >= 3:
            pair = sample_pair(
                rng,
                shift,
                n_support,
                structural_rate,
                prefer_hard or rng.random() < 0.55,
                split,
                easy_rate,
            )
        else:
            pair = sample_curriculum_pair(rng, level, n_support, structural_rate, easy_rate, split)
        if pair is None:
            continue
        rows.extend(pair)
    if len(rows) < batch:
        raise RuntimeError("could not fill a membership batch")
    return rows[:batch]


def build_curriculum_split(rng: random.Random, n_languages: int, level: int, n_support: int = 4) -> list[Episode]:
    """Fresh positives and near misses from one curriculum level. Train alphabet only."""
    episodes: list[Episode] = []
    guard = 0
    split = f"level{level}"
    while len(episodes) < n_languages * 4 and guard < n_languages * 16:
        guard += 1
        drawn = _curriculum_language(rng, level, n_support)
        if drawn is None:
            continue
        parts, support, pattern, compiled = drawn
        rows: list[Episode] = []
        for _ in range(6):
            if sum(item.label for item in rows) >= 2:
                break
            positive = _positive(parts, compiled, rng, False)
            if positive is None:
                continue
            query, flags = positive
            rows.append(Episode(support, query, 1, pattern, split, "positive", flags))
        for _ in range(8):
            if sum(1 for item in rows if item.label == 0) >= 2:
                break
            broken = _break_positive(parts, compiled, rng)
            episode = (
                Episode(support, broken[0], 0, pattern, split, broken[1])
                if broken is not None
                else _negative_episode(parts, support, pattern, compiled, rng, False, 0.0, split, 0.0)
            )
            if episode is not None:
                rows.append(episode)
        positives = [item for item in rows if item.label == 1][:2]
        negatives = [item for item in rows if item.label == 0][:2]
        if len(positives) == 2 and len(negatives) == 2:
            episodes.extend(positives + negatives)
    return episodes


def build_split(
    rng: random.Random,
    n_languages: int,
    shift: bool,
    n_support: int,
    structural_rate: float,
    prefer_hard: bool,
    split: str,
) -> list[Episode]:
    """Two fresh positives and two near misses for each language."""
    episodes: list[Episode] = []
    guard = 0
    while len(episodes) < n_languages * 4 and guard < n_languages * 12:
        guard += 1
        drawn = _language(rng, shift, n_support, prefer_hard)
        if drawn is None:
            continue
        parts, support, pattern, compiled = drawn
        rows: list[Episode] = []
        for _ in range(6):
            if sum(item.label for item in rows) >= 2:
                break
            positive = _positive(parts, compiled, rng, shift)
            if positive is None:
                continue
            query, flags = positive
            rows.append(Episode(support, query, 1, pattern, split, "positive", flags))
        for _ in range(8):
            if sum(1 for item in rows if item.label == 0) >= 2:
                break
            episode = _negative_episode(
                parts, support, pattern, compiled, rng, shift, structural_rate, split, easy_rate=0.0
            )
            if episode is not None:
                rows.append(episode)
        positives = [item for item in rows if item.label == 1][:2]
        negatives = [item for item in rows if item.label == 0][:2]
        if len(positives) == 2 and len(negatives) == 2:
            episodes.extend(positives + negatives)
    return episodes


def _compile_decision(pattern: str, query: str) -> bool:
    try:
        return re.compile(pattern).fullmatch(query) is not None
    except re.error:
        return False


def heuristic_pattern(support: list[str]) -> str:
    try:
        return infer_regex(support)
    except Exception:
        return "^$"


def rules_pattern(support: list[str]) -> str:
    heuristic = heuristic_pattern(support)
    try:
        shown, _pool = expanded_candidates(support, k=8, pool_limit=80)
    except Exception:
        return heuristic
    best = heuristic
    try:
        best_cost = pattern_cost(heuristic, support, True)
    except Exception:
        best_cost = None
    for cand in shown:
        try:
            cost = pattern_cost(cand.pattern, support, True)
        except Exception:
            continue
        if best_cost is None or cost < best_cost - 1e-9:
            best, best_cost = cand.pattern, cost
        elif abs(cost - best_cost) <= 1e-9 and cand.pattern == heuristic:
            best = heuristic
    return best


def decisions_for(episodes: list[Episode], pattern_of) -> list[bool]:
    cache: dict[tuple[str, ...], str] = {}
    chosen: list[bool] = []
    for episode in episodes:
        key = tuple(episode.support)
        pattern = cache.get(key)
        if pattern is None:
            pattern = pattern_of(list(episode.support))
            cache[key] = pattern
        chosen.append(_compile_decision(pattern, episode.query))
    return chosen


def auroc(labels: list[int], scores: list[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average = (index + 1 + end) / 2
        for cursor in range(index, end):
            if pairs[cursor][1] == 1:
                rank_sum += average
        index = end
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def summarize(episodes: list[Episode], accept: list[bool], scores: list[float] | None = None) -> dict:
    tp = tn = fp = fn = 0
    kinds: dict[str, list[int]] = {}
    for episode, decision in zip(episodes, accept):
        correct = int((episode.label == 1 and decision) or (episode.label == 0 and not decision))
        bucket = kinds.setdefault(episode.kind, [0, 0])
        bucket[0] += 1
        bucket[1] += correct
        if episode.label == 1 and decision:
            tp += 1
        elif episode.label == 1:
            fn += 1
        elif decision:
            fp += 1
        else:
            tn += 1
    total = max(len(episodes), 1)
    report = {
        "n": len(episodes),
        "accuracy": (tp + tn) / total,
        "balanced": ((tp / (tp + fn)) if tp + fn else 0.0) + ((tn / (tn + fp)) if tn + fp else 0.0),
        "tpr": tp / (tp + fn) if tp + fn else 0.0,
        "tnr": tn / (tn + fp) if tn + fp else 0.0,
        "kinds": {name: {"n": counts[0], "accuracy": counts[1] / counts[0]} for name, counts in sorted(kinds.items())},
    }
    report["balanced"] = (report["tpr"] + report["tnr"]) / 2
    if scores is not None:
        report["auroc"] = auroc([episode.label for episode in episodes], scores)
    return report


def disagreement(labels: list[int], model_accept: list[bool], other_accept: list[bool]) -> dict:
    recovery_n = recovery_ok = preserve_n = preserve_ok = 0
    for label, model_bit, other_bit in zip(labels, model_accept, other_accept):
        other_correct = other_bit == bool(label)
        model_correct = model_bit == bool(label)
        if other_correct:
            preserve_n += 1
            preserve_ok += int(model_correct)
        else:
            recovery_n += 1
            recovery_ok += int(model_correct)
    return {
        "heuristic_errors": recovery_n,
        "recovery": recovery_ok / recovery_n if recovery_n else 0.0,
        "heuristic_correct": preserve_n,
        "preservation": preserve_ok / preserve_n if preserve_n else 0.0,
    }
