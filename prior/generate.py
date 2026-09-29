"""Compositional ambiguity tasks. The frozen expander still proposes candidates.

Latent languages are products of factors (class, width, affix, optionality,
repeat, interval, lexicon), not a hand-listed menu. Three combinations are
held out of training:

- optional prefix + character class
- repeat + numeric interval
- lexicon + literal suffix

A task is kept only when the sampled program is in the expander's pool and a
probe accepts that program while rejecting the next wider one.
"""

from __future__ import annotations

import json
import math
import random
import string
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from pattern_induction.charset import charset_spec
from pattern_induction.latent import expand_programs
from pattern_induction.lexicon import LEXICONS
from pattern_induction.program import Prog, _MEMBERSHIP, describe, execute
from prior.dataset import MAX_CANDIDATES
from prior.features import FEATURE_DIM, candidate_features
from prior.tree import programs_json
from prior.support_encode import SUPPORT_DIM, embed_supports, load_frozen_encoder

SEED = 2507
_LEX = {item.name: sorted(item.members) for item in LEXICONS}

# Train may use each of these factors. It must not use the held-out products.
TRAIN_FAMILIES = (
    "interval",
    "interval",
    "prefix_interval",
    "prefix_interval",
    "width",
    "prefix_width",
    "charset_width",
    "prefix_charset",
    "prefix_optional",
    "suffix_width",
    "repeat_width",
    "lexicon",
    "prefix_lexicon",
    "stem_lexicon",
)
OOD_FAMILIES = (
    "prefix_charset_optional",
    "repeat_interval",
    "lexicon_suffix",
)
_HOLDOUT = frozenset(OOD_FAMILIES)
_SEPARATORS = ("-", "/", ":")
_CHARSETS = (
    ("digit", "0123456789", "A", "cls"),
    ("upper", string.ascii_uppercase, "0", "cls"),
    ("lower", string.ascii_lowercase, "0", "cls"),
    ("hex_upper", "0123456789ABCDEF", "G", "set"),
    ("hex_lower", "0123456789abcdef", "g", "set"),
    ("alnum_upper", string.digits + string.ascii_uppercase, "a", "set"),
    ("alnum_lower", string.digits + string.ascii_lowercase, "A", "set"),
)


def _perfect(prog: Prog, supports: list[str], positives: list[str], negatives: list[str]) -> bool:
    return all(execute(prog, text, supports) for text in positives) and all(
        not execute(prog, text, supports) for text in negatives
    )


def _copy(start: int, end: int, slot: int = 0) -> Prog:
    return Prog("COPY", slot=slot, start=start, end=end)


def _seq(*children: Prog) -> Prog:
    return Prog("SEQ", children=list(children))


def _lit(rng: random.Random, n: int, alphabet: str) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


def _plain_int(rng: random.Random, lo: int, hi: int, width: int) -> str | None:
    floor = lo if width == 1 else max(lo, 10 ** (width - 1))
    if hi < floor:
        return None
    return f"{rng.randint(floor, hi)}"


def _snap_window(rng: random.Random) -> tuple[int, int, int] | None:
    """A width≥3 grid cell. The expander adds this snap before any convention."""
    width = rng.choice((3, 4, 5))
    step = 10 ** (width - 2)
    floor = 10 ** (width - 1)
    start = ((floor + step - 1) // step) * step
    last = 10**width - step
    if start > last:
        return None
    lo = rng.randrange(start, last + 1, step)
    return lo, lo + step - 1, width


def _window(rng: random.Random) -> tuple[int, int, int] | None:
    """A latent numeric window. Wider than a 4-example hull, and numerous.

    Decade cells such as 50–59 are not in the expander's first eight intervals,
    so they are not used as gold. Width-1/2 conventions and width≥3 snaps are.
    """
    if rng.randrange(3) == 0:
        lo, hi = rng.choice(
            ((0, 9), (1, 9), (0, 12), (1, 12), (0, 23), (0, 31), (1, 31), (0, 59), (0, 99), (1, 99))
        )
        width = 1 if hi <= 9 else 2
        return lo, hi, width
    return _snap_window(rng)


def _hull(rng: random.Random, lo: int, hi: int, width: int) -> list[str] | None:
    floor = lo if width == 1 else max(lo, 10 ** (width - 1))
    if hi - floor < 4:
        return None
    band = max(2, (hi - floor) // rng.choice((3, 5, 8)))
    for _ in range(20):
        hlo = rng.randint(floor, hi - 2)
        hhi = rng.randint(hlo + 1, min(hi - 1, hlo + band))
        if hlo <= lo and hhi >= hi:
            continue
        shown = [hlo, hhi, rng.randint(hlo, hhi), rng.randint(hlo, hhi)]
        texts = []
        ok = True
        for value in shown:
            text = f"{value:0{width}d}"
            if len(text) != width or (width > 1 and text.startswith("0")):
                ok = False
                break
            texts.append(text)
        if ok and hlo > lo or hhi < hi:
            if ok:
                return texts
    return None


def _interval_body(rng: random.Random) -> tuple[list[str], list[str], list[str], Prog] | None:
    window = _window(rng)
    if window is None:
        return None
    lo, hi, width = window
    supports = _hull(rng, lo, hi, width)
    if supports is None:
        return None
    outside = [value for value in range(lo, hi + 1) if f"{value:0{width}d}" not in supports]
    if not outside:
        return None
    positives = [f"{lo:0{width}d}", f"{hi:0{width}d}", f"{rng.choice(outside):0{width}d}"]
    negatives = [f"{hi + 1:0{width}d}", "A" * width]
    if lo > 0:
        negatives.append(f"{lo - 1:0{width}d}")
    ref = Prog("INT", spec=f"{lo}:{hi}", lo=width, hi=width)
    if not _perfect(ref, supports, positives, negatives):
        return None
    return supports, positives, negatives, ref


def _interval_example(rng: random.Random):
    return _interval_body(rng)


def _prefix_interval(rng: random.Random):
    body = _interval_body(rng)
    if body is None:
        return None
    numbers, positives, negatives, number = body
    plen = rng.choice((1, 2, 3, 4))
    prefix = _lit(rng, plen, string.ascii_uppercase)
    supports = [prefix + text for text in numbers]
    positives = [prefix + text for text in positives]
    negatives = [prefix + text for text in negatives] + [numbers[0], _lit(rng, plen, string.ascii_uppercase) + numbers[0]]
    ref = _seq(_copy(0, plen), number)
    if _perfect(ref, supports, positives, negatives):
        return supports, positives, negatives, ref
    return None


def _width_spec(rng: random.Random) -> tuple[int, int, int]:
    hi = rng.randint(3, 12)
    # Open downward, or keep the observed width. Both are latent width choices.
    lo = 1 if rng.randrange(2) == 0 else hi
    return lo, hi, hi


def _digit_strings(rng: random.Random, lo: int, hi: int, n: int = 4) -> list[str]:
    lengths = [hi, hi, rng.randint(max(lo, 1), hi), rng.randint(max(lo, 1), hi)]
    if lo == 1 and hi > 1:
        lengths[2] = rng.choice((1, 2, hi))
    texts = []
    for length in lengths[:n]:
        alphabet = "123456789" if length == 1 else "123456789"
        texts.append(_lit(rng, length, alphabet))
    return texts


def _width_body(rng: random.Random):
    lo, hi, _obs = _width_spec(rng)
    supports = _digit_strings(rng, max(lo, 1) if lo == hi else 1, hi)
    # Observed strings stay inside the latent width and do not show every length.
    supports = [text for text in supports if lo <= len(text) <= hi]
    if len(supports) < 4:
        supports = (_digit_strings(rng, hi, hi) + supports)[:4]
    if len({len(text) for text in supports}) == 1 and lo == 1:
        supports[0] = _lit(rng, hi, "123456789")
    positives = ["7" if lo == 1 else _lit(rng, lo, "123456789"), _lit(rng, hi, "123456789")]
    negatives = ["9" * (hi + 1), "a" * hi]
    if lo == hi and hi > 1:
        negatives.append("9" * (hi - 1))
    ref = Prog("CLS", cls="digit", lo=lo, hi=hi)
    if _perfect(ref, supports, positives, negatives):
        return supports, positives, negatives, ref
    return None


def _affix(body, prefix: str, suffix: str):
    supports, positives, negatives, _number = body
    return (
        [prefix + text + suffix for text in supports],
        [prefix + text + suffix for text in positives],
        [prefix + text + suffix for text in negatives],
    )


def _prefix_width(rng: random.Random):
    body = _width_body(rng)
    if body is None:
        return None
    numbers, positives, negatives, number = body
    plen = rng.choice((1, 2, 3, 4))
    prefix = _lit(rng, plen, string.ascii_uppercase)
    supports = [prefix + text for text in numbers]
    positives = [prefix + text for text in positives]
    negatives = [prefix + text for text in negatives] + [numbers[0]]
    ref = _seq(_copy(0, plen), number)
    if _perfect(ref, supports, positives, negatives):
        return supports, positives, negatives, ref
    return None


def _suffix_width(rng: random.Random):
    body = _width_body(rng)
    if body is None:
        return None
    numbers, positives, negatives, number = body
    slen = rng.choice((1, 2, 3))
    suffix = _lit(rng, slen, string.ascii_uppercase)
    supports = [text + suffix for text in numbers]
    positives = [text + suffix for text in positives]
    negatives = [text + suffix for text in negatives] + [numbers[0]]
    start = len(supports[0]) - slen
    ref = _seq(number, _copy(start, start + slen))
    if _perfect(ref, supports, positives, negatives):
        return supports, positives, negatives, ref
    return None


def _charset_field(rng: random.Random, width: int, kind: str, alphabet: str) -> tuple[list[str], str]:
    dropped = rng.choice(alphabet)
    observed = alphabet.replace(dropped, "") or alphabet
    supports = [_lit(rng, width, observed) for _ in range(4)]
    return supports, dropped


def _charset_width(rng: random.Random):
    name, alphabet, outside, kind = _CHARSETS[rng.randrange(len(_CHARSETS))]
    width = rng.randint(2, 8)
    supports, dropped = _charset_field(rng, width, kind, alphabet)
    positives = [dropped + _lit(rng, width - 1, alphabet)]
    negatives = [outside * width, alphabet[0] * (width + 1)]
    if width > 1:
        negatives.append(alphabet[0] * (width - 1))
    if kind == "cls":
        ref = Prog("CLS", cls=name if name in {"digit", "upper", "lower"} else "digit", lo=width, hi=width)
    else:
        ref = Prog("SET", spec=charset_spec(set(alphabet)), lo=width, hi=width)
    if _perfect(ref, supports, positives, negatives):
        return supports, positives, negatives, ref
    return None


def _prefix_charset(rng: random.Random):
    drawn = _charset_width(rng)
    if drawn is None:
        return None
    bodies, positives, negatives, field = drawn
    plen = rng.choice((1, 2, 3))
    prefix = _lit(rng, plen, string.ascii_uppercase)
    sep = rng.choice(_SEPARATORS)
    head = prefix + sep
    supports = [head + text for text in bodies]
    positives = [head + text for text in positives]
    negatives = [head + text for text in negatives] + [bodies[0], _lit(rng, plen, string.ascii_uppercase) + sep + bodies[0]]
    ref = _seq(_copy(0, plen), _copy(plen, plen + 1), field)
    if _perfect(ref, supports, positives, negatives):
        return supports, positives, negatives, ref
    return None


def _prefix_optional(rng: random.Random):
    width = rng.randint(2, 8)
    plen = rng.choice((1, 2, 3, 4))
    prefix = _lit(rng, plen, string.ascii_uppercase)
    sep = rng.choice(_SEPARATORS)
    bare = _digit_strings(rng, width, width)
    supports = [prefix + sep + bare[0], bare[1], prefix + sep + bare[2], bare[3]]
    positives = [_lit(rng, 1, "123456789"), prefix + sep + _lit(rng, width, "123456789")]
    negatives = ["9" * (width + 1), prefix + sep + ("9" * (width + 1)), prefix + _lit(rng, width, "123456789")]
    ref = _seq(
        Prog("OPT", children=[_seq(_copy(0, plen), _copy(plen, plen + 1))]),
        Prog("CLS", cls="digit", lo=1, hi=width),
    )
    if _perfect(ref, supports, positives, negatives):
        return supports, positives, negatives, ref
    return None


def _prefix_charset_optional(rng: random.Random):
    """Held out: an optional prefix in front of a letter class completed to a set."""
    width = rng.choice((2, 3, 4, 5, 6))
    plen = rng.choice((1, 2, 3, 4))
    prefix = _lit(rng, plen, string.ascii_uppercase)
    sep = "-"
    field = lambda: _lit(rng, width, string.ascii_uppercase)  # noqa: E731
    supports = [prefix + sep + field(), field(), prefix + sep + field(), field()]
    mixed = _lit(rng, width - 1, string.ascii_uppercase) + rng.choice(string.digits)
    positives = [mixed, prefix + sep + mixed]
    negatives = [mixed.lower(), prefix + sep + ("A" * (width + 1)), "a" * width, prefix + mixed]
    ref = _seq(
        Prog("OPT", children=[_seq(_copy(0, plen), _copy(plen, plen + 1))]),
        Prog("SET", spec=charset_spec(set(string.digits + string.ascii_uppercase)), lo=width, hi=width),
    )
    if _perfect(ref, supports, positives, negatives):
        return supports, positives, negatives, ref
    return None


def _max_fields(width: int, sep_len: int) -> int:
    """Largest field count whose joined string still fits in 32 characters."""
    if width < 1 or sep_len < 1:
        return 2
    ceiling = (32 + sep_len) // (width + sep_len)
    return max(2, min(9, ceiling))


def _repeat_counts(rng: random.Random, width: int, sep_len: int) -> list[int]:
    ceiling = _max_fields(width, sep_len)
    hi = rng.randint(3, ceiling) if ceiling >= 3 else 2
    lo = 2 if hi <= 3 else rng.choice((2, 3))
    counts = [lo, hi, rng.randint(lo, hi), rng.randint(lo, hi)]
    if len(set(counts)) < 2:
        counts[-1] = hi if counts[0] != hi else lo
    return counts


def _separator(rng: random.Random) -> str:
    return rng.choice(_SEPARATORS + ("--", "::"))


def _zero_field(text: str) -> str:
    return "0" + text[1:] if len(text) > 1 else text


def _within(texts: list[str]) -> bool:
    return all(0 < len(text) <= 32 for text in texts)


def _repeat_width(rng: random.Random):
    """Exact-width digit fields, repeated.

    A leading zero in the head column and in a repeated field makes the frozen
    digit node emit ``digit{W,W}`` as the seed. Width 1 never crosses that
    class inside the expander's cap, so it is not sampled.
    """
    width = rng.choice((2, 3, 4, 5, 6))
    sep = _separator(rng)
    counts = _repeat_counts(rng, width, len(sep))
    if max(counts) < 3 or len(set(counts)) < 2:
        return None
    rows = []
    for count in counts:
        parts = [_lit(rng, width, "123456789") for _ in range(count)]
        rows.append(sep.join(parts))
    rows[0] = _zero_field(rows[0][:width]) + rows[0][width:]
    pieces = rows[1].split(sep)
    pieces[1] = _zero_field(pieces[1])
    rows[1] = sep.join(pieces)
    supports = rows
    pair_lo = min(count - 1 for count in counts)
    pair_hi = max(count - 1 for count in counts)
    if pair_hi > 8 or pair_lo < 1:
        return None
    field = Prog("CLS", cls="digit", lo=width, hi=width)
    ref = _seq(field, Prog("REPEAT", lo=pair_lo, hi=pair_hi, children=[_seq(_copy(width, width + len(sep)), field)]))
    positive = sep.join([_zero_field(_lit(rng, width, "123456789")) for _ in range(pair_lo + 1)])
    wide = sep.join(["9" * (width + 1), "9" * (width + 1)])
    letter = sep.join(["a" * width for _ in range(pair_lo + 1)])
    negatives = [wide, letter]
    if pair_lo > 1:
        # One pair short of the observed minimum. The open quantifier still accepts it.
        negatives.append(sep.join(["1" * width for _ in range(pair_lo)]))
    extra_fields = pair_hi + 2
    if extra_fields - 1 <= 8:
        extra = sep.join(["1" * width for _ in range(extra_fields)])
        if len(extra) <= 32:
            negatives.append(extra)
    if not _within(supports + [positive] + negatives):
        return None
    if _perfect(ref, supports, [positive], negatives):
        return supports, [positive], negatives, ref
    return None


def _same_width(value: int, width: int) -> str | None:
    text = str(value)
    if len(text) != width or (width > 1 and text.startswith("0")):
        return None
    return text


def _repeat_interval(rng: random.Random):
    """Held out: the same repeated shape, with a snapped integer window."""
    window = _snap_window(rng)
    if window is None:
        return None
    lo, hi, width = window
    if hi - lo < 8:
        return None
    sep = _separator(rng)
    counts = _repeat_counts(rng, width, len(sep))
    if max(counts) < 3 or len(set(counts)) < 2:
        return None
    rows = []
    for count in counts:
        parts = []
        for _ in range(count):
            text = _plain_int(rng, lo + 1, hi - 1, width)
            if text is None:
                return None
            parts.append(text)
        rows.append(sep.join(parts))
    supports = rows
    pair_lo = min(count - 1 for count in counts)
    pair_hi = max(count - 1 for count in counts)
    if pair_hi > 8 or pair_lo < 1:
        return None
    field = Prog("INT", spec=f"{lo}:{hi}", lo=width, hi=width)
    ref = _seq(
        field,
        Prog("REPEAT", lo=pair_lo, hi=pair_hi, children=[_seq(_copy(width, width + len(sep)), field)]),
    )
    endpoint = _same_width(lo, width)
    other = _same_width(hi, width)
    outside = _same_width(hi + 1, width) or _same_width(lo - 1, width)
    if endpoint is None or other is None or outside is None:
        return None
    positive = sep.join([endpoint, other] + [endpoint] * max(0, pair_lo - 1))
    negatives = [sep.join([outside, outside]), sep.join(["A" * width, "A" * width])]
    if pair_lo > 1:
        negatives.append(sep.join([endpoint for _ in range(pair_lo)]))
    if not _within(supports + [positive] + negatives):
        return None
    if _perfect(ref, supports, [positive], negatives):
        return supports, [positive], negatives, ref
    return None


def _lexicon_draw(rng: random.Random, names: tuple[str, ...] | None = None):
    pool = names or tuple(_LEX)
    name = pool[rng.randrange(len(pool))]
    members = [token for token in _LEX[name] if token.isalnum() and 1 <= len(token) <= 8]
    take = 4 if len(members) >= 8 else 3
    if len(members) < take + 2:
        return None
    chosen = rng.sample(members, take)
    held = [token for token in members if token not in chosen]
    other = rng.choice(held)
    member_set = set(_LEX[name])
    negatives = []
    for text in (other.lower(), other.upper(), other + "x", "999", "000"):
        if text and text not in member_set and text not in negatives:
            negatives.append(text)
    if len(negatives) < 2:
        return None
    return name, chosen, [other, rng.choice(held)], negatives[:3]


def _lexicon(rng: random.Random):
    drawn = _lexicon_draw(rng)
    if drawn is None:
        return None
    name, chosen, positives, negatives = drawn
    ref = Prog("LEX", spec=name)
    if _perfect(ref, chosen, positives, negatives):
        return chosen, positives, negatives, ref
    return None


def _prefix_lexicon(rng: random.Random):
    drawn = _lexicon_draw(rng)
    if drawn is None:
        return None
    name, chosen, positives, negatives = drawn
    plen = rng.choice((1, 2, 3))
    prefix = _lit(rng, plen, string.ascii_lowercase)
    sep = "-"
    head = prefix + sep
    supports = [head + token for token in chosen]
    positives = [head + token for token in positives]
    negatives = [head + token for token in negatives] + [chosen[0]]
    ref = _seq(_copy(0, plen), _copy(plen, plen + 1), Prog("LEX", spec=name))
    if _perfect(ref, supports, positives, negatives):
        return supports, positives, negatives, ref
    return None


def _stem_lexicon(rng: random.Random):
    drawn = _lexicon_draw(rng, ("file_extension",))
    if drawn is None:
        return None
    name, chosen, positives, negatives = drawn
    length = rng.choice((2, 3, 4, 5, 6))
    stems = [_lit(rng, length, string.ascii_lowercase) for _ in range(len(chosen))]
    supports = [stem + "." + token for stem, token in zip(stems, chosen)]
    pos_stem = _lit(rng, length, string.ascii_lowercase)
    positives = [pos_stem + "." + token for token in positives]
    negatives = [pos_stem + "." + token for token in negatives] + [chosen[0], pos_stem + chosen[0]]
    dot = supports[0].index(".")
    ref = _seq(
        Prog("CLS", cls="lower", lo=length, hi=length),
        _copy(dot, dot + 1),
        Prog("LEX", spec=name),
    )
    if _perfect(ref, supports, positives, negatives):
        return supports, positives, negatives, ref
    return None


def _lexicon_suffix(rng: random.Random):
    drawn = _lexicon_draw(rng, ("iso4217", "month_abbr", "month_abbr_upper", "apparel_size"))
    if drawn is None:
        return None
    name, chosen, positives, negatives = drawn
    slen = rng.choice((1, 2, 3))
    suffix = _lit(rng, slen, string.ascii_lowercase)
    sep = rng.choice(_SEPARATORS)
    tail = sep + suffix
    supports = [token + tail for token in chosen]
    positives = [token + tail for token in positives]
    negatives = [token + tail for token in negatives] + [chosen[0], chosen[0] + sep]
    at = len(chosen[0])
    ref = _seq(Prog("LEX", spec=name), _copy(at, at + 1), _copy(at + 1, at + 1 + slen))
    if _perfect(ref, supports, positives, negatives):
        return supports, positives, negatives, ref
    return None


_BUILDERS = {
    "interval": _interval_example,
    "prefix_interval": _prefix_interval,
    "width": _width_body,
    "prefix_width": _prefix_width,
    "charset_width": _charset_width,
    "prefix_charset": _prefix_charset,
    "prefix_optional": _prefix_optional,
    "suffix_width": _suffix_width,
    "repeat_width": _repeat_width,
    "lexicon": _lexicon,
    "prefix_lexicon": _prefix_lexicon,
    "stem_lexicon": _stem_lexicon,
    "prefix_charset_optional": _prefix_charset_optional,
    "repeat_interval": _repeat_interval,
    "lexicon_suffix": _lexicon_suffix,
}


def build_task(rng: random.Random, family: str, cap: int | None = None) -> dict | None:
    """One labeled task, or None when the frozen pool misses the latent language."""
    if cap is None:
        # A repeated interval is a joint snap of two integer fields. That cross
        # needs a larger share than a single-field rewrite.
        cap = 128 if family == "repeat_interval" else 64
    _MEMBERSHIP.clear()
    drawn = None
    for _ in range(6):
        drawn = _BUILDERS[family](rng)
        if drawn is not None:
            break
    if drawn is None:
        return None
    supports, positives, negatives, ref = drawn
    programs = expand_programs([], supports, cap=cap)
    perfect = [prog for prog in programs if _perfect(prog, supports, positives, negatives)]
    if not perfect:
        return None
    gold_keys = {describe(prog) for prog in perfect}
    signature = describe(ref)
    if signature not in gold_keys:
        return None
    rest = [prog for prog in programs if describe(prog) not in gold_keys]
    rng.shuffle(perfect)
    rng.shuffle(rest)
    chosen = (perfect[:4] + rest)[:MAX_CANDIDATES]
    rng.shuffle(chosen)
    feats = [0.0] * (MAX_CANDIDATES * FEATURE_DIM)
    mask = [0] * MAX_CANDIDATES
    gold_row = [0] * MAX_CANDIDATES
    for index, prog in enumerate(chosen):
        row = candidate_features(prog, supports)
        start = index * FEATURE_DIM
        feats[start : start + FEATURE_DIM] = row
        mask[index] = 1
        gold_row[index] = int(describe(prog) in gold_keys)
    if sum(gold_row) == 0:
        return None
    return {
        "family": family,
        "signature": signature,
        "supports": supports,
        "candidate": feats,
        "programs": programs_json(chosen),
        "mask": mask,
        "gold": gold_row,
    }


def diversity(rows: list[dict]) -> dict:
    """How many distinct latent programs the sampler actually produced."""
    signatures = Counter(row["signature"] for row in rows)
    combos = Counter(row["family"] for row in rows)
    support_keys = Counter(tuple(row["supports"]) for row in rows)
    task_keys = Counter((row["signature"], tuple(row["supports"])) for row in rows)
    total = max(len(rows), 1)
    entropy = 0.0
    if rows:
        entropy = -sum((count / total) * math.log2(count / total) for count in signatures.values())
    singletons = sum(1 for row in rows if signatures[row["signature"]] == 1)
    return {
        "tasks": len(rows),
        "unique_signatures": len(signatures),
        "unique_combinations": len(combos),
        "signature_entropy_bits": round(entropy, 3),
        "top_signatures": [{"signature": name, "count": count} for name, count in signatures.most_common(10)],
        "top_combinations": [{"combination": name, "count": count} for name, count in combos.most_common(10)],
        "exact_duplicate_fraction": round(1 - len(task_keys) / total, 6),
        "duplicate_support_fraction": round(1 - len(support_keys) / total, 6),
        "singleton_signature_fraction": round(singletons / total, 6),
        "max_signature_fraction": round((signatures.most_common(1)[0][1] / total) if signatures else 0.0, 6),
        "novel_signature_fraction": round(len(signatures) / total, 6),
    }


def _worker(payload: tuple) -> tuple[list[dict], int]:
    family_cycle, count, seed = payload
    rng = random.Random(seed)
    rows: list[dict] = []
    attempts = 0
    limit = count * 20
    while len(rows) < count and attempts < limit:
        family = family_cycle[attempts % len(family_cycle)]
        attempts += 1
        task = build_task(rng, family)
        if task is not None:
            rows.append(task)
    return rows, attempts


def _write_chunk(path: Path, rows: list[dict], embeddings: torch.Tensor) -> None:
    table = pa.table(
        {
            "family": [row["family"] for row in rows],
            "support": embeddings.tolist(),
            "candidate": [row["candidate"] for row in rows],
            "programs": [row["programs"] for row in rows],
            "supports": [row["supports"] for row in rows],
            "mask": [row["mask"] for row in rows],
            "gold": [row["gold"] for row in rows],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _collect(families: tuple[str, ...], count: int, seed: int) -> tuple[list[dict], int]:
    import multiprocessing as mp

    workers = min(4, max(1, count // 500 or 1))
    per = [count // workers] * workers
    for index in range(count % workers):
        per[index] += 1
    payloads = [(families, size, seed + index * 100_003) for index, size in enumerate(per) if size]
    with mp.Pool(len(payloads)) as pool:
        parts = pool.map(_worker, payloads)
    rows: list[dict] = []
    attempts = 0
    for group, tried in parts:
        rows.extend(group)
        attempts += tried
    return rows[:count], attempts


def generate_split(root: Path, split: str, families: tuple[str, ...], count: int, seed: int) -> dict:
    rows, attempts = _collect(families, count, seed)
    encoder = load_frozen_encoder(root)
    out = root / "results" / "prior" / "shards"
    written = 0
    for start in range(0, len(rows), 4000):
        chunk = rows[start : start + 4000]
        vectors = []
        for cursor in range(0, len(chunk), 256):
            batch = chunk[cursor : cursor + 256]
            vectors.append(embed_supports(encoder, [row["supports"] for row in batch]))
        embedding = torch.cat(vectors, dim=0) if vectors else torch.zeros(0, SUPPORT_DIM)
        _write_chunk(out / f"{split}-{start // 4000:05d}.parquet", chunk, embedding)
        written += len(chunk)
    report = diversity(rows)
    report.update(
        {
            "split": split,
            "kept": written,
            "attempts": attempts,
            "oracle": written / max(attempts, 1),
        }
    )
    return report


def pilot(n: int, seed: int) -> dict:
    rng = random.Random(seed)
    report = {}
    kept_rows: list[dict] = []
    families = tuple(dict.fromkeys(TRAIN_FAMILIES + OOD_FAMILIES))
    for family in families:
        kept = 0
        for _ in range(n):
            task = build_task(rng, family)
            if task is not None:
                kept += 1
                kept_rows.append(task)
        report[family] = {"kept": kept, "attempts": n, "oracle": kept / n}
    report["diversity"] = diversity(kept_rows)
    report["holdout_disjoint"] = _HOLDOUT.isdisjoint(set(TRAIN_FAMILIES))
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--train", type=int, default=50_000)
    parser.add_argument("--val", type=int, default=10_000)
    parser.add_argument("--ood", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.pilot:
        report = pilot(args.pilot, args.seed)
        print(json.dumps(report, indent=2))
        return
    root = Path(__file__).resolve().parents[1]
    shards = root / "results" / "prior" / "shards"
    if shards.exists():
        for path in shards.glob("*.parquet"):
            path.unlink()
    summary = {
        "seed": args.seed,
        "support_dim": SUPPORT_DIM,
        "feature_dim": FEATURE_DIM,
        "max_candidates": MAX_CANDIDATES,
        "train_families": list(dict.fromkeys(TRAIN_FAMILIES)),
        "ood_families": list(OOD_FAMILIES),
        "splits": [
            generate_split(root, "train", TRAIN_FAMILIES, args.train, args.seed),
            generate_split(root, "val", TRAIN_FAMILIES, args.val, args.seed + 1),
            generate_split(root, "ood", OOD_FAMILIES, args.ood, args.seed + 2),
        ],
    }
    dest = root / "results" / "prior" / "generation.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
