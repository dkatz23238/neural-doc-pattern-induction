"""Character sets and integer ranges beyond the three full classes.

A character set is a union of ranges and literals, such as ``[A-Z0-9]`` or
``[0-9a-f]``. An integer range is a numeric interval, such as 2024 through
2035, not an alternation the inducer has to discover by hand. Both are read
off the supports. A full digit, upper, or lower class stays a class fact, so
the exact fragment does not gain a second copy of a language it already has.
"""

from __future__ import annotations

import string

_DIGIT = string.digits
_UPPER = string.ascii_uppercase
_LOWER = string.ascii_lowercase
_FULL = {"0-9", "A-Z", "a-z"}


def charset_spec(chars: set[str]) -> str:
    """Canonical class body. A trailing hyphen is escaped."""
    ranges: list[tuple[str, str]] = []
    for alphabet in (_DIGIT, _UPPER, _LOWER):
        present = sorted(ch for ch in chars if ch in alphabet)
        if not present:
            continue
        start = prev = present[0]
        for ch in present[1:]:
            if ord(ch) == ord(prev) + 1:
                prev = ch
                continue
            ranges.append((start, prev))
            start = prev = ch
        ranges.append((start, prev))
    literals = sorted(ch for ch in chars if ch not in _DIGIT + _UPPER + _LOWER)
    parts: list[str] = []
    for start, end in ranges:
        parts.append(start if start == end else start + "-" + end)
    for ch in literals:
        if ch == "-":
            parts.append("\\-")
        elif ch == "\\":
            parts.append("\\\\")
        elif ch == "]":
            parts.append("\\]")
        else:
            parts.append(ch)
    return "".join(parts)


def spec_chars(spec: str) -> set[str]:
    """Characters named by a character-set body."""
    return set(_expand(spec))


def _expand(spec: str) -> set[str]:
    found: set[str] = set()
    index = 0
    while index < len(spec):
        if spec[index] == "\\" and index + 1 < len(spec):
            found.add(spec[index + 1])
            index += 2
            continue
        if index + 2 < len(spec) and spec[index + 1] == "-" and spec[index + 2] != "\\":
            start, end = spec[index], spec[index + 2]
            for code in range(ord(start), ord(end) + 1):
                found.add(chr(code))
            index += 3
            continue
        found.add(spec[index])
        index += 1
    return found


def char_in_spec(char: str, spec: str) -> bool:
    return char in _expand(spec)


def spec_matches(spec: str, text: str) -> bool:
    return bool(text) and all(char_in_spec(char, spec) for char in text)


def integer_matches(spec: str, text: str, lo: int, hi: int) -> bool:
    """A digit string whose value lies in ``minimum:maximum``.

    A fixed width keeps leading zeros. A variable width does not, except for
    zero itself.
    """
    if not text.isdigit() or not lo <= len(text) <= hi:
        return False
    if lo != hi and len(text) > 1 and text.startswith("0"):
        return False
    minimum, maximum = (int(part) for part in spec.split(":"))
    return minimum <= int(text) <= maximum


def _blocks(start: str, end: str) -> list[str]:
    if start == end:
        return [start]
    if len(start) == 1:
        return [f"[{start}-{end}]" if start != end else start]
    if start[0] == end[0]:
        return [start[0] + part for part in _blocks(start[1:], end[1:])]
    first = int(start[0])
    last = int(end[0])
    tail = len(start) - 1
    parts = _blocks(start, start[0] + "9" * tail)
    for digit in range(first + 1, last):
        parts.append(str(digit) + "[0-9]" * tail)
    parts.extend(_blocks(str(last) + "0" * tail, end))
    return parts


def interval_regex(minimum: int, maximum: int, width: int) -> str:
    """A regex for the zero-padded integers from ``minimum`` through ``maximum``."""
    if width < 1 or minimum > maximum:
        return ""
    parts = _blocks(f"{minimum:0{width}d}", f"{maximum:0{width}d}")
    unique = list(dict.fromkeys(parts))
    if len(unique) == 1:
        return unique[0]
    return "(?:" + "|".join(unique) + ")"


def integer_regex(spec: str, lo: int, hi: int) -> str:
    minimum, maximum = (int(part) for part in spec.split(":"))
    parts: list[str] = []
    for width in range(lo, hi + 1):
        if lo == hi:
            start, end = minimum, maximum
        else:
            start = 0 if width == 1 else 10 ** (width - 1)
            end = 10**width - 1
            start = max(start, minimum)
            end = min(end, maximum)
            if width == 1 and minimum <= 0 <= maximum:
                start = min(start, 0)
        if start > end:
            continue
        parts.append(interval_regex(start, end, width))
    unique = list(dict.fromkeys(part for part in parts if part))
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    return "(?:" + "|".join(unique) + ")"


def set_body(spec: str) -> str:
    return spec


def _full_or_hull(chars: set[str]) -> list[str]:
    """Broad class union, and the observed hull when it is tighter."""
    if not chars:
        return []
    broad: set[str] = set()
    if any(ch.isdigit() for ch in chars):
        broad.update(_DIGIT)
    if any(ch.isupper() for ch in chars):
        broad.update(_UPPER)
    if any(ch.islower() for ch in chars):
        broad.update(_LOWER)
    broad.update(ch for ch in chars if not ch.isalnum())
    hull: set[str] = {ch for ch in chars if not ch.isalnum()}
    for alphabet, pred in (
        (_DIGIT, str.isdigit),
        (_UPPER, str.isupper),
        (_LOWER, str.islower),
    ):
        present = [ch for ch in chars if pred(ch)]
        if not present:
            continue
        lo, hi = min(present), max(present)
        hull.update(ch for ch in alphabet if lo <= ch <= hi)
    specs: list[str] = []
    for group in (hull, broad):
        spec = charset_spec(group)
        if spec in _FULL or spec in specs:
            continue
        specs.append(spec)
    return specs


def v2_facts(supports: list[str]) -> list:
    """Set and integer facts entailed by the supports.

    Full classes stay with the alignment. A mixed alphabet becomes a character
    set. A digit span whose observed values do not fill that width becomes an
    integer range. The exact-fragment facts are not replaced.
    """
    from pattern_induction.chart import Hyp

    if len(supports) < 2 or any(not text for text in supports):
        return []
    facts: list[Hyp] = []
    if len({len(text) for text in supports}) == 1:
        facts.extend(_equal_length(supports))
    else:
        facts.extend(_affix(supports))
    return _dedup(facts)


def _equal_length(supports: list[str]) -> list:
    """Split on constant separators. A mixed field stays one character set.

    A column of digits inside an alphanumeric token is not its own class:
    four supports can show a digit there and a hidden example can show a
    letter. A field that is entirely digits, such as a year, stays a digit
    span so the integer hull and the full class are both available.
    """
    from pattern_induction.chart import Hyp

    width = len(supports[0])

    def separator(index: int) -> bool:
        chars = [text[index] for text in supports]
        return len(set(chars)) == 1 and not chars[0].isalnum()

    facts = []
    index = 0
    while index < width:
        if separator(index):
            end = index
            while end < width and separator(end):
                end += 1
            for slot in range(len(supports)):
                facts.append(Hyp("copy", slot, index, end, score=0.0))
            index = end
            continue
        end = index
        while end < width and not separator(end):
            end += 1
        facts.extend(_field(supports, index, end))
        index = end
    return facts


def _field(supports: list[str], start: int, end: int) -> list:
    from pattern_induction.chart import Hyp

    columns = [text[start:end] for text in supports]
    if len(set(columns)) == 1:
        return [Hyp("copy", slot, start, end, score=0.0) for slot in range(len(supports))]
    chars = {ch for column in columns for ch in column}
    if all(ch.isdigit() for ch in chars):
        return _digit_span(supports, start, end)
    # A mixed field must not grow a digit or letter class on one column.
    # A field that is itself one letter class still keeps that class.
    facts = _set_span(supports, start, end, classes=_named_class(chars) is not None)
    prefix = start
    while prefix < end and all(text[prefix] == supports[0][prefix] for text in supports):
        prefix += 1
    if prefix > start:
        for slot in range(len(supports)):
            facts.append(Hyp("copy", slot, start, prefix, score=0.0))
        if prefix < end:
            facts.extend(_set_span(supports, prefix, end, classes=False))
    return facts


def _affix(supports: list[str]) -> list:
    prefix = supports[0]
    for text in supports[1:]:
        while prefix and not text.startswith(prefix):
            prefix = prefix[:-1]
    suffix = supports[0]
    for text in supports[1:]:
        while suffix and not text.endswith(suffix):
            suffix = suffix[:-1]
    if prefix and suffix and prefix.endswith(suffix) and len(prefix) + len(suffix) > min(len(text) for text in supports):
        suffix = ""
    facts = []
    middles = []
    for slot, text in enumerate(supports):
        end = len(text) - len(suffix) if suffix else len(text)
        if end < len(prefix):
            return []
        middles.append(text[len(prefix) : end])
        if prefix:
            facts.append(_copy(slot, 0, len(prefix)))
        if suffix:
            facts.append(_copy(slot, end, len(text)))
    if not middles or any(not middle for middle in middles):
        return facts
    facts.extend(_middle(supports, middles, len(prefix)))
    return facts


def _middle(supports: list[str], middles: list[str], origin: int) -> list:
    from pattern_induction.chart import Hyp

    chars = {ch for middle in middles for ch in middle}
    if all(ch.isdigit() for ch in chars):
        widths = {len(middle) for middle in middles}
        values = [int(middle) for middle in middles]
        lo, hi = min(widths), max(widths)
        facts = []
        for slot, middle in enumerate(middles):
            facts.append(Hyp("cls", slot, origin, origin + len(middle), score=0.0, cls="digit", lo=lo, hi=hi))
        if min(values) == max(values):
            return facts
        if len(widths) == 1 and min(values) == 0 and max(values) == 10 ** len(middles[0]) - 1:
            return facts
        if len(widths) == 1 and len(middles[0]) == 1:
            spec = charset_spec(set(middles))
            if spec in _FULL:
                return facts
            facts.extend(
                Hyp("set", slot, origin, origin + 1, score=0.0, cls=spec, lo=1, hi=1) for slot in range(len(middles))
            )
            return facts
        spec = f"{min(values)}:{max(values)}"
        facts.extend(
            Hyp("int", slot, origin, origin + len(middle), score=0.0, cls=spec, lo=lo, hi=hi)
            for slot, middle in enumerate(middles)
        )
        return facts
    specs = _full_or_hull(chars)
    lo, hi = min(len(middle) for middle in middles), max(len(middle) for middle in middles)
    facts = []
    name = _named_class(chars)
    if name is not None:
        for slot, middle in enumerate(middles):
            facts.append(Hyp("cls", slot, origin, origin + len(middle), score=0.0, cls=name, lo=lo, hi=hi))
    if not specs:
        return facts
    for spec in specs:
        for slot, middle in enumerate(middles):
            start = origin
            facts.append(Hyp("set", slot, start, start + len(middle), score=0.0, cls=spec, lo=lo, hi=hi))
    return facts


def _copy(slot: int, start: int, end: int):
    from pattern_induction.chart import Hyp

    return Hyp("copy", slot, start, end, score=0.0)


def _named_class(chars: set[str]) -> str | None:
    if chars and all(ch.isdigit() for ch in chars):
        return "digit"
    if chars and all(ch.isalpha() and ch.isupper() for ch in chars):
        return "upper"
    if chars and all(ch.isalpha() and ch.islower() for ch in chars):
        return "lower"
    return None


def _class_facts(slots, start: int, end: int, name: str, lo: int, hi: int) -> list:
    from pattern_induction.chart import Hyp

    return [Hyp("cls", slot, start, end, score=0.0, cls=name, lo=lo, hi=hi) for slot in slots]


def _digit_span(supports: list[str], start: int, end: int) -> list:
    from pattern_induction.chart import Hyp

    width = end - start
    if width < 1:
        return []
    columns = [item[start:end] for item in supports]
    slots = range(len(supports))
    if any(len(column) != width or not column.isdigit() for column in columns):
        return []
    facts = []
    if len(set(columns)) == 1:
        for index in slots:
            facts.append(Hyp("copy", index, start, end, score=0.0))
        return facts
    facts.extend(_class_facts(slots, start, end, "digit", width, width))
    if width == 1:
        digits = set(columns)
        spec = charset_spec(digits)
        if spec not in _FULL:
            for index in slots:
                facts.append(Hyp("set", index, start, end, score=0.0, cls=spec, lo=1, hi=1))
        return facts
    values = [int(column) for column in columns]
    minimum, maximum = min(values), max(values)
    if minimum == 0 and maximum == 10**width - 1:
        return facts
    spec = f"{minimum}:{maximum}"
    for index in slots:
        facts.append(Hyp("int", index, start, end, score=0.0, cls=spec, lo=width, hi=width))
    return facts


def _set_span(supports: list[str], start: int, end: int, classes: bool = True) -> list:
    from pattern_induction.chart import Hyp

    columns = [text[start:end] for text in supports]
    if any(len(column) != end - start for column in columns):
        return []
    chars = {ch for column in columns for ch in column}
    specs = _full_or_hull(chars)
    width = end - start
    facts = []
    name = _named_class(chars)
    if classes and name is not None:
        facts.extend(_class_facts(range(len(supports)), start, end, name, width, width))
    for spec in specs:
        for slot in range(len(supports)):
            facts.append(Hyp("set", slot, start, end, score=0.0, cls=spec, lo=width, hi=width))
    return facts


def _dedup(facts: list) -> list:
    seen = set()
    rows = []
    for hyp in facts:
        key = (hyp.kind, hyp.slot, hyp.start, hyp.end, hyp.cls, hyp.lo, hyp.hi)
        if key in seen:
            continue
        seen.add(key)
        rows.append(hyp)
    return rows


def _quant(lo: int, hi: int) -> str:
    if lo == hi:
        return "{" + str(lo) + "}"
    return "{" + str(lo) + "," + str(hi) + "}"
