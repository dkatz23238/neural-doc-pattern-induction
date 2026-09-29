"""Facts a support either states or does not.

Repeat, interval, optional, optional scope, and lexicon are the latent facts
of the task family. Charset, width, and affix are read off the strings, because
those are the facts a pooled vector could still be carrying: which character
classes vary, whether a field width is fixed, and whether a literal sits on
the front or the back.

The same string reading is what the explicit-fact ablation matches to a
candidate. It does not use the 160-d support vector.
"""

from __future__ import annotations

from pattern_induction.chart import class_runs
from pattern_induction.document_facts import alignment_kind
from pattern_induction.lexicon import matching_lexicons
from pattern_induction.renaming import DIGIT, LOWER, PUNCT, UPPER, structural_class

REPEAT_FAMILIES = frozenset({"repeat_width", "repeat_interval"})
INTERVAL_FAMILIES = frozenset({"interval", "prefix_interval", "repeat_interval"})
OPTIONAL_FAMILIES = frozenset({"prefix_optional", "prefix_charset_optional"})
LEXICON_FAMILIES = frozenset({"lexicon", "prefix_lexicon", "stem_lexicon", "lexicon_suffix"})
PREFIX_FAMILIES = frozenset(
    {
        "prefix_interval",
        "prefix_width",
        "prefix_charset",
        "prefix_optional",
        "prefix_charset_optional",
        "prefix_lexicon",
    }
)
SUFFIX_FAMILIES = frozenset({"suffix_width", "lexicon_suffix"})

SCOPE_NAME = ("absent", "prefix", "suffix", "internal")
AFFIX_NAME = ("none", "prefix", "suffix", "both")
CHARSET_NAME = ("digit", "upper", "lower", "mixed")
WIDTH_NAME = ("fixed", "count", "open")

SCOPE_ABSENT = 0
SCOPE_PREFIX = 1
SCOPE_SUFFIX = 2
SCOPE_INTERNAL = 3
AFFIX_NONE = 0
AFFIX_PREFIX = 1
AFFIX_SUFFIX = 2
AFFIX_BOTH = 3
CHAR_DIGIT = 0
CHAR_UPPER = 1
CHAR_LOWER = 2
CHAR_MIXED = 3
WIDTH_FIXED = 0
WIDTH_COUNT = 1
WIDTH_OPEN = 2

_ALNUM = {DIGIT, UPPER, LOWER}
_CLASS_CHAR = {DIGIT: "digit", UPPER: "upper", LOWER: "lower"}


def family_flags(family: str) -> dict[str, int]:
    """Latent bits the sampler meant, independent of which program was also perfect."""
    if family in OPTIONAL_FAMILIES:
        scope = SCOPE_PREFIX
    else:
        scope = SCOPE_ABSENT
    if family in PREFIX_FAMILIES:
        affix = AFFIX_PREFIX
    elif family in SUFFIX_FAMILIES:
        affix = AFFIX_SUFFIX
    else:
        affix = AFFIX_NONE
    return {
        "repeat": int(family in REPEAT_FAMILIES),
        "interval": int(family in INTERVAL_FAMILIES),
        "optional": int(family in OPTIONAL_FAMILIES),
        "optional_scope": scope,
        "affix": affix,
        "lexicon": int(family in LEXICON_FAMILIES),
    }


def _runs(text: str) -> list[tuple[int, int, int, str]]:
    return [
        (start, end, structural_class(text[start]), text[start:end]) for start, end in class_runs(text)
    ]


def _signature(row: list[tuple[int, int, int, str]]) -> tuple[int, ...]:
    return tuple(run[2] for run in row)


def _optional_span(supports: list[str]) -> tuple[str, str]:
    """Constant optional affix, or absent when the extra text is not one literal.

    A repeated field changes the run count and can look like an optional
    prefix. The extra piece of a real optional affix is the same text every
    time it appears. A repeated number is not.
    """
    kind = alignment_kind(supports)
    if kind not in {"optional_prefix", "optional_suffix"}:
        return "absent", ""
    rows = [_runs(text) for text in supports]
    sigs = [_signature(row) for row in rows]
    unique = list(dict.fromkeys(sigs))
    if len(unique) != 2:
        return "absent", ""
    short, long = sorted(unique, key=len)
    extra = len(long) - len(short)
    if extra < 1:
        return "absent", ""
    texts: list[str] = []
    for text, row, sig in zip(supports, rows, sigs):
        if sig != long:
            continue
        if kind == "optional_prefix":
            texts.append(text[: row[extra - 1][1]])
        else:
            texts.append(text[row[len(short)][0] :])
    # One longer string is not an affix. A real optional prefix is missing from
    # some supports and present, as the same literal, on at least two.
    if len(texts) < 2 or len(set(texts)) != 1 or not texts[0]:
        return "absent", ""
    role = "prefix" if kind == "optional_prefix" else "suffix"
    return role, texts[0]


def _repeat_fields(text: str) -> list[tuple[int, int, int, str]] | None:
    """Equal-width fields of one class, separated by one punctuation class."""
    runs = _runs(text)
    if len(runs) < 3 or runs[0][2] == PUNCT:
        return None
    content = runs[0][2]
    fields: list[tuple[int, int, int, str]] = []
    index = 0
    while index < len(runs):
        if runs[index][2] != content:
            return None
        fields.append(runs[index])
        index += 1
        if index == len(runs):
            break
        if runs[index][2] != PUNCT:
            return None
        index += 1
        if index >= len(runs):
            return None
    if len(fields) < 2:
        return None
    width = fields[0][1] - fields[0][0]
    if any(field[1] - field[0] != width for field in fields):
        return None
    return fields


def repeat_evidence(supports: list[str]) -> bool:
    """Every support is the same delimited field, and the field count changes."""
    if len(supports) < 2:
        return False
    counts: list[int] = []
    for text in supports:
        fields = _repeat_fields(text)
        if fields is None:
            return False
        counts.append(len(fields))
    return min(counts) >= 2 and len(set(counts)) >= 2


def _constant_ends(supports: list[str]) -> tuple[int, int]:
    rows = [_runs(text) for text in supports]
    if any(len(row) != len(rows[0]) for row in rows):
        return 0, 0
    width = len(rows[0])
    if width == 0:
        return 0, 0
    prefix = 0
    for index in range(width):
        texts = [row[index][3] for row in rows]
        if len(set(texts)) != 1:
            break
        prefix += 1
    suffix = 0
    for index in range(width - 1, -1, -1):
        texts = [row[index][3] for row in rows]
        if len(set(texts)) != 1:
            break
        suffix += 1
    if prefix + suffix >= width:
        suffix = max(0, width - prefix - 1)
    return prefix, suffix


def _punct_affix(supports: list[str]) -> tuple[str, str]:
    """A literal glued on with punctuation, when the field's own runs are not aligned.

    A mixed field does not share a run signature, so the constant prefix is
    invisible to a column alignment. It is still the text before the first
    punctuation when that text is the same on every support.
    """
    rows = [_runs(text) for text in supports]
    if any(not any(run[2] == PUNCT for run in row) for row in rows):
        return "none", ""
    prefixes: list[str] = []
    suffixes: list[str] = []
    for text, row in zip(supports, rows):
        first = next(index for index, run in enumerate(row) if run[2] == PUNCT)
        last = next(index for index in range(len(row) - 1, -1, -1) if row[index][2] == PUNCT)
        prefixes.append(text[: row[first][1]])
        suffixes.append(text[row[last][0] :])
    prefix = len(set(prefixes)) == 1 and bool(prefixes[0]) and all(len(item) < len(text) for item, text in zip(prefixes, supports))
    suffix = len(set(suffixes)) == 1 and bool(suffixes[0]) and all(len(item) < len(text) for item, text in zip(suffixes, supports))
    if prefix and suffix and all(len(prefixes[0]) + len(suffixes[0]) < len(text) for text in supports):
        return "both", prefixes[0]
    if prefix:
        return "prefix", prefixes[0]
    if suffix:
        return "suffix", suffixes[0]
    return "none", ""


def affix_role(supports: list[str]) -> str:
    optional, _text = _optional_span(supports)
    if optional == "prefix":
        return "prefix"
    if optional == "suffix":
        return "suffix"
    if repeat_evidence(supports):
        return "none"
    prefix, suffix = _constant_ends(supports)
    if prefix and suffix:
        return "both"
    if prefix:
        return "prefix"
    if suffix:
        return "suffix"
    role, _span = _punct_affix(supports)
    return role


def _content_runs(supports: list[str]) -> list[list[tuple[int, int, int, str]]]:
    optional, span = _optional_span(supports)
    rows = [_runs(text) for text in supports]
    if optional == "prefix":
        kept = []
        for text, row in zip(supports, rows):
            if text.startswith(span):
                cut = len(span)
                kept.append([run for run in row if run[0] >= cut])
            else:
                kept.append(row)
        return kept
    if optional == "suffix":
        kept = []
        for text, row in zip(supports, rows):
            if text.endswith(span):
                cut = len(text) - len(span)
                kept.append([run for run in row if run[1] <= cut])
            else:
                kept.append(row)
        return kept
    if repeat_evidence(supports):
        return [_repeat_fields(text) or [] for text in supports]
    prefix, suffix = _constant_ends(supports)
    if prefix or suffix:
        width = len(rows[0])
        stop = width - suffix
        return [row[prefix:stop] for row in rows]
    role, span = _punct_affix(supports)
    if role == "prefix":
        return [[run for run in row if run[0] >= len(span)] for row in rows]
    if role == "suffix":
        return [[run for run in row if run[1] <= len(text) - len(span)] for text, row in zip(supports, rows)]
    if role == "both":
        # The prefix span is what _punct_affix returns; the suffix is the shared tail.
        _role, suffix_span = _punct_suffix(supports)
        return [
            [run for run in row if run[0] >= len(span) and run[1] <= len(text) - len(suffix_span)]
            for text, row in zip(supports, rows)
        ]
    return rows


def _punct_suffix(supports: list[str]) -> tuple[str, str]:
    role, _prefix = _punct_affix(supports)
    if role != "both":
        return "none", ""
    rows = [_runs(text) for text in supports]
    suffixes = []
    for text, row in zip(supports, rows):
        last = next(index for index in range(len(row) - 1, -1, -1) if row[index][2] == PUNCT)
        suffixes.append(text[row[last][0] :])
    return "suffix", suffixes[0] if suffixes else ""


def content_charset(supports: list[str]) -> str:
    """Class of the characters that are not a shared literal affix."""
    found: set[int] = set()
    rows = _content_runs(supports)
    aligned = bool(rows) and all(len(row) == len(rows[0]) and len(row) > 0 for row in rows)
    if aligned:
        for index in range(len(rows[0])):
            texts = [row[index][3] for row in rows]
            if len(set(texts)) == 1:
                continue
            for row in rows:
                if row[index][2] in _ALNUM:
                    found.add(row[index][2])
    if not found:
        for row in rows:
            for run in row:
                if run[2] in _ALNUM:
                    found.add(run[2])
    if not found:
        for text in supports:
            for ch in text:
                kind = structural_class(ch)
                if kind in _ALNUM:
                    found.add(kind)
    if len(found) == 1:
        return _CLASS_CHAR[next(iter(found))]
    if len(found) > 1:
        return "mixed"
    return "mixed"


def _field_lengths(supports: list[str]) -> list[int]:
    """One length per support. A mixed field is one span, not one length per class run."""
    if repeat_evidence(supports):
        lengths = []
        for text in supports:
            fields = _repeat_fields(text) or []
            if fields:
                lengths.append(fields[0][1] - fields[0][0])
        return lengths
    rows = _content_runs(supports)
    lengths = []
    for row in rows:
        if row:
            lengths.append(row[-1][1] - row[0][0])
    return lengths


def width_mode(supports: list[str]) -> str:
    """fixed: one string length and one field width. count: only the string length changes. open: the field width changes."""
    lengths = _field_lengths(supports)
    if len(set(lengths)) > 1:
        return "open"
    if len({len(text) for text in supports}) > 1:
        return "count"
    return "fixed"


def content_tokens(supports: list[str]) -> list[set[str]]:
    """Columns a frozen lexicon could be evidenced by. Whole strings are one column."""
    columns = [set(supports)]
    rows = _content_runs(supports)
    if rows and all(len(row) == len(rows[0]) for row in rows):
        for index in range(len(rows[0])):
            columns.append({row[index][3] for row in rows if row})
    return columns


def lexicon_names(supports: list[str]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for column in content_tokens(supports):
        for name in matching_lexicons(column):
            if name not in seen:
                seen.add(name)
                found.append(name)
    return found


def visible_facts(supports: list[str]) -> dict[str, str | bool | tuple[str, ...]]:
    optional, _span = _optional_span(supports)
    scope = {"absent": "absent", "prefix": "prefix", "suffix": "suffix"}[optional]
    return {
        "repeat": repeat_evidence(supports),
        "optional_scope": scope,
        "charset": content_charset(supports),
        "width": width_mode(supports),
        "affix": affix_role(supports),
        "lexicon": tuple(lexicon_names(supports)),
    }


def probe_targets(family: str, supports: list[str]) -> dict[str, int]:
    """One integer label per probe. Scope suffix and internal stay in the head; this corpus does not use them."""
    flags = family_flags(family)
    charset = CHARSET_NAME.index(content_charset(supports))
    width = WIDTH_NAME.index(width_mode(supports))
    return {
        "repeat": flags["repeat"],
        "interval": flags["interval"],
        "optional": flags["optional"],
        "optional_scope": flags["optional_scope"],
        "charset": charset,
        "width": width,
        "affix": flags["affix"],
        "lexicon": flags["lexicon"],
    }


PROBE_NAMES = (
    "repeat",
    "interval",
    "optional",
    "optional_scope",
    "charset",
    "width",
    "affix",
    "lexicon",
)
PROBE_CLASSES = {
    "repeat": ("absent", "present"),
    "interval": ("absent", "present"),
    "optional": ("absent", "present"),
    "optional_scope": SCOPE_NAME,
    "charset": CHARSET_NAME,
    "width": WIDTH_NAME,
    "affix": ("none", "prefix", "suffix"),
    "lexicon": ("absent", "present"),
}
