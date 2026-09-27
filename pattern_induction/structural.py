"""Structural candidate generators.

These propose regexes the local mutator never reaches: merged alphabets,
delimiter structure, repeated groups, optional spans, and alternation across
structurally different examples. Every pattern is anchored and has to be
accepted by the cost-model parser. Hidden strings are not consulted.
"""

from __future__ import annotations

import re
import string

from pattern_induction.candidates import (
    Candidate,
    Node,
    _quantify,
    _single_site,
    anchor,
    compress,
    escape_literal,
    named_class,
    parse_pattern,
    pattern_cost,
)
from pattern_induction.heuristic import infer_regex

DIGITS = set(string.digits)
LOWER = set(string.ascii_lowercase)
UPPER = set(string.ascii_uppercase)
HEX_LOWER = set("0123456789abcdef")

MONTHS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)
EXTENSIONS = ("pdf", "docx", "xlsx", "csv", "txt", "png", "jpg", "jpeg", "json", "xml", "html", "zip")
CATALOGS = (
    ("yesno", ("YES", "NO")),
    ("bool", ("true", "false", "TRUE", "FALSE")),
    ("currency", ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF")),
    ("level", ("DEBUG", "INFO", "WARN", "ERROR", "FATAL")),
    ("doctype", ("Invoice", "Statement", "Receipt", "Quote")),
    ("ampm", ("AM", "PM")),
    ("month", MONTHS),
    ("extension", EXTENSIONS),
    ("size", ("XS", "S", "M", "L", "XL", "XXL")),
    ("http", ("200", "201", "204", "301", "302", "400", "401", "403", "404", "500")),
    ("quarter", ("Q1", "Q2", "Q3", "Q4")),
)

WIDEN = {
    r"\d": "[0-9a-f]",
    "[0-9a-f]": "[A-Za-z0-9]",
    "[0-9A-F]": "[A-Za-z0-9]",
    "[a-z]": "[a-z0-9]",
    "[A-Z]": "[A-Z0-9]",
    "[A-Za-z]": "[A-Za-z0-9]",
    "[a-z0-9]": "[a-z0-9_]",
    "[A-Z0-9]": "[A-Za-z0-9]",
    "[A-Za-z0-9]": r"\w",
}

# Tails that show up in some examples and not others. The pattern is the
# optional form; the detector only fires when the sample is mixed.
TAILS = (
    (re.compile(r" \(\d+\)$"), r"(?: \(\d+\))?"),
    (re.compile(r" bis$"), r"(?: bis)?"),
    (re.compile(r"\.\d+$"), r"(?:\.\d+)?"),
    (re.compile(r"\([a-z]\)$"), r"(?:\([a-z]\))?"),
)

MAX_BODY = 280


def structural_candidates(strings: list[str]) -> list[Candidate]:
    """Patterns from the structural families. The heuristic is not included."""
    if len(strings) < 2:
        return []
    found: dict[str, str] = {}

    def add(body: str, tag: str) -> None:
        if not body or len(body) > MAX_BODY:
            return
        pattern = "^" + body + "$"
        if pattern in found:
            return
        if parse_pattern(pattern) is None:
            return
        found[pattern] = tag

    for body, tag in _structure(strings):
        add(body, tag)
    for body in _mixture(strings):
        add(body, "mixture")
    for body in _primitives(strings):
        add(body, "primitive")
    seeds = [infer_regex(strings)] + ["^" + body + "$" for body, _tag in _structure(strings)[:4]]
    for body in _ast_search(strings, seeds):
        if body.startswith("^") and body.endswith("$"):
            body = body[1:-1]
        add(body, "ast_search")
    return [Candidate(pattern, tag) for pattern, tag in found.items()]


def _structure(strings: list[str]) -> list[tuple[str, str]]:
    tagged: list[tuple[str, str]] = []
    for body in _span_bodies(strings):
        tagged.append((body, "span_merge"))
    for body in _grouping_bodies(strings):
        tagged.append((body, "grouping"))
    for body in _repeat_bodies(strings):
        tagged.append((body, "repeat_motif"))
    for body in _delimiter_bodies(strings):
        tagged.append((body, "delimiter_split"))
    for body in _optional_bodies(strings):
        tagged.append((body, "optional_span"))
    for body in _segment_bodies(strings):
        tagged.append((body, "segment"))
    return tagged


def _dedupe(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.append(item)
    return seen


def _common_ends(strings: list[str]) -> tuple[str, str]:
    """Longest literal prefix and suffix shared by every string."""
    prefix = strings[0]
    suffix = strings[0]
    for text in strings[1:]:
        while prefix and not text.startswith(prefix):
            prefix = prefix[:-1]
        while suffix and not text.endswith(suffix):
            suffix = suffix[1:]
    if prefix and suffix and len(prefix) + len(suffix) > min(len(text) for text in strings):
        overlap = len(prefix) + len(suffix) - min(len(text) for text in strings)
        suffix = suffix[overlap:]
    return prefix, suffix


def _peel(strings: list[str], prefix: str, suffix: str) -> list[str]:
    peeled = []
    for text in strings:
        body = text[len(prefix):] if prefix else text
        if suffix:
            body = body[: len(body) - len(suffix)]
        peeled.append(body)
    return peeled


def _catalog_body(values: list[str]) -> str | None:
    observed = set(values)
    if not observed:
        return None
    for _name, catalog in CATALOGS:
        allowed = set(catalog)
        if observed <= allowed and len(observed) < len(allowed) <= 16:
            alts = "|".join(escape_literal(item) for item in catalog)
            return f"(?:{alts})"
    return None


def _prefer_class(token: str, sample_size: int) -> bool:
    """A literal copied from a handful of examples is often an accident."""
    if sample_size > 8 or not token or any(ch.isdigit() for ch in token):
        return False
    if token.isupper() and len(token) <= 6:
        return False
    if token.islower() and len(token) >= 3:
        return True
    if len(token) >= 4 and token[:1].isupper() and token[1:].islower():
        return True
    return False


def _class_quant(chars: set[str], lo: int, hi: int) -> list[str]:
    if not chars:
        return []
    atom = named_class(chars)
    if atom == ".":
        return []
    options = [_quantify(atom, lo, hi)]
    wider = WIDEN.get(atom)
    if wider:
        options.append(_quantify(wider, lo, hi))
    if lo == hi and lo >= 1:
        options.append(_quantify(atom, 1, None))
    elif hi is not None and hi > lo:
        options.append(_quantify(atom, lo, None))
    return _dedupe(options)


def _length_alts(atom: str, lengths: list[int]) -> str | None:
    unique = sorted(set(lengths))
    if not 2 <= len(unique) <= 4:
        return None
    if unique[-1] - unique[0] + 1 <= len(unique) + 1:
        return None
    parts = [_quantify(atom, length, length) for length in unique]
    return "(?:" + "|".join(parts) + ")"


def piece_options(values: list[str], depth: int) -> list[str]:
    """A few pattern bodies that cover these component strings."""
    if not values:
        return []
    options: list[str] = []
    unique = list(dict.fromkeys(values))
    if len(unique) == 1:
        token = unique[0]
        literal = escape_literal(token)
        catalog = _catalog_body([token])
        if catalog and not token.isupper():
            options.append(catalog)
        if _prefer_class(token, len(values)):
            atom = named_class(set(token))
            if atom != ".":
                options.append(_quantify(atom, 1, None))
                options.append(_quantify(atom, len(token), len(token)))
        options.append(literal)
        if token.isdigit() and 1 <= len(token) <= 3:
            options.append(_quantify(r"\d", len(token), len(token)))
        return _dedupe(options)[:5]

    digit_free = all(not any(ch.isdigit() for ch in value) for value in unique)
    punct_free = all(ch.isalnum() for value in unique for ch in value)
    # Literals are free in the code-length rule, so an alternation of every
    # observed token looks cheaper than a class and then misses hidden values.
    # Use it only when the tokens actually repeat, or when a catalog covers them.
    repeats = len(unique) * 2 <= len(values)
    binary = len(unique) <= 2 and max(len(value) for value in unique) <= 12
    catalog = _catalog_body(unique) if digit_free or all(value.isdigit() for value in unique) else None
    if digit_free and punct_free and len(unique) <= 6 and max(len(value) for value in unique) <= 16 and (repeats or binary):
        ordered = sorted(unique, key=lambda value: (len(value), value))
        alts = "|".join(escape_literal(value) for value in ordered)
        options.append(f"(?:{alts})")
    if catalog:
        options.append(catalog)

    if all(re.fullmatch(r"[A-Z][a-z]+", value) for value in values):
        options.append("[A-Z][a-z]+")

    if depth > 0:
        options.extend(_delimiter_join(values, depth - 1)[:2])
        options.extend(_repeat_of(values)[:2])

    chars = set("".join(values))
    lengths = [len(value) for value in values]
    lo, hi = min(lengths), max(lengths)
    atom = named_class(chars)
    class_opts = _class_quant(chars, lo, hi)
    if class_opts and atom != ".":
        gapped = _length_alts(atom, lengths)
        if gapped:
            options.append(gapped)
        # A handful of examples often share a length by accident. Keep the
        # unbounded form near the front so it is not cut off.
        if lo == hi and lo <= 12 and len(values) <= 8:
            options.append(_quantify(atom, 1, None))
        options.extend(class_opts)
    return _dedupe(options)[:5]


def _uniform_atom(texts: list[str]) -> str | None:
    chars = set("".join(texts))
    if not chars:
        return None
    atom = named_class(chars)
    if atom == ".":
        return None
    return atom


def _repeat_of(values: list[str]) -> list[str]:
    """piece (delim piece)* with a varying repeat count, including zero extras."""
    results: list[str] = []
    for delim in (".", "-", "_", "/", ":"):
        if not any(delim in value for value in values):
            continue
        rows = [value.split(delim) for value in values]
        if any(piece == "" for row in rows for piece in row):
            continue
        flat = [piece for row in rows for piece in row]
        atom = _uniform_atom(flat)
        if atom is None:
            continue
        counts = [len(row) for row in rows]
        lens = [len(piece) for piece in flat]
        unit = _quantify(atom, min(lens), max(lens))
        unit_plus = _quantify(atom, 1, None)
        extra_lo = min(counts) - 1
        extra_hi = max(counts) - 1
        esc = escape_literal(delim)
        bodies = []
        if extra_hi <= 0:
            bodies.append(unit)
        else:
            bodies.append(unit + _quantify(esc + unit, extra_lo, extra_hi))
            wide_lo = max(0, extra_lo - 1)
            wide_hi = extra_hi + 1
            bodies.append(unit_plus + _quantify(esc + unit_plus, wide_lo, wide_hi))
        for body in bodies:
            if body not in results:
                results.append(body)
        if len(results) >= 4:
            break
    return results


def _stable_columns(values: list[str]) -> list[tuple[str, list[list[str]]]]:
    found: list[tuple[str, list[list[str]]]] = []
    chars: set[str] = set()
    for value in values:
        chars.update(ch for ch in value if not ch.isalnum())
    for delim in chars:
        if any(value.count(delim) == 0 for value in values):
            continue
        counts = {value.count(delim) for value in values}
        if len(counts) != 1:
            continue
        count = next(iter(counts))
        if count < 1 or count > 6:
            continue
        rows = [value.split(delim) for value in values]
        if any(piece == "" for row in rows for piece in row):
            continue
        found.append((delim, rows))
    return found


def _delimiter_join(values: list[str], depth: int) -> list[str]:
    results: list[str] = []
    for delim, rows in _stable_columns(values):
        width = len(rows[0])
        columns = [[row[index] for row in rows] for index in range(width)]
        options = [piece_options(column, depth) for column in columns]
        if any(not opts for opts in options):
            continue
        esc = escape_literal(delim)
        tight = esc.join(opts[0] for opts in options)
        results.append(tight)
        relaxed_parts = []
        for opts in options:
            relaxed_parts.append(opts[1] if len(opts) > 1 else opts[0])
        relaxed = esc.join(relaxed_parts)
        if relaxed != tight:
            results.append(relaxed)
        if len(results) >= 4:
            break
    return results


def _delimiter_bodies(strings: list[str]) -> list[str]:
    return _delimiter_join(strings, 2)


def _with_affix(strings: list[str], middle_builder) -> list[str]:
    prefix, suffix = _common_ends(strings)
    middles = _peel(strings, prefix, suffix)
    if any(middle == "" for middle in middles):
        return []
    bodies = middle_builder(middles)
    if not bodies:
        return []
    wrapped = [escape_literal(prefix) + body + escape_literal(suffix) for body in bodies]
    if prefix.endswith(" ") and prefix[:-1]:
        optional = escape_literal(prefix[:-1]) + r"\s?"
        wrapped.extend(optional + body + escape_literal(suffix) for body in bodies[:2])
    return _dedupe(wrapped)


def _grouping_middle(texts: list[str]) -> list[str]:
    if not texts or any(text == "" for text in texts):
        return []
    results: list[str] = []
    for decimal in (".", ","):
        if any(text.count(decimal) != 1 for text in texts):
            continue
        wholes: list[str] = []
        fracs: list[str] = []
        failed = False
        for text in texts:
            whole, frac = text.split(decimal)
            if not frac.isdigit() or not whole:
                failed = True
                break
            wholes.append(whole)
            fracs.append(frac)
        if failed:
            continue
        frac_pat = _quantify(r"\d", min(map(len, fracs)), max(map(len, fracs)))
        decimal_pat = escape_literal(decimal)
        for group in (",", " ", "\u00a0"):
            if group == decimal:
                continue
            if _grouped(wholes, group):
                group_pat = escape_literal(group)
                results.append(rf"\d{{1,3}}(?:{group_pat}\d{{3}})*{decimal_pat}{frac_pat}")
        if all(whole.isdigit() for whole in wholes):
            whole_pat = _quantify(r"\d", min(map(len, wholes)), max(map(len, wholes)))
            results.append(whole_pat + decimal_pat + frac_pat)
    for group in (",", " "):
        if _grouped(texts, group):
            group_pat = escape_literal(group)
            results.append(rf"\d{{1,3}}(?:{group_pat}\d{{3}})*")
    return _dedupe(results)


def _grouped(wholes: list[str], group: str) -> bool:
    if not any(group in whole for whole in wholes):
        return False
    for whole in wholes:
        parts = whole.split(group) if group in whole else [whole]
        if any(not part.isdigit() for part in parts):
            return False
        if len(parts) == 1:
            if not 1 <= len(parts[0]) <= 3:
                return False
            continue
        if not 1 <= len(parts[0]) <= 3:
            return False
        if any(len(part) != 3 for part in parts[1:]):
            return False
    return True


def _grouping_bodies(strings: list[str]) -> list[str]:
    direct = _grouping_middle(strings)
    return _dedupe(direct + _with_affix(strings, _grouping_middle))


def _repeat_middle(texts: list[str]) -> list[str]:
    spaced = _optional_lead(texts)
    results = _repeat_of(texts)
    if spaced is not None:
        lead, stripped = spaced
        for body in _repeat_of(stripped):
            results.append(lead + body)
    return _dedupe(results)


def _optional_lead(texts: list[str]) -> tuple[str, list[str]] | None:
    if not any(text.startswith(" ") for text in texts):
        return None
    if any(text.startswith("  ") or not text.startswith(" ") and False for text in texts):
        return None
    if all(text.startswith(" ") for text in texts):
        return None
    stripped = [text[1:] if text.startswith(" ") else text for text in texts]
    if any(text == "" for text in stripped):
        return None
    return r"\s?", stripped


def _repeat_bodies(strings: list[str]) -> list[str]:
    direct = _repeat_of(strings)
    affixed = _with_affix(strings, _repeat_middle)
    return _dedupe(direct + affixed)


def _optional_bodies(strings: list[str]) -> list[str]:
    results: list[str] = []
    prefix, suffix = _common_ends(strings)
    middles = _peel(strings, prefix, suffix)
    for matcher, tail_pat in TAILS:
        hits = [bool(matcher.search(middle)) for middle in middles]
        if not any(hits) or all(hits):
            continue
        stems = [matcher.sub("", middle) for middle in middles]
        if any(stem == "" for stem in stems):
            continue
        for stem in piece_options(stems, 1)[:4]:
            results.append(escape_literal(prefix) + stem + tail_pat + escape_literal(suffix))
    return _dedupe(results)


def _segment_bodies(strings: list[str]) -> list[str]:
    rows = [_runs(text) for text in strings]
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        return []
    parts: list[str] = []
    relaxed: list[str] = []
    for index in range(width):
        kind = rows[0][index][0]
        texts = [row[index][1] for row in rows]
        if any(row[index][0] != kind for row in rows):
            return []
        if kind == "P":
            if any(text != texts[0] for text in texts):
                return []
            piece = escape_literal(texts[0])
            parts.append(piece)
            relaxed.append(piece)
            continue
        options = piece_options(texts, 1)
        if not options:
            return []
        parts.append(options[0])
        relaxed.append(options[min(1, len(options) - 1)])
    tight = "".join(parts)
    second = "".join(relaxed)
    return _dedupe([tight, second])


def _runs(text: str) -> list[tuple[str, str]]:
    runs: list[tuple[str, str]] = []
    index = 0
    while index < len(text):
        kind = _run_kind(text[index])
        end = index + 1
        if kind == "P":
            while end < len(text) and text[end] == text[index]:
                end += 1
        else:
            while end < len(text) and _run_kind(text[end]) == kind:
                end += 1
        runs.append((kind, text[index:end]))
        index = end
    return runs


def _run_kind(ch: str) -> str:
    if ch.isdigit():
        return "D"
    if ch.islower():
        return "L"
    if ch.isupper():
        return "U"
    if ch.isspace():
        return "S"
    return "P"


def _span_bodies(strings: list[str]) -> list[str]:
    if len({len(text) for text in strings}) != 1:
        return []
    width = len(strings[0])
    if width == 0 or width > 80:
        return []
    columns = [{text[index] for text in strings} for index in range(width)]
    atoms = []
    for chars in columns:
        if len(chars) == 1:
            atoms.append(("lit", next(iter(chars))))
        else:
            atoms.append(("cls", named_class(chars)))
    absorbed = list(atoms)
    for index in range(1, width - 1):
        if atoms[index][0] != "lit":
            continue
        left, right = atoms[index - 1], atoms[index + 1]
        if left[0] != "cls" or right[0] != "cls" or left[1] != right[1]:
            continue
        if _char_in_named(left[1], atoms[index][1]):
            absorbed[index] = ("cls", left[1])
    patterns = [_render_atoms(atoms), _render_atoms(absorbed)]
    everything = set("".join(strings))
    atom = named_class(everything)
    if atom != ".":
        patterns.append(_quantify(atom, width, width))
        wider = WIDEN.get(atom)
        if wider:
            patterns.append(_quantify(wider, width, width))
    return _dedupe(patterns)


def _char_in_named(atom: str, ch: str) -> bool:
    if atom == r"\d":
        return ch.isdigit()
    if atom == r"\w":
        return ch.isalnum() or ch == "_"
    if atom == r"\s":
        return ch.isspace()
    if atom.startswith("[") and atom.endswith("]"):
        # Use the cost-model class reader via a one-character probe.
        from pattern_induction.candidates import _char_in_class

        return _char_in_class(atom, ch)
    return False


def _render_atoms(atoms: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    index = 0
    while index < len(atoms):
        kind, value = atoms[index]
        if kind == "lit":
            end = index + 1
            while end < len(atoms) and atoms[end][0] == "lit":
                end += 1
            text = "".join(atoms[cursor][1] for cursor in range(index, end))
            parts.append(escape_literal(text))
            index = end
            continue
        end = index + 1
        while end < len(atoms) and atoms[end] == ("cls", value):
            end += 1
        if value == ".":
            return ""
        parts.append(_quantify(value, end - index, end - index))
        index = end
    return "".join(parts)


def _signature(text: str) -> tuple[str, ...]:
    parts: list[str] = []
    for ch in text:
        if ch.isdigit():
            kind = "D"
        elif ch.islower():
            kind = "L"
        elif ch.isupper():
            kind = "U"
        elif ch.isspace():
            kind = "S"
        else:
            kind = ch
        if parts and parts[-1] == kind and kind in "DLUS":
            continue
        parts.append(kind)
    return tuple(parts)


def _raw_literal(members: list[str]) -> str | None:
    unique = list(dict.fromkeys(members))
    if len(unique) > 6 or any(len(value) > 16 for value in unique):
        return None
    if any(any(ch.isdigit() for ch in value) for value in unique):
        return None
    ordered = sorted(unique, key=lambda value: (len(value), value))
    if len(ordered) == 1:
        return escape_literal(ordered[0])
    alts = "|".join(escape_literal(value) for value in ordered)
    return f"(?:{alts})"


def _sentinel_branch(members: list[str]) -> str | None:
    """Closed tokens such as N/A, YES, or UNKNOWN, not a single mixed-case sample."""
    if any(any(ch.isdigit() for ch in value) for value in members):
        return None
    unique = list(dict.fromkeys(members))
    if len(unique) > 6:
        return None
    punct = any(not ch.isalnum() for value in unique for ch in value)
    uniform_case = all(value.isupper() or value.islower() for value in unique)
    if punct:
        return _raw_literal(unique)
    if uniform_case and (len(unique) >= 2 or (len(unique) == 1 and len(unique[0]) >= 3)):
        literal = _raw_literal(unique)
        catalog = _catalog_body(unique)
        return catalog or literal
    return None


def _branch(members: list[str]) -> str | None:
    structural = []
    for body, _tag in _structure(members):
        if body not in structural and not _is_member_literal(body, members):
            structural.append(body)
    sentinel = _sentinel_branch(members)
    has_digit = any(any(ch.isdigit() for ch in value) for value in members)
    if has_digit and structural:
        # The code-length rule drops frozen columns such as a month that
        # happened to start with 0 in every visible example.
        return min(structural, key=lambda body: (pattern_cost("^" + body + "$", members, True), len(body)))
    if sentinel and not has_digit:
        return sentinel
    if structural:
        return min(structural, key=lambda body: (pattern_cost("^" + body + "$", members, True), len(body)))
    return sentinel


def _is_member_literal(body: str, members: list[str]) -> bool:
    if len(members) != 1:
        return False
    return body == escape_literal(members[0])


def _sentinel_cluster(members: list[str]) -> bool:
    """A short closed token, not another spelling of the same digit pattern."""
    if any(any(ch.isdigit() for ch in value) for value in members):
        return False
    unique = list(dict.fromkeys(members))
    if len(unique) > 4 or any(len(value) > 12 for value in unique):
        return False
    if any(not ch.isalnum() for value in unique for ch in value):
        return True
    if _catalog_body(unique):
        return True
    return all(value.isupper() or value.islower() for value in unique) and (
        len(unique) == 1 or len(unique) * 2 <= len(members)
    )


def _mixture(strings: list[str]) -> list[str]:
    """Alternate a sentinel with a structured pattern. Do not split two digit shapes."""
    groups: dict[tuple[str, ...], list[str]] = {}
    for text in strings:
        groups.setdefault(_signature(text), []).append(text)
    if not 2 <= len(groups) <= 4:
        return []
    sentinels: list[list[str]] = []
    structured: list[list[str]] = []
    for members in groups.values():
        if _sentinel_cluster(members):
            sentinels.append(members)
        elif any(any(ch.isdigit() for ch in value) for value in members):
            structured.append(members)
        else:
            return []
    if not sentinels or not structured:
        return []
    branches: list[str] = []
    catalogs: list[str] = []
    catalog_used = False
    for members in sentinels + structured:
        branch = _branch(members)
        if not branch:
            return []
        branches.append(branch)
        catalog_branch = _catalog_swap(members, branch)
        catalogs.append(catalog_branch or branch)
        catalog_used = catalog_used or catalog_branch is not None
    results = ["(?:" + "|".join(f"(?:{branch})" for branch in branches) + ")"]
    if catalog_used:
        swapped = "(?:" + "|".join(f"(?:{branch})" for branch in catalogs) + ")"
        if swapped not in results:
            results.append(swapped)
    return results


def _catalog_swap(members: list[str], branch: str) -> str | None:
    unique = list(dict.fromkeys(members))
    catalog = _catalog_body(unique)
    if catalog and catalog != branch and not any(ch.isdigit() for text in unique for ch in text):
        return catalog
    return None


def _primitives(strings: list[str]) -> list[str]:
    results: list[str] = []
    results.extend(_uuid(strings))
    results.extend(_ipv4(strings))
    results.extend(_email(strings))
    results.extend(_url(strings))
    return _dedupe(results)


def _uuid(strings: list[str]) -> list[str]:
    pattern = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
    if not all(pattern.fullmatch(text) for text in strings):
        return []
    hex_atom = "[0-9a-f]" if all(ch in HEX_LOWER or ch == "-" for text in strings for ch in text) else "[0-9A-Fa-f]"
    generic = f"{hex_atom}{{8}}-{hex_atom}{{4}}-{hex_atom}{{4}}-{hex_atom}{{4}}-{hex_atom}{{12}}"
    versions = {text[14] for text in strings}
    variants = {text[19].lower() for text in strings}
    results = [generic]
    if versions == {"4"} and variants <= set("89ab"):
        variant = "[89ab]" if hex_atom == "[0-9a-f]" else "[89abAB]"
        results.append(
            f"{hex_atom}{{8}}-{hex_atom}{{4}}-4{hex_atom}{{3}}-{variant}{hex_atom}{{3}}-{hex_atom}{{12}}"
        )
    return results


def _ipv4(strings: list[str]) -> list[str]:
    rows = []
    for text in strings:
        parts = text.split(".")
        if len(parts) != 4 or any(not part.isdigit() for part in parts):
            return []
        numbers = []
        for part in parts:
            if len(part) > 1 and part.startswith("0"):
                return []
            value = int(part)
            if value > 255:
                return []
            numbers.append(value)
        rows.append(numbers)
    if not rows:
        return []
    loose = r"(?:\d{1,3}\.){3}\d{1,3}"
    octet = r"(?:25[0-5]|2[0-4]\d|1?\d{1,2})"
    return [loose, rf"(?:{octet}\.){{3}}{octet}"]


def _email(strings: list[str]) -> list[str]:
    if not all(text.count("@") == 1 and " " not in text for text in strings):
        return []
    locals_, domains = zip(*(text.split("@") for text in strings))
    if any("." not in domain for domain in domains):
        return []
    lefts = piece_options(list(locals_), 2)[:3]
    rights = piece_options(list(domains), 2)[:2]
    return _dedupe([left + "@" + right for left in lefts for right in rights])[:4]


def _url(strings: list[str]) -> list[str]:
    if not all(text.startswith("https://") or text.startswith("http://") for text in strings):
        return []
    scheme = "https://" if all(text.startswith("https://") for text in strings) else "https?://"
    # https?:// is not worth a special parser fight; keep a concrete scheme when mixed.
    if scheme == "https?://":
        scheme_body = r"https?://"
    else:
        scheme_body = escape_literal(scheme)
    rest = [text.split("://", 1)[1] for text in strings]
    if any("/" not in text for text in rest):
        hosts = piece_options(rest, 2)[:2]
        return [scheme_body + host for host in hosts]
    hosts = []
    paths = []
    for text in rest:
        host, path = text.split("/", 1)
        hosts.append(host)
        paths.append(path)
    host_opts = piece_options(hosts, 2)[:2]
    path_opts = piece_options(paths, 1)[:2]
    return _dedupe([scheme_body + host + "/" + path for host in host_opts for path in path_opts])[:3]


def _ast_search(strings: list[str], seeds: list[str]) -> list[str]:
    """A short beam of grammar edits, ordered by the code-length rule."""
    if len(strings) > 24 or sum(len(text) for text in strings) > 500:
        return []
    seen: set[str] = set()
    frontier: list[tuple[float, str]] = []
    for seed in seeds:
        if not seed or seed in seen:
            continue
        seen.add(seed)
        cost = pattern_cost(seed, strings, True)
        if cost < 1_000_000:
            frontier.append((cost, seed))
    frontier.sort()
    frontier = frontier[:6]
    discovered: list[tuple[float, str]] = []
    for _depth in range(2):
        nxt: list[tuple[float, str]] = []
        for _cost, pattern in frontier:
            for neighbor in _neighbors(pattern)[:16]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                if not _matches(neighbor, strings):
                    continue
                cost = pattern_cost(neighbor, strings, True)
                if cost >= 1_000_000:
                    continue
                nxt.append((cost, neighbor))
                discovered.append((cost, neighbor))
        if not nxt:
            break
        nxt.sort()
        frontier = nxt[:6]
    discovered.sort()
    return [pattern for _cost, pattern in discovered[:12]]


def _matches(pattern: str, strings: list[str]) -> bool:
    try:
        compiled = re.compile(pattern)
    except re.error:
        return False
    try:
        return all(compiled.fullmatch(text) for text in strings)
    except re.error:
        return False


def _neighbors(pattern: str) -> list[str]:
    node = parse_pattern(pattern)
    if node is None:
        return []
    variants: list[Node] = []
    variants.extend(_single_site(node, _relax_quant))
    variants.extend(_single_site(node, _class_step))
    variants.extend(_single_site(node, _digit_literal))
    variants.extend(_merge_adjacent(node))
    rendered: list[str] = []
    for variant in variants:
        try:
            text = anchor(compress(variant))
        except (ValueError, IndexError):
            continue
        if text not in rendered:
            rendered.append(text)
        if len(rendered) >= 16:
            break
    return rendered


def _relax_quant(node: Node) -> list[Node]:
    if node.op not in {"CLS", "DOT", "REPEAT"}:
        return []
    lo, hi = node.lo, node.hi
    options: list[tuple[int, int | None]] = []
    if hi is not None and lo == hi and lo >= 1:
        options.append((lo, None))
        options.append((max(1, lo - 1), lo + 1))
    elif hi is not None and lo < hi:
        options.append((lo, None))
    return [node.replace(lo=new_lo, hi=new_hi) for new_lo, new_hi in options]


def _class_step(node: Node) -> list[Node]:
    if node.op != "CLS":
        return []
    wider = WIDEN.get(node.text)
    if not wider:
        return []
    return [node.replace(text=wider)]


def _digit_literal(node: Node) -> list[Node]:
    if node.op != "LIT" or node.lo != 1 or node.hi != 1:
        return []
    if not node.text.isdigit() or not 1 <= len(node.text) <= 3:
        return []
    return [Node("CLS", text=r"\d", lo=len(node.text), hi=len(node.text))]


def _merge_adjacent(node: Node) -> list[Node]:
    """Merge an interior literal whose character sits in both neighboring classes."""
    if node.op != "SEQ":
        return []
    kids = list(node.kids)
    merged: list[Node] = []
    for index in range(1, len(kids) - 1):
        current = kids[index]
        left, right = kids[index - 1], kids[index + 1]
        if current.op != "LIT" or len(current.text) != 1 or current.lo != 1 or current.hi != 1:
            continue
        if left.op != "CLS" or right.op != "CLS" or left.text != right.text:
            continue
        if not _char_in_named(left.text, current.text):
            continue
        updated = kids[:]
        updated[index] = Node("CLS", text=left.text, lo=1, hi=1)
        merged.append(compress(node.replace(kids=tuple(updated))))
    child_merges: list[Node] = []
    for index, kid in enumerate(kids):
        for variant in _merge_adjacent(kid):
            updated = kids[:]
            updated[index] = variant
            child_merges.append(compress(node.replace(kids=tuple(updated))))
    return merged + child_merges


