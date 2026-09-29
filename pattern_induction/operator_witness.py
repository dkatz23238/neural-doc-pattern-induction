"""Deterministic witnesses for character sets and integer ranges.

The frozen witness head was trained on the exact fragment. A mixed character
set and an integer range disagree on strings that head was not asked to judge.
These probes are built from the supports: a complete mixed alphabet is a
positive, and an integer past the observed endpoints abstains.
"""

from __future__ import annotations

import string

from pattern_induction.charset import spec_chars
from pattern_induction.program import Prog, execute

_DIGIT = set(string.digits)
_UPPER = set(string.ascii_uppercase)
_LOWER = set(string.ascii_lowercase)


def complete_mixed(spec: str) -> bool:
    """True when every class the spec touches is the whole class, and two or more classes appear.

    ``[0-9A-Z]`` is complete. ``[0-8A-Y]`` and ``[2-5]`` are not. A single full
    class is a class fact, not a mixed set.
    """
    chars = spec_chars(spec)
    kinds = 0
    if any(ch.isdigit() for ch in chars):
        if not _DIGIT <= chars:
            return False
        kinds += 1
    if any(ch.isupper() for ch in chars):
        if not _UPPER <= chars:
            return False
        kinds += 1
    if any(ch.islower() for ch in chars):
        if not _LOWER <= chars:
            return False
        kinds += 1
    return kinds >= 2


def operator_signature(prog: Prog) -> tuple:
    """SET and INT nodes, in preorder. Exact-fragment programs sign as empty."""
    found: list[tuple] = []

    def walk(node: Prog) -> None:
        if node.op in {"SET", "INT"}:
            found.append((node.op, node.spec, node.lo, node.hi))
        for child in node.children:
            walk(child)

    walk(prog)
    return tuple(found)


def split_operator_families(families: list[list[Prog]]) -> list[list[Prog]]:
    """Keep hull and broad sets in different families when the battery cannot see the difference.

    Programs with an empty operator signature stay together, so the exact
    fragment grouping is unchanged.
    """
    split: list[list[Prog]] = []
    for group in families:
        buckets: dict[tuple, list[Prog]] = {}
        for prog in group:
            buckets.setdefault(operator_signature(prog), []).append(prog)
        split.extend(buckets.values())
    return split


def _set_specs(prog: Prog) -> list[str]:
    found: list[str] = []

    def walk(node: Prog) -> None:
        if node.op == "SET" and node.spec:
            found.append(node.spec)
        for child in node.children:
            walk(child)

    walk(prog)
    return found


def _class_seen(char: str, supports: list[str]) -> bool:
    if char.isdigit():
        return any(ch.isdigit() for text in supports for ch in text)
    if char.isupper():
        return any(ch.isupper() for text in supports for ch in text)
    if char.islower():
        return any(ch.islower() for text in supports for ch in text)
    return any(char in text for text in supports)


def _splice(supports: list[str], char: str, broad: Prog, narrow: Prog) -> str:
    """One support with a single character replaced, accepted by the broad set and rejected by the hull."""
    for text in supports:
        for index, current in enumerate(text):
            if current == char:
                continue
            probe = text[:index] + char + text[index + 1 :]
            if execute(broad, probe, supports) and not execute(narrow, probe, supports):
                return probe
    return ""


def operator_judgments(families: list[list[Prog]], supports: list[str], limit: int = 2) -> list[tuple[str, float]]:
    """Positive probes for a complete mixed alphabet against a tighter set.

    Two confident votes clear the grammar tie. Integer endpoints are not voted
    here: a value outside the observed min and max abstains, so a tight range
    does not outrank the digit class.
    """
    by_spec: dict[str, Prog] = {}
    for group in families:
        if not group:
            continue
        for spec in _set_specs(group[0]):
            by_spec.setdefault(spec, group[0])
    probes: list[tuple[str, float]] = []
    seen: set[str] = set()
    for broad_spec, broad in by_spec.items():
        if not complete_mixed(broad_spec):
            continue
        bigger = spec_chars(broad_spec)
        for other_spec, narrow in by_spec.items():
            if other_spec == broad_spec:
                continue
            extra = bigger - spec_chars(other_spec)
            if not extra:
                continue
            chars: list[str] = []
            digits = sorted(ch for ch in extra if ch.isdigit())
            uppers = sorted(ch for ch in extra if ch.isupper())
            lowers = sorted(ch for ch in extra if ch.islower())
            if digits:
                chars.append(digits[-1])
            if uppers:
                chars.append(uppers[-1])
            if lowers and len(chars) < 2:
                chars.append(lowers[-1])
            for char in chars:
                if not _class_seen(char, supports):
                    continue
                probe = _splice(supports, char, broad, narrow)
                if not probe or probe in seen:
                    continue
                seen.add(probe)
                probes.append((probe, 0.95))
                if len(probes) >= limit:
                    return probes
    return probes


def integer_abstentions(families: list[list[Prog]], supports: list[str]) -> list[tuple[str, float]]:
    """Boundary integers. Outside the observed endpoints these abstain.

    ``2023`` and ``2028`` distinguish a tight year from a wider one, and from
    ``\\d{4}``. Voting them positive or negative would lock in an endpoint the
    supports do not show. The selector therefore records them at 0.5.
    """
    specs: list[tuple[str, int]] = []

    def walk(node: Prog) -> None:
        if node.op == "INT" and ":" in node.spec:
            specs.append((node.spec, node.lo if node.lo == node.hi else 0))
        for child in node.children:
            walk(child)

    for group in families:
        if group:
            walk(group[0])
    rows: list[tuple[str, float]] = []
    seen: set[str] = set()
    for spec, width in specs:
        minimum, maximum = (int(part) for part in spec.split(":"))
        for value in (minimum - 1, minimum, maximum, maximum + 1):
            if value < 0:
                continue
            text = f"{value:0{width}d}" if width else str(value)
            if text in seen:
                continue
            seen.add(text)
            if any(text == support or text in support for support in supports) and minimum <= value <= maximum:
                continue
            rows.append((text, 0.5))
    return rows


def identifiable_facts(supports: list[str]) -> list:
    """Facts a reader can name from the supports, without a wider hidden guess.

    Integer endpoints stay at the observed min and max. Mixed alphabets stay
    at the observed hull. The complete class union is the guess, so it is left out.
    A digit class on a span that already has an integer range is left out too.
    """
    from pattern_induction.charset import v2_facts
    from pattern_induction.document_facts import document_facts

    facts = list(document_facts(supports)) + list(v2_facts(supports))
    int_spans = {(fact.slot, fact.start, fact.end) for fact in facts if fact.kind == "int"}
    kept = []
    for fact in facts:
        if fact.kind == "set" and complete_mixed(fact.cls):
            continue
        if fact.kind == "cls" and fact.cls == "digit" and (fact.slot, fact.start, fact.end) in int_spans:
            continue
        kept.append(fact)
    return kept


def merge_judgments(
    neural: list[tuple[str, float]], operator: list[tuple[str, float]]
) -> list[tuple[str, float]]:
    """Operator probes override the frozen head on the same string."""
    override = {text: prob for text, prob in operator}
    merged = [(text, override.get(text, prob)) for text, prob in neural]
    seen = {text for text, _prob in neural}
    for text, prob in operator:
        if text not in seen:
            merged.append((text, prob))
    return merged
