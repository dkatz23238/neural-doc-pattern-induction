"""Stability of a candidate language without rebuilding the version space.

Mode A resynthesizes every leave-one-out subset. Mode B keeps the chart and
asks which programs are still licensed on the subset. Mode C scores each
candidate directly: a copy must still be spelled in the subset, an integer
range must still show both endpoints, and a character-set hull must still
show its endpoints. A complete mixed alphabet only needs its classes.
"""

from __future__ import annotations

import string

from pattern_induction.charset import spec_chars
from pattern_induction.operator_witness import complete_mixed
from pattern_induction.program import Prog, execute

_DIGIT = set(string.digits)
_UPPER = set(string.ascii_uppercase)
_LOWER = set(string.ascii_lowercase)


def subsets(supports: list[str]) -> list[list[str]]:
    rows = []
    for index in range(len(supports)):
        subset = [text for cursor, text in enumerate(supports) if cursor != index]
        if len(subset) >= 2:
            rows.append(subset)
    if len(supports) >= 4:
        rows.append(supports[:-2])
    if len(supports) >= 2:
        rows.append(supports[1:] + supports[:1])
    return rows


def _literal(prog: Prog, supports: list[str]) -> str:
    if prog.slot < 0 or prog.slot >= len(supports):
        return ""
    text = supports[prog.slot]
    if prog.start < 0 or prog.end > len(text) or prog.end <= prog.start:
        return ""
    return text[prog.start : prog.end]


def _appears(needle: str, subset: list[str]) -> bool:
    return bool(needle) and any(needle in text for text in subset)


def _class_present(chars: set[str], subset: list[str]) -> bool:
    pool = "".join(subset)
    return any(ch in pool for ch in chars)


def _endpoints_present(chars: set[str], alphabet: set[str], subset: list[str]) -> bool:
    present = sorted(ch for ch in chars if ch in alphabet)
    if not present:
        return True
    return _appears(present[0], subset) and _appears(present[-1], subset)


def licensed(prog: Prog, subset: list[str], supports: list[str]) -> bool:
    """Whether this program's own constraints are still visible on the subset."""

    def walk(node: Prog) -> bool:
        if node.op == "COPY":
            return _appears(_literal(node, supports), subset)
        if node.op == "CLS":
            alphabet = {"digit": _DIGIT, "upper": _UPPER, "lower": _LOWER}.get(node.cls, set())
            return _class_present(alphabet, subset)
        if node.op == "INT":
            if ":" not in node.spec:
                return False
            minimum, maximum = (int(part) for part in node.spec.split(":"))
            width = node.lo if node.lo == node.hi else 0
            left = f"{minimum:0{width}d}" if width else str(minimum)
            right = f"{maximum:0{width}d}" if width else str(maximum)
            return _appears(left, subset) and _appears(right, subset)
        if node.op == "SET":
            chars = spec_chars(node.spec)
            if complete_mixed(node.spec):
                if any(ch.isdigit() for ch in chars) and not _class_present(_DIGIT, subset):
                    return False
                if any(ch.isupper() for ch in chars) and not _class_present(_UPPER, subset):
                    return False
                if any(ch.islower() for ch in chars) and not _class_present(_LOWER, subset):
                    return False
                return True
            if not _endpoints_present(chars, _DIGIT, subset):
                return False
            if not _endpoints_present(chars, _UPPER, subset):
                return False
            if not _endpoints_present(chars, _LOWER, subset):
                return False
            literals = {ch for ch in chars if ch not in _DIGIT and ch not in _UPPER and ch not in _LOWER}
            return all(_appears(ch, subset) for ch in literals)
        return all(walk(child) for child in node.children)

    return walk(prog)


def direct_stability(families: list[list[Prog]], supports: list[str]) -> dict[int, float]:
    """Mode C. Each representative is scored on its own constraints."""
    groups = subsets(supports)
    if not groups:
        return {index: 1.0 for index in range(len(families))}
    scores = {}
    for index, group in enumerate(families):
        if not group:
            scores[index] = 0.0
            continue
        hits = sum(1 for subset in groups if licensed(group[0], subset, supports))
        scores[index] = hits / len(groups)
    return scores


def cached_stability(
    programs: list[Prog],
    supports: list[str],
    signatures: dict[int, tuple],
) -> dict[int, float]:
    """Mode B. A family's battery signature remains when some chart program of that signature is still licensed."""
    from pattern_induction.language import boundary_strings

    groups = subsets(supports)
    if not groups or not signatures:
        return {index: 1.0 for index in signatures}
    battery = boundary_strings(supports)
    wanted = set(signatures.values())
    paired: list[tuple[Prog, tuple]] = []
    for prog in programs:
        signature = tuple(bool(execute(prog, text, supports)) for text in battery)
        if signature in wanted:
            paired.append((prog, signature))
    hits = {index: 0 for index in signatures}
    for subset in groups:
        found = {signature for prog, signature in paired if licensed(prog, subset, supports)}
        for index, signature in signatures.items():
            if signature in found:
                hits[index] += 1
    return {index: hits[index] / len(groups) for index in signatures}
