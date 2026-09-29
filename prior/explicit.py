"""Score candidates by matching them to explicit support facts.

The facts are the ones already read off the strings: a repeated equal-width
field, a constant optional affix, the varying character class, the visible
width, a literal prefix or suffix, and a column that sits inside a frozen
lexicon. Nothing is pooled into the 160-d vector, and nothing is trained.

Each piece agrees (+1) or contradicts (-1). A set that merely covers a pure
observed class is compatible (0): the strings did not show the extra
characters. Weights are fixed. They are not fit on any benchmark.
"""

from __future__ import annotations

from pattern_induction.charset import spec_chars
from pattern_induction.program import Prog
from pattern_induction.renaming import DIGIT, LOWER, UPPER, structural_class
from prior.factors import (
    OPT_ABSENT,
    OPT_FIELD,
    OPT_OTHER,
    OPT_OTHER_AFFIX,
    OPT_PREFIX_AFFIX,
    _leaves,
    _optional_scope,
)
from prior.probe_labels import visible_facts

_CONTENT = {"CLS", "SET", "INT", "LEX"}


def _walk(node: Prog):
    yield node
    for child in node.children:
        yield from _walk(child)


def candidate_pieces(prog: Prog) -> dict:
    """The pieces a candidate claims, without reading the support vector."""
    nodes = list(_walk(prog))
    leaves = _leaves(prog)
    content = [leaf for leaf in leaves if leaf.op in _CONTENT]
    kinds = {leaf.op for leaf in content}
    classes = {leaf.cls for leaf in content if leaf.op == "CLS"}
    if "LEX" in kinds:
        field = "lex"
    elif "INT" in kinds:
        field = "int"
    elif "SET" in kinds:
        field = "set"
    elif "digit" in classes:
        field = "digit"
    elif "upper" in classes:
        field = "upper"
    elif "lower" in classes:
        field = "lower"
    else:
        field = "none"
    widths = [(leaf.lo, leaf.hi) for leaf in content if leaf.op in {"CLS", "SET", "INT"}]
    if not widths:
        width = "none"
    elif all(lo == hi for lo, hi in widths):
        width = "fixed"
    else:
        width = "open"
    scope_id = _optional_scope(prog)
    scope = {
        OPT_ABSENT: "absent",
        OPT_PREFIX_AFFIX: "prefix",
        OPT_OTHER_AFFIX: "suffix",
        OPT_FIELD: "field",
        OPT_OTHER: "other",
    }[scope_id]
    positions = [index for index, leaf in enumerate(leaves) if leaf.op in _CONTENT]
    first = positions[0] if positions else len(leaves)
    last = positions[-1] if positions else -1
    roles: set[str] = set()
    for index, leaf in enumerate(leaves):
        if leaf.op != "COPY":
            continue
        if not positions:
            roles.add("span")
        elif index < first:
            roles.add("prefix")
        elif index > last:
            roles.add("suffix")
        else:
            roles.add("infix")
    if "prefix" in roles and "suffix" in roles:
        affix = "both"
    elif "prefix" in roles:
        affix = "prefix"
    elif "suffix" in roles:
        affix = "suffix"
    else:
        affix = "none"
    covered: set[str] = set()
    for leaf in content:
        if leaf.op == "SET":
            covered |= spec_chars(leaf.spec)
        elif leaf.op == "CLS" and leaf.cls == "digit":
            covered |= set("0123456789")
        elif leaf.op == "CLS" and leaf.cls == "upper":
            covered |= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        elif leaf.op == "CLS" and leaf.cls == "lower":
            covered |= set("abcdefghijklmnopqrstuvwxyz")
    return {
        "repeat": any(node.op == "REPEAT" for node in nodes),
        "optional": scope,
        "field": field,
        "width": width,
        "affix": affix,
        "lexicon": "LEX" in kinds,
        "covers": covered,
    }


def _agree(left: bool, right: bool) -> float:
    return 1.0 if left == right else -1.0


def _covers_observed(pieces: dict, supports: list[str]) -> bool:
    chars = {ch for text in supports for ch in text if structural_class(ch) in {DIGIT, UPPER, LOWER}}
    return bool(chars) and chars <= pieces["covers"]


def _field_score(observed: str, pieces: dict, supports: list[str], has_lexicon: bool) -> float:
    field = pieces["field"]
    if field == "lex":
        # The lexicon piece scores the closed set. A lexicon field is only a contradiction when none is evidenced.
        return 0.0 if has_lexicon else -1.0
    if field == "int":
        return 1.0 if observed == "digit" else -1.0
    if field in {"digit", "upper", "lower"}:
        return 1.0 if observed == field else -1.0
    if field == "set":
        covers = _covers_observed(pieces, supports)
        if observed == "mixed":
            return 1.0 if covers else -1.0
        if covers:
            return 0.0
        return -1.0
    return -1.0


def match_score(facts: dict, pieces: dict, supports: list[str]) -> float:
    """Sum of piece agreements. A compatible set completion adds nothing."""
    has_lexicon = bool(facts["lexicon"])
    score = _agree(bool(facts["repeat"]), bool(pieces["repeat"]))
    score += 1.0 if facts["optional_scope"] == pieces["optional"] else -1.0
    score += 1.0 if facts["affix"] == pieces["affix"] else -1.0
    score += _field_score(str(facts["charset"]), pieces, supports, has_lexicon)
    if facts["width"] == "open":
        score += 1.0 if pieces["width"] == "open" else -1.0
    elif pieces["width"] == "none":
        score += -1.0
    else:
        # count still has a fixed field width; only the string length changes.
        score += 1.0 if pieces["width"] == "fixed" else -1.0
    score += _agree(has_lexicon, bool(pieces["lexicon"]))
    return score


def score_programs(programs: list[Prog], supports: list[str], facts: dict | None = None) -> list[float]:
    facts = facts if facts is not None else visible_facts(supports)
    return [match_score(facts, candidate_pieces(prog), supports) for prog in programs]


def choose(scores: list[float], mask: list[bool]) -> int:
    best = None
    index = 0
    for cursor, (score, live) in enumerate(zip(scores, mask)):
        if not live:
            continue
        if best is None or score > best:
            best = score
            index = cursor
    return index
