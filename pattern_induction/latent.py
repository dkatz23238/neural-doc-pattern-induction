"""Latent candidates that the supports do not literally demonstrate.

The version space already contains what the examples justify. These rewrites
add the nearby generalizations a language might still have been: a wider
bound, a natural character superclass, a conventional numeric interval, an
open repeat, or an alternation lifted to a class. Nothing here is a benchmark
answer. Every candidate is still an executable program, and it is kept only
when it accepts the supports.
"""

from __future__ import annotations

import string

from pattern_induction.charset import charset_spec, spec_chars
from pattern_induction.program import OPEN, Prog, describe, execute

# Conventional intervals a numeric field often means, independent of any benchmark.
# A candidate is used only when it contains every observed value.
_CONVENTIONS: tuple[tuple[int, int], ...] = (
    (0, 9),
    (1, 9),
    (0, 12),
    (1, 12),
    (0, 23),
    (0, 31),
    (1, 31),
    (0, 59),
    (0, 99),
    (1, 99),
    (0, 100),
    (0, 255),
)

# Natural alphabets. A hull is widened only to a family that contains it.
_FAMILIES: tuple[str, ...] = (
    "0-9",
    "A-Z",
    "a-z",
    "0-9A-F",
    "0-9a-f",
    "0-9A-Z",
    "0-9a-z",
    "0-9A-Za-z",
    "0-9A-Z_",
    "0-9a-z_",
    "0-9A-Za-z_",
    "0-9A-Za-z.-",
    "0-9A-Za-z._-",
    "0-9A-Za-z._%+\\-",
    "A-Za-z",
)


def _clone(prog: Prog, children: list[Prog] | None = None, **changes) -> Prog:
    data = {
        "op": prog.op,
        "slot": prog.slot,
        "start": prog.start,
        "end": prog.end,
        "cls": prog.cls,
        "lo": prog.lo,
        "hi": prog.hi,
        "spec": prog.spec,
        "children": list(prog.children if children is None else children),
    }
    data.update(changes)
    return Prog(**data)


def _bounds(lo: int, hi: int) -> list[tuple[int, int]]:
    """Exact, slack, bounded, and open widths around an observed quantifier."""
    lo = max(1, min(lo, hi))
    hi = max(lo, hi)
    found = {(lo, hi)}
    if lo == hi:
        width = lo
        found.add((1, width))
        found.add((max(1, width - 1), min(128, width + 1)))
        found.add((width, max(width, OPEN)))
        found.add((1, max(width, OPEN)))
        if width * 2 <= 128:
            found.add((width * 2, width * 2))
            found.add((width, width * 2))
    else:
        found.add((max(1, lo - 1), hi))
        found.add((lo, min(128, hi + 1)))
        found.add((max(1, lo - 1), min(128, hi + 1)))
        found.add((1, hi))
        found.add((lo, max(hi, OPEN)))
        found.add((1, max(hi, OPEN)))
    return [(a, b) for a, b in sorted(found) if 1 <= a <= b <= 128]


def _snap(minimum: int, maximum: int, step: int) -> tuple[int, int]:
    lo = (minimum // step) * step
    hi = (maximum // step) * step + step - 1
    return lo, hi


def _intervals(minimum: int, maximum: int, width: int) -> list[tuple[int, int]]:
    """Hull, then the snaps and conventions most likely to be the latent range.

    The order is the product order. The first few entries of every field are
    fully crossed, so a century and a month lattice are proposed together.
    """
    ordered: list[tuple[int, int]] = [(minimum, maximum)]
    snaps: list[tuple[int, int]] = []
    for step in (10, 100, 1000):
        if maximum < step and minimum < step:
            continue
        lo, hi = _snap(minimum, maximum, step)
        if hi > 10 ** max(width, 1) + step and width:
            continue
        snaps.append((lo, hi))
        snaps.append((lo, hi + step))
    snaps.sort(key=lambda item: (item[1] - item[0], item[0]))
    conventions = [
        (lo, hi)
        for lo, hi in _CONVENTIONS
        if lo <= minimum and maximum <= hi and (lo < minimum or hi > maximum)
    ]
    conventions.sort(key=lambda item: (item[1] - item[0], item[0]))
    full: list[tuple[int, int]] = []
    if width >= 1:
        span = 10**width - 1
        if maximum <= span:
            full.append((0, span))
    for item in snaps + conventions[:3] + full:
        if item not in ordered and item[0] <= minimum and item[1] >= maximum:
            ordered.append(item)
    return ordered[:8]


def _int_nodes(spec: str, lo: int, hi: int) -> list[Prog]:
    if ":" not in spec:
        return []
    minimum, maximum = (int(part) for part in spec.split(":"))
    width = lo if lo == hi else 0
    nodes = []
    seen = set()
    for start, end in _intervals(minimum, maximum, width):
        if width and end <= 10**width - 1:
            span = (width, width)
        else:
            low = 1 if start == 0 else len(str(start))
            high = max(len(str(end)), low, width)
            span = (low, high)
        key = (f"{start}:{end}", span)
        if key in seen:
            continue
        seen.add(key)
        nodes.append(Prog("INT", spec=f"{start}:{end}", lo=span[0], hi=span[1]))
    nodes.append(Prog("CLS", cls="digit", lo=max(1, lo), hi=max(lo, hi)))
    for bound_lo, bound_hi in _bounds(lo, hi):
        nodes.append(Prog("CLS", cls="digit", lo=bound_lo, hi=bound_hi))
    return nodes


def _set_nodes(spec: str, lo: int, hi: int) -> list[Prog]:
    chars = spec_chars(spec)
    nodes = [Prog("SET", spec=spec, lo=lo, hi=hi)]
    seen = {spec}
    for family in _FAMILIES:
        family_chars = spec_chars(family)
        if not chars or not chars <= family_chars:
            continue
        widened = charset_spec(family_chars)
        if widened in seen:
            continue
        seen.add(widened)
        nodes.append(Prog("SET", spec=widened, lo=lo, hi=hi))
    for bound_lo, bound_hi in _bounds(lo, hi):
        if (bound_lo, bound_hi) == (lo, hi):
            continue
        nodes.append(Prog("SET", spec=spec, lo=bound_lo, hi=bound_hi))
    return nodes


def _class_nodes(cls: str, lo: int, hi: int) -> list[Prog]:
    return [Prog("CLS", cls=cls, lo=bound_lo, hi=bound_hi) for bound_lo, bound_hi in _bounds(lo, hi)]


def _literal(prog: Prog, supports: list[str]) -> str:
    if prog.slot < 0 or prog.slot >= len(supports):
        return ""
    text = supports[prog.slot]
    if prog.start < 0 or prog.end > len(text) or prog.end <= prog.start:
        return ""
    return text[prog.start : prog.end]


def _class_name(char: str) -> str:
    if char.isdigit():
        return "digit"
    if char.isupper():
        return "upper"
    if char.islower():
        return "lower"
    return ""


def _alt_class(prog: Prog, supports: list[str]) -> Prog | None:
    """Lift an alternation of same-class literals to one class or set."""
    literals: list[str] = []

    def walk(node: Prog) -> bool:
        if node.op == "COPY":
            text = _literal(node, supports)
            if not text:
                return False
            literals.append(text)
            return True
        if node.op == "ALT":
            return all(walk(child) for child in node.children)
        return False

    if prog.op != "ALT" or not walk(prog) or len(literals) < 2:
        return None
    columns = list(zip(*literals)) if len({len(text) for text in literals}) == 1 else None
    if columns is not None and all(len({_class_name(ch) for ch in column}) == 1 and _class_name(column[0]) for column in columns):
        name = _class_name(columns[0][0])
        if all(_class_name(ch) == name for column in columns for ch in column):
            return Prog("CLS", cls=name, lo=len(literals[0]), hi=len(literals[0]))
    names = {_class_name(ch) for text in literals for ch in text}
    if "" in names or len(names) != 1:
        return None
    lo = min(len(text) for text in literals)
    hi = max(len(text) for text in literals)
    return Prog("CLS", cls=names.pop(), lo=lo, hi=hi)


def _separator_lifts(children: list[Prog], supports: list[str]) -> list[list[Prog]]:
    """A separator plus a field may be optional, or may repeat, even if the supports show it once."""
    lifts: list[list[Prog]] = []
    for index in range(len(children) - 1):
        left, right = children[index], children[index + 1]
        if left.op != "COPY" or right.op not in {"CLS", "SET", "INT"}:
            continue
        literal = _literal(left, supports)
        if not literal or any(ch.isalnum() for ch in literal):
            continue
        group = Prog("SEQ", children=[left, right])
        optional = children[:index] + [Prog("OPT", children=[group])] + children[index + 2 :]
        repeating = children[:index] + [Prog("OPT", children=[Prog("REPEAT", lo=1, hi=OPEN, children=[group])])] + children[index + 2 :]
        lifts.append(optional)
        lifts.append(repeating)
    return lifts


def _dedupe(programs: list[Prog]) -> list[Prog]:
    seen = set()
    rows = []
    for prog in programs:
        key = describe(prog)
        if key in seen:
            continue
        seen.add(key)
        rows.append(prog)
    return rows


def _product(groups: list[list[Prog]], cap: int) -> list[tuple[Prog, ...]]:
    """Cross the first few alternatives of every node, then single-node swaps.

    A full cross of long lists drops the joint generalization (a wider year
    and a wider month together). The head of each list is small enough to cross.
    """
    if not groups:
        return [()]
    head = [group[:4] if group else [Prog("COPY")] for group in groups]
    total = 1
    for group in head:
        total *= len(group)
    rows: list[tuple[Prog, ...]] = [()]
    if total <= cap:
        for group in head:
            rows = [prefix + (item,) for prefix in rows for item in group]
    else:
        rows = [tuple(group[0] for group in groups)]
        depth = max(len(group) for group in head)
        for index in range(1, depth):
            rows.append(tuple(group[min(index, len(group) - 1)] for group in head))
    seen = {tuple(describe(item) for item in row) for row in rows}
    identity = tuple(group[0] for group in groups)
    for place, group in enumerate(groups):
        for item in group[1:]:
            swapped = identity[:place] + (item,) + identity[place + 1 :]
            key = tuple(describe(part) for part in swapped)
            if key in seen:
                continue
            seen.add(key)
            rows.append(swapped)
            if len(rows) >= cap:
                return rows
    return rows[:cap]


def _node_choices(prog: Prog, supports: list[str], cap: int) -> list[Prog]:
    if prog.op == "SEQ":
        groups = [_node_choices(child, supports, cap) for child in prog.children]
        rows = [_clone(prog, children=list(items)) for items in _product(groups, cap)]
        if prog.children:
            for lifted in _separator_lifts(prog.children, supports):
                rows.append(Prog("SEQ", children=lifted))
        return _dedupe(rows)[:cap]
    if prog.op in {"OPT", "REPEAT"} and prog.children:
        inner = _node_choices(prog.children[0], supports, cap)
        rows = []
        for child in inner:
            if prog.op == "OPT":
                rows.append(_clone(prog, children=[child]))
            else:
                for lo, hi in _bounds(prog.lo, prog.hi):
                    rows.append(_clone(prog, children=[child], lo=lo, hi=hi))
                    if len(rows) >= cap:
                        break
            if len(rows) >= cap:
                break
        return _dedupe(rows)[:cap]
    if prog.op == "ALT":
        lifted = _alt_class(prog, supports)
        rows = [_clone(prog, children=[_node_choices(child, supports, 4)[0] for child in prog.children])]
        if lifted is not None:
            rows.append(lifted)
        return _dedupe(rows)[:cap]
    if prog.op == "INT":
        return _dedupe(_int_nodes(prog.spec, prog.lo, prog.hi))[:cap]
    if prog.op == "SET":
        return _dedupe(_set_nodes(prog.spec, prog.lo, prog.hi))[:cap]
    if prog.op == "CLS":
        return _class_nodes(prog.cls, prog.lo, prog.hi)[:cap]
    return [prog]


def expand_program(prog: Prog, supports: list[str], cap: int = 240) -> list[Prog]:
    """Generalizations of one program, including itself, that still accept the supports."""
    rows = []
    for candidate in _node_choices(prog, supports, cap):
        if all(execute(candidate, text, supports) for text in supports):
            rows.append(candidate)
        if len(rows) >= cap:
            break
    return _dedupe(rows)


_GAP_ORDER = (
    "unseen width/bound",
    "unseen class member",
    "unseen alternative",
    "unseen optional branch",
    "semantic numeric interval",
    "open quantifier",
    "other",
)


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[A-Za-z]+|\d+|[^A-Za-z0-9]+", text)


def _skeleton(text: str) -> tuple[str, ...]:
    rows = []
    for token in _tokens(text):
        if token.isdigit():
            rows.append("D")
        elif token.isalpha():
            rows.append("A")
        else:
            rows.append(token)
    return tuple(rows)


def _gold_open(gold: str) -> bool:
    import re

    return bool(re.search(r"(?<!\\)(?:\+|\*|\{[0-9]+,\})", gold))


def _gold_optional(gold: str) -> bool:
    """A real optional marker. Non-capturing ``(?:`` is not one."""
    return "?" in gold.replace("(?:", "")


def _gold_semantic(gold: str) -> bool:
    """A bounded numeric convention, not a full digit class."""
    import re

    return bool(
        re.search(
            r"0\[1-9\]|1\[0-2\]|2\[0-3\]|\[0-5\]|3\[01\]|\[12\]\\d|\[1-4\]|19\|20|100\|",
            gold,
        )
    )


def _gap_for_positive(text: str, supports: list[str], gold: str) -> str:
    """Why this hidden positive is outside the support-identifiable language."""
    import re

    shape = _skeleton(text)
    aligned = [support for support in supports if _skeleton(support) == shape]
    if not aligned:
        words = re.findall(r"[A-Za-z]{2,}", text)
        blob = "\n".join(supports)
        if any(word not in blob for word in words):
            return "unseen alternative"
        if _gold_optional(gold):
            return "unseen optional branch"
        if _gold_open(gold):
            return "open quantifier"
        return "unseen optional branch"
    reasons = set()
    observed: list[list[str]] = [_tokens(support) for support in aligned]
    tokens = _tokens(text)
    width = len(observed[0]) if observed else 0
    if any(len(row) != width for row in observed) or len(tokens) != width:
        if _gold_open(gold):
            return "open quantifier"
        return "unseen width/bound"
    for index, token in enumerate(tokens):
        column = [row[index] for row in observed if len(row) > index]
        if token.isdigit() and column and all(item.isdigit() for item in column):
            lengths = {len(item) for item in column}
            if len(token) not in lengths:
                reasons.add("unseen width/bound")
            values = [int(item) for item in column]
            if int(token) < min(values) or int(token) > max(values):
                reasons.add("semantic numeric interval" if _gold_semantic(gold) else "unseen class member")
        elif token.isalpha() and column and all(item.isalpha() for item in column):
            if token not in column:
                observed_chars = {ch for item in column for ch in item}
                if any(ch not in observed_chars for ch in token) and len({len(item) for item in column + [token]}) == 1:
                    reasons.add("unseen class member")
                else:
                    reasons.add("unseen alternative")
        elif token not in column:
            observed_chars = {ch for item in column for ch in item}
            if token and all(ch in observed_chars or not ch.isalnum() for ch in token):
                reasons.add("unseen width/bound")
            elif any(ch.isalnum() and ch not in observed_chars for ch in token):
                reasons.add("unseen class member")
    if not reasons:
        if _gold_open(gold):
            return "open quantifier"
        return "other"
    for name in _GAP_ORDER:
        if name in reasons:
            return name
    return "other"


def classify_gap(prog: Prog | None, supports: list[str], gold: str, hidden_positives: list[str]) -> str:
    """Primary ambiguity when the support-identifiable program misses the hidden gold."""
    if prog is None:
        return "other"
    missed = [text for text in hidden_positives if not execute(prog, text, supports)]
    if not missed:
        return "identifiable"
    labels = [_gap_for_positive(text, supports, gold) for text in missed[:8]]
    for name in _GAP_ORDER:
        if name in labels:
            return name
    return "other"


def expand_programs(programs: list[Prog], supports: list[str], cap: int = 480) -> list[Prog]:
    """Union of latent rewrites. The input programs stay in the set."""
    rows: list[Prog] = []
    seen = set()
    per = max(24, cap // max(len(programs), 1))
    for prog in programs:
        for candidate in [prog, *expand_program(prog, supports, per)]:
            key = describe(candidate)
            if key in seen:
                continue
            seen.add(key)
            rows.append(candidate)
            if len(rows) >= cap:
                return rows
    return rows
