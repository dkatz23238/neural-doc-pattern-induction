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
from pattern_induction.lexicon import matching_lexicons
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
    """Exact, then a bounded width, then an open width.

    The observed pair stays first so a product of fields still contains the
    support-identifiable program. Open means a width up to 128: the executor
    treats a class bound as a real maximum, and ``OPEN`` is only 8.
    """
    lo = max(1, min(lo, hi))
    hi = max(lo, hi)
    ordered = [(lo, hi)]

    def add(pair: tuple[int, int]) -> None:
        left, right = pair
        left = max(1, left)
        right = min(128, max(left, right))
        if (left, right) not in ordered:
            ordered.append((left, right))

    # Open sits in the first four so a product of fields can widen them together.
    # One step shorter is the bound a sample of a digit run most often missed.
    if lo > 1:
        add((lo - 1, hi))
    add((1, hi))
    add((lo, 128))
    add((1, 128))
    add((lo, min(128, hi + 1)))
    add((max(1, lo - 1), min(128, hi + 1)))
    if lo == hi and lo * 2 <= 128:
        add((lo, lo * 2))
        add((lo * 2, lo * 2))
    add((lo, max(hi, OPEN)))
    add((1, max(hi, OPEN)))
    return ordered


def _snap(minimum: int, maximum: int, step: int) -> tuple[int, int]:
    lo = (minimum // step) * step
    hi = (maximum // step) * step + step - 1
    return lo, hi


def _intervals(minimum: int, maximum: int, width: int) -> list[tuple[int, int]]:
    """Hull, then the snap or convention a product will actually cross.

    Only the first few intervals of each field are crossed together. A wide
    field therefore leads with its parent grid and the one-step extension of
    that grid (a year hull of 1990–1996 leads with 1900–2099, not with the
    tighter decade). A short field leads with the tightest containing
    convention, so a day hull still meets 1–31 inside that same cross.
    """
    ordered: list[tuple[int, int]] = [(minimum, maximum)]

    def add(item: tuple[int, int]) -> None:
        lo, hi = item
        if lo < 0:
            lo = 0
        if hi < maximum or lo > minimum or hi > 100000:
            return
        if (lo, hi) not in ordered:
            ordered.append((lo, hi))

    if width >= 3:
        step = 10 ** (width - 2)
        lo, hi = _snap(minimum, maximum, step)
        add((lo, hi))
        add((lo, hi + step))
        add((lo - step, hi))
        fine = max(10, step // 10)
        flo, fhi = _snap(minimum, maximum, fine)
        add((flo, fhi))
        add((flo, fhi + fine))
    conventions = [
        (lo, hi)
        for lo, hi in _CONVENTIONS
        if lo <= minimum and maximum <= hi and (lo < minimum or hi > maximum)
    ]
    conventions.sort(key=lambda item: (item[1] - item[0], item[0]))
    for item in conventions:
        add(item)
    snaps: list[tuple[int, int]] = []
    for step in (10, 100, 1000):
        if maximum < step and minimum < step:
            continue
        lo, hi = _snap(minimum, maximum, step)
        snaps.append((lo, hi))
        snaps.append((lo, hi + step))
        snaps.append((max(0, lo - step), hi))
    snaps.sort(key=lambda item: (item[1] - item[0], item[0]))
    for item in snaps:
        add(item)
    if width >= 1:
        span = 10**width - 1
        if maximum <= span:
            add((0, span))
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
    # One shorter string than the observed width. It sits just after the first
    # four intervals, so the four-way cross that dates already use is unchanged
    # and a five-way cross can still see the shorter bound.
    if nodes and lo > 1:
        nodes.insert(min(4, len(nodes)), Prog("CLS", cls="digit", lo=lo - 1, hi=hi))
    nodes.append(Prog("CLS", cls="digit", lo=max(1, lo), hi=max(lo, hi)))
    for bound_lo, bound_hi in _bounds(lo, hi):
        nodes.append(Prog("CLS", cls="digit", lo=bound_lo, hi=bound_hi))
    return nodes


def _set_nodes(spec: str, lo: int, hi: int) -> list[Prog]:
    chars = spec_chars(spec)
    nodes = [Prog("SET", spec=spec, lo=lo, hi=hi)]
    seen = {spec}
    families = []
    for family in _FAMILIES:
        family_chars = spec_chars(family)
        if not chars or not chars <= family_chars:
            continue
        widened = charset_spec(family_chars)
        if widened in seen:
            continue
        seen.add(widened)
        families.append(widened)
    if families:
        nodes.append(Prog("SET", spec=families[0], lo=lo, hi=hi))
    open_hi = 128 if hi < 128 else hi
    if families and (1, open_hi) != (lo, hi):
        nodes.append(Prog("SET", spec=families[0], lo=1, hi=open_hi))
    if open_hi != hi:
        nodes.append(Prog("SET", spec=spec, lo=lo, hi=open_hi))
        if families:
            nodes.append(Prog("SET", spec=families[0], lo=lo, hi=open_hi))
    for widened in families[1:]:
        nodes.append(Prog("SET", spec=widened, lo=lo, hi=hi))
    for bound_lo, bound_hi in _bounds(lo, hi):
        if (bound_lo, bound_hi) == (lo, hi):
            continue
        nodes.append(Prog("SET", spec=spec, lo=bound_lo, hi=bound_hi))
    return nodes


_CLASS_CHARS = {
    "digit": set(string.digits),
    "upper": set(string.ascii_uppercase),
    "lower": set(string.ascii_lowercase),
}


def _class_nodes(cls: str, lo: int, hi: int) -> list[Prog]:
    nodes = [Prog("CLS", cls=cls, lo=bound_lo, hi=bound_hi) for bound_lo, bound_hi in _bounds(lo, hi)[:1]]
    alphabet = _CLASS_CHARS.get(cls, set())
    if cls != "digit" and alphabet:
        # An open word is the usual latent class of a short letter token.
        word = {"lower": "0-9a-z_", "upper": "0-9A-Z_"}.get(cls)
        if word and alphabet < spec_chars(word):
            nodes.append(Prog("SET", spec=charset_spec(spec_chars(word)), lo=1, hi=128))
    if lo == hi and hi < 128:
        nodes.append(Prog("CLS", cls=cls, lo=lo, hi=hi + 1))
    if hi < 128:
        nodes.append(Prog("CLS", cls=cls, lo=lo, hi=128))
    nodes.extend(Prog("CLS", cls=cls, lo=bound_lo, hi=bound_hi) for bound_lo, bound_hi in _bounds(lo, hi)[1:])
    if cls == "digit" or not alphabet:
        return _dedupe_progs(nodes)
    seen = {node.spec for node in nodes if node.op == "SET"}
    for family in _FAMILIES:
        chars = spec_chars(family)
        if not alphabet < chars:
            continue
        widened = charset_spec(chars)
        if widened in seen:
            continue
        seen.add(widened)
        nodes.append(Prog("SET", spec=widened, lo=lo, hi=hi))
    return nodes


def _dedupe_progs(nodes: list[Prog]) -> list[Prog]:
    return _dedupe(nodes)


def _literal(prog: Prog, supports: list[str]) -> str:
    if prog.slot < 0 or prog.slot >= len(supports):
        return ""
    text = supports[prog.slot]
    if prog.start < 0 or prog.end > len(text) or prog.end <= prog.start:
        return ""
    return text[prog.start : prog.end]


def _first_family(spec: str) -> str | None:
    chars = spec_chars(spec)
    if not chars:
        return None
    for family in _FAMILIES:
        if chars <= spec_chars(family):
            widened = charset_spec(spec_chars(family))
            if widened != spec:
                return widened
            return None
    return None


def _joint_family(prog: Prog) -> Prog | None:
    """Widen every character set to its first superclass together.

    A product of fields often widens one set and leaves the others on the
    observed hull. A hex identifier is the superclass of every field at once.
    """
    changed = False

    def walk(node: Prog) -> Prog:
        nonlocal changed
        if node.op == "SET":
            family = _first_family(node.spec)
            if family:
                changed = True
                return _clone(node, spec=family)
            return node
        if node.children:
            return _clone(node, children=[walk(child) for child in node.children])
        return node

    widened = walk(prog)
    return widened if changed else None


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
    varying = sum(1 for group in groups if len(group) > 1)
    head_n = 5 if varying and 5**varying <= cap else 4
    head = [group[:head_n] if group else [Prog("COPY")] for group in groups]
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

        def pick(group: list[Prog], index: int, hold_alt: bool, hold_width: bool) -> Prog:
            if hold_alt and group[0].op == "ALT":
                return group[0]
            chosen = group[min(index, len(group) - 1)]
            # A one-step-longer class rides the same diagonal as a clock
            # convention. Keeping the observed width still lets the convention
            # through, and the longer class remains on the unheld diagonal.
            if hold_width and chosen.hi > group[0].hi:
                return group[0]
            return chosen

        for index in range(1, depth):
            rows.append(tuple(pick(group, index, False, False) for group in head))
            for hold_alt, hold_width in ((True, False), (False, True), (True, True)):
                held = tuple(pick(group, index, hold_alt, hold_width) for group in head)
                if held not in rows:
                    rows.append(held)
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
        # Keep the crossed field generalizations ahead of separator lifts.
        # Lifting every row first crowds the century/month/day cross out of the cap.
        chosen = []
        for items in _product(groups, cap):
            chosen.append(list(items))
            if len(chosen) >= cap:
                break
        rows = [_clone(prog, children=items) for items in chosen]
        for items in chosen:
            if len(rows) >= cap:
                break
            for lifted in _separator_lifts(items, supports):
                rows.append(Prog("SEQ", children=lifted))
                if len(rows) >= cap:
                    break
        return _dedupe(rows)[:cap]
    if prog.op in {"OPT", "REPEAT"} and prog.children:
        inner = _node_choices(prog.children[0], supports, cap)
        rows = []
        # The observed count, paired with each inner generalization, stays at the
        # front. Otherwise a repeated field widens its count before its range,
        # and the joint range never shares the original count.
        for child in inner:
            if prog.op == "OPT":
                rows.append(_clone(prog, children=[child]))
            else:
                rows.append(_clone(prog, children=[child], lo=prog.lo, hi=prog.hi))
            if len(rows) >= cap:
                break
        if prog.op == "REPEAT":
            for child in inner:
                for lo, hi in _bounds(prog.lo, prog.hi):
                    if (lo, hi) == (prog.lo, prog.hi):
                        continue
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


def _seq(children: list[Prog]) -> Prog:
    nodes = [child for child in children if child is not None and not (child.op == "COPY" and child.end <= child.start and child.slot == 0 and not child.spec)]
    if not nodes:
        return Prog("SEQ")
    if len(nodes) == 1:
        return nodes[0]
    return Prog("SEQ", children=nodes)


def _nest_alt(nodes: list[Prog]) -> Prog:
    acc = nodes[0]
    for node in nodes[1:]:
        acc = Prog("ALT", children=[acc, node])
    return acc


def _copy_span(slot: int, start: int, end: int) -> Prog:
    return Prog("COPY", slot=slot, start=start, end=end)


def _raw_tokenize(text: str) -> list[list]:
    tokens: list[list] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isdigit():
            kind = "digit"
            end = index + 1
            while end < len(text) and text[end].isdigit():
                end += 1
        elif char.isupper():
            kind = "upper"
            end = index + 1
            while end < len(text) and text[end].isupper():
                end += 1
        elif char.islower():
            kind = "lower"
            end = index + 1
            while end < len(text) and text[end].islower():
                end += 1
        else:
            kind = "mark"
            end = index + 1
            while end < len(text) and text[end] == char:
                end += 1
        tokens.append([kind, text[index:end], index, end])
        index = end
    return tokens


def _coalesce(tokens: list[list]) -> list[list]:
    """Join a capital with its tail, and join underscore words. Keep 2025Q4 split."""
    merged: list[list] = []
    index = 0
    while index < len(tokens):
        kind, text, start, end = tokens[index]
        if kind == "upper" and len(text) == 1 and index + 1 < len(tokens) and tokens[index + 1][0] == "lower":
            nxt = tokens[index + 1]
            kind, text, end = "title", text + nxt[1], nxt[3]
            index += 2
            if index < len(tokens) and tokens[index][0] == "digit":
                nxt = tokens[index]
                text, end, kind = text + nxt[1], nxt[3], "word"
                index += 1
            merged.append([kind, text, start, end])
            continue
        if kind == "lower" and index + 1 < len(tokens) and tokens[index + 1][0] == "digit":
            nxt = tokens[index + 1]
            merged.append(["word", text + nxt[1], start, nxt[3]])
            index += 2
            continue
        merged.append([kind, text, start, end])
        index += 1
    joined: list[list] = []
    index = 0
    while index < len(merged):
        kind = merged[index][0]
        if (
            index + 2 < len(merged)
            and merged[index + 1][0] == "mark"
            and merged[index + 1][1] == "_"
            and kind in {"lower", "upper", "digit", "word"}
            and merged[index + 2][0] in {"lower", "upper", "digit", "word"}
        ):
            text = merged[index][1] + "_" + merged[index + 2][1]
            start, end = merged[index][2], merged[index + 2][3]
            index += 3
            while (
                index + 1 < len(merged)
                and merged[index][0] == "mark"
                and merged[index][1] == "_"
                and index + 1 < len(merged)
                and merged[index + 1][0] in {"lower", "upper", "digit", "word"}
            ):
                text += "_" + merged[index + 1][1]
                end = merged[index + 1][3]
                index += 2
            joined.append(["word", text, start, end])
            continue
        joined.append(merged[index])
        index += 1
    return joined


def _tokenize(text: str) -> list[list]:
    tokens = _coalesce(_raw_tokenize(text))
    for token in tokens:
        if token[0] == "lower":
            token[0] = "word"
    return tokens


def _signature(tokens: list[list]) -> tuple:
    return tuple(("mark", token[1]) if token[0] == "mark" else token[0] for token in tokens)


def _observed_set(texts: list[str]) -> Prog:
    chars = {char for text in texts for char in text}
    lo = min(len(text) for text in texts)
    hi = max(len(text) for text in texts)
    return Prog("SET", spec=charset_spec(chars), lo=lo, hi=hi)


def _digit_node(texts: list[str]) -> Prog:
    lo = min(len(text) for text in texts)
    hi = max(len(text) for text in texts)
    # A leading zero is a digit string, not an integer. Variable-width integers
    # reject it, so the field would not even accept the supports.
    if any(len(text) > 1 and text.startswith("0") for text in texts):
        return Prog("CLS", cls="digit", lo=lo, hi=hi)
    values = [int(text) for text in texts if text.isdigit()]
    if not values:
        return Prog("CLS", cls="digit", lo=lo, hi=hi)
    return Prog("INT", spec=f"{min(values)}:{max(values)}", lo=lo, hi=hi)


def _letter_node(texts: list[str], kind: str) -> Prog:
    lo = min(len(text) for text in texts)
    hi = max(len(text) for text in texts)
    if all(text.isalpha() and text.islower() for text in texts):
        return Prog("CLS", cls="lower", lo=lo, hi=hi)
    if all(text.isalpha() and text.isupper() for text in texts):
        return Prog("CLS", cls="upper", lo=lo, hi=hi)
    return _observed_set(texts)


def _title_node(texts: list[str]) -> Prog:
    tails = [text[1:] for text in texts if text]
    lo = min((len(tail) for tail in tails), default=1)
    hi = max((len(tail) for tail in tails), default=1)
    lo = max(1, lo)
    return Prog(
        "SEQ",
        children=[
            Prog("CLS", cls="upper", lo=1, hi=1),
            Prog("CLS", cls="lower", lo=lo, hi=max(lo, hi)),
        ],
    )


def _alt_of(tokens_at: list[list]) -> Prog | None:
    seen = set()
    nodes = []
    for token in tokens_at:
        if token[1] in seen:
            continue
        if len(token[1]) > 16:
            return None
        seen.add(token[1])
        nodes.append(_copy_span(0, 0, 0) if False else _copy_span_from(token))
    if len(nodes) < 2 or len(nodes) > 8:
        return None
    return _nest_alt(nodes)


def _copy_span_from(token: list) -> Prog:
    return _copy_span(token[4], token[2], token[3])


def _tag_slots(token_rows: list[list[list]]) -> None:
    for slot, row in enumerate(token_rows):
        for token in row:
            if len(token) == 4:
                token.append(slot)
            else:
                token[4] = slot


def _open_token(tokens: list[list], kind: str) -> Prog:
    texts = [token[1] for token in tokens]
    if kind == "digit" or (kind == "word" and all(text.isdigit() for text in texts)):
        return _digit_node(texts)
    if kind == "title":
        return _title_node(texts)
    if kind in {"lower", "upper", "word"}:
        return _letter_node(texts, kind)
    return _observed_set(texts)


def _column_options(rows: list[list[list]], index: int) -> list[Prog]:
    tokens = [row[index] for row in rows]
    kind = tokens[0][0]
    texts = [token[1] for token in tokens]
    if kind == "mark":
        return [_copy_span_from(tokens[0])]
    if len(set(texts)) == 1:
        # One observed token can still be a sample of its class.
        return [_copy_span_from(tokens[0]), _open_token(tokens, kind)]
    options: list[Prog] = []
    if kind == "digit" or (kind == "word" and all(text.isdigit() for text in texts)):
        options.append(_digit_node(texts))
        prefix = texts[0]
        for text in texts[1:]:
            while prefix and not text.startswith(prefix):
                prefix = prefix[:-1]
        if len(prefix) >= 2 and prefix.isdigit() and all(len(text) > len(prefix) for text in texts):
            rest = [text[len(prefix) :] for text in texts]
            if all(part.isdigit() for part in rest):
                options.append(_seq([_copy_span_from(_prefix_token(tokens[0], prefix)), _digit_node(rest)]))
    elif kind == "title":
        options.append(_title_node(texts))
    elif kind in {"lower", "upper", "word"}:
        options.append(_letter_node(texts, kind))
    else:
        options.append(_observed_set(texts))
    distinct = len(set(texts))
    if 2 <= distinct < len(texts):
        alt = _alt_of(tokens)
        if alt is not None:
            options.append(alt)
    return options


def _prefix_token(token: list, prefix: str) -> list:
    return ["digit", prefix, token[2], token[2] + len(prefix), token[4]]


def _open_choices(rows: list[list[list]]) -> list[list[Prog]]:
    return [_column_options(rows, index) for index in range(len(rows[0]))]


def _programs_from_options(options: list[list[Prog]], supports: list[str]) -> list[Prog]:
    if not options:
        return []
    base = [group[0] for group in options]
    programs = [_seq(base)]
    for place, group in enumerate(options):
        if len(group) < 2:
            continue
        swapped = list(base)
        swapped[place] = group[1]
        programs.append(_seq(swapped))
    programs.extend(_collapse_repeats(base, supports))
    for combo in [base, *(_swap(base, place, group[1]) for place, group in enumerate(options) if len(group) > 1)]:
        programs.extend(_optional_edges(combo))
    return programs


def _swap(base: list[Prog], place: int, node: Prog) -> list[Prog]:
    swapped = list(base)
    swapped[place] = node
    return swapped


def _same_mark(left: Prog, right: Prog, supports: list[str]) -> bool:
    if left.op != "COPY" or right.op != "COPY":
        return False
    literal = _literal(left, supports)
    return bool(literal) and literal == _literal(right, supports)


def _field_kind(node: Prog) -> str:
    if node.op == "INT":
        return "int"
    if node.op == "CLS":
        return "cls:" + node.cls
    if node.op == "SET":
        return "set"
    if node.op == "SEQ":
        return "seq:" + ",".join(_field_kind(child) for child in node.children)
    return node.op


def _merge_fields(nodes: list[Prog]) -> Prog:
    if all(node.op == "INT" and ":" in node.spec for node in nodes):
        lows, highs = [], []
        for node in nodes:
            left, right = (int(part) for part in node.spec.split(":"))
            lows.append(left)
            highs.append(right)
        lo = min(node.lo for node in nodes)
        hi = max(node.hi for node in nodes)
        return Prog("INT", spec=f"{min(lows)}:{max(highs)}", lo=lo, hi=hi)
    if all(node.op == "CLS" and node.cls == nodes[0].cls for node in nodes):
        return Prog("CLS", cls=nodes[0].cls, lo=min(node.lo for node in nodes), hi=max(node.hi for node in nodes))
    chars: set[str] = set()
    for node in nodes:
        if node.op == "SET" and node.spec:
            chars |= spec_chars(node.spec)
        elif node.op == "CLS":
            chars |= _CLASS_CHARS.get(node.cls, set())
        elif node.op == "INT":
            chars |= set(string.digits)
    if not chars:
        return nodes[0]
    return Prog(
        "SET",
        spec=charset_spec(chars),
        lo=min(node.lo for node in nodes),
        hi=max(node.hi for node in nodes),
    )


def _collapse_repeats(children: list[Prog], supports: list[str] | None = None) -> list[Prog]:
    """A field joined by one mark, several times, can be one repeated field."""
    if supports is None or len(children) < 5 or children[1].op != "COPY":
        return []
    count = 0
    index = 1
    fields = [children[0]]
    while index + 1 < len(children):
        sep = children[index]
        nxt = children[index + 1]
        if sep.op != "COPY" or not _same_mark(sep, children[1], supports):
            break
        if children[0].op not in {"INT", "SET", "CLS"} or nxt.op not in {"INT", "SET", "CLS"}:
            break
        fields.append(nxt)
        count += 1
        index += 2
    if count < 2 or index != len(children):
        return []
    merged = _merge_fields(fields)
    unit = _seq([children[1], merged])
    exact = _seq([merged, Prog("REPEAT", lo=count, hi=count, children=[unit])])
    opened = _seq([merged, Prog("REPEAT", lo=1, hi=max(count, OPEN), children=[unit])])
    return [exact, opened]


def _wrap_leading(children: list[Prog]) -> list[Prog] | None:
    if len(children) < 2 or children[0].op not in {"COPY", "ALT", "SEQ"}:
        return None
    end = 1
    if len(children) > 2 and children[1].op == "COPY":
        end = 2
    if end >= len(children):
        return None
    return [Prog("OPT", children=[_seq(children[:end])]), *children[end:]]


def _optional_edges(children: list[Prog]) -> list[Prog]:
    wrapped = _wrap_leading(children)
    if not wrapped:
        return []
    return [_seq(wrapped)]


def _aligned_programs(supports: list[str]) -> list[Prog]:
    rows = [_tokenize(text) for text in supports]
    if any(not row for row in rows):
        return []
    _tag_slots(rows)
    if len({_signature(row) for row in rows}) != 1:
        return []
    return _programs_from_options(_open_choices(rows), supports)


def _columns(token_rows: list[list[list]]) -> list[Prog]:
    if not token_rows or any(len(row) != len(token_rows[0]) for row in token_rows):
        return []
    _tag_slots(token_rows)
    return [group[0] for group in _open_choices(token_rows)]


def _optional_shape_programs(supports: list[str]) -> list[Prog]:
    """An affix or an internal token present in only some supports can be optional."""
    rows = [_tokenize(text) for text in supports]
    if any(not row for row in rows):
        return []
    _tag_slots(rows)
    signatures = [_signature(row) for row in rows]
    unique = list(dict.fromkeys(signatures))
    programs = _variable_repeats(rows)
    if len(unique) != 2:
        return programs
    left, right = unique
    short, long = (left, right) if len(left) < len(right) else (right, left)
    short_rows = [row for row, sig in zip(rows, signatures) if sig == short]
    long_rows = [row for row, sig in zip(rows, signatures) if sig == long]
    if long[len(long) - len(short) :] == short:
        prefix = [row[: len(row) - len(short)] for row in long_rows]
        shared = [row[-len(short) :] for row in rows]
        head = _columns(prefix)
        tail = _columns(shared)
        if head and tail:
            programs.append(_seq([Prog("OPT", children=[_seq(head)]), *tail]))
    if long[: len(short)] == short:
        suffix = [row[len(short) :] for row in long_rows]
        shared = [row[: len(short)] for row in rows]
        head = _columns(shared)
        tail = _columns(suffix)
        if head and tail:
            programs.append(_seq([*head, Prog("OPT", children=[_seq(tail)])]))
    extra = len(long) - len(short)
    for at in range(len(short) + 1):
        if long[:at] != short[:at] or long[at + extra :] != short[at:]:
            continue
        if extra <= 0:
            continue
        left_rows = [row[:at] for row in rows]
        inserted = [row[at : at + extra] for row in long_rows]
        right_rows = [row[at + extra :] if sig == long else row[at:] for row, sig in zip(rows, signatures)]
        parts = []
        if at:
            parts.extend(_columns(left_rows))
        mid = _columns(inserted)
        if not mid:
            continue
        parts.append(Prog("OPT", children=[_seq(mid)]))
        parts.extend(_columns(right_rows))
        programs.append(_seq(parts))
        break
    return programs


def _split_pair_tail(tokens: list[list]) -> tuple[list[list], list[list[list]]] | None:
    best = None
    for start in range(len(tokens)):
        rest = tokens[start:]
        if len(rest) < 2 or len(rest) % 2:
            continue
        unit = rest[:2]
        if unit[0][0] != "mark":
            continue
        pieces = [rest[index : index + 2] for index in range(0, len(rest), 2)]
        if any(piece[0][0] != unit[0][0] or piece[0][1] != unit[0][1] or piece[1][0] != unit[1][0] for piece in pieces):
            continue
        if best is None or len(pieces) > len(best[1]):
            best = (tokens[:start], pieces)
    return best


def _variable_repeats(rows: list[list[list]]) -> list[Prog]:
    parsed = []
    for row in rows:
        split = _split_pair_tail(row)
        if split is None or not split[1]:
            return []
        parsed.append(split)
    if len({_signature(head) for head, _units in parsed}) != 1:
        return []
    if len({_signature(units[0]) for _head, units in parsed}) != 1:
        return []
    counts = [len(units) for _head, units in parsed]
    if max(counts) < 2 or len(set(counts)) < 2:
        return []
    head = _columns([item[0] for item in parsed])
    unit_rows = [piece for _head, units in parsed for piece in units]
    unit = _columns(unit_rows)
    if not unit:
        return []
    low, high = min(counts), max(counts)
    exact = Prog("REPEAT", lo=low, hi=high, children=[_seq(unit)])
    opened = Prog("REPEAT", lo=1, hi=max(high, OPEN), children=[_seq(unit)])
    return [_seq([*head, exact]), _seq([*head, opened])]


_AMOUNT = None


def _amount_pattern():
    global _AMOUNT
    if _AMOUNT is None:
        import re

        _AMOUNT = re.compile(
            r"^(?P<head>\D*)(?P<lead>\d{1,3})(?P<groups>(?:(?P<sep>[,.' \u00a0_])\d{3})*)(?P<frac>(?P<fsep>[.,])\d+)?(?P<tail>\D*)$"
        )
    return _AMOUNT


def _parse_amount(text: str) -> dict | None:
    match = _amount_pattern().fullmatch(text)
    if match is None:
        return None
    groups = match.group("groups") or ""
    seps = {char for char in groups if not char.isdigit()}
    if len(seps) > 1:
        return None
    return {
        "head": match.group("head"),
        "tail": match.group("tail"),
        "sep": next(iter(seps)) if seps else "",
        "lead": match.group("lead"),
        "frac": match.group("frac") or "",
        "fsep": match.group("fsep") or "",
    }


def _find_literal(supports: list[str], literal: str) -> Prog | None:
    if literal == "":
        return None
    for slot, text in enumerate(supports):
        at = text.find(literal)
        if at >= 0:
            return _copy_span(slot, at, at + len(literal))
    return None


def _thousands_programs(supports: list[str]) -> list[Prog]:
    """A thousands group is a 1–3 digit head plus zero or more fixed groups of three."""
    parsed = [_parse_amount(text) for text in supports]
    if any(row is None for row in parsed):
        return []
    rows = [row for row in parsed if row is not None]
    if not any(row["sep"] for row in rows):
        return []
    if len({row["head"] for row in rows}) != 1 or len({row["tail"] for row in rows}) != 1:
        return []
    seps = {row["sep"] for row in rows if row["sep"]}
    if len(seps) != 1:
        return []
    fracs = {row["frac"] for row in rows}
    if any(row["frac"] for row in rows) and any(not row["frac"] for row in rows):
        return []
    if len({row["fsep"] for row in rows}) != 1 or len({len(row["frac"]) for row in rows}) != 1:
        return []
    sep = next(iter(seps))
    if rows[0]["fsep"] and sep == rows[0]["fsep"]:
        return []
    children: list[Prog] = []
    head = _find_literal(supports, rows[0]["head"])
    if head is not None:
        children.append(head)
    lead_hi = max(3, max(len(row["lead"]) for row in rows))
    lead_lo = min(len(row["lead"]) for row in rows)
    children.append(Prog("CLS", cls="digit", lo=min(lead_lo, 1), hi=lead_hi))
    sep_node = _find_literal(supports, sep)
    if sep_node is None:
        return []
    group = _seq([sep_node, Prog("CLS", cls="digit", lo=3, hi=3)])
    children.append(Prog("OPT", children=[Prog("REPEAT", lo=1, hi=OPEN, children=[group])]))
    if rows[0]["frac"]:
        mark = _find_literal(supports, rows[0]["fsep"])
        width = len(rows[0]["frac"]) - 1
        if mark is None or width < 1:
            return []
        children.append(mark)
        children.append(Prog("CLS", cls="digit", lo=width, hi=width))
    tail = _find_literal(supports, rows[0]["tail"])
    if tail is not None:
        children.append(tail)
    return [_seq(children)]


def _peel_field(supports: list[str], start: int, end: int) -> list[Prog]:
    pieces: list[Prog] = []
    index = start
    while index < end:
        if all(text[index] == supports[0][index] for text in supports):
            stop = index + 1
            while stop < end and all(text[stop] == supports[0][stop] for text in supports):
                stop += 1
            pieces.append(_copy_span(0, index, stop))
            index = stop
            continue
        stop = index + 1
        while stop < end and not all(text[stop] == supports[0][stop] for text in supports):
            stop += 1
        columns = [text[index:stop] for text in supports]
        if all(column.isdigit() for column in columns):
            pieces.append(_digit_node(columns))
        else:
            pieces.append(_observed_set(columns))
        index = stop
    return pieces


def _equal_length_programs(supports: list[str]) -> list[Prog]:
    if len({len(text) for text in supports}) != 1:
        return []
    width = len(supports[0])
    children: list[Prog] = []
    index = 0
    fields = 0
    while index < width:
        if all(not text[index].isalnum() and text[index] == supports[0][index] for text in supports):
            stop = index + 1
            while stop < width and all(not text[stop].isalnum() and text[stop] == supports[0][stop] for text in supports):
                stop += 1
            children.append(_copy_span(0, index, stop))
            index = stop
            continue
        stop = index + 1
        while stop < width and not (
            stop < width and all(not text[stop].isalnum() and text[stop] == supports[0][stop] for text in supports)
        ):
            stop += 1
        children.extend(_peel_field(supports, index, stop))
        fields += 1
        index = stop
    if fields < 2:
        return []
    programs = [_seq(children)]
    opened = _open_last(children)
    if opened is not None:
        programs.append(_seq(opened))
    programs.extend(_collapse_repeats(children, supports))
    return programs


def _open_last(children: list[Prog]) -> list[Prog] | None:
    """The last field may be longer than the clipped support window."""
    for index in range(len(children) - 1, -1, -1):
        node = children[index]
        if node.op not in {"SET", "CLS", "INT"}:
            continue
        hi = min(128, max(node.hi * 2, node.hi + 4))
        if hi == node.hi:
            return None
        opened = list(children)
        opened[index] = _clone(node, hi=hi)
        return opened
    return None


def _charset_tails(supports: list[str]) -> list[Prog]:
    """One alphanumeric blob, possibly with one optional tail of the extra width."""
    if not supports or any(not text.isalnum() for text in supports):
        return []
    chars = {char for text in supports for char in text}
    spec = charset_spec(chars)
    lengths = sorted({len(text) for text in supports})
    lo, hi = lengths[0], lengths[-1]
    programs = [Prog("SET", spec=spec, lo=lo, hi=hi)]
    if hi > lo and all(len(text) in {lo, hi} for text in supports) and hi - lo <= 4:
        programs.append(
            _seq(
                [
                    Prog("SET", spec=spec, lo=lo, hi=lo),
                    Prog("OPT", children=[Prog("SET", spec=spec, lo=hi - lo, hi=hi - lo)]),
                ]
            )
        )
    return programs


def _lexicon_programs(supports: list[str]) -> list[Prog]:
    """Replace an observed field with a frozen closed set that contains it.

    The set is not read off the benchmark. Two distinct observed tokens have
    to sit inside the set, and the set has to name something they did not show.
    """
    if len(supports) < 2:
        return []
    programs: list[Prog] = []
    for name in matching_lexicons(set(supports)):
        programs.append(Prog("LEX", spec=name))
    rows = [_tokenize(text) for text in supports]
    if rows and all(rows) and len({_signature(row) for row in rows}) == 1:
        _tag_slots(rows)
        base = _columns(rows)
        if len(base) == len(rows[0]):
            for index in range(len(rows[0])):
                observed = {row[index][1] for row in rows}
                for name in matching_lexicons(observed):
                    children = list(base)
                    children[index] = Prog("LEX", spec=name)
                    programs.append(_seq(children))
    if all("." in text and not text.endswith(".") for text in supports):
        extensions = {text.rsplit(".", 1)[1] for text in supports}
        stems = [text.rsplit(".", 1)[0] for text in supports]
        if all(stems) and all(token.isalnum() for token in extensions):
            chars = {char for stem in stems for char in stem}
            dot = _find_literal(supports, ".")
            if chars and dot is not None:
                stem = Prog(
                    "SET",
                    spec=charset_spec(chars),
                    lo=min(len(text) for text in stems),
                    hi=max(len(text) for text in stems),
                )
                for name in matching_lexicons(extensions):
                    programs.append(_seq([stem, dot, Prog("LEX", spec=name)]))
    return programs


def factor_supports(supports: list[str]) -> list[Prog]:
    """Field programs read off the supports, before latent widening.

    These are generic: shared marks become fields, a thousands group becomes a
    head plus a repeated group of three, a repeated field becomes a quantifier,
    and a token seen in only some supports becomes optional. Closed sets are
    added beside these, from the frozen lexicon bank, not from this factorizer.
    """
    if len(supports) < 2:
        return []
    rows: list[Prog] = []
    rows.extend(_thousands_programs(supports))
    rows.extend(_optional_shape_programs(supports))
    rows.extend(_aligned_programs(supports))
    rows.extend(_equal_length_programs(supports))
    rows.extend(_charset_tails(supports))
    kept = []
    for prog in _dedupe(rows):
        if all(execute(prog, text, supports) for text in supports):
            kept.append(prog)
    return kept


def expand_programs(programs: list[Prog], supports: list[str], cap: int = 480) -> list[Prog]:
    """Union of latent rewrites.

    Every seed is kept. Each seed then receives the same share of its
    group's budget, so the first factored program cannot spend the share
    of a later seed. When the caller passes chart programs, factored
    supports and that chart each keep half of the cap. With no chart
    seeds the factored programs keep the whole cap. A joint superclass
    and a lexicon completion are added beside that budget.
    """
    seeds: list[Prog] = []
    seen = set()

    def add(prog: Prog | None) -> bool:
        if prog is None:
            return False
        key = describe(prog)
        if key in seen:
            return False
        seen.add(key)
        seeds.append(prog)
        return True

    factor: list[Prog] = []
    for prog in factor_supports(supports):
        if add(prog):
            factor.append(prog)
    incoming: list[Prog] = []
    for prog in programs:
        if add(prog):
            incoming.append(prog)
    joints: list[Prog] = []
    for prog in list(seeds):
        joint = _joint_family(prog)
        if add(joint):
            joints.append(joint)
    rows: list[Prog] = []
    emitted = set()

    def emit(candidate: Prog) -> None:
        key = describe(candidate)
        if key in emitted:
            return
        emitted.add(key)
        rows.append(candidate)

    def pour(group: list[Prog], budget: int) -> None:
        if not group or budget <= 0:
            return
        for prog in group:
            emit(prog)
        per = max(1, budget // len(group))
        for prog in group:
            for candidate in expand_program(prog, supports, per):
                emit(candidate)

    if incoming and factor:
        pour(factor, cap // 2)
        pour(incoming, cap // 2)
    elif incoming:
        pour(incoming, cap)
    else:
        pour(factor, cap)
    pour(joints, max(cap // 5, 1))
    # Lexicon completions keep their own budget so they do not evict a structural rewrite.
    extra = 0
    for prog in _lexicon_programs(supports):
        if not all(execute(prog, text, supports) for text in supports):
            continue
        for candidate in [prog, *expand_program(prog, supports, 48)]:
            key = describe(candidate)
            if key in emitted:
                continue
            emitted.add(key)
            rows.append(candidate)
            extra += 1
            if extra >= 80:
                return rows
    return rows
