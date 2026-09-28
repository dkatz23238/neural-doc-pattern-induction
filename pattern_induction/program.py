"""Programs whose semantics are executed, not learned.

A program is a small tree: copy a support span, match a character class,
sequence, repeat, optional, or alternation. The neural proposer predicts that
tree. Membership is this executor. Literals are copied from the support, so an
unseen delimiter is a pointer, not a vocabulary item.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from pattern_induction.compose import (
    Alternation,
    OptionalInternal,
    OptionalTail,
    PrefixAlt,
    PrefixClass,
    PrefixRepeat,
    RepeatNum,
    RepeatOptional,
    _BUILDERS,
)
from pattern_induction.membership import Episode
from pattern_induction.renaming import featurize
from pattern_induction.schema_dsl import Language, SchemaError

OPEN = 8
MAX_PROG = 12
MAX_LEN = 32
MAX_SUPPORT = 4
N_OP = 7
OP_COPY = 0
OP_CLS = 1
OP_SEQ = 2
OP_REPEAT = 3
OP_OPT = 4
OP_ALT = 5
OP_STOP = 6
OP_NAME = ("COPY", "CLS", "SEQ", "REPEAT", "OPT", "ALT", "STOP")
CLS_NAME = ("digit", "upper", "lower")
ROLE_COPY = 1
ROLE_DIGIT = 2
ROLE_UPPER = 3
ROLE_LOWER = 4
SCOPE_REPEAT = 1
SCOPE_OPT = 2
SCOPE_ALT = 3


@dataclass
class Prog:
    op: str
    slot: int = 0
    start: int = 0
    end: int = 0
    cls: str = "digit"
    lo: int = 1
    hi: int = 1
    children: list["Prog"] = field(default_factory=list)

    def nodes(self) -> int:
        return 1 + sum(child.nodes() for child in self.children)


@dataclass
class Task:
    family: str
    supports: list[str]
    program: Prog
    probe: list[tuple[str, bool]]


def class_ok(text: str, name: str) -> bool:
    if name == "digit":
        return text.isdigit()
    if name == "upper":
        return text.isalpha() and text.isupper()
    if name == "lower":
        return text.isalpha() and text.islower()
    return False


def _literal(prog: Prog, supports: list[str]) -> str:
    if prog.slot < 0 or prog.slot >= len(supports):
        return ""
    text = supports[prog.slot]
    if prog.start < 0 or prog.end > len(text) or prog.end <= prog.start:
        return ""
    return text[prog.start : prog.end]


def execute(prog: Prog, text: str, supports: list[str]) -> bool:
    """True when the program matches the whole string."""
    if not text:
        return False
    memo: dict[tuple[int, int], tuple[int, ...]] = {}

    def ends(node: Prog, index: int) -> tuple[int, ...]:
        key = (id(node), index)
        cached = memo.get(key)
        if cached is not None:
            return cached
        if node.op == "COPY":
            literal = _literal(node, supports)
            result = (index + len(literal),) if literal and text.startswith(literal, index) else ()
        elif node.op == "CLS":
            found = []
            lo = max(1, min(node.lo, node.hi))
            hi = max(lo, node.hi)
            for width in range(lo, hi + 1):
                chunk = text[index : index + width]
                if len(chunk) == width and class_ok(chunk, node.cls):
                    found.append(index + width)
            result = tuple(found)
        elif node.op == "OPT" and node.children:
            result = (index,) + ends(node.children[0], index)
        elif node.op == "ALT" and len(node.children) >= 2:
            result = ends(node.children[0], index) + ends(node.children[1], index)
        elif node.op == "SEQ" and node.children:
            positions: tuple[int, ...] = (index,)
            for child in node.children:
                nxt: list[int] = []
                for pos in positions:
                    nxt.extend(ends(child, pos))
                positions = tuple(dict.fromkeys(nxt))
                if not positions:
                    break
            result = positions
        elif node.op == "REPEAT" and node.children:
            hi = max(node.lo, min(node.hi, OPEN))
            found: list[int] = []
            positions = (index,)
            for count in range(1, hi + 1):
                nxt = []
                for pos in positions:
                    nxt.extend(ends(node.children[0], pos))
                positions = tuple(dict.fromkeys(nxt))
                if count >= node.lo:
                    found.extend(positions)
                if not positions:
                    break
            result = tuple(dict.fromkeys(found))
        else:
            result = ()
        memo[key] = result
        return result

    return len(text) in ends(prog, 0)


def accepts_all(prog: Prog, strings: list[str], supports: list[str]) -> bool:
    return all(execute(prog, text, supports) for text in strings)


def agrees(prog: Prog, supports: list[str], probe: list[tuple[str, bool]]) -> bool:
    return all(execute(prog, text, supports) == truth for text, truth in probe)


def to_regex(prog: Prog, supports: list[str]) -> str:
    """Render the program against these supports. The executor remains the judge."""

    def quant(lo: int, hi: int) -> str:
        if lo == hi:
            return "{" + str(lo) + "}"
        if hi >= OPEN:
            return "{" + str(lo) + ",}"
        return "{" + str(lo) + "," + str(hi) + "}"

    def piece(node: Prog) -> str:
        if node.op == "COPY":
            return re.escape(_literal(node, supports))
        if node.op == "CLS":
            body = {"digit": r"\d", "upper": r"[A-Z]", "lower": r"[a-z]"}.get(node.cls, r"\d")
            lo = max(1, min(node.lo, node.hi))
            hi = max(lo, node.hi)
            return body + quant(lo, hi)
        if node.op == "OPT" and node.children:
            return "(?:" + piece(node.children[0]) + ")?"
        if node.op == "ALT" and len(node.children) >= 2:
            return "(?:" + piece(node.children[0]) + "|" + piece(node.children[1]) + ")"
        if node.op == "SEQ":
            return "".join(piece(child) for child in node.children)
        if node.op == "REPEAT" and node.children:
            return "(?:" + piece(node.children[0]) + ")" + quant(node.lo, node.hi)
        return ""

    return "^" + piece(prog) + "$"


def describe(prog: Prog) -> str:
    if prog.op == "COPY":
        return f"COPY({prog.slot}:{prog.start}-{prog.end})"
    if prog.op == "CLS":
        return f"{prog.cls}{{{prog.lo},{prog.hi}}}"
    if prog.op == "REPEAT" and prog.children:
        return f"REP{prog.lo}-{prog.hi}({describe(prog.children[0])})"
    if prog.op == "OPT" and prog.children:
        return f"OPT({describe(prog.children[0])})"
    if prog.op == "ALT":
        return "ALT(" + ",".join(describe(child) for child in prog.children) + ")"
    if prog.op == "SEQ":
        return "SEQ(" + ",".join(describe(child) for child in prog.children) + ")"
    return "?"


def _find_span(supports: list[str], literal: str) -> tuple[int, int, int] | None:
    if not literal:
        return None
    for slot, text in enumerate(supports):
        at = text.find(literal)
        if at >= 0:
            return slot, at, at + len(literal)
    return None


def _copy(slot: int, start: int, end: int) -> Prog:
    return Prog("COPY", slot=slot, start=start, end=end)


def _cls(name: str, lo: int, hi: int) -> Prog:
    return Prog("CLS", cls=name, lo=lo, hi=hi)


def _seq(children: list[Prog]) -> Prog:
    if len(children) == 1:
        return children[0]
    return Prog("SEQ", children=children)


def gold_repeat(lang: RepeatNum, supports: list[str]) -> Prog | None:
    at = supports[0].find(lang.sep)
    if at < 0:
        return None
    unit = _seq([_copy(0, at, at + 1), _cls("digit", 1, 2)])
    return _seq([_cls("digit", 1, 2), Prog("REPEAT", lo=1, hi=OPEN, children=[unit])])


def gold_compose(family: str, lang, supports: list[str]) -> Prog | None:
    if family in {"repeat", "repeat_extra"}:
        return gold_repeat(lang, supports)
    if family == "primitive":
        body = "digit" if lang.body == "D" else "lower"
        leaf = _cls(body, lang.width, lang.width)
        if not lang.prefix:
            return leaf
        if not supports[0].startswith(lang.prefix):
            return None
        return _seq([_copy(0, 0, len(lang.prefix)), leaf])
    if family == "optional":
        if not supports[0].startswith(lang.prefix):
            return None
        hyphen = _find_span(supports, "-")
        if hyphen is None:
            return None
        return _seq([_copy(0, 0, len(lang.prefix)), Prog("OPT", children=[_copy(*hyphen)]), _cls("digit", 3, 3)])
    if family == "alternation":
        for slot, text in enumerate(supports):
            if text == lang.literal:
                return Prog("ALT", children=[_copy(slot, 0, len(text)), _cls("digit", 4, 4)])
        return None
    if family == "optional_internal":
        if not supports[0].startswith(lang.left) or not supports[0].endswith(lang.right):
            return None
        return _seq(
            [
                _copy(0, 0, len(lang.left)),
                Prog("OPT", children=[_cls("digit", lang.width, lang.width)]),
                _copy(0, len(supports[0]) - len(lang.right), len(supports[0])),
            ]
        )
    if family == "prefix_alt":
        if not supports[0].startswith(lang.prefix):
            return None
        literal = _find_span(supports, lang.literal)
        if literal is None:
            return None
        return _seq([_copy(0, 0, len(lang.prefix)), Prog("ALT", children=[_copy(*literal), _cls("digit", lang.width, lang.width)])])
    if family == "prefix_repeat":
        if not supports[0].startswith(lang.prefix):
            return None
        body = gold_repeat(lang.repeat, [text[len(lang.prefix) :] for text in supports])
        if body is None:
            return None
        # The separator span was measured on the suffix. Shift it back onto the full string.
        return _seq([_copy(0, 0, len(lang.prefix)), _shift_copies(body, len(lang.prefix))])
    if family == "repeat_optional":
        body = gold_repeat(lang.repeat, supports)
        tail = _find_span(supports, "-" + lang.suffix)
        if body is None or tail is None:
            return None
        return _seq([body, Prog("OPT", children=[_copy(*tail)])])
    return None


def _shift_copies(prog: Prog, offset: int) -> Prog:
    if prog.op == "COPY":
        return Prog("COPY", slot=prog.slot, start=prog.start + offset, end=prog.end + offset, cls=prog.cls, lo=prog.lo, hi=prog.hi)
    return Prog(
        prog.op,
        slot=prog.slot,
        start=prog.start,
        end=prog.end,
        cls=prog.cls,
        lo=prog.lo,
        hi=prog.hi,
        children=[_shift_copies(child, offset) for child in prog.children],
    )


def _shape(lang: Language, node, supports: list[str]) -> Prog | None:
    if node.kind == "class":
        width = lang.binding.lengths[node.ident]
        return _cls(node.cls, width, width)
    if node.kind == "literal_slot":
        found = _find_span(supports, lang.binding.literals[node.ident])
        if found is None:
            return None
        return _copy(*found)
    if node.kind == "optional":
        child = _shape(lang, node.children[0], supports)
        if child is None:
            return None
        return Prog("OPT", children=[child])
    if node.kind == "repeat":
        child = _shape(lang, node.children[0], supports)
        if child is None:
            return None
        return Prog("REPEAT", lo=node.min_n, hi=node.max_n, children=[child])
    if node.kind == "alt":
        children = [_shape(lang, child, supports) for child in node.children]
        if any(child is None for child in children):
            return None
        return Prog("ALT", children=children)
    if node.kind == "seq":
        children = [_shape(lang, child, supports) for child in node.children]
        if any(child is None for child in children):
            return None
        return _seq(children)
    return None


def program_from_dsl(lang: Language, supports: list[str]) -> Prog | None:
    """Schema structure, with copy spans taken from the supports that show each literal."""
    return _shape(lang, lang.root, supports)


def linearize(prog: Prog) -> list[dict]:
    steps: list[dict] = []

    def walk(node: Prog) -> None:
        steps.append(
            {
                "op": {"COPY": OP_COPY, "CLS": OP_CLS, "SEQ": OP_SEQ, "REPEAT": OP_REPEAT, "OPT": OP_OPT, "ALT": OP_ALT}[node.op],
                "slot": node.slot,
                "start": node.start,
                "end": max(node.end - 1, 0),
                "cls": CLS_NAME.index(node.cls) if node.cls in CLS_NAME else 0,
                "lo": max(0, min(node.lo, 4) - 1),
                "hi": max(0, min(node.hi, OPEN) - 1),
                "nchild": max(0, min(len(node.children), 4) - 2) if node.op == "SEQ" else 0,
            }
        )
        for child in node.children:
            walk(child)

    walk(prog)
    steps.append({"op": OP_STOP, "slot": 0, "start": 0, "end": 0, "cls": 0, "lo": 0, "hi": 0, "nchild": 0})
    return steps


def delinearize(steps: list[dict]) -> Prog | None:
    index = 0

    def parse() -> Prog | None:
        nonlocal index
        if index >= len(steps):
            return None
        step = steps[index]
        index += 1
        op = int(step["op"])
        if op == OP_STOP:
            return None
        if op == OP_COPY:
            return Prog("COPY", slot=int(step["slot"]), start=int(step["start"]), end=int(step["end"]) + 1)
        if op == OP_CLS:
            return Prog("CLS", cls=CLS_NAME[int(step["cls"])], lo=int(step["lo"]) + 1, hi=int(step["hi"]) + 1)
        if op == OP_SEQ:
            count = int(step["nchild"]) + 2
            children = []
            for _ in range(count):
                child = parse()
                if child is None:
                    return None
                children.append(child)
            return Prog("SEQ", children=children)
        if op in {OP_REPEAT, OP_OPT}:
            child = parse()
            if child is None:
                return None
            name = "REPEAT" if op == OP_REPEAT else "OPT"
            return Prog(name, lo=int(step["lo"]) + 1, hi=int(step["hi"]) + 1, children=[child])
        if op == OP_ALT:
            left = parse()
            right = parse()
            if left is None or right is None:
                return None
            return Prog("ALT", children=[left, right])
        return None

    prog = parse()
    if prog is None or index >= len(steps) or int(steps[index]["op"]) != OP_STOP:
        return None
    return prog


def grammar_ok(steps: list[dict]) -> bool:
    stack = [1]
    for step in steps:
        op = int(step["op"])
        if not stack:
            return op == OP_STOP
        if op == OP_STOP:
            return False
        stack[-1] -= 1
        if stack[-1] == 0:
            stack.pop()
        if op == OP_SEQ:
            stack.append(int(step["nchild"]) + 2)
        elif op in {OP_REPEAT, OP_OPT}:
            stack.append(1)
        elif op == OP_ALT:
            stack.append(2)
    return not stack


def _retarget(prog: Prog, fn) -> Prog:
    updated = fn(prog)
    return Prog(
        updated.op,
        slot=updated.slot,
        start=updated.start,
        end=updated.end,
        cls=updated.cls,
        lo=updated.lo,
        hi=updated.hi,
        children=[_retarget(child, fn) for child in updated.children],
    )


def corrupt(prog: Prog, supports: list[str]) -> Prog | None:
    """A nearby program the executor rejects, so training can prefer the gold one."""
    variants = []

    def freeze_repeat(node: Prog) -> Prog:
        if node.op == "REPEAT":
            return Prog(node.op, lo=node.lo, hi=node.lo, children=node.children)
        return node

    def drop_opt(node: Prog) -> Prog:
        if node.op == "OPT" and node.children:
            return node.children[0]
        return node

    def flip_class(node: Prog) -> Prog:
        if node.op == "CLS":
            name = "upper" if node.cls == "digit" else "digit"
            return Prog("CLS", cls=name, lo=node.lo, hi=node.hi)
        return node

    for fn in (freeze_repeat, drop_opt, flip_class):
        variant = _retarget(prog, fn)
        if describe(variant) == describe(prog):
            continue
        if accepts_all(prog, supports, supports) and not accepts_all(variant, supports, supports):
            variants.append(variant)
            break
    return variants[0] if variants else None


def label_string(prog: Prog, text: str, supports: list[str]) -> list[tuple[int, int]] | None:
    """Per-character copy/class role and the innermost operator scope, for one full parse."""

    def parse(node: Prog, index: int, scope: int) -> list[tuple[int, list[tuple[int, int, int]]]]:
        if node.op == "COPY":
            literal = _literal(node, supports)
            end = index + len(literal)
            if not literal or not text.startswith(literal, index):
                return []
            return [(end, [(cursor, ROLE_COPY, scope) for cursor in range(index, end)])]
        if node.op == "CLS":
            role = {"digit": ROLE_DIGIT, "upper": ROLE_UPPER, "lower": ROLE_LOWER}.get(node.cls, ROLE_DIGIT)
            lo = max(1, min(node.lo, node.hi))
            found = []
            for width in range(lo, max(lo, node.hi) + 1):
                chunk = text[index : index + width]
                if len(chunk) == width and class_ok(chunk, node.cls):
                    found.append((index + width, [(cursor, role, scope) for cursor in range(index, index + width)]))
            return found
        if node.op == "OPT" and node.children:
            return [(index, [])] + parse(node.children[0], index, SCOPE_OPT)
        if node.op == "ALT" and len(node.children) >= 2:
            return parse(node.children[0], index, SCOPE_ALT) + parse(node.children[1], index, SCOPE_ALT)
        if node.op == "SEQ":
            options = [(index, [])]
            for child in node.children:
                nxt = []
                for end, paints in options:
                    for child_end, child_paints in parse(child, end, scope):
                        nxt.append((child_end, paints + child_paints))
                options = nxt
                if not options:
                    return []
            return options
        if node.op == "REPEAT" and node.children:
            found = []
            options = [(index, [])]
            hi = max(node.lo, min(node.hi, OPEN))
            for count in range(1, hi + 1):
                nxt = []
                for end, paints in options:
                    for child_end, child_paints in parse(node.children[0], end, SCOPE_REPEAT):
                        nxt.append((child_end, paints + child_paints))
                options = nxt
                if count >= node.lo:
                    found.extend(options)
                if not options:
                    break
            return found
        return []

    hits = [paints for end, paints in parse(prog, 0, 0) if end == len(text)]
    if not hits:
        return None
    labels = [(0, 0) for _ in text]
    for pos, role, scope in hits[0]:
        labels[pos] = (role, scope)
    return labels


def _probe_compose(family: str, lang, rng: random.Random, supports: list[str], count: int) -> list[tuple[str, bool]]:
    rows = []
    for _ in range(count):
        if family == "repeat_extra":
            rows.append((lang.positive(rng, supports, rng.choice((4, 5))), True))
            rows.append((lang.negative(rng, supports, rng.choice((4, 5))), False))
        else:
            rows.append((lang.positive(rng, supports), True))
            rows.append((lang.negative(rng, supports), False))
    return rows


def sample_compose_task(family: str, rng: random.Random, probe_n: int = 6) -> Task:
    last = "sample"
    for _ in range(12):
        lang = _BUILDERS[family](rng)
        supports = lang.support(rng)
        program = gold_compose(family, lang, supports)
        if program is None or len(linearize(program)) > MAX_PROG:
            last = "gold"
            continue
        if not accepts_all(program, supports, supports):
            last = "support"
            continue
        probe = _probe_compose(family, lang, rng, supports, probe_n)
        if not agrees(program, supports, probe):
            last = "probe"
            continue
        return Task(family, supports, program, probe)
    raise RuntimeError(f"compose task failed: {family} {last}")


def sample_dsl_task(compiled, rng: random.Random, probe_n: int = 6) -> Task:
    last = "sample"
    for _ in range(8):
        try:
            lang = Language(compiled, rng)
            supports, _anns = lang.support(rng)
            program = program_from_dsl(lang, supports)
        except SchemaError as exc:
            last = exc.reason
            continue
        if program is None or len(linearize(program)) > MAX_PROG:
            last = "gold"
            continue
        banned = set(supports)
        probe = []
        try:
            for _draw in range(probe_n):
                text, _ann = lang.positive(rng, banned)
                probe.append((text, True))
                probe.append((lang.negative(rng, banned), False))
        except SchemaError as exc:
            last = exc.reason
            continue
        if not accepts_all(program, supports, supports) or not agrees(program, supports, probe):
            last = "semantics"
            continue
        return Task(f"d1:{compiled.ident}", supports, program, probe)
    raise RuntimeError(f"dsl task failed: {last}")


def same_structure(left: list[dict], right: list[dict]) -> bool:
    if len(left) != len(right):
        return False
    keys = ("op", "slot", "start", "end", "cls", "lo", "hi", "nchild")
    return all(all(int(a[key]) == int(b[key]) for key in keys) for a, b in zip(left, right))


def feature_batch(tasks: list[Task]):
    episodes = [
        Episode(task.supports[:MAX_SUPPORT], task.supports[0], 1, task.family, "p1", task.family) for task in tasks
    ]
    return featurize(episodes, max_len=MAX_LEN, max_support=MAX_SUPPORT)
