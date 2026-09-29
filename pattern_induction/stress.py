"""Eval-only stress for the frozen CPU selector.

Generators, the recognizer, the chart, and the witness head stay as they
are. Each cell builds tasks from those generators, or from one rename of
their strings, and scores the already trained selector.
"""

from __future__ import annotations

import random
import string

from pattern_induction.compose import PrefixAlt, PrefixRepeat, RepeatNum, _BUILDERS
from pattern_induction.local_train import OOD
from pattern_induction.program import (
    MAX_LEN,
    Prog,
    Task,
    _copy,
    _find_span,
    _probe_compose,
    _shift_copies,
    accepts_all,
    agrees,
    gold_compose,
    gold_repeat,
)
from pattern_induction.renaming import apply_mapping, build_mapping
from pattern_induction.select_train import _semantic, select_program

STRESS_DRAWS = 8


def _task(family: str, supports: list[str], program: Prog, probe: list[tuple[str, bool]]) -> Task | None:
    if program is None or not supports or any(len(text) > MAX_LEN or not text for text in supports):
        return None
    if any(len(text) > MAX_LEN or not text for text, _label in probe):
        return None
    if not accepts_all(program, supports, supports) or not agrees(program, supports, probe):
        return None
    return Task(family, supports, program, probe)


def _few_support(family: str, rng: random.Random) -> Task | None:
    lang = _BUILDERS[family](rng)
    full = lang.support(rng)
    if family == "repeat_optional":
        tail = "-" + lang.suffix
        present = [text for text in full if text.endswith(tail)]
        absent = [text for text in full if not text.endswith(tail)]
        if not present or not absent:
            return None
        supports = [absent[0], present[0]]
    elif family == "optional_internal":
        short = [text for text in full if text == lang.left + lang.right]
        long = [text for text in full if text != lang.left + lang.right]
        if not short or not long:
            return None
        supports = [short[0], long[0]]
    else:
        supports = full[:2]
    program = gold_compose(family, lang, supports)
    if program is None:
        return None
    return _task(family, supports, program, _probe_compose(family, lang, rng, supports, 3))


def _long_probe(rng: random.Random) -> Task | None:
    lang = RepeatNum(rng)
    supports = lang.support(rng)
    program = gold_repeat(lang, supports)
    if program is None:
        return None
    probe: list[tuple[str, bool]] = []
    for depth in (6, 7):
        probe.append((lang.positive(rng, supports, depth), True))
        parts = ["".join(rng.choice(string.digits) for _ in range(3)) for _ in range(depth + 1)]
        probe.append((lang.sep.join(parts), False))
    return _task("repeat_long", supports, program, probe)


def _rename(family: str, rng: random.Random) -> Task | None:
    lang = _BUILDERS[family](rng)
    supports = lang.support(rng)
    program = gold_compose(family, lang, supports)
    if program is None:
        return None
    probe = _probe_compose(family, lang, rng, supports, 3)
    blob = list(supports) + [text for text, _label in probe]
    mapping = build_mapping(blob, rng, True)
    renamed_supports = [apply_mapping(text, mapping) for text in supports]
    renamed_probe = [(apply_mapping(text, mapping), label) for text, label in probe]
    return _task(family, renamed_supports, program, renamed_probe)


def _adversarial_width(rng: random.Random) -> Task | None:
    lang = RepeatNum(rng)
    supports: list[str] = []
    for repeats in (1, 2, 3):
        found = ""
        for _try in range(60):
            text = lang.sample(rng, repeats)
            parts = text.split(lang.sep)
            if parts and all(len(part) == 2 for part in parts):
                found = text
                break
        if not found:
            return None
        supports.append(found)
    program = gold_repeat(lang, supports)
    if program is None:
        return None
    positive = ""
    for _try in range(60):
        text = lang.sample(rng, 2)
        parts = text.split(lang.sep)
        if any(len(part) == 1 for part in parts) and text not in supports:
            positive = text
            break
    if not positive:
        return None
    wide = lang.sep.join("".join(rng.choice(string.digits) for _ in range(3)) for _ in range(3))
    probe = [(positive, True), (lang.positive(rng, supports, 2), True), (wide, False), (lang.negative(rng, supports, 2), False)]
    return _task("repeat_width", supports, program, probe)


def _lower_prefix(family: str, rng: random.Random) -> Task | None:
    if family not in {"prefix_repeat", "prefix_alt"}:
        return None
    lang = PrefixRepeat(rng) if family == "prefix_repeat" else PrefixAlt(rng)
    supports = lang.support(rng)
    program = gold_compose(family, lang, supports)
    if program is None:
        return None
    probe = _probe_compose(family, lang, rng, supports, 3)
    letters = {ch for text in supports for ch in text if ch.isupper()}
    if not letters:
        return None
    mapping = {ch: ch.lower() for ch in letters}
    lowered_supports = [apply_mapping(text, mapping) for text in supports]
    lowered_probe = [(apply_mapping(text, mapping), label) for text, label in probe]
    return _task(family, lowered_supports, program, lowered_probe)


def _bodies(prefix: str, tail: str, supports: list[str]) -> list[str]:
    rows = []
    for text in supports:
        body = text[len(prefix) :]
        if body.endswith(tail):
            body = body[: -len(tail)]
        rows.append(body)
    return rows


def _nested(rng: random.Random) -> Task | None:
    """Prefix, then an open repeat, then an optional two-letter tail."""
    prefix = "".join(rng.choice(string.ascii_uppercase) for _ in range(2))
    repeat = RepeatNum(rng)
    suffix = "".join(rng.choice(string.ascii_uppercase) for _ in range(2))
    tail = "-" + suffix

    def draw(repeats: int, present: bool) -> str:
        return prefix + repeat.sample(rng, repeats) + (tail if present else "")

    supports = [draw(1, False), draw(2, True), draw(3, False), draw(2, False)]
    if any(len(text) > MAX_LEN for text in supports):
        return None
    body = gold_repeat(repeat, _bodies(prefix, tail, supports))
    found = _find_span(supports, tail)
    if body is None or found is None:
        return None
    program = Prog(
        "SEQ",
        children=[_copy(0, 0, len(prefix)), _shift_copies(body, len(prefix)), Prog("OPT", children=[_copy(*found)])],
    )
    banned = set(supports)

    def positive() -> str:
        for _try in range(30):
            text = draw(rng.choice((1, 2, 3)), bool(rng.randrange(2)))
            if text not in banned:
                return text
        return ""

    def negative() -> str:
        for _try in range(30):
            mode = rng.randrange(3)
            if mode == 0:
                text = "QQ" + repeat.sample(rng, 2)
            elif mode == 1:
                parts = ["".join(rng.choice(string.digits) for _ in range(3)) for _ in range(3)]
                text = prefix + repeat.sep.join(parts)
            else:
                text = prefix + repeat.sample(rng, 2) + suffix
            if text not in banned and not accepts_all(program, [text], supports):
                return text
        return ""

    probe = []
    for _draw in range(3):
        yes = positive()
        no = negative()
        if not yes or not no:
            return None
        probe.append((yes, True))
        probe.append((no, False))
    return _task("prefix_repeat_optional", supports, program, probe)


def _draw_tasks(builder, draws: int, seed: int) -> list[Task]:
    tasks = []
    cursor = 0
    while len(tasks) < draws and cursor < draws * 12:
        task = builder(random.Random(seed + cursor))
        cursor += 1
        if task is not None:
            tasks.append(task)
    return tasks


def stress_cells(draws: int = STRESS_DRAWS) -> list[tuple[str, str, list[Task]]]:
    """Fair cells first. Nested and lower-case prefixes are reported beside them."""
    cells: list[tuple[str, str, list[Task]]] = []
    few: list[Task] = []
    for index, family in enumerate(OOD):
        few.extend(_draw_tasks(lambda rng, family=family: _few_support(family, rng), draws, 7000 + index * 40))
    cells.append(("few_support", "core", few))
    cells.append(("long_probe", "core", _draw_tasks(_long_probe, draws, 8000)))
    renamed: list[Task] = []
    for index, family in enumerate(OOD):
        renamed.extend(_draw_tasks(lambda rng, family=family: _rename(family, rng), draws, 8100 + index * 40))
    cells.append(("rename", "core", renamed))
    cells.append(("adversarial_width", "core", _draw_tasks(_adversarial_width, draws, 8200)))
    lowered: list[Task] = []
    for index, family in enumerate(("prefix_repeat", "prefix_alt")):
        lowered.extend(_draw_tasks(lambda rng, family=family: _lower_prefix(family, rng), draws, 8300 + index * 40))
    cells.append(("lower_prefix", "report", lowered))
    cells.append(("nested", "report", _draw_tasks(_nested, draws, 8400)))
    return cells


def score_rows(scope, witness, prior, tasks: list[Task], with_stability: bool) -> list[dict]:
    rows = []
    for index, task in enumerate(tasks):
        chosen, programs, count = select_program(scope, witness, prior, task, with_stability)
        print(f"  {task.family} {index + 1}/{len(tasks)} languages {count}", flush=True)
        rows.append(
            {
                "family": task.family,
                "top1": 1.0 if chosen is not None and agrees(chosen, task.supports, task.probe) else 0.0,
                "semantic": _semantic(chosen, task),
                "oracle": 1.0 if any(agrees(prog, task.supports, task.probe) for prog in programs) else 0.0,
                "families": count,
            }
        )
    return rows


def aggregate(rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    return {
        "n": len(rows),
        "top1": sum(row["top1"] for row in rows) / n,
        "semantic": sum(row["semantic"] for row in rows) / n,
        "oracle": sum(row["oracle"] for row in rows) / n,
        "families": sum(row["families"] for row in rows) / n,
        "max_families": max((row["families"] for row in rows), default=0),
    }
