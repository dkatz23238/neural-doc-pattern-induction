"""Shared sampling helpers that keep hidden sets consistent with a gold regex."""

from __future__ import annotations

import random
import re
from collections.abc import Callable

from pattern_induction.tasks import Task

ALIENS = [
    "hello",
    "not-a-match",
    "@@@",
    "ABC",
    "123",
    "foo bar",
    "NULL",
    "n/a",
    "...",
    "<missing>",
    "xx",
    "YYYY",
    "0",
    "test@test",
    "http://nope",
    "   ",
    "\t",
    "INV",
    "2026",
    "€",
    "$",
    "PO-1",
    "page",
    "###",
    "O0O0",
    "l1I",
]


def _mutate(text: str, rng: random.Random) -> str:
    if not text:
        return "x"
    ops = rng.randrange(8)
    chars = list(text)
    i = rng.randrange(len(chars))
    if ops == 0 and len(chars) > 1:
        del chars[i]
    elif ops == 1:
        chars.insert(i, rng.choice("0OIl1- /.,"))
    elif ops == 2 and chars[i].isdigit():
        chars[i] = rng.choice("OIlABCDEFG")
    elif ops == 3 and chars[i].isalpha():
        chars[i] = chars[i].swapcase()
    elif ops == 4 and chars[i].isdigit():
        chars[i] = str((int(chars[i]) + rng.randint(1, 9)) % 10)
    elif ops == 5:
        chars.append(rng.choice("0A- "))
    elif ops == 6:
        return text.replace("-", "/", 1) if "-" in text else text + " "
    else:
        chars[i] = rng.choice("xyzXYZ!*")
    mutated = "".join(chars)
    return mutated if mutated != text else text + "x"


def build_task(
    *,
    task_id: str,
    benchmark: str,
    domain: str,
    variant: str,
    gold_regex: str,
    sample: Callable[[random.Random], str],
    rng: random.Random,
    latent_schema: str,
    n_visible: int = 6,
    n_pos: int = 24,
    n_neg: int = 24,
    split: str = "test",
    notes: str = "",
    meta: dict | None = None,
    extra_negative: Callable[[random.Random], str] | None = None,
) -> Task:
    gold = re.compile(gold_regex)
    seen: set[str] = set()

    def draw(unique: bool) -> str:
        fallback = ""
        for _ in range(80):
            value = sample(rng)
            if not gold.fullmatch(value):
                continue
            fallback = value
            if not unique or value not in seen:
                seen.add(value)
                return value
        if fallback:
            return fallback
        raise RuntimeError(f"sampler failed for {task_id} / {gold_regex}")

    visible = [draw(True) for _ in range(n_visible)]
    positives = [draw(False) for _ in range(n_pos)]
    negatives: list[str] = []
    attempts = 0
    while len(negatives) < n_neg and attempts < 4000:
        attempts += 1
        if extra_negative and attempts % 3 == 0:
            candidate = extra_negative(rng)
        elif attempts % 5 == 0:
            candidate = rng.choice(ALIENS)
        else:
            candidate = _mutate(rng.choice(visible + positives), rng)
        if candidate in seen or candidate in negatives:
            continue
        if gold.fullmatch(candidate):
            continue
        negatives.append(candidate)
    if len(negatives) < n_neg:
        raise RuntimeError(f"not enough negatives for {task_id}: {len(negatives)}")
    return Task(
        id=task_id,
        benchmark=benchmark,
        domain=domain,
        variant=variant,
        visible=visible,
        hidden_positives=positives,
        hidden_negatives=negatives,
        gold_regex=gold_regex,
        latent_schema=latent_schema,
        split=split,
        notes=notes,
        meta=meta or {},
    )
