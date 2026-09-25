"""BENCH-D ambiguity: the same latent schemas observed with more examples."""

from __future__ import annotations

import random
import string

from pattern_induction.sample_util import build_task
from pattern_induction.tasks import Task


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))


def _upper(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(n))


SCHEMAS = [
    (
        "two_letters_two_digits",
        r"^[A-Z]{2}\d{2}$",
        lambda rng: _upper(rng, 2) + _digits(rng, 2),
        "Two examples often look like a closed list. The generator is a character class.",
    ),
    (
        "letter_digits",
        r"^[A-Z]\d{3}$",
        lambda rng: _upper(rng, 1) + _digits(rng, 3),
        "Single letter prefix plus three digits.",
    ),
    (
        "optional_dash",
        r"^[A-Z]{3}-?\d{4}$",
        lambda rng: _upper(rng, 3) + ("-" if rng.random() < 0.5 else "") + _digits(rng, 4),
        "The dash is optional, so two examples can hide it.",
    ),
    (
        "alt_prefix",
        r"^(?:INV|PO)-\d{4}$",
        lambda rng: rng.choice(("INV", "PO")) + "-" + _digits(rng, 4),
        "Two prefixes. A two-example draw can show only one.",
    ),
    (
        "variable_digits",
        r"^ID-\d{3,6}$",
        lambda rng: "ID-" + _digits(rng, rng.randint(3, 6)),
        "Width is a range. Short samples understate it.",
    ),
    (
        "date_or_literal_year",
        r"^20\d{2}-\d{2}-\d{2}$",
        lambda rng: f"{rng.randint(2018, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "A short sample can make the year look fixed.",
    ),
    (
        "word_code",
        r"^[a-z]{3,5}_\d{2}$",
        lambda rng: "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 5)))
        + "_"
        + _digits(rng, 2),
        "Lowercase stem of variable width.",
    ),
    (
        "hex_byte",
        r"^(?:[0-9A-F]{2}){2,4}$",
        lambda rng: "".join(rng.choice("0123456789ABCDEF") for _ in range(2 * rng.randint(2, 4))),
        "Even-length hex. Few examples look like fixed literals.",
    ),
]


def generate_bench_d(seed: int = 11) -> list[Task]:
    rng = random.Random(seed)
    tasks: list[Task] = []
    index = 1
    for name, regex, sample, notes in SCHEMAS:
        for n_visible in (2, 3, 5, 10, 20):
            for repeat in range(8):
                tasks.append(
                    build_task(
                        task_id=f"D-{index:04d}-{name}-n{n_visible}-{repeat}",
                        benchmark="D",
                        domain="ambiguity",
                        variant=f"{name}_n{n_visible}",
                        gold_regex=regex,
                        sample=sample,
                        rng=rng,
                        latent_schema=regex,
                        n_visible=n_visible,
                        n_pos=24,
                        n_neg=24,
                        split="test",
                        notes=notes,
                        meta={"n_visible": n_visible, "schema": name},
                    )
                )
                index += 1
    return tasks
