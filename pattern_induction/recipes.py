"""Training distributions for the frozen compositional model.

D0 is the atomic curriculum. D1 is a hand-built matrix of other compositions.
D2–D4 load schemas that the executable compiler has already accepted. Every
recipe keeps the same batch size and the same positive/negative balance.
D1–D4 spend half their steps on the atomic curriculum so the checkpoint rule,
which watches those four families, still sees them.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pattern_induction.compose import TRAIN_FAMILIES, sample_batch
from pattern_induction.schema_dsl import (
    GENERATOR_VERSION,
    CompiledSchema,
    SchemaError,
    compile_schema,
    sample_schema_episode,
)

D1_SPECS = [
    {
        "id": "class_repeat",
        "schema": {
            "type": "seq",
            "children": [
                {"type": "class", "class": "upper", "min": 2, "max": 2},
                {"type": "class", "class": "digit", "min": 1, "max": 2},
                {
                    "type": "repeat",
                    "min": 1,
                    "max": 3,
                    "child": {
                        "type": "seq",
                        "children": [
                            {"type": "literal_slot", "role": "sep", "class": "punct", "min": 1, "max": 1},
                            {"type": "class", "class": "digit", "min": 1, "max": 2},
                        ],
                    },
                },
            ],
        },
    },
    {
        "id": "repeat_suffix",
        "schema": {
            "type": "seq",
            "children": [
                {"type": "class", "class": "digit", "min": 1, "max": 2},
                {
                    "type": "repeat",
                    "min": 1,
                    "max": 3,
                    "child": {
                        "type": "seq",
                        "children": [
                            {"type": "literal_slot", "role": "sep", "class": "punct", "min": 1, "max": 1},
                            {"type": "class", "class": "digit", "min": 1, "max": 2},
                        ],
                    },
                },
                {"type": "literal_slot", "role": "suffix", "class": "upper", "min": 2, "max": 2},
            ],
        },
    },
    {
        "id": "optional_repeat",
        "schema": {
            "type": "seq",
            "children": [
                {"type": "optional", "child": {"type": "literal_slot", "role": "mark", "class": "punct", "min": 1, "max": 1}},
                {"type": "class", "class": "digit", "min": 1, "max": 2},
                {
                    "type": "repeat",
                    "min": 1,
                    "max": 3,
                    "child": {
                        "type": "seq",
                        "children": [
                            {"type": "literal_slot", "role": "sep", "class": "punct", "min": 1, "max": 1},
                            {"type": "class", "class": "digit", "min": 1, "max": 2},
                        ],
                    },
                },
            ],
        },
    },
    {
        "id": "alt_suffix",
        "schema": {
            "type": "seq",
            "children": [
                {
                    "type": "alt",
                    "children": [
                        {"type": "literal_slot", "role": "word", "class": "upper", "min": 3, "max": 3},
                        {"type": "class", "class": "digit", "min": 4, "max": 4},
                    ],
                },
                {"type": "literal_slot", "role": "suffix", "class": "upper", "min": 2, "max": 2},
            ],
        },
    },
    {
        "id": "repeat_repeat",
        "schema": {
            "type": "seq",
            "children": [
                {"type": "class", "class": "digit", "min": 1, "max": 1},
                {
                    "type": "repeat",
                    "min": 1,
                    "max": 2,
                    "child": {
                        "type": "seq",
                        "children": [
                            {"type": "literal_slot", "role": "sep_a", "class": "punct", "min": 1, "max": 1},
                            {"type": "class", "class": "digit", "min": 1, "max": 1},
                        ],
                    },
                },
                {"type": "class", "class": "lower", "min": 1, "max": 1},
                {
                    "type": "repeat",
                    "min": 1,
                    "max": 2,
                    "child": {
                        "type": "seq",
                        "children": [
                            {"type": "literal_slot", "role": "sep_b", "class": "punct", "min": 1, "max": 1},
                            {"type": "class", "class": "lower", "min": 1, "max": 1},
                        ],
                    },
                },
            ],
        },
    },
    {
        "id": "optional_alt",
        "schema": {
            "type": "seq",
            "children": [
                {"type": "optional", "child": {"type": "literal_slot", "role": "mark", "class": "punct", "min": 1, "max": 1}},
                {
                    "type": "alt",
                    "children": [
                        {"type": "literal_slot", "role": "word", "class": "upper", "min": 3, "max": 3},
                        {"type": "class", "class": "digit", "min": 3, "max": 3},
                    ],
                },
            ],
        },
    },
    {
        "id": "class_alt",
        "schema": {
            "type": "seq",
            "children": [
                {"type": "class", "class": "upper", "min": 2, "max": 2},
                {
                    "type": "alt",
                    "children": [
                        {"type": "literal_slot", "role": "word", "class": "lower", "min": 3, "max": 3},
                        {"type": "class", "class": "digit", "min": 3, "max": 3},
                    ],
                },
            ],
        },
    },
    {
        "id": "alt_repeat",
        "schema": {
            "type": "seq",
            "children": [
                {
                    "type": "alt",
                    "children": [
                        {"type": "literal_slot", "role": "word", "class": "upper", "min": 2, "max": 2},
                        {"type": "class", "class": "lower", "min": 2, "max": 2},
                    ],
                },
                {"type": "class", "class": "digit", "min": 1, "max": 1},
                {
                    "type": "repeat",
                    "min": 1,
                    "max": 2,
                    "child": {
                        "type": "seq",
                        "children": [
                            {"type": "literal_slot", "role": "sep", "class": "punct", "min": 1, "max": 1},
                            {"type": "class", "class": "digit", "min": 1, "max": 1},
                        ],
                    },
                },
            ],
        },
    },
]


def compile_bank(records: list[dict], allow_heldout: bool) -> list[CompiledSchema]:
    compiled = []
    for record in records:
        compiled.append(compile_schema(record, allow_heldout=allow_heldout))
    if not compiled:
        raise SchemaError("empty_bank")
    return compiled


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def d1_bank() -> list[CompiledSchema]:
    return compile_bank(D1_SPECS, allow_heldout=False)


def load_schema_bank(path: Path, allow_heldout: bool) -> list[CompiledSchema]:
    return compile_bank(load_jsonl(path), allow_heldout=allow_heldout)


def load_pair_bank(path: Path) -> list[CompiledSchema]:
    """Each side of a minimal pair is its own language. The other side is a confusable negative."""
    compiled: list[CompiledSchema] = []
    for record in load_jsonl(path):
        left = record.get("schema_a") or record.get("A")
        right = record.get("schema_b") or record.get("B")
        ident = str(record.get("id") or "pair")
        compiled.append(compile_schema({"id": f"{ident}:a", "schema": left, "confusable": [right]}, allow_heldout=True))
        compiled.append(compile_schema({"id": f"{ident}:b", "schema": right, "confusable": [left]}, allow_heldout=True))
    if not compiled:
        raise SchemaError("empty_bank")
    return compiled


def sample_recipe_batch(
    recipe: str,
    rng: random.Random,
    step: int,
    size: int,
    bank: list[CompiledSchema] | None = None,
) -> list:
    """One family per batch, half positive. Odd steps of D1–D4 stay on the atomic curriculum."""
    if recipe == "d0" or (recipe != "d0" and step % 2 == 1):
        cursor = (step - 1) // (1 if recipe == "d0" else 2)
        family = TRAIN_FAMILIES[cursor % len(TRAIN_FAMILIES)]
        return sample_batch(family, rng, size)
    if not bank:
        raise SchemaError("empty_bank")
    compiled = bank[((step // 2) - 1) % len(bank)]
    half = size // 2
    episodes = [sample_schema_episode(compiled, rng, 1, f"{recipe}:{compiled.ident}") for _ in range(half)]
    episodes += [sample_schema_episode(compiled, rng, 0, f"{recipe}:{compiled.ident}") for _ in range(size - half)]
    rng.shuffle(episodes)
    return episodes


def recipe_sources(root: Path) -> dict[str, tuple[list[CompiledSchema] | None, bool]]:
    """Return the compiled bank and whether training may include an eval family."""
    validated = root / "data" / "teacher" / "validated"
    d2 = validated / "schemas.jsonl"
    d3 = validated / "minimal_pairs.jsonl"
    d4 = validated / "adversarial_templates.jsonl"
    banks: dict[str, tuple[list[CompiledSchema] | None, bool]] = {
        "d0": (None, False),
        "d1": (d1_bank(), False),
    }
    if d2.exists() and load_jsonl(d2):
        banks["d2"] = (load_schema_bank(d2, allow_heldout=False), False)
    if d3.exists() and load_jsonl(d3):
        banks["d3"] = (load_pair_bank(d3), True)
    if "d2" in banks and "d3" in banks:
        extra = load_schema_bank(d4, allow_heldout=True) if d4.exists() and load_jsonl(d4) else []
        banks["d4"] = (banks["d2"][0] + banks["d3"][0] + extra, True)
    return banks


def generator_stamp(records: list[dict]) -> dict:
    from pattern_induction.schema_dsl import bank_hash

    return {"generator_version": GENERATOR_VERSION, "template_bank_hash": bank_hash(records), "templates": len(records)}
