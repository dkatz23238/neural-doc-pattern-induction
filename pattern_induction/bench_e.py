"""BENCH-E: a sealed external split.

Generated once from a fixed seed. It is not a development set. Do not score
it, print its tasks, or choose a model, a fact, or a grammar production by
looking at it. BENCH-C is the diagnostic set. This file stays closed until
the architecture and the grammar are frozen.
"""

from __future__ import annotations

import hashlib
import json
import random
import string
from pathlib import Path

from pattern_induction.sample_util import build_task
from pattern_induction.tasks import Task, write_jsonl

SEED = 2507
REPLICATES = 4


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))


def _upper(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(n))


def _lower(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def _hex(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.hexdigits.lower()) for _ in range(n))


def _families() -> list[tuple]:
    rows = []

    def add(domain: str, name: str, regex: str, sample, notes: str) -> None:
        rows.append((domain, name, regex, sample, notes))

    add("lot", "desk", r"^LOT-[A-Z]{2}\d{4}$", lambda rng: f"LOT-{_upper(rng, 2)}{_digits(rng, 4)}", "Desk lot.")
    add("bin", "aisle", r"^BIN-\d{2}-[A-Z]{3}$", lambda rng: f"BIN-{_digits(rng, 2)}-{_upper(rng, 3)}", "Aisle bin.")
    add("badge", "day", r"^BG\d{6}$", lambda rng: "BG" + _digits(rng, 6), "Day badge.")
    add("sensor", "rack", r"^SN-[A-Z]{3}-\d{5}$", lambda rng: f"SN-{_upper(rng, 3)}-{_digits(rng, 5)}", "Rack sensor.")
    add("vial", "cap", r"^V\d{2}[A-Z]\d{3}$", lambda rng: f"V{_digits(rng, 2)}{_upper(rng, 1)}{_digits(rng, 3)}", "Capped vial.")
    add(
        "dock",
        "slip",
        r"^DCK-\d{4}(?:-[A-Z]{2})?$",
        lambda rng: "DCK-" + _digits(rng, 4) + (f"-{_upper(rng, 2)}" if rng.random() < 0.5 else ""),
        "Dock slip, optional suffix.",
    )
    add(
        "pallet",
        "kind",
        r"^(?:PLT|CRT)-\d{6}$",
        lambda rng: rng.choice(("PLT", "CRT")) + "-" + _digits(rng, 6),
        "Two pallet marks.",
    )
    add(
        "wafer",
        "cell",
        r"^W-\d{2}-\d{2}-[A-Z]$",
        lambda rng: f"W-{_digits(rng, 2)}-{_digits(rng, 2)}-{_upper(rng, 1)}",
        "Wafer cell.",
    )
    add(
        "lane",
        "bay",
        r"^L\d{1,2}[A-Z]{2}$",
        lambda rng: "L" + _digits(rng, rng.choice((1, 2))) + _upper(rng, 2),
        "Lane bay.",
    )
    add("cassette", "slot", r"^CS\d{8}$", lambda rng: "CS" + _digits(rng, 8), "Cassette slot.")
    add(
        "tray",
        "row",
        r"^TR-[a-z]{3}-\d{3}$",
        lambda rng: f"TR-{_lower(rng, 3)}-{_digits(rng, 3)}",
        "Tray row.",
    )
    add(
        "aliquot",
        "tube",
        r"^ALQ\d{4}[A-Z]$",
        lambda rng: f"ALQ{_digits(rng, 4)}{_upper(rng, 1)}",
        "Aliquot tube.",
    )
    add(
        "gate",
        "pass",
        r"^GP-[A-Z]{4}-\d{2}$",
        lambda rng: f"GP-{_upper(rng, 4)}-{_digits(rng, 2)}",
        "Gate pass.",
    )
    add("token", "hex", r"^[0-9a-f]{8}$", lambda rng: _hex(rng, 8), "Hex token.")
    add(
        "fiscal",
        "year",
        r"^FY(?:2[4-9]|3[0-2])-\d{3}$",
        lambda rng: f"FY{rng.randint(24, 32)}-{_digits(rng, 3)}",
        "Fiscal year window.",
    )
    add(
        "temporary",
        "open",
        r"^TMP-\d+$",
        lambda rng: "TMP-" + _digits(rng, rng.randint(1, 6)),
        "Open temporary id.",
    )
    add(
        "shelf",
        "label",
        r"^Row \d{1,2}$",
        lambda rng: f"Row {rng.randint(1, 40)}",
        "Shelf label with a space.",
    )
    add(
        "wing",
        "side",
        r"^(?:North|South|East|West)-\d{2}$",
        lambda rng: rng.choice(("North", "South", "East", "West")) + "-" + _digits(rng, 2),
        "Wing name.",
    )
    add(
        "container",
        "unit",
        r"^CNU[A-Z]{2}\d{6}$",
        lambda rng: f"CNU{_upper(rng, 2)}{_digits(rng, 6)}",
        "Container unit.",
    )
    add(
        "ticket",
        "span",
        r"^TKT-\d{3,5}$",
        lambda rng: "TKT-" + _digits(rng, rng.choice((3, 4, 5))),
        "Ticket width.",
    )
    add(
        "shelf",
        "code",
        r"^[A-Z]\d{2}-[A-Z]\d{2}$",
        lambda rng: f"{_upper(rng, 1)}{_digits(rng, 2)}-{_upper(rng, 1)}{_digits(rng, 2)}",
        "Shelf code.",
    )
    add(
        "batch",
        "rev",
        r"^BAT-\d{4}(?:-R\d)?$",
        lambda rng: "BAT-" + _digits(rng, 4) + (f"-R{rng.randint(0, 9)}" if rng.random() < 0.5 else ""),
        "Batch with an optional revision.",
    )
    add(
        "coil",
        "pair",
        r"^\d{5}$|^\d{8}$",
        lambda rng: _digits(rng, rng.choice((5, 8))),
        "Two coil widths.",
    )
    add("stamp", "head", r"^77\d{10}$", lambda rng: "77" + _digits(rng, 10), "Fixed head, long body.")
    add(
        "lab",
        "slip",
        r"^lab-[a-z]{4}\d{2}$",
        lambda rng: f"lab-{_lower(rng, 4)}{_digits(rng, 2)}",
        "Lab slip.",
    )
    return rows


def generate_bench_e(seed: int = SEED, replicates: int = REPLICATES) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    index = 1
    for domain, name, regex, sample, notes in _families():
        for replicate in range(replicates):
            tasks.append(
                build_task(
                    task_id=f"E-{index:04d}-{domain}-{name}-r{replicate}",
                    benchmark="E",
                    domain=domain,
                    variant=name,
                    gold_regex=regex,
                    sample=sample,
                    rng=rng,
                    latent_schema=regex,
                    n_visible=6,
                    n_pos=24,
                    n_neg=24,
                    split="sealed",
                    notes=notes,
                    meta={"source": "sealed-external", "holdout": True, "replicate": replicate, "seed": seed},
                )
            )
            index += 1
    return tasks


def seal_bench_e(root: Path, seed: int = SEED) -> dict:
    """Write the split and its seal. The seal records that nothing was scored."""
    tasks = generate_bench_e(seed)
    path = root / "data" / "bench_e.jsonl"
    write_jsonl(path, tasks)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "benchmark": "BENCH-E",
        "role": "untouched external split",
        "scored": False,
        "inspected": False,
        "n": len(tasks),
        "sha256": digest,
        "seed": seed,
        "replicates": REPLICATES,
        "note": "Do not score or inspect until the architecture and the grammar are frozen. BENCH-C is the diagnostic set.",
    }
    seal = root / "results" / "needle" / "bench_e_seal.json"
    seal.parent.mkdir(parents=True, exist_ok=True)
    seal.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
