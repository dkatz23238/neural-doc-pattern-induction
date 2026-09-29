"""Distractor sweep for the version-space chart.

The recognizer stays frozen. Each condition includes every gold local fact.
The version space should still contain a probe-matching program.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from pattern_induction.chart import LOCAL_K, compose, gold_facts
from pattern_induction.local_net import LocalNet, collate_local, hypotheses_from_batch
from pattern_induction.local_train import ALL, ATOMIC, OOD
from pattern_induction.program import OPEN, Prog, Task, agrees, sample_compose_task, sample_dsl_task
from pattern_induction.recipes import d1_bank
from pattern_induction.version_space import assemble, distractors

CONDITIONS = ("gold", "plus1", "plus2", "model")
DRAWS = 16
D1_DRAWS = 4


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _open_repeat(prog: Prog) -> bool:
    if prog.op == "REPEAT" and prog.hi >= OPEN:
        return True
    return any(_open_repeat(child) for child in prog.children)


def _hypotheses(condition: str, task: Task, model: LocalNet | None, rng: random.Random) -> list:
    gold = gold_facts(task.program, task.supports)
    if condition == "gold":
        return gold
    if condition == "plus1":
        return gold + distractors(task.supports, gold, 1, rng)
    if condition == "plus2":
        return gold + distractors(task.supports, gold, 2, rng)
    if model is None:
        raise RuntimeError("the model condition needs the frozen recognizer")
    if condition != "model":
        raise RuntimeError(condition)
    predicted = hypotheses_from_batch(model, collate_local([task]), [task.supports[:4]], LOCAL_K)[0]
    return gold + predicted


def _score(task: Task, hyps) -> dict:
    built = assemble(task.supports, hyps)
    beam = compose(task.supports, hyps, LOCAL_K)
    version_hit = any(agrees(prog, task.supports, task.probe) for prog, _score in built.programs)
    beam_hit = any(agrees(prog, task.supports, task.probe) for prog, _score in beam)
    open_hit = any(
        agrees(prog, task.supports, task.probe) and _open_repeat(prog) for prog, _score in built.programs
    )
    return {
        "version": int(version_hit),
        "beam": int(beam_hit),
        "open": int(open_hit),
        "programs": len(built.programs),
        "truncated": int(built.truncated),
        "library": built.library,
    }


def _mean(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(row[key] for row in rows) / len(rows)


def _load_model(root: Path) -> LocalNet:
    path = root / "results" / "local_chart" / "l1" / "seed_0" / "checkpoint.pt"
    if not path.exists():
        raise RuntimeError(f"missing frozen recognizer at {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = LocalNet()
    model.load_state_dict(state)
    model.eval()
    return model


def _compose_tasks() -> list[Task]:
    tasks = []
    for index, family in enumerate(ALL):
        for draw in range(DRAWS):
            tasks.append(sample_compose_task(family, random.Random(5000 + index * 50 + draw), 3))
    return tasks


def _d1_tasks():
    tasks = []
    bank = d1_bank()
    for index, spec in enumerate(bank):
        for draw in range(D1_DRAWS):
            tasks.append(sample_dsl_task(spec, random.Random(7000 + index * 20 + draw), 2))
    return tasks


def _evaluate(model: LocalNet) -> dict:
    started = time.time()
    compose_rows: dict[str, dict[str, list[dict]]] = {condition: {family: [] for family in ALL} for condition in CONDITIONS}
    for index, task in enumerate(_compose_tasks()):
        family = task.family
        for cond_index, condition in enumerate(CONDITIONS):
            rng = random.Random(9000 + index * 10 + cond_index)
            row = _score(task, _hypotheses(condition, task, model, rng))
            compose_rows[condition][family].append(row)
        if (index + 1) % 16 == 0:
            print(f"version space compose {index + 1}/{DRAWS * len(ALL)} ({time.time() - started:.0f}s)", flush=True)
    d1_tasks = _d1_tasks()
    d1_rows: dict[str, list[dict]] = {condition: [] for condition in CONDITIONS}
    for index, task in enumerate(d1_tasks):
        for cond_index, condition in enumerate(CONDITIONS):
            rng = random.Random(11000 + index * 10 + cond_index)
            row = _score(task, _hypotheses(condition, task, model, rng))
            d1_rows[condition].append(row)
        if (index + 1) % 8 == 0 or index + 1 == len(d1_tasks):
            print(f"version space d1 {index + 1}/{len(d1_tasks)} ({time.time() - started:.0f}s)", flush=True)
    families = {}
    for condition in CONDITIONS:
        families[condition] = {}
        for family in ALL:
            rows = compose_rows[condition][family]
            families[condition][family] = {
                "oracle": _mean(rows, "version"),
                "beam": _mean(rows, "beam"),
                "open_repeat": _mean(rows, "open"),
                "programs": _mean(rows, "programs"),
                "truncated": sum(row["truncated"] for row in rows),
                "n": len(rows),
            }
    d1 = {}
    for condition in CONDITIONS:
        rows = d1_rows[condition]
        d1[condition] = {
            "oracle": _mean(rows, "version"),
            "beam": _mean(rows, "beam"),
            "programs": _mean(rows, "programs"),
            "truncated": sum(row["truncated"] for row in rows),
            "n": len(rows),
        }
    return {"families": families, "d1": d1, "seconds": time.time() - started}


def _overall(block: dict, families: tuple[str, ...]) -> dict:
    rows = [block[family] for family in families]
    n = sum(row["n"] for row in rows)
    return {
        "oracle": sum(row["oracle"] * row["n"] for row in rows) / n,
        "beam": sum(row["beam"] * row["n"] for row in rows) / n,
        "programs": sum(row["programs"] * row["n"] for row in rows) / n,
        "truncated": sum(row["truncated"] for row in rows),
    }


def _write(root: Path, metrics: dict) -> None:
    out = root / "results" / "version_space"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    lines = [
        "# Version-space chart",
        "",
        "P2a freezes the L1 recognizer and replaces the scored beam with a bottom-up version space. Every gold local fact is present in every condition. Extra hypotheses are added on purpose: one per head, two per head, then the frozen model's own top 8. A partial program is dropped only when it misses a support. Distinct repeat bounds stay in the set together.",
        "",
        "The program oracle is the fraction of tasks whose version space contains a support-consistent program that matches the hidden probe. The beam column is the old chart, limited to eight programs, on the same facts. BENCH-C is not scored.",
        "",
        f"Draws per compose family: {DRAWS}. Draws per D1 schema: {D1_DRAWS}. Elapsed {metrics['seconds']:.0f}s.",
        "",
        "| Condition | Compose oracle | Compose beam@8 | Compose programs | Truncated | D1 oracle | D1 beam@8 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    labels = {
        "gold": "Gold facts",
        "plus1": "Gold + 1 distractor/head",
        "plus2": "Gold + 2 distractors/head",
        "model": "Gold ∪ model top 8",
    }
    for condition in CONDITIONS:
        compose = _overall(metrics["families"][condition], ALL)
        d1 = metrics["d1"][condition]
        lines.append(
            f"| {labels[condition]} | {compose['oracle']:.3f} | {compose['beam']:.3f} | {compose['programs']:.1f} | "
            f"{compose['truncated']} | {d1['oracle']:.3f} | {d1['beam']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Gold facts union the model's top 8",
            "",
            "| Family | Version-space oracle | Beam@8 oracle | Open-repeat oracle | Programs | Truncated |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for family in ALL:
        view = metrics["families"]["model"][family]
        lines.append(
            f"| {family} | {view['oracle']:.3f} | {view['beam']:.3f} | {view['open_repeat']:.3f} | "
            f"{view['programs']:.1f} | {view['truncated']} |"
        )
    compose_model = _overall(metrics["families"]["model"], ALL)
    atomic = _overall(metrics["families"]["model"], ATOMIC)
    held = _overall(metrics["families"]["model"], OOD)
    lines.extend(
        [
            "",
            f"With gold facts union the top 8, atomic oracle is {atomic['oracle']:.3f} and held-out oracle is {held['oracle']:.3f}. "
            f"Across all four conditions the compose oracle is "
            + ", ".join(f"{labels[condition]} {_overall(metrics['families'][condition], ALL)['oracle']:.3f}" for condition in CONDITIONS)
            + f". Truncated searches: {sum(_overall(metrics['families'][condition], ALL)['truncated'] for condition in CONDITIONS) + sum(metrics['d1'][condition]['truncated'] for condition in CONDITIONS)}.",
            "",
        ]
    )
    gate = min(_overall(metrics["families"][condition], ALL)["oracle"] for condition in CONDITIONS)
    gate = min(gate, min(metrics["d1"][condition]["oracle"] for condition in CONDITIONS))
    if gate >= 0.99 and compose_model["truncated"] == 0:
        lines.append(
            "The symbolic chart assembles a probe-matching program whenever the gold local facts are present, distractors included. "
            "Held-out local recall is still the open number. Scope relations between spans are the next place to train, and the recognizer weights stay as they are until that training starts. "
            "A pretrained backbone and BENCH-C stay off."
        )
    else:
        lines.append(
            f"The version space reaches {gate:.3f} oracle on the weakest condition. "
            "That is still short of a complete chart, so the recognizer stays frozen."
        )
    lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'summary.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("eval",), default="eval")
    args = parser.parse_args()
    del args
    root = _root()
    torch.set_num_threads(4)
    model = _load_model(root)
    metrics = _evaluate(model)
    metrics["arch"] = "version-space"
    metrics["checkpoint"] = "results/local_chart/l1/seed_0/checkpoint.pt"
    metrics["draws"] = DRAWS
    _write(root, metrics)


if __name__ == "__main__":
    main()
