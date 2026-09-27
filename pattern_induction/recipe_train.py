"""Train the frozen span model on one data recipe.

Optimizer, step count, batch size, loss, checkpoint rule, and the evaluation
slices match the compositional run. The only input that changes is the batch
drawn at each step.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path

import torch

from pattern_induction.compose import TRAIN_FAMILIES, build_split
from pattern_induction.recipes import recipe_sources, sample_recipe_batch
from pattern_induction.schema_dsl import GENERATOR_VERSION, bank_hash
from pattern_induction.span_net import SpanNet, collate, composition_loss, parameter_count

FROZEN = {
    "steps": 1200,
    "batch": 12,
    "lr": 8e-4,
    "weight_decay": 0.01,
    "eval_every": 100,
    "eval_n": 80,
    "final_n": 160,
    "select_families": TRAIN_FAMILIES,
}
COMP_FAMILIES = ("prefix_repeat", "repeat_optional", "prefix_alt")
EXTRA_FAMILIES = ("repeat_extra", "optional_internal")
ALL_FAMILIES = TRAIN_FAMILIES + EXTRA_FAMILIES + COMP_FAMILIES


def _compose_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_compose.py"
    spec = importlib.util.spec_from_file_location("run_compose_frozen", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _worst(slices: dict) -> tuple[str, float]:
    name = min(COMP_FAMILIES, key=lambda item: slices[item]["original"]["auroc"])
    return name, slices[name]["original"]["auroc"]


def _interpretation(mean_auroc: float, worst_auroc: float, invariance: float, extra_auroc: float) -> str:
    if invariance < 0.99:
        return "reject: renaming invariance moved"
    if mean_auroc >= 0.93 and extra_auroc >= 0.90 and worst_auroc >= 0.85:
        return "decoder threshold: mean, worst composition, and repeat extrapolation are all high"
    if mean_auroc > 0.90 and worst_auroc > 0.85:
        return "major success on held-out composition"
    if mean_auroc >= 0.82:
        return "data moved composition, short of the 0.90 bar"
    if mean_auroc >= 0.78 and worst_auroc >= 0.65:
        return "composition rose from the atomic baseline and remains short of the 0.82 band"
    return "data alone is not solving composition"


def _write_curves(path: Path, rows: list[dict]) -> None:
    fields = ["step", "loss", "membership", "aux", "select_auroc", "primitive", "repeat", "optional", "alternation", "seconds"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _failures(episodes, probs) -> list[dict]:
    rows = []
    for episode, prob in zip(episodes, probs):
        decision = prob >= 0.5
        if decision == bool(episode.label):
            continue
        rows.append(
            {
                "family": episode.family,
                "support": episode.support,
                "query": episode.query,
                "label": episode.label,
                "probability": round(prob, 6),
                "pattern": episode.pattern,
            }
        )
    return rows


def train_recipe(root: Path, recipe: str, seed: int = 0) -> dict:
    banks = recipe_sources(root)
    if recipe not in banks:
        raise RuntimeError(f"{recipe} has no validated template bank")
    bank, _exposes = banks[recipe]
    exposes = sorted({item.heldout for item in (bank or []) if item.heldout})
    out = root / "results" / "data_recipe" / recipe / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = SpanNet()
    count = parameter_count(model)
    if not 900_000 <= count <= 1_050_000:
        raise RuntimeError(f"parameter count {count} left the frozen band")
    opt = torch.optim.AdamW(model.parameters(), lr=FROZEN["lr"], weight_decay=FROZEN["weight_decay"])
    compose = _compose_script()
    select_splits = {
        family: build_split(family, random.Random(1000 + index), FROZEN["eval_n"])
        for index, family in enumerate(FROZEN["select_families"])
    }
    best_score = -1.0
    best_state = None
    best_step = 0
    curve = []
    started = time.time()
    for step in range(1, FROZEN["steps"] + 1):
        batch = sample_recipe_batch(recipe, rng, step, FROZEN["batch"], bank)
        packed = collate(batch, max_len=model.max_len, max_support=model.max_support)
        model.train()
        logit, aux = model(*packed[:4])
        loss, membership, aux_loss = composition_loss(logit, aux, *packed[4:])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % FROZEN["eval_every"] == 0 or step == FROZEN["steps"]:
            reports = {
                name: compose._pack(select_splits[name], compose._predict(model, select_splits[name], 32))
                for name in FROZEN["select_families"]
            }
            score = sum(reports[name]["auroc"] for name in FROZEN["select_families"]) / len(FROZEN["select_families"])
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "membership": float(membership),
                "aux": float(aux_loss),
                "select_auroc": score,
                "primitive": reports["primitive"]["auroc"],
                "repeat": reports["repeat"]["auroc"],
                "optional": reports["optional"]["auroc"],
                "alternation": reports["alternation"]["auroc"],
                "seconds": time.time() - started,
            }
            curve.append(row)
            print(
                f"{recipe} seed {seed} step {step}: loss {row['loss']:.3f} select {score:.3f} "
                f"prim {row['primitive']:.3f} rep {row['repeat']:.3f} opt {row['optional']:.3f} "
                f"alt {row['alternation']:.3f} ({row['seconds']:.0f}s)",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    final_splits = {
        family: build_split(family, random.Random(5000 + index), FROZEN["final_n"]) for index, family in enumerate(ALL_FAMILIES)
    }
    slices = {family: compose._score_slice(model, final_splits[family], 32, 8000) for family in ALL_FAMILIES}
    worst_name, worst_auroc = _worst(slices)
    mean_auroc = sum(slices[name]["original"]["auroc"] for name in COMP_FAMILIES) / len(COMP_FAMILIES)
    invariance = min(slices[name]["symmetry"]["invariance"] for name in ALL_FAMILIES)
    extra = slices["repeat_extra"]["original"]["auroc"]
    records = [] if bank is None else [{"id": item.ident, "schema": item.schema} for item in bank]
    metrics = {
        "recipe": recipe,
        "seed": seed,
        "parameters": count,
        "frozen": FROZEN,
        "generator_version": GENERATOR_VERSION,
        "template_bank_hash": bank_hash(records) if records else "atomic",
        "templates": len(records),
        "exposes_eval_family": exposes,
        "selected_step": best_step,
        "select_auroc": best_score,
        "composition_auroc": mean_auroc,
        "worst_composition": {"family": worst_name, "auroc": worst_auroc},
        "repeat_extra_auroc": extra,
        "min_invariance": invariance,
        "interpretation": _interpretation(mean_auroc, worst_auroc, invariance, extra),
        "slices": {
            name: {
                "balanced": slices[name]["original"]["balanced"],
                "auroc": slices[name]["original"]["auroc"],
                "tpr": slices[name]["original"]["tpr"],
                "tnr": slices[name]["original"]["tnr"],
                "member_acceptance": slices[name]["original"]["member_acceptance"],
                "invariance": slices[name]["symmetry"]["invariance"],
                "renamed_auroc": slices[name]["renamed"]["auroc"],
            }
            for name in ALL_FAMILIES
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out / "slice_metrics.json").write_text(json.dumps(slices, indent=2) + "\n")
    _write_curves(out / "curves.csv", curve)
    mistakes = []
    for family in COMP_FAMILIES:
        probs = compose._predict(model, final_splits[family], 32)
        mistakes.extend(_failures(final_splits[family], probs))
    with (out / "failures.jsonl").open("w") as handle:
        for row in mistakes:
            handle.write(json.dumps(row) + "\n")
    blob = bytearray()
    for key in sorted(best_state):
        blob.extend(best_state[key].numpy().tobytes())
    torch.save(best_state, out / "checkpoint.pt")
    (out / "checkpoint.json").write_text(
        json.dumps(
            {
                "selected_step": best_step,
                "select_auroc": best_score,
                "parameters": count,
                "file": "checkpoint.pt",
                "sha256": hashlib.sha256(blob).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"{recipe} seed {seed}: composition {mean_auroc:.3f} worst {worst_name} {worst_auroc:.3f} "
        f"extra {extra:.3f} invariance {invariance:.3f} [{metrics['interpretation']}]",
        flush=True,
    )
    return metrics


def failure_packet(root: Path, recipe: str, seed: int = 0, limit: int = 8) -> str:
    path = root / "results" / "data_recipe" / recipe / f"seed_{seed}" / "failures.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: abs(row["probability"] - 0.5), reverse=True)
    packet = []
    for row in rows[:limit]:
        packet.append(
            {
                "failure_family": row["family"],
                "support": row["support"],
                "query": row["query"],
                "label": row["label"],
                "student_score": row["probability"],
                "gold_pattern": row["pattern"],
            }
        )
    return json.dumps({"failures": packet}, indent=2)
