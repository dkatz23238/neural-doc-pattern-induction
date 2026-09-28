"""Train a hierarchical group model on one frozen data recipe.

The optimizer, step count, batch size, checkpoint rule, and evaluation slices
match the compositional run. H0 is the span model already recorded under
results/data_recipe. H1 and H2 change the mixer above the spans and add
operator-tree targets. They do not change the strings.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from pattern_induction.compose import TRAIN_FAMILIES, build_split
from pattern_induction.group_net import GroupNet, batch_targets, hierarchy_loss
from pattern_induction.recipe_train import ALL_FAMILIES, COMP_FAMILIES, FROZEN, _compose_script, _failures, _worst
from pattern_induction.recipes import recipe_sources, sample_recipe_batch
from pattern_induction.schema_dsl import GENERATOR_VERSION, bank_hash
from pattern_induction.span_net import collate, parameter_count

PARAM_LO = 1_000_000
PARAM_HI = 2_000_000


def _interpretation(mean_auroc: float, worst_auroc: float, invariance: float, extra_auroc: float, atomic_min: float) -> str:
    if invariance < 0.99:
        return "reject: renaming invariance moved"
    if atomic_min < 0.90:
        return "atomic operators slipped"
    if mean_auroc > 0.90 and worst_auroc > 0.85 and extra_auroc > 0.90:
        return "hierarchy gate met"
    if mean_auroc > 0.90 and worst_auroc > 0.85:
        return "composition moved and repeat-depth extrapolation fell"
    if mean_auroc >= 0.82 and extra_auroc <= 0.90:
        return "composition moved and repeat-depth extrapolation is under 0.90"
    if mean_auroc >= 0.82:
        return "composition moved, short of the hierarchy gate"
    return "hierarchy did not clear the composition gate"


def _write_curves(path: Path, rows: list[dict]) -> None:
    fields = ["step", "loss", "membership", "aux", "select_auroc", "primitive", "repeat", "optional", "alternation", "seconds"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def train_hierarchy(root: Path, arch: str, recipe: str, seed: int = 0) -> dict:
    banks = recipe_sources(root)
    if recipe not in banks:
        raise RuntimeError(f"{recipe} has no validated template bank")
    bank, _exposes = banks[recipe]
    exposes = sorted({item.heldout for item in (bank or []) if item.heldout})
    out = root / "results" / "hierarchy" / arch / recipe / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = GroupNet(arch)
    count = parameter_count(model)
    if not PARAM_LO <= count <= PARAM_HI:
        raise RuntimeError(f"{arch} parameter count {count} left the 1M–2M band")
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
        targets = batch_targets(batch, packed[1], packed[2], packed[3], model.max_len)
        edge_logit = model.score_pairs(aux["groups"], targets["pair_src"], targets["pair_dst"])
        loss, membership, aux_loss = hierarchy_loss(logit, aux, *packed[4:], targets, edge_logit)
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
                f"{arch} {recipe} seed {seed} step {step}: loss {row['loss']:.3f} select {score:.3f} "
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
    model.eval()
    final_splits = {
        family: build_split(family, random.Random(5000 + index), FROZEN["final_n"]) for index, family in enumerate(ALL_FAMILIES)
    }
    slices = {family: compose._score_slice(model, final_splits[family], 32, 8000) for family in ALL_FAMILIES}
    worst_name, worst_auroc = _worst(slices)
    mean_auroc = sum(slices[name]["original"]["auroc"] for name in COMP_FAMILIES) / len(COMP_FAMILIES)
    invariance = min(slices[name]["symmetry"]["invariance"] for name in ALL_FAMILIES)
    extra = slices["repeat_extra"]["original"]["auroc"]
    atomic_min = min(slices[name]["original"]["auroc"] for name in TRAIN_FAMILIES)
    records = [] if bank is None else [{"id": item.ident, "schema": item.schema} for item in bank]
    metrics = {
        "arch": arch,
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
        "atomic_min_auroc": atomic_min,
        "min_invariance": invariance,
        "interpretation": _interpretation(mean_auroc, worst_auroc, invariance, extra, atomic_min),
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
                "arch": arch,
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
        f"{arch} {recipe} seed {seed}: composition {mean_auroc:.3f} worst {worst_name} {worst_auroc:.3f} "
        f"extra {extra:.3f} atomic {atomic_min:.3f} invariance {invariance:.3f} [{metrics['interpretation']}]",
        flush=True,
    )
    return metrics
