"""Train local structural hypotheses and measure them before the chart runs.

Checkpoint selection watches atomic local recall only. Held-out compositions,
including repeats of depth 4–5, are scored after selection.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import torch

from pattern_induction.chart import LOCAL_K, compose, gold_facts, recall
from pattern_induction.local_net import LocalNet, collate_local, hypotheses_from_batch, local_loss
from pattern_induction.membership import Episode
from pattern_induction.program import Task, accepts_all, agrees, feature_batch, sample_compose_task, sample_dsl_task
from pattern_induction.recipes import d1_bank
from pattern_induction.renaming import rename_episode
from pattern_induction.schema_dsl import GENERATOR_VERSION, bank_hash
from pattern_induction.span_net import parameter_count

PARAM_LO = 1_000_000
PARAM_HI = 2_000_000
ATOMIC = ("primitive", "repeat", "optional", "alternation")
OOD = ("prefix_repeat", "repeat_optional", "prefix_alt", "optional_internal", "repeat_extra")
ALL = ATOMIC + OOD
STEPS = 1200
BATCH = 8
KINDS = ("copy", "cls", "repeat", "optional", "alt", "pair", "cut")


def _interpretation(local_recall: float, program_oracle: float, invariance: float) -> str:
    if invariance < 0.99:
        return "reject: renaming invariance moved"
    if local_recall < 0.50:
        return "the neural recognizer is the bottleneck: gold local structure is outside the top hypotheses"
    if program_oracle < 0.40:
        return "gold local structure is in the top hypotheses and the chart is not assembling a correct program"
    return "local hypotheses and the chart both carry a large share of the held-out languages"


def _sample_batch(rng: random.Random, step: int, bank, size: int) -> list[Task]:
    tasks = []
    cursor = 0
    while len(tasks) < size and cursor < size * 6:
        cursor += 1
        try:
            if step % 2 == 1:
                family = ATOMIC[((step // 2) + len(tasks)) % len(ATOMIC)]
                tasks.append(sample_compose_task(family, rng, probe_n=2))
            else:
                compiled = bank[((step // 2) + len(tasks)) % len(bank)]
                tasks.append(sample_dsl_task(compiled, rng, probe_n=2))
        except RuntimeError:
            continue
    if len(tasks) < size:
        raise RuntimeError("could not fill a local-hypothesis batch")
    return tasks


def _select_program(programs, supports):
    passed = [(prog, score) for prog, score in programs if accepts_all(prog, supports, supports)]
    if not passed:
        return None
    return max(passed, key=lambda item: item[1])[0]


def _empty_counts() -> dict[str, list[int]]:
    return {kind: [0, 0] for kind in KINDS}


def _add_recall(store: dict[str, list[int]], facts, hyps, k: int) -> None:
    kept = _truncate(hyps, k)
    for kind in KINDS:
        hits, total = recall(facts, kept, kind, bounds=True)
        store[kind][0] += hits
        store[kind][1] += total


def _truncate(hyps, k: int):
    grouped: dict[tuple[str, int], list] = {}
    for hyp in hyps:
        grouped.setdefault((hyp.kind, hyp.slot), []).append(hyp)
    kept = []
    for rows in grouped.values():
        rows.sort(key=lambda item: item.score, reverse=True)
        kept.extend(rows[:k])
    return kept


def _rates(store: dict[str, list[int]]) -> dict[str, float]:
    rates = {}
    for kind, (hits, total) in store.items():
        rates[kind] = hits / total if total else None
    present = [rate for rate in rates.values() if rate is not None]
    rates["mean"] = sum(present) / len(present) if present else 0.0
    return rates


def _score_tasks(model: LocalNet, tasks: list[Task], k: int = LOCAL_K) -> dict:
    recall_k = _empty_counts()
    recall_1 = _empty_counts()
    top1 = oracle = support = 0
    emitted = 0
    for task in tasks:
        batch = collate_local([task])
        hyps = hypotheses_from_batch(model, batch, [task.supports[:4]], k)[0]
        facts = gold_facts(task.program, task.supports)
        _add_recall(recall_k, facts, hyps, k)
        _add_recall(recall_1, facts, hyps, 1)
        programs = compose(task.supports, hyps, k)
        emitted += len(programs)
        if any(agrees(prog, task.supports, task.probe) for prog, _score in programs):
            oracle += 1
        chosen = _select_program(programs, task.supports)
        if chosen is not None:
            support += 1
            if agrees(chosen, task.supports, task.probe):
                top1 += 1
    n = max(len(tasks), 1)
    return {
        "local": _rates(recall_k),
        "local_at_1": _rates(recall_1),
        "local_recall": _rates(recall_k)["mean"],
        "local_recall_at_1": _rates(recall_1)["mean"],
        "top1": top1 / n,
        "oracle": oracle / n,
        "support": support / n,
        "programs": emitted / n,
        "n": len(tasks),
    }


def _invariance(model: LocalNet, tasks: list[Task]) -> float:
    matched = 0
    model.eval()
    for index, task in enumerate(tasks):
        episode = Episode(task.supports[:4], task.supports[0], 1, task.family, "local", task.family)
        renamed = rename_episode(episode, random.Random(8000 + index), True)
        left = feature_batch([task])
        right = feature_batch([Task(task.family, renamed.support, task.program, [])])
        if not all(torch.allclose(a, b) for a, b in zip(left, right)):
            continue
        a = hypotheses_from_batch(model, collate_local([task]), [task.supports], 4)[0]
        b = hypotheses_from_batch(model, collate_local([Task(task.family, renamed.support, task.program, [])]), [renamed.support], 4)[0]
        if _signature(a) == _signature(b):
            matched += 1
    return matched / max(len(tasks), 1)


def _signature(hyps) -> tuple:
    return tuple(sorted((hyp.kind, hyp.slot, hyp.start, hyp.end, hyp.cls, hyp.lo, hyp.hi, hyp.slot2, hyp.start2, hyp.end2, hyp.index) for hyp in hyps))


def train_local(root: Path, seed: int = 0) -> dict:
    bank = d1_bank()
    out = root / "results" / "local_chart" / "l1" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = LocalNet()
    count = parameter_count(model)
    if not PARAM_LO <= count <= PARAM_HI:
        raise RuntimeError(f"local recognizer parameter count {count} left the 1M–2M band")
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    best_score = -1.0
    best_state = None
    best_step = 0
    started = time.time()
    curve = []
    for step in range(1, STEPS + 1):
        tasks = _sample_batch(rng, step, bank, BATCH)
        batch = collate_local(tasks)
        model.train()
        logits = model(batch)
        loss, atom = local_loss(logits, batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == STEPS:
            model.eval()
            select_tasks = [
                sample_compose_task(family, random.Random(1000 + index * 20 + draw), 2)
                for index, family in enumerate(ATOMIC)
                for draw in range(4)
            ]
            reports = {family: _score_tasks(model, [task for task in select_tasks if task.family == family], LOCAL_K) for family in ATOMIC}
            score = sum(reports[family]["local_recall"] for family in ATOMIC) / len(ATOMIC)
            row = {"step": step, "loss": float(loss.detach()), "atom": float(atom), "select_local": score, "seconds": time.time() - started}
            curve.append(row)
            print(
                f"l1 seed {seed} step {step}: loss {row['loss']:.3f} local {score:.3f} ({row['seconds']:.0f}s)",
                flush=True,
            )
            if score >= best_score:
                best_score = score
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    final = {
        family: _score_tasks(
            model,
            [sample_compose_task(family, random.Random(5000 + index * 50 + draw), 3) for draw in range(16)],
            LOCAL_K,
        )
        for index, family in enumerate(ALL)
    }
    inv_tasks = [sample_compose_task("repeat", random.Random(80 + index), 1) for index in range(6)]
    invariance = _invariance(model, inv_tasks)
    ood_local = sum(final[family]["local_recall"] for family in OOD) / len(OOD)
    ood_local_1 = sum(final[family]["local_recall_at_1"] for family in OOD) / len(OOD)
    ood_oracle = sum(final[family]["oracle"] for family in OOD) / len(OOD)
    ood_top1 = sum(final[family]["top1"] for family in OOD) / len(OOD)
    atomic_local = sum(final[family]["local_recall"] for family in ATOMIC) / len(ATOMIC)
    extra = final["repeat_extra"]["local_recall"]
    records = [{"id": item.ident, "schema": item.schema} for item in bank]
    metrics = {
        "arch": "l1",
        "seed": seed,
        "parameters": count,
        "steps": STEPS,
        "batch": BATCH,
        "local_k": LOCAL_K,
        "generator_version": GENERATOR_VERSION,
        "template_bank_hash": bank_hash(records),
        "selected_step": best_step,
        "select_local": best_score,
        "atomic_local": atomic_local,
        "ood_local": ood_local,
        "ood_local_at_1": ood_local_1,
        "ood_oracle": ood_oracle,
        "ood_top1": ood_top1,
        "repeat_extra_local": extra,
        "min_invariance": invariance,
        "interpretation": _interpretation(ood_local, ood_oracle, invariance),
        "families": final,
        "curve": curve,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    blob = bytearray()
    for key in sorted(best_state):
        blob.extend(best_state[key].numpy().tobytes())
    torch.save(best_state, out / "checkpoint.pt")
    (out / "checkpoint.json").write_text(
        json.dumps({"selected_step": best_step, "parameters": count, "sha256": hashlib.sha256(blob).hexdigest()}, indent=2) + "\n"
    )
    print(
        f"l1 seed {seed}: local@8 {ood_local:.3f} local@1 {ood_local_1:.3f} "
        f"oracle@8 {ood_oracle:.3f} top1 {ood_top1:.3f} invariance {invariance:.3f} [{metrics['interpretation']}]",
        flush=True,
    )
    return metrics
