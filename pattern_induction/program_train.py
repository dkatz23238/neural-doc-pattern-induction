"""Train the program proposer. The executor, not the loss, decides membership.

Checkpoint selection watches atomic languages only. Held-out compositions,
including repeats of depth 4–5, are scored after selection.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import torch

from pattern_induction.program import (
    MAX_PROG,
    Task,
    accepts_all,
    agrees,
    linearize,
    sample_compose_task,
    sample_dsl_task,
)
from pattern_induction.proposer import Proposer, _pack_steps, collate_tasks, program_loss, token_logprob
from pattern_induction.recipes import d1_bank
from pattern_induction.renaming import rename_episode
from pattern_induction.schema_dsl import GENERATOR_VERSION, bank_hash
from pattern_induction.membership import Episode
from pattern_induction.program import feature_batch
from pattern_induction.span_net import parameter_count

PARAM_LO = 1_000_000
PARAM_HI = 2_000_000
ATOMIC = ("primitive", "repeat", "optional", "alternation")
OOD = ("prefix_repeat", "repeat_optional", "prefix_alt", "optional_internal", "repeat_extra")
ALL = ATOMIC + OOD
STEPS = 1200
BATCH = 8
PRIOR = 0.02


def _interpretation(ood_top1: float, ood_oracle16: float, extra_top1: float, invariance: float, support: float) -> str:
    if invariance < 0.99:
        return "reject: renaming invariance moved"
    if support < 0.99:
        return "executor filter is not keeping a program that accepts the supports"
    if ood_top1 > 0.90 and ood_oracle16 > 0.98 and extra_top1 > 0.90:
        return "structure induction is working; the emitted program is usually the right language"
    if ood_oracle16 > 0.95 and ood_top1 <= 0.90:
        return "a correct program is in the beam; selection is the remaining gap"
    if ood_oracle16 > 0.90:
        return "the beam often contains a correct program and is short of the 0.98 oracle bar"
    return "the proposer is not yet putting a correct program in the top 16"


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
        raise RuntimeError("could not fill a program batch")
    return tasks


def _corrupt_steps(tasks: list[Task]):
    from pattern_induction.program import corrupt

    programs = []
    mask = []
    for task in tasks:
        bad = corrupt(task.program, task.supports)
        if bad is None or len(linearize(bad)) > MAX_PROG:
            programs.append(task.program)
            mask.append(False)
        else:
            programs.append(bad)
            mask.append(True)
    packed = _pack_steps([Task("corrupt", task.supports, prog, []) for task, prog in zip(tasks, programs)])
    return packed["steps"], torch.tensor(mask)


def _select(beam, supports):
    passed = [(prog, score - PRIOR * prog.nodes()) for prog, score in beam if accepts_all(prog, supports, supports)]
    if not passed:
        return None
    return max(passed, key=lambda item: item[1])[0]


def _score_tasks(model: Proposer, tasks: list[Task], k: int = 16) -> dict:
    top1 = oracle4 = oracle16 = raw1 = support = 0
    beam_n = 0
    for task in tasks:
        batch = collate_tasks([task])
        beam = model.propose(batch["local"], batch["classes"], batch["rel"], batch["pad"], k, [task.supports])[0]
        beam_n += len(beam)
        if beam and agrees(beam[0][0], task.supports, task.probe):
            raw1 += 1
        if any(agrees(prog, task.supports, task.probe) for prog, _score in beam[:4]):
            oracle4 += 1
        if any(agrees(prog, task.supports, task.probe) for prog, _score in beam[:k]):
            oracle16 += 1
        chosen = _select(beam, task.supports)
        if chosen is not None:
            support += 1
            if agrees(chosen, task.supports, task.probe):
                top1 += 1
    n = max(len(tasks), 1)
    return {
        "top1": top1 / n,
        "raw1": raw1 / n,
        "oracle4": oracle4 / n,
        "oracle16": oracle16 / n,
        "support": support / n,
        "beam": beam_n / n,
        "n": len(tasks),
    }


def _invariance(model: Proposer, tasks: list[Task]) -> float:
    matched = 0
    model.eval()
    for index, task in enumerate(tasks):
        episode = Episode(task.supports[:4], task.supports[0], 1, task.family, "p1", task.family)
        renamed = rename_episode(episode, random.Random(8000 + index), True)
        left = feature_batch([task])
        right = feature_batch([Task(task.family, renamed.support, task.program, [])])
        if not all(torch.allclose(a, b) for a, b in zip(left, right)):
            continue
        a = model.propose(*left, 1, [task.supports])[0]
        b = model.propose(*right, 1, [renamed.support])[0]
        if not a and not b:
            matched += 1
            continue
        if a and b and linearize(a[0][0]) == linearize(b[0][0]):
            matched += 1
    return matched / max(len(tasks), 1)


def train_proposer(root: Path, seed: int = 0) -> dict:
    bank = d1_bank()
    out = root / "results" / "programs" / "p1" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = Proposer()
    count = parameter_count(model)
    if not PARAM_LO <= count <= PARAM_HI:
        raise RuntimeError(f"proposer parameter count {count} left the 1M–2M band")
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    best_score = -1.0
    best_state = None
    best_step = 0
    started = time.time()
    curve = []
    for step in range(1, STEPS + 1):
        tasks = _sample_batch(rng, step, bank, BATCH)
        batch = collate_tasks(tasks)
        corrupt_steps, corrupt_mask = _corrupt_steps(tasks)
        model.train()
        encoded, memory, key_pad = model.encode(batch["local"], batch["classes"], batch["rel"], batch["pad"])
        logits = model.decode(memory, key_pad, batch["steps"], MAX_PROG)
        aux = {"role": model.char_role(encoded), "scope": model.char_scope(encoded)}
        gold_lp = token_logprob(logits, batch["steps"])
        corrupt_logits = model.decode(memory, key_pad, corrupt_steps, MAX_PROG)
        corrupt_lp = token_logprob(corrupt_logits, corrupt_steps)
        loss, op_loss, rank = program_loss(logits, aux, batch, gold_lp, corrupt_lp, corrupt_mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == STEPS:
            model.eval()
            select_tasks = [
                sample_compose_task(family, random.Random(1000 + index * 20 + draw), 4)
                for index, family in enumerate(ATOMIC)
                for draw in range(8)
            ]
            reports = {family: _score_tasks(model, [task for task in select_tasks if task.family == family], 8) for family in ATOMIC}
            score = sum(reports[family]["top1"] for family in ATOMIC) / len(ATOMIC)
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "op": float(op_loss),
                "rank": float(rank),
                "select_top1": score,
                "seconds": time.time() - started,
            }
            curve.append(row)
            print(
                f"p1 seed {seed} step {step}: loss {row['loss']:.3f} op {row['op']:.3f} "
                f"select {score:.3f} ({row['seconds']:.0f}s)",
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
            [sample_compose_task(family, random.Random(5000 + index * 50 + draw), 4) for draw in range(24)],
            16,
        )
        for index, family in enumerate(ALL)
    }
    inv_tasks = [sample_compose_task("repeat", random.Random(80 + index), 2) for index in range(8)]
    invariance = _invariance(model, inv_tasks)
    ood_top1 = sum(final[family]["top1"] for family in OOD) / len(OOD)
    ood_oracle4 = sum(final[family]["oracle4"] for family in OOD) / len(OOD)
    ood_oracle16 = sum(final[family]["oracle16"] for family in OOD) / len(OOD)
    ood_support = min(final[family]["support"] for family in OOD)
    extra = final["repeat_extra"]["top1"]
    atomic_top1 = sum(final[family]["top1"] for family in ATOMIC) / len(ATOMIC)
    mean_beam = sum(final[family]["beam"] for family in ALL) / len(ALL)
    records = [{"id": item.ident, "schema": item.schema} for item in bank]
    metrics = {
        "arch": "p1",
        "seed": seed,
        "parameters": count,
        "steps": STEPS,
        "batch": BATCH,
        "generator_version": GENERATOR_VERSION,
        "template_bank_hash": bank_hash(records),
        "selected_step": best_step,
        "select_top1": best_score,
        "atomic_top1": atomic_top1,
        "ood_top1": ood_top1,
        "ood_oracle4": ood_oracle4,
        "ood_oracle16": ood_oracle16,
        "ood_support": ood_support,
        "repeat_extra_top1": extra,
        "mean_beam": mean_beam,
        "min_invariance": invariance,
        "interpretation": _interpretation(ood_top1, ood_oracle16, extra, invariance, ood_support),
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
        f"p1 seed {seed}: ood top1 {ood_top1:.3f} oracle@4 {ood_oracle4:.3f} oracle@16 {ood_oracle16:.3f} "
        f"extra {extra:.3f} support {ood_support:.3f} invariance {invariance:.3f} [{metrics['interpretation']}]",
        flush=True,
    )
    return metrics
