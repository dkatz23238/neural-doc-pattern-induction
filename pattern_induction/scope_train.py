"""Train scope relations on a frozen local recognizer.

Checkpoint selection watches atomic local recall only. Held-out local recall
and the version-space program oracle are measured after the checkpoint is chosen.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import torch

from pattern_induction.chart import LOCAL_K, gold_facts
from pattern_induction.local_train import ALL, ATOMIC, BATCH, OOD, _add_recall, _empty_counts, _rates, _sample_batch, _signature
from pattern_induction.membership import Episode
from pattern_induction.program import OPEN, Prog, Task, agrees, feature_batch, sample_compose_task
from pattern_induction.recipes import d1_bank
from pattern_induction.renaming import rename_episode
from pattern_induction.schema_dsl import GENERATOR_VERSION, bank_hash
from pattern_induction.scope import ScopeNet, collate_scope, hypotheses_from_scope, scope_loss, total_parameter_count
from pattern_induction.version_space import assemble

PARAM_LO = 1_000_000
PARAM_HI = 2_000_000
STEPS = 800
LR = 1e-3
L1_CHECKPOINT = "results/local_chart/l1/seed_0/checkpoint.pt"


def _interpretation(local_recall: float, program_oracle: float, invariance: float) -> str:
    if invariance < 1.0:
        return "renaming invariance is below 1.000, so this checkpoint waits"
    if local_recall < 0.95:
        return "held-out local recall is still the gate, and the chart stays frozen"
    if program_oracle >= 0.90:
        return "held-out local recall cleared 0.95 and the version-space oracle recovered programs from the model's own facts"
    return "held-out local recall cleared 0.95, and the version-space oracle on the model's facts is the measurement that follows"


def _open_repeat(prog: Prog) -> bool:
    if prog.op == "REPEAT" and prog.hi >= OPEN:
        return True
    return any(_open_repeat(child) for child in prog.children)


def _score_tasks(model: ScopeNet, tasks: list[Task], k: int = LOCAL_K, with_oracle: bool = False) -> dict:
    recall_k = _empty_counts()
    recall_1 = _empty_counts()
    oracle = open_hit = emitted = truncated = 0
    for task in tasks:
        batch = collate_scope([task])
        hyps = hypotheses_from_scope(model, batch, [task.supports[:4]], k)[0]
        facts = gold_facts(task.program, task.supports)
        _add_recall(recall_k, facts, hyps, k)
        _add_recall(recall_1, facts, hyps, 1)
        if not with_oracle:
            continue
        built = assemble(task.supports, hyps)
        emitted += len(built.programs)
        truncated += int(built.truncated)
        matched = [prog for prog, _score in built.programs if agrees(prog, task.supports, task.probe)]
        if matched:
            oracle += 1
            if any(_open_repeat(prog) for prog in matched):
                open_hit += 1
    n = max(len(tasks), 1)
    return {
        "local": _rates(recall_k),
        "local_at_1": _rates(recall_1),
        "local_recall": _rates(recall_k)["mean"],
        "local_recall_at_1": _rates(recall_1)["mean"],
        "oracle": oracle / n if with_oracle else None,
        "open_repeat": open_hit / n if with_oracle else None,
        "programs": emitted / n if with_oracle else None,
        "truncated": truncated if with_oracle else None,
        "n": len(tasks),
    }


def _atomic_tasks(draws: int, seed: int) -> list[Task]:
    return [
        sample_compose_task(family, random.Random(seed + index * 20 + draw), 2)
        for index, family in enumerate(ATOMIC)
        for draw in range(draws)
    ]


def _select_score(model: ScopeNet, seed: int) -> float:
    tasks = _atomic_tasks(4, seed)
    reports = {family: _score_tasks(model, [task for task in tasks if task.family == family]) for family in ATOMIC}
    return sum(reports[family]["local_recall"] for family in ATOMIC) / len(ATOMIC)


def _invariance(model: ScopeNet, tasks: list[Task]) -> float:
    matched = 0
    model.local.eval()
    for index, task in enumerate(tasks):
        episode = Episode(task.supports[:4], task.supports[0], 1, task.family, "scope", task.family)
        renamed = rename_episode(episode, random.Random(8000 + index), True)
        left = feature_batch([task])
        right = feature_batch([Task(task.family, renamed.support, task.program, [])])
        if not all(torch.allclose(a, b) for a, b in zip(left, right)):
            continue
        a = hypotheses_from_scope(model, collate_scope([task]), [task.supports], 4)[0]
        b = hypotheses_from_scope(model, collate_scope([Task(task.family, renamed.support, task.program, [])]), [renamed.support], 4)[0]
        if _signature(a) == _signature(b):
            matched += 1
    return matched / max(len(tasks), 1)


def _load_local(root: Path, model: ScopeNet) -> None:
    path = root / L1_CHECKPOINT
    if not path.exists():
        raise RuntimeError(f"missing frozen recognizer at {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.local.load_state_dict(state)
    model.local.eval()
    for param in model.local.parameters():
        param.requires_grad_(False)


def _snapshot(model: ScopeNet) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_scope(root: Path, seed: int = 0) -> dict:
    bank = d1_bank()
    out = root / "results" / "scope" / "p2b" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = ScopeNet()
    _load_local(root, model)
    count = total_parameter_count(model)
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if not PARAM_LO <= count <= PARAM_HI:
        raise RuntimeError(f"scope recognizer parameter count {count} left the 1M–2M band")
    opt = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=LR, weight_decay=0.01)
    started = time.time()
    model.local.eval()
    best_score = _select_score(model, 1000)
    best_state = _snapshot(model)
    best_step = 0
    curve = [{"step": 0, "loss": None, "pair": None, "select_local": best_score, "seconds": time.time() - started}]
    print(f"p2b seed {seed} step 0: local {best_score:.3f} ({curve[-1]['seconds']:.0f}s)", flush=True)
    for step in range(1, STEPS + 1):
        tasks = _sample_batch(rng, step, bank, BATCH)
        batch = collate_scope(tasks)
        model.local.eval()
        logits = model(batch)
        loss, pair = scope_loss(logits, batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_((param for param in model.parameters() if param.requires_grad), 1.0)
        opt.step()
        if step % 200 == 0 or step == STEPS:
            model.local.eval()
            score = _select_score(model, 1000)
            row = {"step": step, "loss": float(loss.detach()), "pair": float(pair), "select_local": score, "seconds": time.time() - started}
            curve.append(row)
            print(f"p2b seed {seed} step {step}: loss {row['loss']:.3f} local {score:.3f} ({row['seconds']:.0f}s)", flush=True)
            if score >= best_score:
                best_score = score
                best_step = step
                best_state = _snapshot(model)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.local.eval()
    for param in model.local.parameters():
        param.requires_grad_(False)
    final = {}
    for index, family in enumerate(ALL):
        tasks = [sample_compose_task(family, random.Random(5000 + index * 50 + draw), 3) for draw in range(16)]
        final[family] = _score_tasks(model, tasks, LOCAL_K, with_oracle=True)
        print(
            f"p2b seed {seed} {family}: local {final[family]['local_recall']:.3f} oracle {final[family]['oracle']:.3f} "
            f"programs {final[family]['programs']:.1f}",
            flush=True,
        )
    inv_tasks = [sample_compose_task("repeat", random.Random(80 + index), 1) for index in range(6)]
    invariance = _invariance(model, inv_tasks)
    ood_local = sum(final[family]["local_recall"] for family in OOD) / len(OOD)
    ood_local_1 = sum(final[family]["local_recall_at_1"] for family in OOD) / len(OOD)
    ood_oracle = sum(final[family]["oracle"] for family in OOD) / len(OOD)
    atomic_local = sum(final[family]["local_recall"] for family in ATOMIC) / len(ATOMIC)
    atomic_oracle = sum(final[family]["oracle"] for family in ATOMIC) / len(ATOMIC)
    records = [{"id": item.ident, "schema": item.schema} for item in bank]
    metrics = {
        "arch": "p2b",
        "seed": seed,
        "parameters": count,
        "trainable": trainable,
        "steps": STEPS,
        "batch": BATCH,
        "lr": LR,
        "local_k": LOCAL_K,
        "generator_version": GENERATOR_VERSION,
        "template_bank_hash": bank_hash(records),
        "frozen_checkpoint": L1_CHECKPOINT,
        "selected_step": best_step,
        "select_local": best_score,
        "atomic_local": atomic_local,
        "atomic_oracle": atomic_oracle,
        "ood_local": ood_local,
        "ood_local_at_1": ood_local_1,
        "ood_oracle": ood_oracle,
        "min_invariance": invariance,
        "truncated": sum(final[family]["truncated"] for family in ALL),
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
        json.dumps({"selected_step": best_step, "parameters": count, "trainable": trainable, "sha256": hashlib.sha256(blob).hexdigest()}, indent=2) + "\n"
    )
    print(
        f"p2b seed {seed}: local@8 {ood_local:.3f} local@1 {ood_local_1:.3f} "
        f"oracle {ood_oracle:.3f} invariance {invariance:.3f} [{metrics['interpretation']}]",
        flush=True,
    )
    return metrics
