"""Select one language from the frozen version space.

Synthesis is unchanged. Families that accept the same strings are merged, a
grammar fit on gold schemas breaks near ties, support subsets vote for
stability, and a tiny witness head judges strings where families disagree.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from pattern_induction.chart import LOCAL_K
from pattern_induction.language import (
    GrammarPrior,
    boundary_strings,
    choose_family,
    disagreements,
    group_languages,
    repeat_extension,
)
from pattern_induction.local_train import ALL, ATOMIC, OOD
from pattern_induction.membership import Episode
from pattern_induction.program import (
    MAX_SUPPORT,
    Task,
    agrees,
    execute,
    feature_batch,
    sample_compose_task,
    sample_dsl_task,
)
from pattern_induction.recipes import d1_bank
from pattern_induction.renaming import rename_episode
from pattern_induction.schema_dsl import GENERATOR_VERSION, bank_hash
from pattern_induction.scope import ScopeNet, collate_scope, hypotheses_from_scope, total_parameter_count
from pattern_induction.version_space import assemble
from pattern_induction.witness_net import WitnessNet, collate_witnesses, witness_loss

SCOPE_CHECKPOINT = "results/scope/p2b/seed_0/checkpoint.pt"
STEPS = 400
BATCH = 16
LR = 1e-3
TIE = 0.3
PARAM_LO = 1_000_000
PARAM_HI = 2_000_000


def _load_scope(root: Path) -> ScopeNet:
    path = root / SCOPE_CHECKPOINT
    if not path.exists():
        raise RuntimeError(f"missing frozen scope recognizer at {path}")
    model = ScopeNet()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.local.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def synthesize(scope: ScopeNet, task: Task) -> list:
    batch = collate_scope([task])
    hyps = hypotheses_from_scope(scope, batch, [task.supports[:MAX_SUPPORT]], LOCAL_K)[0]
    return [prog for prog, _score in assemble(task.supports, hyps).programs]


def _subsets(supports: list[str]) -> list[list[str]]:
    rows = []
    for index in range(len(supports)):
        subset = [text for cursor, text in enumerate(supports) if cursor != index]
        if len(subset) >= 2:
            rows.append(subset)
    if len(supports) >= 4:
        rows.append(supports[:-2])
    if len(supports) >= 2:
        rows.append(supports[1:] + supports[:1])
    return rows


def _bits(prog, supports: list[str], battery: list[str]) -> tuple[bool, ...]:
    return tuple(bool(execute(prog, text, supports)) for text in battery)


def _stability(scope: ScopeNet, task: Task, families: list[list]) -> dict[int, float]:
    battery = boundary_strings(task.supports)
    target = {index: _bits(group[0], task.supports, battery) for index, group in enumerate(families)}
    hits = {index: 0 for index in target}
    subsets = _subsets(task.supports)
    if not subsets:
        return {index: 1.0 for index in target}
    for subset in subsets:
        programs = synthesize(scope, Task(task.family, subset, task.program, []))
        found = {_bits(prog, subset, battery) for prog in programs}
        for index, signature in target.items():
            if signature in found:
                hits[index] += 1
    return {index: hits[index] / len(subsets) for index in target}


def _judgments(witness: WitnessNet, supports: list[str], texts: list[str]) -> list[tuple[str, float]]:
    if not texts:
        return []
    batch = collate_witnesses([(supports, text) for text in texts])
    witness.encoder.eval()
    with torch.no_grad():
        probs = torch.sigmoid(witness(batch)).tolist()
    return list(zip(texts, [float(prob) for prob in probs]))


def select_program(
    scope: ScopeNet,
    witness: WitnessNet,
    prior: GrammarPrior,
    task: Task,
    with_stability: bool = True,
) -> tuple:
    programs = synthesize(scope, task)
    families = group_languages(programs, task.supports)
    if not families:
        return None, programs, 0
    stability: dict[int, float] = {}
    if with_stability and len(families) > 1 and len(task.supports) >= 3:
        stability = _stability(scope, task, families)
    texts = disagreements(families, task.supports) if len(families) > 1 else []
    judgments = _judgments(witness, task.supports, texts)
    chosen = choose_family(families, task.supports, judgments, prior, stability, TIE)
    return chosen, programs, len(families)


def _semantic(prog, task: Task) -> float:
    if prog is None:
        return 0.0
    positive = [text for text, label in task.probe if label]
    negative = [text for text, label in task.probe if not label]
    recall = sum(bool(execute(prog, text, task.supports)) for text in positive) / max(len(positive), 1)
    rejection = sum(not execute(prog, text, task.supports) for text in negative) / max(len(negative), 1)
    return 0.5 * (recall + rejection)


def _fit_prior(rng: random.Random, bank) -> GrammarPrior:
    prior = GrammarPrior()
    for family in ATOMIC:
        for draw in range(40):
            task = sample_compose_task(family, random.Random(rng.randrange(1_000_000)), 1)
            prior.observe(task.program, repeat_bounds=True)
    for draw in range(80):
        compiled = bank[draw % len(bank)]
        try:
            task = sample_dsl_task(compiled, random.Random(rng.randrange(1_000_000)), 1)
        except RuntimeError:
            continue
        prior.observe(task.program, repeat_bounds=False)
    return prior


def _collect(scope: ScopeNet, tasks: list[Task]) -> list[tuple[list[str], str, float]]:
    rows = []
    for task in tasks:
        for text in boundary_strings(task.supports):
            if repeat_extension(text, task.supports):
                continue
            label = 1.0 if execute(task.program, text, task.supports) else 0.0
            rows.append((task.supports, text, label))
    return rows


def _sample_tasks(rng: random.Random, bank, atomic_draws: int, d1_draws: int) -> list[Task]:
    tasks = []
    for family in ATOMIC:
        for _draw in range(atomic_draws):
            tasks.append(sample_compose_task(family, random.Random(rng.randrange(1_000_000)), 1))
    for draw in range(d1_draws):
        compiled = bank[draw % len(bank)]
        try:
            tasks.append(sample_dsl_task(compiled, random.Random(rng.randrange(1_000_000)), 1))
        except RuntimeError:
            continue
    return tasks


def _score_split(scope, witness, prior, tasks: list[Task], with_stability: bool) -> dict:
    top1 = semantic = oracle = families = 0.0
    for task in tasks:
        chosen, programs, count = select_program(scope, witness, prior, task, with_stability)
        families += count
        if any(agrees(prog, task.supports, task.probe) for prog in programs):
            oracle += 1
        if chosen is not None and agrees(chosen, task.supports, task.probe):
            top1 += 1
        semantic += _semantic(chosen, task)
    n = max(len(tasks), 1)
    return {
        "top1": top1 / n,
        "semantic": semantic / n,
        "oracle": oracle / n,
        "families": families / n,
        "n": len(tasks),
    }


def _invariance(scope, witness, prior, tasks: list[Task]) -> float:
    matched = 0
    for index, task in enumerate(tasks):
        episode = Episode(task.supports[:4], task.supports[0], 1, task.family, "select", task.family)
        renamed = rename_episode(episode, random.Random(8100 + index), True)
        other = Task(task.family, renamed.support, task.program, [])
        if not all(torch.allclose(a, b) for a, b in zip(feature_batch([task]), feature_batch([other]))):
            continue
        left, _programs, _count = select_program(scope, witness, prior, task, with_stability=False)
        right, _programs, _count = select_program(scope, witness, prior, other, with_stability=False)
        if left is None or right is None:
            continue
        from pattern_induction.program import describe
        from pattern_induction.language import flatten

        if describe(flatten(left)) == describe(flatten(right)):
            matched += 1
    return matched / max(len(tasks), 1)


def _interpretation(oracle: float, top1: float, worst: float, invariance: float) -> str:
    if invariance < 1.0:
        return "renaming invariance is below 1.000, so this selector waits"
    if oracle < 0.985:
        return "the frozen version-space oracle moved, so selection stops and synthesis stays as it was"
    if top1 >= 0.98 and worst >= 0.95:
        return "semantic top-1 cleared the stretch line on held-out composition"
    if top1 >= 0.97 and worst >= 0.90:
        return "semantic top-1 cleared 97 percent on held-out composition"
    return "held-out semantic top-1 is still short of 97 percent"


def train_selector(root: Path, seed: int = 0) -> dict:
    bank = d1_bank()
    out = root / "results" / "select" / "s1" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    scope = _load_scope(root)
    witness = WitnessNet(scope.local)
    total = total_parameter_count(scope) + sum(param.numel() for param in witness.head.parameters())
    if not PARAM_LO <= total <= PARAM_HI:
        raise RuntimeError(f"selector parameter count {total} left the 1M–2M band")
    prior = _fit_prior(rng, bank)
    print("s1 collecting witness strings", flush=True)
    data_tasks = _sample_tasks(rng, bank, 6, 16)
    rows = _collect(scope, data_tasks)
    if len(rows) < BATCH:
        raise RuntimeError("witness set is too small to train")
    positives = sum(label for _supports, _text, label in rows)
    print(f"s1 witnesses {len(rows)} positive {positives / len(rows):.3f}", flush=True)
    opt = torch.optim.AdamW(witness.head.parameters(), lr=LR, weight_decay=0.01)
    started = time.time()
    curve = []
    select_tasks = [
        sample_compose_task(family, random.Random(1200 + index * 17 + draw), 2)
        for index, family in enumerate(ATOMIC)
        for draw in range(2)
    ]
    base = _score_split(scope, witness, prior, select_tasks, with_stability=False)
    best_score = base["top1"]
    best_step = 0
    best_head = {key: value.detach().cpu().clone() for key, value in witness.head.state_dict().items()}
    curve.append({"step": 0, "loss": None, "select_top1": best_score, "seconds": time.time() - started})
    print(f"s1 seed {seed} step 0: atomic top1 {best_score:.3f} ({curve[-1]['seconds']:.0f}s)", flush=True)
    for step in range(1, STEPS + 1):
        batch_rows = [rows[rng.randrange(len(rows))] for _ in range(BATCH)]
        batch = collate_witnesses([(supports, text) for supports, text, _label in batch_rows])
        target = torch.tensor([label for _supports, _text, label in batch_rows])
        witness.encoder.eval()
        loss = witness_loss(witness(batch), target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(witness.head.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == STEPS:
            witness.encoder.eval()
            report = _score_split(scope, witness, prior, select_tasks, with_stability=False)
            curve.append({"step": step, "loss": float(loss.detach()), "select_top1": report["top1"], "seconds": time.time() - started})
            print(
                f"s1 seed {seed} step {step}: loss {float(loss.detach()):.3f} atomic top1 {report['top1']:.3f} ({curve[-1]['seconds']:.0f}s)",
                flush=True,
            )
            if report["top1"] >= best_score:
                best_score = report["top1"]
                best_step = step
                best_head = {key: value.detach().cpu().clone() for key, value in witness.head.state_dict().items()}
    witness.head.load_state_dict(best_head)
    witness.encoder.eval()
    final = {}
    for index, family in enumerate(ALL):
        tasks = [sample_compose_task(family, random.Random(5000 + index * 50 + draw), 3) for draw in range(16)]
        final[family] = _score_split(scope, witness, prior, tasks, with_stability=True)
        print(
            f"s1 seed {seed} {family}: top1 {final[family]['top1']:.3f} semantic {final[family]['semantic']:.3f} "
            f"oracle {final[family]['oracle']:.3f} families {final[family]['families']:.1f}",
            flush=True,
        )
    inv_tasks = [sample_compose_task("repeat", random.Random(90 + index), 1) for index in range(4)]
    invariance = _invariance(scope, witness, prior, inv_tasks)
    ood_top1 = sum(final[family]["top1"] for family in OOD) / len(OOD)
    ood_semantic = sum(final[family]["semantic"] for family in OOD) / len(OOD)
    ood_oracle = sum(final[family]["oracle"] for family in OOD) / len(OOD)
    worst = min(final[family]["top1"] for family in OOD)
    worst_family = min(OOD, key=lambda family: final[family]["top1"])
    atomic_top1 = sum(final[family]["top1"] for family in ATOMIC) / len(ATOMIC)
    records = [{"id": item.ident, "schema": item.schema} for item in bank]
    metrics = {
        "arch": "s1",
        "seed": seed,
        "parameters": total,
        "witness_parameters": sum(param.numel() for param in witness.head.parameters()),
        "steps": STEPS,
        "batch": BATCH,
        "selected_step": best_step,
        "select_top1": best_score,
        "generator_version": GENERATOR_VERSION,
        "template_bank_hash": bank_hash(records),
        "frozen_checkpoint": SCOPE_CHECKPOINT,
        "atomic_top1": atomic_top1,
        "ood_top1": ood_top1,
        "ood_semantic": ood_semantic,
        "ood_oracle": ood_oracle,
        "worst_family": worst_family,
        "worst_top1": worst,
        "min_invariance": invariance,
        "interpretation": _interpretation(ood_oracle, ood_top1, worst, invariance),
        "families": final,
        "curve": curve,
        "prior": prior.state(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save({"head": best_head, "prior": prior.state(), "selected_step": best_step}, out / "checkpoint.pt")
    print(
        f"s1 seed {seed}: top1 {ood_top1:.3f} semantic {ood_semantic:.3f} oracle {ood_oracle:.3f} "
        f"worst {worst_family} {worst:.3f} invariance {invariance:.3f} [{metrics['interpretation']}]",
        flush=True,
    )
    return metrics
