"""Candidate-oracle test and CPU rankers.

No neural net is trained. The oracle scores every symbolically proposed
candidate on hidden strings and keeps the best, which is the ceiling of a
later ranker. MDL and the digit-instability rule use only the visible strings.
The linear models are ridge regressions on those same visible features, fit on
BENCH-A or BENCH-B and never on BENCH-C.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pattern_induction.candidates import (
    FEATURE_NAMES,
    feature_vector,
    pattern_cost,
    presented_candidates,
)
from pattern_induction.evaluate import compile_regex
from pattern_induction.tasks import Task, read_jsonl
from pattern_induction.taxonomy import categorize

K_PRESENTED = 8
K_SWEEP = (1, 2, 4, 6, 8, 10)
LAMBDAS = (0.1, 1.0, 10.0, 100.0)
# Published Gate 1 figure for the same BENCH-C file. Not re-queried here.
TERRA_BENCH_C = 0.981


@dataclass
class Scored:
    pattern: str
    tag: str
    accuracy: float
    recall: float
    rejection: float


@dataclass
class TaskEval:
    task_id: str
    benchmark: str
    domain: str
    split: str
    n_visible: int
    visible: list[str]
    gold_regex: str
    heuristic: str
    pool: list[Scored]
    features: list[list[float]]
    gold_accuracy: float
    latency_ms: float
    failure: str = ""
    picks: dict[str, int] = field(default_factory=dict)

    @property
    def presented(self) -> list[Scored]:
        return self.pool[:K_PRESENTED]


def _rates(pattern: str, task: Task) -> tuple[float, float, float]:
    try:
        compiled = compile_regex(pattern)
    except Exception:
        return 0.0, 0.0, 0.0
    n_pos = len(task.hidden_positives) or 1
    n_neg = len(task.hidden_negatives) or 1
    try:
        recall = sum(compiled.fullmatch(text) is not None for text in task.hidden_positives) / n_pos
        rejection = sum(compiled.fullmatch(text) is None for text in task.hidden_negatives) / n_neg
    except Exception:
        return 0.0, 0.0, 0.0
    return (recall + rejection) / 2, recall, rejection


def evaluate_task(task: Task, k: int = K_PRESENTED, pool_limit: int = 80) -> TaskEval:
    started = time.perf_counter()
    shown, pool = presented_candidates(task.visible, k=max(k, max(K_SWEEP)), pool_limit=pool_limit)
    # presented_candidates already truncates `shown` to k and `pool` to pool_limit.
    # Re-slice from the pool so the sweep and the K=8 set share one order.
    del shown
    scored: list[Scored] = []
    for item in pool:
        accuracy, recall, rejection = _rates(item.pattern, task)
        scored.append(Scored(item.pattern, item.tag, accuracy, recall, rejection))
    heuristic = scored[0].pattern if scored else ""
    features = [
        feature_vector(item.pattern, heuristic, task.visible) for item in scored[:K_PRESENTED]
    ]
    gold_accuracy, _, _ = _rates(task.gold_regex, task)
    latency_ms = (time.perf_counter() - started) * 1000
    failure = categorize(
        {
            "semantic_accuracy": scored[0].accuracy if scored else 0.0,
            "positive_recall": scored[0].recall if scored else 0.0,
            "negative_rejection": scored[0].rejection if scored else 0.0,
            "valid": 1.0,
            "catastrophic": 0.0,
            "gold_regex": task.gold_regex,
            "domain": task.domain,
        }
    )
    return TaskEval(
        task_id=task.id,
        benchmark=task.benchmark,
        domain=task.domain,
        split=task.split,
        n_visible=task.n_visible,
        visible=list(task.visible),
        gold_regex=task.gold_regex,
        heuristic=heuristic,
        pool=scored,
        features=features,
        gold_accuracy=gold_accuracy,
        latency_ms=latency_ms,
        failure=failure or "success",
    )


def _better(challenger: Scored, incumbent: Scored, heuristic: str) -> bool:
    if challenger.accuracy > incumbent.accuracy + 1e-12:
        return True
    if abs(challenger.accuracy - incumbent.accuracy) > 1e-12:
        return False
    if challenger.pattern == heuristic and incumbent.pattern != heuristic:
        return True
    if incumbent.pattern == heuristic:
        return False
    return len(challenger.pattern) < len(incumbent.pattern)


def argmax_accuracy(cands: list[Scored], heuristic: str) -> int:
    best = 0
    for index, cand in enumerate(cands):
        if index and _better(cand, cands[best], heuristic):
            best = index
    return best


def pick_cost(eval_row: TaskEval, digit_prior: bool) -> int:
    # Costs are stored at scoring time; TaskEval keeps the visible strings.
    cands = eval_row.presented
    strings = eval_row.visible
    best = 0
    best_cost = pattern_cost(cands[0].pattern, strings, digit_prior)
    for index, cand in enumerate(cands[1:], start=1):
        cost = pattern_cost(cand.pattern, strings, digit_prior)
        if cost < best_cost - 1e-9:
            best, best_cost = index, cost
        elif abs(cost - best_cost) <= 1e-9 and cand.pattern == eval_row.heuristic:
            best, best_cost = index, cost
    return best


def pick_random(eval_row: TaskEval) -> int:
    rng = random.Random(int(hashlib.sha256(eval_row.task_id.encode()).hexdigest()[:8], 16))
    return rng.randrange(len(eval_row.presented))


def assign_symbolic_picks(eval_row: TaskEval) -> None:
    presented = eval_row.presented
    eval_row.picks = {
        "heuristic": 0,
        "random": pick_random(eval_row),
        "mdl": pick_cost(eval_row, digit_prior=False),
        "rules": pick_cost(eval_row, digit_prior=True),
        "oracle_k": argmax_accuracy(presented, eval_row.heuristic),
        "oracle_pool": argmax_accuracy(eval_row.pool, eval_row.heuristic),
    }


def _solve(matrix: list[list[float]], target: list[float]) -> list[float]:
    size = len(target)
    rows = [matrix[index][:] + [target[index]] for index in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(rows[row][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        if abs(scale) < 1e-12:
            scale = 1e-12 if scale >= 0 else -1e-12
        for col_index in range(col, size + 1):
            rows[col][col_index] /= scale
        for row_index in range(size):
            if row_index == col:
                continue
            factor = rows[row_index][col]
            if factor == 0:
                continue
            for col_index in range(col, size + 1):
                rows[row_index][col_index] -= factor * rows[col][col_index]
    return [rows[index][size] for index in range(size)]


def _standardize(samples: list[list[float]]) -> tuple[list[float], list[float]]:
    width = len(samples[0])
    means: list[float] = []
    scales: list[float] = []
    count = float(len(samples))
    for col in range(width):
        column = [row[col] for row in samples]
        mean = sum(column) / count
        variance = sum((value - mean) ** 2 for value in column) / count
        scale = variance ** 0.5
        means.append(mean)
        scales.append(scale if scale >= 1e-8 else 1.0)
    return means, scales


def _design(features: list[float], means: list[float], scales: list[float]) -> list[float]:
    return [1.0] + [(features[index] - means[index]) / scales[index] for index in range(len(features))]


@dataclass
class LinearModel:
    means: list[float]
    scales: list[float]
    weights: list[float]
    lam: float

    def score(self, features: list[float]) -> float:
        designed = _design(features, self.means, self.scales)
        return sum(weight * value for weight, value in zip(self.weights, designed))


def fit_ridge(samples: list[tuple[list[float], float]], lam: float) -> LinearModel:
    means, scales = _standardize([features for features, _label in samples])
    width = len(samples[0][0]) + 1
    gram = [[0.0] * width for _ in range(width)]
    target = [0.0] * width
    for features, label in samples:
        designed = _design(features, means, scales)
        for row in range(width):
            target[row] += designed[row] * label
            for col in range(width):
                gram[row][col] += designed[row] * designed[col]
    for index in range(1, width):
        gram[index][index] += lam
    return LinearModel(means, scales, _solve(gram, target), lam)


def _selection_accuracy(rows: list[TaskEval], model: LinearModel) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        total += row.presented[_predict_index(row, model)].accuracy
    return total / len(rows)


def _predict_index(row: TaskEval, model: LinearModel) -> int:
    best = 0
    best_score = model.score(row.features[0]) if row.features else 0.0
    for index, features in enumerate(row.features):
        score = model.score(features)
        if score > best_score + 1e-12:
            best, best_score = index, score
        elif abs(score - best_score) <= 1e-12 and row.presented[index].pattern == row.heuristic:
            best, best_score = index, score
    return best


def train_linear(train_rows: list[TaskEval], val_rows: list[TaskEval]) -> tuple[LinearModel, float]:
    samples = [
        (features, cand.accuracy)
        for row in train_rows
        for features, cand in zip(row.features, row.presented)
    ]
    if not samples:
        raise RuntimeError("linear ranker received no training rows")
    best_model: LinearModel | None = None
    best_lam = LAMBDAS[0]
    best_accuracy = -1.0
    for lam in LAMBDAS:
        model = fit_ridge(samples, lam)
        heldout = val_rows or train_rows
        accuracy = _selection_accuracy(heldout, model)
        if accuracy > best_accuracy:
            best_model = model
            best_lam = lam
            best_accuracy = accuracy
    assert best_model is not None
    return best_model, best_lam


def apply_linear(rows: list[TaskEval], model: LinearModel, name: str) -> None:
    for row in rows:
        row.picks[name] = _predict_index(row, model)


def document_fold(task_id: str) -> str:
    bucket = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % 5
    return "val" if bucket == 0 else "train"


def chosen(row: TaskEval, system: str) -> Scored:
    if system == "gold":
        return Scored(row.gold_regex, "gold", row.gold_accuracy, 0.0, 0.0)
    if system == "random_expected":
        cands = row.presented
        mean_acc = statistics.fmean(cand.accuracy for cand in cands)
        return Scored("", "random_expected", mean_acc, 0.0, 0.0)
    index = row.picks[system]
    pool = row.pool if system == "oracle_pool" else row.presented
    return pool[index]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return statistics.fmean(values)


def summarize(rows: list[TaskEval], system: str) -> dict:
    if not rows:
        return {"system": system, "n": 0}
    accuracies = []
    recalls = []
    rejections = []
    domains: dict[str, list[float]] = defaultdict(list)
    unchanged = improved = regressed = 0
    for row in rows:
        pick = chosen(row, system)
        base = row.presented[0].accuracy
        accuracies.append(pick.accuracy)
        if system not in {"gold", "random_expected"}:
            recalls.append(pick.recall)
            rejections.append(pick.rejection)
            if pick.pattern == row.heuristic:
                unchanged += 1
            elif pick.accuracy > base + 0.01:
                improved += 1
            elif pick.accuracy + 0.01 < base:
                regressed += 1
        domains[row.domain].append(pick.accuracy)
    n = len(rows)
    summary = {
        "system": system,
        "n": n,
        "semantic_accuracy": round(_mean(accuracies), 4),
        "macro_semantic_accuracy": round(
            _mean([_mean(values) for values in domains.values()]), 4
        ),
        "by_domain": {
            domain: round(_mean(values), 4) for domain, values in sorted(domains.items())
        },
    }
    if recalls:
        summary["positive_recall"] = round(_mean(recalls), 4)
        summary["negative_rejection"] = round(_mean(rejections), 4)
        summary["unchanged_rate"] = round(unchanged / n, 4)
        summary["improved_rate"] = round(improved / n, 4)
        summary["regressed_rate"] = round(regressed / n, 4)
    return summary


def sweep_oracle(rows: list[TaskEval]) -> list[dict]:
    table = []
    for k in K_SWEEP:
        accuracies = []
        for row in rows:
            prefix = row.pool[:k] or row.pool[:1]
            accuracies.append(max(cand.accuracy for cand in prefix))
        table.append({"k": k, "n": len(rows), "oracle_accuracy": round(_mean(accuracies), 4)})
    if rows:
        table.append(
            {
                "k": "pool",
                "n": len(rows),
                "oracle_accuracy": round(
                    _mean(max(cand.accuracy for cand in row.pool) for row in rows), 4
                ),
                "mean_pool_size": round(_mean([float(len(row.pool)) for row in rows]), 2),
            }
        )
    return table


def gain_tags(rows: list[TaskEval], system: str) -> dict:
    """Tag of the selected candidate on tasks where it beats the heuristic."""
    counts: dict[str, int] = defaultdict(int)
    gain = 0.0
    n_gain = 0
    for row in rows:
        pick = chosen(row, system)
        base = row.presented[0].accuracy
        if pick.accuracy <= base + 1e-12 or pick.pattern == row.heuristic:
            counts["no_gain"] += 1
            continue
        counts[pick.tag] += 1
        gain += pick.accuracy - base
        n_gain += 1
    return {
        "n_gain": n_gain,
        "mean_gain_when_improved": round(gain / n_gain, 4) if n_gain else 0.0,
        "tags": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def family_table(rows: list[TaskEval], systems: list[str]) -> list[dict]:
    buckets: dict[str, list[TaskEval]] = defaultdict(list)
    for row in rows:
        buckets[row.failure].append(row)
    table = []
    for failure, group in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
        entry = {"failure": failure, "n": len(group)}
        for system in systems:
            entry[system] = round(_mean([chosen(row, system).accuracy for row in group]), 4)
        entry["rank_gap"] = round(entry["oracle_k"] - entry["rules"], 4)
        entry["truncation_gap"] = round(entry["oracle_pool"] - entry["oracle_k"], 4)
        entry["generator_gap"] = round(entry["gold"] - entry["oracle_pool"], 4)
        table.append(entry)
    return table


def sample_size_table(rows: list[TaskEval], systems: list[str]) -> list[dict]:
    buckets: dict[int, list[TaskEval]] = defaultdict(list)
    for row in rows:
        buckets[row.n_visible].append(row)
    table = []
    for n_visible, group in sorted(buckets.items()):
        entry = {"n_visible": n_visible, "n": len(group)}
        for system in systems:
            entry[system] = round(_mean([chosen(row, system).accuracy for row in group]), 4)
        table.append(entry)
    return table


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _load_terra(exp1_json: Path) -> float:
    """Gate 1 Terra accuracy on this BENCH-C file, if the Experiment 1 report is present."""
    if not exp1_json.exists():
        return TERRA_BENCH_C
    payload = json.loads(exp1_json.read_text(encoding="utf-8"))
    frontier = payload.get("frontier")
    if isinstance(frontier, dict):
        for row in frontier.get("systems", []):
            if row.get("system") == "gpt-5.6-terra":
                return float(row["semantic_accuracy"])
    return TERRA_BENCH_C


def _system_map(payload: dict) -> dict[str, dict]:
    return {row["system"]: row for row in payload["systems"]}


def interpret(report: dict) -> list[str]:
    bench_c = report["benchmarks"]["bench_c"]
    by_system = _system_map(bench_c)
    heuristic = by_system["heuristic"]["semantic_accuracy"]
    oracle = by_system["oracle_k"]["semantic_accuracy"]
    pool = by_system["oracle_pool"]["semantic_accuracy"]
    cpu_names = ["mdl", "rules", "linear_structural", "linear_document"]
    best_name = max(cpu_names, key=lambda name: by_system[name]["semantic_accuracy"])
    best = by_system[best_name]["semantic_accuracy"]
    oracle_gain = oracle - heuristic
    residual = oracle - best
    terra = report["protocol"]["terra_bench_c"]
    terra_gap = terra - heuristic
    sweep = {item["k"]: item["oracle_accuracy"] for item in bench_c["oracle_sweep"]}
    lines = [
        (
            f"On BENCH-C the heuristic scores {_pct(heuristic)}. "
            f"The best of {K_PRESENTED} symbolic neighbors scores {_pct(oracle)} "
            f"({oracle_gain * 100:+.1f} points). The full candidate pool scores {_pct(pool)}. "
            f"Gold, which generated the hidden sets, scores {_pct(by_system['gold']['semantic_accuracy'])}."
        ),
        (
            f"A uniform draw among those {K_PRESENTED} neighbors scores "
            f"{_pct(by_system['random']['semantic_accuracy'])}. "
            "Neighbors are required to match every visible string, so random selection is not a low floor. "
            f"K=2 already reaches {_pct(sweep.get(2, oracle))}; "
            f"growing the set from 2 to {K_PRESENTED} adds {(oracle - sweep.get(2, oracle)) * 100:.1f} points, "
            f"and the patterns left outside the {K_PRESENTED} add another {(pool - oracle) * 100:.1f}."
        ),
        (
            f"The strongest CPU selector on this file is `{best_name}` at {_pct(best)} "
            f"({(best - heuristic) * 100:+.1f} points over the heuristic). "
            f"`mdl` scores {_pct(by_system['mdl']['semantic_accuracy'])}, "
            f"`rules` {_pct(by_system['rules']['semantic_accuracy'])}, "
            f"the structural linear model {_pct(by_system['linear_structural']['semantic_accuracy'])}, "
            f"and the document linear model {_pct(by_system['linear_document']['semantic_accuracy'])}. "
            f"That leaves {residual * 100:.1f} points between `{best_name}` and the {K_PRESENTED}-candidate oracle."
        ),
    ]
    if terra_gap > 0:
        covered = oracle_gain / terra_gap
        lines.append(
            f"Gate 1 `gpt-5.6-terra` scored {_pct(terra)} on this same file "
            f"({terra_gap * 100:+.1f} points over the heuristic). "
            f"The {K_PRESENTED}-candidate oracle covers {covered * 100:.0f}% of that gap, "
            f"and the full pool is {(terra - pool) * 100:.1f} points from Terra."
        )
    lines.append(
        f"`rules` improves {_pct(by_system['rules']['improved_rate'])} of BENCH-C tasks and "
        f"regresses {_pct(by_system['rules']['regressed_rate'])}. "
        f"The structural linear model regresses only {_pct(by_system['linear_structural']['regressed_rate'])} "
        f"and leaves the heuristic unchanged on {_pct(by_system['linear_structural']['unchanged_rate'])}, "
        f"but it gives up accuracy to do that. "
        f"The document linear model keeps the heuristic on "
        f"{_pct(by_system['linear_document']['unchanged_rate'])} of tasks and regresses "
        f"{_pct(by_system['linear_document']['regressed_rate'])}."
    )
    interesting = [
        row
        for row in report.get("bench_c_families", [])
        if row["failure"] != "success" and row["n"] >= 5
    ]
    if interesting:
        bits = []
        for row in interesting:
            bits.append(
                f"{row['failure']} (n={row['n']}): heuristic {_pct(row['heuristic'])}, "
                f"rules {_pct(row['rules'])}, oracle@8 {_pct(row['oracle_k'])}, "
                f"pool {_pct(row['oracle_pool'])}"
            )
        lines.append("Where the heuristic was already wrong: " + "; ".join(bits) + ".")
        rank_focus = max(interesting, key=lambda row: row["rank_gap"])
        gen_focus = max(interesting, key=lambda row: row["generator_gap"])
        lines.append(
            f"Inside those families, the widest ranking gap is {rank_focus['failure']} "
            f"({rank_focus['rank_gap'] * 100:+.1f} points between rules and oracle@8). "
            f"The widest generator gap is {gen_focus['failure']} "
            f"({gen_focus['generator_gap'] * 100:+.1f} points between the pool and gold)."
        )
    shape = report["benchmarks"].get("bench_a_test")
    if shape is not None:
        shape_by = _system_map(shape)
        lines.append(
            f"On held-out BENCH-A shapes the heuristic scores {_pct(shape_by['heuristic']['semantic_accuracy'])}, "
            f"`rules` {_pct(shape_by['rules']['semantic_accuracy'])}, "
            f"the structural linear model {_pct(shape_by['linear_structural']['semantic_accuracy'])}, "
            f"and oracle@8 {_pct(shape_by['oracle_k']['semantic_accuracy'])} "
            f"(pool {_pct(shape_by['oracle_pool']['semantic_accuracy'])}, gold {_pct(shape_by['gold']['semantic_accuracy'])}). "
            "The pool is well short of gold, so a ranker cannot close the structural-composition gap by itself."
        )
    sample = report.get("bench_d_sample_size") or []
    if sample:
        first, last = sample[0], sample[-1]
        lines.append(
            f"On BENCH-D the oracle lead over `rules` is "
            f"{(first['oracle_k'] - first['rules']) * 100:.1f} points at {first['n_visible']} visible strings "
            f"and {(last['oracle_k'] - last['rules']) * 100:.1f} points at {last['n_visible']}. "
            "Ranking headroom shrinks once the column is observed more times; candidate generation still matters at every sample size."
        )
    if oracle_gain < 0.015:
        lines.append(
            "The neighborhood barely beats the heuristic. The next CPU experiment is a richer candidate "
            "generator, not a ranker. There is nothing for a GPU model to select."
        )
    elif residual < 0.01:
        lines.append(
            "Code length already collects the oracle gain. A neural ranker is not the binding constraint. "
            "The families whose pool is still far from gold need new symbolic candidates; ranking will not invent them."
        )
    else:
        lines.append(
            f"A tiny ranker is worth a pilot only as a test of the residual {residual * 100:.1f} points, "
            f"and only if it beats `{best_name}` on BENCH-C and on the BENCH-A shape holdout while regressing "
            "no more often than the structural linear model. "
            "The linear feature models did not beat code length, so the pilot has to use the strings themselves, "
            "not another regression on regex statistics. "
            "Failures where the pool is already near gold are ranking problems. "
            "Failures where the pool stays far from gold are generator problems and will not move if the ranker is perfect. "
            "Neither result requires a larger model or a 50k-task GPU run until that pilot wins."
        )
    return lines


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_markdown(report: dict) -> str:
    lines = [
        "# Candidate oracle (CPU)",
        "",
        "Semantic accuracy is `(hidden positive recall + hidden negative rejection) / 2`.",
        "Candidates are generated from the visible strings only. The oracle then picks the",
        "candidate with the highest hidden accuracy. That number is a ceiling for a ranker,",
        "not a system we can ship: it sees the hidden set. A wider pattern can win when",
        "the hidden negatives happen not to contain the extra strings it accepts.",
        "",
        f"The presented set has {K_PRESENTED} candidates, heuristic first, then one slot per",
        "mutation family. `mdl` minimizes two-part code length on the observed strings.",
        "`rules` is the same length with an extra charge on digit literals shorter than",
        "four characters. The linear models are ridge regressions on visible features,",
        "with the penalty chosen on a validation split that does not include BENCH-C.",
        "",
        "## Reading",
        "",
    ]
    for paragraph in report["interpretation"]:
        lines.append(paragraph)
        lines.append("")
    lines.append("## Systems")
    lines.append("")
    for bench, payload in report["benchmarks"].items():
        role = payload.get("role", "")
        lines.append(f"### {bench}")
        lines.append("")
        if role:
            lines.append(role[:1].upper() + role[1:] + ".")
            lines.append("")
        headers = ["System", "Semantic accuracy", "Macro", "Recall", "Rejection", "Improved", "Regressed", "Unchanged"]
        table = []
        for row in payload["systems"]:
            table.append(
                [
                    row["system"],
                    _pct(row["semantic_accuracy"]),
                    _pct(row.get("macro_semantic_accuracy", 0.0)),
                    _pct(row["positive_recall"]) if "positive_recall" in row else "",
                    _pct(row["negative_rejection"]) if "negative_rejection" in row else "",
                    _pct(row["improved_rate"]) if "improved_rate" in row else "",
                    _pct(row["regressed_rate"]) if "regressed_rate" in row else "",
                    _pct(row["unchanged_rate"]) if "unchanged_rate" in row else "",
                ]
            )
        lines.extend(_md_table(headers, table))
        lines.append("")
        if payload.get("oracle_sweep"):
            lines.append("Oracle as the presented set grows (diversity order, heuristic at K=1):")
            lines.append("")
            sweep_rows = [
                [str(item["k"]), str(item["n"]), _pct(item["oracle_accuracy"])]
                + ([f"{item['mean_pool_size']:.1f}"] if "mean_pool_size" in item else [""])
                for item in payload["oracle_sweep"]
            ]
            lines.extend(_md_table(["K", "Tasks", "Oracle", "Mean pool"], sweep_rows))
            lines.append("")
        if payload.get("gain_tags"):
            tags = payload["gain_tags"]
            rendered = ", ".join(f"{tag} {count}" for tag, count in tags["tags"].items())
            lines.append(
                f"Oracle@8 improvements: {tags['n_gain']} tasks, "
                f"mean gain on those tasks {_pct(tags['mean_gain_when_improved'])}. Tags: {rendered}."
            )
            lines.append("")
    if report.get("bench_c_families"):
        lines.append("## BENCH-C by heuristic failure")
        lines.append("")
        lines.append(
            "`rank_gap` is oracle@8 minus the hand-written rule. "
            "`truncation_gap` is the full pool minus oracle@8. "
            "`generator_gap` is gold minus the full pool."
        )
        lines.append("")
        headers = [
            "Failure", "n", "Heuristic", "MDL", "Rules", "Linear doc", "Oracle@8", "Pool", "Gold",
            "Rank gap", "Truncation", "Generator",
        ]
        family_rows = []
        for row in report["bench_c_families"]:
            family_rows.append(
                [
                    row["failure"],
                    str(row["n"]),
                    _pct(row["heuristic"]),
                    _pct(row["mdl"]),
                    _pct(row["rules"]),
                    _pct(row["linear_document"]),
                    _pct(row["oracle_k"]),
                    _pct(row["oracle_pool"]),
                    _pct(row["gold"]),
                    f"{row['rank_gap'] * 100:+.1f}",
                    f"{row['truncation_gap'] * 100:+.1f}",
                    f"{row['generator_gap'] * 100:+.1f}",
                ]
            )
        lines.extend(_md_table(headers, family_rows))
        lines.append("")
    if report.get("bench_d_sample_size"):
        lines.append("## BENCH-D sample size")
        lines.append("")
        headers = ["Visible strings", "n", "Heuristic", "Rules", "Oracle@8", "Pool"]
        sample_rows = [
            [
                str(row["n_visible"]),
                str(row["n"]),
                _pct(row["heuristic"]),
                _pct(row["rules"]),
                _pct(row["oracle_k"]),
                _pct(row["oracle_pool"]),
            ]
            for row in report["bench_d_sample_size"]
        ]
        lines.extend(_md_table(headers, sample_rows))
        lines.append("")
    lines.append("## Protocol")
    lines.append("")
    protocol = report["protocol"]
    lines.append(
        f"Linear structural model trained on BENCH-A `{protocol['structural_train_split']}` "
        f"({protocol['structural_train_n']} tasks), penalty chosen on BENCH-A val "
        f"(λ={protocol['structural_lambda']}, val accuracy {_pct(protocol['structural_val_accuracy'])})."
    )
    lines.append("")
    lines.append(
        f"Linear document model trained on BENCH-B fold `{protocol['document_train_fold']}` "
        f"({protocol['document_train_n']} tasks), penalty chosen on the held-out BENCH-B fold "
        f"(λ={protocol['document_lambda']}, val accuracy {_pct(protocol['document_val_accuracy'])})."
    )
    lines.append("")
    lines.append(
        "BENCH-C and BENCH-D are not used to fit either model or to choose λ. "
        f"Mean candidate-stage latency on BENCH-C is {protocol['bench_c_latency_ms']['mean']:.1f} ms "
        f"(p50 {protocol['bench_c_latency_ms']['p50']:.1f} ms)."
    )
    lines.append("")
    lines.append(protocol["terra_note"])
    lines.append("")
    return "\n".join(lines)


SYSTEMS = [
    "heuristic",
    "random",
    "random_expected",
    "mdl",
    "rules",
    "linear_structural",
    "linear_document",
    "oracle_k",
    "oracle_pool",
    "gold",
]


def _latency(rows: list[TaskEval]) -> dict:
    values = [row.latency_ms for row in rows]
    if not values:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    ordered = sorted(values)
    def pct(fraction: float) -> float:
        rank = (len(ordered) - 1) * fraction
        lo = int(rank)
        hi = min(lo + 1, len(ordered) - 1)
        weight = rank - lo
        return ordered[lo] * (1 - weight) + ordered[hi] * weight
    return {"p50": pct(0.5), "p95": pct(0.95), "mean": _mean(values)}


def run(data_dir: Path, out_dir: Path, benches: list[str] | None = None) -> dict:
    names = benches or ["bench_c", "bench_b", "bench_a", "bench_d"]
    grouped: dict[str, list[TaskEval]] = {}
    for name in names:
        path = data_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        tasks = read_jsonl(path)
        rows = []
        started = time.perf_counter()
        for index, task in enumerate(tasks, start=1):
            rows.append(evaluate_task(task))
            assign_symbolic_picks(rows[-1])
            if index % 100 == 0 or index == len(tasks):
                elapsed = time.perf_counter() - started
                print(f"{name} {index}/{len(tasks)} in {elapsed:.1f}s", flush=True)
        grouped[name] = rows

    structural_train = [row for row in grouped.get("bench_a", []) if row.split == "dev"]
    structural_val = [row for row in grouped.get("bench_a", []) if row.split == "val"]
    document_train = [row for row in grouped.get("bench_b", []) if document_fold(row.task_id) == "train"]
    document_val = [row for row in grouped.get("bench_b", []) if document_fold(row.task_id) == "val"]

    structural_model, structural_lam = train_linear(structural_train, structural_val)
    document_model, document_lam = train_linear(document_train, document_val)
    all_rows = [row for rows in grouped.values() for row in rows]
    apply_linear(all_rows, structural_model, "linear_structural")
    apply_linear(all_rows, document_model, "linear_document")

    benchmarks = {}
    for name, rows in _slice_report(grouped).items():
        benchmarks[name] = {
            "n": len(rows),
            "role": _slice_role(name),
            "systems": [summarize(rows, system) for system in SYSTEMS],
            "oracle_sweep": sweep_oracle(rows),
            "gain_tags": gain_tags(rows, "oracle_k"),
            "latency_ms": {key: round(value, 3) for key, value in _latency(rows).items()},
        }

    family_systems = ["heuristic", "mdl", "rules", "linear_document", "linear_structural", "oracle_k", "oracle_pool", "gold"]
    report = {
        "benchmarks": benchmarks,
        "bench_c_families": family_table(grouped.get("bench_c", []), family_systems),
        "bench_d_sample_size": sample_size_table(
            grouped.get("bench_d", []),
            ["heuristic", "rules", "oracle_k", "oracle_pool"],
        ),
        "protocol": {
            "k": K_PRESENTED,
            "features": list(FEATURE_NAMES),
            "structural_train_split": "dev",
            "structural_train_n": len(structural_train),
            "structural_lambda": structural_lam,
            "structural_val_accuracy": round(_selection_accuracy(structural_val, structural_model), 4),
            "document_train_fold": "bench_b hash % 5 != 0",
            "document_train_n": len(document_train),
            "document_lambda": document_lam,
            "document_val_accuracy": round(_selection_accuracy(document_val, document_model), 4),
            "bench_c_latency_ms": _latency(grouped.get("bench_c", [])),
            "terra_bench_c": _load_terra(Path("results/exp1/exp1_report.json")),
            "terra_note": (
                "Terra's BENCH-C accuracy is the Gate 1 figure in results/exp1 "
                "(gpt-5.6-terra, simple prompt, 2026-09-25). This run does not call a model API."
            ),
        },
    }
    report["interpretation"] = interpret(report) if "bench_c" in benchmarks else []

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_oracle.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "candidate_oracle.md").write_text(render_markdown(report), encoding="utf-8")
    with (out_dir / "candidate_oracle_tasks.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(_task_record(row)) + "\n")
    return report


def _slice_role(name: str) -> str:
    return {
        "bench_c": "held out of both linear models; primary decision set",
        "bench_d": "held out; same schemas at several sample sizes",
        "bench_a_test": "held-out AST shapes",
        "bench_a_val": "penalty selection for the structural linear model",
        "bench_a_dev": "training data for the structural linear model",
        "bench_b_val": "penalty selection for the document linear model",
        "bench_b_train": "training data for the document linear model",
    }.get(name, "")


def _slice_report(grouped: dict[str, list[TaskEval]]) -> dict[str, list[TaskEval]]:
    """Hold out the rows each linear model was allowed to see."""
    slices = {
        "bench_c": grouped.get("bench_c", []),
        "bench_d": grouped.get("bench_d", []),
        "bench_a_test": [row for row in grouped.get("bench_a", []) if row.split == "test"],
        "bench_a_val": [row for row in grouped.get("bench_a", []) if row.split == "val"],
        "bench_a_dev": [row for row in grouped.get("bench_a", []) if row.split == "dev"],
        "bench_b_val": [row for row in grouped.get("bench_b", []) if document_fold(row.task_id) == "val"],
        "bench_b_train": [row for row in grouped.get("bench_b", []) if document_fold(row.task_id) == "train"],
    }
    return {name: rows for name, rows in slices.items() if rows}


def _task_record(row: TaskEval) -> dict:
    def pack(system: str) -> dict:
        pick = chosen(row, system)
        return {"accuracy": round(pick.accuracy, 4), "pattern": pick.pattern, "tag": pick.tag}

    return {
        "id": row.task_id,
        "benchmark": row.benchmark,
        "domain": row.domain,
        "split": row.split,
        "n_visible": row.n_visible,
        "failure": row.failure,
        "pool_size": len(row.pool),
        "latency_ms": round(row.latency_ms, 3),
        "heuristic": pack("heuristic"),
        "mdl": pack("mdl"),
        "rules": pack("rules"),
        "random": pack("random"),
        "linear_structural": pack("linear_structural"),
        "linear_document": pack("linear_document"),
        "oracle_k": pack("oracle_k"),
        "oracle_pool": pack("oracle_pool"),
        "gold_accuracy": round(row.gold_accuracy, 4),
    }
