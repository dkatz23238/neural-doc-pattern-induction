"""Learned selector on a frozen structural pool of 16 candidates.

Train on the BENCH-B document fold. Choose the listwise penalty, the label
temperature, and the confidence margin on the BENCH-B validation fold. BENCH-C
is scored once and is not used to fit or to stop. BENCH-A is left out: its
oracle gap is still a generator problem.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from pattern_induction.candidate_oracle import LAMBDAS, _rates, document_fold, fit_ridge
from pattern_induction.candidates import expanded_candidates, pattern_cost, presented_candidates
from pattern_induction.selector import (
    FEATURE_NAMES,
    K_FROZEN,
    LAMBDAS as LIST_LAMBDAS,
    MARGINS,
    REGRESSION_BUDGET,
    TAUS,
    alt_literal_cost,
    apply_gate,
    fit_listwise,
    pool_matrix,
    probability_gap,
    rank_index,
    score_patterns,
    standardize_apply,
    standardize_fit,
)
from pattern_induction.tasks import read_jsonl
from pattern_induction.taxonomy import categorize


def _pick_cost(patterns: list[str], strings: list[str], cost_of, heuristic: str) -> int:
    best = 0
    best_cost = cost_of(patterns[0])
    for index, pattern in enumerate(patterns[1:], start=1):
        cost = cost_of(pattern)
        if cost < best_cost - 1e-9:
            best, best_cost = index, cost
        elif abs(cost - best_cost) <= 1e-9 and pattern == heuristic and patterns[best] != heuristic:
            best, best_cost = index, cost
    return best


def pack_task(task) -> dict:
    _shown, old_pool = presented_candidates(task.visible, k=8, pool_limit=80)
    structural, _pool = expanded_candidates(task.visible, k=K_FROZEN, pool_limit=160)
    old = [item.pattern for item in old_pool[:8]]
    patterns = [item.pattern for item in structural[:K_FROZEN]]
    if not patterns:
        patterns = old[:1] or ["^$"]
    heuristic = patterns[0]
    scored: dict[str, tuple[float, float, float]] = {}
    for pattern in patterns + old:
        if pattern not in scored:
            scored[pattern] = _rates(pattern, task)
    accuracy = {pattern: rates[0] for pattern, rates in scored.items()}
    costs = {pattern: pattern_cost(pattern, task.visible, True) for pattern in set(patterns + old)}
    alt_costs = {pattern: alt_literal_cost(pattern, task.visible) for pattern in patterns}
    rules8 = old[_pick_cost(old, task.visible, lambda pattern: costs[pattern], old[0])]
    rules16 = patterns[_pick_cost(patterns, task.visible, lambda pattern: costs[pattern], heuristic)]
    alt16 = patterns[_pick_cost(patterns, task.visible, lambda pattern: alt_costs[pattern], heuristic)]
    heur_acc, heur_rec, heur_rej = scored[heuristic]
    failure = categorize(
        {
            "semantic_accuracy": heur_acc,
            "positive_recall": heur_rec,
            "negative_rejection": heur_rej,
            "valid": 1.0,
            "catastrophic": 0.0,
            "gold_regex": task.gold_regex,
            "domain": task.domain,
        }
    )
    return {
        "id": task.id,
        "benchmark": task.benchmark,
        "domain": task.domain,
        "split": task.split,
        "n_visible": task.n_visible,
        "fold": document_fold(task.id) if task.benchmark == "B" else "",
        "failure": failure or "success",
        "heuristic": heuristic,
        "patterns": patterns,
        "accuracy": [accuracy[pattern] for pattern in patterns],
        "matrix": pool_matrix(patterns, heuristic, task.visible),
        "heuristic_accuracy": accuracy[heuristic],
        "rules8_pattern": rules8,
        "rules8_accuracy": accuracy[rules8],
        "rules16_accuracy": accuracy[rules16],
        "alt_accuracy": accuracy[alt16],
        "oracle_accuracy": max(accuracy[pattern] for pattern in patterns),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _movement(accuracy: float, baseline: float) -> str:
    if accuracy > baseline + 0.01:
        return "improved"
    if accuracy < baseline - 0.01:
        return "regressed"
    return "unchanged"


def _pack_system(rows: list[dict], values: list[float]) -> dict:
    movement = {"improved": 0, "regressed": 0, "unchanged": 0}
    by_domain: dict[str, list[float]] = defaultdict(list)
    for row, accuracy in zip(rows, values):
        movement[_movement(accuracy, row["heuristic_accuracy"])] += 1
        by_domain[row["domain"]].append(accuracy)
    total = len(rows) or 1
    return {
        "semantic_accuracy": round(_mean(values), 4),
        "macro": round(_mean([_mean(scores) for scores in by_domain.values()]), 4) if by_domain else 0.0,
        "improved": round(movement["improved"] / total, 4),
        "regressed": round(movement["regressed"] / total, 4),
        "unchanged": round(movement["unchanged"] / total, 4),
        "n": len(rows),
    }


def _normalize(rows: list[dict], means: list[float], scales: list[float]) -> list[list[list[float]]]:
    return [[standardize_apply(vector, means, scales) for vector in row["matrix"]] for row in rows]


def _pointwise_scores(model, matrix: list[list[float]]) -> list[float]:
    return [model.score(vector) for vector in matrix]


def choose_pointwise(train: list[dict], val: list[dict]):
    samples = [
        (vector, accuracy)
        for row in train
        for vector, accuracy in zip(row["matrix"], row["accuracy"])
    ]
    best = None
    best_accuracy = -1.0
    best_lam = LAMBDAS[0]
    for lam in LAMBDAS:
        model = fit_ridge(samples, lam)
        total = 0.0
        for row in val:
            scores = _pointwise_scores(model, row["matrix"])
            total += row["accuracy"][rank_index(scores, row["patterns"], row["heuristic"])]
        accuracy = total / len(val)
        if accuracy > best_accuracy:
            best, best_accuracy, best_lam = model, accuracy, lam
    return best, best_lam, best_accuracy


def choose_listwise(train: list[dict], val: list[dict], means: list[float], scales: list[float]):
    train_x = _normalize(train, means, scales)
    val_x = _normalize(val, means, scales)
    train_y = [row["accuracy"] for row in train]
    val_y = [row["accuracy"] for row in val]
    val_patterns = [row["patterns"] for row in val]
    val_heuristic = [row["heuristic"] for row in val]
    best = None
    best_key = None
    best_accuracy = -1.0
    for tau in TAUS:
        for lam in LIST_LAMBDAS:
            weights, accuracy, step = fit_listwise(
                train_x,
                train_y,
                val_x,
                val_y,
                val_patterns,
                val_heuristic,
                lam=lam,
                tau=tau,
            )
            print(f"  listwise tau={tau} l2={lam} val={accuracy:.4f} step={step}", flush=True)
            if accuracy > best_accuracy:
                best = weights
                best_key = {"tau": tau, "l2": lam, "val_accuracy": accuracy, "step": step}
                best_accuracy = accuracy
    return best, best_key


def _gated_values(rows: list[dict], weights: list[float], means, scales, margin: float) -> list[float]:
    values = []
    matrices = _normalize(rows, means, scales)
    for row, matrix in zip(rows, matrices):
        scores = score_patterns(weights, matrix)
        learned_index = rank_index(scores, row["patterns"], row["heuristic"])
        learned = row["patterns"][learned_index]
        chosen = apply_gate(learned, row["rules8_pattern"], probability_gap(scores), margin)
        if chosen == learned:
            values.append(row["accuracy"][learned_index])
        elif chosen == row["rules8_pattern"]:
            values.append(row["rules8_accuracy"])
        else:
            values.append(row["heuristic_accuracy"])
    return values


def choose_margin(val: list[dict], weights, means, scales) -> tuple[float, dict]:
    """Highest validation accuracy whose regression rate stays within the budget."""
    eligible = []
    recorded = []
    for margin in MARGINS:
        values = _gated_values(val, weights, means, scales, margin)
        packed = _pack_system(val, values)
        packed["margin"] = margin
        recorded.append(packed)
        if packed["regressed"] <= REGRESSION_BUDGET + 1e-12:
            eligible.append(packed)
    pool = eligible or recorded
    # Prefer accuracy, then fewer regressions, then a wider margin.
    chosen = max(pool, key=lambda row: (row["semantic_accuracy"], -row["regressed"], row["margin"]))
    return chosen["margin"], {"grid": recorded, "chosen": chosen, "budget": REGRESSION_BUDGET, "eligible": bool(eligible)}


def _family_table(rows: list[dict], columns: dict[str, list[float]]) -> list[dict]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["failure"]].append(index)
    table = []
    for label, indexes in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        entry = {"failure": label, "n": len(indexes)}
        for name, values in columns.items():
            entry[name] = round(_mean([values[index] for index in indexes]), 4)
        table.append(entry)
    return table


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _family_line(bench_c: dict, label: str) -> dict | None:
    for row in bench_c.get("families", []):
        if row["failure"] == label:
            return row
    return None


def _interpret(bench_c: dict, protocol: dict | None = None, bench_d: dict | None = None) -> list[str]:
    by_name = {row["system"]: row for row in bench_c["systems"]}
    oracle = by_name["oracle_16"]["semantic_accuracy"]
    learned = by_name["listwise"]["semantic_accuracy"]
    gated = by_name["gated"]["semantic_accuracy"]
    rules = by_name["rules_old8"]["semantic_accuracy"]
    alt = by_name["alt_cost"]
    best = max(learned, gated)
    best_name = "listwise" if learned >= gated else "gated"
    best_row = by_name[best_name]
    lines = [
        (
            f"On the frozen 16-candidate pool, BENCH-C oracle is {_pct(oracle)}. "
            f"The old code-length rule scores {_pct(rules)} "
            f"(regressions {_pct(by_name['rules_old8']['regressed'])}). "
            f"Listwise linear scores {_pct(learned)} "
            f"(regressions {_pct(by_name['listwise']['regressed'])}). "
            f"The confidence-gated scorer scores {_pct(gated)} "
            f"(regressions {_pct(by_name['gated']['regressed'])}). "
            f"Charging alternation literals by hand scores {_pct(alt['semantic_accuracy'])} "
            f"(regressions {_pct(alt['regressed'])})."
        )
    ]
    if protocol is not None:
        val_grid = protocol.get("margin_grid", {}).get("grid", [])
        if val_grid:
            strict = [row for row in val_grid if row["margin"] >= 0.05]
            if strict and all(abs(row["semantic_accuracy"] - strict[0]["semantic_accuracy"]) < 1e-9 for row in strict):
                lines.append(
                    f"The chosen margin is {protocol['margin']}. "
                    f"On the BENCH-B validation fold, margin 0 regresses "
                    f"{_pct(val_grid[0]['regressed'])} of tasks and scores "
                    f"{_pct(val_grid[0]['semantic_accuracy'])}, so it stays inside the 2% budget and wins. "
                    f"Every margin of 0.05 or wider scores {_pct(strict[0]['semantic_accuracy'])} there, "
                    "matching the old rule. The gate does not produce a setting between trusting the "
                    "listwise model on every task and falling back on every task."
                )
    broad = _family_line(bench_c, "too_broad")
    success = _family_line(bench_c, "success")
    variable = _family_line(bench_c, "variable-length inference")
    if broad and success and variable:
        lines.append(
            f"The gain sits on the heuristic's too-broad failures "
            f"(old rule {_pct(broad['rules_old8'])}, listwise {_pct(broad['listwise'])}, "
            f"oracle {_pct(broad['oracle_16'])}, n={broad['n']}). "
            f"The success majority moves the other way "
            f"(heuristic {_pct(success['heuristic'])}, listwise {_pct(success['listwise'])}, n={success['n']}). "
            f"Variable-length stays far from the ceiling "
            f"(listwise {_pct(variable['listwise'])}, oracle {_pct(variable['oracle_16'])}, n={variable['n']})."
        )
    if best >= 0.98 and best_row["regressed"] <= 0.02:
        lines.append(
            f"The {best_name} scorer clears the gate: about 98% or better on BENCH-C "
            "with regressions inside 2%. A tiny neural ranker is optional rather than required."
        )
    elif best >= 0.98:
        lines.append(
            "Accuracy reaches the high 90s, and the regression rate is still above 2%. "
            "The confidence gate did not buy the safety budget on this validation choice."
        )
    else:
        gap = oracle - best
        lines.append(
            f"This misses the gate of about 98% BENCH-C with regressions under 2%. "
            f"The cheap scorer leaves {gap * 100:.1f} points versus oracle@16, "
            "and it regresses more tasks than the old rule. "
            "A tiny neural ranker trained listwise on the same frozen pool is the next selector, "
            "not a larger candidate generator. It still has to keep the regression rate near the old rule's."
        )
    if bench_d is not None:
        d_by = {row["system"]: row for row in bench_d["systems"]}
        lines.append(
            f"BENCH-D was not used for selection. There the listwise scorer scores "
            f"{_pct(d_by['listwise']['semantic_accuracy'])}, the old rule "
            f"{_pct(d_by['rules_old8']['semantic_accuracy'])}, and oracle@16 "
            f"{_pct(d_by['oracle_16']['semantic_accuracy'])}."
        )
    lines.append(
        "BENCH-B and BENCH-C share document field families, so this BENCH-C number is an upper bound "
        "on transfer. Temperature, L2, and the margin were frozen on the BENCH-B validation fold "
        "before BENCH-C was scored."
    )
    return lines


def _markdown_table(headers: list[str], body: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return lines


SYSTEMS = (
    ("heuristic", "heuristic"),
    ("rules_old8", "code-length rule, old 8"),
    ("rules_16", "code-length rule, structural 16"),
    ("alt_cost", "code length plus alternation-literal charge"),
    ("pointwise", "pointwise ridge on hidden accuracy"),
    ("listwise", "listwise linear"),
    ("gated", "listwise, else the old rule"),
    ("oracle_16", "oracle@16"),
)


def _group(rows: list[dict], columns: dict[str, list[float]], note: str, families: bool) -> dict:
    systems = []
    for key, _label in SYSTEMS:
        packed = _pack_system(rows, columns[key])
        packed["system"] = key
        systems.append(packed)
    return {
        "note": note,
        "n": len(rows),
        "systems": systems,
        "families": _family_table(rows, {key: columns[key] for key, _label in SYSTEMS}) if families else [],
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Frozen-pool selector",
        "",
        "The structural generator stays as it is. Each task offers 16 candidates, "
        "the prefix of the structural round-robin. A linear scorer is trained listwise: "
        "the target is a softmax over hidden semantic accuracy, and the inputs are visible "
        "features only. If the top two probabilities are close, the scorer keeps the "
        "code-length choice from the original 8-candidate neighborhood.",
        "",
        "Training tasks are the BENCH-B hash fold. Temperature, L2, and the margin are "
        "chosen on the held-out fifth of BENCH-B. BENCH-C is scored once. BENCH-A is not used.",
        "",
        "## Reading",
        "",
    ]
    for paragraph in report["reading"]:
        lines.append(paragraph)
        lines.append("")
    protocol = report["protocol"]
    lines.append("## Protocol")
    lines.append("")
    lines.append(
        f"Listwise temperature {protocol['tau']}, L2 {protocol['l2']}, "
        f"stopped at step {protocol['step']} with validation accuracy {_pct(protocol['val_accuracy'])}. "
        f"Pointwise ridge lambda {protocol['pointwise_l2']} "
        f"(validation {_pct(protocol['pointwise_val'])}). "
        f"Confidence margin {protocol['margin']}. "
        f"Features: {len(FEATURE_NAMES)}."
    )
    lines.append("")
    for name, group in report["groups"].items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(group["note"])
        lines.append("")
        headers = ["System", "Semantic accuracy", "Macro", "Improved", "Regressed", "Unchanged"]
        body = []
        for key, label in SYSTEMS:
            row = next(item for item in group["systems"] if item["system"] == key)
            body.append([
                label,
                _pct(row["semantic_accuracy"]),
                _pct(row["macro"]),
                _pct(row["improved"]),
                _pct(row["regressed"]),
                _pct(row["unchanged"]),
            ])
        lines.extend(_markdown_table(headers, body))
        lines.append("")
        if group["families"]:
            lines.append("Heuristic-failure families:")
            lines.append("")
            headers = ["Failure", "n", "heuristic", "old rule", "listwise", "gated", "oracle@16"]
            body = []
            for row in group["families"]:
                body.append([
                    row["failure"],
                    str(row["n"]),
                    _pct(row["heuristic"]),
                    _pct(row["rules_old8"]),
                    _pct(row["listwise"]),
                    _pct(row["gated"]),
                    _pct(row["oracle_16"]),
                ])
            lines.extend(_markdown_table(headers, body))
            lines.append("")
    return "\n".join(lines)


def _columns_for(rows: list[dict], weights, means, scales, pointwise, margin: float) -> dict[str, list[float]]:
    listwise = []
    gated = _gated_values(rows, weights, means, scales, margin)
    matrices = _normalize(rows, means, scales)
    pointwise_values = []
    for row, matrix in zip(rows, matrices):
        scores = score_patterns(weights, matrix)
        listwise.append(row["accuracy"][rank_index(scores, row["patterns"], row["heuristic"])])
        point_scores = _pointwise_scores(pointwise, row["matrix"])
        pointwise_values.append(row["accuracy"][rank_index(point_scores, row["patterns"], row["heuristic"])])
    return {
        "heuristic": [row["heuristic_accuracy"] for row in rows],
        "rules_old8": [row["rules8_accuracy"] for row in rows],
        "rules_16": [row["rules16_accuracy"] for row in rows],
        "alt_cost": [row["alt_accuracy"] for row in rows],
        "pointwise": pointwise_values,
        "listwise": listwise,
        "gated": gated,
        "oracle_16": [row["oracle_accuracy"] for row in rows],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    out = root / "results" / "exp2"
    out.mkdir(parents=True, exist_ok=True)
    files = ("bench_b", "bench_c", "bench_d")
    packed: list[dict] = []
    for name in files:
        tasks = read_jsonl(data / f"{name}.jsonl")
        started = time.perf_counter()
        for index, task in enumerate(tasks, start=1):
            packed.append(pack_task(task))
            if index % 200 == 0:
                print(f"{name} {index}/{len(tasks)} in {time.perf_counter() - started:.1f}s", flush=True)
        print(f"{name} {len(tasks)}/{len(tasks)} in {time.perf_counter() - started:.1f}s", flush=True)
    train = [row for row in packed if row["benchmark"] == "B" and row["fold"] == "train"]
    val = [row for row in packed if row["benchmark"] == "B" and row["fold"] == "val"]
    print(f"train {len(train)} val {len(val)}", flush=True)
    flat = [vector for row in train for vector in row["matrix"]]
    means, scales = standardize_fit(flat)
    pointwise, point_lam, point_val = choose_pointwise(train, val)
    print(f"pointwise l2={point_lam} val={point_val:.4f}", flush=True)
    weights, list_key = choose_listwise(train, val, means, scales)
    margin, margin_info = choose_margin(val, weights, means, scales)
    print(f"margin {margin} val gated {margin_info['chosen']['semantic_accuracy']:.4f}", flush=True)
    groups = {}
    notes = {
        "B-val": "Validation fold. Used to choose temperature, L2, and the margin.",
        "C": "Untouched document tasks. Families overlap BENCH-B, so this is an upper bound on transfer.",
        "D": "Sample-size curve. Not used to fit or to choose the margin.",
    }
    slices = {
        "B-val": val,
        "C": [row for row in packed if row["benchmark"] == "C"],
        "D": [row for row in packed if row["benchmark"] == "D"],
    }
    for name, rows in slices.items():
        columns = _columns_for(rows, weights, means, scales, pointwise, margin)
        groups[name] = _group(rows, columns, notes[name], families=(name == "C"))
    report = {
        "protocol": {
            "tau": list_key["tau"],
            "l2": list_key["l2"],
            "step": list_key["step"],
            "val_accuracy": list_key["val_accuracy"],
            "pointwise_l2": point_lam,
            "pointwise_val": point_val,
            "margin": margin,
            "margin_grid": margin_info,
            "k": K_FROZEN,
            "features": list(FEATURE_NAMES),
            "n_train": len(train),
            "n_val": len(val),
        },
        "groups": groups,
        "reading": [],
    }
    report["reading"] = _interpret(groups["C"], report["protocol"], groups["D"])
    (out / "selector.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "selector.md").write_text(render_markdown(report), encoding="utf-8")
    print("\n".join(report["reading"]))


if __name__ == "__main__":
    main()
