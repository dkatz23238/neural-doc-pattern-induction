"""Residual override: learn when an alternative beats the old code-length rule.

Train on document compositions. Choose the epoch, the override threshold, and
pooled versus cross-attention on held-out formatting variants. BENCH-C is
scored once. BENCH-A is not used.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from pattern_induction.candidate_oracle import _rates
from pattern_induction.candidates import expanded_candidates, pattern_cost, presented_candidates
from pattern_induction.override import (
    DEFAULT_LAMBDA,
    PAIR_FEATURES,
    ResidualOverride,
    choose_alternative,
    collate_pairs,
    encode_utf8,
    override_loss,
    pair_rows,
    parameter_count,
)
from pattern_induction.override_split import generate_split
from pattern_induction.selector import K_FROZEN
from pattern_induction.tasks import read_jsonl
from pattern_induction.taxonomy import categorize

THRESHOLDS = (-0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2, 0.4)
REGRESSION_BUDGET = 0.02
RETRY_LAMBDA = 10.0


def _rule_index(patterns: list[str], strings: list[str], heuristic: str) -> int:
    best = 0
    best_cost = pattern_cost(patterns[0], strings, True)
    for index, pattern in enumerate(patterns[1:], start=1):
        cost = pattern_cost(pattern, strings, True)
        if cost < best_cost - 1e-9:
            best, best_cost = index, cost
        elif abs(cost - best_cost) <= 1e-9 and pattern == heuristic and patterns[best] != heuristic:
            best, best_cost = index, cost
    return best


def pack_task(task) -> dict:
    structural, _pool = expanded_candidates(task.visible, k=K_FROZEN, pool_limit=160)
    patterns = [item.pattern for item in structural[:K_FROZEN]]
    shown, _old = presented_candidates(task.visible, k=8, pool_limit=80)
    old = [item.pattern for item in shown]
    if not patterns:
        patterns = old[:1] or ["^$"]
    if not old:
        old = patterns[:1]
    heuristic = patterns[0]
    rule = old[_rule_index(old, task.visible, old[0])]
    scored: dict[str, tuple[float, float, float]] = {}

    def rates(pattern: str) -> tuple[float, float, float]:
        if pattern not in scored:
            scored[pattern] = _rates(pattern, task)
        return scored[pattern]

    heur_acc, heur_rec, heur_rej = rates(heuristic)
    rule_acc = rates(rule)[0]
    alts = []
    for pattern in patterns:
        if pattern != rule and pattern not in alts:
            alts.append(pattern)
    alt_accuracy = [rates(pattern)[0] for pattern in alts]
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
    truncated = False
    string_ids = []
    string_pad = []
    for text in task.visible:
        ids, pad, cut = encode_utf8(text)
        truncated = truncated or cut
        string_ids.append(ids)
        string_pad.append(pad)
    rule_ids, rule_pad, cut = encode_utf8(rule)
    truncated = truncated or cut
    alt_ids = []
    alt_pad = []
    for pattern in alts:
        ids, pad, cut = encode_utf8(pattern)
        truncated = truncated or cut
        alt_ids.append(ids)
        alt_pad.append(pad)
    return {
        "id": task.id,
        "benchmark": task.benchmark,
        "domain": task.domain,
        "split": task.split,
        "failure": failure or "success",
        "truncated": truncated,
        "heuristic_accuracy": heur_acc,
        "rules_accuracy": rule_acc,
        "oracle_accuracy": max([rule_acc, *alt_accuracy]) if alt_accuracy else rule_acc,
        "string_ids": string_ids,
        "string_pad": string_pad,
        "rule_ids": rule_ids,
        "rule_pad": rule_pad,
        "alt_ids": alt_ids,
        "alt_pad": alt_pad,
        "alt_accuracy": alt_accuracy,
        "alt_delta": [accuracy - rule_acc for accuracy in alt_accuracy],
        "features": pair_rows(alts, rule, heuristic, task.visible) if alts else [],
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _movement(accuracy: float, baseline: float) -> str:
    if accuracy > baseline + 0.01:
        return "improved"
    if accuracy < baseline - 0.01:
        return "regressed"
    return "unchanged"


def _fit_scale(rows: list[dict]) -> tuple[list[float], list[float]]:
    vectors = [vector for row in rows for vector in row["features"]]
    width = len(PAIR_FEATURES)
    if not vectors:
        return [0.0] * width, [1.0] * width
    count = float(len(vectors))
    means = []
    scales = []
    for col in range(width):
        column = [vector[col] for vector in vectors]
        center = sum(column) / count
        variance = sum((value - center) ** 2 for value in column) / count
        scale = variance ** 0.5
        means.append(center)
        scales.append(scale if scale >= 1e-8 else 1.0)
    return means, scales


def _batches(rows: list[dict], batch_size: int, shuffle: bool, rng: random.Random):
    order = list(range(len(rows)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [rows[index] for index in order[start : start + batch_size]]


def predict_deltas(model, rows: list[dict], mean, scale, batch_size: int) -> list[list[float]]:
    model.eval()
    predicted = [[] for _ in rows]
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = collate_pairs(chunk, mean, scale)
        with torch.inference_mode():
            delta, _logit = model(
                batch["strings"],
                batch["string_pad"],
                batch["string_slot_pad"],
                batch["rule"],
                batch["rule_pad"],
                batch["alts"],
                batch["alt_pad"],
                batch["alt_slot_pad"],
                batch["features"],
            )
        for local, row in enumerate(chunk):
            count = len(row["alt_delta"])
            predicted[start + local] = [float(value) for value in delta[local, :count].tolist()]
    return predicted


def apply_threshold(rows: list[dict], deltas: list[list[float]], threshold: float) -> tuple[list[float], list[bool]]:
    values = []
    overrode = []
    for row, scores in zip(rows, deltas):
        pick = choose_alternative(scores, threshold)
        if pick is None:
            values.append(row["rules_accuracy"])
            overrode.append(False)
        else:
            values.append(row["alt_accuracy"][pick])
            overrode.append(True)
    return values, overrode


def _pack_system(rows: list[dict], values: list[float], overrode: list[bool]) -> dict:
    movement = {"improved": 0, "regressed": 0, "unchanged": 0}
    false_override = 0
    by_domain: dict[str, list[float]] = defaultdict(list)
    for row, accuracy, took in zip(rows, values, overrode):
        movement[_movement(accuracy, row["heuristic_accuracy"])] += 1
        if took and accuracy < row["rules_accuracy"] - 0.01:
            false_override += 1
        by_domain[row["domain"]].append(accuracy)
    total = len(rows) or 1
    return {
        "semantic_accuracy": round(_mean(values), 4),
        "macro": round(_mean([_mean(scores) for scores in by_domain.values()]), 4) if by_domain else 0.0,
        "improved": round(movement["improved"] / total, 4),
        "regressed": round(movement["regressed"] / total, 4),
        "unchanged": round(movement["unchanged"] / total, 4),
        "override_rate": round(sum(overrode) / total, 4),
        "false_override": round(false_override / total, 4),
        "n": len(rows),
    }


def choose_threshold(rows: list[dict], deltas: list[list[float]]) -> tuple[float, dict]:
    """Highest validation accuracy whose regression rate stays inside the budget."""
    recorded = []
    eligible = []
    for threshold in THRESHOLDS:
        values, overrode = apply_threshold(rows, deltas, threshold)
        packed = _pack_system(rows, values, overrode)
        packed["threshold"] = threshold
        recorded.append(packed)
        if packed["regressed"] <= REGRESSION_BUDGET + 1e-12:
            eligible.append(packed)
    pool = eligible or recorded
    chosen = max(pool, key=lambda row: (row["semantic_accuracy"], -row["regressed"], row["threshold"]))
    return chosen["threshold"], {"grid": recorded, "chosen": chosen, "budget": REGRESSION_BUDGET, "eligible": bool(eligible)}


def _selection_score(packed: dict) -> float:
    accuracy = packed["semantic_accuracy"]
    if packed["regressed"] <= REGRESSION_BUDGET:
        return accuracy
    return accuracy - (packed["regressed"] - REGRESSION_BUDGET)


def train_model(name: str, model, train_rows, val_rows, mean, scale, epochs, patience, batch_size, lr, lam, seed):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    best_state = copy.deepcopy(model.state_dict())
    best_score = -1.0e9
    best_epoch = 0
    best_threshold = 0.0
    best_packed = None
    stall = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_false = 0.0
        seen = 0
        for chunk in _batches(train_rows, batch_size, shuffle=True, rng=rng):
            batch = collate_pairs(chunk, mean, scale)
            optimizer.zero_grad(set_to_none=True)
            delta, logit = model(
                batch["strings"],
                batch["string_pad"],
                batch["string_slot_pad"],
                batch["rule"],
                batch["rule_pad"],
                batch["alts"],
                batch["alt_pad"],
                batch["alt_slot_pad"],
                batch["features"],
            )
            loss, parts = override_loss(delta, logit, batch["actual"], ~batch["alt_slot_pad"], lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = len(chunk)
            total_loss += parts["loss"] * size
            total_false += parts["false_override"] * size
            seen += size
        deltas = predict_deltas(model, val_rows, mean, scale, batch_size)
        threshold, info = choose_threshold(val_rows, deltas)
        packed = info["chosen"]
        score = _selection_score(packed)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(seen, 1),
                "train_false": total_false / max(seen, 1),
                "threshold": threshold,
                "val": packed,
                "score": score,
            }
        )
        mark = ""
        if score > best_score + 1e-4:
            best_score = score
            best_epoch = epoch
            best_threshold = threshold
            best_packed = packed
            best_state = copy.deepcopy(model.state_dict())
            stall = 0
            mark = " *"
        else:
            stall += 1
        print(
            f"{name} epoch {epoch:02d} loss {total_loss / max(seen, 1):.3f} "
            f"val {packed['semantic_accuracy']:.3f} reg {packed['regressed']:.3f} "
            f"override {packed['override_rate']:.3f} t={threshold}{mark}",
            flush=True,
        )
        if stall >= patience:
            break
    model.load_state_dict(best_state)
    return {
        "name": name,
        "epoch": best_epoch,
        "threshold": best_threshold,
        "val": best_packed,
        "score": best_score,
        "history": history,
        "parameters": parameter_count(model),
        "lam": lam,
    }


def _constant(rows: list[dict], key: str) -> tuple[list[float], list[bool]]:
    values = [row[key] for row in rows]
    overrode = [False] * len(rows)
    return values, overrode


def _oracle(rows: list[dict]) -> tuple[list[float], list[bool]]:
    values = []
    overrode = []
    for row in rows:
        values.append(row["oracle_accuracy"])
        overrode.append(row["oracle_accuracy"] > row["rules_accuracy"] + 1e-12)
    return values, overrode


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


def _interpret(selected: dict, bench_c: dict) -> list[str]:
    by_name = {row["system"]: row for row in bench_c["systems"]}
    model = by_name[selected["name"]]
    rules = by_name["rules"]
    oracle = by_name["oracle"]
    accuracy = model["semantic_accuracy"]
    regressed = model["regressed"]
    lines = [
        (
            f"The val-selected residual model is {selected['name']} "
            f"(λ={selected['lam']}, epoch {selected['epoch']}, threshold {selected['threshold']}). "
            f"On BENCH-C it scores {_pct(accuracy)} with regressions {_pct(regressed)}. "
            f"The old rule scores {_pct(rules['semantic_accuracy'])} "
            f"(regressions {_pct(rules['regressed'])}). "
            f"Oracle on the same alternatives is {_pct(oracle['semantic_accuracy'])}."
        )
    ]
    if accuracy >= 0.985 and regressed <= 0.012:
        lines.append(
            "This is a very strong result: at least 98.5% with regressions near the old rule's 0.7%."
        )
    elif accuracy >= 0.98 and regressed <= 0.02:
        lines.append(
            "This clears the go bar: at least 98% with regressions inside 2%."
        )
    elif accuracy >= 0.975 and regressed <= 0.02:
        lines.append(
            "This is promising: the mid-97s with regressions inside 2%, short of the 98% go bar."
        )
    elif accuracy <= 0.97 or regressed > 0.03:
        lines.append(
            "This stops the neural selector. Accuracy is at most 97%, or regressions exceed 3%."
        )
    else:
        lines.append(
            "This sits between the stop line and the promising band. "
            "The residual headroom is still open, and this run does not clear either gate."
        )
    lines.append(
        "Epoch and threshold were frozen on held-out formatting variants before BENCH-C was scored. "
        "Training compositions and BENCH-C still share document families, so this number is an upper bound on transfer."
    )
    return lines


def _markdown_table(headers: list[str], body: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_markdown(report: dict) -> str:
    lines = [
        "# Residual override",
        "",
        "The old code-length rule picks R0 from the original 8-candidate neighborhood. "
        "A tiny model then scores the other candidates in the frozen structural 16 by "
        "Δ = hidden accuracy(Ri) − hidden accuracy(R0). It replaces R0 only when the best "
        "predicted Δ clears a threshold. False overrides are weighted more heavily than missed gains.",
        "",
        "Training uses document compositions. Validation uses the same families with held-out "
        "prefixes, separators, and widths. Pooled and cross-attention models are compared on that "
        "validation split. BENCH-C is scored once. BENCH-A is not used.",
        "",
        "## Reading",
        "",
    ]
    for paragraph in report["reading"]:
        lines.append(paragraph)
        lines.append("")
    selected = report["selected"]
    lines.append("## Protocol")
    lines.append("")
    lines.append(
        f"Selected `{selected['name']}` at epoch {selected['epoch']} with threshold {selected['threshold']} "
        f"and λ={selected['lam']}. Validation accuracy {_pct(selected['val']['semantic_accuracy'])}, "
        f"regressions {_pct(selected['val']['regressed'])}. "
        f"Parameters: {selected['parameters']}."
    )
    lines.append("")
    for name, group in report["groups"].items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(group["note"])
        lines.append("")
        headers = ["System", "Semantic accuracy", "Macro", "Improved", "Regressed", "Overrides", "False overrides"]
        body = []
        for row in group["systems"]:
            body.append([
                row["label"],
                _pct(row["semantic_accuracy"]),
                _pct(row["macro"]),
                _pct(row["improved"]),
                _pct(row["regressed"]),
                _pct(row["override_rate"]),
                _pct(row["false_override"]),
            ])
        lines.extend(_markdown_table(headers, body))
        lines.append("")
        if group["families"]:
            lines.append("Heuristic-failure families:")
            lines.append("")
            headers = ["Failure", "n", "heuristic", "old rule", "residual", "oracle"]
            body = []
            for row in group["families"]:
                body.append([
                    row["failure"],
                    str(row["n"]),
                    _pct(row["heuristic"]),
                    _pct(row["rules"]),
                    _pct(row["residual"]),
                    _pct(row["oracle"]),
                ])
            lines.extend(_markdown_table(headers, body))
            lines.append("")
    return "\n".join(lines)


def _system(key: str, label: str, rows, values, overrode) -> dict:
    packed = _pack_system(rows, values, overrode)
    packed["system"] = key
    packed["label"] = label
    return packed


def _evaluate_model(model, rows, mean, scale, batch_size, threshold):
    deltas = predict_deltas(model, rows, mean, scale, batch_size)
    return apply_threshold(rows, deltas, threshold)


def _pack_rows(tasks, label: str) -> list[dict]:
    rows = []
    started = time.perf_counter()
    for index, task in enumerate(tasks, start=1):
        rows.append(pack_task(task))
        if index % 200 == 0 or index == len(tasks):
            print(f"pack {label} {index}/{len(tasks)} in {time.perf_counter() - started:.1f}s", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1600)
    parser.add_argument("--val", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lam", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-final", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "exp2"
    out.mkdir(parents=True, exist_ok=True)
    train_tasks = generate_split(args.train, "train", seed=2026)
    val_tasks = generate_split(args.val, "val", seed=2027)
    train_rows = _pack_rows(train_tasks, "train")
    val_rows = _pack_rows(val_tasks, "val")
    final_rows = {}
    if not args.skip_final:
        data = root / "data"
        final_rows["C"] = _pack_rows(read_jsonl(data / "bench_c.jsonl"), "bench_c")
        final_rows["D"] = _pack_rows(read_jsonl(data / "bench_d.jsonl"), "bench_d")
    mean, scale = _fit_scale(train_rows)
    runs = []
    for cross, tag in ((True, "cross"), (False, "pooled")):
        model = ResidualOverride(cross_attention=cross)
        print(f"{tag} parameters {parameter_count(model)} λ={args.lam}", flush=True)
        fitted = train_model(
            tag,
            model,
            train_rows,
            val_rows,
            mean,
            scale,
            args.epochs,
            args.patience,
            args.batch_size,
            args.lr,
            args.lam,
            args.seed,
        )
        fitted["cross_attention"] = cross
        fitted["state"] = model.state_dict()
        runs.append(fitted)
    runs.sort(key=lambda item: item["score"], reverse=True)
    best = runs[0]
    needs_retry = best["val"]["regressed"] > REGRESSION_BUDGET or best["val"]["semantic_accuracy"] < 0.97
    if needs_retry:
        print(f"retry {best['name']} at λ={RETRY_LAMBDA}", flush=True)
        model = ResidualOverride(cross_attention=best["cross_attention"])
        fitted = train_model(
            f"{best['name']}-l{int(RETRY_LAMBDA)}",
            model,
            train_rows,
            val_rows,
            mean,
            scale,
            args.epochs,
            args.patience,
            args.batch_size,
            args.lr,
            RETRY_LAMBDA,
            args.seed,
        )
        fitted["cross_attention"] = best["cross_attention"]
        fitted["state"] = model.state_dict()
        runs.append(fitted)
        runs.sort(key=lambda item: item["score"], reverse=True)
    selected = runs[0]
    print(
        f"selected {selected['name']} score {selected['score']:.4f} "
        f"val {selected['val']['semantic_accuracy']:.4f}",
        flush=True,
    )
    groups = {}
    notes = {
        "OOD-val": "Held-out formatting variants. Used to choose epoch, threshold, and architecture.",
        "C": "Untouched document tasks, scored once. Families overlap the training compositions.",
        "D": "Sample-size curve. Not used to fit or to choose the threshold.",
    }
    eval_rows = {"OOD-val": val_rows, **final_rows}
    loaded = []
    for run in runs:
        model = ResidualOverride(cross_attention=run["cross_attention"])
        model.load_state_dict(run["state"])
        model.eval()
        loaded.append((run, model))
    for name, rows in eval_rows.items():
        systems = [
            _system("heuristic", "heuristic", rows, *_constant(rows, "heuristic_accuracy")),
            _system("rules", "old code-length rule", rows, *_constant(rows, "rules_accuracy")),
        ]
        columns = {
            "heuristic": [row["heuristic_accuracy"] for row in rows],
            "rules": [row["rules_accuracy"] for row in rows],
            "oracle": [row["oracle_accuracy"] for row in rows],
        }
        residual_values = None
        for run, model in loaded:
            values, overrode = _evaluate_model(model, rows, mean, scale, args.batch_size, run["threshold"])
            label = f"{run['name']} residual"
            systems.append(_system(run["name"], label, rows, values, overrode))
            if run["name"] == selected["name"]:
                residual_values = values
                columns["residual"] = values
        systems.append(_system("oracle", "oracle on R0 and the alternatives", rows, *_oracle(rows)))
        families = _family_table(rows, columns) if name == "C" and residual_values is not None else []
        groups[name] = {"note": notes[name], "n": len(rows), "systems": systems, "families": families}
    public_runs = []
    for run in runs:
        public_runs.append(
            {
                "name": run["name"],
                "cross_attention": run["cross_attention"],
                "lam": run["lam"],
                "epoch": run["epoch"],
                "threshold": run["threshold"],
                "score": run["score"],
                "parameters": run["parameters"],
                "val": run["val"],
                "history": [
                    {
                        "epoch": item["epoch"],
                        "train_loss": item["train_loss"],
                        "threshold": item["threshold"],
                        "val_accuracy": item["val"]["semantic_accuracy"],
                        "val_regressed": item["val"]["regressed"],
                        "val_override": item["val"]["override_rate"],
                    }
                    for item in run["history"]
                ],
            }
        )
    report = {
        "selected": public_runs[0],
        "runs": public_runs,
        "features": list(PAIR_FEATURES),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "groups": groups,
        "reading": [],
    }
    if "C" in groups:
        report["reading"] = _interpret(public_runs[0], groups["C"])
    else:
        report["reading"] = ["Final benchmarks were skipped."]
    (out / "override.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "override.md").write_text(render_markdown(report), encoding="utf-8")
    print("\n".join(report["reading"]))


if __name__ == "__main__":
    main()
