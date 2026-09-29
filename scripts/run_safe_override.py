"""Train a conservative override of the 96.2% symbolic selector.

The alternative is the program Needle already returns. Training uses document
compositions and only the rows where the two programs differ. The threshold is
frozen on held-out formats. BENCH-C is scored once.
"""

from __future__ import annotations

import argparse
import json
import random
import signal
import time
from pathlib import Path

import torch

from pattern_induction.candidate_oracle import _rates
from pattern_induction.candidates import pattern_cost, presented_candidates
from pattern_induction.needle import needle
from pattern_induction.override_split import generate_split
from pattern_induction.program import MAX_LEN, MAX_SUPPORT, _MEMBERSHIP
from pattern_induction.safe_override import (
    EASY_MARGIN,
    FEATURE_NAMES,
    IMPROVE_EPS,
    LAMBDA,
    REGRESSION_BUDGET,
    TARGET_ACCURACY,
    SafeOverride,
    assemble_features,
    choose_threshold,
    clears_minimum,
    clears_target,
    fit_scale,
    override_choice,
    parameter_count,
    safe_loss,
    standardize,
    summarize,
)
from pattern_induction.tasks import read_jsonl
from pattern_induction.taxonomy import categorize
from prior.expansion import freeze_evidence
from prior.relative import LocalScorer, pack_example

CACHE = Path("/tmp/safe-override")
NEEDLE_LIMIT_S = 25


class _Timeout(Exception):
    pass


def _alarm(signum, frame) -> None:
    raise _Timeout()


def _rule(task) -> tuple[str, float]:
    """Code-length choice on the original 8-candidate neighborhood."""
    shown, _pool = presented_candidates(task.visible, k=8, pool_limit=80)
    patterns = [item.pattern for item in shown] or ["^$"]
    heuristic = patterns[0]
    best = 0
    best_cost = pattern_cost(patterns[0], task.visible, True)
    for index, pattern in enumerate(patterns[1:], start=1):
        cost = pattern_cost(pattern, task.visible, True)
        if cost < best_cost - 1e-9:
            best, best_cost = index, cost
        elif abs(cost - best_cost) <= 1e-9 and pattern == heuristic and patterns[best] != heuristic:
            best, best_cost = index, cost
    pattern = patterns[best]
    return pattern, _rates(pattern, task)[0]


def _failure(task, pattern: str) -> str:
    accuracy, positive, negative = _rates(pattern, task)
    label = categorize(
        {
            "semantic_accuracy": accuracy,
            "positive_recall": positive,
            "negative_rejection": negative,
            "valid": 1.0,
            "catastrophic": 0.0,
            "gold_regex": task.gold_regex,
            "domain": task.domain,
        }
    )
    return label or "success"


def _evidence(model: LocalScorer, prog, supports: list[str]) -> float | None:
    if prog is None:
        return None
    packed = pack_example([prog], [False], [True], supports)
    batch = {
        key: value.unsqueeze(0)
        for key, value in packed.items()
        if "hyp" not in key and (key.endswith("_feat") or key.endswith("_mask"))
    }
    with torch.no_grad():
        scores = model.program_scores(batch)
    return float(scores[0, 0])


def _needle(examples: list[str], root: Path):
    _MEMBERSHIP.clear()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(NEEDLE_LIMIT_S)
    try:
        return needle(examples, root)
    except _Timeout:
        return None
    finally:
        signal.alarm(0)


def pack_task(task, evidence: LocalScorer, root: Path, symbolic: str | None = None, symbolic_accuracy: float | None = None, failure: str | None = None) -> dict:
    if symbolic is None or symbolic_accuracy is None:
        symbolic, symbolic_accuracy = _rule(task)
    supports = [text[:MAX_LEN] for text in task.visible if text][:MAX_SUPPORT]
    result = _needle(task.visible, root)
    if result is None or not result.regex:
        alternative = symbolic
        alt_accuracy = symbolic_accuracy
        score = None
        confidence = 0.0
        timed_out = result is None
    else:
        alternative = result.regex
        alt_accuracy = _rates(alternative, task)[0]
        score = _evidence(evidence, result.ast, supports)
        confidence = float(result.confidence)
        timed_out = False
    disagreed = bool(alternative) and alternative != symbolic
    features = assemble_features(
        alternative,
        symbolic,
        task.visible,
        0.0 if score is None else score,
        score is not None,
        confidence,
    )
    if failure is None:
        failure = _failure(task, symbolic)
    return {
        "id": task.id,
        "domain": task.domain,
        "failure": failure,
        "symbolic_accuracy": symbolic_accuracy,
        "alt_accuracy": alt_accuracy if disagreed else symbolic_accuracy,
        "disagreed": disagreed,
        "timed_out": timed_out,
        "label": 1.0 if disagreed and alt_accuracy > symbolic_accuracy + IMPROVE_EPS else 0.0,
        "features": features,
    }


def _write_cache(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_cache(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _pack_split(tasks, evidence, root: Path, label: str) -> list[dict]:
    rows = []
    started = time.perf_counter()
    for index, task in enumerate(tasks, start=1):
        rows.append(pack_task(task, evidence, root))
        if index % 20 == 0 or index == len(tasks):
            elapsed = time.perf_counter() - started
            print(f"pack {label} {index}/{len(tasks)} in {elapsed:.1f}s", flush=True)
    return rows


def _stored_symbolic(root: Path) -> dict[str, dict]:
    stored = {}
    with (root / "results" / "exp2" / "candidate_oracle_tasks.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            stored[row["id"]] = {
                "pattern": row["rules"]["pattern"],
                "accuracy": row["rules"]["accuracy"],
                "failure": row.get("failure") or "success",
            }
    return stored


def _pack_bench(root: Path, evidence, tasks) -> list[dict]:
    stored = _stored_symbolic(root)
    rows = []
    started = time.perf_counter()
    for index, task in enumerate(tasks, start=1):
        fact = stored.get(task.id)
        if fact is None:
            rows.append(pack_task(task, evidence, root))
        else:
            rows.append(
                pack_task(
                    task,
                    evidence,
                    root,
                    symbolic=fact["pattern"],
                    symbolic_accuracy=fact["accuracy"],
                    failure=fact["failure"],
                )
            )
        if index % 20 == 0 or index == len(tasks):
            print(f"pack bench {index}/{len(tasks)} in {time.perf_counter() - started:.1f}s", flush=True)
    return rows


def _tensor(rows: list[dict], mean: list[float], scale: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    features = [standardize(row["features"], mean, scale) for row in rows]
    labels = [row["label"] for row in rows]
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def _probabilities(model: SafeOverride, rows: list[dict], mean: list[float], scale: list[float]) -> list[float]:
    if not rows:
        return []
    model.eval()
    features, _labels = _tensor(rows, mean, scale)
    with torch.no_grad():
        logit = model(features)
        probability = torch.sigmoid(logit)
    return [float(value) for value in probability.tolist()]


def train(train_rows: list[dict], val_rows: list[dict], epochs: int, patience: int, batch_size: int, lr: float, seed: int) -> dict:
    disagreements = [row for row in train_rows if row["disagreed"]]
    if len(disagreements) < 8:
        raise RuntimeError(f"only {len(disagreements)} training disagreements")
    mean, scale = fit_scale(train_rows)
    model = SafeOverride()
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    best_state = {key: value.clone() for key, value in model.state_dict().items()}
    best_score = -1.0e9
    best = None
    stall = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(disagreements)))
        rng.shuffle(order)
        total = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            chunk = [disagreements[index] for index in order[start : start + batch_size]]
            features, labels = _tensor(chunk, mean, scale)
            optimizer.zero_grad(set_to_none=True)
            loss = safe_loss(model(features), labels, LAMBDA)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(chunk)
            seen += len(chunk)
        probabilities = _probabilities(model, val_rows, mean, scale)
        threshold, info = choose_threshold(val_rows, probabilities)
        chosen = info["chosen"]
        score = chosen["semantic_accuracy"] if chosen["eligible"] else chosen["symbolic_accuracy"] - 1.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / max(seen, 1),
                "threshold": threshold,
                "val_accuracy": chosen["semantic_accuracy"],
                "val_symbolic": chosen["symbolic_accuracy"],
                "val_regressed": chosen["regression_rate"],
                "val_net": chosen["net"],
                "eligible": chosen["eligible"],
                "score": score,
            }
        )
        mark = ""
        if score > best_score + 1e-6:
            best_score = score
            best = {
                "epoch": epoch,
                "threshold": threshold,
                "val": {key: value for key, value in chosen.items() if key != "families"},
                "val_families": chosen["families"],
                "score": score,
                "eligible": chosen["eligible"],
            }
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            stall = 0
            mark = " *"
        else:
            stall += 1
        print(
            f"epoch {epoch:02d} loss {total / max(seen, 1):.3f} "
            f"val {chosen['semantic_accuracy']:.4f} sym {chosen['symbolic_accuracy']:.4f} "
            f"net {chosen['net']} reg {chosen['regression_rate']:.3f} t={threshold}{mark}",
            flush=True,
        )
        if stall >= patience:
            break
    model.load_state_dict(best_state)
    return {
        "model": model,
        "mean": mean,
        "scale": scale,
        "history": history,
        "selected": best,
        "parameters": parameter_count(model),
        "train_n": len(train_rows),
        "train_disagreements": len(disagreements),
        "train_positives": sum(1 for row in disagreements if row["label"] > 0.5),
        "val_n": len(val_rows),
        "val_disagreements": sum(1 for row in val_rows if row["disagreed"]),
    }


def _system(name: str, label: str, summary: dict) -> dict:
    packed = {key: value for key, value in summary.items() if key != "families"}
    packed["system"] = name
    packed["label"] = label
    return packed


def _constant(rows: list[dict], key: str) -> dict:
    take = [key == "alt_accuracy" and row["disagreed"] for row in rows]
    if key == "symbolic_accuracy":
        take = [False] * len(rows)
    elif key == "oracle":
        take = [row["disagreed"] and row["alt_accuracy"] > row["symbolic_accuracy"] + 1e-12 for row in rows]
    return summarize(rows, take)


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _reading(report: dict) -> list[str]:
    bench = report["bench"]
    selected = report["selected"]
    val = selected["val"]
    lines = [
        (
            "The factory prior is negative for v1. Typed scales did not reliably beat the frozen "
            "evidence layer on held-out concepts (0.451 / 0.458 against evidence 0.472 / 0.458). "
            "This page asks a smaller question: keep the symbolic winner, and override it only when "
            "a model is sure Needle's existing program is better."
        ),
        (
            f"The val-selected threshold is {selected['threshold']} at epoch {selected['epoch']} "
            f"(λ={LAMBDA}). On the held-out formats the override scores {_pct(val['semantic_accuracy'])} "
            f"against a symbolic {_pct(val['symbolic_accuracy'])}, with net "
            f"{val['net']} (fixed {val['fixed']} minus regressions {val['regressed']})."
        ),
    ]
    lines.append(
        (
            f"On BENCH-C the override scores {_pct(bench['semantic_accuracy'])}. "
            f"The symbolic selector scores {_pct(bench['symbolic_accuracy'])}. "
            f"Needle's alternative scores {_pct(bench['alternative_accuracy'])}. "
            f"The oracle union of those two scores {_pct(bench['oracle_accuracy'])}. "
            f"Errors fixed {bench['fixed']}, regressions {bench['regressed']}, net {bench['net']}."
        )
    )
    easy = next((row for row in bench["families"] if row["failure"] == "success"), None)
    if easy:
        lines.append(
            (
                f"The easy family (heuristic success, n={easy['n']}) is {_pct(easy['symbolic'])} symbolic "
                f"and {_pct(easy['chosen'])} with the override."
            )
        )
    if bench["cleared_target"]:
        lines.append(
            "This clears the practical v1 bar: above 96.2%, inside the 96.7% band, "
            "without a meaningful regression on the easy family. Freeze this override."
        )
    elif bench["cleared_minimum"]:
        lines.append(
            "This beats the symbolic baseline without a meaningful easy-family regression. "
            "It stays short of the 96.7% band."
        )
    else:
        lines.append(
            "This does not clear the v1 bar. Ship the 96.2% symbolic selector together with the "
            "learned evidence and confidence machinery. Do not spend another search on the half point."
        )
    lines.append(
        "Epoch and threshold were frozen on held-out formatting variants before BENCH-C was scored. "
        "Training compositions and BENCH-C share document families, so a gain here is an upper bound on transfer. "
        "BENCH-E was not scored."
    )
    return lines


def _table(headers: list[str], body: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_markdown(report: dict) -> str:
    bench = report["bench"]
    lines = [
        "# Safe correction of the symbolic selector",
        "",
        "The default is the code-length rule. The alternative is the program Needle already returns "
        "for the same supports. The model sees the symbolic winner, that alternative, a summary of the "
        "supports, the frozen local evidence score of Needle's program, and the structural difference "
        "between the two patterns. It may replace the symbolic winner only when "
        "P(alternative is better) is above a threshold.",
        "",
        "Training rows are disagreements only. A wrong override is weighted ten times a missed correction. "
        "The threshold is swept on held-out prefixes, separators, and widths. It has to keep validation "
        f"accuracy at or above the symbolic score, hold the regression rate to {REGRESSION_BUDGET:.0%}, "
        f"and keep the easy family within {EASY_MARGIN:.1%} of the symbolic score. BENCH-C is scored once.",
        "",
        "## Reading",
        "",
    ]
    for paragraph in report["reading"]:
        lines.append(paragraph)
        lines.append("")
    lines.append("## Protocol")
    lines.append("")
    selected = report["selected"]
    lines.append(
        f"λ={LAMBDA} fixed. Parameters {report['parameters']}. "
        f"Training tasks {report['train_n']}, disagreements {report['train_disagreements']}, "
        f"alternatives that improve the symbolic score {report['train_positives']}. "
        f"Validation tasks {report['val_n']}, disagreements {report['val_disagreements']}. "
        f"Selected epoch {selected['epoch']}, threshold {selected['threshold']}. "
        f"Target band {TARGET_ACCURACY:.1%}. Evidence parameters {report['evidence_parameters']}."
    )
    lines.append("")
    lines.append("## Validation")
    lines.append("")
    lines.append("Held-out formatting variants. Used to choose the epoch and the threshold.")
    lines.append("")
    val = report["validation_systems"]
    lines.extend(_metric_table(val))
    lines.append("")
    lines.append("## BENCH-C")
    lines.append("")
    lines.append("Scored once, after the threshold was frozen. Regressions are drops from the symbolic winner.")
    lines.append("")
    lines.extend(_metric_table(report["bench_systems"]))
    lines.append("")
    lines.append(
        f"Minimum bar (above the symbolic score, regression rate ≤ {REGRESSION_BUDGET:.0%}, "
        f"easy family within {EASY_MARGIN:.1%}): {'yes' if bench['cleared_minimum'] else 'no'}. "
        f"Target band (≥ {TARGET_ACCURACY:.1%} with the same constraints): "
        f"{'yes' if bench['cleared_target'] else 'no'}."
    )
    lines.append("")
    if bench["families"]:
        lines.append("Heuristic-failure families:")
        lines.append("")
        headers = ["Failure", "n", "symbolic", "override", "oracle union"]
        body = []
        for row in bench["families"]:
            body.append(
                [
                    row["failure"],
                    str(row["n"]),
                    _pct(row["symbolic"]),
                    _pct(row["chosen"]),
                    _pct(row["oracle"]),
                ]
            )
        lines.extend(_table(headers, body))
        lines.append("")
    lines.append("## Threshold sweep")
    lines.append("")
    lines.append("Validation only.")
    lines.append("")
    headers = ["Threshold", "Accuracy", "Symbolic", "Fixed", "Regressed", "Net", "Eligible"]
    body = []
    for row in report["grid"]:
        body.append(
            [
                f"{row['threshold']}",
                _pct(row["semantic_accuracy"]),
                _pct(row["symbolic_accuracy"]),
                str(row["fixed"]),
                str(row["regressed"]),
                str(row["net"]),
                "yes" if row["eligible"] else "no",
            ]
        )
    lines.extend(_table(headers, body))
    lines.append("")
    return "\n".join(lines)


def _metric_table(systems: list[dict]) -> list[str]:
    headers = ["System", "Semantic accuracy", "Fixed", "Regressed", "Net", "Overrides"]
    body = []
    for row in systems:
        body.append(
            [
                row["label"],
                _pct(row["semantic_accuracy"]),
                str(row["fixed"]),
                str(row["regressed"]),
                str(row["net"]),
                str(row["overrides"]),
            ]
        )
    return _table(headers, body)


def _load_evidence(root: Path) -> LocalScorer:
    saved = torch.load(root / "results" / "prior" / "runs" / "relative_evidence" / "seed_0" / "checkpoint.pt", weights_only=False)
    model = freeze_evidence(LocalScorer())
    model.load_state_dict(saved["model"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=320)
    parser.add_argument("--val", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repack", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    evidence = _load_evidence(root)
    cache = CACHE
    train_path = cache / "train.jsonl"
    val_path = cache / "val.jsonl"
    if args.repack or not train_path.exists():
        train_rows = _pack_split(generate_split(args.train, "train", seed=2026), evidence, root, "train")
        _write_cache(train_path, train_rows)
    else:
        train_rows = _read_cache(train_path)
        print(f"cache train {len(train_rows)}", flush=True)
    if args.repack or not val_path.exists():
        val_rows = _pack_split(generate_split(args.val, "val", seed=2027), evidence, root, "val")
        _write_cache(val_path, val_rows)
    else:
        val_rows = _read_cache(val_path)
        print(f"cache val {len(val_rows)}", flush=True)
    fitted = train(train_rows, val_rows, args.epochs, args.patience, args.batch_size, args.lr, args.seed)
    threshold = fitted["selected"]["threshold"]
    bench_path = cache / "bench.jsonl"
    if args.repack or not bench_path.exists():
        bench_rows = _pack_bench(root, evidence, read_jsonl(root / "data" / "bench_c.jsonl"))
        _write_cache(bench_path, bench_rows)
    else:
        bench_rows = _read_cache(bench_path)
        print(f"cache bench {len(bench_rows)}", flush=True)
    probabilities = _probabilities(fitted["model"], bench_rows, fitted["mean"], fitted["scale"])
    take = [
        override_choice(probability, threshold, bool(row["disagreed"]))
        for probability, row in zip(probabilities, bench_rows)
    ]
    bench = summarize(bench_rows, take)
    bench["cleared_minimum"] = clears_minimum(bench)
    bench["cleared_target"] = clears_target(bench)
    bench["threshold"] = threshold
    val_probabilities = _probabilities(fitted["model"], val_rows, fitted["mean"], fitted["scale"])
    _val_threshold, val_info = choose_threshold(val_rows, val_probabilities)
    symbolic = _constant(bench_rows, "symbolic_accuracy")
    alternative = _constant(bench_rows, "alt_accuracy")
    oracle = _constant(bench_rows, "oracle")
    val_symbolic = _constant(val_rows, "symbolic_accuracy")
    val_alternative = _constant(val_rows, "alt_accuracy")
    val_oracle = _constant(val_rows, "oracle")
    report = {
        "parameters": fitted["parameters"],
        "evidence_parameters": parameter_count(evidence),
        "lambda": LAMBDA,
        "train_n": fitted["train_n"],
        "train_disagreements": fitted["train_disagreements"],
        "train_positives": fitted["train_positives"],
        "val_n": fitted["val_n"],
        "val_disagreements": fitted["val_disagreements"],
        "selected": fitted["selected"],
        "history": fitted["history"],
        "grid": [{key: value for key, value in row.items() if key != "families"} for row in val_info["grid"]],
        "bench": bench,
        "validation_systems": [
            _system("symbolic", "symbolic selector", val_symbolic),
            _system("alternative", "Needle alternative", val_alternative),
            _system("override", "learned override", val_info["chosen"]),
            _system("oracle", "oracle union", val_oracle),
        ],
        "bench_systems": [
            _system("symbolic", "symbolic selector", symbolic),
            _system("alternative", "Needle alternative", alternative),
            _system("override", "learned override", bench),
            _system("oracle", "oracle union", oracle),
        ],
        "bench_e_scored": False,
        "needle_unchanged": True,
    }
    report["reading"] = _reading(report)
    out = root / "results" / "needle"
    run = out / "runs" / "safe_override" / "seed_0"
    run.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": fitted["model"].state_dict(),
            "mean": fitted["mean"],
            "scale": fitted["scale"],
            "threshold": threshold,
            "features": list(FEATURE_NAMES),
            "lambda": LAMBDA,
        },
        run / "checkpoint.pt",
    )
    payload = json.loads(json.dumps(report))
    (run / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "safe_override.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "safe_override.md").write_text(render_markdown(payload))
    print(
        f"bench {bench['semantic_accuracy']:.4f} symbolic {bench['symbolic_accuracy']:.4f} "
        f"oracle {bench['oracle_accuracy']:.4f} net {bench['net']} "
        f"minimum {bench['cleared_minimum']} target {bench['cleared_target']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
