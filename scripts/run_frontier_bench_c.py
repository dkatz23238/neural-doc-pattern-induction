"""Score two frontier models on BENCH-C and write Gate 1."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from pattern_induction.evaluate import aggregate, score_regex
from pattern_induction.frontier import (
    GATE1_MODELS,
    TEMPERATURE_NOTE,
    call_frontier,
    prompts_for,
)
from pattern_induction.tasks import read_jsonl
from pattern_induction.taxonomy import categorize, summarize

HEURISTIC_SEMANTIC = 0.9316


def score_one(model: str, task, prompt_name: str) -> dict:
    prompt = prompts_for(task.visible)[prompt_name]
    started = time.perf_counter()
    error = ""
    pattern = ""
    raw = ""
    try:
        pattern, raw, _usage = call_frontier(model, prompt)
    except Exception as exc:  # noqa: BLE001 — API failures are rows
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - started) * 1000
    score = score_regex(pattern, task) if pattern else score_regex("(?!)", task)
    if not pattern:
        score.valid = False
        score.semantic_accuracy = 0.0
        score.positive_recall = 0.0
        score.negative_rejection = 0.0
        score.matches_visible = False
    row = {
        "id": task.id,
        "benchmark": task.benchmark,
        "domain": task.domain,
        "variant": task.variant,
        "system": model,
        "prompt": prompt_name,
        "model": model,
        "temperature": None,
        "temperature_note": TEMPERATURE_NOTE,
        "date": date.today().isoformat(),
        "regex": pattern,
        "raw": raw,
        "gold_regex": task.gold_regex,
        "latency_ms": latency_ms,
        "valid": 1.0 if score.valid and pattern else 0.0,
        "matches_visible": 1.0 if score.matches_visible else 0.0,
        "positive_recall": score.positive_recall,
        "negative_rejection": score.negative_rejection,
        "semantic_accuracy": score.semantic_accuracy,
        "catastrophic": 1.0 if score.catastrophic else 0.0,
        "char_length": score.complexity["char_length"],
        "alternations": score.complexity["alternations"],
        "capturing_groups": score.complexity["capturing_groups"],
        "nested_quantifiers": score.complexity["nested_quantifiers"],
        "ast_node_count": score.complexity["ast_node_count"],
        "error": error or score.error,
        "timed_out": False,
        "n_visible": task.n_visible,
    }
    row["failure"] = categorize(row)
    return row


def load_done(path: Path) -> dict[str, dict]:
    done = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            done[row["id"]] = row
    return done


def run_model(model: str, tasks, out_path: Path, prompt_name: str, workers: int) -> list[dict]:
    done = load_done(out_path)
    pending = [task for task in tasks if task.id not in done]
    print(f"{model}: {len(done)} done, {len(pending)} pending", flush=True)
    if pending:
        with out_path.open("a", encoding="utf-8") as fh:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(score_one, model, task, prompt_name): task.id for task in pending
                }
                finished = 0
                for future in as_completed(futures):
                    row = future.result()
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    finished += 1
                    if finished % 25 == 0 or finished == len(pending):
                        print(f"{model}: {finished}/{len(pending)}", flush=True)
    rows = [load_done(out_path)[task.id] for task in tasks]
    return rows


def gate_text(summaries: list[dict]) -> str:
    lines = []
    gaps = []
    for summary in summaries:
        gap = summary["semantic_accuracy"] - HEURISTIC_SEMANTIC
        gaps.append(gap)
        lines.append(
            f"{summary['system']} semantic accuracy {summary['semantic_accuracy']:.1%} "
            f"(macro {summary['macro_semantic_accuracy']:.1%}), "
            f"gap vs heuristic {gap:+.1%}."
        )
    if gaps and min(gaps) >= 0.08:
        decision = (
            "Gate 1 is a go: both models clear the heuristic by at least 8 points. "
            "The synthetic training corpus is the next step."
        )
    else:
        decision = (
            "Gate 1 is a no. The models do not clear the heuristic by 8–10 points. "
            "The spec says not to train a neural model."
        )
    return " ".join(lines) + " " + decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/bench_c.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/exp1"))
    parser.add_argument("--prompt", default="simple")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--models", default=",".join(GATE1_MODELS))
    args = parser.parse_args()
    tasks = read_jsonl(args.data)
    if args.limit:
        tasks = tasks[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    models = [part for part in args.models.split(",") if part]
    summaries = []
    for model in models:
        slug = model.replace(".", "_")
        path = args.out / f"bench_c_{slug}.jsonl"
        rows = run_model(model, tasks, path, args.prompt, args.workers)
        summary = aggregate(model, rows).as_dict()
        summary["prompt"] = args.prompt
        summary["n_tasks"] = len(tasks)
        summary["date"] = date.today().isoformat()
        summary["temperature"] = None
        summary["temperature_note"] = TEMPERATURE_NOTE
        summary["taxonomy"] = summarize(rows)
        summaries.append(summary)
        print(
            f"{model} As={summary['semantic_accuracy']:.3f} "
            f"macro={summary['macro_semantic_accuracy']:.3f} "
            f"R+={summary['positive_recall']:.3f} R-={summary['negative_rejection']:.3f}",
            flush=True,
        )
    decision = gate_text(summaries)
    payload = {
        "prompt": args.prompt,
        "date": date.today().isoformat(),
        "n_tasks": len(tasks),
        "heuristic_semantic_accuracy": HEURISTIC_SEMANTIC,
        "temperature_note": TEMPERATURE_NOTE,
        "models_requested": ["gpt-6-luna", "gpt-6-terra"],
        "models_called": models,
        "model_note": "gpt-6-terra returns model_not_found. Terra is gpt-5.6-terra.",
        "systems": summaries,
        "gate1": decision,
    }
    (args.out / "frontier_bench_c.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(decision, flush=True)


if __name__ == "__main__":
    main()
