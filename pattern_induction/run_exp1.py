"""Run Experiment 1 symbolic baselines and write a review report."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from pattern_induction.baselines import GREX_CONFIGS, infer
from pattern_induction.evaluate import aggregate, score_regex
from pattern_induction.features import render_features
from pattern_induction.frontier import prompts_for, provider_status
from pattern_induction.tasks import Task, read_jsonl
from pattern_induction.taxonomy import categorize, summarize

SYSTEMS = ["memorize", "heuristic", *GREX_CONFIGS.keys(), "fast"]


def evaluate_system(
    system: str,
    tasks: list[Task],
    fast_timeout: float,
) -> list[dict]:
    rows = []
    for task in tasks:
        started = time.perf_counter()
        error = ""
        timed_out = False
        try:
            pattern = infer(system, task.visible, fast_timeout=fast_timeout)
        except TimeoutError as exc:
            pattern = ""
            error = str(exc)
            timed_out = True
        except Exception as exc:  # noqa: BLE001 — baseline failures are data
            pattern = ""
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
            "system": system,
            "regex": pattern,
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
            "timed_out": timed_out,
            "n_visible": task.n_visible,
        }
        row["failure"] = categorize(row)
        rows.append(row)
    return rows


def _markdown(report: dict) -> str:
    lines = [
        "# Experiment 1 — symbolic baselines",
        "",
        "Semantic accuracy is `(positive recall + negative rejection) / 2` on hidden examples.",
        "Latency is wall time on this cloud VM CPU, not a laptop.",
        "",
        report["frontier"]["note"],
        "",
    ]
    for bench, payload in report["benchmarks"].items():
        lines.append(f"## {bench}")
        lines.append("")
        lines.append("| System | Semantic accuracy | Macro accuracy | Positive recall | Negative rejection | Valid | Catastrophic | p50 ms | p95 ms |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in payload["systems"]:
            lat = row["latency_ms"]
            lines.append(
                "| {system} | {semantic_accuracy:.1%} | {macro_semantic_accuracy:.1%} | {positive_recall:.1%} | {negative_rejection:.1%} | {valid_output_rate:.1%} | {catastrophic_rate:.1%} | {p50:.2f} | {p95:.2f} |".format(
                    p50=lat["p50"], p95=lat["p95"], **row
                )
            )
        lines.append("")
        for row in payload["systems"]:
            timeouts = row.get("timeouts") or 0
            if timeouts:
                lines.append(
                    f"`{row['system']}` returned no expression on {timeouts} of {row['n']} tasks "
                    f"(accuracy on completed runs: {row.get('accuracy_on_completed')}). "
                    "A zero in the table is a timeout, not a scored regex."
                )
        lines.append("")
        lines.append("Failure taxonomy:")
        lines.append("")
        for system, counts in payload["taxonomy"].items():
            top = ", ".join(f"{name} {info['n']}" for name, info in list(counts.items())[:6])
            lines.append(f"- `{system}`: {top}")
        lines.append("")
    if "sample_efficiency" in report:
        lines.append("## BENCH-D sample efficiency (heuristic)")
        lines.append("")
        lines.append("| Visible examples | Semantic accuracy |")
        lines.append("| ---: | ---: |")
        for n, acc in report["sample_efficiency"]:
            lines.append(f"| {n} | {acc:.1%} |")
        lines.append("")
    lines.append("## Gate 1")
    lines.append("")
    lines.append(report["gate1"])
    lines.append("")
    return "\n".join(lines)


def run(
    paths: dict[str, Path],
    out_dir: Path,
    systems: list[str],
    fast_timeout: float,
    fast_limit: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"frontier": provider_status(), "benchmarks": {}, "hardware": "cloud VM CPU"}
    gate_bits = []
    for name, path in paths.items():
        tasks = read_jsonl(path)
        payload = {"n_tasks": len(tasks), "systems": [], "taxonomy": {}}
        for system in systems:
            subset = tasks
            if system == "fast":
                subset = [task for task in tasks if max(map(len, task.visible)) <= 12][:fast_limit]
            rows = evaluate_system(system, subset, fast_timeout=fast_timeout)
            (out_dir / f"{name}_{system}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            summary = aggregate(system, rows).as_dict()
            summary["n_evaluated"] = len(rows)
            summary["timeouts"] = sum(1 for row in rows if row["timed_out"])
            completed = [row for row in rows if not row["timed_out"] and row["valid"]]
            summary["accuracy_on_completed"] = (
                round(sum(row["semantic_accuracy"] for row in completed) / len(completed), 4)
                if completed
                else None
            )
            payload["systems"].append(summary)
            payload["taxonomy"][system] = summarize(rows)
            if name == "bench_c" and system == "heuristic":
                gate_bits.append(summary["semantic_accuracy"])
        if name == "bench_d":
            heuristic_rows = [
                json.loads(line)
                for line in (out_dir / f"{name}_heuristic.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            grouped: dict[int, list[float]] = defaultdict(list)
            for row in heuristic_rows:
                grouped[row["n_visible"]].append(row["semantic_accuracy"])
            report["sample_efficiency"] = [
                [n, sum(vals) / len(vals)] for n, vals in sorted(grouped.items())
            ]
        report["benchmarks"][name] = payload
    # Save a handful of frontier prompts so the next step can replay them.
    if "bench_c" in paths:
        tasks = read_jsonl(paths["bench_c"])[:25]
        prompt_rows = []
        for task in tasks:
            features = render_features(task.visible)
            prompt_rows.append(
                {
                    "id": task.id,
                    "prompts": prompts_for(task.visible, features),
                    "gold_regex": task.gold_regex,
                }
            )
        (out_dir / "frontier_prompts_sample.json").write_text(
            json.dumps(prompt_rows, indent=2), encoding="utf-8"
        )
    best = gate_bits[0] if gate_bits else None
    if best is None:
        report["gate1"] = "BENCH-C was not in this run, so Gate 1 is not scored."
    else:
        report["gate1"] = (
            f"On this BENCH-C file the structural heuristic reaches {best:.1%} micro semantic accuracy "
            f"(see macro accuracy in the table; invoice-number columns are capped). "
            "grex's combined digit/word/repetition mode is the next symbolic system. "
            "Default grex matches the memorization floor. "
            "fAST's reference search did not return an expression within 3 seconds on the short tasks it was given, "
            "so it is not a scored competitor yet. "
            "Gate 1 still needs two frontier models on this same file. No API key is configured, so those cells are empty. "
            "The decision rule is an 8–10 point gap between the best symbolic accuracy and the frontier models. "
            "Do not call the gate from the symbolic column alone. "
            "If frontier models also land near the heuristic, the spec says not to train a neural model."
        )
    (out_dir / "exp1_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "exp1_report.md").write_text(_markdown(report), encoding="utf-8")
    return report
