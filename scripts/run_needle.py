"""Rescore the frozen selector, then stress it, then score BENCH-C if it holds.

The witness head is not retrained. BENCH-C is not a training or checkpoint signal.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from pattern_induction.local_train import ALL, ATOMIC, OOD
from pattern_induction.program import MAX_LEN, MAX_SUPPORT, Prog, Task, execute, sample_compose_task, to_regex
from pattern_induction.scope import total_parameter_count
from pattern_induction.select_train import (
    _interpretation,
    _invariance,
    _score_split,
    load_selector,
    select_program,
)
from pattern_induction.bench_split import aggregate as split_aggregate
from pattern_induction.bench_split import fill_synthesis, label_tasks, render_markdown
from pattern_induction.stress import aggregate, score_rows, stress_cells
from pattern_induction.tasks import read_jsonl
from pattern_induction.transfer_eval import evaluate as evaluate_transfer


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}", flush=True)


def _gate(root: Path) -> dict:
    scope, witness, prior = load_selector(root)
    started = time.time()
    final = {}
    for index, family in enumerate(ALL):
        tasks = [sample_compose_task(family, random.Random(5000 + index * 50 + draw), 3) for draw in range(16)]
        final[family] = _score_split(scope, witness, prior, tasks, with_stability=True)
        print(
            f"gate {family}: top1 {final[family]['top1']:.3f} semantic {final[family]['semantic']:.3f} "
            f"oracle {final[family]['oracle']:.3f} languages {final[family]['families']:.1f}",
            flush=True,
        )
    inv_tasks = [sample_compose_task("repeat", random.Random(90 + index), 1) for index in range(4)]
    invariance = _invariance(scope, witness, prior, inv_tasks)
    ood_top1 = sum(final[family]["top1"] for family in OOD) / len(OOD)
    ood_semantic = sum(final[family]["semantic"] for family in OOD) / len(OOD)
    ood_oracle = sum(final[family]["oracle"] for family in OOD) / len(OOD)
    worst_family = min(OOD, key=lambda family: final[family]["top1"])
    worst = final[worst_family]["top1"]
    atomic_top1 = sum(final[family]["top1"] for family in ATOMIC) / len(ATOMIC)
    metrics = {
        "arch": "needle-cpu",
        "contingent_optional": True,
        "retrained": False,
        "parameters": int(total_parameter_count(scope) + sum(param.numel() for param in witness.parameters())),
        "witness_parameters": int(sum(param.numel() for param in witness.parameters())),
        "atomic_top1": atomic_top1,
        "ood_top1": ood_top1,
        "ood_semantic": ood_semantic,
        "ood_oracle": ood_oracle,
        "worst_family": worst_family,
        "worst_top1": worst,
        "min_invariance": invariance,
        "interpretation": _interpretation(ood_oracle, ood_top1, worst, invariance),
        "families": final,
        "seconds": time.time() - started,
    }
    _write(root / "results" / "needle" / "gate.json", metrics)
    print(
        f"gate top1 {ood_top1:.3f} oracle {ood_oracle:.3f} worst {worst_family} {worst:.3f} "
        f"invariance {invariance:.3f} [{metrics['interpretation']}]",
        flush=True,
    )
    return metrics


def _stress(root: Path) -> dict:
    scope, witness, prior = load_selector(root)
    started = time.time()
    cells = []
    core_rows = []
    for name, role, tasks in stress_cells():
        print(f"stress {name} n={len(tasks)}", flush=True)
        rows = score_rows(scope, witness, prior, tasks, with_stability=True)
        summary = aggregate(rows)
        by_family: dict[str, list] = defaultdict(list)
        for row in rows:
            by_family[row["family"]].append(row)
        summary["role"] = role
        summary["name"] = name
        summary["by_family"] = {family: aggregate(group) for family, group in by_family.items()}
        cells.append(summary)
        print(
            f"stress {name}: top1 {summary['top1']:.3f} oracle {summary['oracle']:.3f} "
            f"languages {summary['families']:.1f} max {summary['max_families']:.0f}",
            flush=True,
        )
        if role == "core":
            core_rows.extend(rows)
    core = aggregate(core_rows)
    core_cells = [row for row in cells if row["role"] == "core"]
    worst = min(core_cells, key=lambda row: row["top1"])
    survives = core["n"] > 0 and core["top1"] >= 0.97 and core["oracle"] >= 0.97 and worst["top1"] >= 0.875
    payload = {
        "draws": 8,
        "stability": True,
        "survives": survives,
        "core": core,
        "worst_core": worst["name"] if worst["role"] == "core" else "",
        "worst_core_top1": worst["top1"] if worst["role"] == "core" else 0.0,
        "cells": cells,
        "seconds": time.time() - started,
    }
    _write(root / "results" / "needle" / "stress.json", payload)
    print(
        f"stress core top1 {core['top1']:.3f} oracle {core['oracle']:.3f} "
        f"worst {payload['worst_core']} {payload['worst_core_top1']:.3f} survives {survives}",
        flush=True,
    )
    return payload


def _bench(root: Path, force: bool) -> dict | None:
    stress_path = root / "results" / "needle" / "stress.json"
    if not stress_path.exists():
        raise RuntimeError("stress has not been run")
    stress = json.loads(stress_path.read_text())
    if not stress.get("survives") and not force:
        print("stress stayed off the 99 percent line, so BENCH-C is not scored", flush=True)
        return None
    scope, witness, prior = load_selector(root)
    tasks = read_jsonl(root / "data" / "bench_c.jsonl")
    started = time.time()
    rows = []
    running = 0.0
    long_visible = 0
    for index, bench in enumerate(tasks):
        visible = [text[:MAX_LEN] for text in bench.visible[:MAX_SUPPORT]]
        if any(len(text) > MAX_LEN for text in bench.visible[:MAX_SUPPORT]):
            long_visible += 1
        dummy = Prog("COPY", slot=0, start=0, end=min(1, len(visible[0]))) if visible and visible[0] else Prog("COPY")
        chosen, _programs, count = select_program(
            scope, witness, prior, Task(bench.domain, visible, dummy, []), with_stability=True
        )
        positives = bench.hidden_positives
        negatives = bench.hidden_negatives
        if chosen is None:
            recall = 0.0
            rejection = 1.0
            rendered = ""
        else:
            recall = sum(bool(execute(chosen, text, visible)) for text in positives) / max(len(positives), 1)
            rejection = sum(not execute(chosen, text, visible) for text in negatives) / max(len(negatives), 1)
            rendered = to_regex(chosen, visible)
        accuracy = 0.5 * (recall + rejection)
        running += accuracy
        rows.append(
            {
                "id": bench.id,
                "domain": bench.domain,
                "accuracy": accuracy,
                "recall": recall,
                "rejection": rejection,
                "languages": count,
                "regex": rendered,
                "gold_regex": bench.gold_regex,
            }
        )
        if index % 25 == 0 or index + 1 == len(tasks):
            print(f"bench {index + 1}/{len(tasks)} acc {running / (index + 1):.3f} ({time.time() - started:.0f}s)", flush=True)
    by_domain: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row["accuracy"])
    semantic = sum(row["accuracy"] for row in rows) / max(len(rows), 1)
    macro = sum(sum(values) / len(values) for values in by_domain.values()) / max(len(by_domain), 1)
    abstained = sum(1 for row in rows if not row["regex"])
    misses = sorted(rows, key=lambda row: row["accuracy"])[:12]
    if semantic >= 0.98:
        decision = "GPU work is a performance and robustness Pareto study"
    elif semantic < 0.942:
        decision = "external generalization is the next research target"
    else:
        decision = "BENCH-C sits between the old symbolic rule and the synthetic result"
    payload = {
        "n": len(rows),
        "semantic_accuracy": semantic,
        "macro_semantic_accuracy": macro,
        "abstained": abstained,
        "long_visible": long_visible,
        "old_rule": 0.962,
        "candidate_oracle_16": 0.995,
        "decision": decision,
        "domains": {name: sum(values) / len(values) for name, values in sorted(by_domain.items())},
        "lowest": misses,
        "rows": rows,
        "seconds": time.time() - started,
    }
    _write(root / "results" / "needle" / "bench_c.json", payload)
    print(f"bench semantic {semantic:.3f} macro {macro:.3f} [{decision}]", flush=True)
    return payload


def _split(root: Path) -> dict:
    scored_path = root / "results" / "needle" / "bench_c.json"
    if not scored_path.exists():
        raise RuntimeError("BENCH-C has not been scored")
    scored = json.loads(scored_path.read_text())
    saved_rows = scored.get("rows") or []
    by_id = {row["id"]: row for row in saved_rows}
    benchmarks = read_jsonl(root / "data" / "bench_c.jsonl")
    saved_selector = len(by_id) == len(benchmarks)
    if not saved_selector:
        by_id = {bench.id: {"accuracy": 0.0} for bench in benchmarks}
    scope, witness, prior = load_selector(root)
    rows = label_tasks(benchmarks, by_id)
    fill_synthesis(scope, witness, prior, benchmarks, rows, saved_selector)
    summary = split_aggregate(rows)
    if scored.get("semantic_accuracy") is not None:
        summary["official_semantic_accuracy"] = scored["semantic_accuracy"]
    payload = {"summary": summary, "rows": rows}
    _write(root / "results" / "needle" / "bench_split.json", payload)
    text = render_markdown(summary)
    dest = root / "results" / "needle" / "bench_split.md"
    dest.write_text(text)
    print(f"wrote {dest}", flush=True)
    print(text, flush=True)
    return payload


def _summary(root: Path) -> None:
    dest = root / "results" / "needle" / "summary.md"
    lines = [
        "# Needle CPU",
        "",
        "The scope recognizer, the version-space chart, and the witness head stay frozen. A contingent delimiter, a punctuation mark that some supports omit, adds an optional suffix when it is the last separator. Shared separators do not fire. The witness head is not retrained.",
        "",
    ]
    gate_path = root / "results" / "needle" / "gate.json"
    if gate_path.exists():
        row = json.loads(gate_path.read_text())
        lines.extend(
            [
                f"Parameters: {row['parameters']} ({row['witness_parameters']} in the witness head).",
                "",
                "| Family | Semantic top-1 | Semantic accuracy | Version-space oracle | Languages |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for family in ALL:
            view = row["families"][family]
            lines.append(
                f"| {family} | {view['top1']:.3f} | {view['semantic']:.3f} | {view['oracle']:.3f} | {view['families']:.1f} |"
            )
        lines.extend(
            [
                "",
                f"Atomic semantic top-1 {row['atomic_top1']:.3f}. Held-out semantic top-1 {row['ood_top1']:.3f}. "
                f"Held-out semantic accuracy {row['ood_semantic']:.3f}. Held-out version-space oracle {row['ood_oracle']:.3f}. "
                f"Worst held-out family {row['worst_family']} {row['worst_top1']:.3f}. "
                f"Renaming invariance {row['min_invariance']:.3f}.",
                "",
                row["interpretation"] + ".",
                "",
            ]
        )
    stress_path = root / "results" / "needle" / "stress.json"
    if stress_path.exists():
        stress = json.loads(stress_path.read_text())
        lines.extend(
            [
                "## Stress",
                "",
                "Eight draws per generator setting. Stability stays on. Core cells are two supports, probes at repeat depth 6–7, a within-class rename onto a held-out mark, and width-2 supports whose language still allows a width-1 piece. A lower-case prefix and a prefix-plus-repeat-plus-optional tail are reported beside those cells.",
                "",
                "| Cell | Role | Semantic top-1 | Oracle | Languages | Max languages |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for cell in stress["cells"]:
            lines.append(
                f"| {cell['name']} | {cell['role']} | {cell['top1']:.3f} | {cell['oracle']:.3f} | {cell['families']:.1f} | {cell['max_families']:.0f} |"
            )
            for family, view in cell.get("by_family", {}).items():
                if len(cell.get("by_family", {})) > 1:
                    lines.append(
                        f"| {cell['name']} / {family} | {cell['role']} | {view['top1']:.3f} | {view['oracle']:.3f} | {view['families']:.1f} | {view['max_families']:.0f} |"
                    )
        nested = next((cell for cell in stress["cells"] if cell["name"] == "nested"), None)
        nested_note = ""
        if nested is not None and nested["top1"] < 1.0:
            nested_note = (
                " On the nested stack, top-1 equals the oracle. The misses are the draws whose optional hyphen is also the repeat separator: that character is in every support, so the contingent rule stays quiet and the frozen top-8 does not name the tail."
            )
        lines.extend(
            [
                "",
                f"Core semantic top-1 {stress['core']['top1']:.3f}. Core oracle {stress['core']['oracle']:.3f}. "
                f"Worst core cell {stress['worst_core']} {stress['worst_core_top1']:.3f}.{nested_note}",
                "",
            ]
        )
    bench_path = root / "results" / "needle" / "bench_c.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text())
        lines.extend(
            [
                "## BENCH-C",
                "",
                f"Scored once on {bench['n']} tasks, using the first four visible strings and at most 32 characters of each. "
                f"Semantic accuracy {bench['semantic_accuracy']:.3f}. Macro {bench['macro_semantic_accuracy']:.3f}. "
                f"The old code-length rule on eight candidates scored {bench['old_rule']:.3f}. "
                f"The structural pool oracle at 16 scored {bench['candidate_oracle_16']:.3f}.",
                "",
            ]
        )
        if (root / "results" / "needle" / "bench_split.md").exists():
            lines.append("The split below separates grammar coverage, synthesis, and selection.")
        else:
            lines.append(bench["decision"] + ".")
        lines.append("")
    elif stress_path.exists() and not json.loads(stress_path.read_text()).get("survives"):
        lines.extend(["## BENCH-C", "", "The stress suite left the synthetic line, so BENCH-C stays unscored.", ""])
    split_path = root / "results" / "needle" / "bench_split.md"
    if split_path.exists():
        lines.append(split_path.read_text().rstrip())
        lines.append("")
    transfer_path = root / "results" / "needle" / "transfer.md"
    if transfer_path.exists():
        lines.append(transfer_path.read_text().rstrip())
        lines.append("")
    seal_path = root / "results" / "needle" / "bench_e_seal.json"
    if seal_path.exists():
        seal = json.loads(seal_path.read_text())
        lines.extend(
            [
                "## BENCH-E",
                "",
                f"Sealed external split, {seal['n']} tasks, sha256 `{seal['sha256']}`. "
                "Scored: no. Inspected: no. BENCH-C remains the diagnostic set.",
                "",
            ]
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n")
    print(f"wrote {dest}", flush=True)


def _transfer(root: Path) -> dict:
    payload = evaluate_transfer(root)
    dest = root / "results" / "needle" / "transfer.json"
    _write(dest, payload)
    lines = [
        "## In-fragment transfer",
        "",
        "The model and the grammar stay frozen. A nonempty chart is not rebuilt. "
        "An empty chart gains facts from an alignment of class runs: a shared text is a copy, "
        "a varying run is that class at the observed widths, a fixed run sequence that some supports omit is optional, "
        "and two unrelated skeletons are an alternation. Class width is not held to the repeat cap.",
        "",
        f"Exact tasks {payload['exact_n']}. Empty charts before {payload['exact_empty_before']}, "
        f"after {payload['exact_empty_after']} ({payload['exact_empty_rate']:.3f}). "
        f"Exact-slice synthesis oracle {payload['exact_synthesis_oracle']:.3f}. "
        f"Exact-slice selector accuracy {payload['exact_selector_accuracy']:.3f}. "
        f"Headline selector accuracy, with previously empty tasks rescored and every other task left as scored, "
        f"{payload['headline_selector_accuracy']:.3f}.",
        "",
        f"Rescored {payload['rescored_empty']} previously empty exact tasks in {payload['seconds']:.0f}s. "
        f"Timeouts {payload['timeouts']}. A timeout still keeps a program found before the clock; "
        f"the selector is then counted as an abstention.",
        "",
        "| Alignment | Empty exact tasks | Filled | Hidden-set perfect | Selector accuracy |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, view in payload["by_alignment"].items():
        lines.append(
            f"| {name} | {view['n']} | {view['filled']} | {view['synthesis_perfect']} | {view['selector_accuracy']:.3f} |"
        )
    lines.extend(["",])
    path = root / "results" / "needle" / "transfer.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"transfer exact oracle {payload['exact_synthesis_oracle']:.3f} "
        f"empty {payload['exact_empty_after']}/{payload['exact_n']} "
        f"headline {payload['headline_selector_accuracy']:.3f}",
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("gate", "stress", "bench", "split", "transfer", "summary"), required=True)
    parser.add_argument("--force-bench", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = _root()
    if args.stage == "gate":
        _gate(root)
    elif args.stage == "stress":
        _stress(root)
    elif args.stage == "bench":
        _bench(root, args.force_bench)
    elif args.stage == "split":
        _split(root)
    elif args.stage == "transfer":
        _transfer(root)
    _summary(root)


if __name__ == "__main__":
    main()
