"""Exact-slice synthesis oracle after aligned facts fill empty charts.

The recognizer, the grammar, and every nonempty chart stay as they were.
Tasks whose frozen chart was empty are assembled again with document
alignment. This pass does not score BENCH-E.
"""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path

from pattern_induction.document_facts import alignment_kind
from pattern_induction.program import MAX_LEN, MAX_SUPPORT, Prog, Task, agrees, execute
from pattern_induction.select_train import load_selector, select_program, synthesize_assembly
from pattern_induction.tasks import read_jsonl


class _Timeout(Exception):
    pass


def _visible(texts: list[str]) -> list[str]:
    return [text[:MAX_LEN] for text in texts[:MAX_SUPPORT]]


def _probe(bench) -> list[tuple[str, bool]]:
    return [(text, True) for text in bench.hidden_positives] + [(text, False) for text in bench.hidden_negatives]


def _accuracy(prog, visible: list[str], bench) -> float:
    positives = bench.hidden_positives
    negatives = bench.hidden_negatives
    if prog is None:
        return 0.5
    recall = sum(bool(execute(prog, text, visible)) for text in positives) / max(len(positives), 1)
    rejection = sum(not execute(prog, text, visible) for text in negatives) / max(len(negatives), 1)
    return 0.5 * (recall + rejection)


def _alarm(signum, frame) -> None:
    raise _Timeout()


def evaluate(root: Path, limit_s: float = 90.0) -> dict:
    scope, witness, prior = load_selector(root)
    saved = json.loads((root / "results" / "needle" / "bench_split.json").read_text())
    by_id = {row["id"]: row for row in saved["rows"]}
    benches = read_jsonl(root / "data" / "bench_c.jsonl")
    signal.signal(signal.SIGALRM, _alarm)
    updated: dict[str, dict] = {}
    started = time.perf_counter()
    # Only the exact tasks whose chart was empty. Other tasks keep the frozen score.
    empty_rows = [
        bench
        for bench in benches
        if by_id[bench.id]["dsl"] == "exact" and by_id[bench.id].get("languages", 0) == 0
    ]
    for index, bench in enumerate(empty_rows):
        visible = _visible(bench.visible)
        previous = by_id[bench.id]
        kind = alignment_kind(visible)
        task = Task(bench.domain, visible, Prog("COPY"), [])
        programs: list = []
        perfect = False
        count = 0
        accuracy = 0.5
        timed_out = False
        signal.alarm(int(limit_s))
        try:
            assembly = synthesize_assembly(scope, task)
            programs = [prog for prog, _score in assembly.programs]
            perfect = bool(programs) and any(agrees(prog, visible, _probe(bench)) for prog in programs)
            chosen, _again, count = select_program(scope, witness, prior, task, with_stability=True)
            accuracy = _accuracy(chosen, visible, bench)
        except _Timeout:
            timed_out = True
        finally:
            signal.alarm(0)
        updated[bench.id] = {
            "id": bench.id,
            "domain": bench.domain,
            "dsl": previous["dsl"],
            "alignment": kind,
            "languages": count,
            "programs": len(programs),
            "synthesis_perfect": perfect,
            "selector_accuracy": accuracy,
            "timed_out": timed_out,
        }
        if index % 10 == 0 or index + 1 == len(empty_rows):
            exact_done = [row for row in updated.values() if row["dsl"] == "exact"]
            filled = sum(1 for row in exact_done if row["programs"])
            print(
                f"transfer {index + 1}/{len(empty_rows)} {bench.domain} {kind} "
                f"programs {len(programs)} perfect {int(perfect)} "
                f"exact-filled {filled}/{len(exact_done)} ({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
    exact = [by_id[bench.id] for bench in benches if by_id[bench.id]["dsl"] == "exact"]
    perfect = 0
    empty_after = 0
    accuracies = []
    by_kind: dict[str, list[dict]] = {}
    for row in exact:
        fresh = updated.get(row["id"])
        if fresh is None:
            perfect += int(bool(row.get("synthesis_perfect")))
            accuracies.append(row["selector_accuracy"])
            continue
        perfect += int(bool(fresh["synthesis_perfect"]))
        empty_after += int(fresh["programs"] == 0)
        accuracies.append(fresh["selector_accuracy"])
        by_kind.setdefault(fresh["alignment"], []).append(fresh)
    headline = []
    for bench in benches:
        fresh = updated.get(bench.id)
        if fresh is None:
            headline.append(by_id[bench.id]["selector_accuracy"])
        else:
            headline.append(fresh["selector_accuracy"])
    kind_summary = {}
    for name, rows in sorted(by_kind.items()):
        kind_summary[name] = {
            "n": len(rows),
            "filled": sum(1 for row in rows if row["programs"]),
            "synthesis_perfect": sum(1 for row in rows if row["synthesis_perfect"]),
            "selector_accuracy": sum(row["selector_accuracy"] for row in rows) / len(rows),
        }
    payload = {
        "exact_n": len(exact),
        "exact_empty_before": sum(1 for row in exact if row.get("languages", 0) == 0),
        "exact_empty_after": empty_after,
        "exact_empty_rate": empty_after / len(exact),
        "exact_synthesis_oracle": perfect / len(exact),
        "exact_selector_accuracy": sum(accuracies) / len(accuracies),
        "headline_selector_accuracy": sum(headline) / len(headline),
        "rescored_empty": len(updated),
        "timeouts": sum(1 for row in updated.values() if row["timed_out"]),
        "by_alignment": kind_summary,
        "rows": list(updated.values()),
        "seconds": time.perf_counter() - started,
        "dsl_unchanged": True,
        "model_unchanged": True,
        "nonempty_charts_unchanged": True,
    }
    return payload
