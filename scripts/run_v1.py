"""Frozen v1 evaluation.

Scores the code-length selector. It does not train, and it does not load the
residual override. BENCH-C is the development record. BENCH-E is scored only
with --bench-e, once, after that selector is frozen.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pattern_induction.candidate_oracle import _rates
from pattern_induction.needle import needle
from pattern_induction.symbolic import symbolic_pick
from pattern_induction.tasks import read_jsonl

SHOWCASE = (
    ("invoice", ["INV-2024-10021", "INV-2026-88310", "INV-2025-00418"]),
    ("version", ["1.2.0", "1.4.3", "2.0.1"]),
    ("section", ["1.2", "1.10", "3.4"]),
    ("filename", ["report.pdf", "notes.txt", "budget.csv"]),
    ("purchase order", ["PO-1000", "PO-1042", "PO-1108"]),
    ("customer", ["CUS-DE1234", "CUS-SO9281", "CUS-AM0042"]),
)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _clip(texts: list[str]) -> list[str]:
    return [text[:32] for text in texts if text][:4]


def score_tasks(tasks, product: bool) -> dict:
    accuracies = []
    started = time.perf_counter()
    for index, task in enumerate(tasks, start=1):
        visible = _clip(task.visible) if product else list(task.visible)
        if len(visible) < 2:
            accuracies.append(0.0)
        else:
            accuracies.append(_rates(symbolic_pick(visible).pattern, task)[0])
        if index % 50 == 0 or index == len(tasks):
            print(f"score {index}/{len(tasks)} in {time.perf_counter() - started:.1f}s", flush=True)
    return {
        "n": len(accuracies),
        "semantic_accuracy": _mean(accuracies),
        "seconds": time.perf_counter() - started,
    }


def _stored_rules(root: Path) -> float:
    ids = {task.id for task in read_jsonl(root / "data" / "bench_c.jsonl")}
    scores = []
    with (root / "results" / "exp2" / "candidate_oracle_tasks.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in ids:
                scores.append(row["rules"]["accuracy"])
    return _mean(scores)


def _rss_kb() -> int:
    status = Path("/proc/self/status").read_text()
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return 0


def showcase(root: Path) -> list[dict]:
    rows = []
    for name, examples in SHOWCASE:
        decision_started = time.perf_counter()
        symbolic_pick(examples)
        decision_seconds = time.perf_counter() - decision_started
        started = time.perf_counter()
        result = needle(examples, root)
        rows.append(
            {
                "name": name,
                "examples": list(examples),
                "regex": result.regex,
                "confidence": round(result.confidence, 4),
                "ambiguous": result.ambiguous,
                "alternatives": result.alternatives,
                "method": result.method,
                "has_ast": result.ast is not None,
                "decision_seconds": decision_seconds,
                "seconds": time.perf_counter() - started,
            }
        )
        print(f"showcase {name} {result.regex} {time.perf_counter() - started:.2f}s", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-e", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    stored = _stored_rules(root)
    bench_c = score_tasks(read_jsonl(root / "data" / "bench_c.jsonl"), product=False)
    product = score_tasks(read_jsonl(root / "data" / "bench_c.jsonl"), product=True)
    rows = showcase(root)
    payload = {
        "selector": "code-length with the digit-literal charge, eight presented candidates",
        "bench_c_stored": stored,
        "bench_c_visible": bench_c,
        "bench_c_product_clip": product,
        "showcase": rows,
        "rss_kb": _rss_kb(),
        "override_in_inference": False,
    }
    if args.bench_e:
        payload["bench_e"] = score_tasks(read_jsonl(root / "data" / "bench_e.jsonl"), product=False)
        payload["bench_e_product_clip"] = score_tasks(read_jsonl(root / "data" / "bench_e.jsonl"), product=True)
    out = root / "results" / "needle" / "v1.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in payload if key != "showcase"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
