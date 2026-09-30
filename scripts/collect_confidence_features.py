"""Record the signals behind needle()'s confidence on every BENCH-C task.

Each row keeps the code-length winner, its score on the hidden strings, the
selector's margin and alternatives, and what the recognizer returned without a
time limit, with the seconds it took. `fit_confidence.py` applies the time
limit afterwards. BENCH-E is not read.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "needle" / "confidence_features.jsonl"
CEILING = 90.0


def _row(task_obj: dict) -> dict:
    from pattern_induction.candidate_oracle import _rates
    from pattern_induction.candidates import semantic_key
    from pattern_induction.needle import _learned
    from pattern_induction.program import MAX_LEN, MAX_SUPPORT
    from pattern_induction.symbolic import symbolic_pick
    from pattern_induction.tasks import Task

    task = Task.from_obj(task_obj)
    supports = [text for text in task.visible if text]
    pick = symbolic_pick(supports)
    accuracy, recall, rejection = _rates(pick.pattern, task)
    clipped = [text[:MAX_LEN] for text in supports[:MAX_SUPPORT]]
    started = time.perf_counter()
    learned = _learned(clipped, ROOT, CEILING)
    seconds = time.perf_counter() - started
    answered = learned is not None and learned.regex is not None
    return {
        "id": task.id,
        "domain": task.domain,
        "regex": pick.pattern,
        "accuracy": accuracy,
        "recall": recall,
        "rejection": rejection,
        "margin": pick.margin if math.isfinite(pick.margin) else None,
        "near_tie": int(pick.ambiguous),
        "n_alternatives": len(pick.alternatives),
        "cost": pick.cost,
        "learned": int(answered),
        "learned_regex": learned.regex if answered else None,
        "agrees": int(answered and semantic_key(learned.regex) == semantic_key(pick.pattern)),
        "prior_conf": learned.prior_conf if answered else None,
        "n_programs": learned.n_programs if learned is not None else 0,
        "timeout": int(learned is None),
        "recognizer_seconds": seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    tasks = [json.loads(line) for line in (ROOT / "data" / "bench_c.jsonl").open()]
    started = time.perf_counter()
    rows = []
    with Pool(args.workers) as pool:
        for index, row in enumerate(pool.imap_unordered(_row, tasks, chunksize=1), start=1):
            rows.append(row)
            if index % 25 == 0:
                print(f"{index}/{len(tasks)} {time.perf_counter() - started:.0f}s", flush=True)
    order = {task["id"]: index for index, task in enumerate(tasks)}
    rows.sort(key=lambda row: order[row["id"]])
    OUT.write_text("".join(json.dumps(row) + "\n" for row in rows))


if __name__ == "__main__":
    main()
