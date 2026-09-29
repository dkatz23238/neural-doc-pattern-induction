"""Time the frozen needle() path on the showcase fields.

The symbolic winner is the decision. The default times that decision and the
full call. ``--stages`` also times candidate generation, code-length scoring,
chart assembly, and selection. It does not train and it does not open BENCH-E.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from pattern_induction.candidates import pattern_cost, presented_candidates
from pattern_induction.needle import needle
from pattern_induction.program import MAX_SUPPORT, Prog, Task, _MEMBERSHIP
from pattern_induction.select_train import load_selector, select_program, synthesize_assembly
from pattern_induction.symbolic import symbolic_pick

SHOWCASE = (
    ("invoice", ["INV-2024-10021", "INV-2026-88310", "INV-2025-00418"], r"^INV-\d{4}-\d{5}$"),
    ("version", ["1.2.0", "1.4.3", "2.0.1"], r"^(?:\d\.){2}\d$"),
    ("section", ["1.2", "1.10", "3.4"], r"^\d\.\d{1,2}$"),
    (
        "filename",
        ["report.pdf", "notes.txt", "budget.csv"],
        r"^[a-z]{5,6}\.(?:pdf|docx|xlsx|csv|txt|png|jpg|jpeg|json|xml|html|zip)$",
    ),
    ("purchase order", ["PO-1000", "PO-1042", "PO-1108"], r"^PO-\d{4}$"),
    ("customer", ["CUS-DE1234", "CUS-SO9281", "CUS-AM0042"], r"^CUS-[A-Z]{2}\d{4}$"),
)


def _rss_kb() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", action="store_true", help="time assembly and selection separately")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    load_selector(root)
    print(f"load {time.perf_counter() - started:.3f}s rss_kb {_rss_kb()}")
    for name, examples, expected in SHOWCASE:
        t0 = time.perf_counter()
        pick = symbolic_pick(examples)
        decision_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        result = needle(examples, root)
        needle_s = time.perf_counter() - t1
        if result.regex != expected or pick.pattern != expected:
            raise SystemExit(f"{name} regex changed: {result.regex}")
        line = (
            f"{name} decision {decision_s * 1000:.1f}ms needle {needle_s * 1000:.1f}ms "
            f"confidence {result.confidence:.4f} ambiguous {str(result.ambiguous).lower()} "
            f"regex {result.regex}"
        )
        if args.stages:
            _MEMBERSHIP.clear()
            t2 = time.perf_counter()
            shown, _pool = presented_candidates(examples, k=8, pool_limit=80)
            candidates_s = time.perf_counter() - t2
            t3 = time.perf_counter()
            for item in shown:
                pattern_cost(item.pattern, examples, True)
            cost_s = time.perf_counter() - t3
            scope, witness, prior = load_selector(root)
            task = Task("profile", examples[:MAX_SUPPORT], Prog("COPY"), [])
            _MEMBERSHIP.clear()
            t4 = time.perf_counter()
            assembly = synthesize_assembly(scope, task, v2=True)
            assemble_s = time.perf_counter() - t4
            _MEMBERSHIP.clear()
            t5 = time.perf_counter()
            select_program(
                scope,
                witness,
                prior,
                task,
                with_stability=True,
                v2=True,
                stability_mode="direct",
                operator_witnesses=True,
            )
            select_s = time.perf_counter() - t5
            line += (
                f" candidates {candidates_s * 1000:.1f}ms cost {cost_s * 1000:.1f}ms "
                f"assemble {assemble_s * 1000:.1f}ms select {select_s * 1000:.1f}ms "
                f"programs {len(assembly.programs)}"
            )
        print(line)
    print(f"rss_kb {_rss_kb()}")


if __name__ == "__main__":
    main()
