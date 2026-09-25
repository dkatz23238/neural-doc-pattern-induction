"""Entry point for the first symbolic experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from pattern_induction.run_exp1 import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/exp1"))
    parser.add_argument(
        "--systems",
        default="memorize,heuristic,grex_default,grex_digits,grex_words,grex_repetitions,grex_combined",
    )
    parser.add_argument("--include-fast", action="store_true")
    parser.add_argument("--fast-timeout", type=float, default=6.0)
    parser.add_argument("--fast-limit", type=int, default=20)
    args = parser.parse_args()
    systems = [part for part in args.systems.split(",") if part]
    if args.include_fast:
        systems.append("fast")
    paths = {}
    for name in ("bench_c", "bench_b", "bench_d", "bench_a"):
        path = args.data / f"{name}.jsonl"
        if path.exists():
            paths[name] = path
    report = run(paths, args.out, systems, args.fast_timeout, args.fast_limit)
    for bench, payload in report["benchmarks"].items():
        print(bench)
        for row in payload["systems"]:
            print(
                f"  {row['system']:18} As={row['semantic_accuracy']:.3f} "
                f"R+={row['positive_recall']:.3f} R-={row['negative_rejection']:.3f} "
                f"p50={row['latency_ms']['p50']:.2f}ms n={row['n']}"
            )


if __name__ == "__main__":
    main()
