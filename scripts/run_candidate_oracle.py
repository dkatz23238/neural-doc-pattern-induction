"""Score the symbolic candidate neighborhood and the CPU rankers."""

from __future__ import annotations

import argparse
from pathlib import Path

from pattern_induction.candidate_oracle import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/exp2"))
    parser.add_argument(
        "--benches",
        default="bench_c,bench_b,bench_a,bench_d",
        help="Comma-separated benchmark files to score.",
    )
    args = parser.parse_args()
    benches = [part for part in args.benches.split(",") if part]
    report = run(args.data, args.out, benches)
    if "bench_c" in report["benchmarks"]:
        print("bench_c")
        for row in report["benchmarks"]["bench_c"]["systems"]:
            print(f"  {row['system']:22} {row['semantic_accuracy']:.3f}")
    for paragraph in report.get("interpretation", []):
        print(paragraph)
        print()


if __name__ == "__main__":
    main()
