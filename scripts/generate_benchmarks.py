"""Generate the review benchmarks and a larger BENCH-A structural split."""

from __future__ import annotations

import argparse
from pathlib import Path

from pattern_induction.bench_a import generate_bench_a
from pattern_induction.bench_b import generate_bench_b
from pattern_induction.bench_c import generate_bench_c
from pattern_induction.bench_d import generate_bench_d
from pattern_induction.tasks import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--bench-b", type=int, default=2000)
    parser.add_argument("--bench-a-dev", type=int, default=2000)
    parser.add_argument("--bench-a-val", type=int, default=400)
    parser.add_argument("--bench-a-test", type=int, default=400)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print("BENCH-C")
    bench_c = generate_bench_c()
    write_jsonl(args.out / "bench_c.jsonl", bench_c)
    print(f"  {len(bench_c)} tasks")
    print("BENCH-B")
    bench_b = generate_bench_b(args.bench_b)
    write_jsonl(args.out / "bench_b.jsonl", bench_b)
    print(f"  {len(bench_b)} tasks")
    print("BENCH-D")
    bench_d = generate_bench_d()
    write_jsonl(args.out / "bench_d.jsonl", bench_d)
    print(f"  {len(bench_d)} tasks")
    print("BENCH-A")
    bench_a = generate_bench_a(args.bench_a_dev, args.bench_a_val, args.bench_a_test)
    write_jsonl(args.out / "bench_a.jsonl", bench_a)
    print(f"  {len(bench_a)} tasks")


if __name__ == "__main__":
    main()
