"""Command line for the frozen needle() decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pattern_induction.needle import NeedleError, needle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Induce a regex from example strings.")
    parser.add_argument("examples", nargs="*", help="positive examples")
    parser.add_argument("--file", type=Path, help="read one example per line")
    parser.add_argument("--json", action="store_true", help="print regex, confidence, alternatives, and method")
    parser.add_argument("--root", type=Path, default=None, help="checkout that holds the frozen checkpoints")
    args = parser.parse_args(argv)
    examples = list(args.examples)
    if args.file is not None:
        examples.extend(line for line in args.file.read_text().splitlines() if line)
    if not examples:
        parser.error("pass examples as arguments or via --file")
    try:
        result = needle(examples, args.root)
    except NeedleError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        print(
            json.dumps(
                {
                    "regex": result.regex,
                    "confidence": result.confidence,
                    "alternatives": result.alternatives,
                    "method": result.method,
                    "ambiguous": result.ambiguous,
                }
            )
        )
    else:
        print(result.regex)
        print(f"confidence {result.confidence:.4f} method {result.method} ambiguous {str(result.ambiguous).lower()}")
        for pattern in result.alternatives:
            print(f"alternative {pattern}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
