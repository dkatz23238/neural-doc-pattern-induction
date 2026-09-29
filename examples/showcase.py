"""Six field examples for the frozen needle() decision.

Run from the repo root:

    uv run python examples/showcase.py
"""

from __future__ import annotations

from pathlib import Path

from pattern_induction.needle import needle

ROOT = Path(__file__).resolve().parents[1]

FIELDS = (
    ("invoice", ["INV-2024-10021", "INV-2026-88310", "INV-2025-00418"]),
    ("version", ["1.2.0", "1.4.3", "2.0.1"]),
    ("section", ["1.2", "1.10", "3.4"]),
    ("filename", ["report.pdf", "notes.txt", "budget.csv"]),
    ("purchase order", ["PO-1000", "PO-1042", "PO-1108"]),
    ("customer", ["CUS-DE1234", "CUS-SO9281", "CUS-AM0042"]),
)


def main() -> None:
    for name, examples in FIELDS:
        result = needle(list(examples), ROOT)
        print(name)
        print(f"  examples {', '.join(examples)}")
        print(f"  regex {result.regex}")
        print(f"  confidence {result.confidence:.4f} ambiguous {str(result.ambiguous).lower()} method {result.method}")
        for pattern in result.alternatives:
            print(f"  alternative {pattern}")


if __name__ == "__main__":
    main()
