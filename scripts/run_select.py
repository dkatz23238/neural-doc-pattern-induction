"""Select a language from the frozen version space.

The recognizer and the chart stay frozen. BENCH-C is not scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pattern_induction.local_train import ALL
from pattern_induction.select_train import train_selector


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _summary(root: Path) -> None:
    path = root / "results" / "select" / "s1" / "seed_0" / "metrics.json"
    lines = [
        "# Language selection",
        "",
        "S1 leaves the scope recognizer and the version-space chart frozen. Syntactic copies and programs that accept the same strings are merged. A grammar fit on gold schemas scores operator choice, class width conditional on the class, repeat bounds, optionality, and literal length. Repeat bounds come from the atomic compose languages, where an unobserved longer repeat stays open. A small head on the frozen relational encoder, plus contrast features for width, letter affixes, and same-width retints, judges short strings where the remaining languages disagree. One confident witness stays inside the grammar tie; two that split the same way eliminate a language. Stability on support subsets breaks a remaining tie after the grammar.",
        "",
        "The checkpoint is chosen on atomic semantic top-1. Held-out composition is scored after that choice. BENCH-C is not scored.",
        "",
    ]
    if not path.exists():
        lines.append("S1 has not been trained yet.")
        lines.append("")
    else:
        row = json.loads(path.read_text())
        lines.extend(
            [
                f"Parameters: {row['parameters']} ({row['witness_parameters']} in the witness head). Selected step: {row['selected_step']}.",
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
    dest = root / "results" / "select" / "summary.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n")
    print(f"wrote {dest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "summary"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = _root()
    if args.stage == "summary":
        _summary(root)
    else:
        train_selector(root, args.seed)
        _summary(root)


if __name__ == "__main__":
    main()
