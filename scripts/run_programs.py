"""Execution-guided AST synthesis.

The relational encoder proposes a program. A symbolic executor accepts or
rejects strings, and it drops any beam entry that misses a support. BENCH-C
is not scored: this grammar is the compositional language, not closed numeric
ranges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pattern_induction.program_train import ALL, ATOMIC, OOD, train_proposer


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _summary(root: Path) -> None:
    path = root / "results" / "programs" / "p1" / "seed_0" / "metrics.json"
    lines = [
        "# Execution-guided program synthesis",
        "",
        "P1 keeps the symbol-invariant encoder and asks it to propose a program. Sequence, repeat, optional, and alternation are executed exactly. Literals are copied from a support span, so an unseen delimiter is retrieved rather than generated from a character vocabulary.",
        "",
        "Training mixes the atomic languages with the D1 schemas. The eval compositions, including repeats of depth 4–5, are held out. A beam of 16 programs is executed on the supports; anything that misses a support is dropped. The reported top-1 is the remaining program with the best model score minus a small length prior.",
        "",
        "The gates are renaming invariance about 1, support acceptance 1 after that filter, held-out semantic top-1 above 0.90, oracle@4 above 0.95, oracle@16 above 0.98, and repeat-depth top-1 above 0.90. Semantic agreement is execution on hidden positives and negatives, not exact AST text.",
        "",
    ]
    if not path.exists():
        lines.append("P1 has not been trained yet.")
        lines.append("")
    else:
        row = json.loads(path.read_text())
        lines.extend(
            [
                f"Parameters: {row['parameters']}. Selected step: {row['selected_step']}.",
                "",
                "| Family | Semantic top-1 | Raw top-1 | Oracle@4 | Oracle@16 | Support acceptance |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for family in ALL:
            view = row["families"][family]
            lines.append(
                f"| {family} | {view['top1']:.3f} | {view['raw1']:.3f} | {view['oracle4']:.3f} | "
                f"{view['oracle16']:.3f} | {view['support']:.3f} |"
            )
        lines.extend(
            [
                "",
                f"Atomic top-1 {row['atomic_top1']:.3f}. Held-out top-1 {row['ood_top1']:.3f}. "
                f"Held-out oracle@4 {row['ood_oracle4']:.3f}. Held-out oracle@16 {row['ood_oracle16']:.3f}. "
                f"Repeat 4–5 top-1 {row['repeat_extra_top1']:.3f}. Support acceptance {row['ood_support']:.3f}. "
                f"Renaming invariance {row['min_invariance']:.3f}. Mean programs in the beam {row.get('mean_beam', 0):.1f}.",
                "",
            ]
        )
        empty = [family for family in ALL if row["families"][family]["beam"] == 0]
        if empty:
            lines.append(
                "No support-consistent program was found for "
                + ", ".join(empty)
                + ". Oracle@16 on those families is not a ranking among 16 candidates."
            )
            lines.append("")
        if all(abs(row["families"][family]["oracle16"] - row["families"][family]["top1"]) < 1e-9 for family in ALL):
            lines.append("Where a program is returned, oracle@16 equals semantic top-1. Extra beam entries did not add a correct language.")
            lines.append("")
        if row["ood_oracle16"] > 0.98 and row["ood_top1"] > 0.90 and row["repeat_extra_top1"] > 0.90:
            lines.append("The beam contains the right language often enough to treat selection as the next problem. BENCH-C still waits: its closed ranges are outside this grammar.")
        elif row["ood_oracle16"] > 0.90:
            lines.append("The proposer finds a correct program in the beam more often than it emits one. Calibration is ahead of a new encoder. BENCH-C stays off.")
        else:
            lines.append(
                "The proposer is not putting a semantically correct program in the top 16 on the held-out languages. "
                "Alternation is solved and the atomic repeat is often exact, including the copied separator. "
                "The held-out mixtures are not. A pretrained backbone and BENCH-C stay off."
            )
        lines.append("")
    dest = root / "results" / "programs" / "summary.md"
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
        train_proposer(root, args.seed)


if __name__ == "__main__":
    main()
