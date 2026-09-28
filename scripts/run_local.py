"""Local structural hypotheses composed by a chart.

The recognizer scores spans, repeats, optionals, alternation pairs, and cuts
independently. The chart assembles programs. BENCH-C is not scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pattern_induction.local_train import ALL, ATOMIC, OOD, train_local


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _fmt(value) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _summary(root: Path) -> None:
    path = root / "results" / "local_chart" / "l1" / "seed_0" / "metrics.json"
    lines = [
        "# Local structural hypotheses",
        "",
        "L1 keeps the symbol-invariant encoder and scores local facts independently: copy spans, class spans, repeat units, optional spans, alternation pairs, repeated-span pairs, and sequence cuts. Several alternatives stay alive. A chart stitches those facts into programs, and the executor checks them.",
        "",
        "The first gate is local recall. If the gold fact is missing from the top 8 hypotheses, the recognizer is the bottleneck. If the gold facts are present and the chart's program oracle stays low, the chart is the bottleneck.",
        "",
        "Training mixes the atomic languages with the D1 schemas. The eval compositions, including repeats of depth 4–5, are held out. BENCH-C is not scored.",
        "",
    ]
    if not path.exists():
        lines.append("L1 has not been trained yet.")
        lines.append("")
    else:
        row = json.loads(path.read_text())
        lines.extend(
            [
                f"Parameters: {row['parameters']}. Selected step: {row['selected_step']}.",
                "",
                "| Family | Local recall@8 | Local recall@1 | Program oracle@8 | Semantic top-1 | Support acceptance | Programs |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for family in ALL:
            view = row["families"][family]
            lines.append(
                f"| {family} | {view['local_recall']:.3f} | {view['local_recall_at_1']:.3f} | {view['oracle']:.3f} | "
                f"{view['top1']:.3f} | {view['support']:.3f} | {view['programs']:.1f} |"
            )
        lines.extend(["", "| Family | Copy | Class | Repeat | Optional | Alternation | Repeat pair | Cut |", "| --- | --- | --- | --- | --- | --- | --- | --- |"])
        for family in ALL:
            local = row["families"][family]["local"]
            lines.append(
                "| "
                + " | ".join(
                    [family]
                    + [_fmt(local.get(kind)) for kind in ("copy", "cls", "repeat", "optional", "alt", "pair", "cut")]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                f"Atomic local recall@8 {row['atomic_local']:.3f}. Held-out local recall@8 {row['ood_local']:.3f}. "
                f"Held-out local recall@1 {row['ood_local_at_1']:.3f}. Held-out program oracle@8 {row['ood_oracle']:.3f}. "
                f"Held-out semantic top-1 {row['ood_top1']:.3f}. Repeat-depth local recall {row['repeat_extra_local']:.3f}. "
                f"Renaming invariance {row['min_invariance']:.3f}.",
                "",
                row["interpretation"] + ".",
                "",
            ]
        )
        if row["ood_local"] < 0.50:
            lines.append("The next change belongs in the recognizer. The chart is not the place to look while the gold spans are missing from the top hypotheses. A pretrained backbone and BENCH-C stay off.")
        elif row["ood_oracle"] < 0.40:
            lines.append("The recognizer is offering the gold local structure. The next change belongs in the chart. BENCH-C stays off.")
        else:
            lines.append("Local hypotheses and symbolic composition both move on the held-out languages. BENCH-C still waits on a higher program oracle.")
        lines.append("")
    dest = root / "results" / "local_chart" / "summary.md"
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
        train_local(root, args.seed)


if __name__ == "__main__":
    main()
