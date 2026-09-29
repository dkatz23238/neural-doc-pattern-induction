"""Scope relations on the frozen local recognizer.

The version-space chart stays frozen. BENCH-C is not scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pattern_induction.local_train import ALL, OOD
from pattern_induction.scope_train import train_scope


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _fmt(value) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _summary(root: Path) -> None:
    path = root / "results" / "scope" / "p2b" / "seed_0" / "metrics.json"
    lines = [
        "# Scope relations",
        "",
        "P2b freezes the version-space chart and the L1 recognizer. New heads read the generator's groups: same repeated unit, containment, group start and end, optional child, alternation branch, and shared prefix or suffix. The training objective is recall at 8. Each strong class or repeat span keeps a second bound, and alternation is the frozen bilinear plus a residual.",
        "",
        "The checkpoint is chosen on atomic local recall. Held-out local recall and the version-space oracle on the model's own top 8 are measured after that choice. BENCH-C is not scored.",
        "",
    ]
    if not path.exists():
        lines.append("P2b has not been trained yet.")
        lines.append("")
    else:
        row = json.loads(path.read_text())
        lines.extend(
            [
                f"Parameters: {row['parameters']} ({row['trainable']} trained). Selected step: {row['selected_step']}.",
                "",
                "| Family | Local recall@8 | Local recall@1 | Version-space oracle | Open-repeat oracle | Programs | Truncated |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for family in ALL:
            view = row["families"][family]
            lines.append(
                f"| {family} | {view['local_recall']:.3f} | {view['local_recall_at_1']:.3f} | {view['oracle']:.3f} | "
                f"{view['open_repeat']:.3f} | {view['programs']:.1f} | {view['truncated']} |"
            )
        lines.extend(
            [
                "",
                "| Family | Copy | Class | Repeat | Optional | Alternation | Repeat pair | Cut |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for family in ALL:
            local = row["families"][family]["local"]
            lines.append(
                "| "
                + " | ".join([family] + [_fmt(local.get(kind)) for kind in ("copy", "cls", "repeat", "optional", "alt", "pair", "cut")])
                + " |"
            )
        lines.extend(
            [
                "",
                f"Atomic local recall@8 {row['atomic_local']:.3f}. Atomic version-space oracle {row['atomic_oracle']:.3f}. "
                f"Held-out local recall@8 {row['ood_local']:.3f}. Held-out local recall@1 {row['ood_local_at_1']:.3f}. "
                f"Held-out version-space oracle {row['ood_oracle']:.3f}. Renaming invariance {row['min_invariance']:.3f}. "
                f"Truncated searches: {row['truncated']}.",
                "",
                row["interpretation"] + ".",
                "",
            ]
        )
        if row["ood_local"] >= 0.95 and row["min_invariance"] >= 1.0 and row["ood_oracle"] < 0.90:
            lines.append(
                "Local recall cleared the scope gate. Program selection waits until the version-space oracle on these facts is higher. "
                "A pretrained backbone and BENCH-C stay off."
            )
        elif row["ood_local"] >= 0.95 and row["ood_oracle"] >= 0.90:
            held_programs = sum(row["families"][family]["programs"] for family in OOD) / len(OOD)
            lines.append(
                "Local recall and the version-space oracle both cleared their lines on the model's own facts. "
                f"Held-out tasks leave {held_programs:.0f} probe-consistent programs on average. "
                "Choosing among those programs is the following experiment. A pretrained backbone and BENCH-C stay off."
            )
        else:
            lines.append(
                "Held-out local recall is still the gate. The chart stays frozen. A pretrained backbone and BENCH-C stay off."
            )
        lines.append("")
    dest = root / "results" / "scope" / "summary.md"
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
        train_scope(root, args.seed)
        _summary(root)


if __name__ == "__main__":
    main()
