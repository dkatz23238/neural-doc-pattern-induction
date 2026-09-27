"""Data-recipe study on the frozen compositional model.

Teacher calls are optional and cached. Training reads the validated template
bank. BENCH-C is not scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pattern_induction.recipe_train import COMP_FAMILIES, train_recipe
from pattern_induction.teacher import generate, load_ledger


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _summary(root: Path) -> None:
    rows = []
    base = root / "results" / "data_recipe"
    if not base.exists():
        return
    for recipe_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        for metrics_path in sorted(recipe_dir.glob("seed_*/metrics.json")):
            rows.append(json.loads(metrics_path.read_text()))
    if not rows:
        return
    lines = [
        "# Data recipes for compositional generalization",
        "",
        "The span model, the loss, the step count, and the evaluation slices are the ones from the compositional run. The training distribution is the only change.",
        "D0 is the atomic curriculum. D1 adds a procedural matrix of other compositions and holds the eval languages out. D2 and D3 train on schemas a teacher proposed and the compiler accepted. D4 adds templates aimed at the remaining failures.",
        "D3 and D4 may include the eval families as contrasts. D0, D1, and D2 do not. Repeat counts in teacher schemas stay at or below 3, so four and five repetitions remain an extrapolation test.",
        "Held-out composition AUROC is the mean of prefix+repeat, repeat+optional, and prefix+alternation. The worst of those three is reported beside it. Renaming invariance below 0.99 rejects a recipe.",
        "",
        "| Recipe | Seed | Composition mean | Worst | Repeat 4–5 | Invariance | Exposes eval family | Reading |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        worst = row["worst_composition"]
        lines.append(
            f"| {row['recipe']} | {row['seed']} | {row['composition_auroc']:.3f} | "
            f"{worst['family']} {worst['auroc']:.3f} | {row['repeat_extra_auroc']:.3f} | "
            f"{row['min_invariance']:.3f} | {row['exposes_eval_family']} | {row['interpretation']} |"
        )
    lines.extend(["", "## Slices", ""])
    for row in rows:
        lines.append(f"### {row['recipe']} seed {row['seed']}")
        lines.append("")
        lines.append("| Slice | Balanced | AUROC | Member acceptance | Invariance |")
        lines.append("| --- | --- | --- | --- | --- |")
        for name in (
            "primitive",
            "repeat",
            "optional",
            "alternation",
            "repeat_extra",
            "optional_internal",
            *COMP_FAMILIES,
        ):
            view = row["slices"][name]
            lines.append(
                f"| {name} | {view['balanced']:.3f} | {view['auroc']:.3f} | "
                f"{view['member_acceptance']:.3f} | {view['invariance']:.3f} |"
            )
        lines.append("")
    ledger = load_ledger(root)
    lines.append(f"Teacher spend recorded in the ledger: ${ledger['spent_usd']:.4f} across {ledger['calls']} calls.")
    lines.append("")
    path = base / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("teacher", "train", "mine-teacher", "summary"), required=True)
    parser.add_argument("--recipe", default="d0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="composition_v1.md")
    parser.add_argument("--kind", choices=("schemas", "pairs"), default="schemas")
    parser.add_argument("--allow-heldout", action="store_true")
    parser.add_argument("--from-recipe", default="d2")
    args = parser.parse_args()
    root = _root()
    if args.stage == "teacher":
        packet = ""
        if "{failure_packet}" in (root / "data" / "teacher" / "prompts" / args.prompt).read_text():
            from pattern_induction.recipe_train import failure_packet

            packet = failure_packet(root, args.from_recipe, args.seed)
        metadata = generate(
            root,
            args.prompt,
            args.kind,
            args.allow_heldout,
            failure_packet=packet,
        )
        print(json.dumps(metadata, indent=2), flush=True)
    elif args.stage == "train":
        train_recipe(root, args.recipe, args.seed)
    elif args.stage == "mine-teacher":
        from pattern_induction.recipe_train import failure_packet

        packet = failure_packet(root, args.from_recipe, args.seed)
        metadata = generate(
            root,
            "failure_analysis_v1.md",
            "schemas",
            True,
            failure_packet=packet,
        )
        print(json.dumps(metadata, indent=2), flush=True)
    else:
        _summary(root)


if __name__ == "__main__":
    main()
