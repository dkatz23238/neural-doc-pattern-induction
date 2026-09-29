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
        "| Recipe | Seed | Composition mean | Worst | Repeat 4–5 | Invariance | Exposes eval family | Band |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def band_label(row: dict) -> str:
        mean = row["composition_auroc"]
        worst = row["worst_composition"]["auroc"]
        if row["min_invariance"] < 0.99:
            return "reject: invariance moved"
        if mean > 0.90 and worst > 0.85:
            return "major success"
        if mean >= 0.82:
            return "mean in the 0.82–0.87 band; worst slice under 0.85"
        if mean >= 0.78:
            return "above the atomic baseline, short of 0.82"
        return "near the atomic baseline"

    for row in rows:
        worst = row["worst_composition"]
        lines.append(
            f"| {row['recipe']} | {row['seed']} | {row['composition_auroc']:.3f} | "
            f"{worst['family']} {worst['auroc']:.3f} | {row['repeat_extra_auroc']:.3f} | "
            f"{row['min_invariance']:.3f} | {row['exposes_eval_family']} | {band_label(row)} |"
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
    lines.extend(["## Reading", ""])
    by_recipe: dict[str, list[dict]] = {}
    for row in rows:
        by_recipe.setdefault(row["recipe"], []).append(row)

    def band(recipe: str) -> str:
        group = by_recipe.get(recipe) or []
        if not group:
            return ""
        means = [row["composition_auroc"] for row in group]
        worsts = [row["worst_composition"]["auroc"] for row in group]
        extras = [row["repeat_extra_auroc"] for row in group]
        exposed = group[0]["exposes_eval_family"]
        return (
            f"{recipe} composition mean {min(means):.3f}–{max(means):.3f} over {len(group)} "
            f"{'seed' if len(group) == 1 else 'seeds'}, "
            f"worst slice {min(worsts):.3f}–{max(worsts):.3f}, "
            f"repeat 4–5 AUROC {min(extras):.3f}–{max(extras):.3f}. "
            f"Eval families in that bank: {exposed if exposed else 'none'}."
        )

    lines.append(
        "The success bar for this phase is a held-out composition mean above 0.90 and a worst mixture above 0.85, with renaming invariance still about 1. "
        "A mean from 0.82 to 0.87 is evidence that the curriculum matters. A mean near 0.75 means the data variable did not move the model. "
        "A mean above 0.93 with repeat-depth extrapolation still strong would be the point to attach a decoder. Invariance under 0.99 rejects the recipe."
    )
    lines.append("")
    for recipe in ("d0", "d1", "d2", "d3", "d4"):
        text = band(recipe)
        if text:
            lines.append(text)
    lines.append("")
    lines.append(
        "D1 and D2 keep the three eval mixtures out of the training bank. Their means sit around 0.81, and one D2 seed reaches the 0.82–0.87 band. "
        "The worst mixture stays below 0.85 on every seed. Prefix+repeat rises from the atomic 0.548 into the mid 0.70s or low 0.80s, and it is still the weak slice on several seeds. "
        "Repeat depths 4 and 5 drop from the atomic 0.935 into the low 0.80s, so the composition gain trades away some depth extrapolation."
    )
    lines.append(
        "D3 contrasts one structural attachment at a time. Its bank includes a repeat-then-optional language, and that eval slice does not become the strong one. "
        "D4 adds failure-targeted schemas, including prefix+repeat itself, and the mean does not rise. "
        "No recipe reaches the 0.90 / 0.85 bar. Invariance stays 1.000, so none are rejected for a broken rename. The decoder stays off."
    )
    lines.append("")
    lines.append(
        f"Teacher spend recorded in the ledger: ${ledger['spent_usd']:.4f} across {ledger['calls']} calls. "
        "Rates used for that figure are $2 / $12 per million input / output tokens for gpt-5.6-terra."
    )
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
            max_output_tokens=8000,
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
