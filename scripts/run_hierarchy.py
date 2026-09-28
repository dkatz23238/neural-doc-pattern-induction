"""Hierarchical composition on the frozen data recipes.

H0 is the span model from the data-recipe study. H1 adds a group transformer.
H2 replaces that mixer with typed group relations. Recipes, steps, and the
evaluation slices stay put. BENCH-C is not scored, and this stage does not
call the teacher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pattern_induction.hierarchy_train import train_hierarchy
from pattern_induction.recipe_train import COMP_FAMILIES, TRAIN_FAMILIES


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_rows(root: Path) -> list[dict]:
    rows = []
    recipe_root = root / "results" / "data_recipe"
    for recipe in ("d0", "d1"):
        base = recipe_root / recipe
        if not base.exists():
            continue
        for path in sorted(base.glob("seed_*/metrics.json")):
            row = json.loads(path.read_text())
            row["arch"] = "h0"
            rows.append(row)
    hier = root / "results" / "hierarchy"
    if hier.exists():
        for path in sorted(hier.glob("*/*/seed_*/metrics.json")):
            rows.append(json.loads(path.read_text()))
    return rows


def _gate(row: dict) -> str:
    mean = row["composition_auroc"]
    worst = row["worst_composition"]["auroc"]
    extra = row["repeat_extra_auroc"]
    inv = row["min_invariance"]
    atomic = row.get("atomic_min_auroc")
    if atomic is None:
        atomic = min(row["slices"][name]["auroc"] for name in TRAIN_FAMILIES)
    if inv < 0.99:
        return "reject: invariance moved"
    if atomic < 0.90:
        return "atomic operators slipped"
    if mean > 0.90 and worst > 0.85 and extra > 0.90:
        return "hierarchy gate met"
    if mean > 0.90 and worst > 0.85:
        return "composition moved, extrapolation under 0.90"
    if extra < 0.90 and mean >= 0.82:
        return "partial composition gain, extrapolation under 0.90"
    if mean >= 0.82:
        return "above 0.82, short of the gate"
    if mean >= 0.78:
        return "above the atomic H0 mean, short of 0.82"
    return "near the atomic H0 mean"


def _summary(root: Path) -> None:
    rows = _load_rows(root)
    lines = [
        "# Hierarchical group composition",
        "",
        "The data-recipe study is the control. H0 is that span model. H1 and H2 keep its character features, span encoder, optimizer, 1200 steps, batch, checkpoint rule, and evaluation slices.",
        "H1 pools spans into groups and mixes the groups with a transformer. H2 mixes them by typed edges: parent and child, next sibling, same repeated unit, and alignment from the query group to a support group with the same structural signature.",
        "D0 is the atomic curriculum. D1 is the procedural matrix that holds the eval languages out. Repeat counts in that bank stay at or below 3, so four and five repetitions remain an extrapolation test.",
        "The gate is a held-out composition mean above 0.90, a worst mixture above 0.85, repeat depths 4–5 above 0.90, atomic operators still near solved, and renaming invariance still about 1. Meeting it is the point at which a decoder would come back.",
        "",
        "| Arch | Recipe | Seed | Params | Composition mean | Worst | Repeat 4–5 | Atomic min | Invariance | Gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        worst = row["worst_composition"]
        atomic = row.get("atomic_min_auroc")
        if atomic is None:
            atomic = min(row["slices"][name]["auroc"] for name in TRAIN_FAMILIES)
        lines.append(
            f"| {row['arch']} | {row['recipe']} | {row['seed']} | {row['parameters']} | "
            f"{row['composition_auroc']:.3f} | {worst['family']} {worst['auroc']:.3f} | "
            f"{row['repeat_extra_auroc']:.3f} | {atomic:.3f} | {row['min_invariance']:.3f} | {_gate(row)} |"
        )
    lines.extend(["", "## Slices", ""])
    for row in rows:
        if row["arch"] == "h0" and (row["seed"] != 0 or row["recipe"] not in {"d0", "d1"}):
            continue
        lines.append(f"### {row['arch']} {row['recipe']} seed {row['seed']}")
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
    lines.extend(["## Reading", ""])
    lines.append(
        "H0 on D0 is the frozen atomic baseline: composition mean 0.748, worst slice prefix+repeat 0.548, repeat 4–5 at 0.935, invariance 1.000. "
        "H0 on D1 is the best stable data control: the mean sits around 0.81 and repeat 4–5 falls into the low 0.80s. "
        "A hierarchical run has to beat the H0 number on the same recipe and still clear the absolute gate."
    )
    lines.append("")
    interesting = [row for row in rows if row["arch"] != "h0"]
    if not interesting:
        lines.append("H1 and H2 have not been trained yet.")
    else:
        for row in interesting:
            worst = row["worst_composition"]
            lines.append(
                f"{row['arch']} {row['recipe']} seed {row['seed']}: composition {row['composition_auroc']:.3f}, "
                f"worst {worst['family']} {worst['auroc']:.3f}, repeat 4–5 {row['repeat_extra_auroc']:.3f}, "
                f"invariance {row['min_invariance']:.3f}. {_gate(row)}."
            )
        met = [row for row in interesting if _gate(row) == "hierarchy gate met"]
        lines.append("")
        if met:
            lines.append("At least one hierarchical run met the gate. That is the point to consider a decoder.")
        else:
            lines.append(
                "No hierarchical run met the gate. Invariance is the rename check, and a drop below 0.99 would reject the architecture on its own. "
                "The decoder stays off until the mean, the worst mixture, and repeat-depth extrapolation clear together."
            )
    lines.append("")
    path = root / "results" / "hierarchy" / "summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "summary"), required=True)
    parser.add_argument("--arch", choices=("h1", "h2"), default="h2")
    parser.add_argument("--recipe", default="d0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = _root()
    if args.stage == "summary":
        _summary(root)
    else:
        train_hierarchy(root, args.arch, args.recipe, args.seed)


if __name__ == "__main__":
    main()
