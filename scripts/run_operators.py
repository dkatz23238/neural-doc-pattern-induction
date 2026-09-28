"""Gold-schema operator algebra on the frozen data recipes.

H3-oracle evaluates a shared SEQ / REPEAT / OPTIONAL / ALT module on the
generator's tree. It does not infer that tree, and it does not call the
teacher. H2 remains the control. Structure induction stays off until this
oracle shows that the algebra itself composes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pattern_induction.operator_train import train_operator
from pattern_induction.recipe_train import COMP_FAMILIES, TRAIN_FAMILIES


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load(path: Path, arch: str | None = None) -> dict | None:
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    if arch is not None:
        row["arch"] = arch
    return row


def _rows(root: Path) -> list[dict]:
    rows = []
    for recipe in ("d0", "d1"):
        row = _load(root / "results" / "data_recipe" / recipe / "seed_0" / "metrics.json", "h0")
        if row is not None:
            rows.append(row)
        row = _load(root / "results" / "hierarchy" / "h2" / recipe / "seed_0" / "metrics.json")
        if row is not None:
            rows.append(row)
        row = _load(root / "results" / "operator_algebra" / "h3o" / recipe / "seed_0" / "metrics.json")
        if row is not None:
            rows.append(row)
    return rows


def _gate(row: dict) -> str:
    if row["arch"] == "h3o":
        return row.get("interpretation") or ""
    mean = row["composition_auroc"]
    worst = row["worst_composition"]["auroc"]
    extra = row["repeat_extra_auroc"]
    inv = row["min_invariance"]
    if inv < 0.99:
        return "reject: invariance moved"
    if mean > 0.90 and worst > 0.85 and extra > 0.90:
        return "composition gate met"
    return "control"


def _summary(root: Path) -> None:
    rows = _rows(root)
    lines = [
        "# Operator algebra",
        "",
        "H2 is the typed group-graph control. H3-oracle keeps that trunk, those recipes, the 1200-step budget, and the same evaluation slices. The change is the mixer: one shared function each for sequence, repeat, optional, and alternation, folded over the generator's operator tree.",
        "The model is given that tree. Support strings still supply the leaf bindings. The query's annotation is not an input. Repeat bounds are the schema's min and max, so a query with four or five copies is the same repeat operator the support showed at two to four.",
        "The oracle bar is a held-out composition mean above 0.95, because the structure is handed in. The product gate stays a mean above 0.90, a worst mixture above 0.85, repeat depths 4–5 above 0.90, and renaming invariance about 1. A mean near 0.80 means this operator algebra does not compose, and structure induction should not be built on it.",
        "",
        "| Arch | Recipe | Seed | Params | Composition mean | Worst | Repeat 4–5 | Atomic min | Invariance | Reading |",
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
    oracle = [row for row in rows if row["arch"] == "h3o"]
    lines.extend(["## Reading", ""])
    if not oracle:
        lines.append("H3-oracle has not been trained yet.")
    else:
        for row in oracle:
            worst = row["worst_composition"]
            lines.append(
                f"H3-oracle {row['recipe'].upper()} seed {row['seed']}: composition {row['composition_auroc']:.3f}, "
                f"worst {worst['family']} {worst['auroc']:.3f}, repeat 4–5 {row['repeat_extra_auroc']:.3f}, "
                f"atomic min {row['atomic_min_auroc']:.3f}, invariance {row['min_invariance']:.3f}, "
                f"{row['parameters']} parameters. {row['interpretation']}."
            )
        lines.append("")
        ready = [row for row in oracle if row["composition_auroc"] > 0.95 and row["worst_composition"]["auroc"] > 0.85 and row["repeat_extra_auroc"] > 0.90]
        if ready:
            lines.append("The gold tree is enough for the shared modules to compose. The next experiment is the same algebra with the tree inferred from the support strings.")
        else:
            lines.append("The gold tree was not enough. H3-induced and latent-schema distillation stay off, and the decoder stays off.")
    lines.append("")
    path = root / "results" / "operator_algebra" / "summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "summary"), required=True)
    parser.add_argument("--arch", choices=("h3o",), default="h3o")
    parser.add_argument("--recipe", default="d0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = _root()
    if args.stage == "summary":
        _summary(root)
    else:
        train_operator(root, args.arch, args.recipe, args.seed)


if __name__ == "__main__":
    main()
