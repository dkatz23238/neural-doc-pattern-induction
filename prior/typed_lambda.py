"""One scale per ambiguity, so a head that does not help can sit out.

The heads and features are the expansion prior's. The program score is

    evidence + λ_charset * charset + λ_width * width
             + λ_integer * integer + λ_repeat * repeat

Each λ is a softplus, initialized at 0.01. Softplus can shrink toward 0.
A head whose scale stays there leaves the program ranking with the frozen
evidence scorer.

Checkpoints maximize the mean of synthetic validation top-1 and factory
validation top-1. Factory held-out is not part of that choice.

The gate, fixed before the run: on factory held-out, program accuracy on
each ambiguity slice is at least the evidence accuracy on those same rows;
factory held-out overall and every synthetic family stay within 0.02 of
evidence. Both seeds have to clear that bar.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from prior.expansion import (
    HEADS,
    WIDTH_OPEN,
    ExpansionPrior,
    _batch,
    _fmt,
    evaluate,
    parameter_count,
    renaming_invariance,
)
from prior.factory_mix import mix_batch

INIT_LAMBDA = 0.01
FAMILY_MARGIN = 0.02
OWN_SLICES = ("charset_expansion", "open_width", "integer_expansion", "loose_repeat")
_SLICE_SPEC = {
    "charset_expansion": ("charset", "expansion"),
    "open_width": ("width", "open"),
    "integer_expansion": ("integer", "expansion"),
    "loose_repeat": ("repeat", "expansion"),
}


def _init_raw() -> float:
    return math.log(math.expm1(INIT_LAMBDA))


class TypedLambdaPrior(ExpansionPrior):
    """The same four heads, each with its own scale."""

    def __init__(self, hidden: int = 32):
        super().__init__(hidden)
        self.raw_lambda = torch.nn.Parameter(torch.full((len(HEADS),), _init_raw()))

    def scales(self) -> dict[str, torch.Tensor]:
        values = F.softplus(self.raw_lambda)
        return {name: values[index] for index, name in enumerate(HEADS)}

    def scale(self) -> torch.Tensor:
        return F.softplus(self.raw_lambda).mean()

    def combined_scores(self, batch: dict[str, torch.Tensor], lam: float | None = None) -> torch.Tensor:
        evidence = batch["evidence"].detach()
        total = torch.zeros(batch["mask"].shape, device=batch["mask"].device)
        scales = F.softplus(self.raw_lambda)
        for index, name in enumerate(HEADS):
            scored = self.scores_from(name, batch[f"{name}_feat"], batch[f"{name}_expand"])
            total = total + scales[index] * (scored * batch[f"{name}_mask"]).sum(dim=-1)
        return evidence + total


def selection_score(synthetic_top1: float, factory_top1: float) -> float:
    """Equal weight on the two concept-disjoint validation sets."""
    return 0.5 * (synthetic_top1 + factory_top1)


def slice_mask(batch: dict[str, torch.Tensor], gate: str) -> torch.Tensor:
    """Rows where this gate's gold hypothesis and a rival are both in the pool."""
    name, kind = _SLICE_SPEC[gate]
    gold = batch[f"{name}_hyp_gold"] > 0.5
    mask = batch[f"{name}_hyp_mask"] > 0.5
    expand = batch[f"{name}_hyp_expand"] > 0.5
    other = (~gold & mask).any(dim=-1)
    if kind == "expansion":
        return (gold & expand & mask).any(dim=-1) & other
    opened = batch[f"{name}_hyp"][:, :, WIDTH_OPEN] > 0.5
    return (gold & mask & opened).any(dim=-1) & other


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@torch.no_grad()
def slice_program(model: TypedLambdaPrior, data: dict, seed: int, batch_size: int = 256) -> dict:
    """Program top-1 against evidence, on the rows that belong to each gate."""
    model.eval()
    buckets = {name: {"combined": [], "evidence": []} for name in OWN_SLICES}
    count = len(data["family"])
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        batch = _batch(data, torch.arange(start, stop), seed)
        combined = model.combined_scores(batch)
        mask = batch["mask"]
        gold = batch["gold"]
        choice = combined.masked_fill(~mask, -1e9).argmax(dim=-1)
        evidence_choice = batch["evidence"].masked_fill(~mask, -1e9).argmax(dim=-1)
        rows = torch.arange(choice.shape[0])
        hit = gold[rows, choice]
        evidence_hit = gold[rows, evidence_choice]
        for name in OWN_SLICES:
            chosen = slice_mask(batch, name)
            if not bool(chosen.any()):
                continue
            buckets[name]["combined"].extend(float(bit) for bit in hit[chosen].tolist())
            buckets[name]["evidence"].extend(float(bit) for bit in evidence_hit[chosen].tolist())
    return {
        name: {
            "combined": _mean(rows["combined"]),
            "evidence": _mean(rows["evidence"]),
            "n": len(rows["combined"]),
        }
        for name, rows in buckets.items()
    }


def degraded_families(report: dict, margin: float = FAMILY_MARGIN) -> list[dict]:
    found = []
    for split, family_key, evidence_key in (
        ("val", "val_by_family", "val_evidence_by_family"),
        ("ood", "ood_by_family", "ood_evidence_by_family"),
    ):
        for name, score in report.get(family_key, {}).items():
            evidence = report[evidence_key][name]
            if score < evidence - margin:
                found.append(
                    {"split": split, "family": name, "combined": score, "evidence": evidence}
                )
    return found


def missed_slices(slices: dict) -> list[dict]:
    """Factory held-out slices that fall under evidence."""
    found = []
    for name in OWN_SLICES:
        row = slices[name]
        if row["n"] == 0:
            continue
        if row["combined"] < row["evidence"]:
            found.append({"slice": name, **row})
    return found


def degraded_slices(slices: dict | None, margin: float = FAMILY_MARGIN) -> list[dict]:
    """Synthetic slices more than `margin` under evidence."""
    if not slices:
        return [{"slice": "missing", "combined": 0.0, "evidence": 0.0, "n": 0}]
    found = []
    for name in OWN_SLICES:
        row = slices[name]
        if row["n"] == 0:
            continue
        if row["combined"] < row["evidence"] - margin:
            found.append({"slice": name, **row})
    return found


def seed_passes(report: dict) -> bool:
    held = report["factory_heldout"]
    overall = held["top1"] >= held["evidence_top1"] - FAMILY_MARGIN
    return (
        overall
        and not missed_slices(report["heldout_slices"])
        and not degraded_families(report)
        and not degraded_slices(report.get("val_slices"))
        and not degraded_slices(report.get("ood_slices"))
    )


def seed_retains(report: dict) -> bool:
    """Integer and loose-repeat held-out programs beat evidence, not only match it."""
    for name in ("integer_expansion", "loose_repeat"):
        row = report["heldout_slices"][name]
        if row["n"] == 0 or not row["combined"] > row["evidence"]:
            return False
    return True


def _lambda_map(model: TypedLambdaPrior) -> dict[str, float]:
    return {name: float(value.detach()) for name, value in model.scales().items()}


def train_typed(
    seed: int,
    synthetic: dict,
    factory_train: dict,
    synthetic_val: dict,
    factory_val: dict,
    out: Path,
    steps: int = 20000,
    batch_size: int = 32,
) -> dict:
    torch.manual_seed(seed)
    model = TypedLambdaPrior()
    decay = [param for name, param in model.named_parameters() if name != "raw_lambda"]
    opt = torch.optim.AdamW(
        [{"params": decay}, {"params": [model.raw_lambda], "weight_decay": 0.0}],
        lr=1e-3,
        weight_decay=0.01,
    )
    syn_pool = synthetic["ambiguous"].nonzero(as_tuple=False).flatten()
    fac_pool = factory_train["ambiguous"].nonzero(as_tuple=False).flatten()
    print(
        f"seed {seed} synthetic ambiguous {int(syn_pool.numel())} "
        f"factory ambiguous {int(fac_pool.numel())} parameters {parameter_count(model)}",
        flush=True,
    )
    if int(syn_pool.numel()) == 0 or int(fac_pool.numel()) == 0:
        raise SystemExit("the mixture needs ambiguous rows on both sides")
    best = {"score": -1.0, "step": 0, "state": None}
    history = []
    step = 0
    passes = max(int(syn_pool.numel()), batch_size)
    while step < steps:
        for _ in range(0, passes, batch_size):
            batch = mix_batch(synthetic, factory_train, seed, syn_pool, fac_pool, batch_size)
            opt.zero_grad(set_to_none=True)
            loss = model.loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 2000 == 0 or step >= steps:
                synthetic_report = evaluate(model, synthetic_val, seed)
                factory_report = evaluate(model, factory_val, seed)
                score = selection_score(synthetic_report["top1"], factory_report["top1"])
                scales = _lambda_map(model)
                print(
                    f"seed {seed} step {step} select {score:.4f} "
                    f"syn {synthetic_report['top1']:.4f} factory {factory_report['top1']:.4f} "
                    f"lam {scales['charset']:.3f}/{scales['width']:.3f}/"
                    f"{scales['integer']:.3f}/{scales['repeat']:.3f}",
                    flush=True,
                )
                history.append(
                    {
                        "step": step,
                        "selection": score,
                        "val_top1": synthetic_report["top1"],
                        "factory_val_top1": factory_report["top1"],
                        "lambda": scales,
                    }
                )
                if score >= best["score"]:
                    best = {
                        "score": score,
                        "step": step,
                        "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    }
            if step >= steps:
                break
    model.load_state_dict(best["state"])
    synthetic_report = evaluate(model, synthetic_val, seed)
    factory_report = evaluate(model, factory_val, seed)
    report = {
        "seed": seed,
        "parameters": parameter_count(model),
        "best_step": best["step"],
        "selection": selection_score(synthetic_report["top1"], factory_report["top1"]),
        "lambda": _lambda_map(model),
        "val_top1": synthetic_report["top1"],
        "val_evidence_top1": synthetic_report["evidence_top1"],
        "val_by_family": synthetic_report["by_family"],
        "val_evidence_by_family": synthetic_report["evidence_by_family"],
        "val_gates": synthetic_report["gates"],
        "val_chose_expansion": synthetic_report["chose_expansion"],
        "factory_val_top1": factory_report["top1"],
        "factory_val_evidence_top1": factory_report["evidence_top1"],
        "factory_val_gates": factory_report["gates"],
        "factory_val_chose_expansion": factory_report["chose_expansion"],
        "history": history,
    }
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "report": report}, out / "checkpoint.pt")
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def finish_report(
    model_path: Path,
    seed: int,
    synthetic_val: dict,
    synthetic_ood: dict,
    heldout: dict,
) -> dict:
    saved = torch.load(model_path, weights_only=False)
    report = dict(saved["report"])
    model = TypedLambdaPrior()
    model.load_state_dict(saved["model"])
    ood = evaluate(model, synthetic_ood, seed)
    held = evaluate(model, heldout, seed)
    report["ood_top1"] = ood["top1"]
    report["ood_evidence_top1"] = ood["evidence_top1"]
    report["ood_by_family"] = ood["by_family"]
    report["ood_evidence_by_family"] = ood["evidence_by_family"]
    report["ood_gates"] = ood["gates"]
    report["ood_chose_expansion"] = ood["chose_expansion"]
    report["factory_heldout"] = {
        "top1": held["top1"],
        "evidence_top1": held["evidence_top1"],
        "n": held["n"],
        "gates": held["gates"],
        "chose_expansion": held["chose_expansion"],
    }
    report["heldout_slices"] = slice_program(model, heldout, seed)
    report["val_slices"] = slice_program(model, synthetic_val, seed)
    report["ood_slices"] = slice_program(model, synthetic_ood, seed)
    report["renaming_invariance"] = renaming_invariance(model)
    report["degraded_families"] = degraded_families(report)
    report["missed_slices"] = missed_slices(report["heldout_slices"])
    report["degraded_val_slices"] = degraded_slices(report["val_slices"])
    report["degraded_ood_slices"] = degraded_slices(report["ood_slices"])
    report["passed"] = seed_passes(report)
    report["retained"] = seed_retains(report)
    torch.save({"model": model.state_dict(), "report": report}, model_path)
    (model_path.parent / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"seed {seed} best {report['best_step']} select {report['selection']:.4f} "
        f"held {held['top1']:.4f} evidence {held['evidence_top1']:.4f} "
        f"pass {report['passed']} retain {report['retained']}",
        flush=True,
    )
    return report


def _cell(row: dict) -> str:
    if row["n"] == 0:
        return "n=0"
    return f"{row['combined']:.3f} vs {row['evidence']:.3f} (n={row['n']})"


def _family_rows(report: dict) -> list[str]:
    lines = []
    for split, family_key, evidence_key in (
        ("val", "val_by_family", "val_evidence_by_family"),
        ("ood", "ood_by_family", "ood_evidence_by_family"),
    ):
        for name in report.get(family_key, {}):
            evidence = report[evidence_key][name]
            combined = report[family_key][name]
            lines.append(
                f"| `{name}` | {split} | {_fmt(evidence)} | {_fmt(combined)} | {_fmt(combined - evidence)} |"
            )
    return lines


def render_markdown(payload: dict) -> str:
    seeds = payload["seeds"]
    left, right = seeds[0], seeds[1] if len(seeds) > 1 else seeds[0]
    lines = [
        "## Typed scales",
        "",
        "The heads and features stay the expansion prior's. Each ambiguity has its own softplus scale, initialized at 0.01. The program score is evidence plus the sum of scale times that head. A scale can shrink toward 0, and then that head leaves the ranking with the frozen evidence scorer.",
        "",
        "Training batches stay 16 synthetic ambiguous rows and 16 factory ambiguous rows. The loss, margin, hinge, optimizer, and 20,000-step budget are unchanged. The checkpoint maximizes the mean of synthetic validation top-1 and factory validation top-1. Factory held-out is scored after that choice.",
        "",
        f"Parameters {payload['parameters']}. "
        f"Scales at the selected checkpoints, charset / width / integer / repeat: "
        f"{_fmt(left['lambda']['charset'])} / {_fmt(left['lambda']['width'])} / {_fmt(left['lambda']['integer'])} / {_fmt(left['lambda']['repeat'])}"
        f" and {_fmt(right['lambda']['charset'])} / {_fmt(right['lambda']['width'])} / {_fmt(right['lambda']['integer'])} / {_fmt(right['lambda']['repeat'])}.",
        "",
        f"Selected factory validation top-1 {_fmt(left['factory_val_top1'])} / {_fmt(right['factory_val_top1'])} "
        f"against evidence {_fmt(left['factory_val_evidence_top1'])} / {_fmt(right['factory_val_evidence_top1'])}. "
        f"Synthetic validation {_fmt(left['val_top1'])} / {_fmt(right['val_top1'])} "
        f"against evidence {_fmt(left['val_evidence_top1'])} / {_fmt(right['val_evidence_top1'])}.",
        "",
        f"The gate requires both seeds. On factory held-out, program accuracy on each ambiguity slice is at least the evidence accuracy. Overall held-out accuracy, every synthetic family, and every synthetic ambiguity slice stay within {FAMILY_MARGIN:.2f} of evidence. Retention is separate: integer and loose-repeat held-out programs have to beat evidence.",
        "",
        f"Seed 0 {'clears' if left['passed'] else 'misses'} the gate and {'retains' if left['retained'] else 'does not retain'} integer and loose repeat. "
        f"Seed 1 {'clears' if right['passed'] else 'misses'} the gate and {'retains' if right['retained'] else 'does not retain'} integer and loose repeat.",
        "",
        "### Factory held-out programs",
        "",
        "The slice numbers are program top-1 on the rows in that gate, combined against evidence on the same rows.",
        "",
        "| Slice | Seed 0 | Seed 1 |",
        "| --- | ---: | ---: |",
        f"| All held-out | {_fmt(left['factory_heldout']['top1'])} vs {_fmt(left['factory_heldout']['evidence_top1'])} (n={left['factory_heldout']['n']}) | {_fmt(right['factory_heldout']['top1'])} vs {_fmt(right['factory_heldout']['evidence_top1'])} (n={right['factory_heldout']['n']}) |",
    ]
    for name in OWN_SLICES:
        lines.append(f"| `{name}` | {_cell(left['heldout_slices'][name])} | {_cell(right['heldout_slices'][name])} |")
    lines.extend(
        [
            "",
            "### Synthetic programs on the same slices",
            "",
            "A slice more than 0.02 under evidence is a miss.",
            "",
            "| Slice | Val seed 0 | Val seed 1 | OOD seed 0 | OOD seed 1 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in OWN_SLICES:
        lines.append(
            f"| `{name}` | {_cell(left['val_slices'][name])} | {_cell(right['val_slices'][name])} | "
            f"{_cell(left['ood_slices'][name])} | {_cell(right['ood_slices'][name])} |"
        )
    lines.extend(
        [
            "",
            "### Local gates",
            "",
            "Local accuracy is the head's own argmax. It does not pass through the scale. Chose-rate is how often that argmax is an expansion, or an open width.",
            "",
            "| Gate | Synthetic val seed 0 | seed 1 | Synthetic OOD seed 0 | seed 1 | Factory held-out seed 0 | seed 1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in OWN_SLICES:
        cells = []
        for report, split in (
            (left, "val"),
            (right, "val"),
            (left, "ood"),
            (right, "ood"),
        ):
            gate = report[f"{split}_gates"][name]
            cells.append(f"{gate['accuracy']:.3f} (n={gate['n']})" if gate["n"] else "n=0")
        for report in (left, right):
            gate = report["factory_heldout"]["gates"][name]
            cells.append(f"{gate['accuracy']:.3f} (n={gate['n']})" if gate["n"] else "n=0")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    lines.extend(["", "| Chose | Synthetic val seed 0 | seed 1 | Synthetic OOD seed 0 | seed 1 | Factory held-out seed 0 | seed 1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for name in OWN_SLICES:
        cells = []
        for report, split in ((left, "val"), (right, "val"), (left, "ood"), (right, "ood")):
            cells.append(_fmt(report[f"{split}_chose_expansion"][name]))
        for report in (left, right):
            cells.append(_fmt(report["factory_heldout"]["chose_expansion"][name]))
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "### Synthetic families",
            "",
            "Combined minus evidence. A family more than 0.02 under evidence is a miss.",
            "",
        ]
    )
    for report in (left, right):
        lines.append(f"Seed {report['seed']}")
        lines.append("")
        lines.append("| Family | Split | Evidence | Combined | Delta |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        lines.extend(_family_rows(report))
        lines.append("")
    dropped = []
    for report in (left, right):
        for row in report["degraded_families"]:
            dropped.append(
                f"seed {report['seed']} `{row['family']}` {row['split']} {_fmt(row['combined'])} vs evidence {_fmt(row['evidence'])}"
            )
        for row in report["missed_slices"]:
            dropped.append(
                f"seed {report['seed']} held-out `{row['slice']}` {_fmt(row['combined'])} vs evidence {_fmt(row['evidence'])} (n={row['n']})"
            )
        for label, key in (("val", "degraded_val_slices"), ("ood", "degraded_ood_slices")):
            for row in report[key]:
                dropped.append(
                    f"seed {report['seed']} {label} `{row['slice']}` {_fmt(row['combined'])} vs evidence {_fmt(row['evidence'])} (n={row['n']})"
                )
        held = report["factory_heldout"]
        if held["top1"] < held["evidence_top1"] - FAMILY_MARGIN:
            dropped.append(
                f"seed {report['seed']} held-out {_fmt(held['top1'])} vs evidence {_fmt(held['evidence_top1'])}"
            )
    lines.extend(
        [
            "### Gate",
            "",
            "Both seeds clear the gate and retain integer and loose repeat."
            if payload["passed"] and payload["retained"]
            else "Both seeds clear the gate. Integer and loose repeat do not both beat evidence."
            if payload["passed"]
            else "The gate is missed.",
            "",
        ]
    )
    if dropped:
        lines.append("Misses:")
        lines.append("")
        for row in dropped:
            lines.append(f"- {row}")
        lines.append("")
    lines.extend(
        [
            f"Letter-rename invariance {_fmt(left['renaming_invariance'])} / {_fmt(right['renaming_invariance'])}.",
            "",
            _reading(left, right),
            "",
            "BENCH-E stays sealed. `needle()` is unchanged. The shared-scale checkpoints are not retrained.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _reading(left: dict, right: dict) -> str:
    return "\n\n".join(
        [
            "### Reading",
            "The gate is missed. Separate scales do keep some of the factory signal, and they stop the open-width collapse from the shared scale.",
            "On factory held-out, integer programs beat evidence on both seeds, 0.556 against 0.500 / 0.463 (n=54). Loose-repeat programs beat evidence on seed 1 by one row, 0.100 against 0.000, and tie evidence at 0.000 on seed 0 (n=10). Charset and open-width programs on those held-out rows tie evidence. Overall held-out is 0.451 / 0.458 against evidence 0.472 / 0.458. Seed 1 matches evidence. Seed 0 is 0.021 under, which is past the 0.02 bar.",
            "The open-width head no longer drags the synthetic validation slice. Local open-width accuracy stays 0.632 / 0.632, where the shared scale had fallen to 0.525 / 0.404, and the chose-open rate on the synthetic slices stays 0. Charset scale stays small on seed 0 (0.020) and moderate on seed 1 (0.291). Width scale settles near 0.46 / 0.48. Integer scale settles near 1.13 / 1.32. Repeat scale grows to 2.11 / 3.23.",
            "The integer head still learns a window the synthetic gold does not use. Local integer accuracy on synthetic validation is 0.146 / 0.157. Program accuracy on that validation slice stays within 0.02 of evidence. On synthetic OOD the same slice falls to 0.480 / 0.450 against evidence 0.526 / 0.473, and `repeat_interval` falls with it. Seed 0 also drops `prefix_charset`, `prefix_lexicon`, and `prefix_optional` by just over 0.02. `repeat_width` and `charset_width` stay with evidence.",
            "Factory integer programs can be kept ahead of evidence without the shared scale's open-width damage. The repeat preference shows up in the head (held-out local accuracy 0.600) and barely in the chosen program. Synthetic OOD integer programs still move by more than 0.02. That is short of a v1 where every other family stays put, so the teacher bank stays at this size.",
        ]
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-cache", type=Path, default=Path("/tmp/expansion-prior"))
    parser.add_argument("--factory-cache", type=Path, default=Path("/tmp/factory-mix"))
    parser.add_argument("--out", type=Path, default=Path("results/factory/runs/typed"))
    parser.add_argument("--page", type=Path, default=Path("results/factory/typed.md"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--steps", type=int, default=20000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print("loading caches", flush=True)
    synthetic = {
        name: torch.load(args.synthetic_cache / f"{name}.pt", weights_only=False) for name in ("train", "val", "ood")
    }
    factory = {
        name: torch.load(args.factory_cache / f"{name}.pt", weights_only=False)
        for name in ("train", "validation", "heldout")
    }
    reports = []
    for seed in args.seeds:
        train_typed(
            seed,
            synthetic["train"],
            factory["train"],
            synthetic["val"],
            factory["validation"],
            args.out / f"seed_{seed}",
            steps=args.steps,
        )
        report = finish_report(
            args.out / f"seed_{seed}" / "checkpoint.pt",
            seed,
            synthetic["val"],
            synthetic["ood"],
            factory["heldout"],
        )
        reports.append(report)
    payload = {
        "parameters": reports[0]["parameters"],
        "passed": all(report["passed"] for report in reports),
        "retained": all(report["retained"] for report in reports),
        "seeds": reports,
        "init_lambda": INIT_LAMBDA,
        "family_margin": FAMILY_MARGIN,
        "selection": "mean of synthetic validation top-1 and factory validation top-1",
    }
    dest = args.page.with_suffix(".json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    if len(reports) >= 2:
        args.page.write_text(render_markdown(payload))
    print(f"wrote {dest} pass {payload['passed']}", flush=True)


if __name__ == "__main__":
    main()
