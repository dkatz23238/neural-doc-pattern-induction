"""One comparison: the frozen expansion prior, with and without the concept bank.

The 1,158-parameter evidence scorer and the 1,765-parameter prior stay as
they are. λ, the evidence margin, the hinge, the optimizer, the batch size,
and the 20,000-step budget match ``prior.expansion``. Each batch is 16 rows
from the synthetic ambiguous pool and 16 rows from the factory ambiguous
pool. Checkpoints are chosen by synthetic validation top-1.

Factory rows are support configurations of accepted templates. The template
program is bound to those supports. A neighbor stays in the pool only when it
also accepts every support. A candidate is gold when it agrees with the
template program on the positives, the near misses, and the corner strings.
Whole concept families stay on the side the split file assigned.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from pattern_induction.factory import (
    FactoryError,
    _edge_strings,
    bind_program,
    family_key,
    load_templates,
    materialize,
)
from pattern_induction.program import _MEMBERSHIP, execute
from prior.expansion import (
    ExpansionPrior,
    _batch,
    _chose_cell,
    _draw,
    _fmt,
    _gate_cell,
    evaluate,
    freeze_evidence,
    pack_example as pack_expansion,
    parameter_count,
    renaming_invariance,
)
from prior.relative import LocalScorer, pack_example as pack_evidence

ROOT = Path(__file__).resolve().parents[1]
SPLIT_NAMES = ("train", "validation", "heldout")


def probe_texts(record: dict) -> list[str]:
    """Positives, near misses, and corner strings. Order is stable."""
    texts: list[str] = []

    def add(text: object) -> None:
        if isinstance(text, str) and text and text not in texts:
            texts.append(text)

    for text in record.get("positives") or []:
        add(text)
    for text in record.get("near_misses") or []:
        add(text)
    program = record.get("program")
    if isinstance(program, dict):
        for text in _edge_strings(program):
            add(text)
    for neighbor in record.get("neighbors") or []:
        if isinstance(neighbor, dict):
            for text in _edge_strings(neighbor):
                add(text)
    return texts


def label_candidates(record: dict, supports: list[str]) -> tuple[list, list[bool]] | None:
    """Template program first, then neighbors that also accept the supports.

    Returns None when the rebound template misses a positive, accepts a near
    miss, or has no neighbor that disagrees on the probe.
    """
    try:
        gold = bind_program(record["program"], supports)
    except FactoryError:
        return None
    positives = [text for text in record.get("positives") or [] if isinstance(text, str)]
    misses = [text for text in record.get("near_misses") or [] if isinstance(text, str)]
    if any(not execute(gold, text, supports) for text in list(supports) + positives):
        return None
    if any(execute(gold, text, supports) for text in misses):
        return None
    texts = probe_texts(record)
    truth = [execute(gold, text, supports) for text in texts]
    programs = [gold]
    flags = [True]
    for neighbor in record.get("neighbors") or []:
        if not isinstance(neighbor, dict):
            continue
        try:
            prog = bind_program(neighbor, supports)
        except FactoryError:
            continue
        if any(not execute(prog, text, supports) for text in supports):
            continue
        agreed = all(execute(prog, text, supports) == bit for text, bit in zip(texts, truth))
        programs.append(prog)
        flags.append(agreed)
    if len(programs) < 2 or all(flags):
        return None
    return programs, flags


def support_sets(record: dict, seed: int = 2507) -> list[tuple[str, list[str]]]:
    try:
        tasks = materialize(record, seed=seed, limit=8)
    except FactoryError:
        return []
    found = []
    for task in tasks:
        supports = [text for text in task.get("supports") or [] if isinstance(text, str)]
        if len(supports) >= 2:
            found.append((str(task.get("configuration") or ""), supports))
    return found


def _stack(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = [key for key, value in rows[0].items() if torch.is_tensor(value)]
    return {key: torch.stack([row[key] for row in rows]) for key in keys}


def load_evidence(runs: Path, seeds: list[int]) -> dict[int, LocalScorer]:
    models = {}
    for seed in seeds:
        path = runs / f"seed_{seed}" / "checkpoint.pt"
        saved = torch.load(path, weights_only=False)
        model = freeze_evidence(LocalScorer())
        model.load_state_dict(saved["model"])
        models[seed] = model
    return models


def _score(rows: list[dict[str, torch.Tensor]], model: LocalScorer, chunk: int = 64) -> torch.Tensor:
    pieces = []
    with torch.no_grad():
        for start in range(0, len(rows), chunk):
            group = rows[start : start + chunk]
            batch = {key: torch.stack([row[key] for row in group]) for key in group[0]}
            pieces.append(model.program_scores(batch).cpu())
    return torch.cat(pieces, dim=0)


def build_split(
    records: list[dict],
    ids: list[str],
    models: dict[int, LocalScorer],
    seed: int = 2507,
) -> dict | None:
    """Pack one side of the concept split. Ids outside ``ids`` are ignored."""
    allowed = set(ids)
    chosen = sorted((record for record in records if record.get("id") in allowed), key=lambda record: record["id"])
    packed: list[dict[str, torch.Tensor]] = []
    evidence_rows: list[dict[str, torch.Tensor]] = []
    template_ids: list[str] = []
    configurations: list[str] = []
    families: list[str] = []
    for record in chosen:
        _MEMBERSHIP.clear()
        for name, supports in support_sets(record, seed=seed):
            labeled = label_candidates(record, supports)
            if labeled is None:
                continue
            programs, gold_flags = labeled
            mask = [True] * len(programs)
            packed.append(pack_expansion(programs, gold_flags, mask, supports))
            evidence_rows.append(pack_evidence(programs, gold_flags, mask, supports))
            template_ids.append(str(record["id"]))
            configurations.append(name)
            families.append(family_key(record))
    if not packed:
        return None
    batch: dict = _stack(packed)
    batch["family"] = ["factory"] * len(packed)
    batch["template_id"] = template_ids
    batch["configuration"] = configurations
    batch["family_key"] = families
    batch["version"] = 1
    for model_seed, model in models.items():
        batch[f"evidence_{model_seed}"] = _score(evidence_rows, model)
    return batch


def split_counts(batch: dict | None) -> dict:
    if not batch:
        return {"rows": 0, "ambiguous": 0, "templates": 0, "configurations": {}}
    configurations: dict[str, int] = {}
    for name in batch["configuration"]:
        configurations[name] = configurations.get(name, 0) + 1
    return {
        "rows": len(batch["template_id"]),
        "ambiguous": int(batch["ambiguous"].sum()),
        "templates": len(set(batch["template_id"])),
        "configurations": configurations,
    }


def assert_disjoint(splits: dict[str, dict | None]) -> None:
    groups = {name: set(batch["family_key"]) for name, batch in splits.items() if batch}
    names = list(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = groups[left] & groups[right]
            if shared:
                shown = ", ".join(sorted(shared)[:5])
                raise SystemExit(f"concept families sit on two sides ({left}, {right}): {shown}")


def slice_counts(batch: dict, seed: int = 0) -> dict[str, int]:
    """How many rows land in each gate. The count does not depend on the weights."""
    report = evaluate(ExpansionPrior(), batch, seed)
    return {name: int(report["gates"][name]["n"]) for name in report["gates"]}


def mix_batch(
    synthetic: dict,
    factory: dict,
    seed: int,
    syn_pool: torch.Tensor,
    fac_pool: torch.Tensor,
    batch_size: int = 32,
) -> dict[str, torch.Tensor]:
    """16 synthetic rows and 16 factory rows. The loss then sees one batch of 32."""
    half = batch_size // 2
    left = _batch(synthetic, _draw(syn_pool, half), seed)
    right = _batch(factory, _draw(fac_pool, half), seed)
    keys = [key for key in left if key in right and torch.is_tensor(left[key])]
    return {key: torch.cat([left[key], right[key]], dim=0) for key in keys}


@torch.no_grad()
def exact_rates(model: ExpansionPrior, data: dict, seed: int, batch_size: int = 256) -> dict:
    """Share of rows whose chosen program is the template, which is candidate 0."""
    model.eval()
    exact: list[float] = []
    evidence_exact: list[float] = []
    count = len(data["family"])
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        batch = _batch(data, torch.arange(start, stop), seed)
        choice = model.combined_scores(batch).masked_fill(~batch["mask"], -1e9).argmax(dim=-1)
        evidence_choice = batch["evidence"].masked_fill(~batch["mask"], -1e9).argmax(dim=-1)
        exact.extend(float(bit) for bit in (choice == 0).tolist())
        evidence_exact.extend(float(bit) for bit in (evidence_choice == 0).tolist())
    total = float(len(exact) or 1)
    return {
        "exact_template": sum(exact) / total,
        "evidence_exact_template": sum(evidence_exact) / total,
        "n": len(exact),
    }


def _jsonable(report: dict) -> dict:
    encoded = json.loads(json.dumps(report))
    return encoded


def train_mixed(
    seed: int,
    synthetic: dict,
    factory: dict,
    val: dict,
    out: Path,
    steps: int = 20000,
    batch_size: int = 32,
    extras: dict[str, dict] | None = None,
) -> dict:
    torch.manual_seed(seed)
    model = ExpansionPrior()
    decay = [param for name, param in model.named_parameters() if name != "raw_lambda"]
    opt = torch.optim.AdamW(
        [{"params": decay}, {"params": [model.raw_lambda], "weight_decay": 0.0}],
        lr=1e-3,
        weight_decay=0.01,
    )
    syn_pool = synthetic["ambiguous"].nonzero(as_tuple=False).flatten()
    fac_pool = factory["ambiguous"].nonzero(as_tuple=False).flatten()
    print(
        f"seed {seed} synthetic ambiguous {int(syn_pool.numel())} "
        f"factory ambiguous {int(fac_pool.numel())} parameters {parameter_count(model)}",
        flush=True,
    )
    if int(syn_pool.numel()) == 0 or int(fac_pool.numel()) == 0:
        raise SystemExit("the mixture needs ambiguous rows on both sides")
    best = {"top1": -1.0, "step": 0, "state": None}
    history = []
    step = 0
    passes = max(int(syn_pool.numel()), batch_size)
    while step < steps:
        for _ in range(0, passes, batch_size):
            batch = mix_batch(synthetic, factory, seed, syn_pool, fac_pool, batch_size)
            opt.zero_grad(set_to_none=True)
            loss = model.loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 2000 == 0 or step >= steps:
                report = evaluate(model, val, seed)
                gates = report["gates"]
                print(
                    f"seed {seed} step {step} val {report['top1']:.4f} "
                    f"evidence {report['evidence_top1']:.4f} lam {report['lambda']:.3f} "
                    f"charset {gates['charset_expansion']['accuracy']:.3f} "
                    f"open {gates['open_width']['accuracy']:.3f} "
                    f"integer {gates['integer_expansion']['accuracy']:.3f} "
                    f"loose {gates['loose_repeat']['accuracy']:.3f}",
                    flush=True,
                )
                history.append(
                    {
                        "step": step,
                        "val_top1": report["top1"],
                        "val_evidence": report["evidence_top1"],
                        "lambda": report["lambda"],
                        "charset_expansion": gates["charset_expansion"]["accuracy"],
                        "open_width": gates["open_width"]["accuracy"],
                        "integer_expansion": gates["integer_expansion"]["accuracy"],
                        "loose_repeat": gates["loose_repeat"]["accuracy"],
                    }
                )
                if report["top1"] >= best["top1"]:
                    best = {
                        "top1": report["top1"],
                        "step": step,
                        "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    }
            if step >= steps:
                break
    model.load_state_dict(best["state"])
    val_report = evaluate(model, val, seed)
    report = {
        "seed": seed,
        "parameters": parameter_count(model),
        "evidence_parameters": int(synthetic.get("evidence_parameters", 0)) or parameter_count(LocalScorer()),
        "best_step": best["step"],
        "lambda": val_report["lambda"],
        "ambiguous_train": int(syn_pool.numel()),
        "factory_ambiguous": int(fac_pool.numel()),
        "train_n": len(synthetic["family"]),
        "factory_rows": len(factory["family"]),
        "val_top1": val_report["top1"],
        "val_evidence_top1": val_report["evidence_top1"],
        "val_by_family": val_report["by_family"],
        "val_evidence_by_family": val_report["evidence_by_family"],
        "val_gates": val_report["gates"],
        "val_chose_expansion": val_report["chose_expansion"],
        "val_flip_rate": val_report["flip_rate"],
        "val_flip_median_margin": val_report["flip_median_margin"],
        "history": history,
    }
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "report": report}, out / "checkpoint.pt")
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def finish_report(model_path: Path, seed: int, ood: dict, extras: dict[str, dict]) -> dict:
    saved = torch.load(model_path, weights_only=False)
    report = dict(saved["report"])
    model = ExpansionPrior()
    model.load_state_dict(saved["model"])
    ood_report = evaluate(model, ood, seed)
    report["ood_top1"] = ood_report["top1"]
    report["ood_evidence_top1"] = ood_report["evidence_top1"]
    report["ood_by_family"] = ood_report["by_family"]
    report["ood_evidence_by_family"] = ood_report["evidence_by_family"]
    report["ood_gates"] = ood_report["gates"]
    report["ood_chose_expansion"] = ood_report["chose_expansion"]
    report["ood_flip_rate"] = ood_report["flip_rate"]
    report["ood_flip_median_margin"] = ood_report["flip_median_margin"]
    report["renaming_invariance"] = renaming_invariance(model)
    for name, data in extras.items():
        held = evaluate(model, data, seed)
        held.update(exact_rates(model, data, seed))
        report[name] = _jsonable(held)
    torch.save({"model": saved["model"], "report": report}, model_path)
    (model_path.parent / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"seed {seed} best {report['best_step']} val {report['val_top1']:.4f} "
        f"ood {report['ood_top1']:.4f} rename {report['renaming_invariance']:.3f}",
        flush=True,
    )
    return report


def evaluate_saved(path: Path, data: dict, seed: int) -> dict:
    saved = torch.load(path, weights_only=False)
    model = ExpansionPrior()
    model.load_state_dict(saved["model"])
    report = evaluate(model, data, seed)
    report.update(exact_rates(model, data, seed))
    return _jsonable(report)


def _baseline_view(metrics: dict, heldout: dict, validation: dict) -> dict:
    keep = (
        "seed",
        "parameters",
        "evidence_parameters",
        "best_step",
        "lambda",
        "val_top1",
        "val_evidence_top1",
        "ood_top1",
        "ood_evidence_top1",
        "val_by_family",
        "ood_by_family",
        "val_evidence_by_family",
        "ood_evidence_by_family",
        "val_gates",
        "ood_gates",
        "val_chose_expansion",
        "ood_chose_expansion",
        "val_flip_rate",
        "ood_flip_rate",
        "renaming_invariance",
        "ambiguous_train",
        "train_n",
    )
    view = {key: metrics[key] for key in keep if key in metrics}
    view["factory_heldout"] = heldout
    view["factory_validation"] = validation
    return view


def _pair(left: dict, right: dict, key: str) -> str:
    return f"{_fmt(left.get(key, 0.0))} / {_fmt(right.get(key, 0.0))}"


def _held_cell(report: dict, field: str) -> str:
    held = report.get("factory_heldout") or {}
    if not held or int(held.get("n", 0)) == 0:
        return "n=0"
    if field in ("top1", "evidence_top1", "exact_template", "evidence_exact_template"):
        return f"{held[field]:.3f} (n={int(held['n'])})"
    gate = held.get("gates", {}).get(field)
    if not gate or gate.get("n", 0) == 0:
        return "n=0"
    return f"{gate['accuracy']:.3f} (n={gate['n']})"


def _held_chose(report: dict, field: str) -> str:
    held = report.get("factory_heldout") or {}
    gate = (held.get("gates") or {}).get(field) or {}
    if not gate or gate.get("n", 0) == 0:
        return "n=0"
    return _fmt((held.get("chose_expansion") or {}).get(field, 0.0))


def render_markdown(payload: dict) -> str:
    base = payload["baseline"]
    mix = payload["mixture"]
    left_b, right_b = base[0], base[1] if len(base) > 1 else base[0]
    left_m, right_m = mix[0], mix[1] if len(mix) > 1 else mix[0]
    counts = payload["factory_counts"]
    lines = [
        "## Factory mixture",
        "",
        "The evidence scorer stays the frozen 1,158-parameter model. The prior stays the 1,765-parameter expansion prior: same heads, same λ initialization, same evidence margin of 1, same hinge, same AdamW settings, same 20,000 steps, same batch of 32. Each batch is 16 rows from the synthetic ambiguous pool and 16 rows from the factory ambiguous pool. The checkpoint is the one with the highest synthetic validation top-1.",
        "",
        "A factory row is a support configuration of one accepted template. The template program and any neighbor that also accepts those supports form the pool. Gold programs are the ones that agree with the template on its positives, near misses, and corner strings. Families come from `data/factory/manifests/split_v1.json` and are not split.",
        "",
        f"Factory rows {counts['train']['rows']} train / {counts['validation']['rows']} validation / {counts['heldout']['rows']} held out. "
        f"Ambiguous rows {counts['train']['ambiguous']} / {counts['validation']['ambiguous']} / {counts['heldout']['ambiguous']}. "
        f"Templates {counts['train']['templates']} / {counts['validation']['templates']} / {counts['heldout']['templates']}.",
        "",
        f"Parameters {payload['parameters']}. Evidence parameters {payload['evidence_parameters']}. "
        f"λ at the mixture checkpoints {_fmt(left_m['lambda'])} / {_fmt(right_m['lambda'])}. "
        f"Synthetic-only λ {_fmt(left_b['lambda'])} / {_fmt(right_b['lambda'])}.",
        "",
        "### Gates",
        "",
        "Local accuracy is strict argmax over the compatible pool. The chose-an-expansion rate is the preference.",
        "",
        "| Gate | Split | Synthetic-only seed 0 | Synthetic-only seed 1 | Mixture seed 0 | Mixture seed 1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, label in (
        ("charset_expansion", "Charset expansion"),
        ("open_width", "Open width"),
        ("integer_expansion", "Integer-range expansion"),
        ("loose_repeat", "Loose repeat"),
    ):
        for split in ("val", "ood"):
            lines.append(
                f"| {label} | {split} | {_gate_cell(left_b, split, name)} | {_gate_cell(right_b, split, name)} | "
                f"{_gate_cell(left_m, split, name)} | {_gate_cell(right_m, split, name)} |"
            )
    lines.extend(
        [
            "",
            "| Slice | Split | Synthetic-only seed 0 | Synthetic-only seed 1 | Mixture seed 0 | Mixture seed 1 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, label in (
        ("charset_expansion", "Chose charset expansion"),
        ("open_width", "Chose open width"),
        ("integer_expansion", "Chose integer expansion"),
        ("loose_repeat", "Chose loose repeat"),
    ):
        for split in ("val", "ood"):
            lines.append(
                f"| {label} | {split} | {_chose_cell(left_b, split, name)} | {_chose_cell(right_b, split, name)} | "
                f"{_chose_cell(left_m, split, name)} | {_chose_cell(right_m, split, name)} |"
            )
    lines.extend(
        [
            "",
            "### Whole program on the synthetic sets",
            "",
            f"Validation top-1, synthetic-only {_pair(left_b, right_b, 'val_top1')} "
            f"against evidence {_pair(left_b, right_b, 'val_evidence_top1')}. "
            f"Mixture {_pair(left_m, right_m, 'val_top1')} "
            f"against evidence {_pair(left_m, right_m, 'val_evidence_top1')}.",
            "",
            f"OOD top-1, synthetic-only {_pair(left_b, right_b, 'ood_top1')} "
            f"against evidence {_pair(left_b, right_b, 'ood_evidence_top1')}. "
            f"Mixture {_pair(left_m, right_m, 'ood_top1')} "
            f"against evidence {_pair(left_m, right_m, 'ood_evidence_top1')}.",
            "",
            "| Family | Split | Synthetic-only seed 0 evidence | combined | seed 1 evidence | combined | Mixture seed 0 evidence | combined | seed 1 evidence | combined |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    families: list[tuple[str, str]] = []
    for split_name, family_key_name in (("val", "val_by_family"), ("ood", "ood_by_family")):
        for report in (left_b, right_b):
            for name in report.get(family_key_name, {}):
                item = (name, split_name)
                if item not in families:
                    families.append(item)
    for name, split_name in families:
        family_field = "val_by_family" if split_name == "val" else "ood_by_family"
        evidence_field = "val_evidence_by_family" if split_name == "val" else "ood_evidence_by_family"
        cells = []
        for report in (left_b, right_b, left_m, right_m):
            cells.append(report.get(evidence_field, {}).get(name, 0.0))
            cells.append(report.get(family_field, {}).get(name, 0.0))
        lines.append("| `" + name + "` | " + split_name + " | " + " | ".join(_fmt(value) for value in cells) + " |")
    lines.extend(
        [
            "",
            "### Held-out templates",
            "",
            "Probe-perfect top-1 counts a neighbor that agrees with the template on the probe. Exact template is candidate 0, the template program itself. Factory validation is reported and was not used to pick the checkpoint.",
            "",
            "| Measure | Synthetic-only seed 0 | Synthetic-only seed 1 | Mixture seed 0 | Mixture seed 1 |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| Probe-perfect top-1 | {_held_cell(left_b, 'top1')} | {_held_cell(right_b, 'top1')} | {_held_cell(left_m, 'top1')} | {_held_cell(right_m, 'top1')} |",
            f"| Evidence top-1 | {_held_cell(left_b, 'evidence_top1')} | {_held_cell(right_b, 'evidence_top1')} | {_held_cell(left_m, 'evidence_top1')} | {_held_cell(right_m, 'evidence_top1')} |",
            f"| Exact template | {_held_cell(left_b, 'exact_template')} | {_held_cell(right_b, 'exact_template')} | {_held_cell(left_m, 'exact_template')} | {_held_cell(right_m, 'exact_template')} |",
            f"| Evidence exact template | {_held_cell(left_b, 'evidence_exact_template')} | {_held_cell(right_b, 'evidence_exact_template')} | {_held_cell(left_m, 'evidence_exact_template')} | {_held_cell(right_m, 'evidence_exact_template')} |",
        ]
    )
    for name, label in (
        ("charset_expansion", "Charset expansion"),
        ("open_width", "Open width"),
        ("integer_expansion", "Integer-range expansion"),
        ("loose_repeat", "Loose repeat"),
    ):
        lines.append(
            f"| {label} | {_held_cell(left_b, name)} | {_held_cell(right_b, name)} | "
            f"{_held_cell(left_m, name)} | {_held_cell(right_m, name)} |"
        )
        lines.append(
            f"| Chose {label.lower()} | {_held_chose(left_b, name)} | {_held_chose(right_b, name)} | "
            f"{_held_chose(left_m, name)} | {_held_chose(right_m, name)} |"
        )
    lines.extend(
        [
            "",
            f"Letter-rename invariance, mixture {_fmt(left_m.get('renaming_invariance', 0.0))} / {_fmt(right_m.get('renaming_invariance', 0.0))}. "
            f"Synthetic-only {_fmt(left_b.get('renaming_invariance', 0.0))} / {_fmt(right_b.get('renaming_invariance', 0.0))}.",
            "",
            "BENCH-E stays sealed. `needle()` is unchanged. The synthetic-only checkpoints in `results/prior/runs/expansion_prior` are the comparison point and were not retrained.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def pack_bank(cache: Path, evidence_runs: Path, seeds: list[int]) -> dict[str, dict]:
    records = load_templates(ROOT)
    split = json.loads((ROOT / "data" / "factory" / "manifests" / "split_v1.json").read_text())
    models = load_evidence(evidence_runs, seeds)
    built: dict[str, dict | None] = {}
    for name in SPLIT_NAMES:
        print(f"packing factory {name}", flush=True)
        built[name] = build_split(records, split[name], models)
        print(f"factory {name} {split_counts(built[name])}", flush=True)
    assert_disjoint(built)
    cache.mkdir(parents=True, exist_ok=True)
    stored = {}
    for name, batch in built.items():
        if batch is None:
            raise SystemExit(f"factory {name} produced no rows")
        counts = slice_counts(batch, seeds[0])
        batch["slice_counts"] = counts
        print(f"factory {name} slices {counts}", flush=True)
        path = cache / f"{name}.pt"
        torch.save(batch, path)
        stored[name] = batch
    return stored


def _counts_payload(splits: dict[str, dict]) -> dict:
    payload = {}
    for name in SPLIT_NAMES:
        payload[name] = split_counts(splits[name])
        payload[name]["slices"] = dict(splits[name].get("slice_counts") or {})
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pack", "train", "all"), default="all")
    parser.add_argument("--cache", type=Path, default=Path("/tmp/factory-mix"))
    parser.add_argument("--synthetic-cache", type=Path, default=Path("/tmp/expansion-prior"))
    parser.add_argument("--evidence-runs", type=Path, default=Path("results/prior/runs/relative_evidence"))
    parser.add_argument("--baseline", type=Path, default=Path("results/prior/runs/expansion_prior"))
    parser.add_argument("--out", type=Path, default=Path("results/factory/runs/mix"))
    parser.add_argument("--page", type=Path, default=Path("results/factory/mix.md"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--steps", type=int, default=20000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    seeds = list(args.seeds)
    if args.stage in ("pack", "all"):
        factory = pack_bank(args.cache, args.evidence_runs, seeds)
    else:
        factory = {name: torch.load(args.cache / f"{name}.pt", weights_only=False) for name in SPLIT_NAMES}
    if args.stage == "pack":
        return
    print("loading synthetic cache", flush=True)
    synthetic = {
        name: torch.load(args.synthetic_cache / f"{name}.pt", weights_only=False) for name in ("train", "val", "ood")
    }
    extras = {"factory_heldout": factory["heldout"], "factory_validation": factory["validation"]}
    baseline = []
    for seed in seeds:
        metrics = json.loads((args.baseline / f"seed_{seed}" / "metrics.json").read_text())
        print(f"scoring synthetic-only seed {seed} on factory held-out", flush=True)
        heldout = evaluate_saved(args.baseline / f"seed_{seed}" / "checkpoint.pt", factory["heldout"], seed)
        validation = evaluate_saved(args.baseline / f"seed_{seed}" / "checkpoint.pt", factory["validation"], seed)
        baseline.append(_baseline_view(metrics, heldout, validation))
    reports = []
    for seed in seeds:
        report = train_mixed(
            seed,
            synthetic["train"],
            factory["train"],
            synthetic["val"],
            args.out / f"seed_{seed}",
            steps=args.steps,
        )
        report = finish_report(args.out / f"seed_{seed}" / "checkpoint.pt", seed, synthetic["ood"], extras)
        reports.append(report)
    payload = {
        "parameters": reports[0]["parameters"],
        "evidence_parameters": reports[0]["evidence_parameters"],
        "factory_counts": _counts_payload(factory),
        "baseline": baseline,
        "mixture": reports,
        "steps": args.steps,
        "batch": "16 synthetic ambiguous rows + 16 factory ambiguous rows",
        "selection": "synthetic validation top-1",
    }
    dest = args.page.with_suffix(".json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    if len(reports) >= 2 and len(baseline) >= 2:
        args.page.write_text(render_markdown(payload))
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
