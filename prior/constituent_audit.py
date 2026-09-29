"""Probe constituent state instead of the pooled support vector.

The recognizer stays frozen. Each fact is read from typed class runs and,
separately, from the frozen span hidden of those runs. Training uses the
train families. The held-out compositions are the measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from prior.audit import _family_index, _labels, _mean_pair, _rate, load_rows
from prior.constituents import (
    NEURAL_DIM,
    STRUCTURE_DIM,
    class_blind,
    digit_constancy,
    encode_split,
    field_matrix,
    field_class,
    last_right_constant,
    load_encoder,
    numeric_coverage,
    repeat_summary,
    run_presence,
    suffix_role,
)
from prior.probe_labels import PROBE_CLASSES, PROBE_NAMES
from prior.probes import balanced_accuracy, parameter_count, predict, standardize, train_probe

# The three facts the pooled vector dropped, and where a composition must still show them.
GATES = {
    "interval": "repeat_interval",
    "optional_scope": "prefix_charset_optional",
    "affix": "lexicon_suffix",
}
GATE = 0.90
CONTROLS = {
    "repeat": ("val", "repeat_width"),
    "repeat_holdout": ("ood", "repeat_interval"),
    "charset": ("ood", "prefix_charset_optional"),
    "optional": ("ood", "prefix_charset_optional"),
    "lexicon": ("ood", "lexicon_suffix"),
}


def _view_tensor(structure: torch.Tensor, neural: torch.Tensor, view: str) -> torch.Tensor:
    if view == "structure":
        return structure
    if view == "neural":
        return neural
    if view == "typed":
        return torch.cat([structure, neural], dim=-1)
    raise ValueError(view)


def _probe_view(name: str, kind: str, train_x, train_y, val_x, val_y, ood_x, ood_y, val_index, ood_index, seeds) -> dict:
    n_class = len(PROBE_CLASSES[name])
    runs = []
    for seed in seeds:
        print(f"probe {name} {kind} seed {seed}", flush=True)
        model, meta = train_probe(kind, train_x, train_y, val_x, val_y, n_class, seed, epochs=20)
        pred_val = predict(model, val_x)
        pred_ood = predict(model, ood_x)
        runs.append(
            {
                "seed": seed,
                "parameters": meta["parameters"],
                "best_epoch": meta["best"]["epoch"],
                "val_balanced": balanced_accuracy(pred_val, val_y, n_class),
                "ood_balanced": balanced_accuracy(pred_ood, ood_y, n_class),
                "val": {family: _rate(pred_val, val_y, index) for family, index in sorted(val_index.items())},
                "ood": {family: _rate(pred_ood, ood_y, index) for family, index in sorted(ood_index.items())},
            }
        )
    return {
        "parameters": parameter_count(kind, n_class, int(train_x.shape[-1])),
        "seeds": runs,
        "val_balanced": _mean_pair(runs, "val_balanced"),
        "ood_balanced": _mean_pair(runs, "ood_balanced"),
    }


def _gate_row(block: dict, fact: str, family: str) -> dict:
    seeds = []
    for seed in block["seeds"]:
        stats = seed["ood"][family]
        seeds.append(stats["accuracy"])
    return {
        "seeds": seeds,
        "mean": sum(seeds) / len(seeds),
        "clears": all(value > GATE for value in seeds),
    }


def run(data_dir: Path, out: Path, seeds: tuple[int, ...], cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    splits = {}
    labels = {}
    encoded = {}
    encoder = None
    for split in ("train", "val", "ood"):
        path = cache / f"{split}.pt"
        rows = load_rows(data_dir, split, with_programs=False)
        splits[split] = rows
        labels[split] = _labels(rows["family"], rows["supports"])
        if path.exists():
            print(f"loading {path}", flush=True)
            saved = torch.load(path, weights_only=True)
            encoded[split] = saved
            continue
        if encoder is None:
            print("loading frozen recognizer", flush=True)
            encoder = load_encoder(Path("."))
        print(f"encoding {split} {len(rows['family'])}", flush=True)
        structure, neural = encode_split(encoder, rows["supports"])
        saved = {"structure": structure, "neural": neural}
        torch.save(saved, path)
        encoded[split] = saved
    report = {
        "structure_dim": STRUCTURE_DIM,
        "neural_dim": NEURAL_DIM,
        "gate": GATE,
        "gates": GATES,
        "views": {},
    }
    val_index = _family_index(splits["val"]["family"])
    ood_index = _family_index(splits["ood"]["family"])
    for view in ("structure", "neural", "typed"):
        print(f"view {view}", flush=True)
        train_x, val_x, ood_x = standardize(
            _view_tensor(encoded["train"]["structure"], encoded["train"]["neural"], view),
            _view_tensor(encoded["val"]["structure"], encoded["val"]["neural"], view),
            _view_tensor(encoded["ood"]["structure"], encoded["ood"]["neural"], view),
        )
        kinds = ("linear", "tiny") if view == "typed" else ("linear",)
        view_report = {"probes": {}}
        for name in PROBE_NAMES:
            view_report["probes"][name] = {}
            for kind in kinds:
                view_report["probes"][name][kind] = _probe_view(
                    name,
                    kind,
                    train_x,
                    labels["train"][name],
                    val_x,
                    labels["val"][name],
                    ood_x,
                    labels["ood"][name],
                    val_index,
                    ood_index,
                    seeds,
                )
        linear = {fact: _gate_row(view_report["probes"][fact]["linear"], fact, family) for fact, family in GATES.items()}
        view_report["held_out"] = linear
        view_report["controls"] = {}
        for label, (split, family) in CONTROLS.items():
            fact = "charset" if label == "charset" else label.split("_")[0]
            if label == "repeat_holdout":
                fact = "repeat"
            seeds_acc = [seed[split][family]["accuracy"] for seed in view_report["probes"][fact]["linear"]["seeds"]]
            view_report["controls"][label] = {"family": family, "split": split, "seeds": seeds_acc}
        report["views"][view] = view_report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    return report


def run_relations(data_dir: Path, out: Path, seeds: tuple[int, ...], cache: Path) -> dict:
    """Class-blind roles, and within-field column constancy, on the cached run layout."""
    report = json.loads(out.read_text()) if out.exists() else {"views": {}}
    labels, structure, field, families = _load_field(data_dir, cache)
    views = {
        "class_blind": {split: class_blind(structure[split]) for split in structure},
        "field": {split: torch.cat([structure[split], field[split]], dim=-1) for split in structure},
    }
    val_index = _family_index(families["val"])
    ood_index = _family_index(families["ood"])
    report.setdefault("views", {})
    for view, tensors in views.items():
        print(f"view {view}", flush=True)
        train_x, val_x, ood_x = standardize(tensors["train"], tensors["val"], tensors["ood"])
        view_report = {"probes": {}}
        for name in PROBE_NAMES:
            view_report["probes"][name] = {
                "linear": _probe_view(
                    name,
                    "linear",
                    train_x,
                    labels["train"][name],
                    val_x,
                    labels["val"][name],
                    ood_x,
                    labels["ood"][name],
                    val_index,
                    ood_index,
                    seeds,
                )
            }
        view_report["held_out"] = {
            fact: _gate_row(view_report["probes"][fact]["linear"], fact, family) for fact, family in GATES.items()
        }
        report["views"][view] = view_report
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    return report


def _load_field(data_dir: Path, cache: Path):
    labels = {}
    structure = {}
    field = {}
    families = {}
    for split in ("train", "val", "ood"):
        rows = load_rows(data_dir, split, with_programs=False)
        families[split] = rows["family"]
        labels[split] = _labels(rows["family"], rows["supports"])
        saved = torch.load(cache / f"{split}.pt", weights_only=True)
        structure[split] = saved["structure"]
        field_path = cache / f"{split}-field.pt"
        if field_path.exists():
            print(f"loading {field_path}", flush=True)
            field[split] = torch.load(field_path, weights_only=True)
        else:
            print(f"field relations {split}", flush=True)
            field[split] = field_matrix(rows["supports"])
            torch.save(field[split], field_path)
    return labels, structure, field, families


def run_slices(data_dir: Path, out: Path, seeds: tuple[int, ...], cache: Path) -> dict:
    """Read each held-out fact from the constituent that should carry it."""
    report = json.loads(out.read_text()) if out.exists() else {"views": {}}
    labels, structure, field, families = _load_field(data_dir, cache)
    views = {
        "digit_constancy": {split: digit_constancy(field[split]) for split in field},
        "suffix_role": {split: suffix_role(structure[split]) for split in structure},
        "right_constant": {split: last_right_constant(structure[split]) for split in structure},
    }
    val_index = _family_index(families["val"])
    ood_index = _family_index(families["ood"])
    report.setdefault("views", {})
    for view, tensors in views.items():
        print(f"view {view} dim {int(tensors['train'].shape[-1])}", flush=True)
        train_x, val_x, ood_x = standardize(tensors["train"], tensors["val"], tensors["ood"])
        view_report = {"probes": {}}
        for name in PROBE_NAMES:
            view_report["probes"][name] = {
                "linear": _probe_view(
                    name,
                    "linear",
                    train_x,
                    labels["train"][name],
                    val_x,
                    labels["val"][name],
                    ood_x,
                    labels["ood"][name],
                    val_index,
                    ood_index,
                    seeds,
                )
            }
        view_report["held_out"] = {
            fact: _gate_row(view_report["probes"][fact]["linear"], fact, family) for fact, family in GATES.items()
        }
        report["views"][view] = view_report
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    return report


def run_numeric(data_dir: Path, out: Path, seeds: tuple[int, ...], cache: Path) -> dict:
    """One number per support: the tightest digit constituent, not where it sits."""
    report = json.loads(out.read_text()) if out.exists() else {"views": {}}
    labels = {}
    families = {}
    coverage = {}
    for split in ("train", "val", "ood"):
        rows = load_rows(data_dir, split, with_programs=False)
        families[split] = rows["family"]
        labels[split] = _labels(rows["family"], rows["supports"])
        path = cache / f"{split}-numeric.pt"
        if path.exists():
            print(f"loading {path}", flush=True)
            coverage[split] = torch.load(path, weights_only=True)
        else:
            print(f"numeric coverage {split}", flush=True)
            coverage[split] = numeric_coverage(rows["supports"])
            torch.save(coverage[split], path)
    val_index = _family_index(families["val"])
    ood_index = _family_index(families["ood"])
    print(f"view numeric dim {int(coverage['train'].shape[-1])}", flush=True)
    train_x, val_x, ood_x = standardize(coverage["train"], coverage["val"], coverage["ood"])
    view_report = {"probes": {}}
    for name in PROBE_NAMES:
        view_report["probes"][name] = {
            "linear": _probe_view(
                name,
                "linear",
                train_x,
                labels["train"][name],
                val_x,
                labels["val"][name],
                ood_x,
                labels["ood"][name],
                val_index,
                ood_index,
                seeds,
            )
        }
    view_report["held_out"] = {
        fact: _gate_row(view_report["probes"][fact]["linear"], fact, family) for fact, family in GATES.items()
    }
    report.setdefault("views", {})
    report["views"]["numeric"] = view_report
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    return report


def run_compact(data_dir: Path, out: Path, seeds: tuple[int, ...], cache: Path) -> dict:
    """The slices the matcher is allowed to query, each checked on its own."""
    report = json.loads(out.read_text()) if out.exists() else {"views": {}}
    labels = {}
    families = {}
    structure = {}
    for split in ("train", "val", "ood"):
        rows = load_rows(data_dir, split, with_programs=False)
        families[split] = rows["family"]
        labels[split] = _labels(rows["family"], rows["supports"])
        saved = torch.load(cache / f"{split}.pt", weights_only=True)
        structure[split] = saved["structure"]
    views = {
        "repeat_summary": {split: repeat_summary(structure[split]) for split in structure},
        "run_presence": {split: run_presence(structure[split]) for split in structure},
        "field_class": {split: field_class(structure[split]) for split in structure},
    }
    val_index = _family_index(families["val"])
    ood_index = _family_index(families["ood"])
    report.setdefault("views", {})
    for view, tensors in views.items():
        print(f"view {view} dim {int(tensors['train'].shape[-1])}", flush=True)
        train_x, val_x, ood_x = standardize(tensors["train"], tensors["val"], tensors["ood"])
        view_report = {"probes": {}}
        for name in PROBE_NAMES:
            view_report["probes"][name] = {
                "linear": _probe_view(
                    name,
                    "linear",
                    train_x,
                    labels["train"][name],
                    val_x,
                    labels["val"][name],
                    ood_x,
                    labels["ood"][name],
                    val_index,
                    ood_index,
                    seeds,
                )
            }
        view_report["held_out"] = {
            fact: _gate_row(view_report["probes"][fact]["linear"], fact, family) for fact, family in GATES.items()
        }
        report["views"][view] = view_report
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("results/prior/shards"))
    parser.add_argument("--out", type=Path, default=Path("results/prior/constituents.json"))
    parser.add_argument("--cache", type=Path, default=Path("/tmp/constituent-state"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--relations", action="store_true")
    parser.add_argument("--slices", action="store_true")
    parser.add_argument("--numeric", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.compact:
        run_compact(args.data, args.out, tuple(args.seeds), args.cache)
    elif args.numeric:
        run_numeric(args.data, args.out, tuple(args.seeds), args.cache)
    elif args.slices:
        run_slices(args.data, args.out, tuple(args.seeds), args.cache)
    elif args.relations:
        run_relations(args.data, args.out, tuple(args.seeds), args.cache)
    else:
        run(args.data, args.out, tuple(args.seeds), args.cache)


if __name__ == "__main__":
    main()
