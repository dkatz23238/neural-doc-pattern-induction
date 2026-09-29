"""Representation sufficiency audit.

Train independent probes on the frozen 160-d support vector, and score the
same candidates by matching explicit string facts. The generator, the
encoder, and the candidate pool stay as they are.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch

from prior.explicit import choose, score_programs
from prior.probe_labels import (
    AFFIX_NAME,
    CHARSET_NAME,
    PROBE_CLASSES,
    PROBE_NAMES,
    SCOPE_NAME,
    WIDTH_NAME,
    family_flags,
    probe_targets,
    visible_facts,
)
from prior.probes import balanced_accuracy, parameter_count, predict, standardize, train_probe
from prior.support_encode import SUPPORT_DIM
from prior.tree import programs_from_json

HOLDOUTS = ("prefix_charset_optional", "repeat_interval", "lexicon_suffix")


def load_rows(data_dir: Path, split: str, with_programs: bool) -> dict:
    paths = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no {split} shards in {data_dir}")
    columns = ["family", "support", "supports"]
    if with_programs:
        columns.extend(["programs", "mask", "gold"])
    table = pq.read_table(paths, columns=columns)
    support = torch.tensor(table.column("support").to_pylist(), dtype=torch.float32)
    if support.shape[-1] != SUPPORT_DIM:
        raise RuntimeError(f"{split} support width {support.shape[-1]} is not {SUPPORT_DIM}")
    payload = {
        "family": table.column("family").to_pylist(),
        "support": support,
        "supports": table.column("supports").to_pylist(),
    }
    if with_programs:
        payload["programs"] = table.column("programs").to_pylist()
        payload["mask"] = table.column("mask").to_pylist()
        payload["gold"] = table.column("gold").to_pylist()
    return payload


def _labels(families: list[str], supports: list[list[str]]) -> dict[str, torch.Tensor]:
    columns = {name: [] for name in PROBE_NAMES}
    for family, group in zip(families, supports):
        target = probe_targets(family, group)
        for name in PROBE_NAMES:
            columns[name].append(target[name])
    return {name: torch.tensor(values, dtype=torch.long) for name, values in columns.items()}


def _family_index(families: list[str]) -> dict[str, torch.Tensor]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, family in enumerate(families):
        grouped[family].append(index)
    return {name: torch.tensor(rows, dtype=torch.long) for name, rows in grouped.items()}


def _rate(pred: torch.Tensor, target: torch.Tensor, index: torch.Tensor, label: int | None = None) -> dict:
    if index.numel() == 0:
        return {"n": 0, "accuracy": None, "recall": None}
    chosen = target[index]
    guessed = pred[index]
    report = {
        "n": int(index.numel()),
        "accuracy": float((guessed == chosen).float().mean()),
    }
    if label is None:
        report["recall"] = report["accuracy"]
        return report
    mask = chosen == label
    report["positives"] = int(mask.sum())
    report["recall"] = float((guessed[mask] == label).float().mean()) if int(mask.sum()) else None
    report["predicted_positive"] = float((guessed == label).float().mean())
    return report


def _mean_pair(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def run_probes(train: dict, val: dict, ood: dict, seeds: tuple[int, ...] = (0, 1)) -> tuple[dict, dict]:
    train_y = _labels(train["family"], train["supports"])
    val_y = _labels(val["family"], val["supports"])
    ood_y = _labels(ood["family"], ood["supports"])
    train_x, val_x, ood_x = standardize(train["support"], val["support"], ood["support"])
    val_index = _family_index(val["family"])
    ood_index = _family_index(ood["family"])
    report: dict = {"probes": {}, "seeds": list(seeds)}
    linear_seed0: dict = {}
    for name in PROBE_NAMES:
        n_class = len(PROBE_CLASSES[name])
        per_kind: dict = {}
        for kind in ("linear", "tiny"):
            runs = []
            for seed in seeds:
                print(f"probe {name} {kind} seed {seed}", flush=True)
                model, meta = train_probe(
                    kind,
                    train_x,
                    train_y[name],
                    val_x,
                    val_y[name],
                    n_class,
                    seed,
                )
                pred_val = predict(model, val_x)
                pred_ood = predict(model, ood_x)
                if kind == "linear" and seed == seeds[0]:
                    linear_seed0[name] = model
                runs.append(
                    {
                        "seed": seed,
                        "parameters": meta["parameters"],
                        "best_epoch": meta["best"]["epoch"],
                        "val_balanced": balanced_accuracy(pred_val, val_y[name], n_class),
                        "ood_balanced": balanced_accuracy(pred_ood, ood_y[name], n_class),
                        "val": {
                            family: _rate(pred_val, val_y[name], index)
                            for family, index in sorted(val_index.items())
                        },
                        "ood": {
                            family: _rate(pred_ood, ood_y[name], index)
                            for family, index in sorted(ood_index.items())
                        },
                    }
                )
            per_kind[kind] = {
                "parameters": parameter_count(kind, n_class),
                "seeds": runs,
                "val_balanced": _mean_pair(runs, "val_balanced"),
                "ood_balanced": _mean_pair(runs, "ood_balanced"),
            }
        report["probes"][name] = {
            "classes": list(PROBE_CLASSES[name]),
            "linear": per_kind["linear"],
            "tiny": per_kind["tiny"],
        }
    joints = {
        "model": "linear",
        "seed": seeds[0],
        "val": joint_readout(linear_seed0, val_x, val_y, val["family"]),
        "ood": joint_readout(linear_seed0, ood_x, ood_y, ood["family"]),
    }
    return report, joints


def joint_readout(models: dict[str, torch.nn.Module], features: torch.Tensor, labels: dict[str, torch.Tensor], families: list[str]) -> dict:
    pred = {name: predict(models[name], features) for name in models}
    index = _family_index(families)
    report = {}
    specs = {
        "repeat_interval": (("repeat", None), ("interval", None)),
        "prefix_charset_optional": (("optional", 1), ("optional_scope", 1), ("charset", None)),
        "lexicon_suffix": (("lexicon", 1), ("affix", 2)),
    }
    for family, pieces in specs.items():
        rows = index.get(family)
        if rows is None or rows.numel() == 0:
            report[family] = {"n": 0}
            continue
        ok = torch.ones(rows.numel(), dtype=torch.bool)
        detail = {}
        for name, _label in pieces:
            guess = pred[name][rows]
            truth = labels[name][rows]
            piece_ok = guess == truth
            ok = ok & piece_ok
            detail[name] = float(piece_ok.float().mean())
        report[family] = {"n": int(rows.numel()), "all_correct": float(ok.float().mean()), "pieces": detail}
    return report


def fact_agreement(split: dict) -> dict:
    """How often the string facts match the latent family bits. This checks the reader, not the vector."""
    totals: dict[str, list[float]] = defaultdict(list)
    by_family: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for family, supports in zip(split["family"], split["supports"]):
        flags = family_flags(family)
        seen = visible_facts(supports)
        checks = {
            "repeat": float(bool(seen["repeat"]) == bool(flags["repeat"])),
            "optional": float((seen["optional_scope"] == "prefix") == bool(flags["optional"])),
            "affix": float(AFFIX_NAME[flags["affix"]] == seen["affix"]),
        }
        for name, value in checks.items():
            totals[name].append(value)
            by_family[family][name].append(value)
    return {
        "overall": {name: sum(rows) / len(rows) for name, rows in totals.items()},
        "by_family": {
            family: {name: sum(rows) / len(rows) for name, rows in checks.items()}
            for family, checks in sorted(by_family.items())
        },
    }


def explicit_split(split: dict) -> dict:
    hits: list[float] = []
    tie_hits: list[float] = []
    by_family: dict[str, list[float]] = defaultdict(list)
    tie_family: dict[str, list[float]] = defaultdict(list)
    n = len(split["family"])
    for index in range(n):
        if index and index % 5000 == 0:
            print(f"explicit {index}/{n}", flush=True)
        programs = programs_from_json(split["programs"][index])
        supports = split["supports"][index]
        facts = visible_facts(supports)
        scores = score_programs(programs, supports, facts)
        mask = [bool(bit) for bit in split["mask"][index]]
        gold = [bool(bit) for bit in split["gold"][index]]
        width = min(len(scores), len(mask), len(gold))
        choice = choose(scores[:width], mask[:width])
        hit = 1.0 if gold[choice] else 0.0
        best = scores[choice]
        tied = [cursor for cursor in range(width) if mask[cursor] and scores[cursor] == best]
        tie_hit = 1.0 if any(gold[cursor] for cursor in tied) else 0.0
        family = split["family"][index]
        hits.append(hit)
        tie_hits.append(tie_hit)
        by_family[family].append(hit)
        tie_family[family].append(tie_hit)
    return {
        "n": n,
        "top1": sum(hits) / max(n, 1),
        "tie_hit": sum(tie_hits) / max(n, 1),
        "by_family": {name: sum(rows) / len(rows) for name, rows in sorted(by_family.items())},
        "tie_by_family": {name: sum(rows) / len(rows) for name, rows in sorted(tie_family.items())},
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("results/prior/shards"))
    parser.add_argument("--out", type=Path, default=Path("results/prior/probes.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()
    torch.set_num_threads(4)
    print("loading shards", flush=True)
    train = load_rows(args.data, "train", with_programs=False)
    val = load_rows(args.data, "val", with_programs=True)
    ood = load_rows(args.data, "ood", with_programs=True)
    print(f"rows train {len(train['family'])} val {len(val['family'])} ood {len(ood['family'])}", flush=True)
    agreement = {
        "train": fact_agreement(train),
        "val": fact_agreement(val),
        "ood": fact_agreement(ood),
    }
    print("fact agreement", json.dumps(agreement["train"]["overall"]), flush=True)
    probes, joints = run_probes(train, val, ood, seeds=tuple(args.seeds))
    print("explicit matching", flush=True)
    explicit = {"val": explicit_split(val), "ood": explicit_split(ood)}
    payload = {
        "support_dim": SUPPORT_DIM,
        "question": "Does the frozen 160-d support vector contain the factors a compositional prior would need?",
        "train_rows": len(train["family"]),
        "val_rows": len(val["family"]),
        "ood_rows": len(ood["family"]),
        "holdouts": list(HOLDOUTS),
        "fact_agreement": agreement,
        "probes": probes,
        "joints": joints,
        "explicit": explicit,
        "explicit_rule": (
            "Each of repeat, optional scope, affix, field class, width, and lexicon "
            "adds 1 when the candidate agrees with the string fact and subtracts 1 when it contradicts. "
            "A set that covers a pure observed class scores 0 on the field piece. Weights are not tuned."
        ),
        "class_names": {
            "optional_scope": list(SCOPE_NAME),
            "charset": list(CHARSET_NAME),
            "width": list(WIDTH_NAME),
            "affix": list(AFFIX_NAME[:3]),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
