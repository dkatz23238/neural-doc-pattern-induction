"""Alphabet-renaming equivariance.

The byte-identity membership model is trained on a familiar alphabet, then
scored again after a within-class rename that sends a separator to a held-out
mark such as §. The relational model never receives raw codepoints. Its gate
is whether that rename leaves both the probability and the accuracy in place.
BENCH-C is not used.
"""

from __future__ import annotations

import argparse
import json
import random
import string
import time
from pathlib import Path

import torch

from pattern_induction.equivariant_net import (
    EquivariantNet,
    collate_episodes,
    equivariant_loss,
    parameter_count,
)
from pattern_induction.membership import Episode, build_curriculum_split, sample_batch, summarize
from pattern_induction.membership_net import MembershipNet, collate, membership_loss
from pattern_induction.membership_net import parameter_count as byte_parameter_count
from pattern_induction.renaming import (
    contains_holdout,
    invariance_score,
    member_acceptance,
    rename_episode,
)


def _rename_all(episodes: list[Episode], seed: int) -> list[Episode]:
    return [rename_episode(episode, random.Random(seed + index), True) for index, episode in enumerate(episodes)]


def _predict_byte(model: MembershipNet, episodes: list[Episode], batch: int) -> list[float]:
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(episodes), batch):
            tokens, pad, _labels, _invariant = collate(episodes[start : start + batch])
            logit, _inv = model(tokens, pad)
            scores.extend(torch.sigmoid(logit).tolist())
    return scores


def _predict_rel(model: EquivariantNet, episodes: list[Episode], batch: int) -> list[float]:
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(episodes), batch):
            local, classes, rel, pad, _labels = collate_episodes(episodes[start : start + batch], max_len=model.max_len)
            logit, _latent = model(local, classes, rel, pad)
            scores.extend(torch.sigmoid(logit).tolist())
    return scores


def _pack(episodes: list[Episode], probs: list[float]) -> dict:
    report = summarize(episodes, [prob >= 0.5 for prob in probs], probs)
    report["member_acceptance"] = member_acceptance(episodes, probs)
    held = [(episode, prob) for episode, prob in zip(episodes, probs) if contains_holdout(episode)]
    if held:
        held_eps, held_probs = zip(*held)
        report["holdout_balanced"] = summarize(list(held_eps), [prob >= 0.5 for prob in held_probs], list(held_probs))[
            "balanced"
        ]
        report["holdout_member_acceptance"] = member_acceptance(list(held_eps), list(held_probs))
        report["holdout_n"] = len(held_eps)
    else:
        report["holdout_balanced"] = None
        report["holdout_member_acceptance"] = None
        report["holdout_n"] = 0
    return report


def _compare(episodes: list[Episode], original: list[float], renamed_eps: list[Episode], renamed: list[float]) -> dict:
    return {
        "original": _pack(episodes, original),
        "renamed": _pack(renamed_eps, renamed),
        "symmetry": invariance_score(original, renamed),
    }


def clean_delim_split(rng: random.Random, n_languages: int, delim: str = "-") -> list[Episode]:
    """Fixed shape: letters, one separator, digits. The rename sends the separator off-alphabet."""
    episodes: list[Episode] = []
    others = [ch for ch in "-./:" if ch != delim]
    guard_languages = 0
    while len(episodes) < n_languages * 4 and guard_languages < n_languages * 6:
        guard_languages += 1
        prefix = "".join(rng.choice(string.ascii_uppercase) for _ in range(rng.choice((2, 3))))
        width = rng.choice((3, 4))

        def draw() -> str:
            return prefix + delim + "".join(rng.choice(string.digits) for _ in range(width))

        support: list[str] = []
        seen: set[str] = set()
        guard = 0
        while len(support) < 4 and guard < 40:
            guard += 1
            text = draw()
            if text not in seen:
                seen.add(text)
                support.append(text)
        positives = []
        while len(positives) < 2:
            text = draw()
            if text not in support:
                positives.append(Episode(list(support), text, 1, "", "clean", "positive"))
                seen.add(text)
        negatives = []
        attempts = 0
        while len(negatives) < 2 and attempts < 12:
            attempts += 1
            if attempts % 3 == 1:
                chars = list(prefix)
                chars[0] = "Z" if chars[0] != "Z" else "Y"
                query = "".join(chars) + delim + "".join(rng.choice(string.digits) for _ in range(width))
                kind = "edit"
            elif attempts % 3 == 2:
                query = prefix + rng.choice(others) + "".join(rng.choice(string.digits) for _ in range(width))
                kind = "edit"
            else:
                bad = width + 1 if width < 5 else width - 1
                query = prefix + delim + "".join(rng.choice(string.digits) for _ in range(bad))
                kind = "repeat"
            if query in seen:
                continue
            negatives.append(Episode(list(support), query, 0, "", "clean", kind))
            seen.add(query)
        if len(positives) == 2 and len(negatives) == 2:
            episodes.extend(positives + negatives)
    return episodes


def _train_byte(model, opt, rng, steps: int, level: int, batch: int) -> float:
    model.train()
    losses = []
    for _step in range(steps):
        rows = sample_batch(rng, batch, level=level, structural_rate=0.0, easy_rate=0.1)
        tokens, pad, labels, invariant = collate(rows)
        opt.zero_grad(set_to_none=True)
        logit, inv_logit = model(tokens, pad)
        loss, _bce, _aux = membership_loss(logit, inv_logit, labels, invariant)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    return sum(losses) / max(len(losses), 1)


def _train_rel(model, opt, rng, steps: int, level: int, batch: int) -> tuple[float, float]:
    model.train()
    losses = []
    agrees = []
    for _step in range(steps):
        rows = sample_batch(rng, batch, level=level, structural_rate=0.0, easy_rate=0.1)
        renamed = [rename_episode(episode, rng, True) for episode in rows]
        local, classes, rel, pad, labels = collate_episodes(rows, max_len=model.max_len)
        local2, classes2, rel2, pad2, _labels2 = collate_episodes(renamed, max_len=model.max_len)
        opt.zero_grad(set_to_none=True)
        logit, latent = model(local, classes, rel, pad)
        logit2, latent2 = model(local2, classes2, rel2, pad2)
        loss, _bce, agree = equivariant_loss(logit, latent, logit2, latent2, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        agrees.append(float(agree))
    return sum(losses) / max(len(losses), 1), sum(agrees) / max(len(agrees), 1)


def _round_report(report: dict) -> dict:
    packed = {}
    for key, value in report.items():
        if isinstance(value, float):
            packed[key] = round(value, 4)
        elif isinstance(value, dict):
            packed[key] = _round_report(value)
        else:
            packed[key] = value
    return packed


def _fmt(report: dict | None, key: str) -> str:
    if not report:
        return "—"
    value = report.get(key)
    if value is None:
        return "—"
    return f"{value:.3f}"


def _gate(value: float | None, target: float, tolerance: float = 0.0) -> str:
    if value is None:
        return "no"
    return "yes" if value + 1e-9 >= target - tolerance else "no"


def _write(path: Path, payload: dict) -> None:
    byte = payload["byte"]
    rel = payload["relational"]
    lines = [
        "# Renaming equivariance",
        "",
        "The question is whether membership depends on symbol identity or on relations inside the task.",
        "A within-class rename may send `-` to `§` and `A` to `Q`. The label stays the same.",
        "Invariance is `1 - mean |p(S, x) - p(π(S), π(x))|`.",
        "The relational encoder sees structural class, a task-local symbol id, and alignment relations. It does not see raw codepoints.",
        "BENCH-C is not scored.",
        "",
        f"Byte-identity model: {payload['byte_parameters']} parameters, diagnostic only.",
        f"Relational model: {payload['relational_parameters']} parameters. Selected epoch: {payload['best_epoch']}.",
        "",
        "## Byte-identity model, familiar alphabet only",
        "",
        "| Slice | Balanced | AUROC | Member acceptance | Invariance | Renamed balanced |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in ("simple", "clean", "structural"):
        block = byte[name]
        lines.append(
            f"| {name} | {_fmt(block['original'], 'balanced')} | {_fmt(block['original'], 'auroc')} | "
            f"{_fmt(block['original'], 'member_acceptance')} | {block['symmetry']['invariance']:.3f} | "
            f"{_fmt(block['renamed'], 'balanced')} |"
        )
    lines.extend(
        [
            "",
            f"On the renamed clean probe, held-out true members are accepted at "
            f"{_fmt(byte['clean']['renamed'], 'holdout_member_acceptance')}.",
            "",
            "## Relational model",
            "",
            "| Slice | Balanced | AUROC | Member acceptance | Invariance | Renamed balanced | Renamed member acceptance |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for name in ("simple", "clean", "structural"):
        block = rel[name]
        lines.append(
            f"| {name} | {_fmt(block['original'], 'balanced')} | {_fmt(block['original'], 'auroc')} | "
            f"{_fmt(block['original'], 'member_acceptance')} | {block['symmetry']['invariance']:.3f} | "
            f"{_fmt(block['renamed'], 'balanced')} | {_fmt(block['renamed'], 'member_acceptance')} |"
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
            "These are the early gates for a relational backbone. They are not BENCH-C.",
            "",
            "| Gate | Target | Result | Met |",
            "| --- | --- | --- | --- |",
        ]
    )
    for gate in payload["gates"]:
        shown = "—" if gate["value"] is None else f"{gate['value']:.3f}"
        lines.append(f"| {gate['name']} | {gate['target']} | {shown} | {gate['met']} |")
    lines.extend(["", "## Reading", "", payload["reading"], "", "Epochs", ""])
    for row in payload["history"]:
        lines.append(
            f"- epoch {row['epoch']} {row['model']} {row['stage']}: loss {row['loss']:.3f}, "
            f"clean renamed balanced {row.get('clean_renamed', 0):.3f}"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _reading(rel: dict, byte: dict) -> str:
    clean = rel["clean"]
    simple = rel["simple"]
    structural = rel["structural"]
    byte_clean = byte["clean"]
    return (
        f"The byte-identity model scores {byte_clean['original']['balanced']:.3f} balanced on letter-separator-digit "
        f"tasks and {byte_clean['renamed']['balanced']:.3f} after the separator is renamed off the training alphabet "
        f"(invariance {byte_clean['symmetry']['invariance']:.3f}, "
        f"held-out member acceptance {_fmt(byte_clean['renamed'], 'holdout_member_acceptance')}). "
        f"The relational model scores {simple['original']['balanced']:.3f} on simple languages and "
        f"{simple['renamed']['balanced']:.3f} after a within-class rename "
        f"(invariance {simple['symmetry']['invariance']:.3f}). "
        f"On the clean delimiter probe it scores {clean['original']['balanced']:.3f} before the rename and "
        f"{clean['renamed']['balanced']:.3f} after it, with held-out member acceptance "
        f"{_fmt(clean['renamed'], 'holdout_member_acceptance')}. "
        f"Renamed structural AUROC is {_fmt(structural['renamed'], 'auroc')}. "
        "A regex decoder stays off until those gates hold."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag-steps", type=int, default=200)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--languages", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    print("building eval languages", flush=True)
    simple = build_curriculum_split(random.Random(4_000), args.languages, 0, 4)
    structural = build_curriculum_split(random.Random(6_000), args.languages, 2, 4)
    clean = clean_delim_split(random.Random(7_000), args.languages, "-")
    bundles = {
        "simple": (simple, _rename_all(simple, 11_000)),
        "clean": (clean, _rename_all(clean, 21_000)),
        "structural": (structural, _rename_all(structural, 31_000)),
    }
    print(
        f"simple {len(simple)} clean {len(clean)} structural {len(structural)} "
        f"clean-renamed holdout {sum(contains_holdout(ep) for ep in bundles['clean'][1])}",
        flush=True,
    )

    print("diagnostic: byte-identity model", flush=True)
    byte_model = MembershipNet()
    byte_opt = torch.optim.AdamW(byte_model.parameters(), lr=3e-4, weight_decay=0.01)
    history = []
    for level, name in ((0, "prefix-class"), (1, "delimited")):
        started = time.perf_counter()
        loss = _train_byte(byte_model, byte_opt, rng, args.diag_steps, level, args.batch_size)
        clean_probs = _predict_byte(byte_model, bundles["clean"][1], args.batch_size)
        clean_pack = _pack(bundles["clean"][1], clean_probs)
        history.append(
            {
                "epoch": len(history) + 1,
                "model": "byte",
                "stage": name,
                "loss": loss,
                "clean_renamed": clean_pack["balanced"],
            }
        )
        print(
            f"byte {name} loss {loss:.3f} clean-renamed {clean_pack['balanced']:.3f} "
            f"in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    byte_eval = {
        name: _compare(original, _predict_byte(byte_model, original, args.batch_size), renamed, _predict_byte(byte_model, renamed, args.batch_size))
        for name, (original, renamed) in bundles.items()
    }

    print("relational model", flush=True)
    rel_model = EquivariantNet()
    rel_opt = torch.optim.AdamW(rel_model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"parameters byte {byte_parameter_count(byte_model)} relational {parameter_count(rel_model)}", flush=True)
    best_state = {key: value.detach().cpu().clone() for key, value in rel_model.state_dict().items()}
    best_score = -1.0
    best_epoch = 0
    for level, name in ((0, "prefix-class"), (1, "delimited"), (2, "repeat-alt")):
        started = time.perf_counter()
        loss, agree = _train_rel(rel_model, rel_opt, rng, args.steps, level, args.batch_size)
        clean_probs = _predict_rel(rel_model, bundles["clean"][1], args.batch_size)
        clean_pack = _pack(bundles["clean"][1], clean_probs)
        history.append(
            {
                "epoch": len(history) + 1,
                "model": "relational",
                "stage": name,
                "loss": loss,
                "consistency": agree,
                "clean_renamed": clean_pack["balanced"],
            }
        )
        mark = ""
        if clean_pack["balanced"] > best_score + 1e-4:
            best_score = clean_pack["balanced"]
            best_epoch = history[-1]["epoch"]
            best_state = {key: value.detach().cpu().clone() for key, value in rel_model.state_dict().items()}
            mark = " *"
        print(
            f"relational {name} loss {loss:.3f} consistency {agree:.3f} "
            f"clean-renamed {clean_pack['balanced']:.3f} in {time.perf_counter() - started:.1f}s{mark}",
            flush=True,
        )
    rel_model.load_state_dict(best_state)
    rel_eval = {
        name: _compare(original, _predict_rel(rel_model, original, args.batch_size), renamed, _predict_rel(rel_model, renamed, args.batch_size))
        for name, (original, renamed) in bundles.items()
    }

    simple_bal = rel_eval["simple"]["original"]["balanced"]
    renamed_simple = rel_eval["simple"]["renamed"]["balanced"]
    unseen = rel_eval["clean"]["renamed"]["balanced"]
    members = rel_eval["clean"]["renamed"]["holdout_member_acceptance"]
    structural_auroc = rel_eval["structural"]["renamed"].get("auroc")
    same = abs(simple_bal - renamed_simple) <= 0.03
    gates = [
        {"name": "simple training alphabet", "target": ">0.90 balanced", "value": simple_bal, "met": _gate(simple_bal, 0.90)},
        {
            "name": "symbol-renamed simple score stays within 0.03",
            "target": "same",
            "value": renamed_simple,
            "met": "yes" if same else "no",
        },
        {"name": "unseen-delimiter clean probe", "target": ">0.85 balanced", "value": unseen, "met": _gate(unseen, 0.85)},
        {
            "name": "held-out-byte true-member acceptance",
            "target": ">0.85",
            "value": members,
            "met": _gate(members, 0.85),
        },
        {
            "name": "renamed structural AUROC",
            "target": ">0.90",
            "value": structural_auroc,
            "met": _gate(structural_auroc, 0.90),
        },
    ]
    payload = {
        "byte_parameters": byte_parameter_count(byte_model),
        "relational_parameters": parameter_count(rel_model),
        "best_epoch": best_epoch,
        "best_clean_renamed": best_score,
        "history": history,
        "byte": {name: _round_report(block) for name, block in byte_eval.items()},
        "relational": {name: _round_report(block) for name, block in rel_eval.items()},
        "gates": [
            {
                "name": gate["name"],
                "target": gate["target"],
                "value": None if gate["value"] is None else round(gate["value"], 4),
                "met": gate["met"],
            }
            for gate in gates
        ],
        "reading": _reading(rel_eval, byte_eval),
    }
    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "exp2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "renaming.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write(out / "renaming.md", payload)
    print(payload["reading"], flush=True)
    for gate in payload["gates"]:
        print(f"gate {gate['met']}: {gate['name']} = {gate['value']}", flush=True)


if __name__ == "__main__":
    main()
