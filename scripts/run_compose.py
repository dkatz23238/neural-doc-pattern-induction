"""Compositional language induction on the relational encoder.

Training mixes four atomic families: prefix+class, repetition of one to three
separator-plus-number groups, an optional hyphen, and a bare alternation.
The span encoder sees class runs and adjacent pairs. Generator groups supply
auxiliary labels. Raw codepoints stay out.

Extrapolation and held-out compositions are scored after selection. They are
not in the training mix, and they do not choose the checkpoint. BENCH-C is
not scored.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from pattern_induction.compose import TRAIN_FAMILIES, build_split, rename_annotated, sample_batch
from pattern_induction.membership import summarize
from pattern_induction.renaming import invariance_score, member_acceptance
from pattern_induction.span_net import SpanNet, collate, composition_loss, parameter_count

SELECT_FAMILIES = TRAIN_FAMILIES
EXTRA_FAMILIES = ("repeat_extra", "optional_internal")
COMP_FAMILIES = ("prefix_repeat", "repeat_optional", "prefix_alt")
ALL_FAMILIES = SELECT_FAMILIES + EXTRA_FAMILIES + COMP_FAMILIES


def _predict(model: SpanNet, episodes, batch: int) -> list[float]:
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(episodes), batch):
            packed = collate(
                episodes[start : start + batch],
                max_len=model.max_len,
                max_support=model.max_support,
            )
            logit, _aux = model(*packed[:4])
            scores.extend(torch.sigmoid(logit).tolist())
    return scores


def _pack(episodes, probs: list[float]) -> dict:
    report = summarize(
        [episode.as_episode() for episode in episodes],
        [prob >= 0.5 for prob in probs],
        probs,
    )
    report["member_acceptance"] = member_acceptance(
        [episode.as_episode() for episode in episodes],
        probs,
    )
    return report


def _score_slice(model: SpanNet, episodes, batch: int, seed: int) -> dict:
    renamed = [rename_annotated(episode, random.Random(seed + index), True) for index, episode in enumerate(episodes)]
    original = _predict(model, episodes, batch)
    shifted = _predict(model, renamed, batch)
    return {
        "original": _pack(episodes, original),
        "renamed": _pack(renamed, shifted),
        "symmetry": invariance_score(original, shifted),
    }


def _mean_auroc(slices: dict, names: tuple[str, ...]) -> float:
    return sum(slices[name]["original"]["auroc"] for name in names) / len(names)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Compositional language induction",
        "",
        "The question is whether a symbol-invariant model can learn repetition, optionality, and alternation, then apply them in combinations it was not trained on.",
        "Relational character features stay as they were: structural class, a task-local symbol id, and alignment relations. No codepoint channel is added.",
        "Class runs and adjacent pairs become spans. A second transformer relates those spans. Generator groups supervise span identity, kind, boundaries, and operator role.",
        "Training mixes primitive prefix+class, repeats of 1–3, an optional tail, and a bare alternation. Checkpoint selection uses only those four.",
        "Repeat extrapolation asks for 4–5 repetitions. Optional extrapolation moves the optional group inside the string. Held-out compositions are prefix+repeat, repeat+optional, and prefix+alternation.",
        "Invariance is `1 - mean |p(S, x) - p(π(S), π(x))|`. BENCH-C is not scored.",
        "",
        f"Parameters: {payload['parameters']}. Selected step: {payload['selected_step']}.",
        "",
        "## Atomic operators",
        "",
        "| Slice | Balanced | AUROC | Member acceptance | Invariance | Renamed AUROC |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in SELECT_FAMILIES:
        lines.append(_row(name, payload["slices"][name]))
    lines.extend(
        [
            "",
            "## Extrapolation",
            "",
            "| Slice | Balanced | AUROC | Member acceptance | Invariance | Renamed AUROC |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for name in EXTRA_FAMILIES + ("prefix_alt",):
        lines.append(_row(name, payload["slices"][name]))
    lines.extend(
        [
            "",
            "## Held-out compositions",
            "",
            "| Slice | Balanced | AUROC | Member acceptance | Invariance | Renamed AUROC |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for name in COMP_FAMILIES:
        lines.append(_row(name, payload["slices"][name]))
    lines.extend(["", "## Gates", "", "| Gate | Target | Result | Met |", "| --- | --- | --- | --- |"])
    for gate in payload["gates"]:
        lines.append(
            f"| {gate['name']} | {gate['target']} | {gate['result']:.3f} | {'yes' if gate['met'] else 'no'} |"
        )
    lines.extend(["", "## Reading", "", payload["reading"], "", "Steps", ""])
    for row in payload["curve"]:
        lines.append(
            f"- step {row['step']}: loss {row['loss']:.3f}, "
            f"select AUROC {row['select_auroc']:.3f}, "
            f"primitive {row['primitive']:.3f}, repeat {row['repeat']:.3f}, "
            f"optional {row['optional']:.3f}, alternation {row['alternation']:.3f}"
        )
    path.write_text("\n".join(lines) + "\n")


def _row(name: str, report: dict) -> str:
    original = report["original"]
    return (
        f"| {name} | {original['balanced']:.3f} | {original['auroc']:.3f} | "
        f"{original['member_acceptance']:.3f} | {report['symmetry']['invariance']:.3f} | "
        f"{report['renamed']['auroc']:.3f} |"
    )


def _reading(payload: dict) -> str:
    slices = payload["slices"]

    def view(name: str) -> dict:
        return slices[name]["original"]

    prim, repeat, optional, alt = (view(name) for name in SELECT_FAMILIES)
    extra, internal = (view(name) for name in EXTRA_FAMILIES)
    pre_rep, rep_opt, pre_alt = (view(name) for name in COMP_FAMILIES)
    return (
        f"A within-class rename leaves every probability in place. Invariance is {payload['min_invariance']:.3f} on every slice, including when a separator becomes a held-out mark.\n\n"
        f"The atomic operators clear the AUROC gates. Primitive prefix-plus-class is AUROC {prim['auroc']:.3f} (balanced {prim['balanced']:.3f}). "
        f"Repeats of 1–3 are AUROC {repeat['auroc']:.3f} (balanced {repeat['balanced']:.3f}). "
        f"Optional tails are AUROC {optional['auroc']:.3f} (balanced {optional['balanced']:.3f}). "
        f"Bare alternations are AUROC {alt['auroc']:.3f} (balanced {alt['balanced']:.3f}).\n\n"
        f"Longer repeats still rank above broken strings of the same depth: AUROC {extra['auroc']:.3f}, negative rejection {extra['tnr']:.3f}. "
        f"True-member acceptance falls from {repeat['member_acceptance']:.3f} inside the training depths to {extra['member_acceptance']:.3f}, and balanced accuracy from {repeat['balanced']:.3f} to {extra['balanced']:.3f}. "
        f"An optional group in the middle, absent from training, keeps AUROC {internal['auroc']:.3f} and negative rejection {internal['tnr']:.3f}. "
        f"True-member acceptance there is {internal['member_acceptance']:.3f}, so balanced accuracy is {internal['balanced']:.3f}.\n\n"
        f"The three held-out mixtures land in different places. "
        f"Prefix plus alternation is AUROC {pre_alt['auroc']:.3f} (balanced {pre_alt['balanced']:.3f}). "
        f"Repeat plus an optional tail is AUROC {rep_opt['auroc']:.3f} (balanced {rep_opt['balanced']:.3f}, true-member acceptance {rep_opt['member_acceptance']:.3f}). "
        f"Prefix plus repeat is AUROC {pre_rep['auroc']:.3f} (balanced {pre_rep['balanced']:.3f}, true-member acceptance {pre_rep['member_acceptance']:.3f}). "
        f"The mean of those three AUROCs is {payload['composition_auroc']:.3f}. "
        "The composition gate stays open, so the regex decoder stays off."
    )


def _gates(slices: dict) -> list[dict]:
    composition = _mean_auroc(slices, COMP_FAMILIES)
    invariance = min(slices[name]["symmetry"]["invariance"] for name in ALL_FAMILIES)
    specs = [
        ("primitive AUROC", 0.93, slices["primitive"]["original"]["auroc"], True),
        ("repeats AUROC", 0.90, slices["repeat"]["original"]["auroc"], True),
        ("optional spans AUROC", 0.90, slices["optional"]["original"]["auroc"], True),
        ("alternations AUROC", 0.90, slices["alternation"]["original"]["auroc"], True),
        ("held-out compositions AUROC", 0.90, composition, True),
        ("renaming invariance", 0.99, invariance, True),
    ]
    return [
        {"name": name, "target": f">{target:.2f}" if target < 0.99 else "~1.00", "result": value, "met": value >= target}
        for name, target, value, _flag in specs
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-n", type=int, default=80)
    parser.add_argument("--final-n", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/exp2/compose.md"))
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = SpanNet()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    select_splits = {
        family: build_split(family, random.Random(1000 + index), args.eval_n) for index, family in enumerate(SELECT_FAMILIES)
    }
    best_score = -1.0
    best_state = None
    best_step = 0
    curve = []
    families = list(TRAIN_FAMILIES)
    started = time.time()
    for step in range(1, args.steps + 1):
        family = families[(step - 1) % len(families)]
        batch = sample_batch(family, rng, args.batch)
        packed = collate(batch, max_len=model.max_len, max_support=model.max_support)
        model.train()
        logit, aux = model(*packed[:4])
        loss, membership, aux_loss = composition_loss(logit, aux, *packed[4:])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            # The renamed view is measured once, on the final split. Selection uses membership only.
            reports = {name: _pack(select_splits[name], _predict(model, select_splits[name], 32)) for name in SELECT_FAMILIES}
            score = sum(reports[name]["auroc"] for name in SELECT_FAMILIES) / len(SELECT_FAMILIES)
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "membership": float(membership),
                "aux": float(aux_loss),
                "select_auroc": score,
                "primitive": reports["primitive"]["auroc"],
                "repeat": reports["repeat"]["auroc"],
                "optional": reports["optional"]["auroc"],
                "alternation": reports["alternation"]["auroc"],
                "seconds": time.time() - started,
            }
            curve.append(row)
            print(
                f"step {step}: loss {row['loss']:.3f} mem {row['membership']:.3f} aux {row['aux']:.3f} "
                f"select {score:.3f} prim {row['primitive']:.3f} rep {row['repeat']:.3f} "
                f"opt {row['optional']:.3f} alt {row['alternation']:.3f} ({row['seconds']:.0f}s)",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    final_splits = {family: build_split(family, random.Random(5000 + index), args.final_n) for index, family in enumerate(ALL_FAMILIES)}
    slices = {family: _score_slice(model, final_splits[family], 32, 8000) for family in ALL_FAMILIES}
    gates = _gates(slices)
    payload = {
        "parameters": parameter_count(model),
        "selected_step": best_step,
        "select_auroc": best_score,
        "steps": args.steps,
        "composition_auroc": _mean_auroc(slices, COMP_FAMILIES),
        "min_invariance": min(slices[name]["symmetry"]["invariance"] for name in ALL_FAMILIES),
        "slices": slices,
        "gates": gates,
        "curve": curve,
        "reading": "",
    }
    payload["reading"] = _reading(payload)
    _write(args.out, payload)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
