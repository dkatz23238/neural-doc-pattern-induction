"""Stage 1: does a small set-conditioned model learn the language?

The model sees a few positive strings and a query, and predicts membership.
It trains on synthetic schemas whose delimiter bytes stay in a closed set.
Selection uses a shifted split: held-out delimiters, longer spans, and the
harder constructions. BENCH-C is scored once and does not stop training.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from pattern_induction.membership import (
    HELD_OUT,
    build_split,
    decisions_for,
    disagreement,
    heuristic_pattern,
    rules_pattern,
    sample_batch,
    summarize,
)
from pattern_induction.membership_net import MembershipNet, collate, membership_loss, parameter_count
from pattern_induction.tasks import read_jsonl


def _accept(scores: list[float], threshold: float = 0.5) -> list[bool]:
    return [score >= threshold for score in scores]


def _predict(model: MembershipNet, episodes, batch: int) -> list[float]:
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(episodes), batch):
            tokens, pad, _labels, _invariant = collate(episodes[start : start + batch])
            logit, _inv = model(tokens, pad)
            scores.extend(torch.sigmoid(logit).tolist())
    return scores


def _round(report: dict) -> dict:
    out = {}
    for key, value in report.items():
        if isinstance(value, float):
            out[key] = round(value, 4)
        elif key == "kinds":
            out[key] = {
                name: {"n": item["n"], "accuracy": round(item["accuracy"], 4)}
                for name, item in value.items()
            }
        else:
            out[key] = value
    return out


def _bench_episodes(tasks, max_len_bytes: int):
    from pattern_induction.membership import Episode, fits

    episodes = []
    skipped = 0
    for task in tasks:
        support = [text for text in task.visible if text][:6]
        if len(support) < 2:
            continue
        queries = [(text, 1, "positive") for text in task.hidden_positives]
        queries += [(text, 0, "negative") for text in task.hidden_negatives]
        for text, label, kind in queries:
            if not fits(text) or len(text.encode("utf-8")) > max_len_bytes:
                skipped += 1
                continue
            if any(len(item.encode("utf-8")) > max_len_bytes for item in support):
                skipped += 1
                continue
            episodes.append(Episode(support, text, label, task.gold_regex, "bench_c", kind))
    return episodes, skipped


def _line(name: str, report: dict) -> str:
    auroc = report.get("auroc")
    extra = f", AUROC {auroc:.3f}" if auroc is not None else ""
    return (
        f"| {name} | {report['accuracy']:.3f} | {report['balanced']:.3f} | "
        f"{report['tpr']:.3f} | {report['tnr']:.3f} | {report['n']}{extra} |"
    )


def _write_markdown(path: Path, payload: dict) -> None:
    ood = payload["ood"]
    lines = [
        "# Membership backbone",
        "",
        "Stage 1 asks whether a small model can represent the language behind a few strings.",
        "The input is the support set and one query. The output is the probability that the query belongs.",
        "Training schemas never contain `#`, `~`, `*`, `=`, or `§`. The selection split uses those bytes, longer classes, and the harder constructions (repeats, alternations, optional spans).",
        "Bytes keep a frozen identity code, so an unseen delimiter still matches itself, and a pointer copies the aligned support state onto the query.",
        "An auxiliary head marks literal and delimiter bytes on positive queries. There is no regex decoder in this run.",
        "",
        f"Trainable parameters: {payload['parameters']}. Best epoch: {payload['best_epoch']}.",
        "The checkpoint is the one with the highest shifted balanced accuracy. BENCH-C was not used to stop.",
        "",
        "## Shifted holdout",
        "",
        "| System | Accuracy | Balanced | TPR | TNR | N |",
        "| --- | --- | --- | --- | --- | --- |",
        _line("membership model", ood["model"]),
        _line("structural heuristic", ood["heuristic"]),
        _line("code-length rule", ood["rules"]),
        "",
        "Balanced accuracy is (TPR + TNR) / 2, the same shape as semantic accuracy.",
        "",
        f"On the {ood['held_out']['model']['n']} queries that actually contain a held-out byte, "
        f"the model balanced score is {ood['held_out']['model']['balanced']:.3f} "
        f"(AUROC {ood['held_out']['model'].get('auroc', 0):.3f}) and the heuristic is "
        f"{ood['held_out']['heuristic']['balanced']:.3f}.",
        "",
        f"Where the heuristic is wrong ({ood['versus_heuristic']['heuristic_errors']} queries), "
        f"the model is right {ood['versus_heuristic']['recovery']:.1%} of the time. "
        f"Where the heuristic is right ({ood['versus_heuristic']['heuristic_correct']} queries), "
        f"the model stays right {ood['versus_heuristic']['preservation']:.1%} of the time.",
        "",
        f"The same comparison against the code-length rule: recovery "
        f"{ood['versus_rules']['recovery']:.1%} of {ood['versus_rules']['heuristic_errors']} errors, "
        f"preservation {ood['versus_rules']['preservation']:.1%}.",
        "",
        "## Few-shot and in-distribution",
        "",
        f"Shifted languages shown with only 2 supports: model {payload['ood_n2']['model']['balanced']:.3f}, "
        f"heuristic {payload['ood_n2']['heuristic']['balanced']:.3f}, "
        f"rule {payload['ood_n2']['rules']['balanced']:.3f}.",
        "",
        f"In-distribution schemas, not used for selection: model {payload['id']['model']['balanced']:.3f}, "
        f"heuristic {payload['id']['heuristic']['balanced']:.3f}.",
        "",
        "## BENCH-C probe",
        "",
        payload["bench_c"]["note"],
        "",
        "| System | Accuracy | Balanced | TPR | TNR | N |",
        "| --- | --- | --- | --- | --- | --- |",
        _line("membership model", payload["bench_c"]["model"]),
        _line("structural heuristic", payload["bench_c"]["heuristic"]),
        _line("code-length rule", payload["bench_c"]["rules"]),
        "",
        "## Reading",
        "",
        payload["reading"],
        "",
        "Epochs",
        "",
    ]
    for row in payload["history"]:
        lines.append(
            f"- epoch {row['epoch']}: loss {row['loss']:.3f}, "
            f"shifted balanced {row['ood_balanced']:.3f}, "
            f"in-distribution balanced {row['id_balanced']:.3f}"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _evaluate_split(model, episodes, batch: int) -> dict:
    scores = _predict(model, episodes, batch)
    accepted = _accept(scores)
    heuristic = decisions_for(episodes, heuristic_pattern)
    rules = decisions_for(episodes, rules_pattern)
    labels = [episode.label for episode in episodes]
    held = [episode for episode in episodes if episode.held_out()]
    held_index = [index for index, episode in enumerate(episodes) if episode.held_out()]
    held_scores = [scores[index] for index in held_index]
    held_model = _accept(held_scores)
    return {
        "model": _round(summarize(episodes, accepted, scores)),
        "heuristic": _round(summarize(episodes, heuristic)),
        "rules": _round(summarize(episodes, rules)),
        "versus_heuristic": disagreement(labels, accepted, heuristic),
        "versus_rules": disagreement(labels, accepted, rules),
        "held_out": {
            "model": _round(summarize(held, held_model, held_scores)) if held else {"n": 0, "balanced": 0.0, "auroc": 0.0},
            "heuristic": _round(summarize(held, [heuristic[index] for index in held_index])) if held else {"n": 0, "balanced": 0.0},
        },
    }


def _reading(ood: dict) -> str:
    model = ood["model"]["balanced"]
    heuristic = ood["heuristic"]["balanced"]
    gap = model - heuristic
    recovery = ood["versus_heuristic"]["recovery"]
    preservation = ood["versus_heuristic"]["preservation"]
    if gap >= 0.05 and preservation >= 0.9:
        return (
            f"The membership model is {gap:.1%} balanced points above the structural heuristic "
            "on the shifted split, and it keeps most of the heuristic's correct decisions. "
            "That is evidence the network learned a transferable language, not only a candidate preference."
        )
    if recovery >= 0.15 and preservation >= 0.9 and gap >= 0.0:
        return (
            f"Overall the model is {gap:+.1%} balanced points from the heuristic. "
            f"It recovers {recovery:.1%} of the heuristic's mistakes and preserves {preservation:.1%} of its successes. "
            "The gap is real on the ambiguous slice, and it is not yet a large overall win."
        )
    return (
        f"On the shifted split the model scores {model:.3f} balanced against the heuristic's {heuristic:.3f} "
        f"({gap:+.1%} points). Recovery of heuristic errors is {recovery:.1%} and preservation is {preservation:.1%}. "
        "This does not yet show a membership model that is substantially better than the symbolic inducer "
        "on unseen structure. The representation is not adequate to carry a decoder on its own."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--languages", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-bench-c", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    print("building shifted and in-distribution episodes", flush=True)
    ood = build_split(random.Random(1_000), args.languages, True, 6, 0.45, True, "ood")
    ood_n2 = build_split(random.Random(2_000), max(40, args.languages // 3), True, 2, 0.35, True, "ood2")
    id_eval = build_split(random.Random(3_000), max(40, args.languages // 3), False, 6, 0.25, False, "id")
    held = sum(episode.held_out() for episode in ood)
    print(
        f"ood {len(ood)} held-out-byte {held} ood2 {len(ood_n2)} id {len(id_eval)}",
        flush=True,
    )
    model = MembershipNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"parameters {parameter_count(model)}", flush=True)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_score = -1.0
    best_epoch = 0
    stall = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        losses = []
        for _step in range(args.steps):
            batch = sample_batch(rng, args.batch_size, shift=False, structural_rate=0.3)
            tokens, pad, labels, invariant = collate(batch)
            optimizer.zero_grad(set_to_none=True)
            logit, inv_logit = model(tokens, pad)
            loss, _bce, _aux = membership_loss(logit, inv_logit, labels, invariant)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        mean_loss = sum(losses) / max(len(losses), 1)
        ood_scores = _predict(model, ood, args.batch_size)
        id_scores = _predict(model, id_eval, args.batch_size)
        ood_report = summarize(ood, _accept(ood_scores), ood_scores)
        id_report = summarize(id_eval, _accept(id_scores), id_scores)
        history.append(
            {
                "epoch": epoch,
                "loss": mean_loss,
                "ood_balanced": ood_report["balanced"],
                "ood_accuracy": ood_report["accuracy"],
                "ood_auroc": ood_report["auroc"],
                "id_balanced": id_report["balanced"],
            }
        )
        mark = ""
        if ood_report["balanced"] > best_score + 1e-4:
            best_score = ood_report["balanced"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stall = 0
            mark = " *"
        else:
            stall += 1
        print(
            f"epoch {epoch:02d} loss {mean_loss:.3f} shifted {ood_report['balanced']:.3f} "
            f"auroc {ood_report['auroc']:.3f} id {id_report['balanced']:.3f} "
            f"in {time.perf_counter() - started:.1f}s{mark}",
            flush=True,
        )
        if stall >= args.patience:
            break
    model.load_state_dict(best_state)
    print("scoring final splits", flush=True)
    ood_final = _evaluate_split(model, ood, args.batch_size)
    ood_n2_final = _evaluate_split(model, ood_n2, args.batch_size)
    id_final = _evaluate_split(model, id_eval, args.batch_size)
    bench = {
        "note": "BENCH-C was not scored.",
        "model": {"accuracy": 0, "balanced": 0, "tpr": 0, "tnr": 0, "n": 0},
        "heuristic": {"accuracy": 0, "balanced": 0, "tpr": 0, "tnr": 0, "n": 0},
        "rules": {"accuracy": 0, "balanced": 0, "tpr": 0, "tnr": 0, "n": 0},
    }
    if not args.skip_bench_c:
        root = Path(__file__).resolve().parents[1]
        tasks = read_jsonl(root / "data" / "bench_c.jsonl")
        episodes, skipped = _bench_episodes(tasks, model.max_len)
        print(f"bench-c queries {len(episodes)} skipped {skipped}", flush=True)
        bench_eval = _evaluate_split(model, episodes, args.batch_size)
        bench = {
            "note": (
                f"Untouched document tasks, scored once, not used to stop. "
                f"Queries longer than {model.max_len} bytes were dropped ({skipped}). "
                f"Held-out characters in this probe are {''.join(sorted(HELD_OUT))}."
            ),
            "model": bench_eval["model"],
            "heuristic": bench_eval["heuristic"],
            "rules": bench_eval["rules"],
            "versus_heuristic": bench_eval["versus_heuristic"],
        }
        print(
            f"C model {bench['model']['balanced']:.3f} heuristic {bench['heuristic']['balanced']:.3f}",
            flush=True,
        )
    payload = {
        "parameters": parameter_count(model),
        "best_epoch": best_epoch,
        "best_ood_balanced": best_score,
        "history": history,
        "ood": ood_final,
        "ood_n2": ood_n2_final,
        "id": id_final,
        "bench_c": bench,
        "reading": _reading(ood_final),
        "protocol": {
            "selection": "shifted synthetic balanced accuracy",
            "held_out_bytes": sorted(HELD_OUT),
            "train_structural_negative_rate": 0.3,
            "aux": "invariant versus variable bytes on positive queries",
        },
    }
    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "exp2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "membership.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(out / "membership.md", payload)
    print(payload["reading"], flush=True)


if __name__ == "__main__":
    main()
