"""CPU pilot of the tiny candidate ranker.

Train on BENCH-A dev only, choose the epoch by BENCH-A val semantic accuracy,
and score BENCH-A test plus untouched BENCH-C. A second cross-attention run
trains on the BENCH-B document fold. BENCH-C is never used to fit or to stop.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from pattern_induction.candidate_oracle import _rates, document_fold
from pattern_induction.candidates import pattern_cost, presented_candidates
from pattern_induction.ranker import (
    TAU,
    TinyRegexRanker,
    collate_tasks,
    encode_utf8,
    parameter_count,
    soft_cross_entropy,
)
from pattern_induction.tasks import read_jsonl
from pattern_induction.taxonomy import categorize


def pack_task(task) -> dict:
    shown, _pool = presented_candidates(task.visible, k=8)
    patterns = []
    accuracies = []
    recalls = []
    rejections = []
    cand_ids = []
    cand_pad = []
    truncated = False
    for cand in shown:
        accuracy, recall, rejection = _rates(cand.pattern, task)
        ids, pad, cut = encode_utf8(cand.pattern)
        truncated = truncated or cut
        patterns.append(cand.pattern)
        accuracies.append(accuracy)
        recalls.append(recall)
        rejections.append(rejection)
        cand_ids.append(ids)
        cand_pad.append(pad)
    string_ids = []
    string_pad = []
    for text in task.visible:
        ids, pad, cut = encode_utf8(text)
        truncated = truncated or cut
        string_ids.append(ids)
        string_pad.append(pad)
    failure = categorize(
        {
            "semantic_accuracy": accuracies[0],
            "positive_recall": recalls[0],
            "negative_rejection": rejections[0],
            "valid": 1.0,
            "catastrophic": 0.0,
            "gold_regex": task.gold_regex,
            "domain": task.domain,
        }
    )
    rules_at = _argmin_cost(patterns, task.visible)
    oracle_at = max(range(len(accuracies)), key=lambda index: (accuracies[index], -index))
    return {
        "id": task.id,
        "benchmark": task.benchmark,
        "domain": task.domain,
        "split": task.split,
        "n_visible": task.n_visible,
        "failure": failure or "success",
        "patterns": patterns,
        "accuracies": accuracies,
        "string_ids": string_ids,
        "string_pad": string_pad,
        "cand_ids": cand_ids,
        "cand_pad": cand_pad,
        "truncated": truncated,
        "heuristic_at": 0,
        "rules_at": rules_at,
        "oracle_at": oracle_at,
    }


def _argmin_cost(patterns: list[str], visible: list[str]) -> int:
    best = 0
    best_cost = pattern_cost(patterns[0], visible, True)
    for index, pattern in enumerate(patterns[1:], start=1):
        cost = pattern_cost(pattern, visible, True)
        if cost < best_cost - 1e-9:
            best = index
            best_cost = cost
    return best


def load_rows(data_dir: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    names = ["bench_a", "bench_b", "bench_c", "bench_d"]
    for name in names:
        path = data_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        tasks = read_jsonl(path)
        rows = []
        started = time.perf_counter()
        for index, task in enumerate(tasks, start=1):
            rows.append(pack_task(task))
            if index % 200 == 0 or index == len(tasks):
                print(f"pack {name} {index}/{len(tasks)} in {time.perf_counter() - started:.1f}s", flush=True)
        grouped[name] = rows
    return grouped


def _batches(rows: list[dict], batch_size: int, shuffle: bool, rng: random.Random):
    order = list(range(len(rows)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [rows[index] for index in order[start : start + batch_size]]


def epoch_metrics(model, rows, batch_size, optimizer, train: bool, tau: float, rng: random.Random) -> dict:
    model.train(train)
    total_loss = 0.0
    total_sem = 0.0
    total_top1 = 0.0
    count = 0
    for batch_rows in _batches(rows, batch_size, shuffle=train, rng=rng):
        batch = collate_tasks(batch_rows)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(
                batch["strings"],
                batch["string_pad"],
                batch["string_slot_pad"],
                batch["candidates"],
                batch["cand_pad"],
                batch["cand_slot_pad"],
            )
            loss = soft_cross_entropy(logits, batch["accuracy"], ~batch["cand_slot_pad"], tau)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        pred = logits.argmax(dim=-1)
        chosen = batch["accuracy"].gather(1, pred[:, None])[:, 0]
        target = batch["accuracy"].masked_fill(batch["cand_slot_pad"], -1.0).argmax(dim=-1)
        size = pred.shape[0]
        total_loss += float(loss.detach()) * size
        total_sem += float(chosen.detach().sum())
        total_top1 += float((pred == target).sum())
        count += size
    return {
        "loss": total_loss / max(count, 1),
        "semantic_accuracy": total_sem / max(count, 1),
        "top1": total_top1 / max(count, 1),
        "n": count,
    }


def fit_model(name: str, model, train_rows, val_rows, epochs: int, patience: int, batch_size: int, lr: float, tau: float, seed: int) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    best_epoch = 0
    stall = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_m = epoch_metrics(model, train_rows, batch_size, optimizer, True, tau, rng)
        val_m = epoch_metrics(model, val_rows, batch_size, optimizer, False, tau, rng)
        history.append({"epoch": epoch, "train": train_m, "val": val_m})
        mark = ""
        if val_m["semantic_accuracy"] > best_val + 1e-4:
            best_val = val_m["semantic_accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stall = 0
            mark = " *"
        else:
            stall += 1
        print(
            f"{name} epoch {epoch:02d} loss {train_m['loss']:.3f} "
            f"train {train_m['semantic_accuracy']:.3f} val {val_m['semantic_accuracy']:.3f}{mark}",
            flush=True,
        )
        if stall >= patience:
            break
    model.load_state_dict(best_state)
    return {
        "name": name,
        "best_epoch": best_epoch,
        "best_val_semantic_accuracy": best_val,
        "epochs_ran": len(history),
        "seconds": time.perf_counter() - started,
        "parameters": parameter_count(model),
        "history": history,
    }


def predict(model, rows, batch_size: int) -> list[int]:
    model.eval()
    picks = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = collate_tasks(rows[start : start + batch_size])
            logits = model(
                batch["strings"],
                batch["string_pad"],
                batch["string_slot_pad"],
                batch["candidates"],
                batch["cand_pad"],
                batch["cand_slot_pad"],
            )
            picks.extend(int(index) for index in logits.argmax(dim=-1).tolist())
    return picks


def _chosen_accuracy(row: dict, index: int) -> float:
    return row["accuracies"][index]


def summarize(rows: list[dict], pick_of) -> dict:
    if not rows:
        return {"n": 0}
    accuracies = []
    improved = regressed = unchanged = 0
    domains: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        accuracy = _chosen_accuracy(row, pick_of(row))
        base = row["accuracies"][row["heuristic_at"]]
        accuracies.append(accuracy)
        domains[row["domain"]].append(accuracy)
        if abs(accuracy - base) <= 0.01 and pick_of(row) == row["heuristic_at"]:
            unchanged += 1
        elif accuracy > base + 0.01:
            improved += 1
        elif accuracy + 0.01 < base:
            regressed += 1
        else:
            unchanged += 1
    n = len(rows)
    macro = sum(sum(vals) / len(vals) for vals in domains.values()) / len(domains)
    return {
        "n": n,
        "semantic_accuracy": round(sum(accuracies) / n, 4),
        "macro_semantic_accuracy": round(macro, 4),
        "improved_rate": round(improved / n, 4),
        "regressed_rate": round(regressed / n, 4),
        "unchanged_rate": round(unchanged / n, 4),
    }


def family_table(rows: list[dict], pickers: dict) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["failure"]].append(row)
    table = []
    for failure, group in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
        entry = {"failure": failure, "n": len(group)}
        for name, pick_of in pickers.items():
            entry[name] = round(sum(_chosen_accuracy(row, pick_of(row)) for row in group) / len(group), 4)
        table.append(entry)
    return table


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_markdown(report: dict) -> str:
    lines = [
        "# CPU ranker pilot",
        "",
        "The ranker sees visible strings and the same 8 symbolic candidates as the oracle",
        "experiment. Labels are a softmax over hidden semantic accuracy with temperature",
        f"{report['protocol']['tau']}. BENCH-C is not used for training or epoch selection.",
        "The pooled model replaces candidate-to-example attention with an MLP on the mean",
        "of the example vectors, sized to about the same parameter count.",
        "",
        "## Reading",
        "",
    ]
    for paragraph in report["interpretation"]:
        lines.append(paragraph)
        lines.append("")
    lines.append("## Systems")
    lines.append("")
    for name, payload in report["splits"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(payload["role"])
        lines.append("")
        headers = ["System", "Semantic accuracy", "Macro", "Improved", "Regressed", "Unchanged"]
        table = []
        for row in payload["systems"]:
            table.append(
                [
                    row["system"],
                    _pct(row["semantic_accuracy"]),
                    _pct(row["macro_semantic_accuracy"]),
                    _pct(row["improved_rate"]),
                    _pct(row["regressed_rate"]),
                    _pct(row["unchanged_rate"]),
                ]
            )
        lines.extend(_md_table(headers, table))
        lines.append("")
    if report.get("bench_c_families"):
        lines.append("## BENCH-C by heuristic failure")
        lines.append("")
        headers = ["Failure", "n"] + report["family_systems"]
        table = []
        for row in report["bench_c_families"]:
            table.append(
                [row["failure"], str(row["n"])] + [_pct(row[name]) for name in report["family_systems"]]
            )
        lines.extend(_md_table(headers, table))
        lines.append("")
    lines.append("## Training")
    lines.append("")
    for run in report["runs"]:
        lines.append(
            f"`{run['name']}` has {run['parameters']} parameters, selected epoch {run['best_epoch']} "
            f"of {run['epochs_ran']} by validation semantic accuracy "
            f"({_pct(run['best_val_semantic_accuracy'])}), in {run['seconds']:.0f}s."
        )
        lines.append("")
        headers = ["Epoch", "Train loss", "Train semantic", "Val semantic", "Train top-1"]
        table = []
        for item in run["history"]:
            table.append(
                [
                    str(item["epoch"]),
                    f"{item['train']['loss']:.3f}",
                    _pct(item["train"]["semantic_accuracy"]),
                    _pct(item["val"]["semantic_accuracy"]),
                    _pct(item["train"]["top1"]),
                ]
            )
        lines.extend(_md_table(headers, table))
        lines.append("")
    protocol = report["protocol"]
    lines.append("## Protocol")
    lines.append("")
    lines.append(
        f"AdamW learning rate {protocol['lr']}, weight decay 0.01, gradient clip 1, "
        f"batch size {protocol['batch_size']}, soft-label temperature {protocol['tau']}, "
        f"maximum {protocol['epochs']} epochs, patience {protocol['patience']} on validation semantic accuracy. "
        f"Byte length cap {protocol['max_len']}. Truncated inputs: {protocol['truncated_inputs']}."
    )
    lines.append("")
    lines.append(protocol["note"])
    lines.append("")
    return "\n".join(lines)


def interpret(report: dict) -> list[str]:
    splits = report["splits"]
    test = {row["system"]: row for row in splits["bench_a_test"]["systems"]}
    bench_c = {row["system"]: row for row in splits["bench_c"]["systems"]}
    lines = [
        (
            f"On held-out BENCH-A shapes, the code-length rule scores {_pct(test['rules']['semantic_accuracy'])} "
            f"and the 8-candidate oracle scores {_pct(test['oracle_k']['semantic_accuracy'])}. "
            f"Cross-attention, trained on BENCH-A dev, scores {_pct(test['cross_structural']['semantic_accuracy'])}. "
            f"The pooled model scores {_pct(test['pooled_structural']['semantic_accuracy'])}."
        ),
        (
            f"On untouched BENCH-C, the rule scores {_pct(bench_c['rules']['semantic_accuracy'])} "
            f"and the oracle scores {_pct(bench_c['oracle_k']['semantic_accuracy'])}. "
            f"The structural cross-attention model scores {_pct(bench_c['cross_structural']['semantic_accuracy'])} "
            f"(regressions {_pct(bench_c['cross_structural']['regressed_rate'])}). "
            f"The document-trained cross-attention model scores {_pct(bench_c['cross_document']['semantic_accuracy'])} "
            f"(regressions {_pct(bench_c['cross_document']['regressed_rate'])})."
        ),
    ]
    structural_gain = test["cross_structural"]["semantic_accuracy"] - test["rules"]["semantic_accuracy"]
    document_gain = bench_c["cross_document"]["semantic_accuracy"] - bench_c["rules"]["semantic_accuracy"]
    attn_gain = test["cross_structural"]["semantic_accuracy"] - test["pooled_structural"]["semantic_accuracy"]
    if structural_gain < 0.01 and document_gain < 0.01:
        lines.append(
            "Neither training distribution beats the code-length rule by a point on its decision split. "
            "Train loss can still fall. The ranking signal in this candidate set is not something these "
            "tiny nets pick up beyond compression, so a larger GPU run of the same ranker is not justified."
        )
    elif structural_gain >= 0.01 and document_gain < 0.01:
        lines.append(
            "The structural holdout moves and BENCH-C does not. The net can rank synthetic shapes it was "
            "trained on, and that preference does not transfer to the document fields where the rule still leads."
        )
    elif document_gain >= 0.01 and structural_gain < 0.01:
        lines.append(
            "BENCH-C moves after document training, while the shape holdout does not. That BENCH-C number "
            "shares pattern families with BENCH-B, so it is an upper bound on transfer rather than a structural result."
        )
    else:
        lines.append(
            "The ranker beats the rule on both the shape holdout and BENCH-C. That is the result that makes "
            "a larger training run worth doing, still on this candidate set, with regressions kept in view."
        )
    if abs(attn_gain) < 0.005:
        lines.append(
            "Cross-attention and the pooled model are within half a point on BENCH-A test. "
            "Candidate-specific attention is not earning its keep at this size."
        )
    elif attn_gain > 0:
        lines.append(
            f"Cross-attention leads the pooled model by {attn_gain * 100:.1f} points on BENCH-A test."
        )
    else:
        lines.append(
            f"The pooled model leads cross-attention by {-attn_gain * 100:.1f} points on BENCH-A test."
        )
    return lines


def _system_rows(rows: list[dict], model_picks: dict[str, list[int]]) -> list[dict]:
    def constant(index_name: str):
        return lambda row, index_name=index_name: row[index_name]

    pickers = {
        "heuristic": constant("heuristic_at"),
        "rules": constant("rules_at"),
        "oracle_k": constant("oracle_at"),
    }
    for name, picks in model_picks.items():
        by_id = {row["id"]: pick for row, pick in zip(rows, picks)}
        pickers[name] = lambda row, by_id=by_id: by_id[row["id"]]
    summarized = []
    for name, pick_of in pickers.items():
        payload = summarize(rows, pick_of)
        payload["system"] = name
        summarized.append(payload)
    return summarized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/exp2"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    grouped = load_rows(args.data)
    bench_a = grouped["bench_a"]
    structural_train = [row for row in bench_a if row["split"] == "dev"]
    structural_val = [row for row in bench_a if row["split"] == "val"]
    structural_test = [row for row in bench_a if row["split"] == "test"]
    document_train = [row for row in grouped["bench_b"] if document_fold(row["id"]) == "train"]
    document_val = [row for row in grouped["bench_b"] if document_fold(row["id"]) == "val"]

    runs = []
    picks: dict[str, dict[str, list[int]]] = defaultdict(dict)

    def train_and_score(name: str, cross: bool, train_rows, val_rows, eval_sets: dict[str, list[dict]]) -> None:
        torch.manual_seed(args.seed)
        model = TinyRegexRanker(cross_attention=cross)
        print(f"training {name} params={parameter_count(model)}", flush=True)
        run = fit_model(
            name,
            model,
            train_rows,
            val_rows,
            args.epochs,
            args.patience,
            args.batch_size,
            args.lr,
            args.tau,
            args.seed,
        )
        runs.append(run)
        for split_name, rows in eval_sets.items():
            picks[split_name][name] = predict(model, rows, args.batch_size)

    def _baseline(rows: list[dict], key: str) -> float:
        return sum(row["accuracies"][row[key]] for row in rows) / len(rows)

    print(
        f"A-val rules {_baseline(structural_val, 'rules_at'):.3f} "
        f"oracle {_baseline(structural_val, 'oracle_at'):.3f} "
        f"A-test rules {_baseline(structural_test, 'rules_at'):.3f} "
        f"oracle {_baseline(structural_test, 'oracle_at'):.3f} "
        f"C rules {_baseline(grouped['bench_c'], 'rules_at'):.3f} "
        f"oracle {_baseline(grouped['bench_c'], 'oracle_at'):.3f}",
        flush=True,
    )

    eval_sets = {
        "bench_a_dev": structural_train,
        "bench_a_val": structural_val,
        "bench_a_test": structural_test,
        "bench_c": grouped["bench_c"],
        "bench_d": grouped.get("bench_d", []),
        "bench_b_val": document_val,
    }
    train_and_score("cross_structural", True, structural_train, structural_val, eval_sets)
    train_and_score("pooled_structural", False, structural_train, structural_val, eval_sets)
    train_and_score("cross_document", True, document_train, document_val, eval_sets)

    roles = {
        "bench_a_test": "Held-out AST shapes. This is the structural decision split.",
        "bench_a_val": "Used to choose the epoch for the structural models.",
        "bench_a_dev": "Training data for the structural models.",
        "bench_c": "Untouched document tasks. Not used to train or to stop.",
        "bench_d": "Same schemas at several sample sizes. Not used to train or to stop.",
        "bench_b_val": "Used to choose the epoch for the document model. Families overlap BENCH-C.",
    }
    split_rows = {
        "bench_a_test": structural_test,
        "bench_a_val": structural_val,
        "bench_a_dev": structural_train,
        "bench_c": grouped["bench_c"],
        "bench_d": grouped.get("bench_d", []),
        "bench_b_val": document_val,
    }
    splits = {}
    for name, rows in split_rows.items():
        if not rows:
            continue
        splits[name] = {"role": roles[name], "systems": _system_rows(rows, picks[name])}

    family_systems = ["heuristic", "rules", "oracle_k", "cross_structural", "pooled_structural", "cross_document"]
    family_pickers = {}
    bench_c_rows = grouped["bench_c"]
    family_pickers["heuristic"] = lambda row: row["heuristic_at"]
    family_pickers["rules"] = lambda row: row["rules_at"]
    family_pickers["oracle_k"] = lambda row: row["oracle_at"]
    for name in ("cross_structural", "pooled_structural", "cross_document"):
        by_id = {row["id"]: pick for row, pick in zip(bench_c_rows, picks["bench_c"][name])}
        family_pickers[name] = lambda row, by_id=by_id: by_id[row["id"]]

    truncated = sum(1 for rows in grouped.values() for row in rows if row["truncated"])
    report = {
        "runs": runs,
        "splits": splits,
        "bench_c_families": family_table(bench_c_rows, family_pickers),
        "family_systems": family_systems,
        "protocol": {
            "tau": args.tau,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "max_len": 128,
            "seed": args.seed,
            "truncated_inputs": truncated,
            "structural_train_n": len(structural_train),
            "document_train_n": len(document_train),
            "note": (
                "Structural models train on BENCH-A dev and stop on BENCH-A val semantic accuracy. "
                "The document model trains on the BENCH-B hash fold used in the oracle experiment and "
                "stops on the held-out fifth of BENCH-B. Soft targets use hidden semantic accuracy. "
                "Model inputs are only the visible strings and the candidate regex text."
            ),
        },
    }
    report["interpretation"] = interpret(report)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ranker_pilot.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "ranker_pilot.md").write_text(render_markdown(report), encoding="utf-8")
    print("\n".join(report["interpretation"]))


if __name__ == "__main__":
    main()
