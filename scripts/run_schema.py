"""Train a tiny set encoder to predict a latent schema, then compile it.

The stream is synthetic and labeled by the schema that generated the strings.
Epochs stop on a shifted format split: other delimiters and longer prefixes.
BENCH-C is scored once and is not used to fit or to stop. BENCH-A is not used.
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

from pattern_induction.candidates import pattern_cost, presented_candidates
from pattern_induction.evaluate import score_regex
from pattern_induction.heuristic import infer_regex
from pattern_induction.schema import (
    MAX_LEN,
    MAX_PARTS,
    MAX_STRINGS,
    PROPERTY_NAMES,
    PAD,
    compile_parts,
    encode_strings,
    from_regex,
    parts_to_arrays,
    properties,
    sample_task,
)
from pattern_induction.schema_net import SchemaNet, logits_to_parts, parameter_count, schema_loss
from pattern_induction.tasks import read_jsonl
from pattern_induction.taxonomy import categorize


def _rule_pattern(strings: list[str]) -> str:
    shown, _pool = presented_candidates(strings, k=8, pool_limit=80)
    patterns = [item.pattern for item in shown] or [infer_regex(strings)]
    best = 0
    best_cost = pattern_cost(patterns[0], strings, True)
    for index, pattern in enumerate(patterns[1:], start=1):
        cost = pattern_cost(pattern, strings, True)
        if cost < best_cost - 1e-9:
            best = index
            best_cost = cost
    return patterns[best]


def _semantic(pattern: str | None, task) -> float:
    if not pattern:
        return 0.0
    try:
        return score_regex(pattern, task).semantic_accuracy
    except Exception:
        return 0.0


def _matches_visible(pattern: str | None, strings: list[str]) -> bool:
    if not pattern:
        return False
    try:
        import re

        compiled = re.compile(pattern)
    except re.error:
        return False
    return all(compiled.fullmatch(text) is not None for text in strings)


def _collate(rows: list[tuple]) -> dict[str, torch.Tensor]:
    batch = len(rows)
    n_strings = min(MAX_STRINGS, max(len(strings) for _parts, strings, _pattern in rows))
    tokens = torch.full((batch, n_strings, MAX_LEN), PAD, dtype=torch.long)
    token_pad = torch.ones((batch, n_strings, MAX_LEN), dtype=torch.bool)
    slot_pad = torch.ones((batch, n_strings), dtype=torch.bool)
    target = {
        name: torch.zeros((batch, MAX_PARTS), dtype=torch.long)
        for name in ("kind", "optional", "charset", "lo", "hi", "rep_lo", "rep_hi", "delim")
    }
    target["chars"] = torch.zeros((batch, MAX_PARTS, 16), dtype=torch.long)
    for index, (parts, strings, _pattern) in enumerate(rows):
        ids, pad = encode_strings(strings)
        for slot, (row_ids, row_pad) in enumerate(zip(ids, pad)):
            tokens[index, slot] = torch.tensor(row_ids, dtype=torch.long)
            token_pad[index, slot] = torch.tensor(row_pad, dtype=torch.bool)
            slot_pad[index, slot] = False
        arrays = parts_to_arrays(parts)
        for name in ("kind", "optional", "charset", "lo", "hi", "rep_lo", "rep_hi", "delim"):
            target[name][index] = torch.tensor(arrays[name], dtype=torch.long)
        target["chars"][index] = torch.tensor(arrays["chars"], dtype=torch.long)
    target["tokens"] = tokens
    target["token_pad"] = token_pad
    target["slot_pad"] = slot_pad
    return target


def _predict(model, strings: list[str]) -> str | None:
    ids, pad = encode_strings(strings)
    tokens = torch.full((1, max(len(ids), 1), MAX_LEN), PAD, dtype=torch.long)
    token_pad = torch.ones((1, max(len(ids), 1), MAX_LEN), dtype=torch.bool)
    slot_pad = torch.ones((1, max(len(ids), 1)), dtype=torch.bool)
    for slot, (row_ids, row_pad) in enumerate(zip(ids, pad)):
        tokens[0, slot] = torch.tensor(row_ids, dtype=torch.long)
        token_pad[0, slot] = torch.tensor(row_pad, dtype=torch.bool)
        slot_pad[0, slot] = False
    model.eval()
    with torch.inference_mode():
        logits = model(tokens, token_pad, slot_pad, None)
    parts = logits_to_parts(logits)[0]
    return compile_parts(parts)


def _val_metrics(model, rows: list[tuple], batch_size: int) -> tuple[float, float]:
    """Exact compiled-regex match, and kind accuracy through the first END part."""
    exact = 0
    kind_hit = 0
    kind_slots = 0
    model.eval()
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = _collate(chunk)
        with torch.inference_mode():
            logits = model(batch["tokens"], batch["token_pad"], batch["slot_pad"], None)
        decoded = logits_to_parts(logits)
        for parts, (gold, _strings, pattern) in zip(decoded, chunk):
            if compile_parts(parts) == pattern:
                exact += 1
            for pred, truth in zip(parts, gold):
                kind_slots += 1
                kind_hit += int(pred.kind == truth.kind)
                if truth.kind == 0:
                    break
    total = max(len(rows), 1)
    return exact / total, kind_hit / max(kind_slots, 1)


def _train_epoch(model, optimizer, steps: int, batch_size: int, rng: random.Random) -> float:
    model.train()
    total = 0.0
    for _ in range(steps):
        rows = [sample_task(rng, rng.choice((4, 5, 6, 8)), shift=False) for _ in range(batch_size)]
        batch = _collate(rows)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["tokens"], batch["token_pad"], batch["slot_pad"], batch["chars"])
        loss = schema_loss(logits, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach())
    return total / max(steps, 1)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _movement(accuracy: float, baseline: float) -> str:
    if accuracy > baseline + 0.01:
        return "improved"
    if accuracy < baseline - 0.01:
        return "regressed"
    return "unchanged"


def _pack(rows: list[dict], key: str) -> dict:
    movement = {"improved": 0, "regressed": 0, "unchanged": 0}
    by_domain: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        movement[_movement(row[key], row["heuristic"])] += 1
        by_domain[row["domain"]].append(row[key])
    total = len(rows) or 1
    return {
        "semantic_accuracy": round(_mean([row[key] for row in rows]), 4),
        "macro": round(_mean([_mean(scores) for scores in by_domain.values()]), 4) if by_domain else 0.0,
        "improved": round(movement["improved"] / total, 4),
        "regressed": round(movement["regressed"] / total, 4),
        "unchanged": round(movement["unchanged"] / total, 4),
        "n": len(rows),
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _properties_report(rows: list[dict]) -> dict[str, float]:
    hits = {name: 0 for name in PROPERTY_NAMES}
    total = 0
    for row in rows:
        gold = properties(row["gold"])
        pred = properties(row["raw_predicted"]) if row["raw_predicted"] else None
        if gold is None:
            continue
        total += 1
        for name in PROPERTY_NAMES:
            if pred is not None and pred[name] == gold[name]:
                hits[name] += 1
    return {name: round(hits[name] / total, 4) if total else 0.0 for name in (*PROPERTY_NAMES,)} | {"n": total}


def _interpret(bench_c: dict, expressible: float, val_exact: float) -> list[str]:
    by_name = {row["system"]: row for row in bench_c["systems"]}
    schema = by_name["schema"]
    rules = by_name["rules"]
    heuristic = by_name["heuristic"]
    props = bench_c["properties"]
    prefix = props.get("prefix", 0.0)
    lines = [
        (
            f"The schema compiler covers {expressible * 100:.1f}% of BENCH-C exactly. "
            f"On the shifted validation split the greedy schema matches the generator regex "
            f"{_pct(val_exact)} of the time. "
            f"On BENCH-C the compiled schema scores {_pct(schema['semantic_accuracy'])} "
            f"with regressions {_pct(schema['regressed'])}. "
            f"The heuristic scores {_pct(heuristic['semantic_accuracy'])} and the old rule "
            f"{_pct(rules['semantic_accuracy'])}."
        ),
        (
            f"Literal-prefix recovery on BENCH-C is {_pct(prefix)}. "
            f"Repeat recovery is {_pct(props.get('repeat', 0.0))}, "
            f"optional recovery is {_pct(props.get('optional', 0.0))}, "
            f"and alternation recovery is {_pct(props.get('alternation', 0.0))}."
        ),
    ]
    if schema["semantic_accuracy"] >= rules["semantic_accuracy"] + 0.01 and schema["regressed"] <= 0.02:
        lines.append(
            "The compiled schema beats the old rule by at least a point with regressions inside 2%. "
            "The latent facts are carrying the decision."
        )
    elif props.get("prefix", 0.0) >= 0.8 and schema["semantic_accuracy"] < rules["semantic_accuracy"]:
        lines.append(
            "The model recovers constants more often than it recovers a regex that beats the rule. "
            "The schema language, not the candidate list, is the limit on this run."
        )
    else:
        lines.append(
            "Surface ranking was already set aside. This run asks whether a tiny set encoder "
            "reads the language off six strings. The BENCH-C score is the transfer check, "
            "and it was not used to choose the epoch."
        )
    return lines


def _markdown_table(headers: list[str], body: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_markdown(report: dict) -> str:
    lines = [
        "# Latent schema",
        "",
        "A tiny set encoder reads the visible strings and predicts a short schema: "
        "literal spans, character classes, repeated groups, optional tails, and literal alternations. "
        "A deterministic compiler turns that schema into a regex. Nothing in the model ranks a candidate pool.",
        "",
        "Training tasks are sampled from the schema grammar. The epoch is chosen on a shifted split "
        "with held-out delimiters and longer prefixes. BENCH-C is scored once.",
        "",
        "## Reading",
        "",
    ]
    for paragraph in report["reading"]:
        lines.append(paragraph)
        lines.append("")
    lines.append("## Protocol")
    lines.append("")
    protocol = report["protocol"]
    lines.append(
        f"Parameters {protocol['parameters']}. Epoch {protocol['epoch']} of {protocol['epochs_ran']}. "
        f"Shifted-split exact regex match {_pct(protocol['val_exact'])}. "
        f"In-distribution exact match {_pct(protocol['id_exact'])}, logged and not used to stop. "
        f"Streamed tasks per epoch: {protocol['tasks_per_epoch']}."
    )
    lines.append("")
    for name, group in report["groups"].items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(group["note"])
        lines.append("")
        headers = ["System", "Semantic accuracy", "Macro", "Improved", "Regressed"]
        body = []
        for row in group["systems"]:
            body.append([
                row["label"],
                _pct(row["semantic_accuracy"]),
                _pct(row["macro"]),
                _pct(row["improved"]),
                _pct(row["regressed"]),
            ])
        lines.extend(_markdown_table(headers, body))
        lines.append("")
        if group.get("properties"):
            lines.append("Property recovery against the gold regex:")
            lines.append("")
            headers = ["Property", "Accuracy"]
            body = [[key, _pct(value)] for key, value in group["properties"].items() if key != "n"]
            lines.extend(_markdown_table(headers, body))
            lines.append("")
        if group.get("families"):
            lines.append("Heuristic-failure families:")
            lines.append("")
            headers = ["Failure", "n", "heuristic", "old rule", "schema"]
            body = []
            for row in group["families"]:
                body.append([
                    row["failure"],
                    str(row["n"]),
                    _pct(row["heuristic"]),
                    _pct(row["rules"]),
                    _pct(row["schema"]),
                ])
            lines.extend(_markdown_table(headers, body))
            lines.append("")
    return "\n".join(lines)


def _score_benchmark(model, tasks) -> list[dict]:
    rows = []
    started = time.perf_counter()
    for index, task in enumerate(tasks, start=1):
        predicted = _predict(model, task.visible)
        if not _matches_visible(predicted, task.visible):
            predicted_score = _semantic(predicted, task)
            usable = None
        else:
            predicted_score = _semantic(predicted, task)
            usable = predicted
        heuristic = infer_regex(task.visible)
        heur_score = _semantic(heuristic, task)
        rule = _rule_pattern(task.visible)
        gold_parts = from_regex(task.gold_regex)
        ceiling_pattern = compile_parts(gold_parts) if gold_parts else None
        ceiling_score = _semantic(ceiling_pattern, task) if ceiling_pattern else 0.0
        failure = categorize(
            {
                "semantic_accuracy": heur_score,
                "positive_recall": heur_score,
                "negative_rejection": heur_score,
                "valid": 1.0,
                "catastrophic": 0.0,
                "gold_regex": task.gold_regex,
                "domain": task.domain,
            }
        )
        rows.append(
            {
                "domain": task.domain,
                "failure": failure or "success",
                "gold": task.gold_regex,
                "predicted": usable,
                "raw_predicted": predicted,
                "heuristic": heur_score,
                "rules": _semantic(rule, task),
                "schema": predicted_score if usable else 0.0,
                "schema_fallback": predicted_score if usable else heur_score,
                "ceiling": ceiling_score if ceiling_pattern and ceiling_score >= 0.999 else None,
            }
        )
        if index % 200 == 0 or index == len(tasks):
            print(f"score {index}/{len(tasks)} in {time.perf_counter() - started:.1f}s", flush=True)
    return rows


def _systems(rows: list[dict]) -> list[dict]:
    packed = []
    for key, label in (
        ("heuristic", "heuristic"),
        ("rules", "old code-length rule"),
        ("schema", "compiled schema"),
        ("schema_fallback", "schema, else heuristic"),
    ):
        row = _pack(rows, key)
        row["system"] = key
        row["label"] = label
        packed.append(row)
    ceilings = [row["ceiling"] for row in rows if row["ceiling"] is not None]
    packed.append(
        {
            "system": "ceiling",
            "label": "gold schema, expressible tasks only",
            "semantic_accuracy": round(_mean(ceilings), 4) if ceilings else 0.0,
            "macro": round(_mean(ceilings), 4) if ceilings else 0.0,
            "improved": 0.0,
            "regressed": 0.0,
            "unchanged": 0.0,
            "n": len(ceilings),
        }
    )
    return packed


def _families(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["failure"]].append(row)
    table = []
    for label, items in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        table.append(
            {
                "failure": label,
                "n": len(items),
                "heuristic": round(_mean([item["heuristic"] for item in items]), 4),
                "rules": round(_mean([item["rules"] for item in items]), 4),
                "schema": round(_mean([item["schema"] for item in items]), 4),
            }
        )
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-final", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    val_rows = [sample_task(random.Random(10_000 + index), 6, shift=True) for index in range(args.val)]
    id_rows = [sample_task(random.Random(20_000 + index), 6, shift=False) for index in range(min(128, args.val))]
    model = SchemaNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"parameters {parameter_count(model)}", flush=True)
    best_state = copy.deepcopy(model.state_dict())
    best_exact = -1.0
    best_kind = -1.0
    best_epoch = 0
    stall = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        loss = _train_epoch(model, optimizer, args.steps, args.batch_size, rng)
        exact, kind_acc = _val_metrics(model, val_rows, args.batch_size)
        id_exact, id_kind = _val_metrics(model, id_rows, args.batch_size)
        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "val_exact": exact,
                "val_kind": kind_acc,
                "id_exact": id_exact,
                "id_kind": id_kind,
            }
        )
        mark = ""
        score = (exact, kind_acc)
        if score > (best_exact, best_kind) and (exact > best_exact + 1e-4 or kind_acc > best_kind + 1e-4):
            best_exact = exact
            best_kind = kind_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stall = 0
            mark = " *"
        else:
            stall += 1
        print(
            f"epoch {epoch:02d} loss {loss:.3f} shifted {exact:.3f} kind {kind_acc:.3f} "
            f"id {id_exact:.3f} in {time.perf_counter() - started:.1f}s{mark}",
            flush=True,
        )
        if stall >= args.patience:
            break
    model.load_state_dict(best_state)
    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "exp2"
    out.mkdir(parents=True, exist_ok=True)
    groups = {}
    if not args.skip_final:
        data = root / "data"
        for name, filename, note in (
            ("C", "bench_c.jsonl", "Untouched document tasks, scored once."),
            ("D", "bench_d.jsonl", "Sample-size curve. Not used to fit or to stop."),
        ):
            tasks = read_jsonl(data / filename)
            rows = _score_benchmark(model, tasks)
            expressible = sum(row["ceiling"] is not None for row in rows) / max(len(rows), 1)
            groups[name] = {
                "note": note,
                "n": len(rows),
                "expressible": round(expressible, 4),
                "systems": _systems(rows),
                "properties": _properties_report(rows) if name == "C" else {},
                "families": _families(rows) if name == "C" else [],
            }
            print(
                f"{name} schema {groups[name]['systems'][2]['semantic_accuracy']:.3f} "
                f"expressible {expressible:.3f}",
                flush=True,
            )
    report = {
        "protocol": {
            "parameters": parameter_count(model),
            "epoch": best_epoch,
            "epochs_ran": len(history),
            "val_exact": best_exact,
            "id_exact": history[best_epoch - 1]["id_exact"] if history else 0.0,
            "tasks_per_epoch": args.steps * args.batch_size,
            "history": history,
        },
        "groups": groups,
        "reading": [],
    }
    if "C" in groups:
        report["reading"] = _interpret(groups["C"], groups["C"]["expressible"], best_exact)
    else:
        report["reading"] = ["Final benchmarks were skipped."]
    (out / "schema.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "schema.md").write_text(render_markdown(report), encoding="utf-8")
    print("\n".join(report["reading"]))


if __name__ == "__main__":
    main()
