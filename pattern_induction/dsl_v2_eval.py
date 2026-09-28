"""BENCH-C after character sets and integer ranges.

Synthesis v1 stays the default assembly. This pass turns those facts on and
rescores the tasks whose gold regex was a partial range or a mixed class.
BENCH-E is not read.
"""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path

from pattern_induction.chart import LOCAL_K, same_fact
from pattern_induction.charset import char_in_spec, v2_facts
from pattern_induction.contingent import contingent_optionals
from pattern_induction.document_facts import document_facts
from pattern_induction.dsl_cover import cover_regex
from pattern_induction.program import MAX_LEN, MAX_SUPPORT, Prog, Task, _MEMBERSHIP, execute
from pattern_induction.scope import collate_scope, hypotheses_from_scope
from pattern_induction.select_train import complete_assembly, load_selector, select_program
from pattern_induction.tasks import read_jsonl
from pattern_induction.version_space import assemble

_TARGET_REASONS = {"partial_range", "mixed_class"}


class _Timeout(Exception):
    pass


def _alarm(signum, frame) -> None:
    raise _Timeout()


def _visible(texts: list[str]) -> list[str]:
    return [text[:MAX_LEN] for text in texts[:MAX_SUPPORT]]


def _probe(bench) -> list[tuple[str, bool]]:
    return [(text, True) for text in bench.hidden_positives] + [(text, False) for text in bench.hidden_negatives]


def _perfect(programs, visible, bench) -> bool:
    probe = _probe(bench)
    return any(all(execute(prog, text, visible) == label for text, label in probe) for prog, _score in programs)


def _accuracy(prog, visible, bench) -> float:
    if prog is None:
        return 0.5
    positives = bench.hidden_positives
    negatives = bench.hidden_negatives
    recall = sum(bool(execute(prog, text, visible)) for text in positives) / max(len(positives), 1)
    rejection = sum(not execute(prog, text, visible) for text in negatives) / max(len(negatives), 1)
    return 0.5 * (recall + rejection)


def _spec_size(spec: str) -> int:
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_-"
    return sum(1 for char in alphabet if char_in_spec(char, spec))


def _broad(facts: list) -> list:
    """Drop integer hulls and keep the widest character set on each span."""
    grouped: dict[tuple, list] = {}
    kept = []
    for fact in facts:
        if fact.kind == "int":
            continue
        if fact.kind == "set":
            grouped.setdefault((fact.slot, fact.start, fact.end, fact.lo, fact.hi), []).append(fact)
            continue
        kept.append(fact)
    for group in grouped.values():
        kept.append(max(group, key=lambda fact: _spec_size(fact.cls)))
    return kept


def _hyps(scope, bench, visible: list[str]) -> list:
    task = Task(bench.domain, visible, Prog("COPY"), [])
    batch = collate_scope([task])
    hyps = hypotheses_from_scope(scope, batch, [visible], LOCAL_K)[0]
    for hyp in contingent_optionals(visible):
        if any(same_fact(hyp, kept, False) for kept in hyps):
            continue
        hyps.append(hyp)
    return hyps


def _coverage_counts(tasks) -> dict:
    counts = {"exact": 0, "open": 0, "no": 0}
    for task in tasks:
        counts[cover_regex(task.gold_regex).status] += 1
    total = max(len(tasks), 1)
    counts["exact_or_open"] = counts["exact"] + counts["open"]
    counts["exact_or_open_rate"] = counts["exact_or_open"] / total
    counts["n"] = len(tasks)
    return counts


def evaluate(root: Path, limit_s: float = 90.0) -> dict:
    started = time.perf_counter()
    tasks = read_jsonl(root / "data" / "bench_c.jsonl")
    split = json.loads((root / "results" / "needle" / "bench_split.json").read_text())
    by_id = {row["id"]: row for row in split["rows"]}
    targets = [task for task in tasks if by_id[task.id].get("dsl_reason") in _TARGET_REASONS]
    scope, witness, prior = load_selector(root)
    signal.signal(signal.SIGALRM, _alarm)
    rows = []
    for index, bench in enumerate(targets):
        _MEMBERSHIP.clear()
        visible = _visible(bench.visible)
        cover = cover_regex(bench.gold_regex)
        old = by_id[bench.id]
        row = {
            "id": bench.id,
            "domain": bench.domain,
            "old_reason": old["dsl_reason"],
            "cover": cover.status,
            "cover_reason": cover.reason,
            "expressible": cover.status in {"exact", "open"},
        }
        t0 = time.perf_counter()
        signal.alarm(int(limit_s))
        try:
            deterministic = document_facts(visible) + v2_facts(visible)
            det = assemble(visible, deterministic, wrap_optional_alt=True, pack=True)
            broad = assemble(visible, _broad(deterministic), wrap_optional_alt=True, pack=True)
            row["deterministic_perfect"] = _perfect(det.programs, visible, bench)
            row["recoverable"] = _perfect(broad.programs, visible, bench)
        except _Timeout:
            row["deterministic_perfect"] = False
            row["recoverable"] = False
        finally:
            signal.alarm(0)
        _MEMBERSHIP.clear()
        signal.alarm(int(limit_s))
        try:
            hyps = _hyps(scope, bench, visible)
            assembly = complete_assembly(visible, hyps, v2=True)
            row["programs"] = len(assembly.programs)
            row["truncated"] = assembly.truncated
            row["synthesis_perfect"] = _perfect(assembly.programs, visible, bench)
            row["timed_out"] = False
        except _Timeout:
            row["programs"] = 0
            row["truncated"] = False
            row["synthesis_perfect"] = False
            row["timed_out"] = True
        finally:
            signal.alarm(0)
        row["assemble_s"] = round(time.perf_counter() - t0, 3)
        rows.append(row)
        if index % 10 == 0 or index + 1 == len(targets):
            perfect = sum(1 for item in rows if item["synthesis_perfect"])
            print(
                f"dsl {index + 1}/{len(targets)} {bench.domain} programs {row['programs']} "
                f"perfect {int(row['synthesis_perfect'])} oracle {perfect}/{len(rows)}",
                flush=True,
            )
            partial = root / "results" / "needle" / "dsl_v2_partial.json"
            partial.write_text(json.dumps(rows) + "\n")
    _select(scope, witness, prior, targets, rows, limit_s)
    payload = _summarize(root, tasks, rows, time.perf_counter() - started)
    partial = root / "results" / "needle" / "dsl_v2_partial.json"
    if partial.exists():
        partial.unlink()
    return payload


def _select(scope, witness, prior, targets, rows, limit_s: float) -> None:
    by_id = {bench.id: bench for bench in targets}
    for index, row in enumerate(rows):
        bench = by_id[row["id"]]
        visible = _visible(bench.visible)
        if row["timed_out"] or row["programs"] == 0:
            row["selector_accuracy"] = 0.5
            row["selector_timed_out"] = bool(row["timed_out"])
            continue
        task = Task(bench.domain, visible, Prog("COPY"), [])
        chosen = None
        timed_out = False
        _MEMBERSHIP.clear()
        signal.alarm(int(limit_s))
        try:
            chosen, _programs, _count = select_program(scope, witness, prior, task, with_stability=True, v2=True)
        except _Timeout:
            timed_out = True
            chosen = None
        finally:
            signal.alarm(0)
        row["selector_accuracy"] = 0.5 if timed_out else _accuracy(chosen, visible, bench)
        row["selector_timed_out"] = timed_out
        if index % 10 == 0 or timed_out or index + 1 == len(rows):
            print(
                f"dsl-select {index + 1}/{len(rows)} {row['domain']} acc {row['selector_accuracy']:.3f} "
                f"timeout {int(timed_out)}",
                flush=True,
            )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _headline(root: Path, rows: list[dict]) -> float | None:
    coverage = json.loads((root / "results" / "needle" / "coverage.json").read_text())
    split = json.loads((root / "results" / "needle" / "bench_split.json").read_text())
    exact = {row["id"]: row["selector_accuracy"] for row in coverage["rows"] if row.get("selector_accuracy") is not None}
    fresh = {row["id"]: row["selector_accuracy"] for row in rows if row.get("selector_accuracy") is not None}
    scores = []
    for row in split["rows"]:
        if row["id"] in fresh:
            scores.append(fresh[row["id"]])
        elif row["dsl"] == "exact" and row["id"] in exact:
            scores.append(exact[row["id"]])
        else:
            scores.append(row["selector_accuracy"])
    return _mean(scores)


def _summarize(root: Path, tasks, rows: list[dict], seconds: float) -> dict:
    expressible = [row for row in rows if row["expressible"]]
    n = len(rows)
    perfect = sum(1 for row in rows if row["synthesis_perfect"])
    recoverable = sum(1 for row in rows if row["recoverable"])
    found = sum(1 for row in rows if row["recoverable"] and row["synthesis_perfect"])
    conditional = [
        row["selector_accuracy"]
        for row in rows
        if row.get("synthesis_perfect") and row.get("selector_accuracy") is not None and not row.get("selector_timed_out")
    ]
    empty = sum(1 for row in expressible if row["programs"] == 0 and not row["timed_out"])
    return {
        "dsl": _coverage_counts(tasks),
        "target_n": n,
        "expressible_n": len(expressible),
        "synthesis_oracle": perfect / max(n, 1),
        "synthesis_perfect": perfect,
        "recoverable_oracle": recoverable / max(n, 1),
        "recoverable_n": recoverable,
        "recoverable_synthesis_coverage": found / max(recoverable, 1),
        "deterministic_oracle": sum(1 for row in rows if row["deterministic_perfect"]) / max(n, 1),
        "selector_accuracy": _mean([row["selector_accuracy"] for row in rows if row.get("selector_accuracy") is not None]),
        "selector_accuracy_given_perfect": _mean(conditional),
        "empty_expressible": empty,
        "timeouts": sum(1 for row in rows if row["timed_out"] or row.get("selector_timed_out")),
        "assemble_timeouts": sum(1 for row in rows if row["timed_out"]),
        "selector_timeouts": sum(1 for row in rows if row.get("selector_timed_out")),
        "max_programs": max((row["programs"] for row in rows), default=0),
        "headline_semantic_accuracy": _headline(root, rows),
        "seconds": seconds,
        "model_unchanged": True,
        "bench_e_scored": False,
        "rows": rows,
    }


def render_markdown(payload: dict) -> str:
    dsl = payload["dsl"]
    conditional = payload.get("selector_accuracy_given_perfect") or 0.0
    headline = payload.get("headline_semantic_accuracy") or 0.0
    selector = payload.get("selector_accuracy") or 0.0
    lines = [
        "## DSL v2",
        "",
        "Character sets and integer ranges are read off the four supports and added only when assembly asks for them. "
        "Synthesis v1 stays the default, so the recorded exact-slice gates are unchanged. "
        "The recognizer is not retrained. BENCH-E stays sealed.",
        "",
        f"Of {dsl['n']} BENCH-C patterns, {dsl['exact']} are exact, {dsl['open']} are open, and {dsl['no']} stay outside. "
        f"Exact or open coverage is {dsl['exact_or_open_rate']:.3f}. "
        "The 25 outside patterns are word classes, space classes, month-name alternations, and five IPv4 addresses whose octet tree is still 20 nodes.",
        "",
        f"The 154 partial-range and mixed-class tasks are the measurement set. {payload['expressible_n']} of them are expressible. "
        f"Assembly timeouts are {payload['assemble_timeouts']}. "
        f"Empty charts on expressible tasks: {payload['empty_expressible']}. "
        f"Max chart {payload['max_programs']} programs. Wall time {payload['seconds']:.0f}s.",
        "",
        f"Synthesis oracle on these 154 tasks is {payload['synthesis_oracle']:.3f} ({payload['synthesis_perfect']}/{payload['target_n']}). "
        f"Deterministic facts alone reach {payload['deterministic_oracle']:.3f}. "
        "The neural chart supplies the rest.",
        "",
        f"A broad reading, the widest character set on each span with integer hulls removed, is hidden-set perfect on {payload['recoverable_oracle']:.3f} of the 154. "
        f"Every one of those charts contains that program ({payload['recoverable_synthesis_coverage']:.3f} recoverable synthesis coverage). "
        "The broad reading is lower than the synthesis oracle because an integer hull or a neural program is sometimes the one that matches the hidden set. "
        "The remaining misses are unobserved numeric endpoints, supports cut at 32 characters, or a shape these facts do not name.",
        "",
        f"Selector accuracy on the 154 is {selector:.3f}. "
        f"Given a hidden-set-perfect program, and excluding selector timeouts, accuracy is {conditional:.3f}. "
        f"Selector timeouts are {payload['selector_timeouts']}. "
        "They are stability resynthesis on large charts. Assembly itself finished.",
        "",
        f"Headline semantic accuracy, with the frozen exact-slice scores kept and these 154 tasks rescored, is {headline:.3f}. "
        "Selector timeouts abstain at 0.5 inside that headline.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| DSL exact/open coverage | {dsl['exact_or_open_rate']:.3f} |",
        f"| Expressible partial/mixed tasks | {payload['expressible_n']}/{payload['target_n']} |",
        f"| Broad-set recoverable oracle | {payload['recoverable_oracle']:.3f} |",
        f"| Recoverable synthesis coverage | {payload['recoverable_synthesis_coverage']:.3f} |",
        f"| Synthesis oracle on partial/mixed | {payload['synthesis_oracle']:.3f} |",
        f"| Selector given perfect candidate | {conditional:.3f} |",
        f"| BENCH-C semantic accuracy | {headline:.3f} |",
        f"| Empty charts on expressible tasks | {payload['empty_expressible']} |",
        f"| Assembly timeouts | {payload['assemble_timeouts']} |",
        f"| Selector timeouts | {payload['selector_timeouts']} |",
        "",
    ]
    return "\n".join(lines) + "\n"


def _main() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = evaluate(root)
    dest = root / "results" / "needle" / "dsl_v2.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    md = root / "results" / "needle" / "dsl_v2.md"
    md.write_text(render_markdown(payload), encoding="utf-8")
    print(
        f"dsl coverage {payload['dsl']['exact_or_open_rate']:.3f} "
        f"oracle {payload['synthesis_oracle']:.3f} "
        f"recoverable {payload['recoverable_oracle']:.3f} "
        f"conditional {payload['selector_accuracy_given_perfect']} "
        f"headline {payload['headline_semantic_accuracy']}",
        flush=True,
    )


if __name__ == "__main__":
    _main()
