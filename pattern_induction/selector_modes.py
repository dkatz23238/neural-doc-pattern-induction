"""Selector runtime, operator witnesses, and the old-rule union on BENCH-C.

Assembly stays as it was. Selection either resynthesizes a subset, reuses the
chart, or scores the candidate that was already built. Character-set witnesses
are deterministic. The 1.226M recognizer is not trained. BENCH-E is not read.
"""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path

from pattern_induction.chart import LOCAL_K, same_fact
from pattern_induction.contingent import contingent_optionals
from pattern_induction.dsl_cover import cover_regex
from pattern_induction.operator_witness import identifiable_facts, operator_signature
from pattern_induction.program import MAX_LEN, MAX_SUPPORT, Prog, Task, _MEMBERSHIP, execute, to_regex
from pattern_induction.scope import collate_scope, hypotheses_from_scope
from pattern_induction.select_train import LARGE_CHART, complete_assembly, load_selector, select_from_programs
from pattern_induction.tasks import read_jsonl
from pattern_induction.version_space import assemble

_TARGET_REASONS = {"partial_range", "mixed_class"}
# Fixed before the rescore. Not fit to BENCH-C labels.
CONFIDENCE_MIN = 0.70
STABILITY_MIN = 1.0
AGREEMENT_TASKS = 12
AGREEMENT_CHART = 80
A_LIMIT_S = 25


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
    return any(all(execute(prog, text, visible) == label for text, label in probe) for prog in programs)


def _one_perfect(prog, visible, bench) -> bool:
    if prog is None:
        return False
    return all(execute(prog, text, visible) == label for text, label in _probe(bench))


def _accuracy(prog, visible, bench) -> float:
    if prog is None:
        return 0.5
    positives = bench.hidden_positives
    negatives = bench.hidden_negatives
    recall = sum(bool(execute(prog, text, visible)) for text in positives) / max(len(positives), 1)
    rejection = sum(not execute(prog, text, visible) for text in negatives) / max(len(negatives), 1)
    return 0.5 * (recall + rejection)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def _hyps(scope, bench, visible: list[str]) -> list:
    task = Task(bench.domain, visible, Prog("COPY"), [])
    batch = collate_scope([task])
    hyps = hypotheses_from_scope(scope, batch, [visible], LOCAL_K)[0]
    for hyp in contingent_optionals(visible):
        if any(same_fact(hyp, kept, False) for kept in hyps):
            continue
        hyps.append(hyp)
    return hyps


def _operator_count(prog: Prog) -> int:
    return len(operator_signature(prog))


def _pick_identifiable(programs, supports: list[str]):
    best = None
    best_key = None
    for prog, _score in programs:
        if not all(execute(prog, text, supports) for text in supports):
            continue
        nodes = _operator_count(prog)
        key = (nodes, -prog.nodes())
        if best_key is None or key > best_key:
            best_key = key
            best = prog
    return best


def _previous_needle(root: Path) -> dict[str, float]:
    coverage = json.loads((root / "results" / "needle" / "coverage.json").read_text())
    split = json.loads((root / "results" / "needle" / "bench_split.json").read_text())
    dsl = json.loads((root / "results" / "needle" / "dsl_v2.json").read_text())
    scores = {row["id"]: row["selector_accuracy"] for row in split["rows"]}
    for row in coverage["rows"]:
        if row.get("selector_accuracy") is not None:
            scores[row["id"]] = row["selector_accuracy"]
    for row in dsl["rows"]:
        if row.get("selector_accuracy") is not None:
            scores[row["id"]] = row["selector_accuracy"]
    return scores


def _old_scores(root: Path) -> dict[str, float]:
    scores = {}
    with (root / "results" / "exp2" / "candidate_oracle_tasks.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            scores[row["id"]] = row["rules"]["accuracy"]
    return scores


def _headline(root: Path, fresh: dict[str, float]) -> float | None:
    coverage = json.loads((root / "results" / "needle" / "coverage.json").read_text())
    split = json.loads((root / "results" / "needle" / "bench_split.json").read_text())
    exact = {row["id"]: row["selector_accuracy"] for row in coverage["rows"] if row.get("selector_accuracy") is not None}
    scores = []
    for row in split["rows"]:
        if row["id"] in fresh:
            scores.append(fresh[row["id"]])
        elif row["dsl"] == "exact" and row["id"] in exact:
            scores.append(exact[row["id"]])
        else:
            scores.append(row["selector_accuracy"])
    return _mean(scores)


def _union(old: dict[str, float], needle: dict[str, float]) -> dict:
    ids = [task_id for task_id in old if task_id in needle]
    paired = [(old[task_id], needle[task_id]) for task_id in ids]
    needle_better = sum(1 for left, right in paired if right > left + 1e-9)
    old_better = sum(1 for left, right in paired if left > right + 1e-9)
    ties = len(paired) - needle_better - old_better
    return {
        "n": len(paired),
        "old": _mean([left for left, _right in paired]),
        "needle": _mean([right for _left, right in paired]),
        "oracle_union": _mean([max(left, right) for left, right in paired]),
        "needle_better": needle_better,
        "old_better": old_better,
        "tie": ties,
    }


def _take_needle(row: dict) -> bool:
    """Inference-time rule. Thresholds are the constants above, not a search."""
    if row.get("selector_timed_out") or row.get("timed_out"):
        return False
    if row.get("truncated"):
        return False
    programs = row.get("programs") or 0
    if not 0 < programs < LARGE_CHART:
        return False
    confident = bool(row.get("witness_decided") or row.get("families", 0) <= 1 or row.get("confidence", 0.0) >= CONFIDENCE_MIN)
    stable = bool(row.get("families", 0) <= 1 or row.get("stability", 0.0) >= STABILITY_MIN)
    return confident and stable


def _agreement_ids(root: Path) -> set[str]:
    path = root / "results" / "needle" / "dsl_v2.json"
    if not path.exists():
        return set()
    payload = json.loads(path.read_text())
    rows = [
        row
        for row in payload["rows"]
        if not row.get("selector_timed_out") and not row.get("timed_out") and 0 < row.get("programs", 0) <= AGREEMENT_CHART
    ]
    rows.sort(key=lambda row: row["programs"])
    return {row["id"] for row in rows[:AGREEMENT_TASKS]}


def evaluate(root: Path, limit_s: float = 90.0) -> dict:
    started = time.perf_counter()
    tasks = read_jsonl(root / "data" / "bench_c.jsonl")
    split = json.loads((root / "results" / "needle" / "bench_split.json").read_text())
    by_id = {row["id"]: row for row in split["rows"]}
    targets = [task for task in tasks if by_id[task.id].get("dsl_reason") in _TARGET_REASONS]
    old = _old_scores(root)
    previous = _previous_needle(root)
    agree_on = _agreement_ids(root)
    scope, witness, prior = load_selector(root)
    signal.signal(signal.SIGALRM, _alarm)
    rows = []
    agreements = []
    for index, bench in enumerate(targets):
        _MEMBERSHIP.clear()
        visible = _visible(bench.visible)
        cover = cover_regex(bench.gold_regex)
        row = {
            "id": bench.id,
            "domain": bench.domain,
            "old_reason": by_id[bench.id]["dsl_reason"],
            "dsl": by_id[bench.id]["dsl"],
            "cover": cover.status,
            "old_score": old.get(bench.id),
            "previous_needle": previous.get(bench.id),
        }
        signal.alarm(int(limit_s))
        try:
            facts = identifiable_facts(visible)
            ident_chart = assemble(visible, facts, wrap_optional_alt=True, pack=True)
            identifiable = _pick_identifiable(ident_chart.programs, visible)
            row["identifiable_accuracy"] = _accuracy(identifiable, visible, bench)
            row["identifiable_perfect"] = _one_perfect(identifiable, visible, bench)
            hyps = _hyps(scope, bench, visible)
            assembly = complete_assembly(visible, hyps, v2=True)
            programs = [prog for prog, _score in assembly.programs]
            row["programs"] = len(programs)
            row["truncated"] = assembly.truncated
            row["synthesis_perfect"] = _perfect(programs, visible, bench)
            row["timed_out"] = False
        except _Timeout:
            programs = []
            row["programs"] = 0
            row["truncated"] = False
            row["synthesis_perfect"] = False
            row["timed_out"] = True
            row["identifiable_accuracy"] = 0.5
            row["identifiable_perfect"] = False
        finally:
            signal.alarm(0)
        chosen = None
        info = {"stability": 0.0, "confidence": 0.0, "margin": 0.0, "families": 0, "witness_decided": False, "mode": "direct"}
        select_started = time.perf_counter()
        timed_out = False
        if row["timed_out"] or not programs:
            row["selector_accuracy"] = 0.5
            row["selector_timed_out"] = bool(row["timed_out"])
            row["select_s"] = 0.0
        else:
            _MEMBERSHIP.clear()
            signal.alarm(30)
            try:
                chosen, _programs, _count, info = select_from_programs(
                    programs,
                    visible,
                    witness,
                    prior,
                    stability_mode="direct",
                    with_stability=True,
                    operator_witnesses=True,
                )
            except _Timeout:
                timed_out = True
                chosen = None
            finally:
                signal.alarm(0)
            row["select_s"] = round(time.perf_counter() - select_started, 3)
            row["selector_timed_out"] = timed_out
            row["selector_accuracy"] = 0.5 if timed_out else _accuracy(chosen, visible, bench)
        row["confidence"] = info.get("confidence", 0.0)
        row["margin"] = info.get("margin", 0.0)
        row["stability"] = info.get("stability", 0.0)
        row["families"] = info.get("families", 0)
        row["witness_decided"] = info.get("witness_decided", False)
        row["regex"] = "" if chosen is None else to_regex(chosen, visible)
        row["needle_score"] = row["selector_accuracy"]
        row["difference"] = None if row["old_score"] is None else row["needle_score"] - row["old_score"]
        row["oracle_pair"] = None if row["old_score"] is None else max(row["old_score"], row["needle_score"])
        row["routed_to"] = "needle" if _take_needle(row) else "old"
        if bench.id in agree_on and programs:
            agreements.append(_compare_modes(scope, witness, prior, bench, visible, programs, A_LIMIT_S))
        rows.append(row)
        if index % 10 == 0 or index + 1 == len(targets):
            print(
                f"modes {index + 1}/{len(targets)} {bench.domain} programs {row['programs']} "
                f"acc {row['selector_accuracy']:.3f} select {row['select_s']:.2f}s",
                flush=True,
            )
            (root / "results" / "needle" / "selector_modes_partial.json").write_text(json.dumps({"rows": rows, "agreement": agreements}) + "\n")
    fresh = {row["id"]: row["needle_score"] for row in rows}
    needle_scores = dict(previous)
    needle_scores.update(fresh)
    union_before = _union(old, previous)
    union_after = _union(old, needle_scores)
    routed = []
    for task_id, old_score in old.items():
        if task_id in fresh:
            match = next(row for row in rows if row["id"] == task_id)
            routed.append(match["needle_score"] if match["routed_to"] == "needle" else old_score)
        else:
            routed.append(old_score)
    select_times = [row["select_s"] for row in rows if not row.get("selector_timed_out") and not row.get("timed_out")]
    finished_perfect = [
        row["selector_accuracy"]
        for row in rows
        if row.get("synthesis_perfect") and not row.get("selector_timed_out") and not row.get("timed_out")
    ]
    agreed = [row for row in agreements if row.get("a_finished")]
    payload = {
        "selector_timeouts": sum(1 for row in rows if row.get("selector_timed_out")),
        "assemble_timeouts": sum(1 for row in rows if row.get("timed_out")),
        "select_p50_s": _percentile(select_times, 0.50),
        "select_p95_s": _percentile(select_times, 0.95),
        "select_max_s": max(select_times) if select_times else None,
        "selector_accuracy": _mean([row["selector_accuracy"] for row in rows]),
        "selector_accuracy_given_perfect": _mean(finished_perfect),
        "headline_semantic_accuracy": _headline(root, fresh),
        "hidden_set_oracle": _mean([1.0 if row.get("synthesis_perfect") else 0.0 for row in rows]),
        "support_identifiable_oracle": _mean([1.0 if row.get("identifiable_perfect") else 0.0 for row in rows]),
        "support_identifiable_accuracy": _mean([row["identifiable_accuracy"] for row in rows if row.get("identifiable_accuracy") is not None]),
        "union_before": union_before,
        "union_after": union_after,
        "router_accuracy": _mean(routed),
        "router_took_needle": sum(1 for row in rows if row["routed_to"] == "needle"),
        "agreement_n": len(agreed),
        "agreement_b": _mean([1.0 if row["b_same"] else 0.0 for row in agreed]) if agreed else None,
        "agreement_c": _mean([1.0 if row["c_same"] else 0.0 for row in agreed]) if agreed else None,
        "confidence_min": CONFIDENCE_MIN,
        "stability_min": STABILITY_MIN,
        "chart_max": LARGE_CHART,
        "model_unchanged": True,
        "bench_e_scored": False,
        "seconds": time.perf_counter() - started,
        "agreement": agreements,
        "rows": rows,
    }
    partial = root / "results" / "needle" / "selector_modes_partial.json"
    if partial.exists():
        partial.unlink()
    return payload


def _compare_modes(scope, witness, prior, bench, visible, programs, limit_s: float) -> dict:
    task = Task(bench.domain, visible, Prog("COPY"), [])
    row = {"id": bench.id, "programs": len(programs), "a_finished": False, "b_same": False, "c_same": False}
    chosen = {}
    for mode in ("resynthesize", "cached", "direct"):
        _MEMBERSHIP.clear()
        signal.alarm(int(limit_s if mode == "resynthesize" else 30))
        try:
            picked, _programs, _count, _info = select_from_programs(
                programs,
                visible,
                witness,
                prior,
                stability_mode=mode,
                with_stability=True,
                operator_witnesses=False,
                scope=scope,
                task=task,
                v2=True,
            )
            chosen[mode] = "" if picked is None else to_regex(picked, visible)
        except _Timeout:
            chosen[mode] = None
        finally:
            signal.alarm(0)
    if chosen.get("resynthesize") is None:
        return row
    row["a_finished"] = True
    row["a"] = chosen["resynthesize"]
    row["b_same"] = chosen.get("cached") == chosen["resynthesize"]
    row["c_same"] = chosen.get("direct") == chosen["resynthesize"]
    return row


def render_markdown(payload: dict) -> str:
    before = payload["union_before"]
    after = payload["union_after"]
    conditional = payload.get("selector_accuracy_given_perfect") or 0.0
    headline = payload.get("headline_semantic_accuracy") or 0.0
    p50 = payload.get("select_p50_s") or 0.0
    p95 = payload.get("select_p95_s") or 0.0
    slow = payload.get("select_max_s") or 0.0
    lines = [
        "## Selector witnesses",
        "",
        "The recognizer stays the frozen 1.226M model. Selection no longer rebuilds the version space "
        "to score stability. A complete mixed character set is preferred to its observed hull by a probe "
        "that changes one character of a class the supports already show. An integer outside the observed "
        "endpoints abstains, so the digit class keeps that tie. BENCH-E stays sealed.",
        "",
        f"On the 154 partial-range and mixed-class tasks, selector timeouts are {payload['selector_timeouts']}. "
        f"Select latency is p50 {p50:.3f}s, p95 {p95:.3f}s, max {slow:.3f}s. "
        f"Given a hidden-set-perfect chart, selector accuracy is {conditional:.3f}. "
        f"The BENCH-C headline, with these 154 scores replacing the previous ones, is {headline:.3f}.",
        "",
        f"Where mode A finished on {payload['agreement_n']} small charts, mode B agreed on {payload.get('agreement_b') or 0:.3f} "
        f"and mode C agreed on {payload.get('agreement_c') or 0:.3f}. "
        "The reported selector uses mode C plus the operator probes.",
        "",
        f"Old symbolic rule {before['old']:.3f}. Previous Needle {before['needle']:.3f}. "
        f"Oracle union before this pass {before['oracle_union']:.3f}. "
        f"After this pass the Needle score is {after['needle']:.3f} and the oracle union is {after['oracle_union']:.3f}. "
        f"Needle is strictly better on {after['needle_better']} tasks, the old rule on {after['old_better']}, "
        f"and they tie on {after['tie']}.",
        "",
        "The router uses only inference-time signals: a witness decision or grammar confidence of at least "
        f"{payload['confidence_min']:.2f}, stability {payload['stability_min']:.1f} on the direct check, "
        f"and a chart smaller than {payload['chart_max']} that was not truncated. Otherwise it keeps the old rule. "
        f"It takes Needle on {payload['router_took_needle']} of the 154 tasks. "
        f"Routed BENCH-C accuracy is {payload['router_accuracy']:.3f}.",
        "",
        f"Support-identifiable oracle, the observed integer hull and the observed character hull, "
        f"is {payload['support_identifiable_oracle']:.3f} perfect on the hidden set "
        f"(mean accuracy {payload['support_identifiable_accuracy']:.3f}). "
        f"The hidden-set synthesis oracle is {payload['hidden_set_oracle']:.3f}. "
        "The gap is the part a finite hidden sample can reward without the supports showing it.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Oracle union, old + previous Needle | {before['oracle_union']:.3f} |",
        f"| Oracle union, old + this Needle | {after['oracle_union']:.3f} |",
        f"| BENCH-C after selector timeouts | {headline:.3f} |",
        f"| Selector given perfect candidate | {conditional:.3f} |",
        f"| Selector timeouts | {payload['selector_timeouts']} |",
        f"| Select p50 / p95 / max seconds | {p50:.3f} / {p95:.3f} / {slow:.3f} |",
        f"| Routed accuracy | {payload['router_accuracy']:.3f} |",
        f"| Support-identifiable oracle | {payload['support_identifiable_oracle']:.3f} |",
        f"| Hidden-set synthesis oracle | {payload['hidden_set_oracle']:.3f} |",
        "",
    ]
    return "\n".join(lines) + "\n"


def _main() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = evaluate(root)
    dest = root / "results" / "needle" / "selector_modes.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    md = root / "results" / "needle" / "selector_modes.md"
    md.write_text(render_markdown(payload), encoding="utf-8")
    print(
        f"timeouts {payload['selector_timeouts']} conditional {payload['selector_accuracy_given_perfect']} "
        f"headline {payload['headline_semantic_accuracy']} union {payload['union_after']['oracle_union']} "
        f"router {payload['router_accuracy']}",
        flush=True,
    )


if __name__ == "__main__":
    _main()
