"""Exact-slice coverage after alignment facts join nonempty charts.

The frozen recognizer stays as it is. Facts the four supports already show
are added even when the chart is nonempty, and equivalent concatenations
collapse while the chart is built. This pass does not score BENCH-E.
"""

from __future__ import annotations

import json
import re
import signal
import time
from pathlib import Path

from pattern_induction.chart import LOCAL_K, same_fact
from pattern_induction.contingent import contingent_optionals
from pattern_induction.document_facts import alignment_kind, document_facts
from pattern_induction.program import MAX_LEN, MAX_SUPPORT, Prog, Task, _MEMBERSHIP, agrees, execute
from pattern_induction.scope import collate_scope, hypotheses_from_scope
from pattern_induction.select_train import load_selector, select_program
from pattern_induction.tasks import read_jsonl
from pattern_induction.version_space import assemble


class _Timeout(Exception):
    pass


def _alarm(signum, frame) -> None:
    raise _Timeout()


def _visible(texts: list[str]) -> list[str]:
    return [text[:MAX_LEN] for text in texts[:MAX_SUPPORT]]


def _probe(bench) -> list[tuple[str, bool]]:
    return [(text, True) for text in bench.hidden_positives] + [(text, False) for text in bench.hidden_negatives]


def _accuracy(prog, visible: list[str], bench) -> float:
    positives = bench.hidden_positives
    negatives = bench.hidden_negatives
    if prog is None:
        return 0.5
    recall = sum(bool(execute(prog, text, visible)) for text in positives) / max(len(positives), 1)
    rejection = sum(not execute(prog, text, visible) for text in negatives) / max(len(negatives), 1)
    return 0.5 * (recall + rejection)


def _hyps(scope, bench, visible: list[str]) -> list:
    task = Task(bench.domain, visible, Prog("COPY"), [])
    batch = collate_scope([task])
    hyps = hypotheses_from_scope(scope, batch, [visible], LOCAL_K)[0]
    for hyp in contingent_optionals(visible):
        if any(same_fact(hyp, kept, False) for kept in hyps):
            continue
        hyps.append(hyp)
    return hyps


def _run_widths(text: str, kind: str) -> list[int]:
    pred = {"digit": str.isdigit, "upper": str.isupper, "lower": str.islower}[kind]
    widths = []
    start = None
    for index, ch in enumerate(text + "\0"):
        if index < len(text) and pred(ch):
            if start is None:
                start = index
        elif start is not None:
            widths.append(index - start)
            start = None
    return widths


_QUANT = (
    (r"\\d", "digit"),
    (r"\[0-9\]", "digit"),
    (r"\[A-Z\]", "upper"),
    (r"\[a-z\]", "lower"),
)


def gap_kind(gold: str, supports: list[str]) -> str:
    """Why a gold pattern is or is not fixed by the four supports.

    A bound is unobserved when either endpoint is missing from that class's run widths.
    A branch is unobserved when its literal never occurs. An optional is
    unobserved when every support has the same presence for that literal.
    """
    reasons = []
    for token, kind in _QUANT:
        for match in re.finditer(token + r"\{(\d+)(?:,(\d+))?\}", gold):
            lo = int(match.group(1))
            hi = int(match.group(2) or match.group(1))
            if lo == hi:
                continue
            seen = {width for text in supports for width in _run_widths(text, kind)}
            if lo not in seen or hi not in seen:
                reasons.append("unobserved_bound")
                break
        if reasons:
            break
    for match in re.finditer(r"\((?:\?:)?([^()]*\|[^()]*)\)", gold):
        branches = [re.sub(r"\\(.)", r"\1", part) for part in match.group(1).split("|")]
        literals = [part for part in branches if part and not re.search(r"[\\\[\]{}*+?]", part)]
        if literals and any(all(part not in text for text in supports) for part in literals):
            reasons.append("unobserved_branch")
            break
    for match in re.finditer(r"\((?:\?:)?([^()]*)\)\?", gold):
        body = re.sub(r"\\(.)", r"\1", match.group(1))
        if body and not re.search(r"[\\\[\]{}*+?]", body):
            present = sum(body in text for text in supports)
            if present == 0 or present == len(supports):
                reasons.append("unobserved_optional")
                break
    if not reasons:
        return "visible"
    return reasons[0]


def evaluate(root: Path, limit_s: float = 90.0, select: bool = True) -> dict:
    scope, witness, prior = load_selector(root)
    saved = json.loads((root / "results" / "needle" / "bench_split.json").read_text())
    by_id = {row["id"]: row for row in saved["rows"]}
    benches = read_jsonl(root / "data" / "bench_c.jsonl")
    exact = [bench for bench in benches if by_id[bench.id]["dsl"] == "exact"]
    signal.signal(signal.SIGALRM, _alarm)
    rows = []
    started = time.perf_counter()
    for index, bench in enumerate(exact):
        visible = _visible(bench.visible)
        previous = by_id[bench.id]
        kind = alignment_kind(visible)
        _MEMBERSHIP.clear()
        hyps = _hyps(scope, bench, visible)
        extra = [hyp for hyp in document_facts(visible) if not any(same_fact(hyp, kept, True) for kept in hyps)]
        probe = _probe(bench)
        programs: list = []
        perfect = False
        truncated = False
        assemble_s = 0.0
        timed_out = False
        signal.alarm(int(limit_s))
        try:
            t0 = time.perf_counter()
            if extra:
                assembly = assemble(visible, hyps + extra, wrap_optional_alt=True, pack=True)
            else:
                assembly = assemble(visible, hyps)
            assemble_s = time.perf_counter() - t0
            programs = [prog for prog, _score in assembly.programs]
            truncated = bool(assembly.truncated)
            perfect = bool(programs) and any(agrees(prog, visible, probe) for prog in programs)
        except _Timeout:
            timed_out = True
        finally:
            signal.alarm(0)
        align_perfect = False
        if extra and not timed_out:
            signal.alarm(int(limit_s))
            try:
                aligned = assemble(visible, document_facts(visible), wrap_optional_alt=True, pack=True)
                align_perfect = bool(aligned.programs) and any(agrees(prog, visible, probe) for prog, _score in aligned.programs)
            except _Timeout:
                align_perfect = False
            finally:
                signal.alarm(0)
        was_empty = previous.get("languages", 0) == 0
        was_perfect = bool(previous.get("synthesis_perfect"))
        row = {
            "id": bench.id,
            "domain": bench.domain,
            "alignment": kind,
            "was_empty": was_empty,
            "was_perfect": was_perfect,
            "extra_facts": len(extra),
            "programs": len(programs),
            "truncated": truncated,
            "synthesis_perfect": perfect,
            "alignment_perfect": align_perfect,
            "assemble_s": round(assemble_s, 3),
            "timed_out": timed_out,
            "gap": gap_kind(bench.gold_regex, visible) if not perfect else "",
            "gold": bench.gold_regex,
            "supports": visible,
        }
        rows.append(row)
        if (index + 1) % 20 == 0:
            partial = root / "results" / "needle" / "coverage_partial.json"
            partial.write_text(json.dumps(rows) + "\n")
        if index % 10 == 0 or index + 1 == len(exact):
            done = [item for item in rows if item["synthesis_perfect"]]
            print(
                f"coverage {index + 1}/{len(exact)} {bench.domain} programs {len(programs)} "
                f"perfect {int(perfect)} oracle {len(done)}/{len(rows)} ({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
    if select:
        _select(scope, witness, prior, exact, rows, limit_s)
        _reuse_selector(rows, by_id)
    payload = _summarize(rows, time.perf_counter() - started)
    return payload


def _reuse_selector(rows: list[dict], by_id: dict) -> None:
    for row in rows:
        if row.get("selector_reused"):
            row["selector_accuracy"] = by_id[row["id"]].get("selector_accuracy", 0.5)


def _select(scope, witness, prior, exact, rows: list[dict], limit_s: float) -> None:
    by_id = {bench.id: bench for bench in exact}
    for index, row in enumerate(rows):
        if row["timed_out"] or row["programs"] == 0:
            row["selector_accuracy"] = 0.5
            row["selector_timed_out"] = bool(row["timed_out"])
            row["selector_skipped"] = True
            continue
        # A chart with no new fact is the frozen chart. Selection is unchanged.
        if row["extra_facts"] == 0 and not row["was_empty"]:
            row["selector_accuracy"] = None
            row["selector_timed_out"] = False
            row["selector_skipped"] = False
            row["selector_reused"] = True
            continue
        bench = by_id[row["id"]]
        visible = row["supports"]
        task = Task(bench.domain, visible, Prog("COPY"), [])
        chosen = None
        timed_out = False
        signal.alarm(int(limit_s))
        try:
            chosen, _programs, _count = select_program(scope, witness, prior, task, with_stability=True)
        except _Timeout:
            timed_out = True
            chosen = None
        finally:
            signal.alarm(0)
        row["selector_accuracy"] = 0.5 if timed_out else _accuracy(chosen, visible, bench)
        row["selector_timed_out"] = timed_out
        row["selector_skipped"] = False
        row["selector_reused"] = False
        if index % 10 == 0 or timed_out or index + 1 == len(rows):
            print(
                f"select {index + 1}/{len(rows)} {row['domain']} programs {row['programs']} "
                f"acc {row['selector_accuracy']:.3f} timeout {int(timed_out)}",
                flush=True,
            )


def _selector_mean(rows: list[dict]) -> float | None:
    scored = [row["selector_accuracy"] for row in rows if row.get("selector_accuracy") is not None]
    if not scored:
        return None
    return sum(scored) / len(scored)


def _selector_given_perfect(rows: list[dict]) -> float | None:
    scored = [
        row["selector_accuracy"]
        for row in rows
        if row.get("synthesis_perfect") and row.get("selector_accuracy") is not None and not row.get("selector_timed_out")
    ]
    if not scored:
        return None
    return sum(scored) / len(scored)


def _summarize(rows: list[dict], seconds: float) -> dict:
    n = len(rows)
    perfect = sum(1 for row in rows if row["synthesis_perfect"])
    empty = sum(1 for row in rows if row["programs"] == 0 and not row["timed_out"])
    nonempty_misses = [row for row in rows if not row["was_empty"] and not row["was_perfect"]]
    recovered = [row for row in nonempty_misses if row["synthesis_perfect"]]
    still = [row for row in nonempty_misses if not row["synthesis_perfect"]]
    visible_misses = [row for row in still if row["gap"] == "visible"]
    unobserved = [row for row in still if row["gap"] != "visible"]
    kept_perfect = sum(1 for row in rows if row["was_perfect"] and row["synthesis_perfect"])
    lost_perfect = [row["id"] for row in rows if row["was_perfect"] and not row["synthesis_perfect"]]
    empty_perfect = sum(1 for row in rows if row["was_empty"] and row["synthesis_perfect"])
    # Identifiability ceiling: every exact task except misses that need an unobserved fact.
    ceiling_misses = unobserved
    # Also count formerly empty tasks that are still imperfect for an unobserved reason.
    empty_unobserved = [
        row for row in rows if row["was_empty"] and not row["synthesis_perfect"] and row["gap"] != "visible"
    ]
    identifiable_failures = {row["id"] for row in ceiling_misses} | {row["id"] for row in empty_unobserved}
    return {
        "exact_n": n,
        "exact_empty": empty,
        "exact_synthesis_oracle": perfect / n,
        "exact_synthesis_perfect": perfect,
        "assemble_timeouts": sum(1 for row in rows if row["timed_out"]),
        "previously_perfect_kept": kept_perfect,
        "previously_perfect_lost": lost_perfect,
        "empty_now_perfect": empty_perfect,
        "nonempty_misses": len(nonempty_misses),
        "nonempty_misses_recovered": len(recovered),
        "nonempty_misses_remaining": len(still),
        "remaining_visible": len(visible_misses),
        "remaining_unobserved": len(unobserved),
        "identifiable_failures": len(identifiable_failures),
        "recoverable_oracle": (n - len(identifiable_failures)) / n,
        "alignment_perfect_on_misses": sum(1 for row in nonempty_misses if row["alignment_perfect"]),
        "selector_timeouts": sum(1 for row in rows if row.get("selector_timed_out")),
        "selector_accuracy": _selector_mean(rows),
        "selector_accuracy_given_perfect": _selector_given_perfect(rows),
        "seconds": seconds,
        "dsl_unchanged": True,
        "model_unchanged": True,
        "rows": rows,
    }


def _refresh_gaps(rows: list[dict]) -> None:
    for row in rows:
        row["gap"] = "" if row["synthesis_perfect"] else gap_kind(row["gold"], row["supports"])


def _write_coverage(root: Path, rows: list[dict], started: float) -> dict:
    payload = _summarize(rows, time.perf_counter() - started)
    dest = root / "results" / "needle" / "coverage.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--no-select", action="store_true")
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    if args.select_only:
        started = time.perf_counter()
        saved = json.loads((args.root / "results" / "needle" / "coverage.json").read_text())
        rows = saved["rows"]
        _refresh_gaps(rows)
        scope, witness, prior = load_selector(args.root)
        benches = read_jsonl(args.root / "data" / "bench_c.jsonl")
        split = json.loads((args.root / "results" / "needle" / "bench_split.json").read_text())
        by_id = {row["id"]: row for row in split["rows"]}
        exact = [bench for bench in benches if by_id[bench.id]["dsl"] == "exact"]
        signal.signal(signal.SIGALRM, _alarm)
        _select(scope, witness, prior, exact, rows, 90.0)
        _reuse_selector(rows, by_id)
        payload = _write_coverage(args.root, rows, started)
    else:
        payload = evaluate(args.root, select=not args.no_select)
        dest = args.root / "results" / "needle" / "coverage.json"
        dest.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"oracle {payload['exact_synthesis_oracle']:.3f} "
        f"empty {payload['exact_empty']} "
        f"recovered {payload['nonempty_misses_recovered']}/{payload['nonempty_misses']} "
        f"ceiling {payload['recoverable_oracle']:.3f} "
        f"selector {payload.get('selector_accuracy')} "
        f"conditional {payload.get('selector_accuracy_given_perfect')} "
        f"timeouts {payload.get('selector_timeouts')}",
        flush=True,
    )


if __name__ == "__main__":
    _main()
