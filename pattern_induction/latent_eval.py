"""Candidate-expansion oracle and the BENCH-C ambiguity taxonomy.

No prior is trained. The question is whether generic latent rewrites put the
hidden language in the candidate set. BENCH-E is not read.
"""

from __future__ import annotations

import json
import signal
import time
from collections import Counter
from pathlib import Path

from pattern_induction.dsl_cover import cover_regex
from pattern_induction.latent import classify_gap, expand_program, expand_programs
from pattern_induction.operator_witness import identifiable_facts
from pattern_induction.program import MAX_LEN, MAX_SUPPORT, Prog, _MEMBERSHIP, describe, execute
from pattern_induction.selector_modes import _hyps, _pick_identifiable
from pattern_induction.select_train import complete_assembly, load_selector
from pattern_induction.tasks import read_jsonl
from pattern_induction.version_space import assemble

_ORDER = (
    "unseen width/bound",
    "unseen class member",
    "unseen alternative",
    "unseen optional branch",
    "semantic numeric interval",
    "open quantifier",
    "other",
)


class _Timeout(Exception):
    pass


def _alarm(signum, frame) -> None:
    raise _Timeout()


def _visible(texts: list[str]) -> list[str]:
    return [text[:MAX_LEN] for text in texts[:MAX_SUPPORT]]


def _perfect_prog(programs, visible, bench) -> bool:
    probe = [(text, True) for text in bench.hidden_positives] + [(text, False) for text in bench.hidden_negatives]
    return any(all(execute(prog, text, visible) == label for text, label in probe) for prog in programs)


def _stored_perfect(root: Path) -> dict[str, bool]:
    perfect: dict[str, bool] = {}
    coverage = json.loads((root / "results" / "needle" / "coverage.json").read_text())
    for row in coverage["rows"]:
        if "synthesis_perfect" in row:
            perfect[row["id"]] = bool(row["synthesis_perfect"])
    dsl = json.loads((root / "results" / "needle" / "dsl_v2.json").read_text())
    for row in dsl["rows"]:
        if "synthesis_perfect" in row:
            perfect[row["id"]] = bool(row["synthesis_perfect"])
    split = json.loads((root / "results" / "needle" / "bench_split.json").read_text())
    for row in split["rows"]:
        perfect.setdefault(row["id"], bool(row.get("synthesis_perfect")))
    return perfect


def _identifiable(visible: list[str]):
    chart = assemble(visible, identifiable_facts(visible), wrap_optional_alt=True, pack=True)
    return _pick_identifiable(chart.programs, visible)


def _seeds(programs: list[Prog], identifiable: Prog | None) -> list[Prog]:
    rows: list[Prog] = []
    seen = set()

    def add(prog: Prog | None) -> None:
        if prog is None:
            return
        key = describe(prog)
        if key in seen:
            return
        seen.add(key)
        rows.append(prog)

    add(identifiable)
    ordered = sorted(programs, key=lambda prog: (prog.nodes(), describe(prog)))
    for prog in ordered:
        add(prog)
        if len(rows) >= 5:
            break
    return rows


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def evaluate(root: Path, limit_s: float = 45.0) -> dict:
    started = time.perf_counter()
    tasks = read_jsonl(root / "data" / "bench_c.jsonl")
    stored = _stored_perfect(root)
    split = {row["id"]: row for row in json.loads((root / "results" / "needle" / "bench_split.json").read_text())["rows"]}
    scope = None
    signal.signal(signal.SIGALRM, _alarm)
    rows = []
    for index, bench in enumerate(tasks):
        _MEMBERSHIP.clear()
        visible = _visible(bench.visible)
        cover = cover_regex(bench.gold_regex)
        identifiable = _identifiable(visible)
        gap = classify_gap(identifiable, visible, bench.gold_regex, bench.hidden_positives)
        row = {
            "id": bench.id,
            "domain": bench.domain,
            "cover": cover.status,
            "reason": split.get(bench.id, {}).get("dsl_reason") or cover.reason,
            "gap": gap,
            "stored_perfect": bool(stored.get(bench.id)),
        }
        if row["stored_perfect"]:
            row["current_perfect"] = True
            row["expanded_perfect"] = True
            row["candidates"] = split.get(bench.id, {}).get("programs")
            row["expanded_candidates"] = row["candidates"]
            row["assembled"] = False
        else:
            if scope is None:
                scope, _witness, _prior = load_selector(root)
            signal.alarm(int(limit_s))
            programs: list[Prog] = []
            try:
                assembly = complete_assembly(visible, _hyps(scope, bench, visible), v2=True)
                programs = [prog for prog, _score in assembly.programs]
                row["assembled"] = True
                row["timed_out"] = False
            except _Timeout:
                row["assembled"] = False
                row["timed_out"] = True
            finally:
                signal.alarm(0)
            row["current_perfect"] = _perfect_prog(programs, visible, bench) if programs else False
            seeds = _seeds(programs, identifiable)
            signal.alarm(20)
            try:
                expanded = expand_programs(seeds, visible, cap=400) if seeds else []
            except _Timeout:
                expanded = seeds
            finally:
                signal.alarm(0)
            # The chart itself remains a candidate. Expansion only adds to it.
            row["candidates"] = len(programs)
            row["chart_candidates"] = len(programs)
            row["rewrite_candidates"] = len(expanded)
            row["expanded_candidates"] = len({describe(prog) for prog in programs + expanded})
            pool = expanded if row["current_perfect"] else programs + expanded
            row["expanded_perfect"] = row["current_perfect"] or _perfect_prog(pool, visible, bench)
        rows.append(row)
        if index % 20 == 0 or index + 1 == len(tasks):
            current = sum(1 for item in rows if item["current_perfect"])
            widened = sum(1 for item in rows if item["expanded_perfect"])
            print(
                f"latent {index + 1}/{len(tasks)} {bench.domain} current {current}/{len(rows)} expanded {widened}/{len(rows)}",
                flush=True,
            )
            (root / "results" / "needle" / "latent_oracle_partial.json").write_text(json.dumps(rows) + "\n")
    payload = _summarize(rows, time.perf_counter() - started)
    partial = root / "results" / "needle" / "latent_oracle_partial.json"
    if partial.exists():
        partial.unlink()
    return payload


def _summarize(rows: list[dict], seconds: float) -> dict:
    n = len(rows)
    gaps = [row["gap"] for row in rows if row["gap"] != "identifiable"]
    counts = Counter(gaps)
    expanded_rows = [row for row in rows if not row.get("stored_perfect")]
    recovered = sum(1 for row in expanded_rows if row["expanded_perfect"] and not row["current_perfect"])
    return {
        "n": n,
        "current_synthesis_oracle": sum(1 for row in rows if row["current_perfect"]) / max(n, 1),
        "expanded_synthesis_oracle": sum(1 for row in rows if row["expanded_perfect"]) / max(n, 1),
        "current_perfect": sum(1 for row in rows if row["current_perfect"]),
        "expanded_perfect": sum(1 for row in rows if row["expanded_perfect"]),
        "recovered": recovered,
        "mean_chart": _mean([row["candidates"] for row in rows if isinstance(row.get("candidates"), int)]),
        "mean_rewrites": _mean(
            [row["rewrite_candidates"] for row in rows if isinstance(row.get("rewrite_candidates"), int)]
        ),
        "mean_expanded_candidates": _mean(
            [row["expanded_candidates"] for row in expanded_rows if isinstance(row.get("expanded_candidates"), int)]
        ),
        "identifiable_n": sum(1 for row in rows if row["gap"] == "identifiable"),
        "gap_n": len(gaps),
        "taxonomy": {name: counts.get(name, 0) for name in _ORDER},
        "assemble_timeouts": sum(1 for row in rows if row.get("timed_out")),
        "seconds": seconds,
        "trained": False,
        "bench_e_scored": False,
        "rows": rows,
    }


def render_markdown(payload: dict) -> str:
    taxonomy = payload["taxonomy"]
    lines = [
        "## Latent candidate oracle",
        "",
        "The support-identifiable program is what the four examples justify. "
        "BENCH-C often asks for the latent rule those examples did not reveal. "
        "This pass adds generic rewrites — wider bounds, conventional numeric intervals, "
        "natural character superclasses, open repeats, alternations lifted to a class, "
        "and field structure read off the supports, including thousands groups, repeated fields, and optional affixes — "
        "and asks whether the hidden language is then in the set. No prior is trained. BENCH-E stays sealed.",
        "",
        f"Current synthesis oracle on {payload['n']} tasks is {payload['current_synthesis_oracle']:.3f} "
        f"({payload['current_perfect']}/{payload['n']}). "
        f"Expanded synthesis oracle is {payload['expanded_synthesis_oracle']:.3f} "
        f"({payload['expanded_perfect']}/{payload['n']}). "
        f"The rewrites recover {payload['recovered']} tasks the current chart missed. "
        f"On the tasks that were reassembled, the mean chart has {payload['mean_chart'] or 0:.0f} programs "
        f"and the mean latent rewrite set has {payload.get('mean_rewrites') or 0:.0f}. "
        f"Assembly timeouts: {payload['assemble_timeouts']}.",
        "",
        f"The support-identifiable program is hidden-perfect on {payload['identifiable_n']} tasks. "
        f"The other {payload['gap_n']} fall into the ambiguity classes below. "
        "Those classes are the curriculum for a later prior, not extra benchmark answers.",
        "",
        "| Ambiguity | Tasks |",
        "| --- | ---: |",
    ]
    for name in _ORDER:
        lines.append(f"| {name} | {taxonomy.get(name, 0)} |")
    lines.extend(
        [
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Current synthesis oracle | {payload['current_synthesis_oracle']:.3f} |",
            f"| Expanded synthesis oracle | {payload['expanded_synthesis_oracle']:.3f} |",
            f"| Recovered by expansion | {payload['recovered']} |",
            f"| Mean chart on reassembled tasks | {payload['mean_chart'] or 0:.0f} |",
            f"| Mean latent rewrites on reassembled tasks | {payload.get('mean_rewrites') or 0:.0f} |",
            "",
        ]
    )
    oracle = payload["expanded_synthesis_oracle"]
    if oracle >= 0.99:
        lines.append(
            "The expanded oracle is high enough that the hidden language is usually in the set. "
            "The next experiment is to rank it. No prior is trained in this pass.\n"
        )
    else:
        lines.append(
            "The expanded oracle is not yet above 0.98, so these rewrites are not a basis for training a language prior. "
            "What remains is mostly closed lexicons, supports clipped at 32 characters, and optional branches "
            "that the four examples never show. BENCH-E stays sealed.\n"
        )
    return "\n".join(lines) + "\n"


def _main() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = evaluate(root)
    (root / "results" / "needle" / "latent_oracle.json").write_text(json.dumps(payload, indent=2) + "\n")
    (root / "results" / "needle" / "latent_oracle.md").write_text(render_markdown(payload), encoding="utf-8")
    print(
        f"current {payload['current_synthesis_oracle']:.3f} expanded {payload['expanded_synthesis_oracle']:.3f} "
        f"recovered {payload['recovered']} identifiable {payload['identifiable_n']}",
        flush=True,
    )


if __name__ == "__main__":
    _main()
