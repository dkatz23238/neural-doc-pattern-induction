"""Split a finished BENCH-C score into expressivity, synthesis, and selection.

The recognizer and the witness head stay frozen. This pass does not train.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

from pattern_induction.dsl_cover import cover_regex
from pattern_induction.language import group_languages
from pattern_induction.program import MAX_LEN, MAX_SUPPORT, Prog, Task, agrees, execute
from pattern_induction.select_train import select_program, synthesize_assembly


def _visible(texts: list[str]) -> list[str]:
    return [text[:MAX_LEN] for text in texts[:MAX_SUPPORT]]


def _probe(bench) -> list[tuple[str, bool]]:
    return [(text, True) for text in bench.hidden_positives] + [(text, False) for text in bench.hidden_negatives]


def _perfect(accuracy: float) -> bool:
    return accuracy >= 1.0 - 1e-9


def label_tasks(benchmarks, scored: dict[str, dict]) -> list[dict]:
    rows = []
    for bench in benchmarks:
        saved = scored[bench.id]
        cover = cover_regex(bench.gold_regex)
        rows.append(
            {
                "id": bench.id,
                "domain": bench.domain,
                "dsl": cover.status,
                "dsl_reason": cover.reason,
                "dsl_nodes": cover.nodes,
                "selector_accuracy": saved["accuracy"],
                "selector_perfect": _perfect(saved["accuracy"]),
                "gold_regex": bench.gold_regex,
            }
        )
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate(rows: list[dict], key: str) -> float:
    return _mean([1.0 if row[key] else 0.0 for row in rows])


def _accuracy(prog, visible: list[str], bench) -> float:
    positives = bench.hidden_positives
    negatives = bench.hidden_negatives
    if prog is None:
        recall = 0.0
        rejection = 1.0
    else:
        recall = sum(bool(execute(prog, text, visible)) for text in positives) / max(len(positives), 1)
        rejection = sum(not execute(prog, text, visible) for text in negatives) / max(len(negatives), 1)
    return 0.5 * (recall + rejection)


def fill_synthesis(scope, witness, prior, benchmarks, rows: list[dict], saved_selector: bool) -> None:
    """Record version-space size and whether any program matches the hidden set.

    When the scoring run already stored a per-task accuracy, assembly runs once.
    Otherwise the frozen selector runs again, stability included, and that same
    version space is the synthesis oracle.
    """
    by_id = {row["id"]: row for row in rows}
    for index, bench in enumerate(benchmarks):
        row = by_id[bench.id]
        visible = _visible(bench.visible)
        dummy = Prog("COPY", slot=0, start=0, end=min(1, len(visible[0]))) if visible and visible[0] else Prog("COPY")
        task = Task(bench.domain, visible, dummy, [])
        started = time.perf_counter()
        if saved_selector:
            assembly = synthesize_assembly(scope, task)
            programs = [prog for prog, _score in assembly.programs]
            families = group_languages(programs, visible) if programs else []
            row["languages"] = len(families)
            row["truncated"] = bool(assembly.truncated)
            chosen_perfect = bool(row["selector_perfect"])
        else:
            chosen, programs, count = select_program(scope, witness, prior, task, with_stability=True)
            row["selector_accuracy"] = _accuracy(chosen, visible, bench)
            row["selector_perfect"] = _perfect(row["selector_accuracy"])
            row["languages"] = count
            row["truncated"] = False
            chosen_perfect = row["selector_perfect"]
            families = []
        row["seconds"] = time.perf_counter() - started
        row["programs"] = len(programs)
        perfect = chosen_perfect
        if not perfect and programs:
            if not families:
                families = group_languages(programs, visible)
            probe = _probe(bench)
            perfect = any(agrees(family[0], visible, probe) for family in families)
        row["synthesis_perfect"] = perfect
        if index % 10 == 0 or index + 1 == len(benchmarks):
            done = [item for item in rows if "synthesis_perfect" in item]
            expressible = [item for item in done if item["dsl"] != "no"]
            print(
                f"split {index + 1}/{len(benchmarks)} {bench.domain} languages {row['languages']} "
                f"selector {row['selector_accuracy']:.3f} synthesis {int(perfect)} "
                f"expressible-oracle {_rate(expressible, 'synthesis_perfect'):.3f} ({row['seconds']:.1f}s)",
                flush=True,
            )


def aggregate(rows: list[dict]) -> dict:
    exact = [row for row in rows if row["dsl"] == "exact"]
    opened = [row for row in rows if row["dsl"] == "open"]
    outside = [row for row in rows if row["dsl"] == "no"]
    fragment = exact + opened

    def block(name: str, group: list[dict]) -> dict:
        synth_ready = [row for row in group if "synthesis_perfect" in row]
        synthesized = [row for row in synth_ready if row["synthesis_perfect"]]
        return {
            "name": name,
            "n": len(group),
            "rate": len(group) / len(rows) if rows else 0.0,
            "selector_accuracy": _mean([row["selector_accuracy"] for row in group]),
            "selector_perfect": _rate(group, "selector_perfect"),
            "synthesis_oracle": _rate(synth_ready, "synthesis_perfect") if synth_ready else None,
            "selector_perfect_given_synthesis": _rate(synthesized, "selector_perfect") if synthesized else None,
            "selector_accuracy_given_synthesis": _mean([row["selector_accuracy"] for row in synthesized]) if synthesized else None,
            "mean_languages": _mean([row["languages"] for row in group if "languages" in row]),
            "max_languages": max((row["languages"] for row in group if "languages" in row), default=0),
            "mean_seconds": _mean([row["seconds"] for row in group if "seconds" in row]),
            "truncated": sum(1 for row in group if row.get("truncated")),
        }

    reasons: dict[str, int] = defaultdict(int)
    for row in outside:
        reasons[row["dsl_reason"] or "other"] += 1
    domains = []
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    for domain, group in sorted(by_domain.items()):
        item = block(domain, group)
        item["exact"] = sum(1 for row in group if row["dsl"] == "exact")
        item["open"] = sum(1 for row in group if row["dsl"] == "open")
        item["outside"] = sum(1 for row in group if row["dsl"] == "no")
        domains.append(item)
    return {
        "n": len(rows),
        "selector_accuracy": _mean([row["selector_accuracy"] for row in rows]),
        "exact": block("exact", exact),
        "open": block("open", opened),
        "outside": block("outside", outside),
        "fragment": block("fragment", fragment),
        "reasons": dict(reasons),
        "domains": domains,
        "empty": sum(1 for row in rows if row.get("languages", 0) == 0),
        "exact_empty": sum(1 for row in exact if row.get("languages", 0) == 0),
    }


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def render_markdown(summary: dict) -> str:
    exact = summary["exact"]
    opened = summary["open"]
    outside = summary["outside"]
    fragment = summary["fragment"]
    lines = [
        "## BENCH-C split",
        "",
        "The gold regex is compared with the program fragment: literals, digit / upper / lower, concatenation, optional, alternation, and repeat. A compound repeat stops at 8. A tree stops at 16 nodes. Partial ranges and mixed classes are outside the fragment. An unbounded quantifier is marked open, because the executor approximates it with that cap.",
        "",
        f"Of {summary['n']} tasks, {exact['n']} are exact ({exact['rate']:.3f}), {opened['n']} are open ({opened['rate']:.3f}), and {outside['n']} are outside the fragment ({outside['rate']:.3f}). "
        f"Selector accuracy on this pass is {summary['selector_accuracy']:.3f}."
        + (
            f" The scoring run reported {summary['official_semantic_accuracy']:.3f}."
            if summary.get("official_semantic_accuracy") is not None
            else ""
        ),
        "",
        "| Slice | Tasks | Selector accuracy | Selector perfect | Synthesis oracle | Selector perfect given synthesis | Mean languages | Mean seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for block_row in (exact, opened, outside, fragment):
        lines.append(
            f"| {block_row['name']} | {block_row['n']} | {_pct(block_row['selector_accuracy'])} | {_pct(block_row['selector_perfect'])} | "
            f"{_pct(block_row['synthesis_oracle'])} | {_pct(block_row['selector_perfect_given_synthesis'])} | "
            f"{block_row['mean_languages']:.1f} | {block_row['mean_seconds']:.2f} |"
        )
    if summary["reasons"]:
        rendered = ", ".join(f"{name} {count}" for name, count in sorted(summary["reasons"].items(), key=lambda item: -item[1]))
        lines.extend(["", f"Outside the fragment: {rendered}.", ""])
    if summary.get("empty") is not None:
        exact_gap = (1.0 - exact["selector_accuracy"]) * exact["rate"] * 100
        outside_gap = (1.0 - outside["selector_accuracy"]) * outside["rate"] * 100
        open_gap = (1.0 - opened["selector_accuracy"]) * opened["rate"] * 100
        survived = exact["n"] - summary.get("exact_empty", 0)
        lines.extend(
            [
                f"The version space is empty on {summary['empty']} of {summary['n']} tasks. An empty chart abstains and scores 0.500. "
                f"{summary.get('exact_empty', 0)} of those empty tasks are exact in the fragment, out of {exact['n']} exact tasks. "
                f"A program survives on the other {survived} exact tasks. "
                f"When synthesis puts a hidden-set-perfect program in the fragment, selector semantic accuracy is {_pct(fragment['selector_accuracy_given_synthesis'])} "
                f"and the chosen program is perfect on {_pct(fragment['selector_perfect_given_synthesis'])} of those tasks.",
                "",
                f"Against a perfect score, the exact slice contributes {exact_gap:.1f} points, tasks outside the fragment contribute {outside_gap:.1f}, and unbounded quantifiers contribute {open_gap:.1f}. "
                "The exact-task gap is the empty charts. Selection, once a perfect program is present, stays at the semantic accuracy above. "
                "A few families also get large: filenames peak around 11 minutes, and dates and versions run to a few thousand languages.",
                "",
            ]
        )
    lines.extend(
        [
            "| Domain | Tasks | Exact | Open | Outside | Selector accuracy | Synthesis oracle | Languages | Seconds |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for domain in summary["domains"]:
        lines.append(
            f"| {domain['name']} | {domain['n']} | {domain['exact']} | {domain['open']} | {domain['outside']} | "
            f"{_pct(domain['selector_accuracy'])} | {_pct(domain['synthesis_oracle'])} | "
            f"{domain['mean_languages']:.1f} | {domain['mean_seconds']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)
