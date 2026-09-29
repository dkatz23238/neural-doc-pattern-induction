"""Candidate recall under the frozen benchmark contract and the product contract.

The legacy contract still shows the inducer four examples clipped at 32
characters. The product contract shows those same four strings whole. Closed
sets come from the frozen lexicon bank. No prior is trained, and BENCH-E is
not read.
"""

from __future__ import annotations

import json
import re
import signal
import time
from pathlib import Path

from re._constants import BRANCH, LITERAL, MAX_REPEAT, SUBPATTERN
from re._parser import parse as parse_regex

from pattern_induction.latent import expand_programs
from pattern_induction.latent_eval import _identifiable, _seeds
from pattern_induction.lexicon import closes_alternation
from pattern_induction.program import MAX_LEN, MAX_SUPPORT, Prog, _MEMBERSHIP, execute
from pattern_induction.select_train import complete_assembly, load_selector
from pattern_induction.selector_modes import _hyps
from pattern_induction.tasks import read_jsonl


class _Timeout(Exception):
    pass


def _alarm(signum, frame) -> None:
    raise _Timeout()


def _literal_alternations(pattern: str) -> list[set[str]]:
    try:
        tokens = parse_regex(pattern)
    except re.error:
        return []
    found: list[set[str]] = []

    def walk(nodes) -> None:
        for op, arg in nodes:
            if op is BRANCH:
                _flags, arms = arg
                literals: list[str] = []
                literal_arms = True
                for arm in arms:
                    if arm and all(token[0] is LITERAL for token in arm):
                        literals.append("".join(chr(token[1]) for token in arm))
                    else:
                        literal_arms = False
                if literal_arms and len(literals) >= 3:
                    found.append(set(literals))
                for arm in arms:
                    walk(arm)
            elif op is SUBPATTERN:
                walk(arg[-1])
            elif op is MAX_REPEAT:
                walk(arg[2])

    walk(tokens)
    return found


def is_closed_set(pattern: str) -> bool:
    return any(closes_alternation(literals) for literals in _literal_alternations(pattern))


def _perfect(programs: list[Prog], supports: list[str], positives: list[str], negatives: list[str]) -> bool:
    if not programs:
        return False
    for prog in programs:
        if all(execute(prog, text, supports) for text in positives) and all(
            not execute(prog, text, supports) for text in negatives
        ):
            return True
    return False


def _expand(seeds: list[Prog], supports: list[str], limit_s: float) -> tuple[list[Prog], bool]:
    signal.alarm(int(limit_s))
    try:
        return expand_programs(seeds, supports, cap=400), False
    except _Timeout:
        return list(seeds), True
    finally:
        signal.alarm(0)


def _legacy_pool(root: Path, scope_box: list, bench, supports: list[str], assemble_s: float, expand_s: float):
    """Chart seeds plus latent rewrites. ``supports`` are already clipped."""
    identifiable = _identifiable(supports)
    if scope_box[0] is None:
        scope_box[0], _witness, _prior = load_selector(root)
    signal.alarm(int(assemble_s))
    programs: list[Prog] = []
    timed = False
    try:
        assembly = complete_assembly(supports, _hyps(scope_box[0], bench, supports), v2=True)
        programs = [prog for prog, _score in assembly.programs]
    except _Timeout:
        timed = True
    finally:
        signal.alarm(0)
    seeds = _seeds(programs, identifiable)
    expanded, expand_timed = _expand(seeds, supports, expand_s)
    return programs + expanded, timed or expand_timed


def evaluate(root: Path, limit_s: float = 25.0) -> dict:
    started = time.perf_counter()
    previous = json.loads((root / "results" / "needle" / "latent_oracle.json").read_text())
    by_id = {row["id"]: row for row in previous["rows"]}
    tasks = read_jsonl(root / "data" / "bench_c.jsonl")
    signal.signal(signal.SIGALRM, _alarm)
    scope_box: list = [None]
    rows = []
    timeouts = 0
    assemble_s = 45.0
    for index, bench in enumerate(tasks):
        _MEMBERSHIP.clear()
        prior = by_id[bench.id]
        closed = is_closed_set(bench.gold_regex)
        legacy_supports = [text[:MAX_LEN] for text in bench.visible[:MAX_SUPPORT]]
        full_supports = list(bench.visible[:MAX_SUPPORT])
        # The chart bit is unchanged, so a chart success stays a success.
        # Everything the chart missed is assembled again and rewritten, so an
        # older expansion win is not carried forward.
        legacy = bool(prior["current_perfect"])
        product = legacy
        legacy_added = 0
        product_added = 0
        if not legacy:
            programs, timed = _legacy_pool(root, scope_box, bench, legacy_supports, assemble_s, limit_s)
            timeouts += int(timed)
            legacy_added = len(programs)
            legacy = _perfect(programs, legacy_supports, bench.hidden_positives, bench.hidden_negatives)
            product = legacy
            # Full strings stay on the symbolic factorizer. The recognizer is
            # not run on a support longer than the frozen 32-character window.
            if not product and full_supports != legacy_supports:
                programs, timed = _expand([], full_supports, limit_s)
                timeouts += int(timed)
                product_added = len(programs)
                product = _perfect(programs, full_supports, bench.hidden_positives, bench.hidden_negatives)
            elif not product:
                product_added = legacy_added
        row = {
            "id": bench.id,
            "domain": bench.domain,
            "variant": bench.variant,
            "closed_set": closed,
            "current_perfect": bool(prior["current_perfect"]),
            "prior_expanded_perfect": bool(prior["expanded_perfect"]),
            "legacy_perfect": legacy,
            "product_perfect": product,
            "legacy_added": legacy_added,
            "product_added": product_added,
            "clipped": any(len(text) > MAX_LEN for text in bench.visible[:MAX_SUPPORT]),
        }
        rows.append(row)
        if index % 25 == 0 or index + 1 == len(tasks):
            legacy_n = sum(1 for item in rows if item["legacy_perfect"])
            product_n = sum(1 for item in rows if item["product_perfect"])
            print(
                f"recall {index + 1}/{len(tasks)} {bench.domain} legacy {legacy_n}/{len(rows)} product {product_n}/{len(rows)}",
                flush=True,
            )
            (root / "results" / "needle" / "recall_oracle_partial.json").write_text(json.dumps(rows) + "\n")
    payload = _summarize(rows, previous, time.perf_counter() - started, timeouts)
    partial = root / "results" / "needle" / "recall_oracle_partial.json"
    if partial.exists():
        partial.unlink()
    return payload


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row[key]) / len(rows)


def _summarize(rows: list[dict], previous: dict, seconds: float, timeouts: int) -> dict:
    n = len(rows)
    closed = [row for row in rows if row["closed_set"]]
    structural = [row for row in rows if not row["closed_set"]]
    legacy_perfect = sum(1 for row in rows if row["legacy_perfect"])
    product_perfect = sum(1 for row in rows if row["product_perfect"])
    missed = [row for row in rows if not row["product_perfect"]]
    return {
        "n": n,
        "current_synthesis_oracle": previous["current_synthesis_oracle"],
        "prior_expanded_oracle": previous["expanded_synthesis_oracle"],
        "legacy_expanded_oracle": legacy_perfect / max(n, 1),
        "product_oracle": product_perfect / max(n, 1),
        "structural_oracle": _rate(structural, "product_perfect"),
        "closed_set_recall": _rate(closed, "product_perfect"),
        "legacy_perfect": legacy_perfect,
        "product_perfect": product_perfect,
        "structural_n": len(structural),
        "structural_perfect": sum(1 for row in structural if row["product_perfect"]),
        "closed_n": len(closed),
        "closed_perfect": sum(1 for row in closed if row["product_perfect"]),
        "legacy_gained": sum(1 for row in rows if row["legacy_perfect"] and not row["prior_expanded_perfect"]),
        "legacy_lost": sum(1 for row in rows if row["prior_expanded_perfect"] and not row["legacy_perfect"]),
        "product_gained_by_full_string": sum(
            1 for row in rows if row["product_perfect"] and not row["legacy_perfect"]
        ),
        "clipped_tasks": sum(1 for row in rows if row["clipped"]),
        "remaining_product_misses": len(missed),
        "remaining_by_domain": _domains(missed),
        "assemble_timeouts": timeouts,
        "seconds": seconds,
        "trained": False,
        "bench_e_scored": False,
        "max_len_legacy": MAX_LEN,
        "rows": rows,
    }


def _domains(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["domain"]] = counts.get(row["domain"], 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def render_markdown(payload: dict) -> str:
    lines = [
        "## Candidate recall",
        "",
        "The legacy contract is unchanged: four examples, each clipped at 32 characters, "
        "the same chart as the latent-oracle pass. The product contract uses those four "
        "strings without that clip. Closed sets are proposed from a frozen bank "
        "(ISO 4217, Gregorian month names, RFC 1945 and the IANA status registry, "
        "common filename suffixes, mass units, letter sizes). "
        "An integer interval and a literal alternation now count as one node, "
        "so a 20-node IPv4 expression and a month-name alternation fit the 16-node cap. "
        "No prior is trained. BENCH-E stays sealed.",
        "",
        f"Legacy expanded oracle is {payload['legacy_expanded_oracle']:.3f} "
        f"({payload['legacy_perfect']}/{payload['n']}), "
        f"up from {payload['prior_expanded_oracle']:.3f} "
        f"({payload['legacy_gained']} gained, {payload['legacy_lost']} lost). "
        f"Product-contract oracle is {payload['product_oracle']:.3f} "
        f"({payload['product_perfect']}/{payload['n']}). "
        f"Full strings add {payload['product_gained_by_full_string']} tasks the clipped contract still misses. "
        f"Expansion timeouts: {payload['assemble_timeouts']}.",
        "",
        f"Non-lexicon structural tasks: {payload['structural_oracle']:.3f} "
        f"({payload['structural_perfect']}/{payload['structural_n']}). "
        f"Closed-set candidate recall: {payload['closed_set_recall']:.3f} "
        f"({payload['closed_perfect']}/{payload['closed_n']}).",
        "",
        "| Metric | Value | Gate |",
        "| --- | ---: | ---: |",
        f"| Legacy BENCH-C expanded oracle | {payload['legacy_expanded_oracle']:.3f} | 0.95 |",
        f"| Full-input product-contract oracle | {payload['product_oracle']:.3f} | 0.98 |",
        f"| Non-lexicon structural oracle | {payload['structural_oracle']:.3f} | 0.98 |",
        f"| Closed-set candidate recall | {payload['closed_set_recall']:.3f} | 0.95 |",
        f"| Expansion timeouts | {payload['assemble_timeouts']} | 0 |",
        "",
        "Tasks the product contract still misses, by domain:",
        "",
    ]
    if payload["remaining_by_domain"]:
        for domain, count in payload["remaining_by_domain"].items():
            lines.append(f"- {domain}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    product = payload["product_oracle"]
    structural = payload["structural_oracle"]
    if product >= 0.98 and structural >= 0.98 and payload["closed_set_recall"] >= 0.95:
        lines.append(
            "The product-contract candidate oracle is above 0.98, so the latent language is usually in the set. "
            "That earns a learned prior. This pass does not train one. BENCH-E stays sealed.\n"
        )
    else:
        lines.append(
            "The product-contract oracle is not yet above 0.98, so this is not a basis for training a language prior. "
            "What remains is structure the four examples never show: an optional branch, a width the clip still "
            "does not explain, or a closed set whose published superset also accepts a hidden negative. "
            "BENCH-E stays sealed.\n"
        )
    return "\n".join(lines) + "\n"


def _main() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = evaluate(root)
    out = root / "results" / "needle"
    (out / "recall_oracle.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "recall_oracle.md").write_text(render_markdown(payload), encoding="utf-8")
    print(
        "legacy",
        round(payload["legacy_expanded_oracle"], 3),
        "product",
        round(payload["product_oracle"], 3),
        "structural",
        round(payload["structural_oracle"], 3),
        "closed",
        round(payload["closed_set_recall"], 3),
        "timeouts",
        payload["assemble_timeouts"],
    )


if __name__ == "__main__":
    _main()
