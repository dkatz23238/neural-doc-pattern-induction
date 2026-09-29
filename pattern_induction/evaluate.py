"""Semantic evaluation of an inferred regex against a hidden set."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from pattern_induction.tasks import Task

CATASTROPHIC_BODIES = {
    ".*",
    ".+",
    "\\w*",
    "\\w+",
    "\\d*",
    "\\d+",
    "\\s*",
    "\\s+",
    "\\S*",
    "\\S+",
    "[\\s\\S]*",
    "[\\s\\S]+",
    "[\\d\\D]*",
    "[\\d\\D]+",
    "[A-Za-z0-9]*",
    "[A-Za-z0-9]+",
    "[A-Za-z0-9_]*",
    "[A-Za-z0-9_]+",
    "\\w{0,}",
    ".{0,}",
    ".{1,}",
}

PROBES = [
    "x",
    "hello",
    "@@@",
    "not a document field",
    "zzzzzzzzzzzzzzzz",
    "12",
    "AB",
    "...",
    "foo/bar",
    "???###",
]


@dataclass
class Score:
    valid: bool
    matches_visible: bool
    positive_recall: float
    negative_rejection: float
    semantic_accuracy: float
    catastrophic: bool
    complexity: dict
    error: str = ""
    accepted_positives: int = 0
    rejected_negatives: int = 0
    n_positives: int = 0
    n_negatives: int = 0


def _strip_anchors(pattern: str) -> str:
    body = pattern.strip()
    if body.startswith("^"):
        body = body[1:]
    if body.endswith("$"):
        body = body[:-1]
    while body.startswith("(?:") and body.endswith(")"):
        inner = body[3:-1]
        if inner.count("(") == inner.count(")"):
            body = inner
        else:
            break
    return body


def is_catastrophic(pattern: str, compiled: re.Pattern | None) -> bool:
    body = _strip_anchors(pattern).replace(" ", "")
    if body in CATASTROPHIC_BODIES:
        return True
    if re.fullmatch(r"(\.\*|\.\+|\.\{\d*,\d*\}|\\w[\*\+]|\\d[\*\+])", body):
        return True
    if compiled is None:
        return False
    hits = 0
    for probe in PROBES:
        try:
            if compiled.fullmatch(probe):
                hits += 1
        except re.error:
            return False
    return hits >= 6


def regex_complexity(pattern: str) -> dict:
    alternations = len(re.findall(r"(?<!\\)\|", pattern))
    capturing = len(re.findall(r"(?<!\\)\((?!\?)", pattern))
    quantifiers = re.findall(r"(?<!\\)(?:[+*?]|\{[0-9,]+\})", pattern)
    nested = 0
    depth = 0
    i = 0
    group_depth_at_quant = False
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "(":
            depth += 1
        elif ch == ")" and depth:
            depth -= 1
        elif ch in "+*?" or (ch == "{" and depth > 0):
            if depth > 0:
                group_depth_at_quant = True
        if ch in "+*?" and group_depth_at_quant and depth > 1:
            nested += 1
        i += 1
    nodes = (
        alternations
        + capturing
        + len(quantifiers)
        + len(re.findall(r"\\[dDwWsS]", pattern))
        + len(re.findall(r"\[[^\]]+\]", pattern))
    )
    return {
        "char_length": len(pattern),
        "ast_node_count": max(nodes, 1 if pattern else 0),
        "alternations": alternations,
        "capturing_groups": capturing,
        "quantifiers": len(quantifiers),
        "nested_quantifiers": nested,
    }


_COMPILED: dict[str, re.Pattern] = {}
_COMPILED_CAP = 8192


def compile_regex(pattern: str) -> re.Pattern:
    cached = _COMPILED.get(pattern)
    if cached is not None:
        return cached
    compiled = re.compile(pattern)
    if len(_COMPILED) < _COMPILED_CAP:
        _COMPILED[pattern] = compiled
    return compiled


def _rate(hits: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return hits / total


def score_regex(pattern: str, task: Task) -> Score:
    n_pos = len(task.hidden_positives)
    n_neg = len(task.hidden_negatives)
    complexity = regex_complexity(pattern)
    try:
        compiled = compile_regex(pattern)
    except re.error as exc:
        return Score(
            valid=False,
            matches_visible=False,
            positive_recall=0.0,
            negative_rejection=0.0,
            semantic_accuracy=0.0,
            catastrophic=False,
            complexity=complexity,
            error=str(exc),
            n_positives=n_pos,
            n_negatives=n_neg,
        )
    try:
        visible_ok = all(compiled.fullmatch(s) for s in task.visible)
        acc_pos = sum(1 for s in task.hidden_positives if compiled.fullmatch(s))
        rej_neg = sum(1 for s in task.hidden_negatives if not compiled.fullmatch(s))
    except re.error as exc:
        return Score(
            valid=False,
            matches_visible=False,
            positive_recall=0.0,
            negative_rejection=0.0,
            semantic_accuracy=0.0,
            catastrophic=False,
            complexity=complexity,
            error=str(exc),
            n_positives=n_pos,
            n_negatives=n_neg,
        )
    recall = _rate(acc_pos, n_pos)
    rejection = _rate(rej_neg, n_neg)
    return Score(
        valid=True,
        matches_visible=visible_ok,
        positive_recall=recall,
        negative_rejection=rejection,
        semantic_accuracy=(recall + rejection) / 2,
        catastrophic=is_catastrophic(pattern, compiled),
        complexity=complexity,
        accepted_positives=acc_pos,
        rejected_negatives=rej_neg,
        n_positives=n_pos,
        n_negatives=n_neg,
    )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


@dataclass
class Aggregate:
    system: str
    n: int
    semantic_accuracy: float
    positive_recall: float
    negative_rejection: float
    valid_output_rate: float
    visible_match_rate: float
    catastrophic_rate: float
    latency_ms: dict
    mean_char_length: float
    mean_alternations: float
    by_domain: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "system": self.system,
            "n": self.n,
            "semantic_accuracy": round(self.semantic_accuracy, 4),
            "positive_recall": round(self.positive_recall, 4),
            "negative_rejection": round(self.negative_rejection, 4),
            "valid_output_rate": round(self.valid_output_rate, 4),
            "visible_match_rate": round(self.visible_match_rate, 4),
            "catastrophic_rate": round(self.catastrophic_rate, 4),
            "latency_ms": {k: round(v, 3) for k, v in self.latency_ms.items()},
            "mean_char_length": round(self.mean_char_length, 2),
            "mean_alternations": round(self.mean_alternations, 3),
            "macro_semantic_accuracy": round(
                statistics.fmean(self.by_domain.values()) if self.by_domain else 0.0, 4
            ),
            "by_domain": self.by_domain,
        }


def aggregate(system: str, rows: list[dict]) -> Aggregate:
    n = len(rows)
    if n == 0:
        return Aggregate(system, 0, 0, 0, 0, 0, 0, 0, {}, 0, 0)

    def mean(key: str) -> float:
        return statistics.fmean(r[key] for r in rows)

    lat = [r["latency_ms"] for r in rows]
    domains: dict[str, list] = {}
    for row in rows:
        domains.setdefault(row["domain"], []).append(row["semantic_accuracy"])
    by_domain = {
        domain: round(statistics.fmean(vals), 4) for domain, vals in sorted(domains.items())
    }
    return Aggregate(
        system=system,
        n=n,
        semantic_accuracy=mean("semantic_accuracy"),
        positive_recall=mean("positive_recall"),
        negative_rejection=mean("negative_rejection"),
        valid_output_rate=mean("valid"),
        visible_match_rate=mean("matches_visible"),
        catastrophic_rate=mean("catastrophic"),
        latency_ms={
            "p50": percentile(lat, 0.50),
            "p95": percentile(lat, 0.95),
            "p99": percentile(lat, 0.99),
            "mean": statistics.fmean(lat),
        },
        mean_char_length=mean("char_length"),
        mean_alternations=mean("alternations"),
        by_domain=by_domain,
    )
