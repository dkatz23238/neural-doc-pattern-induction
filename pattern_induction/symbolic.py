"""v1 decision rule: code length with the digit-literal charge.

The pick is the minimum two-part cost on the eight candidates the symbolic
selector presents. A tie keeps the heuristic pattern. Learned scores do not
enter this function. The constant below is one bit of that cost, not a
threshold fit to a benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

from pattern_induction.candidates import pattern_cost, presented_candidates

PRESENTED = 8
POOL_LIMIT = 80
AMBIGUITY_BITS = 1.0


@dataclass(frozen=True)
class SymbolicPick:
    pattern: str
    cost: float
    margin: float
    alternatives: tuple[str, ...]
    ambiguous: bool


def symbolic_pick(strings: list[str]) -> SymbolicPick:
    """Choose the code-length winner for these visible strings."""
    if len(strings) < 2:
        raise ValueError("the symbolic selector needs at least two strings")
    shown, _pool = presented_candidates(strings, k=PRESENTED, pool_limit=POOL_LIMIT)
    patterns = [item.pattern for item in shown] or ["^$"]
    heuristic = patterns[0]
    costs = [pattern_cost(pattern, strings, True) for pattern in patterns]
    best = 0
    for index, cost in enumerate(costs[1:], start=1):
        if cost < costs[best] - 1e-9:
            best = index
        elif abs(cost - costs[best]) <= 1e-9 and patterns[index] == heuristic:
            best = index
    winner = patterns[best]
    winner_cost = costs[best]
    ranked = sorted(
        ((costs[index], patterns[index]) for index in range(len(patterns)) if patterns[index] != winner),
        key=lambda item: (item[0], item[1]),
    )
    seen: list[str] = []
    for cost, pattern in ranked:
        # An equal-cost spelling is the same explanation. The tie-break already chose.
        if cost <= winner_cost + 1e-9:
            continue
        if pattern not in seen:
            seen.append(pattern)
    stricter = [cost for cost, _pattern in ranked if cost > winner_cost + 1e-9]
    margin = (min(stricter) - winner_cost) if stricter else float("inf")
    return SymbolicPick(
        pattern=winner,
        cost=winner_cost,
        margin=margin,
        alternatives=tuple(seen),
        ambiguous=margin < AMBIGUITY_BITS,
    )
