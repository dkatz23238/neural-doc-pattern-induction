"""End-to-end few-shot interface over the frozen recognizer.

Positive examples go through the neural chart and the deterministic facts,
then version-space synthesis, equivalence collapse, and witness selection.
The result is a semantic program and the regex it compiles to. When several
languages survive, the other regexes are returned as alternatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from pattern_induction.program import MAX_LEN, MAX_SUPPORT, Prog, Task, to_regex
from pattern_induction.select_train import load_selector, select_program

_MODELS: tuple | None = None


@dataclass
class NeedleResult:
    regex: str
    ast: Prog | None
    confidence: float
    alternatives: list[str]
    method: str


def needle(examples: list[str], root: Path | None = None) -> NeedleResult:
    """Induce a regex from two to four positive examples."""
    supports = [text[:MAX_LEN] for text in examples if text][:MAX_SUPPORT]
    if len(supports) < 2:
        return NeedleResult("", None, 0.0, [], "abstain")
    base = Path(root) if root is not None else Path(".")
    scope, witness, prior = _models(base)
    task = Task("needle", supports, Prog("COPY"), [])
    chosen, programs, _count = select_program(scope, witness, prior, task, with_stability=True, v2=True)
    if chosen is None:
        return NeedleResult("", None, 0.0, [], "abstain")
    regex = to_regex(chosen, supports)
    ranked = _ranked(programs, supports, prior)
    alternatives = [item for item, _score in ranked if item != regex][:4]
    return NeedleResult(regex, chosen, _confidence(ranked, regex), alternatives, _method(programs))


def _models(root: Path):
    global _MODELS
    resolved = root.resolve()
    if _MODELS is None or _MODELS[0] != resolved:
        _MODELS = (resolved, load_selector(resolved))
    return _MODELS[1]


def _uses_v2(prog: Prog) -> bool:
    if prog.op in {"SET", "INT"}:
        return True
    return any(_uses_v2(child) for child in prog.children)


def _method(programs: list[Prog]) -> str:
    if any(_uses_v2(prog) for prog in programs):
        return "dsl-v2"
    return "synthesis-v1"


def _ranked(programs: list[Prog], supports: list[str], prior) -> list[tuple[str, float]]:
    best: dict[str, float] = {}
    for prog in programs:
        rendered = to_regex(prog, supports)
        score = prior.score(prog)
        if rendered not in best or score > best[rendered]:
            best[rendered] = score
    return sorted(best.items(), key=lambda item: item[1], reverse=True)


def _confidence(ranked: list[tuple[str, float]], chosen: str) -> float:
    if len(ranked) <= 1:
        return 0.95
    scores = {regex: score for regex, score in ranked}
    top = scores.get(chosen, ranked[0][1])
    others = [score for regex, score in ranked if regex != chosen]
    gap = top - max(others)
    return 1.0 / (1.0 + math.exp(-gap))
