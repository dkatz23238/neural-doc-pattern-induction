"""v1 few-shot interface.

The decision is the code-length rule on the eight candidates the symbolic
selector presents. The frozen recognizer may attach the matching program, a
confidence, and alternatives. It does not replace the symbolic winner. The
residual override is not loaded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from pattern_induction.program import MAX_LEN, MAX_SUPPORT, Prog, Task, to_regex
from pattern_induction.select_train import load_selector, select_program
from pattern_induction.symbolic import symbolic_pick

_MODELS: tuple | None = None


class NeedleError(ValueError):
    """The examples are not a list of strings."""


@dataclass
class NeedleResult:
    regex: str
    ast: Prog | None
    confidence: float
    alternatives: list[str]
    method: str
    ambiguous: bool = False


def needle(examples: list[str], root: Path | None = None) -> NeedleResult:
    """Induce a regex from two to four positive examples.

    Each string is clipped to 32 characters. Every non-empty example is shown
    to the symbolic selector. The recognizer reads the first four of those.
    One example, or none, abstains. The returned regex is the symbolic winner.
    ``ambiguous`` is true when another presented candidate is within one bit
    of that cost, or when the recognizer proposes a different program.
    """
    supports = _supports(examples)
    if len(supports) < 2:
        return NeedleResult("", None, 0.0, [], "abstain", False)
    pick = symbolic_pick(supports)
    learned = _learned(supports[:MAX_SUPPORT], Path(root) if root is not None else Path("."))
    agrees = learned is None or learned[0] == pick.pattern
    alternatives = _alternatives(pick.alternatives, None if learned is None else learned[0], pick.pattern)
    ast = learned[1] if learned is not None and learned[0] == pick.pattern else None
    ambiguous = pick.ambiguous or not agrees
    confidence = _confidence(pick.margin, agrees, None if learned is None else learned[2])
    return NeedleResult(pick.pattern, ast, confidence, alternatives, "symbolic", ambiguous)


def _supports(examples: list[str]) -> list[str]:
    if isinstance(examples, (str, bytes)):
        raise NeedleError("examples must be a list of strings, not a single string")
    if not isinstance(examples, (list, tuple)):
        raise NeedleError("examples must be a list of strings")
    cleaned: list[str] = []
    for text in examples:
        if not isinstance(text, str):
            raise NeedleError("every example must be a string")
        if text:
            cleaned.append(text[:MAX_LEN])
    return cleaned


def _models(root: Path):
    global _MODELS
    resolved = root.resolve()
    if _MODELS is None or _MODELS[0] != resolved:
        _MODELS = (resolved, load_selector(resolved))
    return _MODELS[1]


def _learned(supports: list[str], root: Path) -> tuple[str, Prog | None, float] | None:
    """Recognizer hypothesis. Missing checkpoints leave the symbolic answer in place."""
    try:
        scope, witness, prior = _models(root)
    except (RuntimeError, OSError, FileNotFoundError):
        return None
    task = Task("needle", supports, Prog("COPY"), [])
    chosen, programs, _count = select_program(
        scope,
        witness,
        prior,
        task,
        with_stability=True,
        v2=True,
        stability_mode="direct",
        operator_witnesses=True,
    )
    if chosen is None:
        return None
    regex = to_regex(chosen, supports)
    return regex, chosen, _prior_confidence(programs, supports, prior, regex)


def _prior_confidence(programs: list[Prog], supports: list[str], prior, chosen: str) -> float:
    ranked = _ranked(programs, supports, prior)
    if len(ranked) <= 1:
        return 0.95
    scores = {regex: score for regex, score in ranked}
    top = scores.get(chosen, ranked[0][1])
    others = [score for regex, score in ranked if regex != chosen]
    gap = top - max(others)
    return 1.0 / (1.0 + math.exp(-gap))


def _ranked(programs: list[Prog], supports: list[str], prior) -> list[tuple[str, float]]:
    best: dict[str, float] = {}
    for prog in programs:
        rendered = to_regex(prog, supports)
        score = prior.score(prog)
        if rendered not in best or score > best[rendered]:
            best[rendered] = score
    return sorted(best.items(), key=lambda item: (-item[1], item[0]))


def _alternatives(symbolic: tuple[str, ...], learned: str | None, winner: str) -> list[str]:
    ordered: list[str] = []
    if learned and learned != winner:
        ordered.append(learned)
    for pattern in symbolic:
        if pattern != winner and pattern not in ordered:
            ordered.append(pattern)
        if len(ordered) == 4:
            break
    return ordered


def _confidence(margin: float, agrees: bool, prior_confidence: float | None) -> float:
    """Report certainty. This number does not choose the regex.

    ``margin`` is extra bits of code length charged to the runner-up.
    When the recognizer agrees, its own gap is averaged in. When it disagrees,
    the report stays low and the symbolic regex is unchanged.
    """
    finite = margin if math.isfinite(margin) else 8.0
    strength = 1.0 - math.exp(-max(finite, 0.0))
    if agrees and prior_confidence is not None:
        return 0.5 * (0.5 + 0.5 * strength) + 0.5 * min(max(prior_confidence, 0.0), 1.0)
    if agrees:
        return 0.5 + 0.5 * strength
    return 0.25 * strength
