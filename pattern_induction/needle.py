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

from pattern_induction.candidates import semantic_key
from pattern_induction.program import MAX_LEN, MAX_SUPPORT, Prog, Task, _MEMBERSHIP, _PIECE, to_regex
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
    """Induce a regex from two or more positive examples.

    Each string is clipped to 32 characters. Every non-empty example is shown
    to the symbolic selector. The recognizer reads the first four of those.
    One example, or none, abstains. The returned regex is the symbolic winner.
    ``ambiguous`` is true when another presented candidate is within one bit
    of that cost, or when the recognizer proposes a different program.
    Alternatives differ in the language they describe, not in spelling.
    """
    supports = _supports(examples)
    if len(supports) < 2:
        return NeedleResult("", None, 0.0, [], "abstain", False)
    pick = symbolic_pick(supports)
    learned = _learned(supports[:MAX_SUPPORT], Path(root) if root is not None else Path("."))
    learned_regex = None if learned is None else learned[0]
    same_language = learned_regex is not None and semantic_key(learned_regex) == semantic_key(pick.pattern)
    agrees = learned is None or same_language
    alternatives = _alternatives(pick.alternatives, learned_regex, pick.pattern)
    ast = learned[1] if same_language else None
    ambiguous = pick.ambiguous or not agrees
    confidence = _confidence(pick.margin, agrees, None if learned is None else learned[2], pick.ambiguous)
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
    _MEMBERSHIP.clear()
    _PIECE.clear()
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
    """Up to four languages, not spellings of the winner."""
    ordered: list[str] = []
    seen = {semantic_key(winner)}
    incoming: list[str] = []
    if learned and learned != winner:
        incoming.append(learned)
    incoming.extend(symbolic)
    for pattern in incoming:
        if pattern == winner:
            continue
        key = semantic_key(pattern)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(pattern)
        if len(ordered) == 4:
            break
    return ordered


def _confidence(
    margin: float,
    agrees: bool,
    prior_confidence: float | None,
    ambiguous: bool = False,
) -> float:
    """Report certainty. This number does not choose the regex.

    Three factors move it. ``margin`` is extra bits of code length charged to
    the cheapest strictly costlier candidate. ``agrees`` is whether the
    recognizer's regex is the symbolic winner; its grammar gap counts only
    then. ``ambiguous`` is the one-bit band on that cost. A wide gap with
    agreement can sit high. Disagreement, or a gap under one bit, stays
    below certainty. The symbolic regex is unchanged either way.
    """
    finite = margin if math.isfinite(margin) else 8.0
    finite = min(max(finite, 0.0), 8.0)
    margin_factor = 1.0 - math.exp(-finite)
    if agrees and prior_confidence is not None:
        evidence = min(max(prior_confidence, 0.0), 1.0)
    elif agrees:
        evidence = 1.0
    else:
        evidence = 0.5
    clarity = 0.6 if ambiguous else 1.0
    score = margin_factor * (0.5 + 0.5 * evidence) * clarity
    if not agrees:
        score = min(score, 0.8)
    if ambiguous:
        score = min(score, 0.7)
    return score
