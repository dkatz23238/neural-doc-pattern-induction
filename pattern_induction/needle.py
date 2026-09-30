"""v1 few-shot interface.

The decision is the code-length rule on the eight candidates the symbolic
selector presents. The frozen recognizer may attach the matching program, a
confidence, and alternatives. It does not replace the symbolic winner. The
residual override is not loaded.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

from pattern_induction.candidates import semantic_key
from pattern_induction.confidence import Signals, fitted_confidence, load_fit
from pattern_induction.program import (
    MAX_LEN,
    MAX_SUPPORT,
    Prog,
    SearchTimeout,
    Task,
    _MEMBERSHIP,
    _PIECE,
    set_deadline,
    to_regex,
)
from pattern_induction.select_train import load_selector, select_program
from pattern_induction.symbolic import symbolic_pick

TIME_LIMIT = 5.0
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


def needle(examples: list[str], root: Path | None = None, time_limit: float | None = TIME_LIMIT) -> NeedleResult:
    """Induce a regex from two or more positive examples.

    Every non-empty example is shown whole to the symbolic selector. The
    recognizer reads the first four of those, each clipped to 32 characters,
    and stops after ``time_limit`` seconds of search; the symbolic winner is
    returned either way. One example, or none, abstains. ``ambiguous`` is true
    when another presented candidate is within one bit of that cost, or when
    the recognizer proposes a different program. Alternatives differ in the
    language they describe, not in spelling.
    """
    supports = _supports(examples)
    if len(supports) < 2:
        return NeedleResult("", None, 0.0, [], "abstain", False)
    base = Path(root) if root is not None else Path(".")
    pick = symbolic_pick(supports)
    clipped = [text[:MAX_LEN] for text in supports[:MAX_SUPPORT]]
    learned = _learned(clipped, base, time_limit)
    learned_regex = learned.regex if learned is not None else None
    same_language = learned_regex is not None and semantic_key(learned_regex) == semantic_key(pick.pattern)
    agrees = learned_regex is None or same_language
    alternatives = _alternatives(pick.alternatives, learned_regex, pick.pattern)
    ast = learned.program if same_language else None
    ambiguous = pick.ambiguous or not agrees
    fit = load_fit(base)
    if fit is None:
        prior_conf = learned.prior_conf if learned_regex is not None else None
        confidence = _confidence(pick.margin, agrees, prior_conf, pick.ambiguous)
    else:
        signals = Signals(
            regex=pick.pattern,
            supports=supports,
            margin=pick.margin,
            near_tie=pick.ambiguous,
            n_alternatives=len(pick.alternatives),
            cost=pick.cost,
            learned=learned_regex is not None,
            agrees=same_language,
            prior_conf=learned.prior_conf if learned is not None else None,
            n_programs=learned.n_programs if learned is not None else 0,
        )
        confidence = fitted_confidence(signals, fit)
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
            cleaned.append(text)
    return cleaned


def _models(root: Path):
    global _MODELS
    resolved = root.resolve()
    if _MODELS is None or _MODELS[0] != resolved:
        _MODELS = (resolved, load_selector(resolved))
    return _MODELS[1]


@dataclass(frozen=True)
class _Learned:
    regex: str | None
    program: Prog | None
    prior_conf: float | None
    n_programs: int


def _learned(supports: list[str], root: Path, time_limit: float | None = TIME_LIMIT) -> _Learned | None:
    """Recognizer hypothesis. Missing checkpoints, or a search past the time limit, return None."""
    _MEMBERSHIP.clear()
    _PIECE.clear()
    try:
        scope, witness, prior = _models(root)
    except (RuntimeError, OSError, FileNotFoundError):
        return None
    task = Task("needle", supports, Prog("COPY"), [])
    set_deadline(None if time_limit is None else time.perf_counter() + time_limit)
    try:
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
            return _Learned(None, None, None, len(programs or []))
        regex = to_regex(chosen, supports)
        return _Learned(regex, chosen, _prior_confidence(programs, supports, prior, regex), len(programs))
    except SearchTimeout:
        return None
    finally:
        set_deadline(None)


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
