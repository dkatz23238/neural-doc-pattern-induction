"""Fitted confidence for the v1 decision.

A logistic model over signals needle() already computes. The target is that the
code-length winner scores at least 0.99 on BENCH-C's hidden strings. The
weights come from `scripts/fit_confidence.py` and live in
`results/needle/confidence_fit.json`. The score does not change the regex.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from pattern_induction.program import MAX_LEN

FIT_PATH = Path("results") / "needle" / "confidence_fit.json"

NAMES = (
    "gap in bits",
    "no stricter runner-up",
    "runner-up within 1 bit",
    "network answered",
    "network agrees",
    "network's own margin",
    "log #network candidates",
    "#examples",
    "log max length",
    "example over 32 chars",
    "#rule alternatives",
    "bits per character",
    "log regex length",
    "regex has |",
    "regex has open repeat",
    "regex #ranges",
    "regex #classes",
)


@dataclass(frozen=True)
class Signals:
    regex: str
    supports: list[str]
    margin: float
    near_tie: bool
    n_alternatives: int
    cost: float
    learned: bool
    agrees: bool
    prior_conf: float | None
    n_programs: int


def features(signals: Signals) -> list[float]:
    supports = signals.supports
    mean_len = sum(len(text) for text in supports) / len(supports)
    margin = 8.0 if not math.isfinite(signals.margin) else min(signals.margin, 8.0)
    prior = signals.prior_conf if signals.prior_conf is not None else 0.5
    regex = signals.regex
    return [
        margin,
        float(not math.isfinite(signals.margin)),
        float(signals.near_tie),
        float(signals.learned),
        float(signals.agrees),
        prior if signals.agrees else 0.5,
        math.log1p(signals.n_programs),
        float(len(supports)),
        math.log(max(len(text) for text in supports)),
        float(any(len(text) > MAX_LEN for text in supports)),
        float(signals.n_alternatives),
        signals.cost / max(1.0, mean_len * len(supports)),
        math.log(len(regex)),
        float("|" in regex),
        float(any(token in regex for token in ("+", "*", ",}"))),
        float(regex.count(",")),
        float(regex.count("[")),
    ]


_FITS: dict[Path, dict | None] = {}


def load_fit(root: Path) -> dict | None:
    path = (root / FIT_PATH).resolve()
    if path not in _FITS:
        _FITS[path] = json.loads(path.read_text()) if path.exists() else None
    return _FITS[path]


def fitted_confidence(signals: Signals, fit: dict) -> float:
    if tuple(fit["names"]) != NAMES:
        raise ValueError("confidence fit was made for different features")
    values = features(signals)
    logit = fit["w"][0] + sum(
        weight * (value - mu) / sd for weight, value, mu, sd in zip(fit["w"][1:], values, fit["mu"], fit["sd"])
    )
    return 1.0 / (1.0 + math.exp(-logit))
