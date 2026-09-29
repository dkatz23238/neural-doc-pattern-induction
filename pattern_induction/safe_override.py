"""Conservative override of the symbolic selector.

The code-length rule stays the choice. A small model may replace it with one
existing alternative only when the probability that the alternative has higher
hidden accuracy clears a threshold. Agreements keep the symbolic winner.
Wrong overrides are weighted more heavily than missed corrections.

The threshold is chosen on held-out formatting variants. BENCH-C is scored once.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pattern_induction.override import PAIR_FEATURES, pair_rows

IMPROVE_EPS = 0.01
# A predicted override that is not an improvement is weighted by this factor.
# Fixed before BENCH-C. Not a search.
LAMBDA = 10.0
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
# Fraction of tasks allowed to fall more than IMPROVE_EPS below the symbolic score.
REGRESSION_BUDGET = 0.02
# The easy family is the heuristic-success slice. A drop larger than this is meaningful.
EASY_FAMILY = "success"
EASY_MARGIN = 0.005
# Practical band above the 96.2% symbolic baseline. The minimum bar is any gain.
TARGET_ACCURACY = 0.967

SUPPORT_FEATURES = (
    "n_supports",
    "mean_len",
    "max_len",
    "charset_size",
    "digit_fraction",
)
EXTRA_FEATURES = SUPPORT_FEATURES + (
    "alt_evidence",
    "evidence_present",
    "confidence",
)
FEATURE_NAMES = PAIR_FEATURES + EXTRA_FEATURES


def support_features(strings: list[str]) -> list[float]:
    """Visible summary of the supports. Character identities are not one-hot encoded."""
    texts = [text for text in strings if text][:4]
    if not texts:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    lengths = [len(text) for text in texts]
    blob = "".join(texts)
    total = len(blob) or 1
    digits = sum(character.isdigit() for character in blob)
    return [
        float(len(texts)),
        sum(lengths) / len(texts),
        float(max(lengths)),
        float(len(set(blob))),
        digits / total,
    ]


def assemble_features(
    alternative: str,
    symbolic: str,
    strings: list[str],
    alt_evidence: float,
    evidence_present: bool,
    confidence: float,
) -> list[float]:
    """Symbolic winner, the alternative, support summaries, evidence, and their gap."""
    pair = pair_rows([alternative], symbolic, symbolic, strings)[0]
    tail = support_features(strings)
    tail.extend(
        (
            float(alt_evidence),
            1.0 if evidence_present else 0.0,
            float(confidence),
        )
    )
    if len(pair) + len(tail) != len(FEATURE_NAMES):
        raise RuntimeError("feature width drifted")
    return pair + tail


class SafeOverride(nn.Module):
    """Logit for P(the alternative has higher hidden accuracy)."""

    def __init__(self, n_features: int = len(FEATURE_NAMES), hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(int(param.numel()) for param in model.parameters())


def safe_loss(logit: torch.Tensor, label: torch.Tensor, lam: float = LAMBDA) -> torch.Tensor:
    """BCE. Rows whose alternative is not better contribute ``lam`` times as much."""
    weight = torch.where(label > 0.5, torch.ones_like(logit), torch.full_like(logit, lam))
    return F.binary_cross_entropy_with_logits(logit, label, weight=weight)


def override_choice(probability: float, threshold: float, disagreed: bool) -> bool:
    """Default is the symbolic winner. A tie in the regex is not an override."""
    if not disagreed:
        return False
    return probability > threshold


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict], take_alt: list[bool]) -> dict:
    """Errors fixed and regressions are measured against the symbolic winner."""
    chosen: list[float] = []
    fixed = 0
    regressed = 0
    overrides = 0
    families: dict[str, dict[str, list[float]]] = {}
    for row, take in zip(rows, take_alt):
        accuracy = row["alt_accuracy"] if take else row["symbolic_accuracy"]
        chosen.append(accuracy)
        if take:
            overrides += 1
        if accuracy > row["symbolic_accuracy"] + IMPROVE_EPS:
            fixed += 1
        elif accuracy < row["symbolic_accuracy"] - IMPROVE_EPS:
            regressed += 1
        bucket = families.setdefault(
            row.get("failure") or EASY_FAMILY,
            {"symbolic": [], "chosen": [], "oracle": []},
        )
        bucket["symbolic"].append(row["symbolic_accuracy"])
        bucket["chosen"].append(accuracy)
        bucket["oracle"].append(max(row["symbolic_accuracy"], row["alt_accuracy"]))
    total = len(rows) or 1
    easy = families.get(EASY_FAMILY)
    family_rows = []
    for name, bucket in sorted(families.items(), key=lambda item: (-len(item[1]["chosen"]), item[0])):
        family_rows.append(
            {
                "failure": name,
                "n": len(bucket["chosen"]),
                "symbolic": _mean(bucket["symbolic"]),
                "chosen": _mean(bucket["chosen"]),
                "oracle": _mean(bucket["oracle"]),
            }
        )
    return {
        "n": len(rows),
        "semantic_accuracy": _mean(chosen),
        "symbolic_accuracy": _mean([row["symbolic_accuracy"] for row in rows]),
        "alternative_accuracy": _mean([row["alt_accuracy"] for row in rows]),
        "oracle_accuracy": _mean([max(row["symbolic_accuracy"], row["alt_accuracy"]) for row in rows]),
        "fixed": fixed,
        "regressed": regressed,
        "net": fixed - regressed,
        "fixed_rate": fixed / total,
        "regression_rate": regressed / total,
        "overrides": overrides,
        "override_rate": overrides / total,
        "false_overrides": regressed,
        "easy_n": len(easy["chosen"]) if easy else 0,
        "easy_accuracy": _mean(easy["chosen"]) if easy else None,
        "easy_symbolic": _mean(easy["symbolic"]) if easy else None,
        "families": family_rows,
    }


def _easy_ok(summary: dict) -> bool:
    if not summary["easy_n"] or summary["easy_accuracy"] is None or summary["easy_symbolic"] is None:
        return True
    return summary["easy_accuracy"] + 1e-12 >= summary["easy_symbolic"] - EASY_MARGIN


def eligible(summary: dict) -> bool:
    """A threshold may ship only when it does not give back the symbolic baseline."""
    if summary["semantic_accuracy"] + 1e-12 < summary["symbolic_accuracy"]:
        return False
    if summary["regression_rate"] > REGRESSION_BUDGET + 1e-12:
        return False
    return _easy_ok(summary)


def choose_threshold(rows: list[dict], probabilities: list[float]) -> tuple[float, dict]:
    """Highest validation accuracy inside the regression budget. Ties stay conservative."""
    recorded = []
    eligible_rows = []
    for threshold in THRESHOLDS:
        take = [
            override_choice(probability, threshold, bool(row["disagreed"]))
            for probability, row in zip(probabilities, rows)
        ]
        summary = summarize(rows, take)
        summary["threshold"] = threshold
        summary["eligible"] = eligible(summary)
        recorded.append(summary)
        if summary["eligible"]:
            eligible_rows.append(summary)
    pool = eligible_rows or [row for row in recorded if row["threshold"] == 1.0]
    chosen = max(
        pool,
        key=lambda row: (row["semantic_accuracy"], -row["regression_rate"], row["threshold"]),
    )
    return chosen["threshold"], {"grid": recorded, "chosen": chosen, "eligible": bool(eligible_rows)}


def clears_minimum(summary: dict) -> bool:
    return (
        summary["semantic_accuracy"] > summary["symbolic_accuracy"] + 1e-4
        and summary["regression_rate"] <= REGRESSION_BUDGET + 1e-12
        and _easy_ok(summary)
    )


def clears_target(summary: dict) -> bool:
    return summary["semantic_accuracy"] + 1e-12 >= TARGET_ACCURACY and clears_minimum(summary)


def fit_scale(rows: list[dict]) -> tuple[list[float], list[float]]:
    vectors = [row["features"] for row in rows if row["disagreed"]]
    width = len(FEATURE_NAMES)
    if not vectors:
        return [0.0] * width, [1.0] * width
    count = float(len(vectors))
    means = []
    scales = []
    for col in range(width):
        column = [vector[col] for vector in vectors]
        center = sum(column) / count
        variance = sum((value - center) ** 2 for value in column) / count
        scale = variance ** 0.5
        means.append(center)
        scales.append(scale if scale >= 1e-8 else 1.0)
    return means, scales


def standardize(vector: list[float], mean: list[float], scale: list[float]) -> list[float]:
    return [(value - center) / width for value, center, width in zip(vector, mean, scale)]
