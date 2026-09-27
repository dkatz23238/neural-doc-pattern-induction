"""Frozen-pool selector: listwise linear scores and a confidence gate.

The structural generator is not changed here. A candidate is described with
visible features only. Hidden semantic accuracy is a training label, never a
feature. The code-length rule remains the fallback when the learned score has
no margin.
"""

from __future__ import annotations

import math

from pattern_induction.candidates import (
    class_size,
    description_nodes,
    observed_data_bits,
    parse_pattern,
    pattern_cost,
)

K_FROZEN = 16
# Label temperature and L2 are chosen on a validation fold, never on BENCH-C.
TAUS = (0.05, 0.1, 0.25)
LAMBDAS = (0.01, 0.1, 1.0, 10.0)
# Probability gap between the top two learned scores. 0 always trusts the model.
MARGINS = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7)
# A gated model is eligible when its validation regression rate stays under this.
REGRESSION_BUDGET = 0.02

BASE_FEATURES = (
    "data_per_string",
    "desc_bits",
    "is_heuristic",
    "literal_chars",
    "alt_literal_chars",
    "n_alternations",
    "n_alt_arms",
    "class_breadth",
    "n_unbounded",
    "range_slack",
    "n_optional",
    "n_exact",
    "n_nodes",
    "fixed_fraction",
    "pattern_len",
    "has_dot",
    "has_hex",
    "n_repeats",
    "digit_cost",
)
RELATIVE_FEATURES = (
    "data_per_string",
    "desc_bits",
    "literal_chars",
    "alt_literal_chars",
    "class_breadth",
    "n_nodes",
    "fixed_fraction",
    "digit_cost",
    "n_unbounded",
)
FEATURE_NAMES = BASE_FEATURES + tuple(f"rel_{name}" for name in RELATIVE_FEATURES)


def _quant_flags(node, stats: dict) -> None:
    if node.hi is None:
        stats["n_unbounded"] += 1.0
    elif node.lo == node.hi and node.lo > 0:
        stats["n_exact"] += 1.0
    elif node.hi is not None and node.lo != node.hi:
        stats["range_slack"] += float(node.hi - node.lo)
    if node.lo == 0:
        stats["n_optional"] += 1.0


def _walk(node, under_alt: bool, stats: dict) -> None:
    if node.op == "ALT":
        stats["n_alternations"] += 1.0
        stats["n_alt_arms"] += float(len(node.kids))
        for kid in node.kids:
            _walk(kid, True, stats)
        return
    if node.op == "SEQ":
        for kid in node.kids:
            _walk(kid, under_alt, stats)
        return
    if node.op == "REPEAT":
        stats["n_repeats"] += 1.0
        _quant_flags(node, stats)
        if node.kids:
            _walk(node.kids[0], under_alt, stats)
        return
    if node.op == "LIT":
        count = len(node.text) * max(node.lo, 1)
        stats["literal_chars"] += float(count)
        if under_alt:
            stats["alt_literal_chars"] += float(count)
        _quant_flags(node, stats)
        return
    if node.op == "CLS":
        mid = float(node.lo if node.hi is None else (node.lo + node.hi) / 2)
        stats["class_breadth"] += mid * math.log2(max(class_size(node.text), 2))
        stats["class_slots"] += mid
        if "0-9a-f" in node.text or "0-9A-F" in node.text:
            stats["has_hex"] = 1.0
        _quant_flags(node, stats)
        return
    if node.op == "DOT":
        mid = float(node.lo if node.hi is None else (node.lo + node.hi) / 2)
        stats["class_breadth"] += mid * math.log2(95)
        stats["class_slots"] += mid
        stats["has_dot"] = 1.0
        _quant_flags(node, stats)


def _empty_stats() -> dict:
    return {
        "literal_chars": 0.0,
        "alt_literal_chars": 0.0,
        "n_alternations": 0.0,
        "n_alt_arms": 0.0,
        "class_breadth": 0.0,
        "class_slots": 0.0,
        "n_unbounded": 0.0,
        "range_slack": 0.0,
        "n_optional": 0.0,
        "n_exact": 0.0,
        "has_dot": 0.0,
        "has_hex": 0.0,
        "n_repeats": 0.0,
    }


def raw_features(pattern: str, heuristic: str, strings: list[str]) -> dict[str, float]:
    """Visible-only description of one candidate."""
    stats = _empty_stats()
    node = parse_pattern(pattern)
    if node is not None:
        _walk(node, False, stats)
        nodes = float(description_nodes(node))
    else:
        nodes = 0.0
    n_strings = max(len(strings), 1)
    data = observed_data_bits(pattern, strings, False)
    if data is None:
        data = 1_000_000.0
    digit_cost = pattern_cost(pattern, strings, True)
    literal = stats["literal_chars"]
    slots = stats["class_slots"]
    return {
        "data_per_string": data / n_strings,
        "desc_bits": 1.5 * nodes,
        "is_heuristic": 1.0 if pattern == heuristic else 0.0,
        "literal_chars": literal,
        "alt_literal_chars": stats["alt_literal_chars"],
        "n_alternations": stats["n_alternations"],
        "n_alt_arms": stats["n_alt_arms"],
        "class_breadth": stats["class_breadth"],
        "n_unbounded": stats["n_unbounded"],
        "range_slack": stats["range_slack"],
        "n_optional": stats["n_optional"],
        "n_exact": stats["n_exact"],
        "n_nodes": nodes,
        "fixed_fraction": literal / (literal + slots + 1e-6),
        "pattern_len": float(len(pattern)),
        "has_dot": stats["has_dot"],
        "has_hex": stats["has_hex"],
        "n_repeats": stats["n_repeats"],
        "digit_cost": digit_cost / n_strings,
    }


def pool_matrix(patterns: list[str], heuristic: str, strings: list[str]) -> list[list[float]]:
    """Feature rows for one task, including within-pool relative features."""
    raws = [raw_features(pattern, heuristic, strings) for pattern in patterns]
    means = {
        name: sum(raw[name] for raw in raws) / len(raws)
        for name in RELATIVE_FEATURES
    }
    rows = []
    for raw in raws:
        vector = [raw[name] for name in BASE_FEATURES]
        vector.extend(raw[name] - means[name] for name in RELATIVE_FEATURES)
        rows.append(vector)
    return rows


def alt_literal_cost(pattern: str, strings: list[str]) -> float:
    """Code length plus a charge for literals that live inside an alternation.

    Those literals are how the old rule memorizes the visible strings. Shared
    prefixes outside an alternation stay on the original cost.
    """
    base = pattern_cost(pattern, strings, True)
    features = raw_features(pattern, pattern, strings)
    return base + features["alt_literal_chars"] * math.log2(95) * max(len(strings), 1)


def softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    peak = max(scores)
    weights = [math.exp(score - peak) for score in scores]
    total = sum(weights) or 1.0
    return [weight / total for weight in weights]


def standardize_fit(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    width = len(rows[0])
    count = float(len(rows))
    means = []
    scales = []
    for col in range(width):
        column = [row[col] for row in rows]
        mean = sum(column) / count
        variance = sum((value - mean) ** 2 for value in column) / count
        scale = math.sqrt(variance)
        means.append(mean)
        scales.append(scale if scale >= 1e-8 else 1.0)
    return means, scales


def standardize_apply(row: list[float], means: list[float], scales: list[float]) -> list[float]:
    return [(value - mean) / scale for value, mean, scale in zip(row, means, scales)]


def _dot(weights: list[float], row: list[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, row))


def rank_index(scores: list[float], patterns: list[str], heuristic: str) -> int:
    best = 0
    for index, score in enumerate(scores):
        if score > scores[best] + 1e-12:
            best = index
        elif abs(score - scores[best]) <= 1e-12:
            if patterns[index] == heuristic and patterns[best] != heuristic:
                best = index
            elif patterns[best] != heuristic and len(patterns[index]) < len(patterns[best]):
                best = index
    return best


def probability_gap(scores: list[float]) -> float:
    if len(scores) < 2:
        return 1.0
    ordered = sorted(softmax(scores), reverse=True)
    return ordered[0] - ordered[1]


def apply_gate(learned: str, baseline: str, prob_gap: float, threshold: float) -> str:
    """Use the learned pattern only when the top two probabilities separate."""
    if prob_gap + 1e-12 >= threshold:
        return learned
    return baseline


def listwise_loss(weights: list[float], rows: list[list[list[float]]], targets: list[list[float]], lam: float) -> float:
    total = 0.0
    count = 0
    for matrix, target in zip(rows, targets):
        scores = [_dot(weights, row) for row in matrix]
        pred = softmax(scores)
        total += -sum(gold * math.log(max(prob, 1e-12)) for gold, prob in zip(target, pred))
        count += 1
    penalty = lam * sum(weight * weight for weight in weights)
    return total / max(count, 1) + penalty


def fit_listwise(
    train_rows: list[list[list[float]]],
    train_accuracy: list[list[float]],
    val_rows: list[list[list[float]]],
    val_accuracy: list[list[float]],
    val_patterns: list[list[str]],
    val_heuristic: list[str],
    lam: float,
    tau: float,
    steps: int = 40,
    lr: float = 0.2,
    patience: int = 6,
) -> tuple[list[float], float, int]:
    """Gradient descent on soft cross-entropy. Stops on validation semantic accuracy."""
    width = len(train_rows[0][0])
    weights = [0.0] * width
    # Start near the code-length rule: higher digit cost should lower the score.
    cost_index = BASE_FEATURES.index("digit_cost")
    if cost_index < width:
        weights[cost_index] = -1.0
    targets = [softmax([accuracy / tau for accuracy in row]) for row in train_accuracy]
    best_weights = weights[:]
    best_accuracy = -1.0
    best_step = 0
    stale = 0
    step_lr = lr
    for step in range(steps):
        grad = [lam * 2.0 * weight for weight in weights]
        for matrix, target in zip(train_rows, targets):
            scores = [_dot(weights, row) for row in matrix]
            pred = softmax(scores)
            for row, prob, gold in zip(matrix, pred, target):
                scale = prob - gold
                for index, value in enumerate(row):
                    grad[index] += scale * value / len(train_rows)
        norm = math.sqrt(sum(value * value for value in grad)) or 1.0
        if norm > 5.0:
            grad = [value * 5.0 / norm for value in grad]
        weights = [weight - step_lr * part for weight, part in zip(weights, grad)]
        step_lr *= 0.98
        accuracy = _selection_accuracy(weights, val_rows, val_accuracy, val_patterns, val_heuristic)
        if accuracy > best_accuracy + 1e-4:
            best_accuracy = accuracy
            best_weights = weights[:]
            best_step = step + 1
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_weights, best_accuracy, best_step


def _selection_accuracy(
    weights: list[float],
    rows: list[list[list[float]]],
    accuracy: list[list[float]],
    patterns: list[list[str]],
    heuristic: list[str],
) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for matrix, acc, pats, heur in zip(rows, accuracy, patterns, heuristic):
        scores = [_dot(weights, row) for row in matrix]
        total += acc[rank_index(scores, pats, heur)]
    return total / len(rows)


def score_patterns(weights: list[float], matrix: list[list[float]]) -> list[float]:
    return [_dot(weights, row) for row in matrix]
