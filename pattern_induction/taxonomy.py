"""Rule-based failure categories for semantic misses."""

from __future__ import annotations

import re


def gold_features(gold: str) -> dict:
    return {
        "alternation": "|" in gold,
        "optional": bool(re.search(r"(?<!\\)\?", gold)),
        "variable_length": bool(re.search(r"\{[0-9]+,[0-9]+\}", gold))
        or bool(re.search(r"(?<!\\)[*+]", gold)),
        "char_class": bool(re.search(r"\\[dDwWsS]|\[[^\]]+\]", gold)),
    }


def categorize(row: dict) -> str:
    """Primary failure category. Empty string means the case succeeded."""
    accuracy = row["semantic_accuracy"]
    if accuracy >= 0.9 and row["positive_recall"] >= 0.85 and row["negative_rejection"] >= 0.85:
        return ""
    if not row["valid"]:
        return "invalid_output"
    if row.get("timed_out"):
        return "timeout"
    if row["catastrophic"] or (
        row["negative_rejection"] < 0.35 and row["positive_recall"] > 0.9
    ):
        return "too_broad"
    features = gold_features(row["gold_regex"])
    domain = row.get("domain", "")
    if row["positive_recall"] < 0.6 and row["negative_rejection"] >= 0.8:
        if features["variable_length"]:
            return "variable-length inference"
        if features["optional"]:
            return "optional region"
        if features["alternation"]:
            return "alternation"
        if "ocr" in domain:
            return "OCR noise"
        return "too narrow"
    if row["negative_rejection"] < 0.6 and row["positive_recall"] >= 0.8:
        if features["char_class"] and not features["variable_length"]:
            return "character-class inference"
        return "too broad"
    if row["positive_recall"] < 0.75 and row["negative_rejection"] < 0.75:
        if domain in {"email", "url", "currency", "phone", "postal_code", "filename"}:
            return "semantic class recognition"
        return "mixed population"
    if features["optional"]:
        return "optional region"
    if features["alternation"]:
        return "alternation"
    if features["variable_length"]:
        return "variable-length inference"
    if features["char_class"]:
        return "character-class inference"
    if "ocr" in domain:
        return "OCR noise"
    if domain in {"email", "url", "currency", "phone", "date", "postal_code"}:
        return "format interpretation"
    return "common-prefix/suffix"


def summarize(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        label = categorize(row) or "success"
        counts[label] = counts.get(label, 0) + 1
    total = len(rows) or 1
    return {
        label: {"n": n, "rate": round(n / total, 4)}
        for label, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    }
