"""Structural features for Experiment 2 (LLM ablation) and the hybrid model."""

from __future__ import annotations

from pattern_induction.heuristic import tokenize


def structural_features(strings: list[str]) -> dict:
    if not strings:
        return {}
    prefix = strings[0]
    suffix = strings[0]
    for text in strings[1:]:
        while prefix and not text.startswith(prefix):
            prefix = prefix[:-1]
        while suffix and not text.endswith(suffix):
            suffix = suffix[:-1]
    lengths = sorted({len(s) for s in strings})
    token_sigs = ["".join(tok.kind for tok in tokenize(s)) for s in strings]
    columns = None
    if len(lengths) == 1:
        width = lengths[0]
        columns = []
        for i in range(width):
            chars = sorted({s[i] for s in strings})
            columns.append("".join(chars)[:12])
    return {
        "common_prefix": prefix,
        "common_suffix": suffix,
        "lengths": lengths,
        "token_signatures": sorted(set(token_sigs)),
        "columns": columns,
        "n": len(strings),
    }


def render_features(strings: list[str]) -> str:
    feat = structural_features(strings)
    lines = [
        f"common_prefix: {feat['common_prefix']!r}",
        f"common_suffix: {feat['common_suffix']!r}",
        f"lengths: {feat['lengths']}",
        f"token_signatures: {feat['token_signatures']}",
    ]
    if feat["columns"] is not None:
        lines.append("columns: " + " | ".join(feat["columns"]))
    return "\n".join(lines)
