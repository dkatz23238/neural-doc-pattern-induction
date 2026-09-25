"""Frontier-model prompts for Experiment 1 and the Experiment 2 ablation.

No network call is made unless the selected provider key is present.
Scores are never invented.
"""

from __future__ import annotations

import os

SIMPLE_PROMPT = """Infer the most useful generalized regular expression describing the common structure of these strings.

Return only the regex.

<examples>
{examples}
</examples>
"""

GUARDED_PROMPT = """Infer the most useful generalized regular expression describing the common structure of these strings.

Return only the regex.
Do not answer with .*, .+, or \\w+.
Do not memorize the examples as an alternation of the full strings.
Do not over-generalize past the structure the examples support.

<examples>
{examples}
</examples>
"""

FEATURE_PROMPT = """Infer the most useful generalized regular expression for these strings.
You are also given deterministic structural features. Use them.
Return only the regex.

<examples>
{examples}
</examples>

<features>
{features}
</features>
"""


def format_examples(strings: list[str]) -> str:
    return "\n".join(strings)


def prompts_for(strings: list[str], features: str = "") -> dict[str, str]:
    examples = format_examples(strings)
    out = {
        "simple": SIMPLE_PROMPT.format(examples=examples),
        "guarded": GUARDED_PROMPT.format(examples=examples),
    }
    if features:
        out["features"] = FEATURE_PROMPT.format(examples=examples, features=features)
    return out


def provider_status() -> dict:
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "note": (
            "Frontier baselines are part of Gate 1 but were not executed in this run "
            "because neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set."
        ),
    }
