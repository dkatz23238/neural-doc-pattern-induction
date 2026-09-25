"""Frontier-model prompts for Experiment 1 and the Experiment 2 ablation.

No network call is made unless the selected provider key is present.
Scores are never invented.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

# Gate 1 pair. gpt-6-terra is not on this key (model_not_found). The Terra
# model that is available is gpt-5.6-terra. Neither accepts temperature.
GATE1_MODELS = ("gpt-6-luna", "gpt-5.6-terra")
TEMPERATURE_NOTE = (
    "These models reject temperature. Chat completions accepts only the default "
    "value 1, and the Responses API rejects the temperature parameter. "
    "Calls omit temperature."
)

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


def extract_regex(text: str) -> str:
    """Take the first regex line from a model reply. Empty if there is none."""
    if not text:
        return ""
    fenced = re.search(r"```(?:regex|re|python)?\s*\n?(.*?)```", text, re.S | re.I)
    body = fenced.group(1) if fenced else text
    for line in body.splitlines():
        line = line.strip().strip("`").strip()
        if not line or line.lower().startswith(("here", "the regex", "regex:")):
            continue
        if line.lower().startswith("regex:"):
            line = line.split(":", 1)[1].strip()
        return line
    return ""


def response_text(payload: dict) -> str:
    if payload.get("output_text"):
        return payload["output_text"]
    parts: list[str] = []
    for item in payload.get("output") or []:
        if item.get("type") != "message":
            continue
        for block in item.get("content") or []:
            if block.get("type") == "output_text" and block.get("text"):
                parts.append(block["text"])
    return "\n".join(parts)


def call_frontier(model: str, prompt: str, timeout: float = 90.0) -> tuple[str, str, dict]:
    """Call the Responses API. Returns (regex, raw_text, usage)."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    body = json.dumps(
        {"model": model, "input": prompt, "max_output_tokens": 400}
    ).encode()
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    delay = 1.0
    last_error = ""
    for _ in range(6):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode())
            raw = response_text(payload)
            return extract_regex(raw), raw, payload.get("usage") or {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:400]
            last_error = f"HTTP {exc.code}: {detail}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError(last_error) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(delay)
        delay = min(delay * 2, 20.0)
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
    raise RuntimeError(last_error or "frontier call failed")


def provider_status() -> dict:
    openai = bool(os.environ.get("OPENAI_API_KEY"))
    anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if openai:
        note = (
            "Gate 1 calls gpt-6-luna and gpt-5.6-terra. "
            "gpt-6-terra is not available on this key. "
            + TEMPERATURE_NOTE
        )
    else:
        note = (
            "Frontier baselines are part of Gate 1 but were not executed in this run "
            "because neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set."
        )
    return {"openai": openai, "anthropic": anthropic, "note": note}
