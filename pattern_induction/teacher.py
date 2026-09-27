"""Stateless teacher calls.

Every request is stored with its prompt, the model id, the raw response, a
hash, and the compiler's accept/reject decision. Rebuilding a recipe reads
those files. It does not call the model again.

The teacher proposes schemas. schema_dsl.py decides which strings are members.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

from pattern_induction.frontier import response_text
from pattern_induction.schema_dsl import (
    GENERATOR_VERSION,
    Language,
    SchemaError,
    compile_schema,
    languages_differ,
)

TEACHER_MODEL = "gpt-5.6-terra"
FALLBACK_MODELS = ("gpt-4.1",)
# Published per-million rates used to enforce the cap. The ledger stores the
# token counts so the dollar figure can be recomputed if the rate changes.
PRICE_PER_MILLION = {
    "gpt-5.6-terra": (2.0, 12.0),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4o": (2.5, 10.0),
}
BUDGET_USD = 12.0
COMPILER_VERSION = GENERATOR_VERSION


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _ledger_path(root: Path) -> Path:
    return root / "teacher_runs" / "ledger.json"


def load_ledger(root: Path) -> dict:
    path = _ledger_path(root)
    if not path.exists():
        return {"spent_usd": 0.0, "calls": 0, "budget_usd": BUDGET_USD, "events": []}
    return json.loads(path.read_text())


def _save_ledger(root: Path, ledger: dict) -> None:
    path = _ledger_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2) + "\n")


def estimate_cost(model: str, usage: dict) -> float:
    input_rate, output_rate = PRICE_PER_MILLION.get(model, (10.0, 50.0))
    in_tok = float(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    out_tok = float(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    return in_tok / 1_000_000 * input_rate + out_tok / 1_000_000 * output_rate


def call_teacher(model: str, prompt: str, max_output_tokens: int, timeout: float = 120.0) -> dict:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    body = json.dumps({"model": model, "input": prompt, "max_output_tokens": max_output_tokens}).encode()
    delay = 1.0
    last_error = ""
    for _ in range(4):
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:500]
            last_error = f"HTTP {exc.code}: {detail}"
            if exc.code == 404 or "model_not_found" in detail:
                raise RuntimeError(last_error) from exc
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError(last_error) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(delay)
        delay = min(delay * 2, 16.0)
    raise RuntimeError(last_error or "teacher call failed")


def extract_payload(text: str) -> dict:
    raw = text.strip()
    fenced = raw
    if "```" in raw:
        start = raw.find("```")
        rest = raw[start + 3 :]
        if "\n" in rest:
            rest = rest.split("\n", 1)[1]
        end = rest.find("```")
        fenced = rest[:end] if end >= 0 else rest
    decoder = json.JSONDecoder()
    for candidate in (fenced, raw):
        candidate = candidate.strip()
        for index, char in enumerate(candidate):
            if char not in "{[":
                continue
            try:
                value, _end = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
            if isinstance(value, list):
                return {"schemas": value}
    raise SchemaError("no_json")


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def validate_schema_record(record: dict, allow_heldout: bool, rng: random.Random) -> tuple[dict | None, str]:
    try:
        compiled = compile_schema(record, allow_heldout=allow_heldout)
        lang = Language(compiled, rng)
        lang.support(rng)
        lang.positive(rng, set())
        lang.negative(rng, set())
    except SchemaError as exc:
        return None, exc.reason
    stored = {"id": compiled.ident, "schema": compiled.schema, "generator_version": COMPILER_VERSION}
    if compiled.confusable:
        stored["confusable"] = compiled.confusable
    if compiled.heldout:
        stored["exposes_eval_family"] = compiled.heldout
    return stored, ""


def validate_pair_record(record: dict, rng: random.Random) -> tuple[dict | None, str]:
    left = record.get("schema_a") or record.get("A")
    right = record.get("schema_b") or record.get("B")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None, "pair_missing_schema"
    try:
        compiled_left = compile_schema({"schema": left}, allow_heldout=True)
        compiled_right = compile_schema({"schema": right}, allow_heldout=True)
    except SchemaError as exc:
        return None, exc.reason
    different = False
    for _ in range(4):
        try:
            if languages_differ(Language(compiled_left, rng), Language(compiled_right, rng), rng):
                different = True
                break
        except SchemaError:
            continue
    if not different:
        return None, "not_different"
    ident = str(record.get("id") or compiled_left.ident)
    stored = {
        "id": ident,
        "distinction": str(record.get("distinction") or ""),
        "schema_a": compiled_left.schema,
        "schema_b": compiled_right.schema,
        "generator_version": COMPILER_VERSION,
        "exposes_eval_family": [name for name in (compiled_left.heldout, compiled_right.heldout) if name],
    }
    return stored, ""


def _next_run_dir(root: Path) -> Path:
    base = root / "teacher_runs"
    base.mkdir(parents=True, exist_ok=True)
    existing = [int(path.name.split("_")[-1]) for path in base.glob("run_*") if path.name.split("_")[-1].isdigit()]
    run_id = (max(existing) + 1) if existing else 1
    path = base / f"run_{run_id:04d}"
    path.mkdir(parents=True)
    return path


def reserve(root: Path, model: str, max_output_tokens: int) -> None:
    ledger = load_ledger(root)
    _input_rate, output_rate = PRICE_PER_MILLION.get(model, (10.0, 50.0))
    # Reserve the output cap plus a small input allowance before the call.
    reservation = max_output_tokens / 1_000_000 * output_rate + 0.05
    if ledger["spent_usd"] + reservation > BUDGET_USD:
        raise RuntimeError(
            f"teacher budget would pass ${BUDGET_USD:.2f}: spent ${ledger['spent_usd']:.4f}, reserve ${reservation:.4f}"
        )


def generate(
    root: Path,
    prompt_name: str,
    kind: str,
    allow_heldout: bool,
    failure_packet: str = "",
    model: str = TEACHER_MODEL,
    max_output_tokens: int = 2500,
) -> dict:
    """One teacher call. kind is 'schemas' or 'pairs'."""
    prompt_path = root / "data" / "teacher" / "prompts" / prompt_name
    template = prompt_path.read_text()
    prompt = template.replace("{failure_packet}", failure_packet) if failure_packet else template
    reserve(root, model, max_output_tokens)
    started = time.time()
    try:
        payload = call_teacher(model, prompt, max_output_tokens)
    except RuntimeError as exc:
        if "model_not_found" not in str(exc):
            raise
        payload = None
        for fallback in FALLBACK_MODELS:
            try:
                payload = call_teacher(fallback, prompt, max_output_tokens)
                model = fallback
                break
            except RuntimeError as nested:
                if "model_not_found" not in str(nested):
                    raise
        if payload is None:
            raise
    raw = response_text(payload)
    usage = payload.get("usage") or {}
    cost = estimate_cost(model, usage)
    response_hash = _sha(raw)
    run_dir = _next_run_dir(root)
    request = {
        "model": model,
        "prompt_file": str(prompt_path.relative_to(root)),
        "prompt_sha256": _sha(template),
        "prompt": prompt,
        "max_output_tokens": max_output_tokens,
        "kind": kind,
        "allow_heldout": allow_heldout,
        "compiler_version": COMPILER_VERSION,
    }
    (run_dir / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    (run_dir / "response.json").write_text(json.dumps({"raw": raw, "usage": usage, "payload": payload}, indent=2) + "\n")
    accepted: list[dict] = []
    rejected: list[dict] = []
    try:
        parsed = extract_payload(raw)
    except SchemaError as exc:
        parsed = {}
        rejected.append({"reason": exc.reason, "raw_sha256": response_hash})
    rng = random.Random(0)
    rows = parsed.get("pairs") if kind == "pairs" else parsed.get("schemas") or []
    if not isinstance(rows, list):
        rows = []
        rejected.append({"reason": "payload_shape", "raw_sha256": response_hash})
    for row in rows:
        if not isinstance(row, dict):
            rejected.append({"reason": "row_not_object"})
            continue
        if kind == "pairs":
            stored, reason = validate_pair_record(row, rng)
            target = root / "data" / "teacher" / "validated" / "minimal_pairs.jsonl"
        else:
            stored, reason = validate_schema_record(row, allow_heldout, rng)
            name = "adversarial_templates.jsonl" if "failure" in prompt_name else "schemas.jsonl"
            target = root / "data" / "teacher" / "validated" / name
        if stored is None:
            rejected.append({"reason": reason, "id": row.get("id"), "row": row})
            continue
        stored["response_sha256"] = response_hash
        stored["prompt"] = prompt_name
        stored["model"] = model
        accepted.append(stored)
        _append_jsonl(target, stored)
    for item in rejected:
        item["response_sha256"] = response_hash
        item["prompt"] = prompt_name
        _append_jsonl(root / "data" / "teacher" / "rejected" / "invalid_schemas.jsonl", item)
    _append_jsonl(
        root / "data" / "teacher" / "raw" / "batches.jsonl",
        {"run": run_dir.name, "model": model, "prompt": prompt_name, "raw": raw, "usage": usage},
    )
    ledger = load_ledger(root)
    ledger["spent_usd"] = round(ledger["spent_usd"] + cost, 6)
    ledger["calls"] = ledger["calls"] + 1
    ledger["events"].append(
        {"run": run_dir.name, "model": model, "cost_usd": round(cost, 6), "usage": usage, "accepted": len(accepted)}
    )
    _save_ledger(root, ledger)
    metadata = {
        "model": model,
        "prompt": prompt_name,
        "response_sha256": response_hash,
        "usage": usage,
        "cost_usd": round(cost, 6),
        "spent_usd": ledger["spent_usd"],
        "compiler_version": COMPILER_VERSION,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "seconds": round(time.time() - started, 3),
        "accepted_ids": [row.get("id") for row in accepted],
        "reject_reasons": [row.get("reason") for row in rejected],
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
