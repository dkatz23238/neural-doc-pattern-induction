"""Teacher-driven concept bank.

A model may invent a field, its meaning, and a candidate program. This module
stores the raw response, compiles the program into Needle's DSL, and keeps the
concept only when ``execute`` accepts every positive and rejects every near
miss. A rejected response stays on disk. The expanded training shards are not
written here; a manifest names the bank hash, the parser version, and the seed
that would regenerate them.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import string
import time
from pathlib import Path

from pattern_induction.charset import charset_spec, spec_chars
from pattern_induction.lexicon import LEXICONS, lexicon_members
from pattern_induction.program import OPEN, Prog, describe, execute
from pattern_induction.renaming import apply_mapping

PARSER_VERSION = "factory-4"
GENERATOR_VERSION = "factory-gen-1"
TEACHER_MODEL = "gpt-5.6-terra"
# gpt-6-terra is not on this key. The Terra model that answers is gpt-5.6-terra.
PRICE_PER_MILLION = {"gpt-5.6-terra": (2.0, 12.0)}
BUDGET_USD = 10.0
# Shapes the procedural sampler in prior/generate.py actually emits.
# A teacher template outside this set is a novel composition.
PROCEDURAL_SHAPES = frozenset(
    {
        "int",
        "cls",
        "set",
        "lex",
        "seq(lit,int)",
        "seq(lit,cls)",
        "seq(cls,lit)",
        "seq(lit,lit,cls)",
        "seq(lit,lit,set)",
        "seq(opt(seq(lit,lit)),cls)",
        "seq(opt(seq(lit,lit)),set)",
        "seq(cls,repeat(seq(lit,cls)))",
        "seq(int,repeat(seq(lit,int)))",
        "seq(lit,lit,lex)",
        "seq(cls,lit,lex)",
        "seq(lex,lit,lit)",
    }
)
CAMPAIGN_DOMAINS = (
    "software",
    "finance",
    "legal",
    "healthcare",
    "logistics",
    "manufacturing",
    "energy",
    "agriculture",
    "retail",
    "telecom",
    "government",
    "scientific_data",
    "document_processing",
    "cloud_infrastructure",
    "education",
    "insurance",
)
_GROUP_STEMS = (
    ("semantic version", "semver"),
    ("semver", "semver"),
    ("iso 6346", "container"),
    ("container", "container"),
    ("iso 4217", "currency"),
    ("iso4217", "currency"),
    ("currency code", "currency"),
    ("air waybill", "waybill"),
    ("waybill", "waybill"),
    ("un/locode", "locode"),
    ("locode", "locode"),
    ("aba routing", "routing"),
    ("routing transit", "routing"),
    ("bluebook", "citation"),
    ("u.s. reports", "citation"),
    ("westlaw", "citation"),
    ("nodeport", "kubernetes"),
    ("kubernetes", "kubernetes"),
    ("medicare", "medicare"),
    ("hl7", "hl7"),
    ("treasury", "treasury"),
    ("bloomberg", "bloomberg"),
)
_GENERIC_TYPES = frozenset({"identifier", "id", "code", "string", "field", "number", "token", "unspecified", ""})
_TEACHER_REASONS = frozenset(
    {
        "duplicate_example",
        "duplicate",
        "overlap",
        "near_miss_accepted",
        "positive_rejected",
        "too_few_positives",
        "missing_identity",
        "memorized_literal",
        "ambiguity_rejected",
        "bad_neighbors",
        "concept_not_object",
        "bad_ambiguity",
    }
)
_PARSER_REASONS = frozenset(
    {
        "no_json",
        "no_concepts",
        "node_not_object",
        "unknown_op",
        "bad_class",
        "bad_int",
        "bad_set",
        "bad_opt",
        "bad_repeat",
    }
)
_DSL_REASONS = frozenset(
    {
        "bad_width",
        "bad_literal",
        "unbound_literal",
        "bad_bounds",
        "repeat_cap",
        "nested_repeat",
        "too_many_nodes",
        "unknown_lexicon",
        "int_width",
        "bad_alt",
        "leading_zero",
    }
)
MAX_NODES = 16
MAX_TEXT = 32
LEXICON_NAMES = frozenset(item.name for item in LEXICONS)
_CLASS_ALPHABET = {
    "digit": string.digits,
    "upper": string.ascii_uppercase,
    "lower": string.ascii_lowercase,
}


class FactoryError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _walk_count(node: dict) -> int:
    children = []
    if isinstance(node.get("children"), list):
        children.extend(node["children"])
    if isinstance(node.get("child"), dict):
        children.append(node["child"])
    return 1 + sum(_walk_count(child) for child in children if isinstance(child, dict))


def _bounds(item: dict, default_lo: int, default_hi: int) -> tuple[int, int]:
    lo = item.get("lo", default_lo)
    hi = item.get("hi", default_hi)
    if not isinstance(lo, int) or not isinstance(hi, int) or isinstance(lo, bool) or isinstance(hi, bool):
        raise FactoryError("bad_bounds")
    if lo < 1 or hi < lo:
        raise FactoryError("bad_bounds")
    return lo, hi


def _check_width(lo: int, hi: int) -> None:
    if hi > 32 and hi != 128:
        raise FactoryError("bad_width")
    if lo > 32:
        raise FactoryError("bad_width")


_LEX_ALIASES = {
    "iso4217_currency_codes": "iso4217",
    "iso4217_currency_code": "iso4217",
    "iso_4217": "iso4217",
    "iso4217_codes": "iso4217",
}


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _width_pair(left, right) -> bool:
    return _is_int(left) and _is_int(right) and 1 <= left <= right <= 8


def _range_pair(left, right) -> bool:
    return _is_int(left) and _is_int(right) and 0 <= left <= right <= 10**9


def _int_fields(item: dict) -> dict:
    """Read an integer node.

    The prompt puts the numeric window in min/max and the digit width in lo/hi.
    One teacher response swapped those pairs. A pair is a width only when both
    ends sit between 1 and 8, so a year window cannot be mistaken for a width.
    """
    minimum, maximum = item.get("min"), item.get("max")
    lo, hi = item.get("lo"), item.get("hi")
    prompt = _width_pair(lo, hi) and _range_pair(minimum, maximum)
    flipped = _width_pair(minimum, maximum) and _range_pair(lo, hi) and not _width_pair(lo, hi)
    if prompt:
        return {"min": minimum, "max": maximum, "lo": lo, "hi": hi}
    if flipped:
        return {"min": lo, "max": hi, "lo": minimum, "hi": maximum}
    if _range_pair(minimum, maximum) and lo is None and hi is None:
        width = max(len(str(minimum)), len(str(maximum)))
        return {"min": minimum, "max": maximum, "lo": width, "hi": width}
    width = item.get("width")
    if _is_int(width) and 1 <= width <= 8 and _range_pair(lo, hi) and minimum is None and maximum is None:
        return {"min": lo, "max": hi, "lo": width, "hi": width}
    raise FactoryError("bad_int")


def _lex_name(value) -> str:
    if isinstance(value, dict):
        value = value.get("name") or value.get("spec") or ""
    name = str(value or "").strip()
    folded = name.lower().replace("-", "_").replace(" ", "_")
    if folded in LEXICON_NAMES:
        return folded
    return _LEX_ALIASES.get(folded, name)


def _set_node(value) -> dict:
    if isinstance(value, str):
        return {"op": "set", "spec": value, "lo": 1, "hi": 1}
    if isinstance(value, dict):
        spec = value.get("spec")
        if isinstance(spec, str):
            return {"op": "set", "spec": spec, "lo": value.get("lo", 1), "hi": value.get("hi", 1)}
        chars = value.get("chars")
        if isinstance(chars, list):
            value = chars
        else:
            raise FactoryError("bad_set")
    if isinstance(value, list):
        if value and all(isinstance(char, str) and len(char) == 1 for char in value):
            return {"op": "set", "spec": charset_spec(set(value)), "lo": 1, "hi": 1}
        raise FactoryError("bad_set")
    raise FactoryError("bad_set")


def _alias_node(item: dict) -> dict:
    """Rename keys from dialects that already use ``op``."""
    op = str(item.get("op") or "").lower()
    if op == "seq" and not isinstance(item.get("children"), list) and isinstance(item.get("args"), list):
        item = {**item, "op": "seq", "children": item["args"]}
    if op == "lit" and not isinstance(item.get("text"), str) and isinstance(item.get("value"), str):
        item = {**item, "text": item["value"]}
    if op == "cls" and not item.get("class") and item.get("kind"):
        item = {**item, "class": item["kind"]}
    if op in {"repeat", "opt"} and not isinstance(item.get("child"), dict) and isinstance(item.get("node"), dict):
        item = {**item, "child": item["node"]}
    return item


def _lift_multikey(item: dict) -> dict:
    """A node whose op is a key beside lo, hi, min, or max."""
    names = {"seq", "cls", "class", "set", "int", "lex", "lit", "literal", "opt", "alt", "repeat"}
    found = [key for key in item if str(key).lower() in names]
    if len(found) != 1:
        raise FactoryError("node_not_object")
    op = str(found[0]).lower()
    value = item[found[0]]
    if op in {"cls", "class"} and isinstance(value, str):
        return {"op": "cls", "class": value, "lo": item.get("lo", 1), "hi": item.get("hi", 1)}
    if op in {"lit", "literal"} and isinstance(value, str):
        return {"op": "lit", "text": value}
    if op == "int" and not isinstance(value, dict):
        return {"op": "int", **_int_fields(item)}
    if op == "lex" and isinstance(value, str):
        return {"op": "lex", "name": value}
    if op == "set" and isinstance(value, str):
        return {"op": "set", "spec": value, "lo": item.get("lo", 1), "hi": item.get("hi", 1)}
    if op == "seq" and isinstance(value, list):
        return {"op": "seq", "children": value}
    if op == "repeat":
        if isinstance(value, dict) and any(key in value for key in ("child", "node", "lo", "hi")):
            spec = dict(value)
        else:
            spec = {"child": value if isinstance(value, dict) else {}, "lo": item.get("lo", 1), "hi": item.get("hi", 1)}
        for key in ("lo", "hi"):
            spec.setdefault(key, item.get(key, 1))
        if "child" not in spec and "node" in spec:
            spec["child"] = spec["node"]
        return {"op": "repeat", "lo": spec.get("lo", 1), "hi": spec.get("hi", 1), "child": spec.get("child")}
    if op == "opt":
        child = value if isinstance(value, dict) else None
        return {"op": "opt", "child": child}
    if op == "alt" and isinstance(value, list):
        return {"op": "alt", "children": value}
    raise FactoryError("node_not_object")


def _canonicalize(item: dict) -> dict:
    """Lift the single-key dialect (`{"seq":[...]}`, `{"lit":"-"}`) into op nodes."""
    if not isinstance(item, dict):
        raise FactoryError("node_not_object")
    if "op" not in item and isinstance(item.get("type"), str):
        item = {key: value for key, value in item.items() if key != "short"}
        item = {**item, "op": item.pop("type")}
    if "op" in item:
        item = _alias_node(item)
        if item.get("op") == "int" and "min" in item:
            return {**item, **_int_fields(item)}
        if item.get("op") == "lex":
            return {**item, "name": _lex_name(item.get("name") or item.get("spec"))}
        return item
    if len(item) != 1:
        return _canonicalize(_lift_multikey(item))
    op, value = next(iter(item.items()))
    op = str(op).lower()
    if op == "seq":
        if not isinstance(value, list) or not value:
            raise FactoryError("bad_seq")
        if len(value) == 1 and isinstance(value[0], dict):
            return _canonicalize(value[0])
        return {"op": "seq", "children": value}
    if op in {"lit", "literal"}:
        if isinstance(value, str):
            return {"op": "lit", "text": value}
        if isinstance(value, dict) and isinstance(value.get("text") or value.get("value"), str):
            return {"op": "lit", "text": value.get("text") or value["value"]}
        raise FactoryError("bad_literal")
    if op in {"cls", "class"}:
        if isinstance(value, str):
            if value not in _CLASS_ALPHABET and spec_chars(value):
                return {"op": "set", "spec": value, "lo": 1, "hi": 1}
            return {"op": "cls", "class": value, "lo": 1, "hi": 1}
        if isinstance(value, dict):
            return {
                "op": "cls",
                "class": value.get("class") or value.get("cls") or value.get("kind"),
                "lo": value.get("lo", 1),
                "hi": value.get("hi", 1),
            }
        raise FactoryError("bad_class")
    if op == "set":
        return _set_node(value)
    if op == "int":
        if not isinstance(value, dict):
            raise FactoryError("bad_int")
        return {"op": "int", **_int_fields(value)}
    if op == "lex":
        return {"op": "lex", "name": _lex_name(value)}
    if op == "opt":
        child = value.get("child") if isinstance(value, dict) and "child" in value and "op" not in value else value
        if not isinstance(child, dict):
            raise FactoryError("bad_opt")
        return {"op": "opt", "child": child}
    if op == "alt":
        if not isinstance(value, list):
            raise FactoryError("bad_alt")
        return {"op": "alt", "children": value}
    if op == "repeat":
        child = value.get("child") if isinstance(value, dict) else None
        if not isinstance(child, dict) and isinstance(value, dict):
            child = value.get("node")
        if not isinstance(value, dict) or not isinstance(child, dict):
            raise FactoryError("bad_repeat")
        return {"op": "repeat", "lo": value.get("lo", 1), "hi": value.get("hi", 1), "child": child}
    raise FactoryError("unknown_op")


def compile_node(item: dict, repeats: int = 0) -> dict:
    """Normalize one template node. Literals stay as text until a support set binds them."""
    if not isinstance(item, dict):
        raise FactoryError("node_not_object")
    item = _canonicalize(item)
    if "op" not in item:
        raise FactoryError("node_not_object")
    if _walk_count(item) > MAX_NODES:
        raise FactoryError("too_many_nodes")
    op = str(item["op"]).lower()
    if op == "seq":
        children = item.get("children")
        if not isinstance(children, list) or not 2 <= len(children) <= 6:
            raise FactoryError("bad_seq")
        return {"op": "seq", "children": [compile_node(child, repeats) for child in children]}
    if op == "cls":
        name = item.get("class") or item.get("cls")
        lo, hi = _bounds(item, 1, 1)
        _check_width(lo, hi)
        if name not in _CLASS_ALPHABET:
            raise FactoryError("bad_class")
        return {"op": "cls", "class": name, "lo": lo, "hi": hi}
    if op == "set":
        spec = item.get("spec")
        lo, hi = _bounds(item, 1, 1)
        _check_width(lo, hi)
        if not isinstance(spec, str) or not spec or len(spec) > 80 or not spec_chars(spec):
            raise FactoryError("bad_set")
        return {"op": "set", "spec": spec, "lo": lo, "hi": hi}
    if op == "int":
        lo, hi = _bounds(item, 1, 1)
        if hi > 8:
            raise FactoryError("bad_width")
        minimum = item.get("min")
        maximum = item.get("max")
        if not isinstance(minimum, int) or not isinstance(maximum, int):
            raise FactoryError("bad_int")
        if isinstance(minimum, bool) or isinstance(maximum, bool):
            raise FactoryError("bad_int")
        if minimum < 0 or maximum < minimum or maximum > 10**9:
            raise FactoryError("bad_int")
        if len(str(maximum)) > hi:
            raise FactoryError("int_width")
        return {"op": "int", "min": minimum, "max": maximum, "lo": lo, "hi": hi}
    if op == "lex":
        name = item.get("name") or item.get("spec")
        if name not in LEXICON_NAMES:
            raise FactoryError("unknown_lexicon")
        return {"op": "lex", "name": name}
    if op == "lit":
        text = item.get("text")
        if not isinstance(text, str) or not 1 <= len(text) <= 8 or any(ord(char) < 32 for char in text):
            raise FactoryError("bad_literal")
        return {"op": "lit", "text": text}
    if op == "opt":
        child = item.get("child")
        if not isinstance(child, dict):
            raise FactoryError("bad_opt")
        return {"op": "opt", "child": compile_node(child, repeats)}
    if op == "alt":
        children = item.get("children")
        if not isinstance(children, list) or len(children) != 2:
            raise FactoryError("bad_alt")
        return {"op": "alt", "children": [compile_node(child, repeats) for child in children]}
    if op == "repeat":
        if repeats:
            raise FactoryError("nested_repeat")
        lo, hi = _bounds(item, 1, 1)
        if hi > OPEN:
            raise FactoryError("repeat_cap")
        child = item.get("child")
        if not isinstance(child, dict):
            raise FactoryError("bad_repeat")
        return {"op": "repeat", "lo": lo, "hi": hi, "child": compile_node(child, repeats + 1)}
    raise FactoryError("unknown_op")


def _literals(node: dict) -> list[str]:
    found = []
    if node["op"] == "lit":
        found.append(node["text"])
    for key in ("child",):
        if key in node:
            found.extend(_literals(node[key]))
    for child in node.get("children") or []:
        found.extend(_literals(child))
    return found


def _operators(node: dict) -> list[str]:
    found = [node["op"]]
    if "child" in node:
        found.extend(_operators(node["child"]))
    for child in node.get("children") or []:
        found.extend(_operators(child))
    return found


def _lexicons(node: dict) -> list[str]:
    found = [node["name"]] if node["op"] == "lex" else []
    if "child" in node:
        found.extend(_lexicons(node["child"]))
    for child in node.get("children") or []:
        found.extend(_lexicons(child))
    return found


def bind_program(node: dict, supports: list[str]) -> Prog:
    """Replace each literal with a COPY of a span that actually occurs in the supports."""

    def locate(text: str) -> Prog:
        for slot, support in enumerate(supports):
            at = support.find(text)
            if at >= 0:
                return Prog("COPY", slot=slot, start=at, end=at + len(text))
        raise FactoryError("unbound_literal")

    def build(item: dict) -> Prog:
        op = item["op"]
        if op == "seq":
            return Prog("SEQ", children=[build(child) for child in item["children"]])
        if op == "cls":
            return Prog("CLS", cls=item["class"], lo=item["lo"], hi=item["hi"])
        if op == "set":
            return Prog("SET", spec=item["spec"], lo=item["lo"], hi=item["hi"])
        if op == "int":
            return Prog("INT", spec=f"{item['min']}:{item['max']}", lo=item["lo"], hi=item["hi"])
        if op == "lex":
            return Prog("LEX", spec=item["name"])
        if op == "lit":
            return locate(item["text"])
        if op == "opt":
            return Prog("OPT", children=[build(item["child"])])
        if op == "alt":
            return Prog("ALT", children=[build(child) for child in item["children"]])
        if op == "repeat":
            return Prog("REPEAT", lo=item["lo"], hi=item["hi"], children=[build(item["child"])])
        raise FactoryError("unknown_op")

    return build(node)


def _strings(value, limit: int) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= limit:
        raise FactoryError("bad_examples")
    rows = []
    for item in value:
        if not isinstance(item, str) or not 1 <= len(item) <= MAX_TEXT:
            raise FactoryError("bad_examples")
        rows.append(item)
    if len(set(rows)) != len(rows):
        raise FactoryError("duplicate_example")
    return rows


def _edge_strings(node: dict) -> list[str]:
    """A few strings at the corners of this node. Used only to tell two programs apart."""

    def one(item: dict, high: bool) -> str | None:
        op = item["op"]
        if op == "seq":
            parts = []
            for child in item["children"]:
                text = one(child, high)
                if text is None:
                    return None
                parts.append(text)
            return "".join(parts)
        if op == "cls":
            alphabet = _CLASS_ALPHABET[item["class"]]
            char = alphabet[-1] if high else alphabet[0]
            width = item["lo"] if not high else min(item["hi"], 32)
            return char * width
        if op == "set":
            chars = sorted(spec_chars(item["spec"]))
            if not chars:
                return None
            char = chars[-1] if high else chars[0]
            width = item["lo"] if not high else min(item["hi"], 32)
            return char * width
        if op == "int":
            value = item["max"] if high else item["min"]
            width = item["hi"] if high else item["lo"]
            if len(str(value)) > width:
                return None
            rendered = f"{value:0{width}d}" if item["lo"] == item["hi"] else str(value)
            if not item["min"] <= int(rendered) <= item["max"]:
                return None
            return rendered
        if op == "lex":
            members = lexicon_members(item["name"])
            if not members:
                return None
            return members[-1] if high else members[0]
        if op == "lit":
            return item["text"]
        if op == "opt":
            return one(item["child"], high) if high else ""
        if op == "alt":
            return one(item["children"][1 if high else 0], high)
        if op == "repeat":
            piece = one(item["child"], high)
            if piece is None:
                return None
            count = item["hi"] if high else item["lo"]
            return piece * count
        return None

    found = []
    for flag in (False, True):
        text = one(node, flag)
        if text and len(text) <= MAX_TEXT:
            found.append(text)
    return found


def _accepts(program: Prog, text: str, supports: list[str]) -> bool:
    if not text or len(text) > MAX_TEXT:
        return False
    return execute(program, text, supports)


def _differs(gold: dict, other: dict, supports: list[str], probes: list[str]) -> bool:
    try:
        left = bind_program(gold, supports)
        right = bind_program(other, supports)
    except FactoryError:
        return False
    for text in probes:
        if _accepts(left, text, supports) != _accepts(right, text, supports):
            return True
    return False


def validate_concept(raw: dict) -> tuple[dict | None, str]:
    """Return a stored template, or None and a reason. The raw object is not mutated."""
    if not isinstance(raw, dict):
        return None, "concept_not_object"
    try:
        domain = str(raw.get("domain") or "").strip().lower()
        concept = str(raw.get("concept") or "").strip()
        if not domain or not concept or len(concept) > 120:
            raise FactoryError("missing_identity")
        description = str(raw.get("description") or "").strip()
        positives = _strings(raw.get("positives") or raw.get("positive_examples"), 8)
        misses = _strings(raw.get("near_misses") or raw.get("negatives"), 8)
        if len(positives) < 2:
            raise FactoryError("too_few_positives")
        if set(positives) & set(misses):
            raise FactoryError("overlap")
        program = compile_node(raw.get("program") or {})
        for literal in _literals(program):
            if any(literal == text for text in positives):
                raise FactoryError("memorized_literal")
        bound = bind_program(program, positives)
        for text in positives:
            if not _accepts(bound, text, positives):
                raise FactoryError("positive_rejected")
        for text in misses:
            if _accepts(bound, text, positives):
                raise FactoryError("near_miss_accepted")
        probes = list(dict.fromkeys(positives + misses + _edge_strings(program)))
        neighbors = []
        proposed = raw.get("neighbors") or raw.get("nearby") or []
        if proposed is None:
            proposed = []
        if not isinstance(proposed, list):
            raise FactoryError("bad_neighbors")
        for item in proposed[:6]:
            if not isinstance(item, dict):
                continue
            body = item.get("program") or item
            try:
                compiled = compile_node(body)
            except FactoryError:
                continue
            if json.dumps(compiled, sort_keys=True) == json.dumps(program, sort_keys=True):
                continue
            extra = _edge_strings(compiled)
            if _differs(program, compiled, positives, probes + extra):
                neighbors.append(compiled)
        ambiguous = raw.get("ambiguous_supports") or []
        ambiguity_status = "absent"
        if ambiguous:
            if not isinstance(ambiguous, list):
                raise FactoryError("bad_ambiguity")
            supports = _strings(ambiguous, 8)
            if any(not _accepts(bound, text, positives) for text in supports):
                raise FactoryError("ambiguity_rejected")
            witnessed = False
            for neighbor in neighbors:
                try:
                    other = bind_program(neighbor, positives)
                except FactoryError:
                    continue
                if all(_accepts(other, text, positives) for text in supports):
                    witnessed = True
                    break
            ambiguity_status = "witnessed" if witnessed else "unwitnessed"
        kinds = raw.get("ambiguity_types") or raw.get("ambiguity") or []
        if isinstance(kinds, str):
            kinds = [kinds]
        if not isinstance(kinds, list):
            kinds = []
        kinds = [str(kind)[:40] for kind in kinds if isinstance(kind, str)]
        canonical = json.dumps(program, sort_keys=True)
        ident = _sha(f"{domain}|{concept}|{canonical}")[:16]
        record = {
            "id": ident,
            "domain": _slug(domain),
            "concept": concept,
            "description": description[:500],
            "semantic_type": str(raw.get("semantic_type") or "")[:80],
            "positives": positives,
            "near_misses": misses,
            "program": program,
            "neighbors": neighbors,
            "ambiguous_supports": ambiguous if isinstance(ambiguous, list) else [],
            "ambiguity_types": kinds,
            "ambiguity_status": ambiguity_status,
            "operators": sorted(set(_operators(program))),
            "lexicons": sorted(set(_lexicons(program))),
            "signature": "+".join(sorted(set(_operators(program)))),
            "parser_version": PARSER_VERSION,
            "validation": "accepted",
        }
        return record, ""
    except FactoryError as exc:
        return None, exc.reason


def _slug(domain: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", domain.lower()).strip("_")
    return (slug or "other")[:40]


def _sample_piece(node: dict, rng: random.Random, include_optional: bool, narrow: bool) -> str | None:
    op = node["op"]
    if op == "seq":
        parts = []
        for child in node["children"]:
            text = _sample_piece(child, rng, include_optional, narrow)
            if text is None:
                return None
            parts.append(text)
        return "".join(parts)
    if op == "cls":
        alphabet = _CLASS_ALPHABET[node["class"]]
        width = node["lo"] if narrow else rng.randint(node["lo"], min(node["hi"], 32))
        return "".join(rng.choice(alphabet) for _ in range(width))
    if op == "set":
        chars = sorted(spec_chars(node["spec"]))
        width = node["lo"] if narrow else rng.randint(node["lo"], min(node["hi"], 32))
        return "".join(rng.choice(chars) for _ in range(width))
    if op == "int":
        width = node["lo"] if narrow else rng.randint(node["lo"], node["hi"])
        if node["lo"] == node["hi"]:
            low = max(node["min"], 0)
            high = min(node["max"], 10**width - 1)
            if low > high:
                return None
            return f"{rng.randint(low, high):0{width}d}"
        low = max(node["min"], 0 if width == 1 else 10 ** (width - 1))
        high = min(node["max"], 10**width - 1)
        if low > high:
            return None
        return str(rng.randint(low, high))
    if op == "lex":
        members = lexicon_members(node["name"])
        return rng.choice(members) if members else None
    if op == "lit":
        return node["text"]
    if op == "opt":
        if not include_optional:
            return ""
        return _sample_piece(node["child"], rng, include_optional, narrow) or ""
    if op == "alt":
        return _sample_piece(rng.choice(node["children"]), rng, include_optional, narrow)
    if op == "repeat":
        count = node["lo"] if narrow else rng.randint(node["lo"], node["hi"])
        parts = []
        for _ in range(count):
            text = _sample_piece(node["child"], rng, include_optional, narrow)
            if text is None:
                return None
            parts.append(text)
        return "".join(parts)
    return None


def _has_op(node: dict, op: str) -> bool:
    return op in _operators(node)


def _rename_supports(node: dict, supports: list[str], rng: random.Random) -> list[str] | None:
    frozen = set("".join(_literals(node)))
    alphabet = [char for char in string.ascii_uppercase if char not in frozen]
    if len(alphabet) < 4 or "cls" not in _operators(node) and "set" not in _operators(node):
        return None
    pool = alphabet[:]
    rng.shuffle(pool)
    mapping = {char: pool[index] for index, char in enumerate(alphabet)}
    renamed = [apply_mapping(text, mapping) for text in supports]
    if renamed == supports:
        return None
    return renamed


def materialize(record: dict, seed: int = 2507, limit: int = 8) -> list[dict]:
    """A handful of support configurations from one accepted template.

    This does not write a corpus. The same seed and template rebuild the rows.
    """
    rng = random.Random(seed)
    program = record["program"]
    configs: list[tuple[str, list[str]]] = []

    def draw(count: int, include_optional: bool, narrow: bool) -> list[str] | None:
        rows = []
        for _ in range(count * 6):
            text = _sample_piece(program, rng, include_optional, narrow)
            if not text or len(text) > MAX_TEXT or text in rows:
                continue
            rows.append(text)
            if len(rows) == count:
                break
        if len(rows) < count:
            return None
        try:
            bound = bind_program(program, rows)
        except FactoryError:
            return None
        if any(not _accepts(bound, text, rows) for text in rows):
            return None
        return rows

    for count in (2, 3, 4):
        rows = draw(count, True, False)
        if rows:
            configs.append((f"supports_{count}", rows))
    if _has_op(program, "opt"):
        rows = draw(4, False, False)
        if rows:
            configs.append(("missing_optional", rows))
    if any(item in record["operators"] for item in ("cls", "set", "int", "repeat")):
        rows = draw(3, True, True)
        if rows:
            configs.append(("narrow", rows))
    if record.get("ambiguity_status") == "witnessed":
        supports = [text for text in record.get("ambiguous_supports") or [] if isinstance(text, str)][:4]
        if len(supports) >= 2:
            configs.append(("ambiguous", supports))
    base = configs[0][1] if configs else None
    if base:
        renamed = _rename_supports(program, base, rng)
        if renamed:
            try:
                bound = bind_program(program, renamed)
            except FactoryError:
                bound = None
            if bound and all(_accepts(bound, text, renamed) for text in renamed):
                configs.append(("renamed", renamed))
    tasks = []
    for name, supports in configs[:limit]:
        bound = bind_program(program, supports)
        candidates = [bound]
        for neighbor in record.get("neighbors") or []:
            try:
                candidates.append(bind_program(neighbor, supports))
            except FactoryError:
                continue
        tasks.append(
            {
                "template_id": record["id"],
                "configuration": name,
                "supports": supports,
                "gold": describe(bound),
                "candidate_count": len(candidates),
                "generator_version": GENERATOR_VERSION,
                "seed": seed,
            }
        )
    return tasks


def _ast_shape(node: dict) -> str:
    """Tree shape with bounds and vocabularies erased. That is the AST signature."""
    op = node.get("op") or "?"
    children = node.get("children")
    if isinstance(children, list) and children:
        return op + "(" + ",".join(_ast_shape(child) for child in children if isinstance(child, dict)) + ")"
    child = node.get("child")
    if isinstance(child, dict):
        return op + "(" + _ast_shape(child) + ")"
    return op


def prog_shape(prog: Prog) -> str:
    """Same shape string as ``_ast_shape``, read off an executable program."""
    names = {
        "SEQ": "seq",
        "CLS": "cls",
        "SET": "set",
        "INT": "int",
        "LEX": "lex",
        "COPY": "lit",
        "OPT": "opt",
        "ALT": "alt",
        "REPEAT": "repeat",
    }
    name = names.get(prog.op, prog.op.lower())
    if not prog.children:
        return name
    return name + "(" + ",".join(prog_shape(child) for child in prog.children) + ")"


def family_key(record: dict) -> str:
    """A concept family. SemVer wordings share a key, and the split never breaks one."""
    blob = f"{record.get('concept') or ''} {record.get('description') or ''}".lower()
    for stem, key in _GROUP_STEMS:
        if stem in blob:
            return key
    semantic = _slug(str(record.get("semantic_type") or ""))
    domain = record.get("domain") or "other"
    if semantic and semantic not in _GENERIC_TYPES:
        return f"{domain}:{semantic}"
    concept = _slug(str(record.get("concept") or "concept"))
    return f"{domain}:{concept}"


def _has_variable_int(node: dict) -> bool:
    if node.get("op") == "int" and node.get("lo") != node.get("hi"):
        return True
    child = node.get("child")
    if isinstance(child, dict) and _has_variable_int(child):
        return True
    for item in node.get("children") or []:
        if isinstance(item, dict) and _has_variable_int(item):
            return True
    return False


def _leading_zero_failure(row: dict) -> bool:
    """A variable-width integer rejected a positive that keeps a leading zero."""
    program = row.get("program")
    positives = row.get("positives")
    if not isinstance(program, dict) or not isinstance(positives, list):
        return False
    try:
        compiled = compile_node(program)
        bound = bind_program(compiled, [text for text in positives if isinstance(text, str)])
    except FactoryError:
        return False
    if not _has_variable_int(compiled):
        return False
    failed = [text for text in positives if isinstance(text, str) and not _accepts(bound, text, positives)]
    return any(re.search(r"(^|[^0-9])0\d", text) for text in failed)


def classify_rejection(item: dict) -> dict[str, str]:
    """Bucket one rejected concept: teacher mistake, DSL limit, or parser miss."""
    reason = str(item.get("reason") or "unknown")
    row = item.get("row") if isinstance(item.get("row"), dict) else {}
    detail = reason
    if reason == "positive_rejected" and _leading_zero_failure(row):
        bucket, detail = "dsl", "leading_zero"
    elif reason == "bad_examples":
        values = row.get("positives") or row.get("near_misses") or []
        overlong = isinstance(values, list) and any(isinstance(text, str) and len(text) > MAX_TEXT for text in values)
        bucket, detail = ("dsl", "max_length") if overlong else ("teacher", reason)
    elif reason == "bad_seq":
        children = (row.get("program") or {}).get("children") if isinstance(row.get("program"), dict) else None
        bucket = "dsl" if isinstance(children, list) and len(children) > 6 else "parser"
    elif reason in _DSL_REASONS:
        bucket = "dsl"
    elif reason in _PARSER_REASONS:
        bucket = "parser"
    elif reason in _TEACHER_REASONS:
        bucket = "teacher"
    else:
        bucket = "teacher"
    labels = {"dsl": "DSL limitation", "parser": "parser limitation", "teacher": "bad teacher output"}
    return {"bucket": bucket, "bucket_label": labels[bucket], "detail": detail, "reason": reason}


def assign_splits(records: list[dict], seed: int = 2507) -> dict:
    """Put whole concept families into train, validation, or held-out.

    The seed only breaks ties. A family is never divided, so two SemVer
    wordings cannot land on opposite sides.
    """
    groups: dict[str, list[str]] = {}
    for record in records:
        groups.setdefault(family_key(record), []).append(record["id"])

    def tie(key: str) -> int:
        return int(_sha(f"{seed}|{key}")[:8], 16)

    order = sorted(groups, key=lambda key: (-len(groups[key]), tie(key), key))
    targets = {"train": 0.70, "validation": 0.15, "heldout": 0.15}
    buckets: dict[str, list[str]] = {name: [] for name in targets}
    families: dict[str, str] = {}
    for key in order:
        ids = groups[key]
        total = sum(len(rows) for rows in buckets.values())
        name = max(
            targets,
            key=lambda item: (targets[item] - (len(buckets[item]) / max(1, total)), targets[item]),
        )
        buckets[name].extend(ids)
        families[key] = name
    return {
        "seed": seed,
        "train": buckets["train"],
        "validation": buckets["validation"],
        "heldout": buckets["heldout"],
        "families": families,
        "counts": {name: len(rows) for name, rows in buckets.items()},
    }


def diversity(records: list[dict]) -> dict:
    def entropy(groups: dict[str, int]) -> float:
        total = sum(groups.values())
        if total <= 1:
            return 0.0
        return -sum((count / total) * math.log2(count / total) for count in groups.values() if count)

    def share(groups: dict[str, int], n: int) -> float:
        if not groups or n == 0:
            return 0.0
        ordered = sorted(groups.values(), reverse=True)
        return sum(ordered[:n]) / float(sum(ordered))

    programs: dict[str, int] = {}
    domains: dict[str, int] = {}
    signatures: dict[str, int] = {}
    shapes: dict[str, int] = {}
    families: dict[str, int] = {}
    concepts: dict[str, int] = {}
    positives = 0
    novel = 0
    for record in records:
        programs[json.dumps(record["program"], sort_keys=True)] = programs.get(json.dumps(record["program"], sort_keys=True), 0) + 1
        domains[record["domain"]] = domains.get(record["domain"], 0) + 1
        signatures[record["signature"]] = signatures.get(record["signature"], 0) + 1
        shape = _ast_shape(record["program"])
        shapes[shape] = shapes.get(shape, 0) + 1
        if shape not in PROCEDURAL_SHAPES:
            novel += 1
        family = str(record.get("semantic_type") or "unspecified")
        families[family] = families.get(family, 0) + 1
        concepts[record["concept"]] = concepts.get(record["concept"], 0) + 1
        positives += len(record.get("positives") or [])
    return {
        "templates": len(records),
        "unique_programs": len(programs),
        "unique_ast_shapes": len(shapes),
        "concepts": len(concepts),
        "domains": len(domains),
        "semantic_families": len(families),
        "program_entropy": round(entropy(programs), 4),
        "domain_entropy": round(entropy(domains), 4),
        "ast_signature_entropy": round(entropy(shapes), 4),
        "semantic_entropy": round(entropy(families), 4),
        "signature_entropy": round(entropy(signatures), 4),
        "top1_program_share": round(share(programs, 1), 4),
        "top10_signature_share": round(share(signatures, 10), 4),
        "top10_family_share": round(share(families, 10), 4),
        "mean_positives": round(positives / len(records), 4) if records else 0.0,
        "witnessed_ambiguity": sum(1 for record in records if record.get("ambiguity_status") == "witnessed"),
        "novel_compositions": novel,
        "novel_composition_share": round(novel / len(records), 4) if records else 0.0,
        "domain_counts": dict(sorted(domains.items())),
        "family_counts": dict(sorted(families.items())),
    }


def _json_values(text: str) -> list:
    decoder = json.JSONDecoder()
    values = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        bracket = text.find("[", index)
        if start < 0 and bracket < 0:
            break
        if start < 0 or (0 <= bracket < start):
            start = bracket
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        values.append(value)
        index = start + max(end, 1)
    return values


def extract_concepts(text: str) -> list[dict]:
    raw = text.strip()
    fenced = raw
    if "```" in raw:
        start = raw.find("```")
        rest = raw[start + 3 :]
        if "\n" in rest:
            rest = rest.split("\n", 1)[1]
        end = rest.find("```")
        fenced = rest[:end] if end >= 0 else rest
    values = _json_values(fenced) or _json_values(raw)
    concepts: list[dict] = []
    for value in values:
        if isinstance(value, dict) and isinstance(value.get("concepts"), list):
            concepts.extend(item for item in value["concepts"] if isinstance(item, dict))
        elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            concepts.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict) and ("program" in value or "concept" in value):
            concepts.append(value)
    if concepts:
        return concepts
    if not values:
        raise FactoryError("no_json")
    raise FactoryError("no_concepts")


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _known_ids(root: Path) -> set[str]:
    found = set()
    bank = root / "data" / "factory" / "templates"
    if not bank.exists():
        return found
    for path in bank.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                found.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return found


def _run_dir(root: Path) -> Path:
    base = root / "data" / "factory" / "runs"
    base.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d")
    existing = list(base.glob(f"{stamp}_terra_*"))
    number = len(existing) + 1
    path = base / f"{stamp}_terra_{number:03d}"
    path.mkdir(parents=True)
    return path


def ingest_text(
    root: Path,
    raw: str,
    *,
    prompt_name: str,
    prompt_text: str,
    model: str,
    usage: dict | None = None,
    request: dict | None = None,
) -> dict:
    """Store one teacher response, then accept or reject each concept."""
    usage = usage or {}
    response_hash = _sha(raw)
    run = _run_dir(root)
    (run / "response.json").write_text(json.dumps({"raw": raw, "usage": usage, "model": model}, indent=2) + "\n")
    if request is not None:
        (run / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    (run / "usage.json").write_text(json.dumps(usage, indent=2) + "\n")
    known = _known_ids(root)
    accepted: list[dict] = []
    rejected: list[dict] = []
    try:
        concepts = extract_concepts(raw)
    except FactoryError as exc:
        concepts = []
        rejected.append({"reason": exc.reason, "response_sha256": response_hash})
    for concept in concepts:
        stored, reason = validate_concept(concept)
        if stored is None:
            rejected.append({"reason": reason, "concept": concept.get("concept"), "row": concept, "response_sha256": response_hash})
            continue
        if stored["id"] in known:
            rejected.append({"reason": "duplicate", "id": stored["id"], "response_sha256": response_hash})
            continue
        stored.update(
            {
                "prompt": prompt_name,
                "prompt_sha256": _sha(prompt_text),
                "model": model,
                "response_sha256": response_hash,
                "run": run.name,
            }
        )
        known.add(stored["id"])
        accepted.append(stored)
        _append(root / "data" / "factory" / "templates" / f"{stored['domain']}.jsonl", stored)
    for item in rejected:
        item.setdefault("response_sha256", response_hash)
        item["prompt"] = prompt_name
        item["run"] = run.name
        _append(root / "data" / "factory" / "rejected" / "concepts.jsonl", item)
        _append(run / "rejected.jsonl", item)
    for item in accepted:
        _append(run / "accepted.jsonl", item)
    manifest = {
        "run": run.name,
        "model": model,
        "prompt": prompt_name,
        "prompt_sha256": _sha(prompt_text),
        "response_sha256": response_hash,
        "parser_version": PARSER_VERSION,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "reject_reasons": [item.get("reason") for item in rejected],
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_bank_manifest(root)
    return manifest


def reprocess_run(root: Path, run_name: str) -> dict:
    """Parse a stored response again. The raw file stays where it was."""
    run = root / "data" / "factory" / "runs" / run_name
    payload = json.loads((run / "response.json").read_text())
    request = json.loads((run / "request.json").read_text()) if (run / "request.json").exists() else {}
    raw = payload.get("raw") or ""
    known = _known_ids(root)
    accepted: list[dict] = []
    rejected: list[dict] = []
    try:
        concepts = extract_concepts(raw)
    except FactoryError as exc:
        concepts = []
        rejected.append({"reason": exc.reason})
    prompt_name = request.get("prompt_file", "")
    prompt_text = request.get("prompt") or ""
    model = payload.get("model") or request.get("model") or TEACHER_MODEL
    response_hash = _sha(raw)
    for concept in concepts:
        stored, reason = validate_concept(concept)
        if stored is None:
            rejected.append({"reason": reason, "concept": concept.get("concept"), "row": concept})
            continue
        if stored["id"] in known:
            continue
        stored.update(
            {
                "prompt": Path(prompt_name).name if prompt_name else "",
                "prompt_sha256": _sha(prompt_text) if prompt_text else "",
                "model": model,
                "response_sha256": response_hash,
                "run": run.name,
                "reprocessed": True,
            }
        )
        known.add(stored["id"])
        accepted.append(stored)
        _append(root / "data" / "factory" / "templates" / f"{stored['domain']}.jsonl", stored)
        _append(run / "accepted.jsonl", stored)
    for item in rejected:
        item["response_sha256"] = response_hash
        item["run"] = run.name
        item["reprocessed"] = True
        _append(root / "data" / "factory" / "rejected" / "concepts.jsonl", item)
        _append(run / "rejected.jsonl", item)
    summary = {
        "run": run.name,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "parser_version": PARSER_VERSION,
        "reject_reasons": [item.get("reason") for item in rejected],
    }
    (run / "reprocess.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_bank_manifest(root)
    return summary


def load_templates(root: Path) -> list[dict]:
    bank = root / "data" / "factory" / "templates"
    records = []
    if not bank.exists():
        return records
    for path in sorted(bank.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_bank_manifest(root: Path) -> dict:
    records = load_templates(root)
    blob = json.dumps(records, sort_keys=True)
    payload = {
        "dataset": "needle-factory-v1",
        "parser_version": PARSER_VERSION,
        "generator_version": GENERATOR_VERSION,
        "teacher_model": TEACHER_MODEL,
        "teacher_note": "gpt-6-terra is not on this key; calls use gpt-5.6-terra.",
        "template_count": len(records),
        "template_bank_hash": _sha(blob),
        "seed": 2507,
        "num_tasks": 0,
        "diversity": diversity(records),
    }
    path = root / "data" / "factory" / "manifests" / "factory_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _write_split(root: Path, records: list[dict]) -> dict:
    payload = assign_splits(records)
    payload["generator_version"] = GENERATOR_VERSION
    payload["template_count"] = len(records)
    path = root / "data" / "factory" / "manifests" / "split_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _run_rows(root: Path) -> list[dict]:
    base = root / "data" / "factory" / "runs"
    rows = []
    if not base.exists():
        return rows
    for path in sorted(base.glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        request_path = path.parent / "request.json"
        domain = ""
        if request_path.exists():
            domain = str(json.loads(request_path.read_text()).get("domain") or "")
        accepted = manifest.get("accepted", 0)
        rejected_count = manifest.get("rejected", 0)
        reprocess_path = path.parent / "reprocess.json"
        if reprocess_path.exists():
            overlay = json.loads(reprocess_path.read_text())
            accepted = overlay.get("accepted", accepted)
            rejected_count = overlay.get("rejected", rejected_count)
        rows.append(
            {
                "run": manifest.get("run") or path.parent.name,
                "prompt": manifest.get("prompt") or "",
                "domain": domain,
                "accepted": accepted,
                "rejected": rejected_count,
            }
        )
    return rows


def render_report(root: Path) -> str:
    records = load_templates(root)
    report = diversity(records)
    manifest = _write_bank_manifest(root)
    lines = [
        "## Concept factory",
        "",
        "Terra proposes a field and a program. The raw response is stored before parsing. A concept enters the bank only when Needle's executor accepts every positive and rejects every near miss. Rejected concepts stay in `data/factory/rejected/`. Training shards are not committed; the manifest names the bank hash and the generator version that would rebuild them.",
        "",
        f"Parser `{PARSER_VERSION}`. Teacher model `{TEACHER_MODEL}`. "
        "gpt-6-terra is not on this key. "
        f"Templates {report['templates']}. Unique latent programs {report['unique_programs']}. "
        f"AST shapes {report['unique_ast_shapes']}. Concepts {report['concepts']}. "
        f"Teacher domains {report['domains']}. Semantic families {report['semantic_families']}. "
        f"Bank hash `{manifest['template_bank_hash'][:16]}`.",
        "",
        f"Program entropy {report['program_entropy']:.3f} bits. "
        f"AST-signature entropy {report['ast_signature_entropy']:.3f} bits. "
        f"Semantic-family entropy {report['semantic_entropy']:.3f} bits. "
        f"Operator-combination entropy {report['signature_entropy']:.3f} bits. "
        f"Domain entropy {report['domain_entropy']:.3f} bits. "
        f"Top-1 program share {report['top1_program_share']:.3f}. "
        f"Top-10 operator-combination share {report['top10_signature_share']:.3f}. "
        f"Top-10 semantic-family share {report['top10_family_share']:.3f}. "
        f"Mean positives per language {report['mean_positives']:.2f}. "
        f"Witnessed ambiguity {report['witnessed_ambiguity']}. "
        f"Novel-composition share {report['novel_composition_share']:.3f} "
        f"({report['novel_compositions']} templates whose AST shape is outside the procedural sampler).",
        "",
    ]
    split = _write_split(root, records)
    counts = split["counts"]
    lines.append(
        f"Concept-level split, seed 2507: train {counts['train']}, "
        f"validation {counts['validation']}, held out {counts['heldout']}. "
        "A family stays on one side. The evidence scorer and the expansion prior are not retrained on this bank."
    )
    held_families = sorted(key for key, name in split["families"].items() if name == "heldout")
    if held_families:
        shown = ", ".join(f"`{name}`" for name in held_families[:12])
        extra = len(held_families) - 12
        if extra > 0:
            shown += f", and {extra} more"
        lines.append("")
        lines.append(f"Held-out families: {shown}.")
    lines.append("")
    runs = _run_rows(root)
    if runs:
        banked_runs: dict[str, int] = {}
        for record in records:
            name = record.get("run") or ""
            banked_runs[name] = banked_runs.get(name, 0) + 1
        lines.append("| Run | Prompt | Prompt domain | Accepted | Rejected |")
        lines.append("| --- | --- | --- | ---: | ---: |")
        for row in runs:
            accepted = banked_runs.get(row["run"], row["accepted"])
            lines.append(
                f"| {row['run']} | {row['prompt']} | {row['domain']} | {accepted} | {row['rejected']} |"
            )
        lines.append("")
    ledger = _ledger(root)
    if ledger.get("calls"):
        lines.append(
            f"Calls {ledger['calls']}. Spent ${ledger['spent_usd']:.4f} of the ${BUDGET_USD:.0f} factory cap."
        )
        lines.append("")
    if report["domain_counts"]:
        lines.append("| Domain | Templates |")
        lines.append("| --- | ---: |")
        for name, count in report["domain_counts"].items():
            lines.append(f"| {name} | {count} |")
        lines.append("")
    rejected = root / "data" / "factory" / "rejected" / "concepts.jsonl"
    reasons: dict[str, int] = {}
    buckets: dict[str, int] = {}
    details: dict[tuple[str, str], int] = {}
    if rejected.exists():
        banked = {(record.get("run"), record.get("concept")) for record in records}
        latest_items: dict[tuple, dict] = {}
        anonymous_items: list[dict] = []
        for line in rejected.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            concept = item.get("concept")
            if concept:
                latest_items[(item.get("run"), concept)] = item
            else:
                anonymous_items.append(item)
        kept = [item for key, item in latest_items.items() if key not in banked]
        kept.extend(anonymous_items)
        for item in kept:
            view = classify_rejection(item)
            buckets[view["bucket_label"]] = buckets.get(view["bucket_label"], 0) + 1
            pair = (view["bucket_label"], view["detail"])
            details[pair] = details.get(pair, 0) + 1
            reasons[view["reason"]] = reasons.get(view["reason"], 0) + 1
    if buckets:
        lines.append("| Rejection bucket | Count |")
        lines.append("| --- | ---: |")
        for name, count in sorted(buckets.items(), key=lambda item: -item[1]):
            lines.append(f"| {name} | {count} |")
        lines.append("")
        lines.append("| Bucket | Detail | Count |")
        lines.append("| --- | --- | ---: |")
        for (bucket, detail), count in sorted(details.items(), key=lambda item: -item[1]):
            lines.append(f"| {bucket} | {detail} | {count} |")
        lines.append("")
    if reasons:
        lines.append("| Reject reason | Count |")
        lines.append("| --- | ---: |")
        for name, count in sorted(reasons.items(), key=lambda item: -item[1]):
            lines.append(f"| {name} | {count} |")
        lines.append("")
    lines.append("BENCH-E stays sealed. `needle()` is unchanged.")
    lines.append("")
    return "\n".join(lines)


def _ledger(root: Path) -> dict:
    path = root / "data" / "factory" / "ledger.json"
    if not path.exists():
        return {"spent_usd": 0.0, "calls": 0, "budget_usd": BUDGET_USD, "events": []}
    return json.loads(path.read_text())


def _save_ledger(root: Path, ledger: dict) -> None:
    path = root / "data" / "factory" / "ledger.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2) + "\n")


def generate(root: Path, prompt_name: str, domain: str = "", max_output_tokens: int = 5000) -> dict:
    """One Terra call. The response is stored even when every concept is rejected."""
    from pattern_induction.frontier import response_text
    from pattern_induction.teacher import call_teacher, estimate_cost

    prompt_path = root / "data" / "factory" / "prompts" / prompt_name
    template = prompt_path.read_text()
    prompt = template.replace("{domain}", domain or "logistics")
    model = TEACHER_MODEL
    rate = PRICE_PER_MILLION[model][1]
    ledger = _ledger(root)
    reservation = max_output_tokens / 1_000_000 * rate + 0.05
    if ledger["spent_usd"] + reservation > BUDGET_USD:
        raise RuntimeError(f"factory budget would pass ${BUDGET_USD:.2f}")
    started = time.time()
    payload = call_teacher(model, prompt, max_output_tokens)
    raw = response_text(payload)
    usage = payload.get("usage") or {}
    cost = estimate_cost(model, usage)
    request = {
        "model": model,
        "prompt_file": str(prompt_path.relative_to(root)),
        "prompt_sha256": _sha(template),
        "prompt": prompt,
        "domain": domain,
        "max_output_tokens": max_output_tokens,
        "parser_version": PARSER_VERSION,
    }
    manifest = ingest_text(
        root,
        raw,
        prompt_name=prompt_name,
        prompt_text=template,
        model=payload.get("model") or model,
        usage=usage,
        request=request,
    )
    ledger = _ledger(root)
    ledger["spent_usd"] = round(ledger["spent_usd"] + cost, 6)
    ledger["budget_usd"] = BUDGET_USD
    ledger["calls"] = ledger["calls"] + 1
    ledger["events"].append(
        {"run": manifest["run"], "model": model, "cost_usd": round(cost, 6), "usage": usage, "accepted": manifest["accepted"]}
    )
    _save_ledger(root, ledger)
    manifest["cost_usd"] = round(cost, 6)
    manifest["spent_usd"] = ledger["spent_usd"]
    manifest["seconds"] = round(time.time() - started, 3)
    return manifest


def _completed_pairs(root: Path) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    base = root / "data" / "factory" / "runs"
    if not base.exists():
        return found
    for path in base.glob("*/request.json"):
        request = json.loads(path.read_text())
        prompt = Path(str(request.get("prompt_file") or "")).name
        domain = str(request.get("domain") or "")
        if prompt and domain:
            found.add((prompt, domain))
    return found


def campaign_plan() -> list[tuple[str, str]]:
    """Prompt and domain pairs. Pairs already stored on disk are skipped by the runner."""
    plan: list[tuple[str, str]] = []
    for prompt in ("prior_decisions_v1.md", "real_world_concepts_v2.md", "ambiguity_v2.md"):
        for domain in CAMPAIGN_DOMAINS:
            plan.append((prompt, domain))
    already = {"software", "finance", "legal", "healthcare", "logistics"}
    for domain in CAMPAIGN_DOMAINS:
        if domain in already:
            continue
        plan.append(("domain_expansion_v1.md", domain))
    return plan


def campaign(root: Path, target: int = 400, max_output_tokens: int = 8000) -> dict:
    """Keep calling the teacher until the bank reaches ``target`` or the cap."""
    done = _completed_pairs(root)
    results = []
    for prompt, domain in campaign_plan():
        if (prompt, domain) in done:
            continue
        if len(load_templates(root)) >= target:
            break
        try:
            manifest = generate(root, prompt, domain, max_output_tokens)
        except RuntimeError as exc:
            results.append({"prompt": prompt, "domain": domain, "error": str(exc)})
            print(json.dumps(results[-1]), flush=True)
            if "budget" in str(exc):
                break
            continue
        results.append(
            {
                "prompt": prompt,
                "domain": domain,
                "run": manifest.get("run"),
                "accepted": manifest.get("accepted"),
                "rejected": manifest.get("rejected"),
                "cost_usd": manifest.get("cost_usd"),
            }
        )
        print(json.dumps(results[-1]), flush=True)
        page = root / "results" / "factory" / "bank.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(render_report(root))
    return {"templates": len(load_templates(root)), "calls": len(results), "results": results}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generate", "diversity", "preview", "reprocess", "campaign"), required=True)
    parser.add_argument("--run", default="")
    parser.add_argument("--prompt", default="real_world_concepts_v1.md")
    parser.add_argument("--domain", default="logistics")
    parser.add_argument("--max-output-tokens", type=int, default=5000)
    parser.add_argument("--target", type=int, default=400)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.stage == "reprocess":
        if not args.run:
            raise SystemExit("--run is required")
        print(json.dumps(reprocess_run(root, args.run), indent=2), flush=True)
        page = root / "results" / "factory" / "bank.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(render_report(root))
    elif args.stage == "generate":
        manifest = generate(root, args.prompt, args.domain, args.max_output_tokens)
        print(json.dumps(manifest, indent=2), flush=True)
        page = root / "results" / "factory" / "bank.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(render_report(root))
    elif args.stage == "campaign":
        summary = campaign(root, args.target, max(args.max_output_tokens, 8000))
        print(json.dumps({"templates": summary["templates"], "calls": summary["calls"]}), flush=True)
    elif args.stage == "diversity":
        page = root / "results" / "factory" / "bank.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(render_report(root))
        print(f"wrote {page}", flush=True)
    else:
        records = load_templates(root)
        preview = []
        for record in records[:3]:
            preview.extend(materialize(record, limit=args.limit))
        print(json.dumps(preview, indent=2))


if __name__ == "__main__":
    main()
