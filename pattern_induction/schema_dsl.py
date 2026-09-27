"""Executable schema language for compositional training data.

A teacher may propose a schema. This module decides whether the schema is
well formed, which strings it accepts, and which character spans the gold
derivation uses. Those decisions are the labels. A teacher string is never
trusted as a membership bit.

Repeat counts are capped at three so a later test of four and five repetitions
stays an extrapolation test. Schemas that reproduce the frozen eval languages
can be rejected when a recipe is supposed to hold those languages out.
"""

from __future__ import annotations

import hashlib
import json
import random
import string
from dataclasses import dataclass, field

from pattern_induction.compose import (
    KIND_CLS,
    KIND_DELIM,
    KIND_LIT,
    KIND_OPT,
    KIND_REPEAT,
    SEPS,
    AnnotatedEpisode,
    CharAnn,
    ignore_ann,
    paint,
)
from pattern_induction.membership import HELD_OUT

GENERATOR_VERSION = "dsl-1"
MAX_NODES = 14
MAX_DEPTH = 5
MAX_REPEAT = 3
MAX_TEXT = 28
CLASSES = ("digit", "upper", "lower", "punct")


@dataclass
class Node:
    kind: str
    ident: int
    children: list["Node"] = field(default_factory=list)
    cls: str = ""
    min_n: int = 1
    max_n: int = 1
    role: str = ""


@dataclass
class Binding:
    literals: dict[int, str]
    lengths: dict[int, int]


@dataclass
class CompiledSchema:
    root: Node
    schema: dict
    shape: tuple
    ident: str
    heldout: str | None
    confusable: list[dict] = field(default_factory=list)


class SchemaError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _shape(node: Node) -> tuple:
    if node.kind == "seq":
        return ("seq", tuple(_shape(child) for child in node.children))
    if node.kind == "class":
        return ("cls", node.cls)
    if node.kind == "literal_slot":
        return ("lit", node.cls)
    if node.kind == "repeat":
        return ("rep", _shape(node.children[0]))
    if node.kind == "optional":
        return ("opt", _shape(node.children[0]))
    return ("alt", _shape(node.children[0]), _shape(node.children[1]))


def _has(node: Node, kind: str) -> bool:
    if node.kind == kind:
        return True
    return any(_has(child, kind) for child in node.children)


def _repeat_depth(node: Node, seen: int = 0) -> int:
    here = seen + (1 if node.kind == "repeat" else 0)
    child_depths = [_repeat_depth(child, here) for child in node.children]
    return max([here, *child_depths])


def heldout_family(root: Node) -> str | None:
    """Name the frozen eval language this schema would train, if it matches one."""
    if root.kind != "seq" or len(root.children) < 2:
        return None
    first, *rest = root.children
    last = root.children[-1]

    def digit_repeat(node: Node) -> bool:
        return node.kind == "repeat" and _has(node, "class")

    if first.kind == "literal_slot" and first.cls in {"upper", "lower"} and any(digit_repeat(child) for child in rest):
        if not _has(root, "alt") and not _has(root, "optional"):
            return "prefix_repeat"
    if last.kind == "optional" and any(digit_repeat(child) for child in root.children[:-1]):
        return "repeat_optional"
    if first.kind == "literal_slot" and _has(root, "alt") and not _has(root, "repeat"):
        return "prefix_alt"
    if (
        len(root.children) == 3
        and root.children[1].kind == "optional"
        and root.children[0].kind in {"literal_slot", "class"}
        and root.children[2].kind in {"literal_slot", "class"}
        and not _has(root, "repeat")
        and not _has(root, "alt")
    ):
        return "optional_internal"
    return None


def compile_schema(payload: dict, allow_heldout: bool = False) -> CompiledSchema:
    if not isinstance(payload, dict):
        raise SchemaError("schema_not_object")
    raw = payload.get("schema", payload)
    confusable = payload.get("confusable") or payload.get("confusable_structures") or []
    if isinstance(confusable, dict):
        confusable = [confusable]
    if not isinstance(confusable, list):
        raise SchemaError("confusable_not_list")
    counter = {"n": 0, "depth": 0}

    def take() -> int:
        counter["n"] += 1
        if counter["n"] > MAX_NODES:
            raise SchemaError("too_many_nodes")
        return counter["n"]

    def build(item: dict, depth: int) -> Node:
        if depth > MAX_DEPTH:
            raise SchemaError("too_deep")
        if not isinstance(item, dict) or "type" not in item:
            raise SchemaError("node_not_object")
        kind = item["type"]
        if kind == "seq":
            children = item.get("children")
            if not isinstance(children, list) or not 2 <= len(children) <= 4:
                raise SchemaError("bad_seq")
            return Node("seq", take(), [build(child, depth + 1) for child in children])
        if kind == "class":
            cls = item.get("class")
            lo, hi = _bounds(item, 1, 4)
            if cls not in {"digit", "upper", "lower"}:
                raise SchemaError("bad_class")
            return Node("class", take(), [], cls, lo, hi)
        if kind == "literal_slot":
            cls = item.get("class")
            lo, hi = _bounds(item, 1, 4)
            if cls not in CLASSES:
                raise SchemaError("bad_literal_class")
            if cls == "punct" and (lo, hi) != (1, 1):
                raise SchemaError("punct_width")
            return Node("literal_slot", take(), [], cls, lo, hi, str(item.get("role") or ""))
        if kind == "repeat":
            lo, hi = _bounds(item, 1, MAX_REPEAT)
            if hi > MAX_REPEAT or lo > hi:
                raise SchemaError("repeat_too_long")
            child = item.get("child")
            if not isinstance(child, dict):
                raise SchemaError("bad_repeat")
            node = Node("repeat", take(), [build(child, depth + 1)], "", lo, hi)
            if _repeat_depth(node) > 1:
                raise SchemaError("nested_repeat")
            return node
        if kind == "optional":
            child = item.get("child")
            if not isinstance(child, dict):
                raise SchemaError("bad_optional")
            return Node("optional", take(), [build(child, depth + 1)])
        if kind == "alt":
            children = item.get("children")
            if not isinstance(children, list) or len(children) != 2:
                raise SchemaError("bad_alt")
            return Node("alt", take(), [build(child, depth + 1) for child in children])
        raise SchemaError(f"unknown_type:{kind}")

    root = build(raw, 1)
    if root.kind not in {"seq", "alt", "repeat"}:
        raise SchemaError("root_too_small")
    family = heldout_family(root)
    if family and not allow_heldout:
        raise SchemaError(f"heldout:{family}")
    compiled_confusable = []
    for item in confusable:
        if isinstance(item, str):
            continue
        if not isinstance(item, dict):
            continue
        try:
            other = compile_schema({"schema": item.get("schema", item)}, allow_heldout=allow_heldout)
        except SchemaError:
            continue
        compiled_confusable.append(other.schema)
    ident = payload.get("id") or hashlib.sha256(_canon(raw).encode()).hexdigest()[:12]
    return CompiledSchema(root, raw, _shape(root), ident, family, compiled_confusable)


def _bounds(item: dict, default_lo: int, default_hi: int) -> tuple[int, int]:
    lo = item.get("min", default_lo)
    hi = item.get("max", default_hi)
    if isinstance(lo, bool) or isinstance(hi, bool) or not isinstance(lo, int) or not isinstance(hi, int):
        raise SchemaError("bad_bounds")
    if lo < 1 or hi < lo or hi > 4 and item.get("type") != "repeat":
        if item.get("type") == "repeat":
            if lo < 1 or hi < lo:
                raise SchemaError("bad_bounds")
            return lo, hi
        raise SchemaError("bad_bounds")
    return lo, hi


def _canon(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def bank_hash(records: list[dict]) -> str:
    blob = "\n".join(_canon(record) for record in records)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _class_ok(text: str, cls: str) -> bool:
    if cls == "digit":
        return text.isdigit()
    if cls == "upper":
        return text.isalpha() and text.isupper()
    if cls == "lower":
        return text.isalpha() and text.islower()
    return all(ch in SEPS for ch in text)


def _draw_class(rng: random.Random, cls: str, width: int) -> str:
    if cls == "digit":
        pool = string.digits
    elif cls == "upper":
        pool = string.ascii_uppercase
    else:
        pool = string.ascii_lowercase
    return "".join(rng.choice(pool) for _ in range(width))


class Language:
    """One instantiated schema. Membership is the matcher, not the sampler."""

    def __init__(self, compiled: CompiledSchema, rng: random.Random):
        self.compiled = compiled
        self.root = compiled.root
        self.binding = self._bind(rng)
        self.confusable = []
        for raw in compiled.confusable:
            try:
                other = compile_schema({"schema": raw}, allow_heldout=True)
            except SchemaError:
                continue
            self.confusable.append(Language(other, rng))

    def _bind(self, rng: random.Random) -> Binding:
        literals: dict[int, str] = {}
        lengths: dict[int, int] = {}
        used_punct: set[str] = set()

        def walk(node: Node) -> None:
            if node.kind == "class":
                lengths[node.ident] = rng.randint(node.min_n, node.max_n)
            elif node.kind == "literal_slot":
                if node.cls == "punct":
                    pool = [ch for ch in SEPS if ch not in used_punct] or list(SEPS)
                    choice = rng.choice(pool)
                    used_punct.add(choice)
                    literals[node.ident] = choice
                else:
                    width = rng.randint(node.min_n, node.max_n)
                    literals[node.ident] = _draw_class(rng, node.cls, width)
            for child in node.children:
                walk(child)

        walk(self.root)
        return Binding(literals, lengths)

    def accepts(self, text: str) -> bool:
        if not text or len(text) > MAX_TEXT or any(ch in HELD_OUT for ch in text):
            return False
        memo: dict[tuple[int, int], tuple[int, ...]] = {}
        return len(text) in self._ends(self.root, text, 0, memo)

    def _ends(self, node: Node, text: str, index: int, memo: dict[tuple[int, int], tuple[int, ...]]) -> tuple[int, ...]:
        key = (node.ident, index)
        cached = memo.get(key)
        if cached is not None:
            return cached
        if node.kind == "class":
            width = self.binding.lengths[node.ident]
            chunk = text[index : index + width]
            result = (index + width,) if len(chunk) == width and _class_ok(chunk, node.cls) else ()
        elif node.kind == "literal_slot":
            literal = self.binding.literals[node.ident]
            result = (index + len(literal),) if text.startswith(literal, index) else ()
        elif node.kind == "optional":
            result = (index,) + self._ends(node.children[0], text, index, memo)
        elif node.kind == "alt":
            result = self._ends(node.children[0], text, index, memo) + self._ends(node.children[1], text, index, memo)
        elif node.kind == "seq":
            positions = (index,)
            for child in node.children:
                nxt: list[int] = []
                for pos in positions:
                    nxt.extend(self._ends(child, text, pos, memo))
                positions = tuple(dict.fromkeys(nxt))
                if not positions:
                    break
            result = positions
        else:
            found: list[int] = []
            positions = (index,)
            for count in range(1, node.max_n + 1):
                nxt = []
                for pos in positions:
                    nxt.extend(self._ends(node.children[0], text, pos, memo))
                positions = tuple(dict.fromkeys(nxt))
                if count >= node.min_n:
                    found.extend(positions)
                if not positions:
                    break
            result = tuple(dict.fromkeys(found))
        memo[key] = result
        return result

    def derive(self, rng: random.Random, force: dict[int, int] | None = None) -> tuple[str, CharAnn]:
        forced = force or {}
        parts: list[tuple[str, int, tuple[int, int, int]]] = []

        def role(repeat_index: int, optional: bool, alt: bool) -> tuple[int, int, int]:
            return (1 if repeat_index > 0 else 0, 1 if optional else 0, 1 if alt else 0)

        def walk(node: Node, repeat_index: int, optional: bool, alt: bool) -> None:
            if node.kind == "class":
                kind = KIND_OPT if optional else (KIND_REPEAT if repeat_index > 0 else KIND_CLS)
                parts.append((_draw_class(rng, node.cls, self.binding.lengths[node.ident]), kind, role(repeat_index, optional, alt)))
                return
            if node.kind == "literal_slot":
                if optional:
                    kind = KIND_OPT
                elif node.cls == "punct":
                    kind = KIND_DELIM
                elif repeat_index > 0:
                    kind = KIND_REPEAT
                else:
                    kind = KIND_LIT
                parts.append((self.binding.literals[node.ident], kind, role(repeat_index, optional, alt)))
                return
            if node.kind == "optional":
                present = forced[node.ident] if node.ident in forced else rng.randrange(2)
                if present:
                    walk(node.children[0], repeat_index, True, alt)
                return
            if node.kind == "alt":
                branch = forced[node.ident] if node.ident in forced else rng.randrange(2)
                walk(node.children[branch], repeat_index, optional, True)
                return
            if node.kind == "seq":
                for child in node.children:
                    walk(child, repeat_index, optional, alt)
                return
            count = forced[node.ident] if node.ident in forced else rng.randint(node.min_n, node.max_n)
            for cursor in range(count):
                walk(node.children[0], cursor, optional, alt)

        walk(self.root, 0, False, False)
        groups = [([(ch, kind) for ch in text], triple) for text, kind, triple in parts if text]
        if not groups or len("".join(text for text, _kind, _role in parts)) > MAX_TEXT:
            raise SchemaError("sample_bounds")
        text, ann = paint(groups)
        if any(ch in HELD_OUT for ch in text):
            raise SchemaError("heldout_symbol")
        return text, ann

    def sample(self, rng: random.Random, force: dict[int, int] | None = None) -> tuple[str, CharAnn]:
        last_error = "sample_bounds"
        for _ in range(12):
            try:
                text, ann = self.derive(rng, force)
            except SchemaError as exc:
                last_error = exc.reason
                continue
            if self.accepts(text) and len(ann) == len(text):
                return text, ann
        raise SchemaError(last_error)

    def plans(self) -> list[dict[int, int]]:
        alts = []
        optionals = []
        repeats = []

        def walk(node: Node) -> None:
            if node.kind == "alt":
                alts.append(node)
            elif node.kind == "optional":
                optionals.append(node)
            elif node.kind == "repeat" and node.max_n > node.min_n:
                repeats.append(node)
            for child in node.children:
                walk(child)

        walk(self.root)
        plans: list[dict[int, int]] = [{}]
        if alts:
            node = alts[0]
            plans = [{node.ident: 0}, {node.ident: 1}]
        if optionals:
            node = optionals[0]
            plans = [{**plan, node.ident: bit} for plan in plans for bit in (0, 1)]
        if repeats:
            node = repeats[0]
            base = list(plans)
            plans = [{**plan, node.ident: node.min_n} for plan in base]
            plans.append({**base[0], node.ident: node.max_n})
        return plans[:4] or [{}]

    def support(self, rng: random.Random) -> tuple[list[str], list[CharAnn]]:
        texts: list[str] = []
        anns: list[CharAnn] = []
        seen: set[str] = set()
        for plan in self.plans():
            if len(texts) >= 4:
                break
            try:
                text, ann = self.sample(rng, plan)
            except SchemaError:
                continue
            if text in seen:
                continue
            seen.add(text)
            texts.append(text)
            anns.append(ann)
        guard = 0
        while len(texts) < 3 and guard < 20:
            guard += 1
            try:
                text, ann = self.sample(rng, None)
            except SchemaError:
                continue
            if text in seen:
                continue
            seen.add(text)
            texts.append(text)
            anns.append(ann)
        if len(texts) < 3:
            raise SchemaError("thin_support")
        return texts[:4], anns[:4]

    def positive(self, rng: random.Random, banned: set[str]) -> tuple[str, CharAnn]:
        for _ in range(16):
            text, ann = self.sample(rng, None)
            if text not in banned:
                return text, ann
        return self.sample(rng, None)

    def negative(self, rng: random.Random, banned: set[str]) -> str:
        for _ in range(24):
            if self.confusable and rng.randrange(2) == 0:
                try:
                    text, _ann = rng.choice(self.confusable).sample(rng, None)
                except SchemaError:
                    text = ""
            else:
                try:
                    base, _ann = self.sample(rng, None)
                except SchemaError:
                    base = "ABC12"
                text = _mutate(rng, base)
            if text and text not in banned and not self.accepts(text) and len(text) <= MAX_TEXT:
                if not any(ch in HELD_OUT for ch in text):
                    return text
        return "??"


def _mutate(rng: random.Random, text: str) -> str:
    if not text:
        return "Q"
    chars = list(text)
    mode = rng.randrange(4)
    if mode == 0:
        index = rng.randrange(len(chars))
        chars[index] = rng.choice("QZ" if chars[index].isdigit() else "9")
        return "".join(chars)
    if mode == 1 and len(chars) > 2:
        del chars[rng.randrange(len(chars))]
        return "".join(chars)
    if mode == 2:
        chars.insert(rng.randrange(len(chars) + 1), rng.choice(SEPS))
        return "".join(chars)
    return text + rng.choice("Q9")


def languages_differ(left: Language, right: Language, rng: random.Random) -> bool:
    left_only = right_only = 0
    for _ in range(6):
        try:
            text, _ann = left.sample(rng, None)
            if not right.accepts(text):
                left_only += 1
        except SchemaError:
            continue
        try:
            text, _ann = right.sample(rng, None)
            if not left.accepts(text):
                right_only += 1
        except SchemaError:
            continue
    return left_only > 0 and right_only > 0


def episode_from_language(lang: Language, rng: random.Random, label: int, family: str) -> AnnotatedEpisode:
    support, support_ann = lang.support(rng)
    banned = set(support)
    if label == 1:
        query, query_ann = lang.positive(rng, banned)
    else:
        query = lang.negative(rng, banned)
        query_ann = ignore_ann(query)
    if any(ch in HELD_OUT for text in support + [query] for ch in text):
        raise SchemaError("heldout_symbol")
    for text in support:
        if not lang.accepts(text):
            raise SchemaError("support_reject")
    if lang.accepts(query) != bool(label):
        raise SchemaError("label_disagrees")
    pattern = json.dumps({"id": lang.compiled.ident, "schema": lang.compiled.schema}, sort_keys=True)
    return AnnotatedEpisode(support, query, label, family, pattern, support_ann, query_ann)


def sample_schema_episode(compiled: CompiledSchema, rng: random.Random, label: int, family: str) -> AnnotatedEpisode:
    last = "sample_fail"
    for _ in range(8):
        try:
            lang = Language(compiled, rng)
            return episode_from_language(lang, rng, label, family)
        except SchemaError as exc:
            last = exc.reason
    raise SchemaError(last)
