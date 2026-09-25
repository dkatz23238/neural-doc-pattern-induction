"""BENCH-A controlled synthetic patterns with a structural train/test split.

Identical ASTs never cross splits. Test shapes are held out entirely.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass

from pattern_induction.sample_util import build_task
from pattern_induction.tasks import Task

LOWER = "abcdefghijklmnopqrstuvwxyz"
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
ALNUM = LOWER + UPPER + DIGITS
PUNCT = "-_./:#"


@dataclass(frozen=True)
class Node:
    op: str
    value: str = ""
    lo: int = 1
    hi: int = 1
    kids: tuple["Node", ...] = ()

    def shape(self) -> str:
        if self.kids:
            inner = ",".join(kid.shape() for kid in self.kids)
            if self.op == "REPEAT":
                return f"REPEAT({inner})"
            return f"{self.op}({inner})"
        if self.op == "LITERAL":
            return "LITERAL"
        if self.op == "REPEAT":
            return f"REPEAT({self.value})"
        return self.op

    def signature(self) -> str:
        if self.op == "LITERAL":
            return f"LIT:{self.value}"
        if self.op == "REPEAT":
            child = self.kids[0].signature() if self.kids else self.value
            return f"R{self.lo}-{self.hi}({child})"
        if self.op == "OPTIONAL":
            return f"OPT({self.kids[0].signature()})"
        if self.op in {"SEQ", "ALT"}:
            inner = ",".join(kid.signature() for kid in self.kids)
            return f"{self.op}({inner})"
        return self.op


def _split_for_shape(shape: str) -> str:
    bucket = int(hashlib.sha256(shape.encode()).hexdigest()[:8], 16) % 10
    if bucket >= 8:
        return "test"
    if bucket == 7:
        return "val"
    return "dev"


def _rand_literal(rng: random.Random) -> str:
    alphabet = rng.choice([UPPER, LOWER, DIGITS, PUNCT, "INV", "PO", "ID"])
    if alphabet in {"INV", "PO", "ID"}:
        return alphabet
    n = rng.randint(1, 3)
    return "".join(rng.choice(alphabet) for _ in range(n))


def _random_node(rng: random.Random, depth: int) -> Node:
    if depth <= 0 or rng.random() < 0.45:
        leaf = rng.choice(["LITERAL", "DIGIT", "LOWER", "UPPER", "ALPHA", "ALNUM", "WHITESPACE"])
        if leaf == "LITERAL":
            return Node("LITERAL", _rand_literal(rng))
        if rng.random() < 0.55:
            lo = rng.randint(1, 3)
            hi = lo if rng.random() < 0.6 else min(lo + rng.randint(1, 2), 5)
            return Node("REPEAT", lo=lo, hi=hi, kids=(Node(leaf),))
        return Node(leaf)
    op = rng.choice(["SEQ", "SEQ", "ALT", "OPTIONAL", "REPEAT"])
    if op == "SEQ":
        kids = tuple(_random_node(rng, depth - 1) for _ in range(rng.randint(2, 3)))
        return Node("SEQ", kids=kids)
    if op == "ALT":
        kids = tuple(_random_node(rng, depth - 1) for _ in range(2))
        return Node("ALT", kids=kids)
    if op == "OPTIONAL":
        return Node("OPTIONAL", kids=(_random_node(rng, depth - 1),))
    child = _random_node(rng, depth - 1)
    lo = 1
    hi = rng.randint(1, 3)
    return Node("REPEAT", lo=lo, hi=hi, kids=(child,))


def _compile(node: Node) -> str:
    if node.op == "LITERAL":
        return re.escape(node.value)
    if node.op == "DIGIT":
        return r"\d"
    if node.op == "LOWER":
        return r"[a-z]"
    if node.op == "UPPER":
        return r"[A-Z]"
    if node.op == "ALPHA":
        return r"[A-Za-z]"
    if node.op == "ALNUM":
        return r"[A-Za-z0-9]"
    if node.op == "WHITESPACE":
        return r" "
    if node.op == "ANY":
        return r"[A-Za-z0-9]"
    if node.op == "SEQ":
        return "".join(_compile(kid) for kid in node.kids)
    if node.op == "ALT":
        return "(?:" + "|".join(_compile(kid) for kid in node.kids) + ")"
    if node.op == "OPTIONAL":
        return f"(?:{_compile(node.kids[0])})?"
    if node.op == "REPEAT":
        inner = _compile(node.kids[0]) if node.kids else _compile(Node(node.value))
        if len(inner) > 1 and not inner.startswith("(?:") and not inner.startswith("["):
            inner = f"(?:{inner})"
        if node.lo == node.hi:
            return inner if node.lo == 1 else f"{inner}{{{node.lo}}}"
        return f"{inner}{{{node.lo},{node.hi}}}"
    raise ValueError(node.op)


def _sample(node: Node, rng: random.Random) -> str:
    if node.op == "LITERAL":
        return node.value
    if node.op == "DIGIT":
        return rng.choice(DIGITS)
    if node.op == "LOWER":
        return rng.choice(LOWER)
    if node.op == "UPPER":
        return rng.choice(UPPER)
    if node.op == "ALPHA":
        return rng.choice(LOWER + UPPER)
    if node.op == "ALNUM":
        return rng.choice(ALNUM)
    if node.op == "WHITESPACE":
        return " "
    if node.op == "ANY":
        return rng.choice(ALNUM)
    if node.op == "SEQ":
        return "".join(_sample(kid, rng) for kid in node.kids)
    if node.op == "ALT":
        return _sample(rng.choice(node.kids), rng)
    if node.op == "OPTIONAL":
        return _sample(node.kids[0], rng) if rng.random() < 0.5 else ""
    if node.op == "REPEAT":
        count = rng.randint(node.lo, node.hi)
        child = node.kids[0]
        return "".join(_sample(child, rng) for _ in range(count))
    raise ValueError(node.op)


def _too_broad(node: Node) -> bool:
    sig = node.signature()
    if len(sig) > 180:
        return True
    if sig.count("OPTIONAL") > 2 or sig.count("ALT") > 2:
        return True
    return False


def generate_bench_a(
    n_dev: int = 2000,
    n_val: int = 400,
    n_test: int = 400,
    seed: int = 7,
) -> list[Task]:
    """Generate a review-sized structural split.

    The full 100k/10k/10k corpus is the same function with larger counts.
    Shape buckets are fixed, so a larger draw keeps the same held-out shapes.
    """
    rng = random.Random(seed)
    targets = {"dev": n_dev, "val": n_val, "test": n_test}
    counts = {"dev": 0, "val": 0, "test": 0}
    seen_ast: set[str] = set()
    tasks: list[Task] = []
    attempts = 0
    limit = (n_dev + n_val + n_test) * 40
    while any(counts[key] < targets[key] for key in targets) and attempts < limit:
        attempts += 1
        node = _random_node(rng, depth=rng.choice((1, 2, 2, 3)))
        if _too_broad(node):
            continue
        signature = node.signature()
        if signature in seen_ast:
            continue
        split = _split_for_shape(node.shape())
        if counts[split] >= targets[split]:
            continue
        regex = "^" + _compile(node) + "$"
        try:
            compiled = re.compile(regex)
        except re.error:
            continue
        if compiled.fullmatch(""):
            continue
        distinct = set()
        for _ in range(30):
            probe = _sample(node, rng)
            if compiled.fullmatch(probe) and 1 <= len(probe) <= 40:
                distinct.add(probe)
        if len(distinct) < 4:
            continue
        seen_ast.add(signature)
        task = build_task(
            task_id=f"A-{split}-{counts[split]:05d}",
            benchmark="A",
            domain="synthetic",
            variant=node.shape(),
            gold_regex=regex,
            sample=lambda rng, node=node: _sample(node, rng),
            rng=rng,
            latent_schema=signature,
            n_visible=rng.choice((4, 6, 8)),
            n_pos=20,
            n_neg=20,
            split=split,
            notes="Controlled AST sample. ANY is restricted to alnum so hidden negatives stay meaningful.",
            meta={"shape": node.shape()},
        )
        tasks.append(task)
        counts[split] += 1
    if any(counts[key] < targets[key] for key in targets):
        raise RuntimeError(f"short generation {counts} target {targets}")
    return tasks
