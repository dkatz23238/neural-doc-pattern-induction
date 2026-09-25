"""Symbolic baselines: memorization, heuristic, grex, and fAST."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pattern_induction.heuristic import infer_regex

GREX_CONFIGS = {
    "grex_default": [],
    "grex_digits": ["-d"],
    "grex_words": ["-w"],
    "grex_repetitions": ["-r"],
    "grex_combined": ["-d", "-w", "-r"],
}


def memorize_regex(strings: list[str]) -> str:
    body = "|".join(re.escape(s) for s in strings)
    return f"^(?:{body})$"


def run_heuristic(strings: list[str]) -> str:
    return infer_regex(strings)


def run_grex(strings: list[str], flags: list[str], timeout: float = 10.0) -> str:
    grex = os.environ.get("GREX_BIN", "grex")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as fh:
        path = fh.name
        for string in strings:
            fh.write(string.replace("\n", " ") + "\n")
    try:
        proc = subprocess.run(
            [grex, *flags, "--file", path],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        Path(path).unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"grex exit {proc.returncode}")
    pattern = proc.stdout.strip()
    if not pattern:
        raise RuntimeError("grex returned an empty pattern")
    return pattern


def _silence_fast_display() -> None:
    import sys

    import fast  # noqa: F401 — loads the submodule before the name is shadowed

    module = sys.modules["fast.fast"]
    module.ipynb_display_graph = lambda *args, **kwargs: None


def run_fast(strings: list[str], timeout: float = 8.0) -> str:
    """Run the published fAST implementation.

    The search prints progress and tries to render Graphviz previews when it
    accepts an AST. Both are disabled here. The call is killed at ``timeout``.
    """
    _silence_fast_display()
    from fast import fast

    start = time.perf_counter()

    def stop(results: list, elapsed: float) -> bool:
        return bool(results) or elapsed >= timeout or (time.perf_counter() - start) >= timeout

    stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        results = fast(list(strings), stop_condition=stop)
    finally:
        sys.stdout.close()
        sys.stdout = stdout
    if not results:
        raise TimeoutError(f"fAST produced no regex within {timeout:.1f}s")
    _score, ast = results[0]
    pattern = _fast_ast_to_python(ast)
    if not pattern.startswith("^"):
        pattern = "^" + pattern
    if not pattern.endswith("$"):
        pattern = pattern + "$"
    return pattern


def _fast_ast_to_python(ast, node: int | None = None) -> str:
    """Translate a fAST AST into a Python regex.

    fAST writes concatenation as ``.``. A leaf ``.`` is the any-character atom.
    """
    if node is None:
        node = ast.first_child(ast.root)
        if node is None:
            return ""
    label = ast.label(node)
    if ast.is_n_ary(node):
        parts = [_fast_ast_to_python(ast, child) for child in ast.children(node)]
        if label == ".":
            return "".join(parts)
        if label == "|":
            return "(?:" + "|".join(parts) + ")"
        return "(?:" + label.join(parts) + ")"
    if ast.is_unary(node):
        child = _fast_ast_to_python(ast, ast.first_child(node))
        if label in "+*?" and not (child.startswith("(?:") or len(child) == 1 or child.startswith("\\")):
            child = f"(?:{child})"
        return child + label
    if label == ".":
        return "."
    if label in set(r"^$*+?{}[]|()\\"):
        return "\\" + label
    return label


def infer(system: str, strings: list[str], fast_timeout: float = 8.0) -> str:
    if system == "memorize":
        return memorize_regex(strings)
    if system == "heuristic":
        return run_heuristic(strings)
    if system in GREX_CONFIGS:
        return run_grex(strings, GREX_CONFIGS[system])
    if system == "fast":
        return run_fast(strings, timeout=fast_timeout)
    raise KeyError(system)
