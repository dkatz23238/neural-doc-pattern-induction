"""Compact candidate features. No regex strings enter the ranker.

Numeric endpoints and lexicon identity are semantic. A within-class rename
of a letter prefix does not change these features. A rename of a digit does,
because the integer it spells is a different endpoint.
"""

from __future__ import annotations

import math
import re

from pattern_induction.charset import charset_spec, spec_chars
from pattern_induction.latent import _CONVENTIONS, _FAMILIES
from pattern_induction.lexicon import LEXICONS
from pattern_induction.program import Prog

NAMES: tuple[str, ...] = (
    "n_copy",
    "n_cls",
    "n_set",
    "n_int",
    "n_lex",
    "n_seq",
    "n_repeat",
    "n_opt",
    "n_alt",
    "nodes",
    "depth",
    "n_literal",
    "mean_lit_len",
    "max_lit_len",
    "cls_digit",
    "cls_upper",
    "cls_lower",
    "digit_lo",
    "digit_hi",
    "digit_exact",
    "digit_open",
    "int_present",
    "int_log_span",
    "int_exact_width",
    "int_covers",
    "int_hull",
    "int_convention",
    "int_round",
    "set_present",
    "set_family",
    "set_size",
    "set_covers",
    "set_hull",
    "set_exact_width",
    "set_open",
    "lex_present",
    "lex_currency",
    "lex_month",
    "lex_http",
    "lex_extension",
    "lex_mass",
    "lex_size",
    "open_width",
    "support_len_span",
    "support_all_digit",
    "support_all_alpha",
    "rel_round_cover",
    "rel_family_cover",
    "rel_lex_small",
    "rel_hull_varied",
)

FEATURE_DIM = len(NAMES)
INDEX = {name: pos for pos, name in enumerate(NAMES)}

_FAMILY_CHARS = {charset_spec(spec_chars(name)): spec_chars(name) for name in _FAMILIES}
_CONVENTION_SET = set(_CONVENTIONS)
_LEX_GROUP = {
    "iso4217": "lex_currency",
    "month_abbr": "lex_month",
    "month_abbr_upper": "lex_month",
    "month_abbr_lower": "lex_month",
    "month_full": "lex_month",
    "month_full_upper": "lex_month",
    "month_full_lower": "lex_month",
    "http_1_0": "lex_http",
    "http_iana": "lex_http",
    "file_extension": "lex_extension",
    "mass_unit": "lex_mass",
    "apparel_size": "lex_size",
}
# Touch the bank so a rename of LEXICONS fails this import loudly.
assert {item.name for item in LEXICONS} == set(_LEX_GROUP)


def _depth(prog: Prog) -> int:
    if not prog.children:
        return 1
    return 1 + max(_depth(child) for child in prog.children)


def _literal(prog: Prog, supports: list[str]) -> str:
    if prog.slot < 0 or prog.slot >= len(supports):
        return ""
    text = supports[prog.slot]
    if prog.start < 0 or prog.end > len(text) or prog.end <= prog.start:
        return ""
    return text[prog.start : prog.end]


def _round_interval(lo: int, hi: int) -> bool:
    if hi < lo:
        return False
    decade = lo % 10 == 0 and (hi % 10 == 9 or (hi + 1) % 10 == 0)
    hundred = lo % 100 == 0 and hi % 100 == 99
    power = lo == 0 and hi >= 1 and ((hi + 1) & hi) == 0
    return decade or hundred or power


def _support_view(supports: list[str]) -> dict[str, float]:
    lengths = [len(text) for text in supports] or [0]
    chars = {char for text in supports for char in text}
    numbers = [int(run) for text in supports for run in re.findall(r"\d+", text)]
    return {
        "len_min": float(min(lengths)),
        "len_max": float(max(lengths)),
        "len_span": float(max(lengths) - min(lengths)),
        "all_digit": float(bool(supports) and all(text.isdigit() for text in supports)),
        "all_alpha": float(bool(supports) and all(text.isalpha() for text in supports)),
        "chars": chars,
        "num_min": float(min(numbers)) if numbers else 0.0,
        "num_max": float(max(numbers)) if numbers else 0.0,
        "has_num": float(bool(numbers)),
    }


def candidate_values(prog: Prog, supports: list[str]) -> dict[str, float]:
    """One program, one support set, a fixed bag of structural numbers."""
    values = {name: 0.0 for name in NAMES}
    counts = {op: 0 for op in ("COPY", "CLS", "SET", "INT", "LEX", "SEQ", "REPEAT", "OPT", "ALT")}
    literals: list[str] = []
    ints: list[tuple[int, int, int, int]] = []
    sets: list[Prog] = []
    digits: list[Prog] = []
    lexes: list[str] = []
    opens = 0

    def walk(node: Prog) -> None:
        nonlocal opens
        counts[node.op] = counts.get(node.op, 0) + 1
        if node.op == "COPY":
            literals.append(_literal(node, supports))
        elif node.op == "INT" and ":" in node.spec:
            lo, hi = (int(part) for part in node.spec.split(":"))
            ints.append((lo, hi, node.lo, node.hi))
        elif node.op == "SET":
            sets.append(node)
        elif node.op == "CLS" and node.cls == "digit":
            digits.append(node)
        elif node.op == "LEX":
            lexes.append(node.spec)
        if node.op in {"CLS", "SET", "INT", "REPEAT"} and node.hi >= 64:
            opens += 1
        for child in node.children:
            walk(child)

    walk(prog)
    view = _support_view(supports)
    values["n_copy"] = counts["COPY"] / 4
    values["n_cls"] = counts["CLS"] / 4
    values["n_set"] = counts["SET"] / 4
    values["n_int"] = counts["INT"] / 4
    values["n_lex"] = counts["LEX"] / 4
    values["n_seq"] = counts["SEQ"] / 4
    values["n_repeat"] = counts["REPEAT"] / 4
    values["n_opt"] = counts["OPT"] / 4
    values["n_alt"] = counts["ALT"] / 4
    values["nodes"] = prog.nodes() / 16
    values["depth"] = _depth(prog) / 8
    values["n_literal"] = len(literals) / 4
    if literals:
        lengths = [len(text) for text in literals]
        values["mean_lit_len"] = (sum(lengths) / len(lengths)) / 8
        values["max_lit_len"] = max(lengths) / 16
    if digits:
        node = digits[0]
        values["cls_digit"] = 1.0
        values["digit_lo"] = node.lo / 8
        values["digit_hi"] = min(node.hi, 128) / 32
        values["digit_exact"] = float(node.lo == node.hi)
        values["digit_open"] = float(node.hi >= 64)
    values["cls_upper"] = float(any(node.op == "CLS" and node.cls == "upper" for node in _cls_nodes(prog)))
    values["cls_lower"] = float(any(node.op == "CLS" and node.cls == "lower" for node in _cls_nodes(prog)))
    if ints:
        lo, hi, width_lo, width_hi = min(ints, key=lambda item: item[1] - item[0])
        values["int_present"] = 1.0
        values["int_log_span"] = math.log10(max(hi - lo, 1)) / 4
        values["int_exact_width"] = float(width_lo == width_hi)
        covers = view["has_num"] and lo <= view["num_min"] and hi >= view["num_max"]
        hull = covers and lo == view["num_min"] and hi == view["num_max"]
        values["int_covers"] = float(covers)
        values["int_hull"] = float(hull)
        values["int_convention"] = float((lo, hi) in _CONVENTION_SET)
        values["int_round"] = float(_round_interval(lo, hi))
    if sets:
        node = max(sets, key=lambda item: len(spec_chars(item.spec)))
        chars = spec_chars(node.spec)
        values["set_present"] = 1.0
        values["set_family"] = float(charset_spec(chars) in _FAMILY_CHARS or node.spec in _FAMILY_CHARS)
        values["set_size"] = len(chars) / 64
        values["set_covers"] = float(bool(view["chars"]) and view["chars"] <= chars)
        values["set_hull"] = float(bool(view["chars"]) and chars == view["chars"])
        values["set_exact_width"] = float(node.lo == node.hi)
        values["set_open"] = float(node.hi >= 64)
    if lexes:
        values["lex_present"] = 1.0
        for name in lexes:
            group = _LEX_GROUP.get(name)
            if group:
                values[group] = 1.0
    values["open_width"] = float(opens > 0)
    values["support_len_span"] = view["len_span"] / 8
    values["support_all_digit"] = view["all_digit"]
    values["support_all_alpha"] = view["all_alpha"]
    values["rel_round_cover"] = values["int_round"] * values["int_covers"]
    values["rel_family_cover"] = values["set_family"] * values["set_covers"]
    values["rel_lex_small"] = values["lex_present"] * float(prog.nodes() <= 3)
    values["rel_hull_varied"] = values["int_hull"] * float(view["len_span"] > 0)
    return values


def _cls_nodes(prog: Prog) -> list[Prog]:
    found: list[Prog] = []

    def walk(node: Prog) -> None:
        if node.op == "CLS":
            found.append(node)
        for child in node.children:
            walk(child)

    walk(prog)
    return found


def candidate_features(prog: Prog, supports: list[str]) -> list[float]:
    values = candidate_values(prog, supports)
    return [values[name] for name in NAMES]


def symbolic_score(features: list[float]) -> float:
    """Fixed grammar score. Nothing here is fit to a split.

    A covering convention or family outranks the observed hull. Among those,
    a shorter interval or a smaller alphabet ranks higher. A lexicon outranks
    a character class. An open width is a penalty.
    """
    take = features.__getitem__
    convention = take(INDEX["int_convention"]) * take(INDEX["int_covers"])
    rounded = take(INDEX["int_round"]) * take(INDEX["int_covers"])
    family = take(INDEX["set_family"]) * take(INDEX["set_covers"])
    return (
        4.0 * convention
        + 4.0 * rounded
        - 0.4 * take(INDEX["int_log_span"]) * (convention + rounded)
        + 1.0 * take(INDEX["int_hull"])
        + 4.0 * family
        - 2.0 * take(INDEX["set_size"]) * take(INDEX["set_family"])
        + 1.0 * take(INDEX["set_hull"])
        + 5.0 * take(INDEX["lex_present"])
        - 2.0 * take(INDEX["open_width"])
        - 0.5 * take(INDEX["nodes"])
    )
