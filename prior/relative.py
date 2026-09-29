"""Candidate-relative scores from deterministic sufficient statistics.

A repeat node sees its own bounds and the observed repeat counts. It does
not see whether the repeated field is an integer range or a width. An
integer node sees its own endpoints against the observed numbers. A charset
node sees its alphabet against the observed characters. The program score
is the sum of those local scores. There is no global embedding and no
cross-node relation.

The local target is which hypothesis matches a probe-perfect program, not
whether the fact is present.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from pattern_induction.charset import spec_chars
from pattern_induction.latent import _CLASS_CHARS
from pattern_induction.lexicon import matching_lexicons
from pattern_induction.program import OPEN, Prog, advance
from pattern_induction.renaming import PUNCT, structural_class
from prior.features import _literal
from prior.loss import ranking_loss
from prior.probe_labels import (
    _content_runs,
    _field_lengths,
    _optional_span,
    _punct_affix,
    _repeat_fields,
    lexicon_names,
    repeat_evidence,
)
from prior.tree import programs_from_json

HEADS = ("repeat", "integer", "charset", "optional", "literal", "lexicon")
HIDDEN = 16
MAX_NODES = 6
MAX_HYP = 32
REPEAT_DIM = 11
INTEGER_DIM = 14
CHARSET_DIM = 16
OPTIONAL_DIM = 8
LITERAL_DIM = 7
LEXICON_DIM = 4
DIMS = {
    "repeat": REPEAT_DIM,
    "integer": INTEGER_DIM,
    "charset": CHARSET_DIM,
    "optional": OPTIONAL_DIM,
    "literal": LITERAL_DIM,
    "lexicon": LEXICON_DIM,
}


def _signed_log(value: float) -> float:
    if value == 0:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * math.log10(abs(value) + 1.0) / 6.0


def _log_pos(value: float) -> float:
    return math.log10(value + 1.0) / 6.0


def _abstain(dim: int) -> list[float]:
    row = [0.0] * dim
    row[-1] = 1.0
    return row


def _kinds(chars: set[str]) -> tuple[float, float, float]:
    return (
        float(any(char.isdigit() for char in chars)),
        float(any(char.isupper() for char in chars)),
        float(any(char.islower() for char in chars)),
    )


class Observations:
    """Statistics of one support set. No candidate has been applied yet."""

    def __init__(self, supports: list[str]):
        self.supports = supports
        self.repeat_counts = _repeat_counts(supports)
        self.numbers, self.number_widths = _numbers(supports)
        self.chars = _observed_chars(supports)
        self.widths = [width for width in _field_lengths(supports) if width > 0]
        role, text = _optional_span(supports)
        self.optional_role = role
        self.optional_text = text
        self.optional_fraction = _optional_fraction(supports, role, text)
        punct_role, punct_text = _punct_affix(supports)
        self.const_prefix = text if role == "prefix" else (punct_text if punct_role == "prefix" else "")
        self.const_suffix = text if role == "suffix" else (punct_text if punct_role == "suffix" else "")
        self.lexicons = lexicon_names(supports)
        self.separator = _separator(supports)

    @property
    def has_repeat(self) -> bool:
        return len(self.repeat_counts) >= 2 and len(set(self.repeat_counts)) >= 2

    @property
    def repeat_min(self) -> int:
        return min(self.repeat_counts) if self.repeat_counts else 0

    @property
    def repeat_max(self) -> int:
        return max(self.repeat_counts) if self.repeat_counts else 0

    @property
    def number_min(self) -> int:
        return min(self.numbers) if self.numbers else 0

    @property
    def number_max(self) -> int:
        return max(self.numbers) if self.numbers else 0

    @property
    def number_width(self) -> int:
        if not self.number_widths:
            return 0
        return max(set(self.number_widths), key=self.number_widths.count)

    @property
    def field_width(self) -> int:
        if not self.widths:
            return 0
        return max(set(self.widths), key=self.widths.count)


def _repeat_counts(supports: list[str]) -> list[int]:
    """How many times the repeated unit fires. The head field is not one of them."""
    if not repeat_evidence(supports):
        return []
    counts = []
    for text in supports:
        fields = _repeat_fields(text)
        if not fields:
            return []
        counts.append(len(fields) - 1)
    return counts


def _numbers(supports: list[str]) -> tuple[list[int], list[int]]:
    values: list[int] = []
    widths: list[int] = []
    rows = [_repeat_fields(text) or [] for text in supports] if repeat_evidence(supports) else _content_runs(supports)
    for row in rows:
        for run in row:
            text = run[3]
            if text.isdigit():
                values.append(int(text))
                widths.append(len(text))
    return values, widths


def _observed_chars(supports: list[str]) -> set[str]:
    found: set[str] = set()
    rows = [_repeat_fields(text) or [] for text in supports] if repeat_evidence(supports) else _content_runs(supports)
    for row in rows:
        for run in row:
            if run[2] != PUNCT:
                found.update(run[3])
    return found


def _optional_fraction(supports: list[str], role: str, text: str) -> float:
    if role not in {"prefix", "suffix"} or not text or not supports:
        return 0.0
    hits = 0
    for item in supports:
        if role == "prefix" and item.startswith(text):
            hits += 1
        elif role == "suffix" and item.endswith(text):
            hits += 1
    return hits / float(len(supports))


def _separator(supports: list[str]) -> str:
    if not repeat_evidence(supports):
        return ""
    marks: list[str] = []
    for text in supports:
        fields = _repeat_fields(text) or []
        if len(fields) < 2:
            return ""
        marks.append(text[fields[0][1] : fields[1][0]])
    if len(set(marks)) != 1:
        return ""
    return marks[0]


def _walk(node: Prog):
    yield node
    for child in node.children:
        yield from _walk(child)


def _trace(node: Prog, text: str, supports: list[str], cursor: int, end: int) -> dict[int, dict] | None:
    """Spans of one full match of ``node`` on ``text[cursor:end]``.

    The repeat count is how many times this node fired, not a fact about its
    child. An integer or charset records only the characters it consumed.
    """
    if node.op in {"COPY", "CLS", "SET", "INT", "LEX"}:
        if end not in advance(node, text, supports, cursor):
            return None
        return {id(node): {"texts": [text[cursor:end]], "counts": []}}
    if node.op == "OPT" and node.children:
        if cursor == end:
            return {id(node): {"texts": [""], "counts": [0]}}
        child = _trace(node.children[0], text, supports, cursor, end)
        if child is None:
            return None
        child[id(node)] = {"texts": [text[cursor:end]], "counts": [1]}
        return child
    if node.op == "ALT" and len(node.children) >= 2:
        for child in node.children[:2]:
            found = _trace(child, text, supports, cursor, end)
            if found is not None:
                found[id(node)] = {"texts": [text[cursor:end]], "counts": []}
                return found
        return None
    if node.op == "SEQ" and node.children:
        found = _trace_seq(node.children, text, supports, cursor, end)
        if found is None:
            return None
        found[id(node)] = {"texts": [text[cursor:end]], "counts": []}
        return found
    if node.op == "REPEAT" and node.children:
        return _trace_repeat(node, text, supports, cursor, end)
    return None


def _trace_seq(children: list[Prog], text: str, supports: list[str], cursor: int, end: int) -> dict[int, dict] | None:
    if len(children) == 1:
        return _trace(children[0], text, supports, cursor, end)
    for mid in advance(children[0], text, supports, cursor):
        if mid > end:
            continue
        left = _trace(children[0], text, supports, cursor, mid)
        if left is None:
            continue
        right = _trace_seq(children[1:], text, supports, mid, end)
        if right is None:
            continue
        for key, value in right.items():
            slot = left.setdefault(key, {"texts": [], "counts": []})
            slot["texts"].extend(value["texts"])
            slot["counts"].extend(value["counts"])
        return left
    return None


def _trace_repeat(node: Prog, text: str, supports: list[str], cursor: int, end: int) -> dict[int, dict] | None:
    child = node.children[0]
    hi = max(node.lo, min(node.hi, OPEN))
    reached = [{cursor}]
    previous: list[dict[int, int]] = [{}]
    for _count in range(1, hi + 1):
        nxt: set[int] = set()
        back: dict[int, int] = {}
        for pos in reached[-1]:
            for stop in advance(child, text, supports, pos):
                if stop > end or stop <= pos or stop in nxt:
                    continue
                nxt.add(stop)
                back[stop] = pos
        reached.append(nxt)
        previous.append(back)
        if not nxt:
            break
    chosen = None
    for count in range(node.lo, hi + 1):
        if count < len(reached) and end in reached[count]:
            chosen = count
            break
    if chosen is None:
        return None
    positions = [end]
    pos = end
    for count in range(chosen, 0, -1):
        pos = previous[count][pos]
        positions.append(pos)
    positions.reverse()
    records: dict[int, dict] = {id(node): {"texts": [text[cursor:end]], "counts": [chosen]}}
    for start, stop in zip(positions, positions[1:]):
        child_rec = _trace(child, text, supports, start, stop)
        if child_rec is None:
            return None
        for key, value in child_rec.items():
            slot = records.setdefault(key, {"texts": [], "counts": []})
            slot["texts"].extend(value["texts"])
            slot["counts"].extend(value["counts"])
    return records


def _matched(prog: Prog, supports: list[str]) -> dict[int, dict] | None:
    """Per-node texts and firing counts, when every support has one full match."""
    merged: dict[int, dict] = {}
    for text in supports:
        if len(text) not in advance(prog, text, supports, 0):
            return None
        found = _trace(prog, text, supports, 0, len(text))
        if found is None:
            return None
        for key, value in found.items():
            slot = merged.setdefault(key, {"texts": [], "counts": []})
            slot["texts"].extend(value["texts"])
            slot["counts"].extend(value["counts"])
    return merged


class _View:
    """One node's statistics. Unmentioned fields stay the support-level values."""

    def __init__(self, base: Observations, repeat_counts=None, numbers=None, number_widths=None, chars=None, widths=None, lexicons=None):
        self.supports = base.supports
        self.repeat_counts = base.repeat_counts if repeat_counts is None else repeat_counts
        self.numbers = base.numbers if numbers is None else numbers
        self.number_widths = base.number_widths if number_widths is None else number_widths
        self.chars = base.chars if chars is None else chars
        self.widths = base.widths if widths is None else widths
        self.optional_role = base.optional_role
        self.optional_text = base.optional_text
        self.optional_fraction = base.optional_fraction
        self.const_prefix = base.const_prefix
        self.const_suffix = base.const_suffix
        self.lexicons = base.lexicons if lexicons is None else lexicons
        self.separator = base.separator

    @property
    def has_repeat(self) -> bool:
        return len(self.repeat_counts) >= 1

    @property
    def repeat_min(self) -> int:
        return min(self.repeat_counts) if self.repeat_counts else 0

    @property
    def repeat_max(self) -> int:
        return max(self.repeat_counts) if self.repeat_counts else 0

    @property
    def number_min(self) -> int:
        return min(self.numbers) if self.numbers else 0

    @property
    def number_max(self) -> int:
        return max(self.numbers) if self.numbers else 0

    @property
    def number_width(self) -> int:
        if not self.number_widths:
            return 0
        return max(set(self.number_widths), key=self.number_widths.count)

    @property
    def field_width(self) -> int:
        if not self.widths:
            return 0
        return max(set(self.widths), key=self.widths.count)


def _view_for(base: Observations, node: Prog, matched: dict[int, dict] | None) -> _View:
    if not matched or id(node) not in matched:
        return _View(base)
    info = matched[id(node)]
    if node.op == "REPEAT" and info["counts"]:
        return _View(base, repeat_counts=list(info["counts"]))
    if node.op == "INT":
        values: list[int] = []
        widths: list[int] = []
        for text in info["texts"]:
            if text.isdigit():
                values.append(int(text))
                widths.append(len(text))
        if values:
            return _View(base, numbers=values, number_widths=widths)
    if node.op in {"CLS", "SET"}:
        chars: set[str] = set()
        widths = []
        for text in info["texts"]:
            if not text:
                continue
            chars.update(char for char in text if structural_class(char) != PUNCT)
            widths.append(len(text))
        if chars:
            return _View(base, chars=chars, widths=widths)
    if node.op == "LEX":
        names = matching_lexicons({text for text in info["texts"] if text})
        if names:
            return _View(base, lexicons=names)
    return _View(base)


def _alphabet(node: Prog) -> set[str]:
    if node.op == "CLS":
        return set(_CLASS_CHARS.get(node.cls, ()))
    if node.op == "SET":
        return spec_chars(node.spec)
    return set()


def _parse_span(spec: str) -> tuple[int, int] | None:
    if ":" not in spec:
        return None
    left, right = spec.split(":", 1)
    if not left.lstrip("-").isdigit() or not right.lstrip("-").isdigit():
        return None
    return int(left), int(right)


def repeat_features(obs: Observations, node: Prog | None) -> list[float]:
    if node is None:
        return _abstain(REPEAT_DIM)
    has = float(obs.has_repeat)
    obs_lo = obs.repeat_min
    obs_hi = obs.repeat_max
    cand_lo = int(node.lo)
    cand_hi = int(node.hi)
    contains = float(has and cand_lo <= obs_lo and cand_hi >= obs_hi)
    if has and contains:
        slack = ((obs_lo - cand_lo) + (cand_hi - obs_hi)) / 8.0
    elif has:
        slack = -((max(0, cand_lo - obs_lo) + max(0, obs_hi - cand_hi)) / 8.0)
    else:
        slack = 0.0
    return [
        has,
        obs_lo / 8.0,
        obs_hi / 8.0,
        cand_lo / 8.0,
        min(cand_hi, OPEN) / 8.0,
        float(cand_hi >= OPEN),
        float(has and cand_lo == obs_lo),
        float(has and cand_hi == obs_hi),
        contains,
        slack,
        0.0,
    ]


def integer_features(obs: Observations, node: Prog | None) -> list[float]:
    if node is None:
        return _abstain(INTEGER_DIM)
    span = _parse_span(node.spec)
    has = float(bool(obs.numbers) and span is not None)
    obs_lo, obs_hi = obs.number_min, obs.number_max
    cand_lo, cand_hi = span if span is not None else (0, 0)
    contains = float(has and cand_lo <= obs_lo and cand_hi >= obs_hi)
    obs_span = max(obs_hi - obs_lo + 1, 1)
    cand_span = max(cand_hi - cand_lo + 1, 1)
    width = obs.number_width
    return [
        has,
        _log_pos(float(obs_lo)),
        _log_pos(float(obs_hi)),
        _log_pos(float(cand_lo)),
        _log_pos(float(cand_hi)),
        _signed_log(float(obs_lo - cand_lo)),
        _signed_log(float(cand_hi - obs_hi)),
        contains,
        float(has and cand_lo == obs_lo and cand_hi == obs_hi),
        math.log10(cand_span / float(obs_span)) / 4.0 if has else 0.0,
        float(has and node.lo == node.hi == width and width > 0),
        float(has and node.lo <= width <= node.hi and width > 0),
        float(has and cand_lo in obs.numbers and cand_hi in obs.numbers),
        0.0,
    ]


def charset_features(obs: Observations, node: Prog | None) -> list[float]:
    if node is None:
        return _abstain(CHARSET_DIM)
    alphabet = _alphabet(node)
    observed = obs.chars
    has = float(bool(observed) and bool(alphabet))
    covered = observed & alphabet
    contained = float(has and observed <= alphabet)
    coverage = len(covered) / float(len(observed)) if observed else 0.0
    unseen = (len(alphabet - observed) / float(len(alphabet))) if alphabet else 0.0
    obs_kinds = _kinds(observed)
    cand_kinds = _kinds(alphabet)
    width = obs.field_width
    return [
        has,
        contained,
        coverage,
        unseen,
        obs_kinds[0],
        obs_kinds[1],
        obs_kinds[2],
        cand_kinds[0],
        cand_kinds[1],
        cand_kinds[2],
        float(width > 0 and node.lo == node.hi == width),
        float(node.hi >= 64),
        float(obs_kinds == cand_kinds and has),
        float(width > 0 and node.lo <= width <= min(node.hi, 128)),
        len(alphabet) / 62.0,
        0.0,
    ]


def _optional_literal(node: Prog, supports: list[str]) -> str:
    texts = [_literal(child, supports) for child in _walk(node) if child.op == "COPY"]
    texts = [text for text in texts if text]
    return texts[0] if texts else ""


def optional_features(obs: Observations, node: Prog | None) -> list[float]:
    if node is None:
        return _abstain(OPTIONAL_DIM)
    has = float(obs.optional_role in {"prefix", "suffix"})
    literal = _optional_literal(node, obs.supports)
    return [
        has,
        float(obs.optional_role == "prefix"),
        float(obs.optional_role == "suffix"),
        obs.optional_fraction,
        float(bool(obs.optional_text) and literal == obs.optional_text),
        len(literal) / 8.0,
        float(0.0 < obs.optional_fraction < 1.0),
        0.0,
    ]


def literal_features(obs: Observations, node: Prog | None) -> list[float]:
    if node is None:
        return _abstain(LITERAL_DIM)
    text = _literal(node, obs.supports)
    punct = bool(text) and all(structural_class(char) == PUNCT for char in text)
    return [
        len(text) / 8.0,
        float(bool(text) and text == obs.const_prefix),
        float(bool(text) and text == obs.const_suffix),
        float(punct),
        float(bool(text) and text == obs.optional_text),
        float(bool(text) and text == obs.separator),
        0.0,
    ]


def lexicon_features(obs: Observations, node: Prog | None) -> list[float]:
    if node is None:
        return _abstain(LEXICON_DIM)
    return [
        float(bool(obs.lexicons)),
        float(bool(node.spec) and node.spec in obs.lexicons),
        min(len(obs.lexicons), 4) / 4.0,
        0.0,
    ]


_FEATURES = {
    "repeat": (repeat_features, "REPEAT"),
    "integer": (integer_features, "INT"),
    "charset": (charset_features, ("CLS", "SET")),
    "optional": (optional_features, "OPT"),
    "literal": (literal_features, "COPY"),
    "lexicon": (lexicon_features, "LEX"),
}


def _nodes_of(prog: Prog, ops: str | tuple[str, ...]) -> list[Prog]:
    names = (ops,) if isinstance(ops, str) else ops
    return [node for node in _walk(prog) if node.op in names]


def _key(values: list[float]) -> tuple[float, ...]:
    return tuple(round(value, 5) for value in values)


def pack_example(programs: list[Prog], gold: list[bool], mask: list[bool], supports: list[str]) -> dict[str, torch.Tensor]:
    """Node features for every candidate, and one row per distinct hypothesis."""
    obs = Observations(supports)
    matched_cache: dict[int, dict | None] = {}
    packed: dict[str, torch.Tensor] = {}
    for name, (function, ops) in _FEATURES.items():
        dim = DIMS[name]
        nodes = torch.zeros(32, MAX_NODES, dim)
        node_mask = torch.zeros(32, MAX_NODES)
        hypotheses: list[list[float]] = [_abstain(dim)]
        gold_has = any(
            index < len(gold)
            and bool(gold[index])
            and index < len(mask)
            and bool(mask[index])
            and bool(_nodes_of(prog, ops))
            for index, prog in enumerate(programs[:32])
        )
        hyp_gold = [not gold_has]
        seen = {_key(hypotheses[0]): 0}
        for index, prog in enumerate(programs[:32]):
            if index >= len(mask) or not mask[index]:
                continue
            chosen = _nodes_of(prog, ops)[:MAX_NODES]
            alive = bool(gold[index]) if index < len(gold) else False
            if chosen and id(prog) not in matched_cache:
                matched_cache[id(prog)] = _matched(prog, supports)
            matched = matched_cache.get(id(prog))
            for slot, node in enumerate(chosen):
                values = function(_view_for(obs, node, matched), node)
                nodes[index, slot] = torch.tensor(values)
                node_mask[index, slot] = 1.0
                key = _key(values)
                if key not in seen and len(hypotheses) < MAX_HYP:
                    seen[key] = len(hypotheses)
                    hypotheses.append(values)
                    hyp_gold.append(False)
                if key in seen and alive:
                    hyp_gold[seen[key]] = True
        if any(hyp_gold[1:]):
            hyp_gold[0] = False
        if not any(hyp_gold):
            hyp_gold[0] = True
        hyp = torch.zeros(MAX_HYP, dim)
        hyp_mask = torch.zeros(MAX_HYP)
        gold_row = torch.zeros(MAX_HYP)
        for slot, values in enumerate(hypotheses):
            hyp[slot] = torch.tensor(values)
            hyp_mask[slot] = 1.0
            gold_row[slot] = float(hyp_gold[slot])
        packed[f"{name}_feat"] = nodes
        packed[f"{name}_mask"] = node_mask
        packed[f"{name}_hyp"] = hyp
        packed[f"{name}_hyp_mask"] = hyp_mask
        packed[f"{name}_hyp_gold"] = gold_row
    return packed


class LocalScorer(nn.Module):
    """One small network per node type. The program score is their sum."""

    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(DIMS[name], hidden), nn.GELU(), nn.Linear(hidden, 1))
                for name in HEADS
            }
        )

    def program_scores(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        total = torch.zeros(batch["repeat_feat"].shape[:2], device=batch["repeat_feat"].device)
        for name, head in self.heads.items():
            raw = head(batch[f"{name}_feat"]).squeeze(-1)
            total = total + (raw * batch[f"{name}_mask"]).sum(dim=-1)
        return total

    def local_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        losses = []
        for name, head in self.heads.items():
            scores = head(batch[f"{name}_hyp"]).squeeze(-1)
            gold = batch[f"{name}_hyp_gold"] > 0.5
            mask = batch[f"{name}_hyp_mask"] > 0.5
            valid = (gold & mask).any(dim=-1)
            if not bool(valid.any()):
                continue
            losses.append(ranking_loss(scores[valid], gold[valid], mask[valid]))
        if not losses:
            return batch["repeat_feat"].sum() * 0.0
        return torch.stack(losses).mean()


def parameter_count(model: LocalScorer) -> int:
    return sum(int(param.numel()) for param in model.parameters())


def _stack_examples(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = rows[0].keys()
    return {key: torch.stack([row[key] for row in rows]) for key in keys}


def load_examples(data_dir: Path, split: str) -> dict:
    import pyarrow.parquet as pq

    paths = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no {split} shards in {data_dir}")
    table = pq.read_table(paths, columns=["family", "programs", "gold", "mask", "supports"])
    families = table.column("family").to_pylist()
    programs = table.column("programs").to_pylist()
    golds = table.column("gold").to_pylist()
    masks = table.column("mask").to_pylist()
    supports = table.column("supports").to_pylist()
    count = len(families)
    batch: dict = {}
    for name, dim in DIMS.items():
        batch[f"{name}_feat"] = torch.zeros(count, 32, MAX_NODES, dim)
        batch[f"{name}_mask"] = torch.zeros(count, 32, MAX_NODES)
        batch[f"{name}_hyp"] = torch.zeros(count, MAX_HYP, dim)
        batch[f"{name}_hyp_mask"] = torch.zeros(count, MAX_HYP)
        batch[f"{name}_hyp_gold"] = torch.zeros(count, MAX_HYP)
    program_gold = torch.zeros(count, 32, dtype=torch.bool)
    program_mask = torch.zeros(count, 32, dtype=torch.bool)
    for index, (text, gold, mask, group) in enumerate(zip(programs, golds, masks, supports)):
        if index % 5000 == 0:
            print(f"pack {split} {index}", flush=True)
        progs = programs_from_json(text)
        gold_row = [bool(bit) for bit in gold]
        mask_row = [bool(bit) for bit in mask]
        packed = pack_example(progs, gold_row, mask_row, group)
        for key, value in packed.items():
            batch[key][index] = value
        for slot, bit in enumerate(gold_row[:32]):
            program_gold[index, slot] = bit
        for slot, bit in enumerate(mask_row[:32]):
            program_mask[index, slot] = bit
    batch["gold"] = program_gold
    batch["mask"] = program_mask
    batch["family"] = families
    return batch


def cache_split(data_dir: Path, split: str, cache: Path) -> dict:
    path = cache / f"{split}.pt"
    if path.exists():
        print(f"loading {path}", flush=True)
        saved = torch.load(path, weights_only=False)
        return saved
    print(f"packing {split}", flush=True)
    saved = load_examples(data_dir, split)
    cache.mkdir(parents=True, exist_ok=True)
    torch.save(saved, path)
    return saved


@torch.no_grad()
def _accuracy(scores: torch.Tensor, gold: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    filled = scores.masked_fill(~mask, -1e9)
    choice = filled.argmax(dim=-1)
    rows = torch.arange(choice.shape[0])
    return gold[rows, choice]


@torch.no_grad()
def evaluate(model: LocalScorer, data: dict, batch_size: int = 256) -> dict:
    model.eval()
    families = data["family"]
    n = len(families)
    program_hits = []
    local_hits = {name: [] for name in HEADS}
    local_present_hits = {name: [] for name in HEADS}
    evidence_hits = {"repeat": [], "integer": []}
    by_family: dict[str, list[float]] = {}
    local_by_family: dict[str, dict[str, list[float]]] = {name: {} for name in HEADS}
    local_present_by_family: dict[str, dict[str, list[float]]] = {name: {} for name in HEADS}
    evidence_by_family: dict[str, dict[str, list[float]]] = {"repeat": {}, "integer": {}}
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        batch = {key: value[start:stop] if torch.is_tensor(value) else value[start:stop] for key, value in data.items()}
        scores = model.program_scores(batch)
        scores = scores.masked_fill(~batch["mask"], -1e9)
        choice = scores.argmax(dim=-1)
        rows = torch.arange(choice.shape[0])
        hit = batch["gold"][rows, choice]
        program_hits.extend(hit.tolist())
        for family, correct in zip(families[start:stop], hit.tolist()):
            by_family.setdefault(family, []).append(float(correct))
        for name in HEADS:
            local_scores = model.heads[name](batch[f"{name}_hyp"]).squeeze(-1)
            gold = batch[f"{name}_hyp_gold"] > 0.5
            mask = batch[f"{name}_hyp_mask"] > 0.5
            correct = _accuracy(local_scores, gold, mask)
            local_hits[name].extend(correct.tolist())
            present = ~gold[:, 0]
            for offset, (bit, family) in enumerate(zip(correct.tolist(), families[start:stop])):
                local_by_family[name].setdefault(family, []).append(float(bit))
                if bool(present[offset]):
                    local_present_hits[name].append(float(bit))
                    local_present_by_family[name].setdefault(family, []).append(float(bit))
        repeat_scores = model.heads["repeat"](batch["repeat_hyp"]).squeeze(-1)
        repeat_choice = repeat_scores.masked_fill(~(batch["repeat_hyp_mask"] > 0.5), -1e9).argmax(dim=-1)
        tight = (batch["repeat_hyp"][rows, repeat_choice, 6] > 0.5) & (batch["repeat_hyp"][rows, repeat_choice, 7] > 0.5)
        has_repeat = batch["repeat_hyp"][:, 1:, 0].amax(dim=-1) > 0.5
        for family, bit, present in zip(families[start:stop], tight.tolist(), has_repeat.tolist()):
            if present:
                evidence_hits["repeat"].append(float(bit))
                evidence_by_family["repeat"].setdefault(family, []).append(float(bit))
        integer_scores = model.heads["integer"](batch["integer_hyp"]).squeeze(-1)
        integer_choice = integer_scores.masked_fill(~(batch["integer_hyp_mask"] > 0.5), -1e9).argmax(dim=-1)
        hull = batch["integer_hyp"][rows, integer_choice, 8] > 0.5
        has_int = batch["integer_hyp"][:, 1:, 0].amax(dim=-1) > 0.5
        for family, bit, present in zip(families[start:stop], hull.tolist(), has_int.tolist()):
            if present:
                evidence_hits["integer"].append(float(bit))
                evidence_by_family["integer"].setdefault(family, []).append(float(bit))
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "n": n,
        "top1": _mean([float(bit) for bit in program_hits]),
        "by_family": {name: _mean(rows) for name, rows in sorted(by_family.items())},
        "local": {name: _mean(rows) for name, rows in local_hits.items()},
        "local_present": {name: _mean(rows) for name, rows in local_present_hits.items()},
        "local_by_family": {
            name: {family: _mean(rows) for family, rows in sorted(families.items())}
            for name, families in local_by_family.items()
        },
        "local_present_by_family": {
            name: {family: _mean(rows) for family, rows in sorted(families.items())}
            for name, families in local_present_by_family.items()
        },
        "evidence": {name: _mean(rows) for name, rows in evidence_hits.items()},
        "evidence_by_family": {
            name: {family: _mean(rows) for family, rows in sorted(families.items())}
            for name, families in evidence_by_family.items()
        },
    }


def renaming_invariance(model: LocalScorer, n: int = 32, seed: int = 2507) -> float:
    """A within-class letter rename leaves the summed score in place."""
    import random
    import string

    from pattern_induction.renaming import apply_mapping

    rng = random.Random(seed)
    hits = 0
    model.eval()
    for _ in range(n):
        prefix = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
        supports = [prefix + text for text in ("2024", "2025", "2026", "2024")]
        mapping = {}
        pool = [char for char in string.ascii_uppercase if char not in prefix]
        rng.shuffle(pool)
        for char, target in zip(prefix, pool):
            mapping[char] = target
        renamed = [apply_mapping(text, mapping) for text in supports]
        program = Prog(
            "SEQ",
            children=[
                Prog("COPY", slot=0, start=0, end=3),
                Prog("INT", spec="2000:2099", lo=4, hi=4),
            ],
        )
        left = pack_example([program], [True], [True], supports)
        right = pack_example([program], [True], [True], renamed)
        batch = _stack_examples([left, right])
        scores = model.program_scores(batch)
        hits += int(torch.allclose(scores[0], scores[1], atol=1e-5))
    return hits / float(n)


def train_one(seed: int, data_dir: Path, cache: Path, out: Path, steps: int = 20000, batch_size: int = 32) -> dict:
    torch.manual_seed(seed)
    splits = {name: cache_split(data_dir, name, cache) for name in ("train", "val", "ood")}
    model = LocalScorer()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    count = len(splits["train"]["family"])
    order = torch.arange(count)
    best = {"top1": -1.0, "step": 0, "state": None}
    history = []
    step = 0
    while step < steps:
        perm = order[torch.randperm(order.shape[0])]
        for start in range(0, count, batch_size):
            index = perm[start : start + batch_size]
            batch = {
                key: value[index]
                for key, value in splits["train"].items()
                if torch.is_tensor(value)
            }
            opt.zero_grad(set_to_none=True)
            scores = model.program_scores(batch)
            loss = ranking_loss(scores, batch["gold"], batch["mask"]) + model.local_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 2000 == 0 or step >= steps:
                report = evaluate(model, splits["val"])
                print(
                    f"seed {seed} step {step} val {report['top1']:.4f} "
                    f"repeat_node {report['local_present']['repeat']:.3f} "
                    f"charset_node {report['local_present']['charset']:.3f} "
                    f"repeat_tight {report['evidence']['repeat']:.3f}",
                    flush=True,
                )
                history.append({"step": step, "val_top1": report["top1"], "local": report["local"]})
                if report["top1"] >= best["top1"]:
                    best = {
                        "top1": report["top1"],
                        "step": step,
                        "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    }
            if step >= steps:
                break
    model.load_state_dict(best["state"])
    val_report = evaluate(model, splits["val"])
    ood_report = evaluate(model, splits["ood"])
    invariance = renaming_invariance(model)
    report = {
        "seed": seed,
        "parameters": parameter_count(model),
        "best_step": best["step"],
        "val_top1": val_report["top1"],
        "ood_top1": ood_report["top1"],
        "val_by_family": val_report["by_family"],
        "ood_by_family": ood_report["by_family"],
        "val_local": val_report["local"],
        "ood_local": ood_report["local"],
        "val_local_present": val_report["local_present"],
        "ood_local_present": ood_report["local_present"],
        "val_local_by_family": val_report["local_by_family"],
        "ood_local_by_family": ood_report["local_by_family"],
        "val_local_present_by_family": val_report["local_present_by_family"],
        "ood_local_present_by_family": ood_report["local_present_by_family"],
        "val_evidence": val_report["evidence"],
        "ood_evidence": ood_report["evidence"],
        "val_evidence_by_family": val_report["evidence_by_family"],
        "ood_evidence_by_family": ood_report["evidence_by_family"],
        "renaming_invariance": invariance,
        "history": history,
    }
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "report": report}, out / "checkpoint.pt")
    (out / "metrics.json").write_text(__import__("json").dumps(report, indent=2) + "\n")
    print(
        f"seed {seed} best {best['step']} val {val_report['top1']:.4f} ood {ood_report['top1']:.4f} rename {invariance:.3f}",
        flush=True,
    )
    return report


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("results/prior/shards"))
    parser.add_argument("--cache", type=Path, default=Path("/tmp/relative-evidence"))
    parser.add_argument("--out", type=Path, default=Path("results/prior/runs/relative_evidence"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--steps", type=int, default=20000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    reports = [train_one(seed, args.data, args.cache, args.out / f"seed_{seed}", steps=args.steps) for seed in args.seeds]
    dest = Path("results/prior/relative.json")
    dest.write_text(json.dumps({"parameters": reports[0]["parameters"], "seeds": reports}, indent=2) + "\n")
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
