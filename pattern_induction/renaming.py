"""Symbol renaming and the relations that survive it.

Membership should depend on how symbols line up across the support set, not on
which codepoint plays each role. Within a task, symbols get local ids in the
order they first appear. Structural class keeps uppercase, lowercase, digits,
spaces, and separators apart. A within-class rename, including `-` to `§`,
leaves those features unchanged.
"""

from __future__ import annotations

import random
import string
from collections import defaultdict

import torch

from pattern_induction.membership import HELD_OUT, Episode

UPPER, LOWER, DIGIT, SPACE, PUNCT = range(5)
N_CLASS = 5
# Separators share one class, whatever their Unicode category. `§` is a symbol
# and `-` is a dash; both are punctuation for this purpose.
PUNCT_POOL = list("-_./:#$%+@=,~*^|!?&§")
HELD_OUT_PUNCT = [ch for ch in "#~*=§" if ch in HELD_OUT]
POOLS = {
    UPPER: list(string.ascii_uppercase),
    LOWER: list(string.ascii_lowercase),
    DIGIT: list(string.digits),
    SPACE: [" "],
    PUNCT: PUNCT_POOL,
}

# Per-character relations to the support, at the left column and the right column.
# Coverage is the fraction of support strings that have the aligned cell.
# Id-match is the fraction whose symbol there equals this one.
# Constant means every support string has that cell and they all agree.
N_REL = 9
REL_LEFT_COVERAGE = 0
REL_LEFT_ID = 1
REL_LEFT_CLASS = 2
REL_LEFT_CONSTANT = 3
REL_RIGHT_COVERAGE = 4
REL_RIGHT_ID = 5
REL_RIGHT_CLASS = 6
REL_RIGHT_CONSTANT = 7
REL_NEW = 8

MAX_SYMBOLS = 48


def structural_class(ch: str) -> int:
    if ch.isupper():
        return UPPER
    if ch.islower():
        return LOWER
    if ch.isdigit():
        return DIGIT
    if ch.isspace():
        return SPACE
    return PUNCT


def build_mapping(strings: list[str], rng: random.Random, force_holdout: bool) -> dict[str, str]:
    """Within-class bijection. One separator is sent to a held-out mark when asked."""
    grouped: dict[int, list[str]] = defaultdict(list)
    seen = set()
    for text in strings:
        for ch in text:
            if ch in seen:
                continue
            seen.add(ch)
            grouped[structural_class(ch)].append(ch)
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for cls, group in grouped.items():
        pool = [ch for ch in POOLS[cls] if ch not in used]
        rng.shuffle(pool)
        targets: list[str] = []
        if force_holdout and cls == PUNCT and group:
            hold = [ch for ch in HELD_OUT_PUNCT if ch not in used and ch != group[0]]
            if not hold:
                hold = [ch for ch in HELD_OUT_PUNCT if ch not in used]
            if hold:
                targets.append(rng.choice(hold))
                pool = [ch for ch in pool if ch != targets[0]]
        for src in group[len(targets) :]:
            pick = next((ch for ch in pool if ch != src), None)
            if pick is None and pool:
                pick = pool[0]
            if pick is None:
                pick = src
            targets.append(pick)
            if pick in pool:
                pool.remove(pick)
        if len(targets) < len(group):
            for src in group[len(targets) :]:
                targets.append(src)
        for src, dst in zip(group, targets):
            mapping[src] = dst
            used.add(dst)
    return mapping


def apply_mapping(text: str, mapping: dict[str, str]) -> str:
    return "".join(mapping.get(ch, ch) for ch in text)


def rename_episode(episode: Episode, rng: random.Random, force_holdout: bool = True) -> Episode:
    """Same label, new symbols. The gold pattern string is not renamed."""
    mapping = build_mapping(list(episode.support) + [episode.query], rng, force_holdout)
    return Episode(
        [apply_mapping(text, mapping) for text in episode.support],
        apply_mapping(episode.query, mapping),
        episode.label,
        "renamed",
        episode.split,
        episode.kind,
    )


def contains_holdout(episode: Episode) -> bool:
    blob = "".join(episode.support) + episode.query
    return any(ch in HELD_OUT for ch in blob)


def _assign(support: list[str], query: str) -> tuple[dict[str, int], set[str]]:
    mapping: dict[str, int] = {}

    def add(ch: str) -> None:
        if ch not in mapping and len(mapping) < MAX_SYMBOLS:
            mapping[ch] = len(mapping) + 1

    for text in support:
        for ch in text:
            add(ch)
    support_symbols = set(mapping)
    for ch in query:
        add(ch)
    return mapping, support_symbols


def _column(strings: list[str], index: int, ids: dict[str, int], from_right: bool) -> tuple[list[int], list[int]]:
    found_ids = []
    found_cls = []
    for text in strings:
        if index >= len(text):
            continue
        ch = text[-1 - index] if from_right else text[index]
        if ch not in ids:
            continue
        found_ids.append(ids[ch])
        found_cls.append(structural_class(ch))
    return found_ids, found_cls


def featurize(
    episodes: list[Episode],
    max_len: int = 48,
    max_support: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Local ids, structural classes, relation features, and a pad mask.

    Local id 0 is padding. Relation features are zero on pad cells.
    """
    batch = len(episodes)
    slots = max_support + 1
    local = torch.zeros(batch, slots, max_len, dtype=torch.long)
    classes = torch.zeros(batch, slots, max_len, dtype=torch.long)
    rel = torch.zeros(batch, slots, max_len, N_REL)
    pad = torch.ones(batch, slots, max_len, dtype=torch.bool)
    for row, episode in enumerate(episodes):
        support = [text[:max_len] for text in episode.support[:max_support] if text]
        query = episode.query[:max_len]
        if not support or not query:
            continue
        ids, support_symbols = _assign(support, query)
        n_support = len(support)
        placed = [(slot, text) for slot, text in enumerate(support)]
        placed.append((slots - 1, query))
        for slot, text in placed:
            for index, ch in enumerate(text):
                local_id = ids.get(ch, 0)
                if local_id == 0:
                    continue
                local[row, slot, index] = local_id
                classes[row, slot, index] = structural_class(ch)
                pad[row, slot, index] = False
                left_ids, left_cls = _column(support, index, ids, False)
                right_off = len(text) - 1 - index
                right_ids, right_cls = _column(support, right_off, ids, True)
                cls = structural_class(ch)
                rel[row, slot, index, REL_LEFT_COVERAGE] = len(left_ids) / n_support
                rel[row, slot, index, REL_LEFT_ID] = left_ids.count(local_id) / n_support
                rel[row, slot, index, REL_LEFT_CLASS] = left_cls.count(cls) / n_support
                rel[row, slot, index, REL_LEFT_CONSTANT] = float(
                    len(left_ids) == n_support and len(set(left_ids)) == 1
                )
                rel[row, slot, index, REL_RIGHT_COVERAGE] = len(right_ids) / n_support
                rel[row, slot, index, REL_RIGHT_ID] = right_ids.count(local_id) / n_support
                rel[row, slot, index, REL_RIGHT_CLASS] = right_cls.count(cls) / n_support
                rel[row, slot, index, REL_RIGHT_CONSTANT] = float(
                    len(right_ids) == n_support and len(set(right_ids)) == 1
                )
                rel[row, slot, index, REL_NEW] = float(ch not in support_symbols)
    return local, classes, rel, pad


def invariance_score(original: list[float], renamed: list[float]) -> dict[str, float]:
    """1 means renaming a task does not move the membership probability."""
    if not original:
        return {"invariance": 0.0, "mean_abs_delta": 0.0, "decision_agreement": 0.0}
    deltas = [abs(left - right) for left, right in zip(original, renamed)]
    agree = [(left >= 0.5) == (right >= 0.5) for left, right in zip(original, renamed)]
    mean_delta = sum(deltas) / len(deltas)
    return {
        "invariance": 1.0 - mean_delta,
        "mean_abs_delta": mean_delta,
        "decision_agreement": sum(agree) / len(agree),
    }


def member_acceptance(episodes: list[Episode], probs: list[float]) -> float:
    positives = [(episode, prob) for episode, prob in zip(episodes, probs) if episode.label == 1]
    if not positives:
        return 0.0
    return sum(prob >= 0.5 for _episode, prob in positives) / len(positives)
