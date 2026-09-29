"""Identifiability of the candidate-relative packets.

Each hypothesis the local scorer sees is one float vector: the observations
of the span and the candidate's own parameters. This audit hashes those
vectors and asks whether the gold label is a function of the vector.

The training conditional is the ceiling. A packet that is gold in one task
and a decoy in another cannot be classified by any function of itself. The
candidate-pool ceiling ranks each pool by that conditional and breaks ties
uniformly. Validation and held-out rows use the training conditional only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prior.relative import HEADS

SLICES: dict[str, dict[str, tuple]] = {
    "repeat": {
        "tight": (("both", 6, 7),),
        "loose_container": (("pos", 8), ("not_both", 6, 7)),
        "miss": (("pos", 0), ("neg", 8)),
    },
    "integer": {
        "hull": (("pos", 8),),
        "expansion": (("pos", 7), ("neg", 8)),
        "miss": (("pos", 0), ("neg", 7)),
        "width_exact": (("pos", 10),),
        "width_inexact": (("neg", 10), ("pos", 11)),
    },
    "charset": {
        "exact_alphabet": (("pos", 1), ("zero", 3)),
        "expansion": (("pos", 1), ("pos", 3)),
        "miss": (("pos", 0), ("neg", 1)),
        "width_exact": (("pos", 10),),
        "width_open": (("pos", 11),),
    },
    "optional": {"candidate": ()},
    "literal": {"candidate": ()},
    "lexicon": {"candidate": ()},
}


def row_keys(values: np.ndarray) -> np.ndarray:
    """Exact float32 identity of each row. Negative zero matches positive zero."""
    raw = np.ascontiguousarray(values, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError("expected a matrix of packets")
    zeros = raw == 0
    if zeros.any():
        raw = raw.copy()
        raw[zeros] = 0
    width = raw.shape[1] * raw.dtype.itemsize
    return raw.view(np.dtype((np.void, width))).reshape(-1)


def group_labels(keys: np.ndarray, gold: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorted unique packets and their gold / nongold counts."""
    if len(keys) == 0:
        empty = np.array([], dtype=np.void)
        return empty, np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    order = np.argsort(keys)
    ordered = keys[order]
    labels = np.ascontiguousarray(gold, dtype=np.int64)[order]
    change = np.empty(len(ordered), dtype=bool)
    change[0] = True
    change[1:] = ordered[1:] != ordered[:-1]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(ordered)]
    cumulative = np.cumsum(labels)
    positive = cumulative[ends - 1] - np.where(starts > 0, cumulative[starts - 1], 0)
    total = ends - starts
    return ordered[starts], positive.astype(np.int64), (total - positive).astype(np.int64)


def _entropy(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    total = positive + negative
    p = np.divide(positive, total, out=np.zeros(len(total), dtype=np.float64), where=total > 0)
    out = np.zeros(len(p), dtype=np.float64)
    live = (p > 0) & (p < 1)
    out[live] = -(p[live] * np.log2(p[live]) + (1.0 - p[live]) * np.log2(1.0 - p[live]))
    return out


def packet_stats(positive: np.ndarray, negative: np.ndarray) -> dict:
    total = positive + negative
    count = int(total.sum()) if len(total) else 0
    if count == 0:
        return {
            "n": 0,
            "unique": 0,
            "gold_rate": 0.0,
            "conflict_packets": 0,
            "conflict_rate": 0.0,
            "entropy": 0.0,
            "majority_accuracy": 0.0,
            "gold_recall": 0.0,
        }
    p = positive / total
    conflict = (positive > 0) & (negative > 0)
    gold_n = int(positive.sum())
    recalled = int(positive[p >= 0.5].sum())
    return {
        "n": count,
        "unique": int(len(total)),
        "gold_rate": gold_n / count,
        "conflict_packets": int(conflict.sum()),
        "conflict_rate": float(total[conflict].sum() / count),
        "entropy": float((_entropy(positive, negative) * total).sum() / count),
        "majority_accuracy": float(np.maximum(positive, negative).sum() / count),
        "gold_recall": recalled / gold_n if gold_n else 0.0,
    }


def lookup_rates(
    train_keys: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    query: np.ndarray,
    backoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Training P(gold | packet) for each query. Unseen packets get ``backoff``."""
    rates = np.full(len(query), backoff, dtype=np.float64)
    seen = np.zeros(len(query), dtype=bool)
    if len(query) == 0 or len(train_keys) == 0:
        return rates, seen
    index = np.searchsorted(train_keys, query)
    index = np.clip(index, 0, len(train_keys) - 1)
    seen = train_keys[index] == query
    total = positive[index[seen]] + negative[index[seen]]
    rates[seen] = positive[index[seen]] / total
    return rates, seen


def pool_ceiling(scores: np.ndarray, gold: np.ndarray, alive: np.ndarray) -> dict:
    """Expected top-1 if tied scores are broken uniformly, plus the strict hit rate.

    A strict hit requires every hypothesis at the winning score to be gold.
    Identical packets share a score, so a conflicting packet cannot be strict.
    """
    if scores.size == 0:
        return {"n": 0, "top1": 0.0, "strict": 0.0}
    filled = np.where(alive, scores, -np.inf)
    best = filled.max(axis=1)
    valid = np.isfinite(best)
    tied = alive & (filled == best[:, None])
    gold_tied = tied & gold
    n_tied = tied.sum(axis=1)
    n_gold = gold_tied.sum(axis=1)
    expected = np.zeros(len(best), dtype=np.float64)
    expected[valid] = n_gold[valid] / np.maximum(n_tied[valid], 1)
    strict = valid & (n_tied > 0) & (n_gold == n_tied)
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "top1": 0.0, "strict": 0.0}
    return {"n": n, "top1": float(expected[valid].mean()), "strict": float(strict[valid].mean())}


def _candidate_mask(values: np.ndarray) -> np.ndarray:
    return values[:, -1] < 0.5


def _slice_mask(name: str, slice_name: str, values: np.ndarray) -> np.ndarray:
    mask = _candidate_mask(values)
    for rule in SLICES[name][slice_name]:
        kind = rule[0]
        if kind == "pos":
            mask &= values[:, rule[1]] > 0.5
        elif kind == "neg":
            mask &= values[:, rule[1]] <= 0.5
        elif kind == "zero":
            mask &= np.abs(values[:, rule[1]]) <= 1e-6
        elif kind == "both":
            mask &= (values[:, rule[1]] > 0.5) & (values[:, rule[2]] > 0.5)
        elif kind == "not_both":
            mask &= ~((values[:, rule[1]] > 0.5) & (values[:, rule[2]] > 0.5))
        else:
            raise ValueError(kind)
    return mask


def _flatten_alive(values: np.ndarray, mask: np.ndarray, gold: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alive = np.ascontiguousarray(mask > 0.5)
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1, values.shape[-1])
    keep = alive.reshape(-1)
    rows = np.repeat(np.arange(values.shape[0]), values.shape[1])
    return flat[keep], np.ascontiguousarray(gold, dtype=bool).reshape(-1)[keep], rows[keep]


def _head_views(split: dict, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = split[f"{name}_hyp"].numpy()
    mask = split[f"{name}_hyp_mask"].numpy()
    gold = split[f"{name}_hyp_gold"].numpy()
    return values, mask, gold


def _slice_report(flat: np.ndarray, flat_gold: np.ndarray, name: str) -> dict:
    report = {}
    for slice_name in SLICES[name]:
        chosen = _slice_mask(name, slice_name, flat)
        if chosen.any():
            _keys, positive, negative = group_labels(row_keys(flat[chosen]), flat_gold[chosen])
        else:
            _keys, positive, negative = group_labels(np.array([], dtype=np.void), np.zeros(0, dtype=bool))
        report[slice_name] = packet_stats(positive, negative)
    return report


def audit_head(train: dict, others: dict[str, dict], name: str) -> dict:
    values, mask, gold = _head_views(train, name)
    flat, flat_gold, row_index = _flatten_alive(values, mask, gold)
    candidate = _candidate_mask(flat)
    keys = row_keys(flat[candidate])
    train_keys, positive, negative = group_labels(keys, flat_gold[candidate])
    base = float(flat_gold[candidate].mean()) if candidate.any() else 0.0
    abstain = ~candidate
    abstain_rate = float(flat_gold[abstain].mean()) if abstain.any() else 0.0

    def rates_for(split_values: np.ndarray, split_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        split_flat, _, _ = _flatten_alive(split_values, split_mask, np.zeros(split_mask.shape, dtype=bool))
        query = row_keys(split_flat) if len(split_flat) else np.array([], dtype=np.void)
        rates, seen = lookup_rates(train_keys, positive, negative, query, base)
        # The abstain vector is one constant packet. Its training rate replaces the backoff.
        candidate_rows = np.zeros(len(split_flat), dtype=bool)
        if split_flat.size:
            candidate_rows = _candidate_mask(split_flat)
            rates[~candidate_rows] = abstain_rate
            seen[~candidate_rows] = True
        coverage = float(seen[candidate_rows].mean()) if candidate_rows.any() else 0.0
        return rates, seen, coverage

    def ceiling_from(split: dict, rates: np.ndarray, alive_index: np.ndarray) -> dict:
        n, width = split[f"{name}_hyp_mask"].shape
        scores = np.full((n, width), -np.inf, dtype=np.float64)
        alive = split[f"{name}_hyp_mask"].numpy() > 0.5
        positions = np.flatnonzero(alive.reshape(-1))
        scores.reshape(-1)[positions] = rates
        labels = split[f"{name}_hyp_gold"].numpy() > 0.5
        present = labels[:, 0] == False  # noqa: E712 abstain is not the gold
        full = pool_ceiling(scores, labels, alive)
        presented = pool_ceiling(scores[present], labels[present], alive[present])
        return {"all": full, "present": presented, "rows": int(len(alive_index))}

    train_rates, _train_seen, train_coverage = rates_for(values, mask)
    report = {
        "abstain_n": int(abstain.sum()),
        "abstain_gold_rate": abstain_rate,
        "train": {
            "packets": packet_stats(positive, negative),
            "pool": ceiling_from(train, train_rates, row_index),
            "seen_rate": train_coverage,
        },
        "slices": _slice_report(flat, flat_gold, name),
        "by_family": {},
    }
    families = list(train["family"])
    report["by_family"]["train"] = _family_pools(families, ceiling_from(train, train_rates, row_index), values.shape[0], train, name, train_rates)

    for split_name, split in others.items():
        split_values, split_mask, _ = _head_views(split, name)
        split_rates, split_seen, coverage = rates_for(split_values, split_mask)
        pooled = ceiling_from(split, split_rates, np.arange(len(split_rates)))
        split_flat, split_gold, _rows = _flatten_alive(
            split_values, split_mask, split[f"{name}_hyp_gold"].numpy()
        )
        report[split_name] = {
            "pool": pooled,
            "seen_rate": coverage,
            "slices": _slice_report(split_flat, split_gold, name),
        }
        report["by_family"][split_name] = _family_pools(list(split["family"]), pooled, split_values.shape[0], split, name, split_rates)
    return report


def _family_pools(families: list[str], _unused: dict, n: int, split: dict, name: str, rates: np.ndarray) -> dict[str, dict]:
    labels = split[f"{name}_hyp_gold"].numpy() > 0.5
    alive = split[f"{name}_hyp_mask"].numpy() > 0.5
    scores = np.full((n, labels.shape[1]), -np.inf, dtype=np.float64)
    positions = np.flatnonzero(alive.reshape(-1))
    if len(rates) != len(positions):
        raise RuntimeError(f"{name} rate rows {len(rates)} != alive cells {len(positions)}")
    scores.reshape(-1)[positions] = rates
    present = labels[:, 0] == False  # noqa: E712
    out = {}
    names = np.asarray(families)
    for family in sorted(set(families)):
        chosen = names == family
        if not chosen.any():
            continue
        out[family] = {
            "all": pool_ceiling(scores[chosen], labels[chosen], alive[chosen]),
            "present": pool_ceiling(scores[chosen & present], labels[chosen & present], alive[chosen & present]),
        }
    return out


def _program_gold(split: dict) -> np.ndarray:
    return split["gold"].numpy() > 0.5


def _mix(ids: np.ndarray, scale: int, shift: int) -> np.ndarray:
    mixed = ids.astype(np.uint64) * np.uint64(scale)
    return mixed ^ (mixed >> np.uint64(shift))


def _fingerprint(ident: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """Order-invariant 128-bit multiset hash plus the live node count.

    Addition keeps two copies of one packet distinct from one copy. XOR would
    cancel them.
    """
    total_a = np.zeros(ident.shape[:-1], dtype=np.uint64)
    total_b = np.zeros(ident.shape[:-1], dtype=np.uint64)
    count = np.zeros(ident.shape[:-1], dtype=np.uint16)
    for slot in range(ident.shape[-1]):
        live = alive[..., slot]
        total_a = total_a + np.where(live, _mix(ident[..., slot], 0x9E3779B185EBCA87, 33), np.uint64(0))
        total_b = total_b + np.where(live, _mix(ident[..., slot], 0xBF58476D1CE4E5B9, 27), np.uint64(0))
        count = count + live.astype(np.uint16)
    packed = np.zeros(ident.shape[:-1], dtype=[("a", "<u8"), ("b", "<u8"), ("n", "<u2")])
    packed["a"] = total_a
    packed["b"] = total_b
    packed["n"] = count
    width = packed.dtype.itemsize
    return np.ascontiguousarray(packed).view(np.dtype((np.void, width))).reshape(packed.shape)


def _node_layout(split: dict, vocab: dict[bytes, int] | None) -> tuple[np.ndarray, np.ndarray, dict[bytes, int]]:
    """Shared integer ids for every node packet, so programs can share a signature."""
    owned = vocab is None
    if vocab is None:
        vocab = {}
    unseen_start = len(vocab) + 1
    columns: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for name in HEADS:
        feat = np.ascontiguousarray(split[f"{name}_feat"].numpy(), dtype=np.float32)
        mask = split[f"{name}_mask"].numpy() > 0.5
        unique, inverse = np.unique(row_keys(feat.reshape(-1, feat.shape[-1])), return_inverse=True)
        unique_ids = np.zeros(len(unique), dtype=np.int32)
        fresh = 0
        for slot, key in enumerate(unique):
            blob = bytes(key)
            found = vocab.get(blob)
            if found is None:
                if owned:
                    found = len(vocab) + 1
                    vocab[blob] = found
                else:
                    fresh += 1
                    found = unseen_start + fresh
                    vocab[blob] = found
            unique_ids[slot] = found
        columns.append(unique_ids[inverse].reshape(feat.shape[:-1]))
        masks.append(mask)
    return np.concatenate(columns, axis=-1), np.concatenate(masks, axis=-1), vocab


def _program_ceiling(train: dict, others: dict[str, dict]) -> dict:
    """Bayes top-1 over programs given the multiset of their node packets."""
    train_ident, _train_nodes, vocab = _node_layout(train, None)
    train_sig = _fingerprint(train_ident, _train_nodes)
    train_labels = _program_gold(train)
    train_mask = train["mask"].numpy() > 0.5
    keys, positive, negative = group_labels(train_sig.reshape(-1)[train_mask.reshape(-1)], train_labels.reshape(-1)[train_mask.reshape(-1)])
    base = float(train_labels[train_mask].mean()) if train_mask.any() else 0.0

    def apply(split: dict, signatures: np.ndarray) -> dict:
        labels = _program_gold(split)
        alive = split["mask"].numpy() > 0.5
        flat_sig = signatures.reshape(-1)
        flat_alive = alive.reshape(-1)
        own_keys, own_pos, own_neg = group_labels(flat_sig[flat_alive], labels.reshape(-1)[flat_alive])
        rates, seen = lookup_rates(keys, positive, negative, flat_sig, base)
        scores = np.where(alive, rates.reshape(labels.shape), -np.inf)
        families = np.asarray(list(split["family"]))
        by_family = {}
        for family in sorted(set(families.tolist())):
            chosen = families == family
            by_family[family] = pool_ceiling(scores[chosen], labels[chosen], alive[chosen])
        return {
            "all": pool_ceiling(scores, labels, alive),
            "by_family": by_family,
            "packets": packet_stats(own_pos, own_neg),
            "seen_rate": float(seen[flat_alive].mean()) if flat_alive.any() else 0.0,
        }

    report = {
        "train": apply(train, train_sig),
        "unique_signatures": int(len(keys)),
        "base_gold": base,
    }
    for name, split in others.items():
        ident, node_alive, vocab = _node_layout(split, vocab)
        report[name] = apply(split, _fingerprint(ident, node_alive))
    return report


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, (np.floating,)):
        return round(float(value), 6)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def audit(cache: Path) -> dict:
    train = _load(cache, "train")
    others = {name: _load(cache, name) for name in ("val", "ood")}
    heads = {name: audit_head(train, others, name) for name in HEADS}
    programs = _program_ceiling(train, others)
    return {"heads": heads, "programs": _slim_programs(programs)}


def _slim_programs(programs: dict) -> dict:
    """Drop nothing structural; the signature table itself is not serialized."""
    return programs


def _load(cache: Path, split: str) -> dict:
    import torch

    path = cache / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, weights_only=False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/relative-evidence"))
    parser.add_argument("--out", type=Path, default=Path("results/prior/packets.json"))
    args = parser.parse_args()
    report = audit(args.cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)
    for name, head in report["heads"].items():
        packets = head["train"]["packets"]
        pool = head["train"]["pool"]["present"]
        print(
            f"{name:10} conflict {packets['conflict_rate']:.3f} entropy {packets['entropy']:.3f} "
            f"majority {packets['majority_accuracy']:.3f} pool {pool['top1']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
