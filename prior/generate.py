"""Synthetic ambiguity tasks. The frozen expander proposes candidates; this file only labels them.

A task is kept when the latent language is in that candidate pool. The probe
accepts the sampled language and rejects the next wider one, so a hull or an
open class is not marked gold.
"""

from __future__ import annotations

import json
import random
import string
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from pattern_induction.charset import charset_spec
from pattern_induction.latent import expand_programs
from pattern_induction.lexicon import LEXICONS
from pattern_induction.program import Prog, _MEMBERSHIP, describe, execute
from prior.dataset import MAX_CANDIDATES
from prior.features import FEATURE_DIM, candidate_features
from prior.support_encode import SUPPORT_DIM, embed_supports, load_frozen_encoder

SEED = 2507
_LEX = {item.name: sorted(item.members) for item in LEXICONS}

ID_INTERVALS = (
    (0, 9, 1),
    (0, 99, 2),
    (1, 99, 2),
    (0, 59, 2),
    (1, 12, 2),
    (0, 12, 2),
    (2000, 2099, 4),
    (1900, 1999, 4),
)
OOD_INTERVALS = (
    (0, 23, 2),
    (0, 31, 2),
    (1, 31, 2),
    (0, 255, 3),
    (1900, 2099, 4),
)
ID_ALPHABETS = {
    "hex_upper": "0123456789ABCDEF",
    "hex_lower": "0123456789abcdef",
    "digit": "0123456789",
    "upper": string.ascii_uppercase,
    "lower": string.ascii_lowercase,
}
OOD_ALPHABETS = {
    "alnum_upper": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "alnum": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
}
ID_LEXICONS = ("iso4217", "month_abbr", "http_iana")
OOD_LEXICONS = ("file_extension", "apparel_size", "mass_unit")
ID_FAMILIES = ("interval", "charset", "lexicon", "width", "prefix")
OOD_FAMILIES = ("interval_ood", "charset_ood", "lexicon_ood", "width_ood", "composition_ood")
_PURE = {
    "0123456789": "digit",
    string.ascii_uppercase: "upper",
    string.ascii_lowercase: "lower",
}


def _perfect(prog: Prog, supports: list[str], positives: list[str], negatives: list[str]) -> bool:
    return all(execute(prog, text, supports) for text in positives) and all(
        not execute(prog, text, supports) for text in negatives
    )


def _plain(value: int, width: int) -> str | None:
    """Fixed-width digits with no leading zero, so the factorizer emits an integer."""
    text = f"{value:0{width}d}"
    if len(text) != width or (width > 1 and text.startswith("0")):
        return None
    return text


def _interval_example(rng: random.Random, table: tuple[tuple[int, int, int], ...]):
    lo, hi, width = table[rng.randrange(len(table))]
    floor = lo if width == 1 else max(lo, 10 ** (width - 1))
    if hi - floor < 1:
        return None
    span = max(1, (hi - floor) // 4)
    for _ in range(40):
        hlo = rng.randint(floor, hi - 1)
        hhi = rng.randint(hlo, min(hi, hlo + span))
        if hlo <= lo and hhi >= hi:
            continue
        outside = [value for value in range(lo, hi + 1) if value < hlo or value > hhi]
        if not outside:
            continue
        shown = [hlo, hhi, rng.randint(hlo, hhi), rng.randint(hlo, hhi)]
        supports = [_plain(value, width) for value in shown]
        if any(text is None for text in supports):
            continue
        positives = [f"{lo:0{width}d}", f"{hi:0{width}d}", f"{rng.choice(outside):0{width}d}"]
        negatives = [f"{hi + 1:0{width}d}", "A" * width]
        if lo > 0:
            negatives.append(f"{lo - 1:0{width}d}")
        if width > 1:
            negatives.append(str(rng.randint(0, 9)))
        ref = Prog("INT", spec=f"{lo}:{hi}", lo=width, hi=width)
        if _perfect(ref, supports, positives, negatives):
            return supports, positives, negatives, ref
    return None


def _charset_example(rng: random.Random, alphabets: dict[str, str]):
    alphabet = alphabets[rng.choice(list(alphabets))]
    if len(alphabet) < 4:
        return None
    dropped = rng.choice(alphabet)
    observed = [char for char in alphabet if char != dropped]
    width = rng.choice((2, 3, 4))
    supports = ["".join(rng.choice(observed) for _ in range(width)) for _ in range(4)]
    positives = [dropped + "".join(rng.choice(alphabet) for _ in range(width - 1)) for _ in range(3)]
    outside = next(char for char in "Ggz0A!@-" if char not in alphabet)
    negatives = [outside * width, alphabet[0] * (width + 1)]
    if width > 1:
        negatives.append(alphabet[0] * (width - 1))
    pure = _PURE.get(alphabet)
    if pure:
        ref = Prog("CLS", cls=pure, lo=width, hi=width)
    else:
        ref = Prog("SET", spec=charset_spec(set(alphabet)), lo=width, hi=width)
    if _perfect(ref, supports, positives, negatives):
        return supports, positives, negatives, ref
    return None


def _lexicon_example(rng: random.Random, names: tuple[str, ...]):
    name = names[rng.randrange(len(names))]
    members = [token for token in _LEX[name] if token.isalnum() and 1 <= len(token) <= 8]
    take = 4 if len(members) >= 8 else 3
    if len(members) < take + 2:
        return None
    chosen = rng.sample(members, take)
    held = [token for token in members if token not in chosen]
    other = rng.choice(held)
    supports = list(chosen)
    positives = [other, rng.choice(held)]
    typo = list(other)
    typo[0] = "Q" if typo[0] != "Q" else "Z"
    typo_text = "".join(typo)
    member_set = set(_LEX[name])
    negatives = []
    for text in (typo_text, other.lower(), other.upper(), other + "x", "999", "000"):
        if text and text not in member_set and text not in negatives:
            negatives.append(text)
    if len(negatives) < 2:
        return None
    ref = Prog("LEX", spec=name)
    if _perfect(ref, supports, positives, negatives):
        return supports, positives, negatives, ref
    return None


def _width_example(rng: random.Random, highs: tuple[int, ...]):
    gold_hi = highs[rng.randrange(len(highs))]
    obs_lo = min(gold_hi - 1, rng.choice((2, 3)))
    if obs_lo < 2 or obs_lo >= gold_hi:
        return None
    lengths = [obs_lo, gold_hi, rng.randint(obs_lo, gold_hi), rng.randint(obs_lo, gold_hi)]
    supports = ["".join(rng.choice("123456789") for _ in range(length)) for length in lengths]
    positives = ["7", "".join(rng.choice("123456789") for _ in range(gold_hi)), "2" * obs_lo]
    negatives = ["9" * (gold_hi + 1), "a" * obs_lo, ""]
    negatives = [text for text in negatives if text]
    ref = Prog("CLS", cls="digit", lo=1, hi=gold_hi)
    if _perfect(ref, supports, positives, negatives):
        return supports, positives, negatives, ref
    return None


def _prefix_example(rng: random.Random):
    drawn = _interval_example(rng, ID_INTERVALS)
    if drawn is None:
        return None
    numbers, positives, negatives, number = drawn
    prefix = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
    supports = [prefix + text for text in numbers]
    positives = [prefix + text for text in positives]
    negatives = [prefix + text for text in negatives] + [prefix[:-1] + "9" + numbers[0]]
    ref = Prog(
        "SEQ",
        children=[Prog("COPY", slot=0, start=0, end=len(prefix)), number],
    )
    if _perfect(ref, supports, positives, negatives):
        return supports, positives, negatives, ref
    return None


def _composition_example(rng: random.Random):
    members = _LEX["iso4217"]
    chosen = rng.sample(members, 4)
    other = rng.choice([token for token in members if token not in chosen])
    prefix = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
    supports = [f"{prefix}-{code}" for code in chosen]
    positives = [f"{prefix}-{other}", f"{prefix}-{rng.choice(members)}"]
    typo = "Q" + other[1:]
    if typo in members:
        typo = "ZZZ"
    negatives = [f"{prefix}-{typo}", f"XXX-{chosen[0]}", other, f"{prefix}-{other.lower()}"]
    ref = Prog(
        "SEQ",
        children=[Prog("COPY", slot=0, start=0, end=len(prefix) + 1), Prog("LEX", spec="iso4217")],
    )
    if _perfect(ref, supports, positives, negatives):
        return supports, positives, negatives, ref
    return None


_BUILDERS = {
    "interval": lambda rng: _interval_example(rng, ID_INTERVALS),
    "interval_ood": lambda rng: _interval_example(rng, OOD_INTERVALS),
    "charset": lambda rng: _charset_example(rng, ID_ALPHABETS),
    "charset_ood": lambda rng: _charset_example(rng, OOD_ALPHABETS),
    "lexicon": lambda rng: _lexicon_example(rng, ID_LEXICONS),
    "lexicon_ood": lambda rng: _lexicon_example(rng, OOD_LEXICONS),
    "width": lambda rng: _width_example(rng, (4, 5, 6)),
    "width_ood": lambda rng: _width_example(rng, (8, 12)),
    "prefix": _prefix_example,
    "composition_ood": _composition_example,
}


def build_task(rng: random.Random, family: str, cap: int = 48) -> dict | None:
    """One labeled task, or None when the frozen pool misses the latent language."""
    _MEMBERSHIP.clear()
    drawn = _BUILDERS[family](rng)
    if drawn is None:
        return None
    supports, positives, negatives, _ref = drawn
    programs = expand_programs([], supports, cap=cap)
    perfect = [prog for prog in programs if _perfect(prog, supports, positives, negatives)]
    if not perfect:
        return None
    gold_keys = {describe(prog) for prog in perfect}
    rest = [prog for prog in programs if describe(prog) not in gold_keys]
    rng.shuffle(perfect)
    rng.shuffle(rest)
    chosen = (perfect[:4] + rest)[:MAX_CANDIDATES]
    rng.shuffle(chosen)
    feats = [0.0] * (MAX_CANDIDATES * FEATURE_DIM)
    mask = [0] * MAX_CANDIDATES
    gold_row = [0] * MAX_CANDIDATES
    for index, prog in enumerate(chosen):
        row = candidate_features(prog, supports)
        start = index * FEATURE_DIM
        feats[start : start + FEATURE_DIM] = row
        mask[index] = 1
        gold_row[index] = int(describe(prog) in gold_keys)
    if sum(gold_row) == 0:
        return None
    return {"family": family, "supports": supports, "candidate": feats, "mask": mask, "gold": gold_row}


def _worker(payload: tuple) -> tuple[list[dict], int]:
    family_cycle, count, seed = payload
    rng = random.Random(seed)
    rows: list[dict] = []
    attempts = 0
    limit = count * 12
    while len(rows) < count and attempts < limit:
        family = family_cycle[attempts % len(family_cycle)]
        attempts += 1
        task = build_task(rng, family)
        if task is not None:
            rows.append(task)
    return rows, attempts


def _write_chunk(path: Path, rows: list[dict], embeddings: torch.Tensor) -> None:
    table = pa.table(
        {
            "family": [row["family"] for row in rows],
            "support": embeddings.tolist(),
            "candidate": [row["candidate"] for row in rows],
            "mask": [row["mask"] for row in rows],
            "gold": [row["gold"] for row in rows],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _collect(families: tuple[str, ...], count: int, seed: int) -> tuple[list[dict], int]:
    import multiprocessing as mp

    workers = min(4, max(1, count // 500 or 1))
    per = [count // workers] * workers
    for index in range(count % workers):
        per[index] += 1
    payloads = [(families, size, seed + index * 100_003) for index, size in enumerate(per) if size]
    with mp.Pool(len(payloads)) as pool:
        parts = pool.map(_worker, payloads)
    rows: list[dict] = []
    attempts = 0
    for group, tried in parts:
        rows.extend(group)
        attempts += tried
    return rows[:count], attempts


def generate_split(root: Path, split: str, families: tuple[str, ...], count: int, seed: int) -> dict:
    rows, attempts = _collect(families, count, seed)
    encoder = load_frozen_encoder(root)
    out = root / "results" / "prior" / "shards"
    written = 0
    for start in range(0, len(rows), 4000):
        chunk = rows[start : start + 4000]
        vectors = []
        for cursor in range(0, len(chunk), 256):
            batch = chunk[cursor : cursor + 256]
            vectors.append(embed_supports(encoder, [row["supports"] for row in batch]))
        embedding = torch.cat(vectors, dim=0) if vectors else torch.zeros(0, SUPPORT_DIM)
        _write_chunk(out / f"{split}-{start // 4000:05d}.parquet", chunk, embedding)
        written += len(chunk)
    by_family: dict[str, int] = {}
    for row in rows:
        by_family[row["family"]] = by_family.get(row["family"], 0) + 1
    return {"split": split, "kept": written, "attempts": attempts, "oracle": written / max(attempts, 1), "by_family": by_family}


def pilot(n: int, seed: int) -> dict:
    rng = random.Random(seed)
    report = {}
    for family in ID_FAMILIES + OOD_FAMILIES:
        kept = 0
        for _ in range(n):
            if build_task(rng, family) is not None:
                kept += 1
        report[family] = {"kept": kept, "attempts": n, "oracle": kept / n}
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--train", type=int, default=50_000)
    parser.add_argument("--val", type=int, default=10_000)
    parser.add_argument("--ood", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.pilot:
        report = pilot(args.pilot, args.seed)
        print(json.dumps(report, indent=2))
        return
    root = Path(__file__).resolve().parents[1]
    shards = root / "results" / "prior" / "shards"
    if shards.exists():
        for path in shards.glob("*.parquet"):
            path.unlink()
    summary = {
        "seed": args.seed,
        "support_dim": SUPPORT_DIM,
        "feature_dim": FEATURE_DIM,
        "max_candidates": MAX_CANDIDATES,
        "splits": [
            generate_split(root, "train", ID_FAMILIES, args.train, args.seed),
            generate_split(root, "val", ID_FAMILIES, args.val, args.seed + 1),
            generate_split(root, "ood", OOD_FAMILIES, args.ood, args.seed + 2),
        ],
    }
    dest = root / "results" / "prior" / "generation.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
