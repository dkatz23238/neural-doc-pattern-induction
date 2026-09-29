"""Synthetic validation, structural OOD, and renaming invariance.

BENCH-C is not a selection metric. BENCH-E is not read.
"""

from __future__ import annotations

import random
import string

import torch

from pattern_induction.program import Prog
from pattern_induction.renaming import apply_mapping
from prior.dataset import PriorDataset
from prior.features import candidate_features
from prior.model import PriorModel
from prior.support_encode import embed_supports


def move_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


@torch.no_grad()
def evaluate(model: PriorModel, data: PriorDataset, device: torch.device, batch_size: int = 256) -> dict:
    model.eval()
    totals = {"n": 0, "top1": 0.0}
    by_family: dict[str, list[float]] = {}
    for start in range(0, len(data), batch_size):
        index = list(range(start, min(len(data), start + batch_size)))
        payload = {
            "support": data.support[index],
            "candidate": data.candidate[index],
            "mask": data.mask[index],
            "gold": data.gold[index],
            "family": [data.family[i] for i in index],
        }
        for key, value in data.extra.items():
            payload[key] = value[index]
        batch = move_to_device(payload, device)
        scores = model(batch)
        scores = scores.masked_fill(~batch["mask"], -1e9)
        choice = scores.argmax(dim=-1)
        rows = torch.arange(choice.shape[0], device=device)
        hit = batch["gold"][rows, choice]
        totals["n"] += int(choice.shape[0])
        totals["top1"] += float(hit.sum())
        for family, correct in zip(batch["family"], hit.tolist()):
            by_family.setdefault(family, []).append(float(correct))
    n = max(totals["n"], 1)
    return {
        "n": totals["n"],
        "top1": totals["top1"] / n,
        "oracle": 1.0 if totals["n"] else 0.0,
        "by_family": {name: sum(rows) / len(rows) for name, rows in sorted(by_family.items())},
    }


def renaming_invariance(encoder, model: PriorModel | None = None, n: int = 32, seed: int = 2507) -> dict:
    """Letter-prefix tasks stay the same task after a within-class rename.

    Digits and lexicon tokens are semantic endpoints, so they are not renamed.
    """
    rng = random.Random(seed)
    feature_hits = 0
    embed_hits = 0
    score_hits = 0
    scored = 0
    for _ in range(n):
        prefix = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
        numbers = ["2024", "2025", "2026", "2024"]
        supports = [prefix + text for text in numbers]
        mapping = {char: char for char in prefix}
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
        left = candidate_features(program, supports)
        right = candidate_features(program, renamed)
        feature_hits += int(left == right)
        with torch.no_grad():
            vectors = embed_supports(encoder, [supports, renamed])
        embed_hits += int(torch.allclose(vectors[0], vectors[1], atol=1e-5))
        if model is not None and model.kind == "mlp":
            feats = torch.tensor([left, right], dtype=torch.float32)
            batch = {
                "support": vectors,
                "candidate": feats[:, None, :],
                "mask": torch.ones(2, 1, dtype=torch.bool),
                "gold": torch.ones(2, 1, dtype=torch.bool),
            }
            scores = model(batch)
            score_hits += int(torch.allclose(scores[0], scores[1], atol=1e-5))
            scored += 1
        if model is not None and model.kind == "ast":
            from prior.tree import pack_candidates

            packed = [pack_candidates([program], group, 1) for group in (supports, renamed)]
            batch = {
                "support": vectors,
                "relation": torch.stack([item["relation"] for item in packed]),
                "kind": torch.stack([item["kind"] for item in packed]),
                "params": torch.stack([item["params"] for item in packed]),
                "left": torch.stack([item["left"] for item in packed]),
                "right": torch.stack([item["right"] for item in packed]),
                "depth": torch.stack([item["depth"] for item in packed]),
                "root": torch.stack([item["root"] for item in packed]),
                "mask": torch.ones(2, 1, dtype=torch.bool),
                "gold": torch.ones(2, 1, dtype=torch.bool),
            }
            scores = model(batch)
            score_hits += int(torch.allclose(scores[0], scores[1], atol=1e-5))
            scored += 1
    return {
        "n": n,
        "features": feature_hits / n,
        "support_embedding": embed_hits / n,
        "scores": (score_hits / scored) if scored else None,
    }
