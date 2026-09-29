"""Shards of precomputed support vectors and candidate features.

Training does not synthesize regexes. CPU and GPU read the same files.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from prior.features import FEATURE_DIM
from prior.support_encode import SUPPORT_DIM

MAX_CANDIDATES = 32


class PriorDataset(Dataset):
    def __init__(self, support: torch.Tensor, candidate: torch.Tensor, mask: torch.Tensor, gold: torch.Tensor, family: list[str]):
        self.support = support
        self.candidate = candidate
        self.mask = mask
        self.gold = gold
        self.family = family

    def __len__(self) -> int:
        return int(self.support.shape[0])

    def __getitem__(self, index: int) -> dict:
        return {
            "support": self.support[index],
            "candidate": self.candidate[index],
            "mask": self.mask[index],
            "gold": self.gold[index],
            "family": self.family[index],
        }


def _collate(rows: list[dict]) -> dict:
    return {
        "support": torch.stack([row["support"] for row in rows]),
        "candidate": torch.stack([row["candidate"] for row in rows]),
        "mask": torch.stack([row["mask"] for row in rows]),
        "gold": torch.stack([row["gold"] for row in rows]),
        "family": [row["family"] for row in rows],
    }


def load_split(data_dir: Path, split: str, limit: int | None = None) -> PriorDataset:
    paths = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no {split} shards in {data_dir}")
    table = pq.read_table(paths)
    if limit is not None:
        table = table.slice(0, limit)
    support = torch.tensor(table.column("support").to_pylist(), dtype=torch.float32)
    candidate = torch.tensor(table.column("candidate").to_pylist(), dtype=torch.float32)
    mask = torch.tensor(table.column("mask").to_pylist(), dtype=torch.bool)
    gold = torch.tensor(table.column("gold").to_pylist(), dtype=torch.bool)
    family = table.column("family").to_pylist()
    if support.shape[-1] != SUPPORT_DIM or candidate.shape[-1] != MAX_CANDIDATES * FEATURE_DIM:
        raise RuntimeError(
            f"{split} shard shape {tuple(support.shape)} {tuple(candidate.shape)} "
            f"does not match support {SUPPORT_DIM} and {MAX_CANDIDATES}*{FEATURE_DIM}"
        )
    candidate = candidate.view(-1, MAX_CANDIDATES, FEATURE_DIM)
    return PriorDataset(support, candidate, mask, gold, family)


def make_loader(data: PriorDataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
        generator=generator,
        drop_last=False,
    )
