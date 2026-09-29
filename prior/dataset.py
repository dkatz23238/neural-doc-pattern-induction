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
    def __init__(
        self,
        support: torch.Tensor,
        candidate: torch.Tensor,
        mask: torch.Tensor,
        gold: torch.Tensor,
        family: list[str],
        extra: dict[str, torch.Tensor] | None = None,
    ):
        self.support = support
        self.candidate = candidate
        self.mask = mask
        self.gold = gold
        self.family = family
        self.extra = extra or {}

    def __len__(self) -> int:
        return int(self.support.shape[0])

    def __getitem__(self, index: int) -> dict:
        row = {
            "support": self.support[index],
            "candidate": self.candidate[index],
            "mask": self.mask[index],
            "gold": self.gold[index],
            "family": self.family[index],
        }
        for key, value in self.extra.items():
            row[key] = value[index]
        return row


def _collate(rows: list[dict]) -> dict:
    batch = {
        "support": torch.stack([row["support"] for row in rows]),
        "candidate": torch.stack([row["candidate"] for row in rows]),
        "mask": torch.stack([row["mask"] for row in rows]),
        "gold": torch.stack([row["gold"] for row in rows]),
        "family": [row["family"] for row in rows],
    }
    for key in rows[0]:
        if key in batch or not torch.is_tensor(rows[0][key]):
            continue
        batch[key] = torch.stack([row[key] for row in rows])
    return batch


def _pack_trees(programs: list[str], supports: list[list[str]]) -> dict[str, torch.Tensor]:
    from prior.tree import MAX_NODES, PARAM_DIM, REL_DIM, pack_candidates, programs_from_json

    count = len(programs)
    kind = torch.zeros(count, MAX_CANDIDATES, MAX_NODES, dtype=torch.long)
    params = torch.zeros(count, MAX_CANDIDATES, MAX_NODES, PARAM_DIM)
    left = torch.full((count, MAX_CANDIDATES, MAX_NODES), -1, dtype=torch.long)
    right = torch.full((count, MAX_CANDIDATES, MAX_NODES), -1, dtype=torch.long)
    depth = torch.full((count, MAX_CANDIDATES, MAX_NODES), -1, dtype=torch.long)
    root = torch.zeros(count, MAX_CANDIDATES, dtype=torch.long)
    relation = torch.zeros(count, MAX_CANDIDATES, REL_DIM)
    for index, (text, support) in enumerate(zip(programs, supports)):
        packed = pack_candidates(programs_from_json(text), support, MAX_CANDIDATES)
        kind[index] = packed["kind"]
        params[index] = packed["params"]
        left[index] = packed["left"]
        right[index] = packed["right"]
        depth[index] = packed["depth"]
        root[index] = packed["root"]
        relation[index] = packed["relation"]
    return {
        "kind": kind,
        "params": params,
        "left": left,
        "right": right,
        "depth": depth,
        "root": root,
        "relation": relation,
    }


def load_split(data_dir: Path, split: str, limit: int | None = None, kind: str = "mlp") -> PriorDataset:
    paths = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no {split} shards in {data_dir}")
    columns = ["family", "support", "mask", "gold"]
    if kind == "ast":
        columns.extend(["programs", "supports"])
    else:
        columns.append("candidate")
    table = pq.read_table(paths, columns=columns)
    if limit is not None:
        table = table.slice(0, limit)
    support = torch.tensor(table.column("support").to_pylist(), dtype=torch.float32)
    mask = torch.tensor(table.column("mask").to_pylist(), dtype=torch.bool)
    gold = torch.tensor(table.column("gold").to_pylist(), dtype=torch.bool)
    family = table.column("family").to_pylist()
    extra: dict[str, torch.Tensor] = {}
    if kind == "ast":
        if support.shape[-1] != SUPPORT_DIM:
            raise RuntimeError(f"{split} support width {support.shape[-1]} is not {SUPPORT_DIM}")
        candidate = torch.zeros(support.shape[0], MAX_CANDIDATES, FEATURE_DIM)
        extra = _pack_trees(table.column("programs").to_pylist(), table.column("supports").to_pylist())
    else:
        candidate = torch.tensor(table.column("candidate").to_pylist(), dtype=torch.float32)
        if support.shape[-1] != SUPPORT_DIM or candidate.shape[-1] != MAX_CANDIDATES * FEATURE_DIM:
            raise RuntimeError(
                f"{split} shard shape {tuple(support.shape)} {tuple(candidate.shape)} "
                f"does not match support {SUPPORT_DIM} and {MAX_CANDIDATES}*{FEATURE_DIM}"
            )
        candidate = candidate.view(-1, MAX_CANDIDATES, FEATURE_DIM)
    return PriorDataset(support, candidate, mask, gold, family, extra)


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
