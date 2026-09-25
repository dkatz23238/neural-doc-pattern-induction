"""Benchmark task schema and JSONL IO."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Task:
    id: str
    benchmark: str
    domain: str
    variant: str
    visible: list[str]
    hidden_positives: list[str]
    hidden_negatives: list[str]
    gold_regex: str
    latent_schema: str
    split: str = "test"
    n_visible: int = 0
    notes: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.n_visible = len(self.visible)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_obj(obj: dict) -> "Task":
        known = {f for f in Task.__dataclass_fields__}
        payload = {k: v for k, v in obj.items() if k in known}
        return Task(**payload)


def write_jsonl(path: Path, tasks: list[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(task.to_json() + "\n")


def read_jsonl(path: Path) -> list[Task]:
    tasks: list[Task] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(Task.from_obj(json.loads(line)))
    return tasks
