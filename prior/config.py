"""One configuration for every device and every model size."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


SIZES = {
    "0": 0,
    "10k": 10_000,
    "30k": 30_000,
    "100k": 100_000,
    "300k": 300_000,
    "1m": 1_000_000,
    "3m": 3_000_000,
    "5m": 5_000_000,
}


@dataclass
class TrainConfig:
    name: str = "prior_100k"
    kind: str = "mlp"
    device: str = "cpu"
    precision: str = "fp32"
    target_params: int = 100_000
    steps: int = 20_000
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    seed: int = 0
    data_dir: str = "results/prior/shards"
    checkpoint_dir: str = "results/prior/runs"
    num_workers: int = 0
    eval_every: int = 2_000
    grad_clip: float = 1.0
    max_train: int | None = None

    def path(self, root: Path) -> Path:
        return root / self.checkpoint_dir / self.name / f"seed_{self.seed}"


def load_config(path: str | None, overrides: list[str]) -> TrainConfig:
    raw: dict = {}
    if path:
        raw.update(yaml.safe_load(Path(path).read_text()) or {})
    data = {field.name: raw[field.name] for field in fields(TrainConfig) if field.name in raw}
    cfg = TrainConfig(**data)
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"override must be key=value, got {item}")
        key, value = item.split("=", 1)
        key = key.lstrip("-").replace("-", "_")
        if key == "model_size":
            if value not in SIZES:
                raise SystemExit(f"unknown model size {value}")
            cfg.target_params = SIZES[value]
            cfg.kind = "symbolic" if SIZES[value] == 0 else "mlp"
            cfg.name = f"prior_{value}"
            continue
        if key == "mixed_precision":
            cfg.precision = value
            continue
        if not hasattr(cfg, key):
            raise SystemExit(f"unknown config field {key}")
        current = getattr(cfg, key)
        if current is None or isinstance(current, int):
            setattr(cfg, key, int(value) if value not in {"none", "null"} else None)
        elif isinstance(current, float):
            setattr(cfg, key, float(value))
        else:
            setattr(cfg, key, value)
    if cfg.precision not in {"fp32", "bf16"}:
        raise SystemExit(f"precision must be fp32 or bf16, got {cfg.precision}")
    if cfg.kind not in {"mlp", "symbolic", "ast"}:
        raise SystemExit(f"kind must be mlp, symbolic, or ast, got {cfg.kind}")
    return cfg
