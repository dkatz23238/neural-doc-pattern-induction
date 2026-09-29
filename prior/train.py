"""Train the candidate prior. CPU and GPU share this loop.

    device = torch.device(cfg.device)
    model = PriorModel(cfg).to(device)
    for batch in loader:
        batch = move_to_device(batch, device)
        scores = model(batch)
        loss = ranking_loss(scores, batch["gold"], batch["mask"])
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from prior.config import TrainConfig, load_config
from prior.dataset import load_split, make_loader
from prior.evaluate import evaluate, move_to_device, renaming_invariance
from prior.loss import factor_loss, ranking_loss
from prior.model import PriorModel, trainable_parameters
from prior.support_encode import load_frozen_encoder


def _device(cfg: TrainConfig) -> torch.device:
    requested = cfg.device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"device {requested} was requested and CUDA is not available")
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if requested == "cuda" and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return torch.device("cuda", local)
    return torch.device(requested)


def _distributed(device: torch.device) -> bool:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return False
    backend = "nccl" if device.type == "cuda" else "gloo"
    torch.distributed.init_process_group(backend=backend)
    return True


def run(cfg: TrainConfig, root: Path | None = None) -> dict:
    root = root or Path(__file__).resolve().parents[1]
    torch.manual_seed(cfg.seed)
    device = _device(cfg)
    distributed = _distributed(device)
    data_dir = root / cfg.data_dir
    train_data = load_split(data_dir, "train", cfg.max_train, cfg.kind) if cfg.kind != "symbolic" else None
    val_data = load_split(data_dir, "val", kind=cfg.kind)
    ood_data = load_split(data_dir, "ood", kind=cfg.kind)
    model = PriorModel(cfg).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    params = trainable_parameters(raw)
    precision = cfg.precision if device.type == "cuda" or cfg.precision == "bf16" else "fp32"
    optimizer = None
    if params:
        optimizer = torch.optim.AdamW(raw.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = None
    if train_data is not None:
        loader = make_loader(train_data, cfg.batch_size, True, cfg.num_workers, cfg.seed)
    out = cfg.path(root)
    out.mkdir(parents=True, exist_ok=True)
    best = {"val_top1": -1.0, "step": 0}
    history = []
    started = time.perf_counter()
    step = 0
    if loader is not None and optimizer is not None:
        while step < cfg.steps:
            for batch in loader:
                step += 1
                batch = move_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                    if cfg.kind == "factor":
                        scores, aux_logits = model(batch, aux=True)
                        loss = factor_loss(scores, aux_logits, batch["gold"], batch["mask"], batch["aux"])
                    else:
                        scores = model(batch)
                        loss = ranking_loss(scores, batch["gold"], batch["mask"])
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(raw.parameters(), cfg.grad_clip)
                optimizer.step()
                if step % cfg.eval_every == 0 or step >= cfg.steps:
                    val = evaluate(raw, val_data, device)
                    ood = evaluate(raw, ood_data, device)
                    row = {"step": step, "loss": float(loss.detach()), "val_top1": val["top1"], "ood_top1": ood["top1"]}
                    history.append(row)
                    print(
                        f"{cfg.name} seed {cfg.seed} step {step} loss {row['loss']:.4f} "
                        f"val {val['top1']:.3f} ood {ood['top1']:.3f}",
                        flush=True,
                    )
                    if val["top1"] >= best["val_top1"]:
                        best = {"val_top1": val["top1"], "step": step, "ood_top1": ood["top1"]}
                        torch.save({"model": raw.state_dict(), "config": cfg.__dict__, "step": step}, out / "checkpoint.pt")
                if step >= cfg.steps:
                    break
    val = evaluate(raw, val_data, device)
    ood = evaluate(raw, ood_data, device)
    if val["top1"] >= best["val_top1"]:
        best = {"val_top1": val["top1"], "step": step, "ood_top1": ood["top1"]}
        if params:
            torch.save({"model": raw.state_dict(), "config": cfg.__dict__, "step": step}, out / "checkpoint.pt")
    invariance = None
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        encoder = load_frozen_encoder(root)
        invariance = renaming_invariance(encoder, raw if raw.kind in {"mlp", "ast", "factor"} else None)
    payload = {
        "name": cfg.name,
        "kind": cfg.kind,
        "seed": cfg.seed,
        "device": str(device),
        "precision": precision,
        "target_params": cfg.target_params,
        "params": params,
        "width": raw.width,
        "hidden": raw.hidden,
        "tree_dim": raw.tree_dim,
        "rank": getattr(raw, "rank", 0),
        "steps": step,
        "seconds": time.perf_counter() - started,
        "val": val,
        "ood": ood,
        "best": best,
        "history": history,
        "renaming": invariance,
        "trained": bool(params),
        "bench_e_scored": False,
        "bench_c_selected": False,
    }
    if rank == 0:
        (out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    if distributed:
        torch.distributed.destroy_process_group()
    return payload


def main(argv: list[str] | None = None) -> None:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    config_path = None
    if "--config" in args:
        slot = args.index("--config")
        config_path = args[slot + 1]
        del args[slot : slot + 2]
    flags: list[str] = []
    bare: list[str] = []
    index = 0
    while index < len(args):
        item = args[index]
        if item in {"--device", "--model-size", "--steps", "--batch-size", "--mixed-precision"}:
            flags.append(item.lstrip("-").replace("-", "_") + "=" + args[index + 1])
            index += 2
            continue
        bare.append(item)
        index += 1
    cfg = load_config(config_path, flags + bare)
    payload = run(cfg)
    print(
        f"params {payload['params']} val {payload['val']['top1']:.3f} "
        f"ood {payload['ood']['top1']:.3f} renaming {payload['renaming']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
