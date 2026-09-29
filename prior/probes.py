"""Tiny readouts on a frozen 160-d support vector.

A linear probe and one hidden layer of width 32 are trained separately for
each fact. They see the train split only. Validation picks the epoch.
Held-out compositions are not used to choose anything.
"""

from __future__ import annotations

import torch
from torch import nn

from prior.probe_labels import PROBE_CLASSES, PROBE_NAMES
from prior.support_encode import SUPPORT_DIM


class LinearProbe(nn.Module):
    def __init__(self, n_out: int):
        super().__init__()
        self.fc = nn.Linear(SUPPORT_DIM, n_out)

    def forward(self, support: torch.Tensor) -> torch.Tensor:
        return self.fc(support)


class TinyProbe(nn.Module):
    def __init__(self, n_out: int, hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(SUPPORT_DIM, hidden)
        self.fc2 = nn.Linear(hidden, n_out)
        self.hidden = hidden

    def forward(self, support: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(support)))


def make_probe(kind: str, n_out: int) -> nn.Module:
    if kind == "linear":
        return LinearProbe(n_out)
    if kind == "tiny":
        return TinyProbe(n_out)
    raise ValueError(f"unknown probe {kind}")


def parameter_count(kind: str, n_out: int) -> int:
    model = make_probe(kind, n_out)
    return sum(param.numel() for param in model.parameters())


def standardize(train: torch.Tensor, *others: torch.Tensor) -> tuple[torch.Tensor, ...]:
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp(min=1e-6)
    scaled = tuple((block - mean) / std for block in (train, *others))
    return scaled


def balanced_accuracy(pred: torch.Tensor, target: torch.Tensor, n_class: int) -> float:
    recalls = []
    for label in range(n_class):
        mask = target == label
        if int(mask.sum()) == 0:
            continue
        recalls.append(float((pred[mask] == label).float().mean()))
    if not recalls:
        return 0.0
    return sum(recalls) / len(recalls)


def train_probe(
    kind: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    n_class: int,
    seed: int,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-2,
) -> tuple[nn.Module, dict]:
    """Minimize class-balanced cross-entropy. Keep the epoch with the best validation balance."""
    torch.manual_seed(seed)
    model = make_probe(kind, n_class)
    counts = torch.bincount(train_y, minlength=n_class).float()
    present = counts > 0
    weight = torch.zeros(n_class)
    weight[present] = counts[present].sum() / (int(present.sum()) * counts[present])
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best = {"balanced": -1.0, "epoch": 0}
    history = []
    order = torch.arange(train_x.shape[0])
    for epoch in range(epochs):
        model.train()
        perm = order[torch.randperm(order.shape[0])]
        total = 0.0
        seen = 0
        for start in range(0, perm.shape[0], batch_size):
            batch = perm[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(train_x[batch]), train_y[batch])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * int(batch.shape[0])
            seen += int(batch.shape[0])
        model.eval()
        with torch.no_grad():
            pred = model(val_x).argmax(dim=-1)
        score = balanced_accuracy(pred, val_y, n_class)
        history.append({"epoch": epoch + 1, "loss": total / max(seen, 1), "val_balanced": score})
        if score >= best["balanced"]:
            best = {"balanced": score, "epoch": epoch + 1}
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best": best, "history_tail": history[-1], "parameters": parameter_count(kind, n_class)}


@torch.no_grad()
def predict(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model(features).argmax(dim=-1)


def class_names(probe: str) -> tuple[str, ...]:
    if probe not in PROBE_NAMES:
        raise KeyError(probe)
    return PROBE_CLASSES[probe]
