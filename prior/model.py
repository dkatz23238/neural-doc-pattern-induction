"""Candidate scorer. Device and width come from the config, not from a second architecture."""

from __future__ import annotations

import torch
from torch import nn

from prior.config import TrainConfig
from prior.features import FEATURE_DIM, symbolic_score
from prior.support_encode import SUPPORT_DIM


def _estimate(support_dim: int, feat_dim: int, width: int, hidden: int) -> int:
    proj = support_dim * width + width + feat_dim * width + width
    first = (3 * width) * hidden + hidden
    second = hidden * hidden + hidden
    last = hidden + 1
    return proj + first + second + last


def choose_width(target: int, support_dim: int = SUPPORT_DIM, feat_dim: int = FEATURE_DIM) -> tuple[int, int]:
    """Pick a projection width and MLP width near ``target``.

    Hidden size stays between half and four times the projection width, so a
    larger budget cannot collapse into a bottleneck or a one-layer tower.
    """
    if target <= 0:
        return 0, 0
    best: tuple[int, int, int] | None = None
    widths = (8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768)
    hiddens = (
        8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 4096,
    )
    for width in widths:
        for hidden in hiddens:
            if not (0.5 * width <= hidden <= 4 * width):
                continue
            count = _estimate(support_dim, feat_dim, width, hidden)
            if best is None or abs(count - target) < abs(best[0] - target):
                best = (count, width, hidden)
    assert best is not None
    return best[1], best[2]


class PriorModel(nn.Module):
    """Score every candidate against one support vector.

    ``support`` is [B, S], ``candidate`` is [B, C, F]. The product of the two
    projections is the relation between a support and a candidate. A symbolic
    prior has no parameters and ignores the support vector.
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.kind = cfg.kind
        self.support_dim = SUPPORT_DIM
        self.feat_dim = FEATURE_DIM
        if cfg.kind == "symbolic":
            self.width = 0
            self.hidden = 0
            return
        width, hidden = choose_width(cfg.target_params, self.support_dim, self.feat_dim)
        self.width = width
        self.hidden = hidden
        self.support_proj = nn.Linear(self.support_dim, width)
        self.cand_proj = nn.Linear(self.feat_dim, width)
        self.mlp = nn.Sequential(
            nn.Linear(width * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.kind == "symbolic":
            values = batch["candidate"].detach().cpu()
            rows = []
            for group in values:
                rows.append([symbolic_score(item.tolist()) for item in group])
            return torch.tensor(rows, dtype=batch["candidate"].dtype, device=batch["candidate"].device)
        support = self.support_proj(batch["support"])
        candidate = self.cand_proj(batch["candidate"])
        shared = support[:, None, :].expand_as(candidate)
        packed = torch.cat([shared, candidate, shared * candidate], dim=-1)
        return self.mlp(packed).squeeze(-1)


def trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
