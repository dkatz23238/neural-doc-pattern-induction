"""Candidate scorer. Device and width come from the config, not from a second architecture."""

from __future__ import annotations

import torch
from torch import nn

from prior.config import TrainConfig
from prior.factors import FEAT_DIM, FIELD, N_AUX, N_FACTORS, OPTIONAL as FACTOR_OPTIONAL, REPEAT as FACTOR_REPEAT
from prior.features import FEATURE_DIM, symbolic_score
from prior.support_encode import SUPPORT_DIM
from prior.tree import ALT, CHARSET, LEXICON, LEX_NAMES, LITERAL, OPTIONAL, PARAM_DIM, RANGE, REL_DIM, REPEAT, SEQ


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
            self.tree_dim = 0
            return
        if cfg.kind == "ast":
            width, hidden, tree_dim = choose_ast(cfg.target_params)
            self.width = width
            self.hidden = hidden
            self.tree_dim = tree_dim
            self.encoder = TreeEncoder(tree_dim=tree_dim)
            self.support_proj = nn.Linear(self.support_dim, width)
            self.cand_proj = nn.Linear(tree_dim + REL_DIM, width)
            self.mlp = nn.Sequential(
                nn.Linear(width * 3, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            return
        if cfg.kind == "factor":
            hidden = choose_factor(cfg.target_params)
            self.width = hidden
            self.hidden = hidden
            self.tree_dim = 0
            self.factor = FactorEncoder(hidden)
            return
        width, hidden = choose_width(cfg.target_params, self.support_dim, self.feat_dim)
        self.width = width
        self.hidden = hidden
        self.tree_dim = 0
        self.support_proj = nn.Linear(self.support_dim, width)
        self.cand_proj = nn.Linear(self.feat_dim, width)
        self.mlp = nn.Sequential(
            nn.Linear(width * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor], aux: bool = False):
        if self.kind == "factor":
            return self.factor(batch["support"], batch["feat"], batch["present"], aux=aux)
        if self.kind == "symbolic":
            values = batch["candidate"].detach().cpu()
            rows = []
            for group in values:
                rows.append([symbolic_score(item.tolist()) for item in group])
            return torch.tensor(rows, dtype=batch["candidate"].dtype, device=batch["candidate"].device)
        if self.kind == "ast":
            tree = self.encoder(
                batch["kind"],
                batch["params"],
                batch["left"],
                batch["right"],
                batch["depth"],
                batch["root"],
            )
            candidate = self.cand_proj(torch.cat([tree, batch["relation"]], dim=-1))
        else:
            candidate = self.cand_proj(batch["candidate"])
        support = self.support_proj(batch["support"])
        shared = support[:, None, :].expand_as(candidate)
        packed = torch.cat([shared, candidate, shared * candidate], dim=-1)
        return self.mlp(packed).squeeze(-1)


class FactorEncoder(nn.Module):
    """Sum of local preference scores.

    Each factor has its own projection. Repeat features do not include the
    child, and field features do not include the quantifier. The interaction
    is ``tanh(beta) * cosine`` of the two hidden states, detached, so it stays
    inside [-1, 1] and cannot rotate those states into a private pair code.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.support = nn.Linear(SUPPORT_DIM, hidden)
        self.proj = nn.ModuleList(nn.Linear(FEAT_DIM, hidden) for _ in range(N_FACTORS))
        self.aux_h = nn.Parameter(torch.randn(N_AUX, hidden) * 0.02)
        self.aux_z = nn.Parameter(torch.randn(N_AUX, hidden) * 0.02)
        self.aux_bias = nn.Parameter(torch.zeros(N_AUX))
        self.beta_repeat = nn.Parameter(torch.zeros(()))
        self.beta_optional = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        support: torch.Tensor,
        feat: torch.Tensor,
        present: torch.Tensor,
        aux: bool = False,
    ):
        z = torch.nn.functional.gelu(self.support(support))
        present_f = present.to(dtype=feat.dtype).unsqueeze(-1)
        hiddens = []
        total = None
        scale = self.hidden ** 0.5
        for index, layer in enumerate(self.proj):
            hidden = torch.nn.functional.gelu(layer(feat[..., index, :])) * present_f[..., index, :]
            score = (z[:, None, :] * hidden).sum(dim=-1) / scale
            total = score if total is None else total + score
            hiddens.append(hidden)
        total = total + self._interaction(hiddens, present, self.beta_repeat, FACTOR_REPEAT, FIELD)
        total = total + self._interaction(hiddens, present, self.beta_optional, FACTOR_OPTIONAL, FIELD)
        if not aux:
            return total
        stacked = torch.stack(hiddens[:N_AUX], dim=2)
        logits = (stacked * self.aux_h).sum(dim=-1)
        logits = logits + (z[:, None, None, :] * self.aux_z).sum(dim=-1) + self.aux_bias
        return total, logits

    @staticmethod
    def _interaction(hiddens, present, beta, left: int, right: int) -> torch.Tensor:
        cosine = _cosine(hiddens[left].detach(), hiddens[right].detach())
        gate = present[..., left].to(dtype=cosine.dtype) * present[..., right].to(dtype=cosine.dtype)
        return torch.tanh(beta) * cosine * gate


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    denom = left.norm(dim=-1).clamp_min(1e-6) * right.norm(dim=-1).clamp_min(1e-6)
    return (left * right).sum(dim=-1) / denom


class _Mix(nn.Module):
    """One shared composition. The same module is applied at every matching node."""

    def __init__(self, din: int, dout: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, dout), nn.GELU(), nn.Linear(dout, dout))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class TreeEncoder(nn.Module):
    """Recursive candidate encoder. It scores structure; it does not execute it.

    Leaves embed a literal, a character class, an integer range, or a lexicon.
    Optional, repeat, sequence, and alternation each have one function, reused
    at every node of that operator.
    """

    def __init__(self, node_dim: int = 16, tree_dim: int = 48):
        super().__init__()
        self.node_dim = node_dim
        self.tree_dim = tree_dim
        self.literal = nn.Linear(PARAM_DIM, node_dim)
        self.charset = nn.Linear(PARAM_DIM, node_dim)
        self.range = nn.Linear(PARAM_DIM, node_dim)
        self.lexicon = nn.Embedding(len(LEX_NAMES), node_dim)
        self.kind_embed = nn.Embedding(9, node_dim)
        self.optional = _Mix(node_dim, node_dim)
        self.repeat = _Mix(node_dim + 4, node_dim)
        self.seq = _Mix(node_dim * 2, node_dim)
        self.alt = _Mix(node_dim * 2, node_dim)
        self.root = nn.Linear(node_dim, tree_dim)

    def forward(
        self,
        kind: torch.Tensor,
        params: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        depth: torch.Tensor,
        root: torch.Tensor,
    ) -> torch.Tensor:
        # kind: [B, C, N]. Composition is vectorized across the batch and the candidates.
        batch, cands, nodes = kind.shape
        flat_kind = kind.reshape(batch * cands, nodes)
        flat_params = params.reshape(batch * cands, nodes, PARAM_DIM)
        flat_left = left.reshape(batch * cands, nodes)
        flat_right = right.reshape(batch * cands, nodes)
        flat_depth = depth.reshape(batch * cands, nodes)
        flat_root = root.reshape(batch * cands)
        encoded = self._encode(flat_kind, flat_params, flat_left, flat_right, flat_depth)
        gathered = encoded[torch.arange(encoded.shape[0], device=encoded.device), flat_root.clamp(min=0)]
        return self.root(gathered).view(batch, cands, self.tree_dim)

    def _encode(
        self,
        kind: torch.Tensor,
        params: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        rows, nodes = kind.shape
        hidden = torch.zeros(rows, nodes, self.node_dim, device=kind.device, dtype=params.dtype)
        if int(depth.max()) < 0:
            return hidden
        for level in range(int(depth.max()) + 1):
            at = depth == level
            if not bool(at.any()):
                continue
            index = at.nonzero(as_tuple=False)
            batch_i = index[:, 0]
            node_i = index[:, 1]
            code = kind[batch_i, node_i]
            values = params[batch_i, node_i]
            left_h = hidden[batch_i, flat_clamp(left[batch_i, node_i])]
            right_h = hidden[batch_i, flat_clamp(right[batch_i, node_i])]
            out = torch.zeros(batch_i.shape[0], self.node_dim, device=kind.device, dtype=params.dtype)
            out = _fill(out, code == LITERAL, self.literal(values))
            out = _fill(out, code == CHARSET, self.charset(values))
            out = _fill(out, code == RANGE, self.range(values))
            lex_ids = values[:, 0].long().clamp(0, len(LEX_NAMES) - 1)
            out = _fill(out, code == LEXICON, self.lexicon(lex_ids))
            out = _fill(out, code == OPTIONAL, self.optional(left_h))
            out = _fill(out, code == REPEAT, self.repeat(torch.cat([left_h, values[:, :4]], dim=-1)))
            out = _fill(out, code == SEQ, self.seq(torch.cat([left_h, right_h], dim=-1)))
            out = _fill(out, code == ALT, self.alt(torch.cat([left_h, right_h], dim=-1)))
            out = out + self.kind_embed(code.clamp(0, 8))
            out = torch.where((code > 0).unsqueeze(-1), out, torch.zeros_like(out))
            scattered = torch.zeros_like(hidden)
            scattered = scattered.index_put((batch_i, node_i), out)
            hidden = torch.where(at.unsqueeze(-1), scattered, hidden)
        return hidden


def flat_clamp(index: torch.Tensor) -> torch.Tensor:
    return index.clamp(min=0)


def _fill(base: torch.Tensor, sel: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if not bool(sel.any()):
        return base
    return torch.where(sel.unsqueeze(-1), values, base)


def _mix_params(din: int, dout: int) -> int:
    return din * dout + dout + dout * dout + dout


def ast_parameter_count(width: int, hidden: int, tree_dim: int, node_dim: int = 16) -> int:
    leaves = 3 * (PARAM_DIM * node_dim + node_dim)
    lexicon = len(LEX_NAMES) * node_dim
    kind = 9 * node_dim
    mixes = (
        _mix_params(node_dim, node_dim)
        + _mix_params(node_dim + 4, node_dim)
        + _mix_params(node_dim * 2, node_dim)
        + _mix_params(node_dim * 2, node_dim)
    )
    root = node_dim * tree_dim + tree_dim
    support = SUPPORT_DIM * width + width
    candidate = (tree_dim + REL_DIM) * width + width
    mlp = (3 * width) * hidden + hidden + hidden * hidden + hidden + hidden + 1
    return leaves + lexicon + kind + mixes + root + support + candidate + mlp


def choose_ast(target: int) -> tuple[int, int, int]:
    """Projection width, MLP width, and tree width inside the diagnostic band."""
    if target <= 20_000:
        band = (10_000, 20_000)
    elif target <= 50_000:
        band = (30_000, 50_000)
    else:
        band = (80_000, 100_000)
    best: tuple[int, int, int, int] | None = None
    for tree_dim in (32, 48, 64):
        for width in (16, 24, 32, 40, 48, 64, 80, 96, 112, 128):
            for hidden in (16, 24, 32, 48, 64, 80, 96, 128, 160, 192):
                if not (0.5 * width <= hidden <= 4 * width):
                    continue
                count = ast_parameter_count(width, hidden, tree_dim)
                if not (band[0] <= count <= band[1]):
                    continue
                if best is None or abs(count - target) < abs(best[0] - target):
                    best = (count, width, hidden, tree_dim)
    if best is None:
        raise RuntimeError(f"no AST shape in band {band} for target {target}")
    return best[1], best[2], best[3]


def factor_parameter_count(hidden: int) -> int:
    support = SUPPORT_DIM * hidden + hidden
    projections = N_FACTORS * (FEAT_DIM * hidden + hidden)
    auxiliary = N_AUX * (hidden + hidden + 1)
    return support + projections + auxiliary + 2


def choose_factor(target: int) -> int:
    """Hidden width whose factorized scorer lands near ``target``."""
    if target <= 20_000:
        band = (10_000, 20_000)
    else:
        band = (30_000, 50_000)
    best: tuple[int, int] | None = None
    for hidden in (32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192):
        count = factor_parameter_count(hidden)
        if not (band[0] <= count <= band[1]):
            continue
        if best is None or abs(count - target) < abs(best[0] - target):
            best = (count, hidden)
    if best is None:
        raise RuntimeError(f"no factorized shape in band {band} for target {target}")
    return best[1]


def trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
