"""Hierarchical group composition on the frozen span trunk.

H1 pools spans into groups and mixes those groups with a transformer.
H2 replaces that mixer with message passing on typed edges: child, parent,
next sibling, same repeated unit, and support/query alignment.

Both models keep the relational character features and the span encoder.
Membership reads the query's sequence root together with its characters.
Auxiliary heads still predict the generator's character labels, and they add
operator identity, operator scope, and the typed relations between groups.
No codepoint channel is added, so a within-class rename cannot move the score.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pattern_induction.groups import (
    MAX_GROUPS,
    N_EDGE,
    N_GFEAT,
    N_OP,
    N_REL_G,
    group_targets,
    propose_groups,
)
from pattern_induction.span_net import (
    SpanNet,
    _attend,
    _encoder,
    _mean_max,
    _pool_spans,
    composition_loss,
    propose_spans,
)

ARCHS = ("h1", "h2")


def _contained_pool(
    span_h: torch.Tensor,
    span_start: torch.Tensor,
    span_end: torch.Tensor,
    span_pad: torch.Tensor,
    group_start: torch.Tensor,
    group_end: torch.Tensor,
    group_pad: torch.Tensor,
) -> torch.Tensor:
    contained = (
        (span_start[:, :, None, :] >= group_start[:, :, :, None])
        & (span_end[:, :, None, :] <= group_end[:, :, :, None])
        & (span_end[:, :, None, :] > span_start[:, :, None, :])
        & ~span_pad[:, :, None, :]
        & ~group_pad[:, :, :, None]
    )
    weights = contained.float()
    denom = weights.sum(dim=-1).clamp(min=1.0)
    pooled = torch.einsum("bsmd,bsgm->bsgd", span_h, weights) / denom.unsqueeze(-1)
    return pooled * (~group_pad).unsqueeze(-1).float()


class GroupNet(SpanNet):
    def __init__(self, arch: str = "h2", group_layers: int = 2):
        if arch not in ARCHS:
            raise ValueError(arch)
        super().__init__()
        self.arch = arch
        self.max_groups = MAX_GROUPS
        self.group_proj = nn.Linear(N_GFEAT, self.d_model)
        self.group_op = nn.Linear(self.d_model, N_OP)
        self.scope_head = nn.Linear(self.d_model, N_OP)
        self.edge_head = nn.Linear(self.d_model * 2, N_EDGE)
        self.drop = nn.Dropout(0.05)
        if arch == "h1":
            self.group_encoder = _encoder(self.d_model, 4, 320, 0.05, group_layers)
        else:
            self.msg = nn.ModuleList(
                nn.ModuleList(nn.Linear(self.d_model, self.d_model) for _ in range(N_REL_G)) for _ in range(group_layers)
            )
            self.upd = nn.ModuleList(nn.Linear(self.d_model * 2, self.d_model) for _ in range(group_layers))
            self.mix_norm = nn.ModuleList(nn.LayerNorm(self.d_model) for _ in range(group_layers))

    def score_pairs(self, groups: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        batch, slots, count, dim = groups.shape
        flat = groups.reshape(batch, slots * count, dim)
        src = src.to(flat.device).clamp(0, slots * count - 1)
        dst = dst.to(flat.device).clamp(0, slots * count - 1)
        src_h = torch.gather(flat, 1, src.unsqueeze(-1).expand(batch, src.shape[1], dim))
        dst_h = torch.gather(flat, 1, dst.unsqueeze(-1).expand(batch, dst.shape[1], dim))
        return self.edge_head(torch.cat([src_h, dst_h], dim=-1))

    def _pass(
        self,
        hidden: torch.Tensor,
        live: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        rel: torch.Tensor,
        edge_pad: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        batch, count, dim = hidden.shape
        edges = src.shape[1]
        src_h = torch.gather(hidden, 1, src.clamp(0, count - 1).unsqueeze(-1).expand(batch, edges, dim))
        msg = torch.zeros(batch, edges, dim, device=hidden.device, dtype=hidden.dtype)
        for rel_id, proj in enumerate(self.msg[layer]):
            chosen = (rel == rel_id) & ~edge_pad
            projected = proj(src_h)
            msg = torch.where(chosen.unsqueeze(-1), projected, msg)
        msg = msg * (~edge_pad).unsqueeze(-1).float()
        acc = torch.zeros_like(hidden)
        acc.scatter_add_(1, dst.clamp(0, count - 1).unsqueeze(-1).expand(batch, edges, dim), msg)
        counted = torch.zeros(batch, count, 1, device=hidden.device, dtype=hidden.dtype)
        counted.scatter_add_(1, dst.clamp(0, count - 1).unsqueeze(-1), (~edge_pad).unsqueeze(-1).float())
        mean = acc / counted.clamp(min=1.0)
        updated = self.upd[layer](torch.cat([hidden, mean], dim=-1))
        mixed = self.mix_norm[layer](hidden + self.drop(updated))
        return mixed * live.unsqueeze(-1).float()

    def forward(
        self,
        local: torch.Tensor,
        classes: torch.Tensor,
        rel: torch.Tensor,
        pad: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _batch, slots, length = local.shape
        roles = torch.zeros(slots, dtype=torch.long, device=local.device)
        roles[-1] = 1
        which = torch.arange(slots, device=local.device)
        hidden = self.class_emb(classes) + self.local_emb(local) + self.rel_proj(rel)
        hidden = hidden + self.char_pos[:length][None, None, :, :]
        hidden = hidden + self.role(roles)[None, :, None, :] + self.which(which)[None, :, None, :]
        hidden = hidden * (~pad).unsqueeze(-1).float()
        encoded = _attend(self.char_encoder, hidden, pad)

        span_start, span_end, span_feat, span_pad, pos_index = propose_spans(classes, pad, self.max_spans)
        pooled = _pool_spans(encoded, span_start, span_end, span_pad)
        span_h = pooled + self.span_proj(span_feat) + self.span_pos[pos_index]
        span_h = span_h * (~span_pad).unsqueeze(-1).float()
        span_h = _attend(self.span_encoder, span_h, span_pad)

        groups = propose_groups(classes, rel, pad, self.max_groups)
        group_h = _contained_pool(span_h, span_start, span_end, span_pad, groups["start"], groups["end"], groups["pad"])
        group_h = group_h + self.group_proj(groups["feat"])
        group_h = group_h * (~groups["pad"]).unsqueeze(-1).float()
        if self.arch == "h1":
            group_h = _attend(self.group_encoder, group_h, groups["pad"])
        else:
            batch, _slots, count, dim = group_h.shape
            flat = group_h.reshape(batch, _slots * count, dim)
            live = ~groups["pad"].reshape(batch, _slots * count)
            for layer in range(len(self.upd)):
                flat = self._pass(
                    flat,
                    live,
                    groups["edge_src"],
                    groups["edge_dst"],
                    groups["edge_rel"],
                    groups["edge_pad"],
                    layer,
                )
            group_h = flat.reshape(batch, _slots, count, dim)

        char_mean, char_max = _mean_max(encoded[:, -1], pad[:, -1])
        root = group_h[:, -1, 0, :]
        group_mean, _group_max = _mean_max(group_h[:, -1], groups["pad"][:, -1])
        latent = torch.cat([char_mean, char_max, root, group_mean], dim=-1)
        aux = {
            "kind": self.kind_head(encoded),
            "span": self.span_head(encoded),
            "bound": self.bound_head(encoded),
            "role": self.role_head(encoded),
            "scope": self.scope_head(encoded),
            "group_op": self.group_op(group_h),
            "groups": group_h,
        }
        return self.mem(latent).squeeze(-1), aux


def _ce(logit: torch.Tensor, target: torch.Tensor, classes: int) -> torch.Tensor:
    if not bool((target >= 0).any()):
        return logit.new_zeros(())
    return F.cross_entropy(logit.reshape(-1, classes), target.reshape(-1), ignore_index=-1)


def hierarchy_loss(
    logit: torch.Tensor,
    aux: dict[str, torch.Tensor],
    labels: torch.Tensor,
    span_id: torch.Tensor,
    kind: torch.Tensor,
    bound: torch.Tensor,
    role: torch.Tensor,
    mask: torch.Tensor,
    targets: dict[str, torch.Tensor],
    edge_logit: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same membership loss as the span model, plus operator and relation targets."""
    base, membership, aux_loss = composition_loss(logit, aux, labels, span_id, kind, bound, role, mask)
    op_target = targets["op"].to(logit.device)
    scope_target = targets["scope"].to(logit.device)
    pair_target = targets["pair_label"].to(logit.device)
    extra = (
        0.3 * _ce(aux["group_op"], op_target, N_OP)
        + 0.2 * _ce(aux["scope"], scope_target, N_OP)
        + 0.2 * _ce(edge_logit, pair_target, N_EDGE)
    )
    return base + extra, membership, (aux_loss + extra).detach()


def batch_targets(episodes, classes, rel, pad, max_len: int) -> dict[str, torch.Tensor]:
    return group_targets(episodes, classes, rel, pad, max_len)
