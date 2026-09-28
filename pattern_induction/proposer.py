"""Grammar decoder over the relational span encoder.

The decoder proposes a program. It does not decide whether a string is a
member. Copy arguments are support spans, so the character itself stays in
the executor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pattern_induction.program import (
    CLS_NAME,
    MAX_LEN,
    MAX_PROG,
    N_OP,
    OP_ALT,
    OP_CLS,
    OP_COPY,
    OP_NAME,
    OP_OPT,
    OP_REPEAT,
    OP_SEQ,
    OP_STOP,
    Prog,
    Task,
    delinearize,
    feature_batch,
    label_string,
    linearize,
)
from pattern_induction.span_net import SpanNet, _attend, _pool_spans, propose_spans

ARG_DIM = 32


def _pack_steps(tasks: list[Task]) -> dict[str, torch.Tensor]:
    batch = len(tasks)
    fields = ("op", "slot", "start", "end", "cls", "lo", "hi", "nchild")
    steps = {name: torch.zeros(batch, MAX_PROG, dtype=torch.long) for name in fields}
    mask = torch.zeros(batch, MAX_PROG, dtype=torch.bool)
    copy_mask = torch.zeros(batch, MAX_PROG, dtype=torch.bool)
    cls_mask = torch.zeros(batch, MAX_PROG, dtype=torch.bool)
    bound_mask = torch.zeros(batch, MAX_PROG, dtype=torch.bool)
    seq_mask = torch.zeros(batch, MAX_PROG, dtype=torch.bool)
    for row, task in enumerate(tasks):
        tokens = linearize(task.program)[:MAX_PROG]
        for index, token in enumerate(tokens):
            for name in fields:
                steps[name][row, index] = int(token[name])
            mask[row, index] = True
            op = int(token["op"])
            copy_mask[row, index] = op == OP_COPY
            cls_mask[row, index] = op == OP_CLS
            bound_mask[row, index] = op in {OP_CLS, OP_REPEAT}
            seq_mask[row, index] = op == OP_SEQ
    return {"steps": steps, "mask": mask, "copy_mask": copy_mask, "cls_mask": cls_mask, "bound_mask": bound_mask, "seq_mask": seq_mask}


def _pack_labels(tasks: list[Task], slots: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    role = torch.full((len(tasks), slots, length), -1, dtype=torch.long)
    scope = torch.full((len(tasks), slots, length), -1, dtype=torch.long)
    for row, task in enumerate(tasks):
        for slot, text in enumerate(task.supports[: slots - 1]):
            labels = label_string(task.program, text[:length], task.supports)
            if labels is None:
                continue
            for index, (role_id, scope_id) in enumerate(labels[:length]):
                role[row, slot, index] = role_id
                scope[row, slot, index] = scope_id
    return role, scope


def collate_tasks(tasks: list[Task]) -> dict[str, torch.Tensor]:
    local, classes, rel, pad = feature_batch(tasks)
    packed = _pack_steps(tasks)
    role, scope = _pack_labels(tasks, local.shape[1], local.shape[2])
    packed.update({"local": local, "classes": classes, "rel": rel, "pad": pad, "role": role, "scope": scope})
    return packed


class Proposer(SpanNet):
    """Relational spans in, a grammar-constrained program out."""

    def __init__(self):
        super().__init__()
        dim = self.d_model
        self.mem = nn.Identity()
        self.kind_head = nn.Identity()
        self.span_head = nn.Identity()
        self.bound_head = nn.Identity()
        self.role_head = nn.Identity()
        self.bos = nn.Parameter(torch.zeros(dim))
        self.op_emb = nn.Embedding(N_OP, dim)
        self.slot_emb = nn.Embedding(4, ARG_DIM)
        self.start_emb = nn.Embedding(MAX_LEN, ARG_DIM)
        self.end_emb = nn.Embedding(MAX_LEN, ARG_DIM)
        self.cls_emb = nn.Embedding(len(CLS_NAME), ARG_DIM)
        self.lo_emb = nn.Embedding(4, ARG_DIM)
        self.hi_emb = nn.Embedding(8, ARG_DIM)
        self.nchild_emb = nn.Embedding(3, ARG_DIM)
        self.arg_proj = nn.Linear(ARG_DIM * 7, dim)
        self.cell = nn.GRUCell(dim * 2, dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.mix = nn.LayerNorm(dim)
        self.op_head = nn.Linear(dim, N_OP)
        self.slot_head = nn.Linear(dim, 4)
        self.start_head = nn.Linear(dim, MAX_LEN)
        self.end_head = nn.Linear(dim, MAX_LEN)
        self.cls_head = nn.Linear(dim, len(CLS_NAME))
        self.lo_head = nn.Linear(dim, 4)
        self.hi_head = nn.Linear(dim, 8)
        self.nchild_head = nn.Linear(dim, 3)
        self.char_role = nn.Linear(dim, 5)
        self.char_scope = nn.Linear(dim, 4)

    def encode(self, local, classes, rel, pad):
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
        # The last slot repeats the first support so featurize has a query. It is not a program input.
        char_mem = encoded[:, :-1].reshape(encoded.shape[0], -1, encoded.shape[-1])
        char_pad = pad[:, :-1].reshape(pad.shape[0], -1)
        span_mem = span_h[:, :-1].reshape(span_h.shape[0], -1, span_h.shape[-1])
        span_key = span_pad[:, :-1].reshape(span_pad.shape[0], -1)
        memory = torch.cat([char_mem, span_mem], dim=1)
        key_pad = torch.cat([char_pad, span_key], dim=1)
        return encoded, memory, key_pad

    def _args(self, slot, start, end, cls, lo, hi, nchild) -> torch.Tensor:
        pieces = [
            self.slot_emb(slot),
            self.start_emb(start.clamp(0, MAX_LEN - 1)),
            self.end_emb(end.clamp(0, MAX_LEN - 1)),
            self.cls_emb(cls),
            self.lo_emb(lo),
            self.hi_emb(hi),
            self.nchild_emb(nchild),
        ]
        return self.arg_proj(torch.cat(pieces, dim=-1))

    def _step(self, hidden, memory, key_pad, op_vec, arg_vec):
        hidden = self.cell(torch.cat([op_vec, arg_vec], dim=-1), hidden)
        context, _weights = self.attn(hidden[:, None, :], memory, memory, key_padding_mask=key_pad, need_weights=False)
        hidden = self.mix(hidden + context.squeeze(1))
        return hidden, {
            "op": self.op_head(hidden),
            "slot": self.slot_head(hidden),
            "start": self.start_head(hidden),
            "end": self.end_head(hidden),
            "cls": self.cls_head(hidden),
            "lo": self.lo_head(hidden),
            "hi": self.hi_head(hidden),
            "nchild": self.nchild_head(hidden),
        }

    def decode(self, memory, key_pad, steps: dict[str, torch.Tensor] | None, length: int) -> dict[str, torch.Tensor]:
        batch = memory.shape[0]
        hidden = memory.new_zeros(batch, self.d_model)
        op_vec = self.bos.expand(batch, -1)
        arg_vec = memory.new_zeros(batch, self.d_model)
        rows = {name: [] for name in ("op", "slot", "start", "end", "cls", "lo", "hi", "nchild")}
        for index in range(length):
            hidden, logits = self._step(hidden, memory, key_pad, op_vec, arg_vec)
            for name in rows:
                rows[name].append(logits[name])
            if steps is None:
                chosen = {name: logits[name].argmax(dim=-1) for name in rows}
            else:
                chosen = {name: steps[name][:, index] for name in rows}
            op_vec = self.op_emb(chosen["op"])
            arg_vec = self._args(chosen["slot"], chosen["start"], chosen["end"], chosen["cls"], chosen["lo"], chosen["hi"], chosen["nchild"])
        return {name: torch.stack(rows[name], dim=1) for name in rows}

    def forward(self, local, classes, rel, pad, steps) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        encoded, memory, key_pad = self.encode(local, classes, rel, pad)
        logits = self.decode(memory, key_pad, steps, steps["op"].shape[1])
        aux = {"role": self.char_role(encoded), "scope": self.char_scope(encoded)}
        return logits, aux

    def logprob(self, local, classes, rel, pad, steps) -> torch.Tensor:
        _encoded, memory, key_pad = self.encode(local, classes, rel, pad)
        logits = self.decode(memory, key_pad, steps, steps["op"].shape[1])
        token = F.log_softmax(logits["op"], dim=-1)
        chosen = token.gather(-1, steps["op"].unsqueeze(-1)).squeeze(-1)
        mask = steps["op"].new_ones(steps["op"].shape, dtype=torch.bool)
        # Score through the stop token and ignore padding after it.
        seen = torch.zeros(steps["op"].shape[0], dtype=torch.bool, device=steps["op"].device)
        weights = []
        for index in range(steps["op"].shape[1]):
            weights.append(~seen)
            seen = seen | (steps["op"][:, index] == OP_STOP)
        weight = torch.stack(weights, dim=1).float()
        return (chosen * weight).sum(dim=-1)

    @torch.no_grad()
    def propose(self, local, classes, rel, pad, k: int = 16) -> list[list[tuple[Prog, float]]]:
        """Beam of complete programs, best model score first.

        The search varies the opcode, the copied span, the class, the repeat
        bounds, and the sequence arity. A copy that falls outside its support
        is dropped before it can grow.
        """
        self.eval()
        _encoded, memory, key_pad = self.encode(local, classes, rel, pad)
        lengths = (~pad[:, :-1]).sum(dim=-1)
        results = []
        for row in range(local.shape[0]):
            results.append(_beam_row(self, memory[row : row + 1], key_pad[row : row + 1], lengths[row], k))
        return results


def _safe_key(key_pad: torch.Tensor) -> torch.Tensor:
    safe = key_pad.clone()
    empty = safe.all(dim=1)
    safe[empty, 0] = False
    return safe


def _beam_row(model: Proposer, memory, key_pad, lengths, k: int) -> list[tuple[Prog, float]]:
    key_pad = _safe_key(key_pad)
    hidden0 = memory.new_zeros(1, model.d_model)
    # steps, logp, stack, hidden, prev op vector, prev arg vector
    # steps, opcode logp, argument logp, stack, hidden, prev op vector, prev arg vector.
    # Admission uses the mean opcode log-probability first, so a flat span head cannot
    # push every terminal out of the beam. The returned score is the mean full log-probability.
    beams = [([], 0.0, 0.0, (1,), hidden0, model.bos.view(1, -1), memory.new_zeros(1, model.d_model))]
    finished: list[tuple[list[dict], float, float]] = []
    device = memory.device

    def _admit(item) -> tuple[float, float]:
        count = max(len(item[0]), 1)
        return (item[1] / count, item[2] / count)

    for _ in range(MAX_PROG):
        expanded = []
        for steps, op_logp, arg_logp, stack, hidden, op_vec, arg_vec in beams:
            if not stack:
                finished.append((steps, op_logp, arg_logp))
                continue
            hidden, logits = model._step(hidden, memory, key_pad, op_vec, arg_vec)
            log_ops = F.log_softmax(logits["op"], dim=-1)[0]
            for choice, extra in _arg_variants(logits, lengths):
                op = int(choice["op"])
                nxt_stack = _push(stack, op, int(choice["nchild"]))
                # A branch that cannot close before the program limit can never be executed.
                if len(steps) + 1 + sum(nxt_stack) + 1 > MAX_PROG:
                    continue
                expanded.append(
                    (
                        steps + [choice],
                        op_logp + float(log_ops[op]),
                        arg_logp + extra,
                        nxt_stack,
                        hidden,
                        model.op_emb(torch.tensor([op], device=device)),
                        model._args(
                            torch.tensor([choice["slot"]], device=device),
                            torch.tensor([choice["start"]], device=device),
                            torch.tensor([choice["end"]], device=device),
                            torch.tensor([choice["cls"]], device=device),
                            torch.tensor([choice["lo"]], device=device),
                            torch.tensor([choice["hi"]], device=device),
                            torch.tensor([choice["nchild"]], device=device),
                        ),
                    )
                )
        expanded.sort(key=_admit, reverse=True)
        beams = expanded[:k]
        if not beams:
            break
    for steps, op_logp, arg_logp, stack, _hidden, _op, _arg in beams:
        if not stack:
            finished.append((steps, op_logp, arg_logp))
    finished.sort(key=lambda item: (item[1] + item[2]) / max(len(item[0]), 1), reverse=True)
    programs = []
    seen = set()
    for steps, op_logp, arg_logp in finished:
        closed = steps + [{"op": OP_STOP, "slot": 0, "start": 0, "end": 0, "cls": 0, "lo": 0, "hi": 0, "nchild": 0}]
        prog = delinearize(closed)
        if prog is None:
            continue
        key = tuple((step["op"], step["slot"], step["start"], step["end"], step["cls"], step["lo"], step["hi"], step["nchild"]) for step in closed)
        if key in seen:
            continue
        seen.add(key)
        programs.append((prog, (op_logp + arg_logp) / max(len(steps), 1)))
        if len(programs) >= k:
            break
    return programs


def _copy_fits(choice: dict, lengths) -> bool:
    slot = int(choice["slot"])
    if slot < 0 or slot >= int(lengths.shape[0]):
        return False
    limit = int(lengths[slot])
    start = int(choice["start"])
    end = int(choice["end"])
    return limit > 0 and 0 <= start <= end < limit


def _arg_variants(logits: dict[str, torch.Tensor], lengths) -> list[tuple[dict, float]]:
    """Opcode arguments the beam is willing to try, with their log-probability."""
    base = _greedy_args(logits)
    log_slot = F.log_softmax(logits["slot"][0], dim=-1)
    log_start = F.log_softmax(logits["start"][0], dim=-1)
    log_end = F.log_softmax(logits["end"][0], dim=-1)
    log_cls = F.log_softmax(logits["cls"][0], dim=-1)
    log_lo = F.log_softmax(logits["lo"][0], dim=-1)
    log_hi = F.log_softmax(logits["hi"][0], dim=-1)
    log_nchild = F.log_softmax(logits["nchild"][0], dim=-1)
    variants: list[tuple[dict, float]] = []

    def emit(op: int, updates: dict, extra: float) -> None:
        choice = dict(base)
        choice["op"] = op
        choice.update(updates)
        if op == OP_COPY and not _copy_fits(choice, lengths):
            return
        variants.append((choice, extra))

    slots = torch.topk(log_slot, k=2).indices.tolist()
    starts = torch.topk(log_start, k=2).indices.tolist()
    for slot in slots:
        for start in starts:
            end_scores = log_end.clone()
            end_scores[: int(start)] = -1e9
            for end in torch.topk(end_scores, k=2).indices.tolist():
                if int(end) < int(start):
                    continue
                extra = float(log_slot[slot]) + float(log_start[start]) + float(log_end[end])
                emit(OP_COPY, {"slot": int(slot), "start": int(start), "end": int(end)}, extra)

    lo = int(base["lo"])
    hi_scores = log_hi.clone()
    hi_scores[:lo] = -1e9
    class_his = [int(index) for index in torch.topk(hi_scores, k=2).indices.tolist() if int(index) >= lo]
    for cls in torch.topk(log_cls, k=2).indices.tolist():
        for hi in class_his:
            extra = float(log_cls[cls]) + float(log_lo[lo]) + float(log_hi[hi])
            emit(OP_CLS, {"cls": int(cls), "lo": lo, "hi": hi}, extra)

    for nchild in range(3):
        emit(OP_SEQ, {"nchild": nchild}, float(log_nchild[nchild]))

    for lo_index in torch.topk(log_lo, k=2).indices.tolist():
        hi_scores = log_hi.clone()
        hi_scores[: int(lo_index)] = -1e9
        for hi in torch.topk(hi_scores, k=2).indices.tolist():
            if int(hi) < int(lo_index):
                continue
            extra = float(log_lo[lo_index]) + float(log_hi[hi])
            emit(OP_REPEAT, {"lo": int(lo_index), "hi": int(hi)}, extra)

    emit(OP_OPT, {}, 0.0)
    emit(OP_ALT, {}, 0.0)
    return variants


def _greedy_args(logits: dict[str, torch.Tensor]) -> dict:
    start = int(logits["start"][0].argmax())
    end_scores = logits["end"][0].clone()
    end_scores[:start] = -1e9
    end = int(end_scores.argmax())
    if end < start:
        end = min(start, MAX_LEN - 1)
    lo = int(logits["lo"][0].argmax())
    hi_scores = logits["hi"][0].clone()
    hi_scores[:lo] = -1e9
    hi = int(hi_scores.argmax())
    if hi < lo:
        hi = lo
    return {
        "slot": int(logits["slot"][0].argmax()),
        "start": start,
        "end": end,
        "cls": int(logits["cls"][0].argmax()),
        "lo": lo,
        "hi": min(hi, 7),
        "nchild": int(logits["nchild"][0].argmax()),
    }


def _push(stack: tuple[int, ...], op: int, nchild_index: int) -> tuple[int, ...]:
    values = list(stack)
    values[-1] -= 1
    if values[-1] == 0:
        values.pop()
    if op == OP_SEQ:
        values.append(nchild_index + 2)
    elif op in {OP_REPEAT, OP_OPT}:
        values.append(1)
    elif op == OP_ALT:
        values.append(2)
    return tuple(values)


def token_logprob(logits, steps) -> torch.Tensor:
    token = F.log_softmax(logits["op"], dim=-1)
    chosen = token.gather(-1, steps["op"].unsqueeze(-1)).squeeze(-1)
    seen = torch.zeros(steps["op"].shape[0], dtype=torch.bool, device=steps["op"].device)
    weight = []
    for index in range(steps["op"].shape[1]):
        weight.append(~seen)
        seen = seen | (steps["op"][:, index] == OP_STOP)
    return (chosen * torch.stack(weight, dim=1).float()).sum(dim=-1)


def program_loss(logits, aux, batch, gold_lp: torch.Tensor | None = None, corrupt_lp: torch.Tensor | None = None, corrupt_mask: torch.Tensor | None = None):
    op = F.cross_entropy(logits["op"][batch["mask"]], batch["steps"]["op"][batch["mask"]])
    copy = _masked(logits, batch, "slot", "copy_mask") + _masked(logits, batch, "start", "copy_mask") + _masked(logits, batch, "end", "copy_mask")
    cls = _masked(logits, batch, "cls", "cls_mask")
    bounds = _masked(logits, batch, "lo", "bound_mask") + _masked(logits, batch, "hi", "bound_mask")
    nchild = _masked(logits, batch, "nchild", "seq_mask")
    if bool((batch["role"] >= 0).any()):
        role = F.cross_entropy(aux["role"].reshape(-1, 5), batch["role"].reshape(-1), ignore_index=-1)
        scope = F.cross_entropy(aux["scope"].reshape(-1, 4), batch["scope"].reshape(-1), ignore_index=-1)
    else:
        role = op.new_zeros(())
        scope = op.new_zeros(())
    if gold_lp is not None and corrupt_lp is not None and corrupt_mask is not None and bool(corrupt_mask.any()):
        rank = F.softplus(corrupt_lp[corrupt_mask] - gold_lp[corrupt_mask]).mean()
    else:
        rank = op.new_zeros(())
    total = op + 0.5 * copy + 0.5 * cls + 0.5 * bounds + 0.3 * nchild + 0.4 * role + 0.3 * scope + 0.2 * rank
    return total, op.detach(), rank.detach()


def _masked(logits, batch, name: str, mask_name: str) -> torch.Tensor:
    mask = batch[mask_name]
    if not bool(mask.any()):
        return logits["op"].new_zeros(())
    target = batch["steps"][name][mask]
    return F.cross_entropy(logits[name][mask], target)
