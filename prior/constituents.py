"""Constituent state from the frozen recognizer, before the global pool.

Each support keeps its class runs in order. A slot stores the first three
runs and the last run. A run carries its class, its length, whether it
repeats the previous run, the alignment relations on its characters, and,
when the encoder is asked, the frozen span hidden for that run.

Nothing here is mean-pooled across the string. A missing run stays zero, so
a linear probe can see which constituents are present on which support.
"""

from __future__ import annotations

from pathlib import Path

import torch

from pattern_induction.chart import spans_from_ids
from pattern_induction.local_net import LocalNet, _span_tensors
from pattern_induction.membership import Episode
from pattern_induction.program import MAX_LEN, MAX_SUPPORT
from pattern_induction.renaming import DIGIT, LOWER, N_CLASS, N_REL, UPPER, featurize
from prior.support_encode import SUPPORT_DIM, load_frozen_encoder

N_SLOTS = MAX_SUPPORT
N_POS = 4
RUN_FEAT = N_CLASS + 1 + 4 + N_REL
STRUCTURE_DIM = N_SLOTS * N_POS * RUN_FEAT
FIELD_FEAT = 5
FIELD_DIM = N_SLOTS * N_POS * FIELD_FEAT
NEURAL_DIM = N_SLOTS * N_POS * SUPPORT_DIM


def _runs(classes: list[int], pad: list[bool]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    index = 0
    while index < len(classes) and not pad[index]:
        stop = index + 1
        kind = int(classes[index])
        while stop < len(classes) and not pad[stop] and int(classes[stop]) == kind:
            stop += 1
        runs.append((index, stop, kind))
        index = stop
    return runs


def _positions(runs: list[tuple[int, int, int]]) -> list[tuple[int, int, int] | None]:
    """First three class runs, then the last run. Missing positions stay empty."""
    chosen: list[tuple[int, int, int] | None] = []
    for cursor in range(3):
        chosen.append(runs[cursor] if cursor < len(runs) else None)
    chosen.append(runs[-1] if runs else None)
    return chosen


def _same(left: tuple[int, int, int] | None, right: tuple[int, int, int]) -> tuple[float, float]:
    if left is None:
        return 0.0, 0.0
    return (
        float(left[2] == right[2]),
        float((left[1] - left[0]) == (right[1] - right[0])),
    )


def _run_features(
    run: tuple[int, int, int] | None,
    previous: tuple[int, int, int] | None,
    back: tuple[int, int, int] | None,
    rel: list[list[float]],
) -> list[float]:
    if run is None:
        return [0.0] * RUN_FEAT
    start, end, kind = run
    one_hot = [0.0] * N_CLASS
    if 0 <= kind < N_CLASS:
        one_hot[kind] = 1.0
    length = (end - start) / float(MAX_LEN)
    same_class, same_len = _same(previous, run)
    back_class, back_len = _same(back, run)
    if end <= start:
        mean_rel = [0.0] * N_REL
    else:
        cols = rel[start:end]
        mean_rel = [sum(col[axis] for col in cols) / float(len(cols)) for axis in range(N_REL)]
    return [*one_hot, length, same_class, same_len, back_class, back_len, *mean_rel]


def structure_row(classes: list[list[int]], pad: list[list[bool]], rel: list[list[list[float]]]) -> list[float]:
    values: list[float] = []
    for slot in range(N_SLOTS):
        runs = _runs(classes[slot], pad[slot]) if slot < len(classes) else []
        positions = _positions(runs)
        for cursor, run in enumerate(positions):
            if run is None:
                previous = back = None
            elif cursor < 3:
                origin = cursor
                previous = runs[origin - 1] if origin else None
                back = runs[origin - 2] if origin >= 2 else None
            else:
                origin = len(runs) - 1
                previous = runs[origin - 1] if origin else None
                back = runs[origin - 2] if origin >= 2 else None
            slot_rel = rel[slot] if slot < len(rel) else []
            values.extend(_run_features(run, previous, back, slot_rel))
    return values


def _column_constancy(text: str, peers: list[str]) -> float:
    """Fraction of this run's columns that agree across same-width peers."""
    width = len(text)
    if width == 0:
        return 0.0
    same = [peer for peer in peers if len(peer) == width]
    if len(same) < 2:
        return 0.0
    hits = 0
    for index in range(width):
        if len({peer[index] for peer in same}) == 1:
            hits += 1
    return hits / float(width)


def field_relation_row(supports: list[str]) -> list[float]:
    """Within-run alignment. Global left/right constancy does not see inside a repeat."""
    episodes = [
        Episode(list(supports[:MAX_SUPPORT]), supports[0] if supports else "", 1, "prior", "prior", "prior")
    ]
    _local, classes, _rel, pad = featurize(episodes, max_len=MAX_LEN, max_support=MAX_SUPPORT)
    cls_list = classes[0, :-1].tolist()
    pad_list = pad[0, :-1].tolist()
    slot_runs: list[list[tuple[int, int, int, str]]] = []
    for slot in range(N_SLOTS):
        text = supports[slot] if slot < len(supports) else ""
        runs = []
        for start, end, kind in _runs(cls_list[slot], pad_list[slot]):
            runs.append((start, end, kind, text[start:end]))
        slot_runs.append(runs)
    peers: dict[tuple[int, int], list[str]] = {}
    for runs in slot_runs:
        for _start, _end, kind, text in runs:
            peers.setdefault((kind, len(text)), []).append(text)
    values: list[float] = []
    for slot in range(N_SLOTS):
        runs = slot_runs[slot]
        positions = _positions([(start, end, kind) for start, end, kind, _text in runs])
        by_span = {(start, end, kind): text for start, end, kind, text in runs}
        for run in positions:
            if run is None:
                values.extend([0.0] * FIELD_FEAT)
                continue
            text = by_span.get(run, "")
            group = peers.get((run[2], len(text)), [])
            constancy = _column_constancy(text, group)
            count = min(len(group), 16) / 16.0
            kind = run[2]
            values.extend(
                [
                    constancy,
                    count,
                    constancy if kind == DIGIT else 0.0,
                    constancy if kind == UPPER else 0.0,
                    constancy if kind == LOWER else 0.0,
                ]
            )
    return values


def class_blind(structure: torch.Tensor) -> torch.Tensor:
    """Drop character-class identity. Role and alignment stay."""
    blinded = structure.clone()
    for slot in range(N_SLOTS):
        for pos in range(N_POS):
            start = (slot * N_POS + pos) * RUN_FEAT
            blinded[:, start : start + N_CLASS] = 0
    return blinded


def digit_constancy(field: torch.Tensor) -> torch.Tensor:
    """Digit-gated within-field constancy at each kept run."""
    grouped = field.view(field.shape[0], N_SLOTS * N_POS, FIELD_FEAT)
    return grouped[:, :, 2]


def repeat_summary(structure: torch.Tensor) -> torch.Tensor:
    """Whether a later run repeats the class and the length two runs back."""
    grouped = structure.view(structure.shape[0], N_SLOTS, N_POS, RUN_FEAT)
    eq_class = grouped[:, :, :, N_CLASS + 1 + 2].amax(dim=(1, 2))
    eq_len = grouped[:, :, :, N_CLASS + 1 + 3].amax(dim=(1, 2))
    return torch.stack([eq_class, eq_len], dim=-1)


def run_presence(structure: torch.Tensor) -> torch.Tensor:
    """Which kept runs exist. Optional scope is a run that some supports omit."""
    grouped = structure.view(structure.shape[0], N_SLOTS, N_POS, RUN_FEAT)
    present = (grouped[:, :, :, :N_CLASS].sum(dim=-1) > 0).to(dtype=structure.dtype)
    return present.reshape(structure.shape[0], -1)


def field_class(structure: torch.Tensor) -> torch.Tensor:
    """Classes that occur on any kept run. Position is not part of this vector."""
    grouped = structure.view(structure.shape[0], N_SLOTS, N_POS, RUN_FEAT)
    return grouped[:, :, :, :N_CLASS].amax(dim=(1, 2))


def suffix_role(structure: torch.Tensor) -> torch.Tensor:
    """Last-run length, repeat flags, and alignment. Class identity is left out."""
    grouped = structure.view(structure.shape[0], N_SLOTS, N_POS, RUN_FEAT)
    return grouped[:, :, -1, N_CLASS:].reshape(structure.shape[0], -1)


def last_right_constant(structure: torch.Tensor) -> torch.Tensor:
    """Right-constancy of the last class run on each support."""
    grouped = structure.view(structure.shape[0], N_SLOTS, N_POS, RUN_FEAT)
    axis = N_CLASS + 1 + 4 + 7
    return grouped[:, :, -1, axis]


def _digit_runs(text: str) -> list[str]:
    runs: list[str] = []
    index = 0
    while index < len(text):
        if text[index].isdigit():
            stop = index + 1
            while stop < len(text) and text[stop].isdigit():
                stop += 1
            runs.append(text[index:stop])
            index = stop
        else:
            index += 1
    return runs


def numeric_coverage_row(supports: list[str]) -> float:
    """Smallest observed range of a digit constituent, as a fraction of that width.

    Column identity is not this. A tight window and a repeated copy of that
    window both have a small fraction. A digit class spread across its width
    does not. A support with no digit run is the full fraction.
    """
    peers: dict[int, list[int]] = {}
    for text in supports:
        for run in _digit_runs(text):
            peers.setdefault(len(run), []).append(int(run))
    best = 1.0
    found = False
    for width, values in peers.items():
        if len(values) < 2:
            continue
        found = True
        floor = 0 if width == 1 else 10 ** (width - 1)
        full = max(10**width - floor, 1)
        span = (max(values) - min(values) + 1) / float(full)
        if span < best:
            best = span
    return best if found else 1.0


def numeric_coverage(groups: list[list[str]]) -> torch.Tensor:
    values = [numeric_coverage_row(group) for group in groups]
    if not values:
        return torch.zeros(0, 1)
    return torch.tensor(values, dtype=torch.float32).unsqueeze(-1)


def field_matrix(groups: list[list[str]]) -> torch.Tensor:
    rows = [field_relation_row(group) for group in groups]
    if not rows:
        return torch.zeros(0, FIELD_DIM)
    return torch.tensor(rows, dtype=torch.float32)


def structure_matrix(classes: torch.Tensor, pad: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
    """Typed run features for a featurized batch. The query slot is ignored."""
    cls_list = classes[:, :-1].detach().tolist()
    pad_list = pad[:, :-1].detach().tolist()
    rel_list = rel[:, :-1].detach().tolist()
    rows = [structure_row(cls_list[row], pad_list[row], rel_list[row]) for row in range(classes.shape[0])]
    return torch.tensor(rows, dtype=torch.float32)


def _run_span_index(catalog_slot: list[tuple[int, int]], runs: list[tuple[int, int, int]]) -> list[int | None]:
    """Index of each kept run inside the recognizer's span list. Runs are listed first."""
    by_span = {span: index for index, span in enumerate(catalog_slot)}
    indexes: list[int | None] = []
    positions = _positions(runs)
    for run in positions:
        if run is None:
            indexes.append(None)
            continue
        indexes.append(by_span.get((run[0], run[1])))
    return indexes


def neural_from_hidden(
    hidden: torch.Tensor,
    classes: torch.Tensor,
    pad: torch.Tensor,
    catalog: list[list[list[tuple[int, int]]]],
) -> torch.Tensor:
    """Span hiddens for the same run positions. Shape [B, NEURAL_DIM]."""
    batch = hidden.shape[0]
    out = torch.zeros(batch, NEURAL_DIM)
    cls_list = classes[:, :-1].detach().tolist()
    pad_list = pad[:, :-1].detach().tolist()
    for row in range(batch):
        cursor = 0
        for slot in range(N_SLOTS):
            runs = _runs(cls_list[row][slot], pad_list[row][slot])
            indexes = _run_span_index(catalog[row][slot], runs)
            for span_index in indexes:
                if span_index is not None and span_index < hidden.shape[2]:
                    out[row, cursor : cursor + SUPPORT_DIM] = hidden[row, slot, span_index].detach().to(dtype=torch.float32)
                cursor += SUPPORT_DIM
    return out


@torch.no_grad()
def encode_groups(encoder: LocalNet, groups: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Structure features and frozen span hiddens for one batch of support sets."""
    encoder.eval()
    episodes = [
        Episode(list(group[:MAX_SUPPORT]), group[0] if group else "", 1, "prior", "prior", "prior")
        for group in groups
    ]
    local, classes, rel, pad = featurize(episodes, max_len=MAX_LEN, max_support=MAX_SUPPORT)
    start, end, span_pad, catalog = _span_tensors(classes, pad)
    hidden = encoder.encode(local, classes, rel, pad, start, end, span_pad)
    structure = structure_matrix(classes, pad, rel)
    neural = neural_from_hidden(hidden, classes, pad, catalog)
    return structure, neural


def encode_split(encoder: LocalNet, groups: list[list[str]], batch_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    structures = []
    neurals = []
    for start in range(0, len(groups), batch_size):
        structure, neural = encode_groups(encoder, groups[start : start + batch_size])
        structures.append(structure)
        neurals.append(neural)
        if start and start % 2048 == 0:
            print(f"encoded {start}/{len(groups)}", flush=True)
    if not structures:
        return torch.zeros(0, STRUCTURE_DIM), torch.zeros(0, NEURAL_DIM)
    return torch.cat(structures, dim=0), torch.cat(neurals, dim=0)


def load_encoder(root: Path) -> LocalNet:
    encoder = load_frozen_encoder(root)
    encoder.eval()
    return encoder
