"""Local facts read off an alignment of the supports.

A class run is a maximal span of one structural class. When the supports
share one run skeleton, a run that is the same text on every support is a
copy, and a run that varies inside one character class is that class. The
class bounds are the observed widths, including widths above the repeat cap.
A shared literal glued to the front or back of a long varying run is split
off, and the whole-run class stays beside it.

An optional prefix or suffix is a fixed run sequence that some supports
omit, not a separator that happens to be rare. Two unrelated skeletons are
an alternation of the two strings. Three or more skeletons, including a
repeat whose depth changes, stay quiet.
"""

from __future__ import annotations

from dataclasses import dataclass

from pattern_induction.chart import Hyp, class_runs
from pattern_induction.renaming import DIGIT, LOWER, UPPER, structural_class

_CLASS_NAME = {DIGIT: "digit", UPPER: "upper", LOWER: "lower"}


@dataclass(frozen=True)
class _Run:
    cls: int
    start: int
    end: int
    text: str


def alignment_kind(supports: list[str]) -> str:
    """Which alignment, if any, these supports license."""
    kind, _rows = _prepare(supports)
    return kind


def document_facts(supports: list[str]) -> list[Hyp]:
    """Copy, class, optional, and alternation facts entailed by the alignment."""
    kind, rows = _prepare(supports)
    if kind == "equal":
        return _dedup(_aligned_columns(list(enumerate(rows))))
    if kind == "optional_prefix":
        return _dedup(_optional(rows, supports, prefix=True))
    if kind == "optional_suffix":
        return _dedup(_optional(rows, supports, prefix=False))
    if kind == "two_cluster":
        return _dedup(_two_cluster(rows, supports))
    return []


def _prepare(supports: list[str]) -> tuple[str, list[list[_Run]]]:
    if len(supports) < 2 or any(not text for text in supports):
        return "none", []
    rows = [_runs(text) for text in supports]
    if any(not row for row in rows):
        return "none", []
    sigs = [_signature(row) for row in rows]
    unique = list(dict.fromkeys(sigs))
    if len(unique) == 1:
        return "equal", rows
    if len(unique) != 2:
        return "none", rows
    short, long = sorted(unique, key=len)
    if long[len(long) - len(short) :] == short and len(long) > len(short):
        return "optional_prefix", rows
    if long[: len(short)] == short and len(long) > len(short):
        return "optional_suffix", rows
    return "two_cluster", rows


def _runs(text: str) -> list[_Run]:
    found = []
    for start, end in class_runs(text):
        found.append(_Run(structural_class(text[start]), start, end, text[start:end]))
    return found


def _signature(row: list[_Run]) -> tuple[int, ...]:
    return tuple(run.cls for run in row)


def _optional(rows: list[list[_Run]], supports: list[str], *, prefix: bool) -> list[Hyp]:
    sigs = [_signature(row) for row in rows]
    short, long = sorted(set(sigs), key=len)
    extra = len(long) - len(short)
    hyps: list[Hyp] = []
    core: list[tuple[int, list[_Run]]] = []
    holders: list[tuple[int, list[_Run]]] = []
    spans: list[tuple[int, int, int]] = []
    for slot, row in enumerate(rows):
        if sigs[slot] == long:
            if prefix:
                head, tail = row[:extra], row[extra:]
                spans.append((slot, 0, head[-1].end))
            else:
                head, tail = row[: len(short)], row[len(short) :]
                spans.append((slot, tail[0].start, len(supports[slot])))
            holders.append((slot, head if prefix else tail))
            core.append((slot, tail if prefix else head))
        else:
            core.append((slot, row))
    hyps.extend(_aligned_columns(core))
    hyps.extend(_aligned_columns(holders))
    for slot, start, end in spans:
        if end > start:
            hyps.append(Hyp("optional", slot, start, end, score=0.0))
    texts: dict[str, tuple[int, int, int]] = {}
    for slot, start, end in spans:
        texts.setdefault(supports[slot][start:end], (slot, start, end))
    if len(texts) == 2:
        (slot, start, end), (slot2, start2, end2) = texts.values()
        hyps.append(Hyp("copy", slot, start, end, score=0.0))
        hyps.append(Hyp("copy", slot2, start2, end2, score=0.0))
        hyps.append(Hyp("alt", slot, start, end, score=0.0, slot2=slot2, start2=start2, end2=end2))
    return hyps


def _two_cluster(rows: list[list[_Run]], supports: list[str]) -> list[Hyp]:
    groups: dict[tuple[int, ...], list[int]] = {}
    for slot, row in enumerate(rows):
        groups.setdefault(_signature(row), []).append(slot)
    if len(groups) != 2:
        return []
    hyps: list[Hyp] = []
    reps = []
    for slots in groups.values():
        hyps.extend(_aligned_columns([(slot, rows[slot]) for slot in slots]))
        reps.append(slots[0])
    left, right = reps
    hyps.append(
        Hyp(
            "alt",
            left,
            0,
            len(supports[left]),
            score=0.0,
            slot2=right,
            start2=0,
            end2=len(supports[right]),
        )
    )
    return hyps


def _aligned_columns(groups: list[tuple[int, list[_Run]]]) -> list[Hyp]:
    if len(groups) < 1:
        return []
    width = len(groups[0][1])
    if any(len(row) != width for _slot, row in groups):
        return []
    hyps: list[Hyp] = []
    for index in range(width):
        hyps.extend(_column([(slot, row[index]) for slot, row in groups]))
    return hyps


def _column(cells: list[tuple[int, _Run]]) -> list[Hyp]:
    if len(cells) < 2:
        slot, run = cells[0]
        return [Hyp("copy", slot, run.start, run.end, score=0.0)]
    texts = [run.text for _slot, run in cells]
    if len(set(texts)) == 1:
        return [Hyp("copy", slot, run.start, run.end, score=0.0) for slot, run in cells]
    classes = {run.cls for _slot, run in cells}
    if len(classes) != 1:
        return []
    name = _CLASS_NAME.get(next(iter(classes)))
    hyps: list[Hyp] = []
    if name is not None:
        widths = [len(text) for text in texts]
        lo, hi = min(widths), max(widths)
        for slot, run in cells:
            hyps.append(Hyp("cls", slot, run.start, run.end, score=0.0, cls=name, lo=lo, hi=hi))
        if lo == hi:
            hyps.extend(_literal_edges(cells, name))
        elif len(set(widths)) == 2:
            hyps.extend(_width_alternation(cells, name))
    distinct: dict[str, tuple[int, _Run]] = {}
    for slot, run in cells:
        distinct.setdefault(run.text, (slot, run))
    if len(distinct) == 2:
        (slot, run), (slot2, run2) = distinct.values()
        hyps.append(Hyp("copy", slot, run.start, run.end, score=0.0))
        hyps.append(Hyp("copy", slot2, run2.start, run2.end, score=0.0))
        hyps.append(
            Hyp(
                "alt",
                slot,
                run.start,
                run.end,
                score=0.0,
                slot2=slot2,
                start2=run2.start,
                end2=run2.end,
            )
        )
    return hyps


def _literal_edges(cells: list[tuple[int, _Run]], name: str) -> list[Hyp]:
    """Every constant head or tail of a class run, beside the whole-run class.

    The longest shared head is not the only cut. A short literal such as the
    leading ``3`` of a five-digit run stays available when the samples also
    share a longer head.
    """
    texts = [run.text for _slot, run in cells]
    width = len(texts[0])
    if any(len(text) != width for text in texts):
        return []
    pref = _common_prefix(texts)
    suff = _common_suffix(texts)
    hyps: list[Hyp] = []
    cuts: set[tuple[int, int]] = set()
    for k in range(0, len(pref) + 1):
        for m in range(0, len(suff) + 1):
            if k == 0 and m == 0:
                continue
            if k + m >= width:
                continue
            if not (k >= 2 or m >= 2 or width >= 5):
                continue
            mids = [text[k : len(text) - m] if m else text[k:] for text in texts]
            if any(not mid for mid in mids) or len(set(mids)) < 2:
                continue
            cuts.add((k, m))
    for k, m in cuts:
        mid_width = width - k - m
        for slot, run in cells:
            cursor = run.start
            if k:
                hyps.append(Hyp("copy", slot, cursor, cursor + k, score=0.0))
                cursor += k
            mid_end = run.end - m
            hyps.append(Hyp("cls", slot, cursor, mid_end, score=0.0, cls=name, lo=mid_width, hi=mid_width))
            if m:
                hyps.append(Hyp("copy", slot, mid_end, run.end, score=0.0))
    return hyps


def _width_alternation(cells: list[tuple[int, _Run]], name: str) -> list[Hyp]:
    """Two observed widths are each an exact class, and an alternation of the two."""
    exemplars: dict[int, tuple[int, _Run]] = {}
    hyps: list[Hyp] = []
    for slot, run in cells:
        width = len(run.text)
        exemplars.setdefault(width, (slot, run))
        hyps.append(Hyp("cls", slot, run.start, run.end, score=0.0, cls=name, lo=width, hi=width))
    if len(exemplars) != 2:
        return hyps
    (slot, run), (slot2, run2) = exemplars.values()
    hyps.append(
        Hyp(
            "alt",
            slot,
            run.start,
            run.end,
            score=0.0,
            slot2=slot2,
            start2=run2.start,
            end2=run2.end,
        )
    )
    return hyps


def _common_prefix(texts: list[str]) -> str:
    prefix = texts[0]
    for text in texts[1:]:
        while prefix and not text.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


def _common_suffix(texts: list[str]) -> str:
    suffix = texts[0]
    for text in texts[1:]:
        while suffix and not text.endswith(suffix):
            suffix = suffix[:-1]
    return suffix


def _dedup(hyps: list[Hyp]) -> list[Hyp]:
    seen = set()
    rows = []
    for hyp in hyps:
        key = (
            hyp.kind,
            hyp.slot,
            hyp.start,
            hyp.end,
            hyp.cls,
            hyp.lo,
            hyp.hi,
            hyp.slot2,
            hyp.start2,
            hyp.end2,
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(hyp)
    return rows
