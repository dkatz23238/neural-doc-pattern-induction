"""Local structural hypotheses, composed by a chart.

The network scores spans, repeat units, optional spans, alternation pairs,
and sequence cuts independently. Several alternatives stay alive. This chart
turns those facts into programs. The executor remains the membership judge.
"""

from __future__ import annotations

from dataclasses import dataclass

from pattern_induction.program import OPEN, Prog, advance, agrees, ast_id, execute
from pattern_induction.renaming import structural_class

MAX_NGRAM = 3
MAX_CAND = 48
LOCAL_K = 8


@dataclass(frozen=True)
class Hyp:
    kind: str
    slot: int = 0
    start: int = 0
    end: int = 0
    score: float = 1.0
    cls: str = "digit"
    lo: int = 1
    hi: int = 1
    slot2: int = -1
    start2: int = -1
    end2: int = -1
    index: int = -1


@dataclass(frozen=True)
class _Chunk:
    start: int
    end: int
    prog: Prog
    score: float


def class_runs(text: str) -> list[tuple[int, int]]:
    if not text:
        return []
    runs = []
    start = 0
    kind = structural_class(text[0])
    for index, ch in enumerate(text[1:], start=1):
        current = structural_class(ch)
        if current != kind:
            runs.append((start, index))
            start = index
            kind = current
    runs.append((start, len(text)))
    return runs


def spans_from_ids(ids: list[int], cap: int = MAX_CAND) -> list[tuple[int, int]]:
    """Class runs, adjacent run groups, and cuts inside a run.

    A prefix and an alternation branch can share a character class, so the
    boundary between them is not a run edge. Those cuts stay in the inventory.
    """
    if not ids:
        return []
    runs: list[tuple[int, int]] = []
    start = 0
    kind = ids[0]
    for index, current in enumerate(ids[1:], start=1):
        if current != kind:
            runs.append((start, index))
            start = index
            kind = current
    runs.append((start, len(ids)))
    spans: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add(left: int, right: int) -> None:
        if right <= left or (left, right) in seen or len(spans) >= cap:
            return
        seen.add((left, right))
        spans.append((left, right))

    for left, right in runs:
        add(left, right)
    for width in range(2, MAX_NGRAM + 1):
        for cursor in range(len(runs) - width + 1):
            add(runs[cursor][0], runs[cursor + width - 1][1])
    for left, right in runs:
        for cut in range(left + 1, right):
            add(left, cut)
            add(cut, right)
    return spans


def candidate_spans(text: str, cap: int = MAX_CAND) -> list[tuple[int, int]]:
    return spans_from_ids([structural_class(ch) for ch in text], cap)


def _seq(children: list[Prog]) -> Prog:
    if len(children) == 1:
        return children[0]
    return Prog("SEQ", children=children)


def _one(options: list[tuple[int, list]]) -> list[tuple[int, list]]:
    best: dict[int, list] = {}
    for end, events in options:
        if end not in best:
            best[end] = events
    return list(best.items())


def _derive(node: Prog, text: str, supports: list[str], index: int) -> list[tuple[int, list]]:
    if node.op == "COPY":
        literal = supports[node.slot][node.start : node.end] if 0 <= node.slot < len(supports) else ""
        end = index + len(literal)
        if literal and text.startswith(literal, index):
            return [(end, [("copy", index, end)])]
        return []
    if node.op == "CLS":
        found = []
        lo = max(1, min(node.lo, node.hi))
        for width in range(lo, max(lo, node.hi) + 1):
            chunk = text[index : index + width]
            if len(chunk) == width and (
                (node.cls == "digit" and chunk.isdigit())
                or (node.cls == "upper" and chunk.isalpha() and chunk.isupper())
                or (node.cls == "lower" and chunk.isalpha() and chunk.islower())
            ):
                found.append((index + width, [("cls", index, index + width, node.cls, node.lo, node.hi)]))
        return _one(found)
    if node.op == "OPT" and node.children:
        present = []
        for end, events in _derive(node.children[0], text, supports, index):
            if end > index:
                present.append((end, events + [("optional", index, end)]))
        return _one([(index, [])] + present)
    if node.op == "ALT" and len(node.children) >= 2:
        found = []
        for branch, child in enumerate(node.children[:2]):
            for end, events in _derive(child, text, supports, index):
                found.append((end, events + [("alt", index, end, branch)]))
        return _one(found)
    if node.op == "SEQ":
        options = [(index, [])]
        for child_index, child in enumerate(node.children):
            nxt = []
            last = child_index == len(node.children) - 1
            for end, events in options:
                for child_end, child_events in _derive(child, text, supports, end):
                    extra = list(child_events)
                    if not last:
                        extra.append(("cut", child_end))
                    nxt.append((child_end, events + extra))
            options = _one(nxt)
            if not options:
                return []
        return options
    if node.op == "REPEAT" and node.children:
        start = index
        options = [(index, [], index)]
        hi = max(node.lo, min(node.hi, OPEN))
        found = []
        for count in range(1, hi + 1):
            nxt = []
            for end, events, unit_end in options:
                for child_end, child_events in _derive(node.children[0], text, supports, end):
                    if child_end <= end:
                        continue
                    nxt.append((child_end, events + child_events, child_end if count == 1 else unit_end))
            kept: dict[int, tuple[list, int]] = {}
            for end, events, unit_end in nxt:
                if end not in kept:
                    kept[end] = (events, unit_end)
            options = [(end, events, unit_end) for end, (events, unit_end) in kept.items()]
            if count >= node.lo:
                for end, events, unit_end in options:
                    found.append((end, events + [("repeat", start, end, start, unit_end, node.lo, node.hi)]))
            if not options:
                break
        return _one(found)
    return []


def gold_facts(prog: Prog, supports: list[str]) -> list[Hyp]:
    """Every local fact the gold program exposes on the supports."""
    facts: list[Hyp] = []
    alt_left: list[tuple[int, int, int]] = []
    alt_right: list[tuple[int, int, int]] = []
    for slot, text in enumerate(supports):
        if not text or not execute(prog, text, supports):
            continue
        options = [item for item in _derive(prog, text, supports, 0) if item[0] == len(text)]
        if not options:
            continue
        events = options[0][1]
        for event in events:
            kind = event[0]
            if kind == "copy":
                facts.append(Hyp("copy", slot, event[1], event[2]))
            elif kind == "cls":
                facts.append(Hyp("cls", slot, event[1], event[2], cls=event[3], lo=event[4], hi=event[5]))
            elif kind == "repeat":
                facts.append(Hyp("repeat", slot, event[3], event[4], lo=event[5], hi=event[6]))
            elif kind == "optional":
                facts.append(Hyp("optional", slot, event[1], event[2]))
            elif kind == "cut":
                facts.append(Hyp("cut", slot, index=event[1], start=event[1], end=event[1]))
            elif kind == "alt":
                target = alt_left if event[3] == 0 else alt_right
                target.append((slot, event[1], event[2]))
        for left, right, nxt_left, nxt_right in _repeat_pairs_in(prog, text, supports):
            facts.append(Hyp("pair", slot, left, right, slot2=slot, start2=nxt_left, end2=nxt_right))
    if alt_left and alt_right:
        slot, start, end = alt_left[0]
        slot2, start2, end2 = alt_right[0]
        facts.append(Hyp("alt", slot, start, end, slot2=slot2, start2=start2, end2=end2))
    return _unique(facts)


def _repeat_pairs_in(prog: Prog, text: str, supports: list[str]) -> list[tuple[int, int, int, int]]:
    found = []

    def walk(node: Prog, index: int) -> int | None:
        if node.op == "REPEAT" and node.children:
            pos = index
            spans = []
            hi = max(node.lo, min(node.hi, OPEN))
            for _ in range(hi):
                ends = [end for end in advance(node.children[0], text, supports, pos) if end > pos]
                if not ends:
                    break
                end = max(ends)
                spans.append((pos, end))
                pos = end
            for cursor in range(len(spans) - 1):
                found.append((spans[cursor][0], spans[cursor][1], spans[cursor + 1][0], spans[cursor + 1][1]))
            if spans:
                walk(node.children[0], spans[0][0])
            return pos if spans else None
        if node.op == "SEQ":
            cursor = index
            for child in node.children:
                nxt = walk(child, cursor)
                if nxt is None:
                    ends = [end for end in advance(child, text, supports, cursor) if end >= cursor]
                    if not ends:
                        return None
                    nxt = max(ends)
                cursor = nxt
            return cursor
        if node.op == "OPT" and node.children:
            present = walk(node.children[0], index)
            return present if present is not None else index
        if node.op == "ALT":
            for child in node.children:
                if any(end > index for end in advance(child, text, supports, index)):
                    return walk(child, index)
            return None
        ends = [end for end in advance(node, text, supports, index) if end >= index]
        if not ends:
            return None
        return max(ends)

    walk(prog, 0)
    return found


def _unique(facts: list[Hyp]) -> list[Hyp]:
    seen = set()
    rows = []
    for fact in facts:
        key = (fact.kind, fact.slot, fact.start, fact.end, fact.cls, fact.lo, fact.hi, fact.slot2, fact.start2, fact.end2, fact.index)
        if key in seen:
            continue
        seen.add(key)
        rows.append(fact)
    return rows


def _overlaps(left: _Chunk, right: _Chunk) -> bool:
    return not (left.end <= right.start or right.end <= left.start)


def _stitch(
    chunks: list[_Chunk],
    start: int,
    end: int,
    beam: int = 4,
    cuts: dict[int, float] | None = None,
) -> list[tuple[Prog, float]]:
    if start == end:
        return [(Prog("SEQ", children=[]), 0.0)] if False else []
    by_start: dict[int, list[_Chunk]] = {}
    for chunk in chunks:
        if chunk.start < start or chunk.end > end or chunk.end <= chunk.start:
            continue
        by_start.setdefault(chunk.start, []).append(chunk)
    dp: dict[int, list[tuple[tuple[_Chunk, ...], float]]] = {start: [((), 0.0)]}
    seen: set[int] = set()
    while True:
        pending = sorted(pos for pos in dp if pos not in seen and pos <= end)
        if not pending:
            break
        pos = pending[0]
        seen.add(pos)
        rows = sorted(dp[pos], key=lambda item: item[1], reverse=True)[:beam]
        dp[pos] = rows
        if pos == end:
            break
        for chunk in by_start.get(pos, []):
            bonus = cuts.get(chunk.start, 0.0) if cuts and chunk.start != start else 0.0
            for prefix, score in rows:
                dp.setdefault(chunk.end, []).append((prefix + (chunk,), score + chunk.score + bonus))
    finished = []
    for chunks_used, score in dp.get(end, []):
        if not chunks_used:
            continue
        finished.append((_seq([chunk.prog for chunk in chunks_used]), score))
    finished.sort(key=lambda item: item[1], reverse=True)
    return finished[:beam]


def _atoms(slot: int, hyps: list[Hyp]) -> list[_Chunk]:
    chunks = []
    for hyp in hyps:
        if hyp.slot != slot:
            continue
        if hyp.kind == "copy":
            chunks.append(_Chunk(hyp.start, hyp.end, Prog("COPY", slot=slot, start=hyp.start, end=hyp.end), hyp.score))
        elif hyp.kind == "cls":
            chunks.append(_Chunk(hyp.start, hyp.end, Prog("CLS", cls=hyp.cls, lo=hyp.lo, hi=hyp.hi), hyp.score))
    return chunks


def _tile(unit: Prog, text: str, supports: list[str], start: int, hi: int) -> tuple[int, int]:
    pos = start
    count = 0
    limit = max(1, min(hi, OPEN))
    while count < limit:
        ends = [end for end in advance(unit, text, supports, pos) if end > pos]
        if not ends:
            break
        pos = max(ends)
        count += 1
    return count, pos


def _repeat_chunks(
    slot: int,
    text: str,
    supports: list[str],
    atoms: list[_Chunk],
    hyps: list[Hyp],
    cuts: dict[int, float] | None = None,
) -> list[_Chunk]:
    repeats = [hyp for hyp in hyps if hyp.kind == "repeat" and hyp.slot == slot]
    repeats.sort(key=lambda hyp: (hyp.end - hyp.start, hyp.start))
    built: list[_Chunk] = []
    available = list(atoms)
    for hyp in repeats:
        inner = [chunk for chunk in available if chunk.start >= hyp.start and chunk.end <= hyp.end and not any(_overlaps(chunk, other) and chunk.end - chunk.start > other.end - other.start for other in built)]
        # Prefer atoms and already-built shorter repeats that sit inside the unit.
        inner = [chunk for chunk in available + built if hyp.start <= chunk.start and chunk.end <= hyp.end]
        units = _stitch(inner, hyp.start, hyp.end, beam=2, cuts=cuts)
        for unit, score in units:
            count, end = _tile(unit, text, supports, hyp.start, hyp.hi)
            if count < max(1, hyp.lo) or end <= hyp.start:
                continue
            built.append(_Chunk(hyp.start, end, Prog("REPEAT", lo=hyp.lo, hi=hyp.hi, children=[unit]), score + hyp.score))
            break
    return built


def _covering(chunks: list[_Chunk], start: int, end: int, beam: int = 2) -> list[tuple[Prog, float]]:
    return _stitch([chunk for chunk in chunks if chunk.start >= start and chunk.end <= end], start, end, beam)


def compose(supports: list[str], hyps: list[Hyp], limit: int = 8) -> list[tuple[Prog, float]]:
    """Programs entailed by these hypotheses. Ranked by the sum of local scores."""
    programs: list[tuple[Prog, float]] = []
    cells: dict[tuple[int, int, int], list[tuple[Prog, float]]] = {}
    for slot, text in enumerate(supports):
        if not text:
            continue
        atoms = _atoms(slot, hyps)
        for chunk in atoms:
            cells.setdefault((slot, chunk.start, chunk.end), []).append((chunk.prog, chunk.score))
        cuts = {hyp.index: hyp.score for hyp in hyps if hyp.kind == "cut" and hyp.slot == slot}
        repeats = _repeat_chunks(slot, text, supports, atoms, hyps, cuts)
        for chunk in repeats:
            cells.setdefault((slot, chunk.start, chunk.end), []).append((chunk.prog, chunk.score))
        base = atoms + repeats
        programs.extend(_stitch(base, 0, len(text), beam=4, cuts=cuts))
        for hyp in hyps:
            if hyp.kind != "optional" or hyp.slot != slot or hyp.end <= hyp.start:
                continue
            inners = _covering(base, hyp.start, hyp.end, beam=2)
            outside = [chunk for chunk in base if chunk.end <= hyp.start or chunk.start >= hyp.end]
            for inner, score in inners:
                wrapped = _Chunk(hyp.start, hyp.end, Prog("OPT", children=[inner]), score + hyp.score)
                programs.extend(_stitch(outside + [wrapped], 0, len(text), beam=2, cuts=cuts))
        for hyp in hyps:
            if hyp.kind != "repeat" or hyp.slot != slot:
                continue
            region = cells.get((slot, hyp.start, hyp.end))
            # The repeat chunk may be longer than the unit. Find chunks that start at the unit.
            for start, end in list({(chunk.start, chunk.end) for chunk in repeats}):
                if start != hyp.start:
                    continue
                outside = [chunk for chunk in atoms if chunk.end <= start or chunk.start >= end]
                for prog, score in cells.get((slot, start, end), []):
                    if prog.op != "REPEAT":
                        continue
                    programs.extend(_stitch(outside + [_Chunk(start, end, prog, score)], 0, len(text), beam=2, cuts=cuts))
    for hyp in hyps:
        if hyp.kind != "alt":
            continue
        lefts = cells.get((hyp.slot, hyp.start, hyp.end), [])
        rights = cells.get((hyp.slot2, hyp.start2, hyp.end2), [])
        if not lefts or not rights:
            lefts = lefts or _covering(_atoms(hyp.slot, hyps), hyp.start, hyp.end, 1)
            rights = rights or _covering(_atoms(hyp.slot2, hyps), hyp.start2, hyp.end2, 1)
        if not lefts or not rights:
            continue
        left, left_score = lefts[0]
        right, right_score = rights[0]
        branch = Prog("ALT", children=[left, right])
        score = left_score + right_score + hyp.score
        if hyp.start == 0 and hyp.start2 == 0:
            programs.append((branch, score))
            continue
        if hyp.start == hyp.start2 and hyp.start > 0:
            prefixes = cells.get((hyp.slot, 0, hyp.start), [])
            if prefixes:
                prefix, prefix_score = prefixes[0]
                programs.append((_seq([prefix, branch]), score + prefix_score))
    return _dedup(programs, limit)


def _dedup(programs: list[tuple[Prog, float]], limit: int) -> list[tuple[Prog, float]]:
    best: dict[str, tuple[Prog, float]] = {}
    for prog, score in programs:
        if prog.op == "SEQ" and not prog.children:
            continue
        key = _key(prog)
        current = best.get(key)
        if current is None or score > current[1]:
            best[key] = (prog, score)
    ranked = sorted(best.values(), key=lambda item: item[1], reverse=True)
    return ranked[:limit]


_KEYS: dict[int, str] = {}


def _key(prog: Prog) -> str:
    ident = ast_id(prog)
    cached = _KEYS.get(ident)
    if cached is not None:
        return cached
    if prog.op == "COPY":
        text = f"COPY({prog.slot}:{prog.start}-{prog.end})"
    elif prog.op == "CLS":
        text = f"{prog.cls}{{{prog.lo},{prog.hi}}}"
    elif prog.op == "SET":
        text = f"SET[{prog.spec}]{{{prog.lo},{prog.hi}}}"
    elif prog.op == "INT":
        text = f"INT[{prog.spec}]{{{prog.lo},{prog.hi}}}"
    elif prog.op == "REPEAT" and prog.children:
        text = f"REP{prog.lo}-{prog.hi}({_key(prog.children[0])})"
    elif prog.op == "OPT" and prog.children:
        text = f"OPT({_key(prog.children[0])})"
    elif prog.op == "ALT":
        text = "ALT(" + ",".join(_key(child) for child in prog.children) + ")"
    elif prog.op == "SEQ":
        text = "SEQ(" + ",".join(_key(child) for child in prog.children) + ")"
    else:
        text = "?"
    _KEYS[ident] = text
    return text


def same_fact(fact: Hyp, hyp: Hyp, bounds: bool) -> bool:
    if fact.kind != hyp.kind or fact.slot != hyp.slot:
        return False
    if fact.kind == "cut":
        return fact.index == hyp.index
    if fact.kind in {"alt", "pair"}:
        return fact.start == hyp.start and fact.end == hyp.end and fact.slot2 == hyp.slot2 and fact.start2 == hyp.start2 and fact.end2 == hyp.end2
    if fact.start != hyp.start or fact.end != hyp.end:
        return False
    if fact.kind in {"cls", "set", "int"} and fact.cls != hyp.cls:
        return False
    if bounds and fact.kind in {"cls", "repeat", "set", "int"} and (fact.lo != hyp.lo or fact.hi != hyp.hi):
        return False
    return True


def recall(facts: list[Hyp], hyps: list[Hyp], kind: str, bounds: bool = True) -> tuple[int, int]:
    gold = [fact for fact in facts if fact.kind == kind]
    if not gold:
        return 0, 0
    hits = 0
    for fact in gold:
        if any(same_fact(fact, hyp, bounds) for hyp in hyps):
            hits += 1
    return hits, len(gold)


STRICT_KINDS = ("copy", "cls", "repeat", "optional", "alt", "pair", "cut")


def mean_recall(facts: list[Hyp], hyps: list[Hyp], bounds: bool = True) -> float:
    scores = []
    for kind in STRICT_KINDS:
        hits, total = recall(facts, hyps, kind, bounds)
        if total:
            scores.append(hits / total)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def inventory_covers(facts: list[Hyp], supports: list[str]) -> bool:
    spans = {slot: set(candidate_spans(text)) for slot, text in enumerate(supports)}
    for fact in facts:
        if fact.kind == "cut":
            continue
        if fact.kind == "alt":
            if (fact.start, fact.end) not in spans.get(fact.slot, set()):
                return False
            if (fact.start2, fact.end2) not in spans.get(fact.slot2, set()):
                return False
            continue
        if fact.kind == "pair":
            if (fact.start, fact.end) not in spans.get(fact.slot, set()):
                return False
            if (fact.start2, fact.end2) not in spans.get(fact.slot, set()):
                return False
            continue
        if (fact.start, fact.end) not in spans.get(fact.slot, set()):
            return False
    return True


def chart_agrees(prog: Prog, supports: list[str], probe: list[tuple[str, bool]], limit: int = 8) -> bool:
    """True when the version space still contains a probe-matching program.

    `limit` is retained so older callers keep working. The version space does
    not discard a support-consistent program to honor it.
    """
    del limit
    from pattern_induction.version_space import assemble

    built = assemble(supports, gold_facts(prog, supports))
    return any(agrees(candidate, supports, probe) for candidate, _score in built.programs)
