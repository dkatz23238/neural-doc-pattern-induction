"""Bottom-up version space over local hypotheses.

Each hypothesis licenses a chunk at the span it names. Copy and class chunks
are atoms. A repeat hypothesis wraps every program that matches its unit
exactly, and the wrapped program keeps that hypothesis's own bounds. An
optional hypothesis wraps the programs that match its span. An alternation
hypothesis pairs every left program with every right program.

Atoms and repeats form a spine and are stitched left to right. Each optional
and each alternation is substituted into that spine at the span it names,
including an optional prefix in front of an alternation. Two partials merge
only when their ASTs match, so digit{1,2} and digit{1,8} stay side by side
when the supports are too short to separate them. A partial that fails to
match some support is dropped. A partial that still matches is kept even
when another chunk has a higher score.

The cell cap is a safety valve. Hitting it means some support-consistent
program may be missing, and the assembly is marked unfinished.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass

from pattern_induction.chart import STRICT_KINDS, Hyp, _key, candidate_spans, same_fact
from pattern_induction.program import OPEN, Prog, accepts_all, advance, ast_id, class_ok, supports_id

CELL_CAP = 4096
MAX_NODES = 16


@dataclass(frozen=True)
class _Chunk:
    start: int
    end: int
    prog: Prog


@dataclass(frozen=True)
class _AltUse:
    slot: int
    start: int
    prog: Prog


@dataclass
class Assembly:
    programs: list[tuple[Prog, float]]
    truncated: bool
    library: int


def _dedup(programs: list[Prog]) -> list[Prog]:
    found: dict[str, Prog] = {}
    for prog in programs:
        found.setdefault(_key(prog), prog)
    return list(found.values())


def _unique_chunks(chunks: list[_Chunk]) -> list[_Chunk]:
    found: dict[tuple[int, int, str], _Chunk] = {}
    for chunk in chunks:
        found.setdefault((chunk.start, chunk.end, _key(chunk.prog)), chunk)
    return list(found.values())


def _span_ok(text: str, start: int, end: int) -> bool:
    return 0 <= start < end <= len(text)


def _canonical_copy(slot: int, start: int, end: int, supports: list[str]) -> Prog | None:
    if not (0 <= slot < len(supports)) or not _span_ok(supports[slot], start, end):
        return None
    literal = supports[slot][start:end]
    if not literal:
        return None
    for index, text in enumerate(supports):
        at = text.find(literal)
        if at >= 0:
            return Prog("COPY", slot=index, start=at, end=at + len(literal))
    return None


_PACKED: dict[tuple[int, int], Prog] = {}
_PACKED_CAP = 200_000


def _pack(prog: Prog, supports: list[str]) -> Prog:
    key = (ast_id(prog), supports_id(supports))
    cached = _PACKED.get(key)
    if cached is not None:
        return cached
    packed = _pack_tree(prog, supports)
    if len(_PACKED) < _PACKED_CAP:
        _PACKED[key] = packed
    return packed


def _pack_tree(prog: Prog, supports: list[str]) -> Prog:
    """Collapse concatenations that denote one class or one literal.

    Adjacent digit{2} and digit{3} are digit{5}. Adjacent copies of a
    literal that occurs in a support become one copy. The language is
    unchanged. Different widths stay side by side.
    """
    if prog.op == "SEQ":
        flat: list[Prog] = []
        for child in prog.children:
            child = _pack(child, supports)
            if child.op == "SEQ":
                flat.extend(child.children)
            else:
                flat.append(child)
        merged: list[Prog] = []
        for part in flat:
            if merged:
                fused = _fuse(merged[-1], part, supports)
                if fused is not None:
                    merged[-1] = fused
                    continue
            merged.append(part)
        if len(merged) == 1:
            return merged[0]
        return Prog("SEQ", children=merged)
    if prog.op == "OPT" and prog.children:
        return Prog("OPT", children=[_pack(prog.children[0], supports)])
    if prog.op == "REPEAT" and prog.children:
        child = _pack(prog.children[0], supports)
        collapsed = _pack_repeat(prog.lo, prog.hi, child)
        if collapsed is not None:
            return collapsed
        return Prog("REPEAT", lo=prog.lo, hi=prog.hi, children=[child])
    if prog.op == "ALT" and prog.children:
        children = [_pack(child, supports) for child in prog.children[:2]]
        children.sort(key=_key)
        if len(children) == 2:
            collapsed = _pack_alt(children[0], children[1], supports)
            if collapsed is not None:
                return collapsed
        return Prog("ALT", children=children)
    return prog


def _pack_repeat(lo: int, hi: int, child: Prog) -> Prog | None:
    """A repeat whose language is one class becomes that class.

    One required copy is the unit itself. A one-character class repeated
    ``{lo,hi}`` is that class at those bounds. An exact-width class repeated
    a fixed number of times is that class at the product width. A wider unit
    repeated a variable number of times skips lengths, so it stays a repeat.
    """
    if lo == 1 and hi == 1:
        return child
    if child.op == "SET" and child.lo == child.hi:
        if child.lo == 1:
            return Prog("SET", spec=child.spec, lo=lo, hi=hi)
        if lo == hi:
            width = child.lo * lo
            return Prog("SET", spec=child.spec, lo=width, hi=width)
        return None
    if child.op != "CLS" or child.lo != child.hi:
        return None
    if child.lo == 1:
        return Prog("CLS", cls=child.cls, lo=lo, hi=hi)
    if lo == hi:
        width = child.lo * lo
        return Prog("CLS", cls=child.cls, lo=width, hi=width)
    return None


def _pack_alt(left: Prog, right: Prog, supports: list[str]) -> Prog | None:
    """Drop a branch that adds no strings to the other branch."""
    if _key(left) == _key(right):
        return left
    if left.op == "CLS" and right.op == "CLS" and left.cls == right.cls:
        if left.hi + 1 >= right.lo and right.hi + 1 >= left.lo:
            return Prog("CLS", cls=left.cls, lo=min(left.lo, right.lo), hi=max(left.hi, right.hi))
        return None
    if left.op == "CLS" and right.op == "COPY" and _copy_in_class(right, left, supports):
        return left
    if right.op == "CLS" and left.op == "COPY" and _copy_in_class(left, right, supports):
        return right
    return None


def _copy_in_class(copy: Prog, cls: Prog, supports: list[str]) -> bool:
    literal = _copy_text(copy, supports)
    if not literal or not (cls.lo <= len(literal) <= cls.hi):
        return False
    return class_ok(literal, cls.cls)


def _fuse(left: Prog, right: Prog, supports: list[str]) -> Prog | None:
    if left.op == "CLS" and right.op == "CLS" and left.cls == right.cls:
        return Prog("CLS", cls=left.cls, lo=left.lo + right.lo, hi=left.hi + right.hi)
    if left.op == "SET" and right.op == "SET" and left.spec == right.spec:
        return Prog("SET", spec=left.spec, lo=left.lo + right.lo, hi=left.hi + right.hi)
    if left.op == "COPY" and right.op == "COPY":
        literal = _copy_text(left, supports) + _copy_text(right, supports)
        if not literal:
            return None
        for slot, text in enumerate(supports):
            at = text.find(literal)
            if at >= 0:
                return Prog("COPY", slot=slot, start=at, end=at + len(literal))
    return None


def _copy_text(prog: Prog, supports: list[str]) -> str:
    if prog.slot < 0 or prog.slot >= len(supports):
        return ""
    text = supports[prog.slot]
    if prog.start < 0 or prog.end > len(text) or prog.end <= prog.start:
        return ""
    return text[prog.start : prog.end]


def _cat(left: Prog, right: Prog) -> Prog:
    parts: list[Prog] = []
    for piece in (left, right):
        if piece.op == "SEQ":
            parts.extend(piece.children)
        else:
            parts.append(piece)
    if len(parts) == 1:
        return parts[0]
    return Prog("SEQ", children=parts)


def _prefix_ok(prog: Prog, supports: list[str]) -> bool:
    for text in supports:
        if text and not advance(prog, text, supports, 0):
            return False
    return True


def _stitch(
    text: str,
    supports: list[str],
    origin: int,
    goal: int,
    chunks: list[_Chunk],
    floating: list[Prog],
    check_supports: bool,
    pack: bool = False,
) -> tuple[dict[str, tuple[Prog, float]], bool]:
    """Every concatenation of chunks that matches exactly [origin, goal)."""
    if goal <= origin or goal > len(text):
        return {}, False
    by_start: dict[int, list[_Chunk]] = {}
    for chunk in chunks:
        if chunk.start < origin or chunk.end > goal or chunk.end <= chunk.start:
            continue
        by_start.setdefault(chunk.start, []).append(chunk)
    dp: dict[int, dict[str, tuple[Prog | None, float]]] = {origin: {"": (None, 0.0)}}
    heap = [origin]
    pending = {origin}
    done: set[int] = set()
    truncated = False
    while heap:
        pos = heapq.heappop(heap)
        pending.discard(pos)
        if pos in done or pos > goal:
            continue
        done.add(pos)
        if pos == goal:
            continue
        for prog, score in list(dp.get(pos, {}).values()):
            extensions: list[tuple[int, Prog]] = [(chunk.end, chunk.prog) for chunk in by_start.get(pos, [])]
            for piece in floating:
                for end in advance(piece, text, supports, pos):
                    if origin < end <= goal:
                        extensions.append((end, piece))
            for end, piece in extensions:
                if prog is not None and prog.nodes() + piece.nodes() > MAX_NODES:
                    continue
                nxt = piece if prog is None else _cat(prog, piece)
                if pack:
                    nxt = _pack(nxt, supports)
                if nxt.nodes() > MAX_NODES:
                    continue
                if end not in advance(nxt, text, supports, origin):
                    continue
                if check_supports and not _prefix_ok(nxt, supports):
                    continue
                key = _key(nxt)
                bucket = dp.setdefault(end, {})
                if key in bucket:
                    continue
                if len(bucket) >= CELL_CAP:
                    truncated = True
                    continue
                bucket[key] = (nxt, score + 1.0)
                if end not in done and end not in pending:
                    heapq.heappush(heap, end)
                    pending.add(end)
    finished: dict[str, tuple[Prog, float]] = {}
    for key, (prog, score) in dp.get(goal, {}).items():
        if prog is not None:
            finished[key] = (prog, score)
    return finished, truncated


def _atom_chunks(supports: list[str], hyps: list[Hyp]) -> dict[int, list[_Chunk]]:
    chunks: dict[int, list[_Chunk]] = {slot: [] for slot in range(len(supports))}
    for hyp in hyps:
        if hyp.slot < 0 or hyp.slot >= len(supports):
            continue
        text = supports[hyp.slot]
        if not _span_ok(text, hyp.start, hyp.end):
            continue
        prog: Prog | None = None
        if hyp.kind == "copy":
            prog = _canonical_copy(hyp.slot, hyp.start, hyp.end, supports)
        elif hyp.kind == "cls" and hyp.cls in {"digit", "upper", "lower"}:
            # A class width is one node and may pass the repeat cap. Repeat
            # bounds stay capped in `_repeat_chunks`.
            lo = max(1, int(hyp.lo))
            hi = max(lo, int(hyp.hi))
            prog = Prog("CLS", cls=hyp.cls, lo=lo, hi=hi)
        elif hyp.kind == "set" and hyp.cls:
            lo = max(1, int(hyp.lo))
            hi = max(lo, int(hyp.hi))
            prog = Prog("SET", spec=hyp.cls, lo=lo, hi=hi)
        elif hyp.kind == "int" and ":" in hyp.cls:
            lo = max(1, int(hyp.lo))
            hi = max(lo, int(hyp.hi))
            prog = Prog("INT", spec=hyp.cls, lo=lo, hi=hi)
        if prog is None:
            continue
        if hyp.end in advance(prog, text, supports, hyp.start):
            chunks[hyp.slot].append(_Chunk(hyp.start, hyp.end, prog))
    for slot in chunks:
        chunks[slot] = _unique_chunks(chunks[slot])
    return chunks


def _repeat_chunks(
    supports: list[str], hyps: list[Hyp], atoms: dict[int, list[_Chunk]], pack: bool = False
) -> tuple[dict[int, list[_Chunk]], bool]:
    rows = [hyp for hyp in hyps if hyp.kind == "repeat"]
    rows.sort(key=lambda hyp: (hyp.end - hyp.start, hyp.slot, hyp.start))
    chunks: dict[int, list[_Chunk]] = {slot: [] for slot in range(len(supports))}
    truncated = False
    for hyp in rows:
        if hyp.slot < 0 or hyp.slot >= len(supports):
            continue
        text = supports[hyp.slot]
        if not _span_ok(text, hyp.start, hyp.end):
            continue
        pieces = [chunk for chunk in atoms[hyp.slot] if hyp.start <= chunk.start and chunk.end <= hyp.end]
        units, hit = _stitch(text, supports, hyp.start, hyp.end, pieces, [], False, pack)
        truncated = truncated or hit
        lo = max(1, min(int(hyp.lo), OPEN))
        hi = max(lo, min(int(hyp.hi), OPEN))
        for unit, _score in units.values():
            prog = Prog("REPEAT", lo=lo, hi=hi, children=[unit])
            if pack:
                prog = _pack(prog, supports)
            for end in advance(prog, text, supports, hyp.start):
                if end > hyp.start:
                    chunks[hyp.slot].append(_Chunk(hyp.start, end, prog))
    for slot in chunks:
        chunks[slot] = _unique_chunks(chunks[slot])
    return chunks, truncated


def _optional_chunks(
    supports: list[str], hyps: list[Hyp], atoms: dict[int, list[_Chunk]], pack: bool = False
) -> tuple[dict[int, list[_Chunk]], bool]:
    chunks: dict[int, list[_Chunk]] = {slot: [] for slot in range(len(supports))}
    truncated = False
    for hyp in hyps:
        if hyp.kind != "optional" or hyp.slot < 0 or hyp.slot >= len(supports):
            continue
        text = supports[hyp.slot]
        if not _span_ok(text, hyp.start, hyp.end):
            continue
        pieces = [chunk for chunk in atoms[hyp.slot] if hyp.start <= chunk.start and chunk.end <= hyp.end]
        inners, hit = _stitch(text, supports, hyp.start, hyp.end, pieces, [], False, pack)
        truncated = truncated or hit
        for inner, _score in inners.values():
            chunks[hyp.slot].append(_Chunk(hyp.start, hyp.end, Prog("OPT", children=[inner])))
    for slot in chunks:
        chunks[slot] = _unique_chunks(chunks[slot])
    return chunks, truncated


def _alt_programs(
    supports: list[str],
    hyps: list[Hyp],
    atoms: dict[int, list[_Chunk]],
    repeats: dict[int, list[_Chunk]],
    pack: bool = False,
) -> tuple[list[_AltUse], bool]:
    del repeats
    found: dict[tuple[int, int, str], _AltUse] = {}
    truncated = False
    for hyp in hyps:
        if hyp.kind != "alt":
            continue
        if hyp.slot < 0 or hyp.slot >= len(supports) or hyp.slot2 < 0 or hyp.slot2 >= len(supports):
            continue
        left_text = supports[hyp.slot]
        right_text = supports[hyp.slot2]
        if not _span_ok(left_text, hyp.start, hyp.end) or not _span_ok(right_text, hyp.start2, hyp.end2):
            continue
        left_pieces = [chunk for chunk in atoms[hyp.slot] if hyp.start <= chunk.start and chunk.end <= hyp.end]
        right_pieces = [chunk for chunk in atoms[hyp.slot2] if hyp.start2 <= chunk.start and chunk.end <= hyp.end2]
        lefts, left_hit = _stitch(left_text, supports, hyp.start, hyp.end, left_pieces, [], False, pack)
        rights, right_hit = _stitch(right_text, supports, hyp.start2, hyp.end2, right_pieces, [], False, pack)
        truncated = truncated or left_hit or right_hit
        for left, _left_score in lefts.values():
            for right, _right_score in rights.values():
                prog = Prog("ALT", children=[left, right])
                if pack:
                    prog = _pack(prog, supports)
                found.setdefault((hyp.slot, hyp.start, _key(prog)), _AltUse(hyp.slot, hyp.start, prog))
    return list(found.values()), truncated


def _wrap_documented_alts(
    supports: list[str],
    hyps: list[Hyp],
    optionals: dict[int, list[_Chunk]],
    alt_progs: list[Prog],
) -> None:
    """An optional span we added may be an alternation, not a single literal.

    Only hypotheses with score 0, the ones alignment adds after a chart comes
    back empty, take this wrap. Neural optionals keep their own score and
    stay as they were.
    """
    if not alt_progs:
        return
    for hyp in hyps:
        if hyp.kind != "optional" or hyp.score != 0.0:
            continue
        if hyp.slot < 0 or hyp.slot >= len(supports):
            continue
        text = supports[hyp.slot]
        if not _span_ok(text, hyp.start, hyp.end):
            continue
        for alt in alt_progs:
            if hyp.end not in advance(alt, text, supports, hyp.start):
                continue
            optionals[hyp.slot].append(_Chunk(hyp.start, hyp.end, Prog("OPT", children=[alt])))
    for slot in optionals:
        optionals[slot] = _unique_chunks(optionals[slot])


def _join(parts: list[Prog | None]) -> Prog | None:
    prog: Prog | None = None
    for part in parts:
        if part is None:
            continue
        prog = part if prog is None else _cat(prog, part)
    return prog


def assemble(supports: list[str], hyps: list[Hyp], wrap_optional_alt: bool = False, pack: bool = False) -> Assembly:
    """Support-consistent programs entailed by these hypotheses.

    Atoms and repeats form a spine. Each optional and alternation hypothesis
    is substituted into that spine at the span it names, including the case
    where an optional prefix sits in front of an alternation. The score is
    the number of pieces. A shorter derivation ranks first, and no program
    is removed to make room for a higher score.
    """
    atoms = _atom_chunks(supports, hyps)
    repeats, truncated = _repeat_chunks(supports, hyps, atoms, pack)
    optionals, opt_hit = _optional_chunks(supports, hyps, atoms, pack)
    truncated = truncated or opt_hit
    alts, alt_hit = _alt_programs(supports, hyps, atoms, repeats, pack)
    truncated = truncated or alt_hit
    alt_progs = _dedup([alt.prog for alt in alts])
    if wrap_optional_alt:
        documented = [hyp for hyp in hyps if hyp.kind != "alt" or hyp.score == 0.0]
        doc_alts, _doc_hit = _alt_programs(supports, documented, atoms, repeats, pack)
        _wrap_documented_alts(supports, hyps, optionals, _dedup([alt.prog for alt in doc_alts]))
    library = sum(len(rows) for rows in atoms.values()) + sum(len(rows) for rows in repeats.values())
    library += sum(len(rows) for rows in optionals.values()) + len(alt_progs)
    cache: dict[tuple[int, int, int], tuple[list[Prog | None], bool]] = {}

    def cover(slot: int, start: int, end: int) -> list[Prog | None]:
        nonlocal truncated
        if start == end:
            return [None]
        key = (slot, start, end)
        cached = cache.get(key)
        if cached is not None:
            return cached[0]
        text = supports[slot]
        pieces = _unique_chunks(atoms[slot] + repeats[slot])
        finished, hit = _stitch(text, supports, start, end, pieces, [], False, pack)
        truncated = truncated or hit
        progs: list[Prog | None] = [prog for prog, _score in finished.values()]
        cache[key] = (progs, hit)
        return progs

    kept: dict[str, tuple[Prog, float]] = {}

    def emit(parts: list[Prog | None], score: float) -> None:
        prog = _join(parts)
        if prog is not None and pack:
            prog = _pack(prog, supports)
        if prog is None or prog.nodes() > MAX_NODES:
            return
        key = _key(prog)
        if key in kept:
            return
        if accepts_all(prog, supports, supports):
            kept[key] = (prog, score)

    for slot, text in enumerate(supports):
        if not text:
            continue
        pieces = _unique_chunks(atoms[slot] + repeats[slot])
        finished, hit = _stitch(text, supports, 0, len(text), pieces, [], True, pack)
        truncated = truncated or hit
        for prog, score in finished.values():
            emit([prog], score)
        for chunk in optionals[slot]:
            for prefix in cover(slot, 0, chunk.start):
                for suffix in cover(slot, chunk.end, len(text)):
                    emit([prefix, chunk.prog, suffix], 3.0)
        for alt in alts:
            if alt.slot != slot:
                continue
            ends = [end for end in advance(alt.prog, text, supports, alt.start) if end > alt.start]
            for end in ends:
                for prefix in cover(slot, 0, alt.start):
                    for suffix in cover(slot, end, len(text)):
                        emit([prefix, alt.prog, suffix], 3.0)
        for chunk in optionals[slot]:
            for alt in alt_progs:
                ends = [end for end in advance(alt, text, supports, chunk.end) if end > chunk.end]
                for end in ends:
                    for prefix in cover(slot, 0, chunk.start):
                        for suffix in cover(slot, end, len(text)):
                            emit([prefix, chunk.prog, alt, suffix], 4.0)
    ranked = sorted(kept.values(), key=lambda item: (item[1], _key(item[0])))
    return Assembly(ranked, truncated, library)


def _novel(hyp: Hyp, gold: list[Hyp], chosen: list[Hyp]) -> bool:
    if any(same_fact(fact, hyp, True) for fact in gold if fact.kind == hyp.kind):
        return False
    if any(same_fact(fact, hyp, True) for fact in chosen if fact.kind == hyp.kind):
        return False
    return True


def distractors(supports: list[str], gold: list[Hyp], per_head: int, rng: random.Random) -> list[Hyp]:
    """Extra hypotheses that are not gold facts, up to `per_head` of each kind."""
    if per_head <= 0:
        return []
    spans = [candidate_spans(text) for text in supports]
    pools: dict[str, list[Hyp]] = {kind: [] for kind in STRICT_KINDS}
    bounds = ((1, 1), (1, 2), (1, 3), (2, 2), (1, OPEN), (3, 3))
    for slot, text in enumerate(supports):
        for start, end in spans[slot]:
            pools["copy"].append(Hyp("copy", slot, start, end, score=0.4))
            for name in ("digit", "upper", "lower"):
                for lo, hi in bounds:
                    pools["cls"].append(Hyp("cls", slot, start, end, score=0.4, cls=name, lo=lo, hi=hi))
            for lo, hi in bounds:
                pools["repeat"].append(Hyp("repeat", slot, start, end, score=0.4, lo=lo, hi=hi))
            pools["optional"].append(Hyp("optional", slot, start, end, score=0.4))
        for index in range(len(text) + 1):
            pools["cut"].append(Hyp("cut", slot, index, index, score=0.4, index=index))
        for left_start, left_end in spans[slot]:
            for right_start, right_end in spans[slot]:
                if right_start == left_end:
                    pools["pair"].append(
                        Hyp(
                            "pair",
                            slot,
                            left_start,
                            left_end,
                            score=0.4,
                            slot2=slot,
                            start2=right_start,
                            end2=right_end,
                        )
                    )
    for slot in range(len(supports)):
        for other in range(len(supports)):
            if other == slot:
                continue
            rights: dict[int, tuple[int, int]] = {}
            for start, end in spans[other]:
                rights.setdefault(start, (start, end))
            for start, end in spans[slot]:
                match = rights.get(start)
                if match is None:
                    continue
                pools["alt"].append(
                    Hyp("alt", slot, start, end, score=0.4, slot2=other, start2=match[0], end2=match[1])
                )
    chosen: list[Hyp] = []
    for kind in STRICT_KINDS:
        pool = pools[kind]
        rng.shuffle(pool)
        taken = 0
        for hyp in pool:
            if not _novel(hyp, gold, chosen):
                continue
            chosen.append(hyp)
            taken += 1
            if taken >= per_head:
                break
    return chosen
