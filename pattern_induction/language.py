"""Semantic families, a schema grammar, and disagreement witnesses.

The synthesizer stays frozen. These routines shrink its program set to
distinct languages, score those languages with a grammar fit on gold schemas,
and name the short strings where the remaining languages disagree.
"""

from __future__ import annotations

import math
import string
from collections import Counter, deque
from dataclasses import dataclass, field

from pattern_induction.program import OPEN, Prog, class_ok, describe, execute

DFA_CAP = 1500
OPS = ("COPY", "CLS", "SEQ", "REPEAT", "OPT", "ALT")
_FRESH_POOL = {
    "digit": string.digits,
    "upper": string.ascii_uppercase,
    "lower": string.ascii_lowercase,
    "punct": "#@%+=",
}


@dataclass
class _NFA:
    start: int = 0
    accept: set[int] = field(default_factory=set)
    eps: list[list[int]] = field(default_factory=list)
    sym: list[list[list[int]]] = field(default_factory=list)
    n_sym: int = 0

    def new(self) -> int:
        self.eps.append([])
        self.sym.append([[] for _ in range(self.n_sym)])
        return len(self.eps) - 1

    def edge(self, src: int, symbol: int, dst: int) -> None:
        self.sym[src][symbol].append(dst)

    def jump(self, src: int, dst: int) -> None:
        self.eps[src].append(dst)


@dataclass
class _DFA:
    start: int
    accept: set[int]
    trans: list[dict[int, int]]
    n_sym: int
    sink: int


@dataclass
class Family:
    programs: list[Prog]
    key: tuple
    representative: Prog


def flatten(prog: Prog) -> Prog:
    if prog.op == "SEQ":
        children: list[Prog] = []
        for child in prog.children:
            flat = flatten(child)
            if flat.op == "SEQ":
                children.extend(flat.children)
            else:
                children.append(flat)
        if len(children) == 1:
            return children[0]
        return Prog("SEQ", children=children)
    return Prog(
        prog.op,
        slot=prog.slot,
        start=prog.start,
        end=prog.end,
        cls=prog.cls,
        lo=prog.lo,
        hi=prog.hi,
        children=[flatten(child) for child in prog.children],
    )


def _literal(prog: Prog, supports: list[str]) -> str:
    if prog.slot < 0 or prog.slot >= len(supports):
        return ""
    text = supports[prog.slot]
    if prog.start < 0 or prog.end > len(text) or prog.end <= prog.start:
        return ""
    return text[prog.start : prog.end]


def _copy_chars(programs: list[Prog], supports: list[str]) -> set[str]:
    found: set[str] = set()
    for text in supports:
        found.update(text)

    def walk(prog: Prog) -> None:
        if prog.op == "COPY":
            found.update(_literal(prog, supports))
        for child in prog.children:
            walk(child)

    for prog in programs:
        walk(prog)
    return found


def _class_name(ch: str) -> str:
    if ch.isdigit():
        return "digit"
    if ch.isupper():
        return "upper"
    if ch.islower():
        return "lower"
    return "punct"


def _atoms(exact: set[str]) -> list[tuple[str, str]]:
    atoms = [("exact", ch) for ch in sorted(exact)]
    for name, pool in _FRESH_POOL.items():
        if any(ch not in exact for ch in pool):
            atoms.append(("fresh", name))
    return atoms


def _class_atoms(cls: str, atoms: list[tuple[str, str]]) -> list[int]:
    found = []
    for index, (kind, payload) in enumerate(atoms):
        if kind == "exact" and class_ok(payload, cls):
            found.append(index)
        elif kind == "fresh" and payload == cls:
            found.append(index)
    return found


def _atom_char(atom: tuple[str, str], exact: set[str]) -> str | None:
    kind, payload = atom
    if kind == "exact":
        return payload
    for ch in _FRESH_POOL[payload]:
        if ch not in exact:
            return ch
    return None


def _compile(prog: Prog, atoms: list[tuple[str, str]], supports: list[str]) -> _NFA | None:
    nfa = _NFA(n_sym=len(atoms))

    def node(prog: Prog) -> tuple[int, set[int]] | None:
        if prog.op == "COPY":
            literal = _literal(prog, supports)
            if not literal:
                return None
            entry = nfa.new()
            cursor = entry
            for ch in literal:
                try:
                    symbol = atoms.index(("exact", ch))
                except ValueError:
                    return None
                nxt = nfa.new()
                nfa.edge(cursor, symbol, nxt)
                cursor = nxt
            return entry, {cursor}
        if prog.op == "CLS":
            symbols = _class_atoms(prog.cls, atoms)
            if not symbols:
                return None
            lo = max(1, min(prog.lo, prog.hi))
            hi = max(lo, min(max(prog.lo, prog.hi), OPEN))
            entry = nfa.new()
            cursor = entry
            accept: set[int] = set()
            for count in range(1, hi + 1):
                nxt = nfa.new()
                for symbol in symbols:
                    nfa.edge(cursor, symbol, nxt)
                cursor = nxt
                if count >= lo:
                    accept.add(cursor)
            return entry, accept
        if prog.op == "SEQ" and prog.children:
            built = [node(child) for child in prog.children]
            if any(item is None for item in built):
                return None
            entry, accept = built[0]
            for start, done in built[1:]:
                for state in accept:
                    nfa.jump(state, start)
                accept = done
            return entry, accept
        if prog.op == "ALT" and len(prog.children) >= 2:
            built = [node(child) for child in prog.children[:2]]
            if any(item is None for item in built):
                return None
            entry = nfa.new()
            exit_state = nfa.new()
            for start, done in built:
                nfa.jump(entry, start)
                for state in done:
                    nfa.jump(state, exit_state)
            return entry, {exit_state}
        if prog.op == "OPT" and prog.children:
            built = node(prog.children[0])
            if built is None:
                return None
            start, done = built
            entry = nfa.new()
            exit_state = nfa.new()
            nfa.jump(entry, exit_state)
            nfa.jump(entry, start)
            for state in done:
                nfa.jump(state, exit_state)
            return entry, {exit_state}
        if prog.op == "REPEAT" and prog.children:
            hi = max(prog.lo, min(prog.hi, OPEN))
            lo = min(prog.lo, hi)
            entry = nfa.new()
            cursor = entry
            accept = set()
            for count in range(1, hi + 1):
                built = node(prog.children[0])
                if built is None:
                    return None
                start, done = built
                nfa.jump(cursor, start)
                cursor = nfa.new()
                for state in done:
                    nfa.jump(state, cursor)
                if count >= lo:
                    accept.add(cursor)
            return entry, accept
        return None

    built = node(flatten(prog))
    if built is None:
        return None
    nfa.start, nfa.accept = built
    return nfa


def _closure(nfa: _NFA, states: set[int]) -> frozenset[int]:
    seen = set(states)
    stack = list(states)
    while stack:
        state = stack.pop()
        for dst in nfa.eps[state]:
            if dst not in seen:
                seen.add(dst)
                stack.append(dst)
    return frozenset(seen)


def _determinize(nfa: _NFA) -> _DFA | None:
    start = _closure(nfa, {nfa.start})
    index = {start: 0}
    trans: list[dict[int, int]] = [{}]
    accept = set()
    if any(state in nfa.accept for state in start):
        accept.add(0)
    queue = deque([start])
    while queue:
        current = queue.popleft()
        here = index[current]
        for symbol in range(nfa.n_sym):
            reached: set[int] = set()
            for state in current:
                reached.update(nfa.sym[state][symbol])
            if not reached:
                continue
            dest = _closure(nfa, reached)
            if dest not in index:
                if len(index) >= DFA_CAP:
                    return None
                index[dest] = len(index)
                trans.append({})
                if any(state in nfa.accept for state in dest):
                    accept.add(index[dest])
                queue.append(dest)
            trans[here][symbol] = index[dest]
    sink = len(trans)
    return _DFA(0, accept, trans, nfa.n_sym, sink)


def _minimize(dfa: _DFA) -> _DFA:
    count = dfa.sink + 1

    def dest(state: int, symbol: int) -> int:
        if state == dfa.sink:
            return dfa.sink
        return dfa.trans[state].get(symbol, dfa.sink)

    marked = [[False] * count for _ in range(count)]
    for left in range(count):
        for right in range(left + 1, count):
            if (left in dfa.accept) != (right in dfa.accept):
                marked[left][right] = marked[right][left] = True
    changed = True
    while changed:
        changed = False
        for left in range(count):
            for right in range(left + 1, count):
                if marked[left][right]:
                    continue
                for symbol in range(dfa.n_sym):
                    if marked[dest(left, symbol)][dest(right, symbol)]:
                        marked[left][right] = marked[right][left] = True
                        changed = True
                        break
    class_of = [-1] * count
    classes: list[int] = []
    for state in range(count):
        match = next((rep for rep in classes if not marked[state][rep]), None)
        if match is None:
            class_of[state] = len(classes)
            classes.append(state)
        else:
            class_of[state] = class_of[match]
    sink_class = class_of[dfa.sink]
    live = [cls for cls in range(len(classes)) if cls != sink_class]
    remap = {cls: index for index, cls in enumerate(live)}
    remap[sink_class] = len(live)
    trans = [{} for _ in live]
    accept = set()
    for state in range(dfa.sink):
        here = remap[class_of[state]]
        if state in dfa.accept:
            accept.add(here)
        for symbol in range(dfa.n_sym):
            nxt = remap[class_of[dest(state, symbol)]]
            if nxt != remap[sink_class]:
                trans[here][symbol] = nxt
    return _DFA(remap[class_of[dfa.start]], accept, trans, dfa.n_sym, remap[sink_class])


def _canonicalize(dfa: _DFA) -> tuple:
    mapping: dict[int, int] = {dfa.start: 0}
    order = [dfa.start]
    cursor = 0
    while cursor < len(order):
        state = order[cursor]
        cursor += 1
        for symbol in range(dfa.n_sym):
            dest = dfa.trans[state].get(symbol, dfa.sink) if state != dfa.sink else dfa.sink
            if dest not in mapping:
                mapping[dest] = len(mapping)
                if dest != dfa.sink:
                    order.append(dest)
    rows = []
    for state in order:
        dests = tuple(mapping[dfa.trans[state].get(symbol, dfa.sink)] for symbol in range(dfa.n_sym))
        rows.append((state in dfa.accept, dests))
    rows.append((False, tuple(mapping[dfa.sink] for _ in range(dfa.n_sym))))
    return tuple(rows)


def _dfa_key(prog: Prog, atoms: list[tuple[str, str]], supports: list[str]) -> tuple | None:
    nfa = _compile(prog, atoms, supports)
    if nfa is None:
        return None
    dfa = _determinize(nfa)
    if dfa is None or dfa.sink + 1 > 300:
        return None
    return _canonicalize(_minimize(dfa))


def _lcp(rows: list[str]) -> str:
    if not rows:
        return ""
    prefix = rows[0]
    for text in rows[1:]:
        while prefix and not text.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


def _lcs(rows: list[str]) -> str:
    if not rows:
        return ""
    suffix = rows[0]
    for text in rows[1:]:
        while suffix and not text.endswith(suffix):
            suffix = suffix[:-1]
    return suffix


def _longest_run(text: str) -> tuple[int, int]:
    best = (0, 0)
    start = 0
    for index, ch in enumerate(text + "\0"):
        if index == len(text) or _class_name(ch) != _class_name(text[start]):
            if index - start > best[1] - best[0]:
                best = (start, index)
            start = index
    return best


def _edit_run(text: str, delta: int) -> str:
    if not text:
        return ""
    left, right = _longest_run(text)
    if right <= left:
        return text
    if delta > 0:
        return text[:right] + text[left] + text[right:]
    if right - left == 1:
        return text[:left] + text[right:]
    return text[: right - 1] + text[right:]


def _retints(text: str) -> list[str]:
    """Same-width runs filled with one character of that run's class."""
    rows = []
    start = 0
    for index, ch in enumerate(text + "\0"):
        if index == len(text) or _class_name(ch) != _class_name(text[start]):
            run = text[start:index]
            if len(run) >= 1:
                uniform = run[0] * len(run)
                if uniform != run:
                    rows.append(text[:start] + uniform + text[index:])
                other = next((item for item in text if _class_name(item) == _class_name(run[0]) and item != run[0]), "")
                if other:
                    rows.append(text[:start] + (other * len(run)) + text[index:])
            start = index
    return rows


def _swap_run(text: str, supports: list[str]) -> str:
    pool = [ch for item in supports for ch in item]
    for index, ch in enumerate(text):
        other = next((item for item in pool if _class_name(item) != _class_name(ch) and item.isalnum() and ch.isalnum()), "")
        if other:
            return text[:index] + other + text[index + 1 :]
    return ""


def _repeat_variants(text: str) -> list[str]:
    puncts = [ch for ch in text if not ch.isalnum()]
    if not puncts:
        return []
    sep = max(set(puncts), key=puncts.count)
    parts = text.split(sep)
    if len(parts) < 2 or any(part == "" for part in parts):
        return []
    unit = parts[-1]
    rows = [sep.join(parts + [unit]), sep.join(parts + [unit, unit])]
    if len(parts) > 2:
        rows.append(sep.join(parts[:-1]))
    rows.append(parts[0])
    return rows


def boundary_strings(supports: list[str], limit: int = 36) -> list[str]:
    """Short strings around the supports: width edits, extra repeats, shared affixes."""
    rows: list[str] = []

    def add(text: str) -> None:
        if text and len(text) <= 24 and text not in rows and len(rows) < limit:
            rows.append(text)

    for text in supports:
        add(text)
        if len(text) > 1:
            add(text[1:])
            add(text[:-1])
        add(_edit_run(text, 1))
        add(_edit_run(text, -1))
        add(_swap_run(text, supports))
        for tinted in _retints(text):
            add(tinted)
        for variant in _repeat_variants(text):
            add(variant)
    if len(supports) >= 2:
        prefix = _lcp(supports)
        suffix = _lcs(supports)
        if prefix:
            add(prefix)
        if suffix:
            add(suffix)
        if prefix and suffix:
            add(prefix + suffix)
            for text in supports:
                if text.startswith(prefix) and text.endswith(suffix) and len(text) > len(prefix) + len(suffix):
                    middle = text[len(prefix) : len(text) - len(suffix)]
                    add(prefix + middle)
                    add(prefix + _edit_run(middle, 1) + suffix)
    return rows


def repeat_extension(text: str, supports: list[str]) -> bool:
    """A longer copy of a separator pattern the supports already show."""
    puncts = [ch for ch in text if not ch.isalnum()]
    if not puncts:
        return False
    sep = max(set(puncts), key=puncts.count)
    seen = max((item.count(sep) for item in supports), default=0)
    return text.count(sep) > seen and any(sep in item for item in supports)


def _battery_bits(prog: Prog, supports: list[str], battery: list[str]) -> tuple[bool, ...]:
    return tuple(bool(execute(prog, text, supports)) for text in battery)


def group_languages(programs: list[Prog], supports: list[str]) -> list[list[Prog]]:
    """Merge syntactic copies and programs that accept the same language."""
    unique: list[Prog] = []
    seen = set()
    for prog in programs:
        key = describe(flatten(prog))
        if key in seen:
            continue
        seen.add(key)
        unique.append(flatten(prog))
    if not unique:
        return []
    exact = _copy_chars(unique, supports)
    atoms = _atoms(exact)
    battery = boundary_strings(supports)
    groups: dict[tuple, list[Prog]] = {}
    for prog in unique:
        dfa = _dfa_key(prog, atoms, supports)
        bits = _battery_bits(prog, supports, battery)
        key = ("dfa", dfa, bits) if dfa is not None else ("battery", bits)
        groups.setdefault(key, []).append(prog)
    return list(groups.values())


class GrammarPrior:
    """Log-probability of a program under a grammar fit to gold schemas.

    Repeat bounds are counted on the atomic compose languages, where a repeat
    that the supports do not cap stays open. Every other production is counted
    on atomic gold and on the D1 schemas.
    """

    def __init__(self) -> None:
        self.op: Counter[tuple[str, str]] = Counter()
        self.parent: Counter[str] = Counter()
        self.cls: Counter[str] = Counter()
        self.cls_n = 0
        self.bounds: Counter[tuple[str, int, int]] = Counter()
        self.bound_n: Counter[str] = Counter()
        self.cls_bounds: Counter[tuple[str, int, int]] = Counter()
        self.cls_bound_n: Counter[str] = Counter()
        self.repeat_bounds: Counter[tuple[int, int]] = Counter()
        self.repeat_n = 0
        self.copy_len: Counter[int] = Counter()
        self.copy_n = 0
        self.seq_n: Counter[int] = Counter()
        self.seq_total = 0

    def observe(self, prog: Prog, parent: str = "ROOT", repeat_bounds: bool = False) -> None:
        prog = flatten(prog)
        self.op[(parent, prog.op)] += 1
        self.parent[parent] += 1
        if prog.op == "CLS":
            self.cls[prog.cls] += 1
            self.cls_n += 1
            self.bounds[(prog.op, prog.lo, prog.hi)] += 1
            self.bound_n[prog.op] += 1
            self.cls_bounds[(prog.cls, prog.lo, prog.hi)] += 1
            self.cls_bound_n[prog.cls] += 1
        elif prog.op == "REPEAT":
            hi = OPEN if prog.hi >= OPEN else prog.hi
            if repeat_bounds:
                self.repeat_bounds[(prog.lo, hi)] += 1
                self.repeat_n += 1
            else:
                self.bounds[(prog.op, prog.lo, hi)] += 1
                self.bound_n[prog.op] += 1
        elif prog.op == "COPY":
            self.copy_len[min(4, max(1, prog.end - prog.start))] += 1
            self.copy_n += 1
        elif prog.op == "SEQ":
            self.seq_n[min(4, len(prog.children))] += 1
            self.seq_total += 1
        for child in prog.children:
            self.observe(child, prog.op, repeat_bounds)

    def _log(self, count: int, total: int, vocab: int) -> float:
        return math.log((count + 0.5) / (total + 0.5 * max(vocab, 1)))

    def score(self, prog: Prog, parent: str = "ROOT") -> float:
        prog = flatten(prog)
        total = 0.0
        total += self._log(self.op[(parent, prog.op)], self.parent[parent], len(OPS))
        if prog.op == "CLS":
            total += self._log(self.cls[prog.cls], self.cls_n, 3)
            # Width is a property of the class. A one-character letter class
            # should not make a one-digit run look likelier than the digit
            # range the repeat schemas actually use.
            if self.cls_bound_n[prog.cls]:
                total += self._log(self.cls_bounds[(prog.cls, prog.lo, prog.hi)], self.cls_bound_n[prog.cls], 8)
            else:
                total += self._log(self.bounds[(prog.op, prog.lo, prog.hi)], self.bound_n[prog.op], 8)
        elif prog.op == "REPEAT":
            hi = OPEN if prog.hi >= OPEN else prog.hi
            if self.repeat_n:
                total += self._log(self.repeat_bounds[(prog.lo, hi)], self.repeat_n, 6)
            else:
                total += self._log(self.bounds[(prog.op, prog.lo, hi)], self.bound_n[prog.op], 6)
        elif prog.op == "COPY":
            bucket = min(4, max(1, prog.end - prog.start))
            total += self._log(self.copy_len[bucket], self.copy_n, 4)
        elif prog.op == "SEQ":
            total += self._log(self.seq_n[min(4, len(prog.children))], self.seq_total, 3)
        for child in prog.children:
            total += self.score(child, prog.op)
        return total

    def state(self) -> dict:
        return {
            "op": {f"{parent}|{op}": count for (parent, op), count in self.op.items()},
            "parent": dict(self.parent),
            "cls": dict(self.cls),
            "cls_n": self.cls_n,
            "bounds": {f"{op}|{lo}|{hi}": count for (op, lo, hi), count in self.bounds.items()},
            "bound_n": dict(self.bound_n),
            "cls_bounds": {f"{cls}|{lo}|{hi}": count for (cls, lo, hi), count in self.cls_bounds.items()},
            "cls_bound_n": dict(self.cls_bound_n),
            "repeat_bounds": {f"{lo}|{hi}": count for (lo, hi), count in self.repeat_bounds.items()},
            "repeat_n": self.repeat_n,
            "copy_len": {str(key): count for key, count in self.copy_len.items()},
            "copy_n": self.copy_n,
            "seq_n": {str(key): count for key, count in self.seq_n.items()},
            "seq_total": self.seq_total,
        }

    @classmethod
    def from_state(cls, state: dict) -> "GrammarPrior":
        prior = cls()
        prior.op = Counter({tuple(key.split("|")): count for key, count in state["op"].items()})
        prior.parent = Counter(state["parent"])
        prior.cls = Counter(state["cls"])
        prior.cls_n = state["cls_n"]
        prior.bounds = Counter(
            {(parts[0], int(parts[1]), int(parts[2])): count for key, count in state["bounds"].items() for parts in [key.split("|")]}
        )
        prior.bound_n = Counter(state["bound_n"])
        prior.cls_bounds = Counter(
            {(parts[0], int(parts[1]), int(parts[2])): count for key, count in state.get("cls_bounds", {}).items() for parts in [key.split("|")]}
        )
        prior.cls_bound_n = Counter(state.get("cls_bound_n", {}))
        prior.repeat_bounds = Counter(
            {(int(key.split("|")[0]), int(key.split("|")[1])): count for key, count in state["repeat_bounds"].items()}
        )
        prior.repeat_n = state["repeat_n"]
        prior.copy_len = Counter({int(key): count for key, count in state["copy_len"].items()})
        prior.copy_n = state["copy_n"]
        prior.seq_n = Counter({int(key): count for key, count in state["seq_n"].items()})
        prior.seq_total = state["seq_total"]
        return prior


def disagreements(families: list[list[Prog]], supports: list[str], limit: int = 8) -> list[str]:
    """Short strings where at least two families disagree, skipping pure repeat extensions."""
    battery = boundary_strings(supports)
    scored = []
    for text in battery:
        if repeat_extension(text, supports):
            continue
        bits = [execute(group[0], text, supports) for group in families]
        if any(bits) and not all(bits):
            balance = min(sum(bits), len(bits) - sum(bits))
            scored.append((balance, -len(text), text))
    scored.sort(reverse=True)
    return [text for _balance, _length, text in scored[:limit]]


# One confident witness opens a gap smaller than the grammar tie. Two witnesses
# that split the same way clear it. A single bad vote therefore cannot replace
# the schema prior.
WITNESS_GAP = 0.22


def witness_loglik(prog: Prog, supports: list[str], judgments: list[tuple[str, float]]) -> float:
    """Pairwise evidence from confident witnesses. Uncertain votes abstain."""
    total = 0.0
    for text, prob in judgments:
        if repeat_extension(text, supports):
            continue
        if 0.2 < prob < 0.8:
            continue
        accepts = bool(execute(prog, text, supports))
        if prob >= 0.8:
            total += WITNESS_GAP if accepts else 0.0
        else:
            total += WITNESS_GAP if not accepts else 0.0
    return total


def choose_family(
    families: list[list[Prog]],
    supports: list[str],
    judgments: list[tuple[str, float]],
    prior: GrammarPrior,
    stability: dict[int, float] | None = None,
    tie: float = 0.3,
) -> Prog:
    """Witnesses eliminate languages. The grammar, then stability, breaks a near tie."""
    stability = stability or {}
    ranked = []
    for index, group in enumerate(families):
        evidence = witness_loglik(group[0], supports, judgments)
        grammar = max(prior.score(prog) for prog in group)
        representative = max(group, key=lambda prog: (prior.score(prog), -prog.nodes()))
        ranked.append((evidence, grammar, stability.get(index, 0.0), representative))
    ranked.sort(key=lambda item: item[0], reverse=True)
    best = ranked[0][0]
    band = [item for item in ranked if best - item[0] <= tie]
    band.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return band[0][3]
