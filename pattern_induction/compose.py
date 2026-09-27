"""Atomic operators for compositional language induction.

Training covers four families and nothing built by combining them:

- primitive: a constant prefix plus one character class
- repeat: a numeric piece, then the same separator-plus-number one to three times
- optional: a constant prefix, an optional hyphen, and a digit run
- alternation: one fixed literal or one digit run

Extrapolation and compositions are built by the same generators and are not
sampled during training. Repeat extrapolation keeps the support inside one to
three repetitions and asks about four or five. Optional extrapolation moves
the optional piece from the tail to the middle. Alternation extrapolation
puts a constant prefix in front of the two branches. Held-out compositions
are prefix+repeat, repeat+optional tail, and prefix+alternation.

Annotations are the generator's own groups. They ride along as character
indices, so a one-character rename does not move them. Raw codepoints stay
out of the model.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field

from pattern_induction.membership import HELD_OUT, Episode
from pattern_induction.renaming import rename_episode

KIND_LIT = 0
KIND_CLS = 1
KIND_DELIM = 2
KIND_REPEAT = 3
KIND_OPT = 4
KIND_ALT = 5
N_KIND = 6

TRAIN_FAMILIES = ("primitive", "repeat", "optional", "alternation")
SEPS = "-_./"
MAX_GOLD_SPANS = 12


@dataclass
class CharAnn:
    """Per-character latent groups. -1 means the loss should ignore that cell."""

    span: list[int]
    kind: list[int]
    start: list[int]
    end: list[int]
    role: list[list[int]]

    def __len__(self) -> int:
        return len(self.span)


def ignore_ann(text: str) -> CharAnn:
    n = len(text)
    return CharAnn([-1] * n, [-1] * n, [-1] * n, [-1] * n, [[-1, -1, -1] for _ in range(n)])


def paint(groups: list[tuple[list[tuple[str, int]], tuple[int, int, int]]]) -> tuple[str, CharAnn]:
    """Each group is one span: a list of (char, kind) cells and a role triple."""
    span: list[int] = []
    kind: list[int] = []
    start: list[int] = []
    end: list[int] = []
    role: list[list[int]] = []
    chars: list[str] = []
    for span_id, (cells, roles) in enumerate(groups):
        if span_id >= MAX_GOLD_SPANS:
            raise RuntimeError("gold span cap exceeded")
        last = len(cells) - 1
        for offset, (ch, cell_kind) in enumerate(cells):
            chars.append(ch)
            span.append(span_id)
            kind.append(cell_kind)
            start.append(1 if offset == 0 else 0)
            end.append(1 if offset == last else 0)
            role.append([roles[0], roles[1], roles[2]])
    return "".join(chars), CharAnn(span, kind, start, end, role)


@dataclass
class AnnotatedEpisode:
    support: list[str]
    query: str
    label: int
    family: str
    pattern: str
    support_ann: list[CharAnn] = field(default_factory=list)
    query_ann: CharAnn | None = None

    def as_episode(self) -> Episode:
        return Episode(self.support, self.query, self.label, self.pattern, self.family, self.family)


def rename_annotated(episode: AnnotatedEpisode, rng: random.Random, force_holdout: bool = True) -> AnnotatedEpisode:
    """Symbols change. Span indices stay on the same character positions."""
    renamed = rename_episode(episode.as_episode(), rng, force_holdout)
    return AnnotatedEpisode(
        renamed.support,
        renamed.query,
        episode.label,
        episode.family,
        episode.pattern,
        episode.support_ann,
        episode.query_ann,
    )


def _letters(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(n))


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))


def _fresh(rng: random.Random, draw, banned: set[str]) -> str:
    text = draw()
    for _ in range(8):
        if text not in banned:
            return text
        text = draw()
    return text


def _cells(text: str, kind: int) -> list[tuple[str, int]]:
    return [(ch, kind) for ch in text]


# --- languages ----------------------------------------------------------------


class PrefixClass:
    """Constant prefix (possibly empty) and one fixed-width class."""

    def __init__(self, rng: random.Random):
        self.prefix = _letters(rng, rng.choice((0, 2, 2)))
        self.body = rng.choice(("D", "L"))
        self.width = rng.choice((3, 4))
        self.pattern = f"prefix={self.prefix or '-'};body={self.body};width={self.width}"

    def accepts(self, text: str) -> bool:
        if not text.startswith(self.prefix):
            return False
        rest = text[len(self.prefix) :]
        if len(rest) != self.width:
            return False
        if self.body == "D":
            return rest.isdigit()
        return rest.isalpha() and rest.islower()

    def _body(self, rng: random.Random, width: int | None = None) -> str:
        n = self.width if width is None else width
        if self.body == "D":
            return _digits(rng, n)
        return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))

    def _draw(self, rng: random.Random) -> str:
        return self.prefix + self._body(rng)

    def support(self, rng: random.Random) -> list[str]:
        seen: set[str] = set()
        rows = []
        while len(rows) < 3:
            text = _fresh(rng, lambda: self._draw(rng), seen)
            seen.add(text)
            rows.append(text)
        return rows

    def positive(self, rng: random.Random, support: list[str]) -> str:
        return _fresh(rng, lambda: self._draw(rng), set(support))

    def negative(self, rng: random.Random, support: list[str]) -> str:
        banned = set(support)
        for _ in range(30):
            mode = rng.randrange(4)
            if mode == 0 and self.prefix:
                text = _letters(rng, len(self.prefix)) + self._body(rng)
                if text[: len(self.prefix)] == self.prefix:
                    text = "Z" + text[1:]
            elif mode == 1:
                other = self._body(rng)
                if self.body == "D":
                    other = "".join(rng.choice(string.ascii_lowercase) for _ in range(self.width))
                else:
                    other = _digits(rng, self.width)
                text = self.prefix + other
            elif mode == 2:
                text = self.prefix + self._body(rng, self.width + rng.choice((-1, 1)))
            else:
                text = self.prefix + self._body(rng) + rng.choice(SEPS)
            if text and text not in banned and not self.accepts(text):
                return text
        return self.prefix + "!" 

    def annotate(self, text: str) -> CharAnn:
        groups = []
        if self.prefix:
            groups.append((_cells(self.prefix, KIND_LIT), (0, 0, 0)))
        rest = text[len(self.prefix) :]
        groups.append((_cells(rest, KIND_CLS), (0, 0, 0)))
        built, ann = paint(groups)
        if built != text:
            raise RuntimeError("prefix annotation drifted")
        return ann


class RepeatNum:
    """Numeric pieces of width 1 or 2 joined by one fixed separator.

    `repeats` counts separators. Training shows 1, 2, and 3. The language
    itself does not cap the count: four and five separators are still members.
    """

    def __init__(self, rng: random.Random):
        self.sep = rng.choice(SEPS)
        self.pattern = f"repeat;sep={self.sep}"

    def accepts(self, text: str) -> bool:
        if not text or self.sep not in text:
            return False
        if text[0] == self.sep or text[-1] == self.sep:
            return False
        parts = text.split(self.sep)
        if len(parts) < 2:
            return False
        return all(part.isdigit() and 1 <= len(part) <= 2 for part in parts)

    def sample(self, rng: random.Random, repeats: int) -> str:
        parts = [_digits(rng, rng.choice((1, 2))) for _ in range(repeats + 1)]
        return self.sep.join(parts)

    def support(self, rng: random.Random) -> list[str]:
        return [self.sample(rng, repeats) for repeats in (1, 2, 3)]

    def positive(self, rng: random.Random, support: list[str], repeats: int | None = None) -> str:
        choice = repeats if repeats is not None else rng.choice((1, 2, 3))
        return _fresh(rng, lambda: self.sample(rng, choice), set(support))

    def _break(self, rng: random.Random, text: str) -> str:
        mode = rng.randrange(3)
        if mode == 0:
            other = rng.choice([ch for ch in SEPS if ch != self.sep])
            return text.replace(self.sep, other)
        if mode == 1:
            chars = list(text)
            digits = [i for i, ch in enumerate(chars) if ch.isdigit()]
            if not digits:
                return text + "A"
            chars[rng.choice(digits)] = rng.choice(string.ascii_uppercase)
            return "".join(chars)
        parts = text.split(self.sep)
        if len(parts) >= 2:
            parts[rng.randrange(len(parts))] = _digits(rng, 3)
            return self.sep.join(parts)
        return text + self.sep + "A"

    def negative(self, rng: random.Random, support: list[str], repeats: int | None = None) -> str:
        choice = repeats if repeats is not None else rng.choice((1, 2, 3))
        banned = set(support)
        for _ in range(20):
            text = self._break(rng, self.sample(rng, choice))
            if text not in banned and not self.accepts(text):
                return text
        return "A" + self.sep + "B"

    def annotate(self, text: str) -> CharAnn:
        parts = text.split(self.sep)
        groups = [(_cells(parts[0], KIND_CLS), (0, 0, 0))]
        for part in parts[1:]:
            cells = [(self.sep, KIND_DELIM)] + _cells(part, KIND_REPEAT)
            groups.append((cells, (1, 0, 0)))
        built, ann = paint(groups)
        if built != text:
            raise RuntimeError("repeat annotation drifted")
        return ann


class OptionalTail:
    """Three-letter prefix, optional hyphen, three digits."""

    def __init__(self, rng: random.Random):
        self.prefix = _letters(rng, 3)
        self.width = 3
        self.pattern = f"optional;prefix={self.prefix};width={self.width}"

    def accepts(self, text: str) -> bool:
        if not text.startswith(self.prefix):
            return False
        rest = text[len(self.prefix) :]
        if len(rest) == self.width and rest.isdigit():
            return True
        return len(rest) == self.width + 1 and rest[0] == "-" and rest[1:].isdigit()

    def _draw(self, rng: random.Random, hyphen: bool) -> str:
        body = _digits(rng, self.width)
        return self.prefix + ("-" if hyphen else "") + body

    def support(self, rng: random.Random) -> list[str]:
        third = bool(rng.randrange(2))
        return [self._draw(rng, False), self._draw(rng, True), self._draw(rng, third)]

    def positive(self, rng: random.Random, support: list[str]) -> str:
        return _fresh(rng, lambda: self._draw(rng, bool(rng.randrange(2))), set(support))

    def negative(self, rng: random.Random, support: list[str]) -> str:
        banned = set(support)
        for _ in range(30):
            mode = rng.randrange(4)
            if mode == 0:
                text = _letters(rng, 3) + _digits(rng, self.width)
                if text.startswith(self.prefix):
                    text = "Q" + text[1:]
            elif mode == 1:
                text = self.prefix + _digits(rng, self.width + rng.choice((-1, 1)))
            elif mode == 2:
                text = self.prefix + "--" + _digits(rng, self.width)
            else:
                text = self.prefix + _digits(rng, self.width) + "-"
            if text not in banned and not self.accepts(text):
                return text
        return self.prefix + "ABC"

    def annotate(self, text: str) -> CharAnn:
        rest = text[len(self.prefix) :]
        groups = [(_cells(self.prefix, KIND_LIT), (0, 0, 0))]
        if rest.startswith("-"):
            groups.append((_cells("-", KIND_OPT), (0, 1, 0)))
            groups.append((_cells(rest[1:], KIND_CLS), (0, 0, 0)))
        else:
            groups.append((_cells(rest, KIND_CLS), (0, 0, 0)))
        built, ann = paint(groups)
        if built != text:
            raise RuntimeError("optional annotation drifted")
        return ann


class OptionalInternal:
    """Two letters, an optional two-digit group, two letters. Eval only."""

    def __init__(self, rng: random.Random):
        self.left = _letters(rng, 2)
        self.right = _letters(rng, 2)
        self.width = 2
        self.pattern = f"optional-internal;left={self.left};right={self.right}"

    def accepts(self, text: str) -> bool:
        if text == self.left + self.right:
            return True
        if not (text.startswith(self.left) and text.endswith(self.right)):
            return False
        mid = text[len(self.left) : len(text) - len(self.right)]
        return len(mid) == self.width and mid.isdigit()

    def _draw(self, rng: random.Random, present: bool) -> str:
        mid = _digits(rng, self.width) if present else ""
        return self.left + mid + self.right

    def support(self, rng: random.Random) -> list[str]:
        return [self._draw(rng, False), self._draw(rng, True), self._draw(rng, bool(rng.randrange(2)))]

    def positive(self, rng: random.Random, support: list[str]) -> str:
        return _fresh(rng, lambda: self._draw(rng, bool(rng.randrange(2))), set(support))

    def negative(self, rng: random.Random, support: list[str]) -> str:
        banned = set(support)
        for _ in range(30):
            mode = rng.randrange(4)
            if mode == 0:
                text = self.left + _digits(rng, self.width + 1) + self.right
            elif mode == 1:
                text = self.left + self.right + _digits(rng, self.width)
            elif mode == 2:
                text = _digits(rng, self.width) + self.left + self.right
            else:
                text = self.left + "-" + _digits(rng, self.width) + self.right
            if text not in banned and not self.accepts(text):
                return text
        return self.left + "XX" + self.right

    def annotate(self, text: str) -> CharAnn:
        groups = [(_cells(self.left, KIND_LIT), (0, 0, 0))]
        if text != self.left + self.right:
            mid = text[len(self.left) : len(text) - len(self.right)]
            groups.append((_cells(mid, KIND_OPT), (0, 1, 0)))
        groups.append((_cells(self.right, KIND_LIT), (0, 0, 0)))
        built, ann = paint(groups)
        if built != text:
            raise RuntimeError("internal optional annotation drifted")
        return ann


class Alternation:
    """One three-letter literal or one four-digit string."""

    def __init__(self, rng: random.Random):
        self.literal = _letters(rng, 3)
        self.width = 4
        self.pattern = f"alt;lit={self.literal};width={self.width}"

    def accepts(self, text: str) -> bool:
        if text == self.literal:
            return True
        return len(text) == self.width and text.isdigit()

    def _draw(self, rng: random.Random, branch: str) -> str:
        if branch == "lit":
            return self.literal
        return _digits(rng, self.width)

    def support(self, rng: random.Random) -> list[str]:
        third = rng.choice(("lit", "digit"))
        return [self._draw(rng, "lit"), self._draw(rng, "digit"), self._draw(rng, third)]

    def positive(self, rng: random.Random, support: list[str]) -> str:
        branch = rng.choice(("lit", "digit"))
        return _fresh(rng, lambda: self._draw(rng, branch), set(support))

    def negative(self, rng: random.Random, support: list[str]) -> str:
        banned = set(support)
        for _ in range(30):
            mode = rng.randrange(3)
            if mode == 0:
                text = _letters(rng, 3)
                if text == self.literal:
                    text = "Q" + text[1:]
            elif mode == 1:
                text = _digits(rng, self.width + rng.choice((-1, 1)))
            else:
                text = self.literal + _digits(rng, 1)
            if text not in banned and not self.accepts(text):
                return text
        return self.literal.lower()

    def annotate(self, text: str) -> CharAnn:
        if text == self.literal:
            groups = [(_cells(text, KIND_LIT), (0, 0, 1))]
        else:
            groups = [(_cells(text, KIND_CLS), (0, 0, 1))]
        _built, ann = paint(groups)
        return ann


class PrefixAlt:
    """Constant prefix in front of (digits | literal). Eval only."""

    def __init__(self, rng: random.Random):
        self.prefix = _letters(rng, 2)
        self.literal = _letters(rng, 2)
        self.width = 3
        self.pattern = f"prefix-alt;prefix={self.prefix};lit={self.literal};width={self.width}"

    def accepts(self, text: str) -> bool:
        if not text.startswith(self.prefix):
            return False
        rest = text[len(self.prefix) :]
        return rest == self.literal or (len(rest) == self.width and rest.isdigit())

    def _draw(self, rng: random.Random, branch: str) -> str:
        rest = self.literal if branch == "lit" else _digits(rng, self.width)
        return self.prefix + rest

    def support(self, rng: random.Random) -> list[str]:
        third = rng.choice(("lit", "digit"))
        return [self._draw(rng, "lit"), self._draw(rng, "digit"), self._draw(rng, third)]

    def positive(self, rng: random.Random, support: list[str]) -> str:
        branch = rng.choice(("lit", "digit"))
        return _fresh(rng, lambda: self._draw(rng, branch), set(support))

    def negative(self, rng: random.Random, support: list[str]) -> str:
        banned = set(support)
        for _ in range(30):
            mode = rng.randrange(3)
            if mode == 0:
                text = _letters(rng, 2) + self.literal
                if text.startswith(self.prefix):
                    text = "Q" + text[1:]
            elif mode == 1:
                text = self.prefix + _digits(rng, self.width + 1)
            else:
                text = self.prefix + _letters(rng, 2)
                if text == self.prefix + self.literal:
                    text = self.prefix + "QQ"
            if text not in banned and not self.accepts(text):
                return text
        return self.literal + self.prefix

    def annotate(self, text: str) -> CharAnn:
        rest = text[len(self.prefix) :]
        groups = [(_cells(self.prefix, KIND_LIT), (0, 0, 0))]
        if rest == self.literal:
            groups.append((_cells(rest, KIND_LIT), (0, 0, 1)))
        else:
            groups.append((_cells(rest, KIND_CLS), (0, 0, 1)))
        built, ann = paint(groups)
        if built != text:
            raise RuntimeError("prefix-alt annotation drifted")
        return ann


class PrefixRepeat:
    """Constant prefix plus the repeat language. Eval only."""

    def __init__(self, rng: random.Random):
        self.prefix = _letters(rng, 2)
        self.repeat = RepeatNum(rng)
        self.pattern = f"prefix-repeat;prefix={self.prefix};{self.repeat.pattern}"

    def accepts(self, text: str) -> bool:
        return text.startswith(self.prefix) and self.repeat.accepts(text[len(self.prefix) :])

    def _draw(self, rng: random.Random, repeats: int) -> str:
        return self.prefix + self.repeat.sample(rng, repeats)

    def support(self, rng: random.Random) -> list[str]:
        return [self._draw(rng, repeats) for repeats in (1, 2, 3)]

    def positive(self, rng: random.Random, support: list[str]) -> str:
        return _fresh(rng, lambda: self._draw(rng, rng.choice((1, 2, 3))), set(support))

    def negative(self, rng: random.Random, support: list[str]) -> str:
        banned = set(support)
        for _ in range(20):
            if rng.randrange(2) == 0:
                text = self.repeat.sample(rng, rng.choice((1, 2, 3)))
            else:
                text = self.prefix + self.repeat._break(rng, self.repeat.sample(rng, rng.choice((1, 2, 3))))
            if text not in banned and not self.accepts(text):
                return text
        return self.prefix + "AA"

    def annotate(self, text: str) -> CharAnn:
        body = text[len(self.prefix) :]
        groups = [(_cells(self.prefix, KIND_LIT), (0, 0, 0))]
        parts = body.split(self.repeat.sep)
        groups.append((_cells(parts[0], KIND_CLS), (0, 0, 0)))
        for part in parts[1:]:
            cells = [(self.repeat.sep, KIND_DELIM)] + _cells(part, KIND_REPEAT)
            groups.append((cells, (1, 0, 0)))
        built, ann = paint(groups)
        if built != text:
            raise RuntimeError("prefix-repeat annotation drifted")
        return ann


class RepeatOptional:
    """Repeat language plus an optional hyphen and two-letter tail. Eval only."""

    def __init__(self, rng: random.Random):
        self.repeat = RepeatNum(rng)
        self.suffix = _letters(rng, 2)
        self.pattern = f"repeat-optional;suffix={self.suffix};{self.repeat.pattern}"

    def _tail(self) -> str:
        return "-" + self.suffix

    def _body_of(self, text: str) -> tuple[str, bool]:
        tail = self._tail()
        if text.endswith(tail) and self.repeat.accepts(text[: -len(tail)]):
            return text[: -len(tail)], True
        return text, False

    def accepts(self, text: str) -> bool:
        body, _present = self._body_of(text)
        return self.repeat.accepts(body)

    def _draw(self, rng: random.Random, repeats: int, tail: bool) -> str:
        body = self.repeat.sample(rng, repeats)
        return body + (self._tail() if tail else "")

    def support(self, rng: random.Random) -> list[str]:
        return [
            self._draw(rng, 1, False),
            self._draw(rng, 2, True),
            self._draw(rng, 3, False),
            self._draw(rng, 2, False),
        ]

    def positive(self, rng: random.Random, support: list[str]) -> str:
        return _fresh(
            rng,
            lambda: self._draw(rng, rng.choice((1, 2, 3)), bool(rng.randrange(2))),
            set(support),
        )

    def negative(self, rng: random.Random, support: list[str]) -> str:
        banned = set(support)
        for _ in range(20):
            mode = rng.randrange(3)
            if mode == 0:
                text = self.repeat._break(rng, self.repeat.sample(rng, rng.choice((1, 2, 3))))
            elif mode == 1:
                text = self.repeat.sample(rng, rng.choice((1, 2, 3))) + self.suffix
            else:
                text = self.repeat.sample(rng, 2) + "-" + _letters(rng, 2)
                if text.endswith(self._tail()):
                    text = text[:-1] + "Q"
            if text not in banned and not self.accepts(text):
                return text
        return "QQ" + self._tail()

    def annotate(self, text: str) -> CharAnn:
        body, present = self._body_of(text)
        parts = body.split(self.repeat.sep)
        groups = [(_cells(parts[0], KIND_CLS), (0, 0, 0))]
        for part in parts[1:]:
            cells = [(self.repeat.sep, KIND_DELIM)] + _cells(part, KIND_REPEAT)
            groups.append((cells, (1, 0, 0)))
        if present:
            groups.append((_cells(self._tail(), KIND_OPT), (0, 1, 0)))
        built, ann = paint(groups)
        if built != text:
            raise RuntimeError("repeat-optional annotation drifted")
        return ann


_BUILDERS = {
    "primitive": PrefixClass,
    "repeat": RepeatNum,
    "optional": OptionalTail,
    "alternation": Alternation,
    "repeat_extra": RepeatNum,
    "optional_internal": OptionalInternal,
    "prefix_repeat": PrefixRepeat,
    "repeat_optional": RepeatOptional,
    "prefix_alt": PrefixAlt,
}


def sample_episode(family: str, rng: random.Random, label: int | None = None) -> AnnotatedEpisode:
    if family not in _BUILDERS:
        raise KeyError(family)
    if label is None:
        label = rng.randrange(2)
    lang = _BUILDERS[family](rng)
    support = lang.support(rng)
    if family == "repeat_extra":
        repeats = rng.choice((4, 5))
        if label == 1:
            query = lang.positive(rng, support, repeats)
        else:
            query = lang.negative(rng, support, repeats)
    elif label == 1:
        query = lang.positive(rng, support)
    else:
        query = lang.negative(rng, support)
    return _finish_with_support(lang, family, label, support, query)


def _finish_with_support(lang, family: str, label: int, support: list[str], query: str) -> AnnotatedEpisode:
    if any(ch in HELD_OUT for text in list(support) + [query] for ch in text):
        raise RuntimeError("held-out symbol leaked into a training string")
    for text in support:
        if not lang.accepts(text):
            raise RuntimeError(f"support left the language: {text} / {lang.pattern}")
    if lang.accepts(query) != bool(label):
        raise RuntimeError(f"label disagrees with the language: {query} / {lang.pattern} / {label}")
    support_ann = [lang.annotate(text) for text in support]
    query_ann = lang.annotate(query) if label == 1 else ignore_ann(query)
    for text, ann in list(zip(support, support_ann)) + [(query, query_ann)]:
        if len(ann) != len(text):
            raise RuntimeError("annotation length drifted")
    return AnnotatedEpisode(support, query, label, family, lang.pattern, support_ann, query_ann)


def sample_batch(family: str, rng: random.Random, size: int) -> list[AnnotatedEpisode]:
    half = size // 2
    episodes = [sample_episode(family, rng, 1) for _ in range(half)]
    episodes += [sample_episode(family, rng, 0) for _ in range(size - half)]
    rng.shuffle(episodes)
    return episodes


def build_split(family: str, rng: random.Random, n: int) -> list[AnnotatedEpisode]:
    half = n // 2
    episodes = [sample_episode(family, rng, 1) for _ in range(half)]
    episodes += [sample_episode(family, rng, 0) for _ in range(n - half)]
    rng.shuffle(episodes)
    return episodes
