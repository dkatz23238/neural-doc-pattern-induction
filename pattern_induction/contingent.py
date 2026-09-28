"""Optional suffixes licensed by a delimiter some supports omit.

A punctuation character that occurs in a strict subset of the supports is a
contingent delimiter. When that character is the last separator in a support,
the suffix from that delimiter token through the end of the string is an
optional hypothesis. A separator that appears in every support, such as a
repeat mark, does not fire. Assembly still requires every support to match,
so a suffix that is actually required never becomes a program.
"""

from __future__ import annotations

from pattern_induction.chart import Hyp
from pattern_induction.renaming import PUNCT, structural_class


def contingent_optionals(supports: list[str]) -> list[Hyp]:
    """One optional hypothesis per support whose last separator is contingent."""
    if len(supports) < 2:
        return []
    present: dict[str, int] = {}
    for text in supports:
        for ch in set(text):
            if structural_class(ch) == PUNCT:
                present[ch] = present.get(ch, 0) + 1
    rare = {ch for ch, count in present.items() if count < len(supports)}
    if not rare:
        return []
    hyps: list[Hyp] = []
    seen: set[tuple[int, int, int]] = set()
    for slot, text in enumerate(supports):
        last = -1
        for index, ch in enumerate(text):
            if structural_class(ch) == PUNCT:
                last = index
        if last < 0 or text[last] not in rare:
            continue
        start = last
        while start > 0 and text[start - 1] == text[last]:
            start -= 1
        if start <= 0 or start >= len(text):
            continue
        key = (slot, start, len(text))
        if key in seen:
            continue
        seen.add(key)
        hyps.append(Hyp("optional", slot=slot, start=start, end=len(text), score=0.0))
    return hyps
