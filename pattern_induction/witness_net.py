"""A small head on the frozen relational encoder.

The question is narrow: given the supports and one string where candidate
languages disagree, is that string in the language? The encoder stays frozen.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pattern_induction.membership import Episode
from pattern_induction.program import MAX_LEN, MAX_SUPPORT
from pattern_induction.renaming import featurize
from pattern_induction.span_net import _attend, _mean_max


def _kind(ch: str) -> str:
    if ch.isdigit():
        return "digit"
    if ch.isupper():
        return "upper"
    if ch.islower():
        return "lower"
    return "punct"


def _runs(text: str) -> list[tuple[str, str]]:
    runs: list[tuple[str, str]] = []
    if not text:
        return runs
    start = 0
    for index in range(1, len(text) + 1):
        if index == len(text) or _kind(text[index]) != _kind(text[start]):
            runs.append((_kind(text[start]), text[start:index]))
            start = index
    return runs


def _run_lengths(text: str, kind: str) -> list[int]:
    return [len(content) for name, content in _runs(text) if name == kind]


def _shared_affix(supports: list[str], from_end: bool) -> str:
    if not supports:
        return ""
    affix = supports[0]
    for text in supports[1:]:
        while affix and not (text.endswith(affix) if from_end else text.startswith(affix)):
            affix = affix[:-1] if not from_end else affix[1:]
    return affix


def _retint_flags(supports: list[str], query: str) -> tuple[float, float, float]:
    """Digit retint, fixed-letter retint, varying-letter retint. Same skeleton only."""
    query_runs = _runs(query)
    skeleton = tuple((name, len(content)) for name, content in query_runs)
    matched = []
    for text in supports:
        runs = _runs(text)
        if tuple((name, len(content)) for name, content in runs) == skeleton:
            matched.append(runs)
    if not matched:
        return 0.0, 0.0, 0.0
    nearest = min(matched, key=lambda runs: sum(left[1] != right[1] for left, right in zip(query_runs, runs)))
    changed = [index for index, (left, right) in enumerate(zip(query_runs, nearest)) if left[1] != right[1]]
    if not changed:
        return 0.0, 0.0, 0.0
    kinds = {query_runs[index][0] for index in changed}
    if kinds == {"digit"}:
        return 1.0, 0.0, 0.0
    if kinds and kinds <= {"upper", "lower"}:
        fixed = True
        for index in changed:
            observed = {runs[index][1] for runs in matched}
            if len(observed) != 1:
                fixed = False
        if fixed:
            return 0.0, 1.0, 0.0
        return 0.0, 0.0, 1.0
    return 0.0, 0.0, 0.0


# Excess width, a broken letter affix, a digit retint, or a fixed-letter retint.
# These are the disagreements the geometry head can see directly. Deficits stay
# out: a shorter run is still inside an open lower bound.
DECISIVE = (8, 10, 12, 14, 15, 16, 17)


def geometry(supports: list[str], query: str) -> list[float]:
    """Class-and-count features. Renaming inside a class leaves them unchanged.

    The last flags name the disagreement the head is asked to judge: a class
    run past the widths in the supports, a broken letter affix, or a same-width
    retint of digits versus a fixed letter span.
    """

    def run(text: str, pred) -> int:
        best = current = 0
        for ch in text:
            if pred(ch):
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    def puncts(text: str) -> int:
        return sum(not ch.isalnum() for ch in text)

    def span(kind: str) -> tuple[float, float]:
        query_len = max(_run_lengths(query, kind), default=0)
        observed = [length for text in supports for length in _run_lengths(text, kind)]
        if not observed or query_len == 0:
            return 0.0, 0.0
        return float(query_len > max(observed)), float(query_len < min(observed))

    longest = max((len(text) for text in supports), default=1)
    widest = max((run(text, str.isdigit) for text in supports), default=0)
    support_puncts = {ch for text in supports for ch in text if not ch.isalnum()}
    query_puncts = {ch for ch in query if not ch.isalnum()}
    digit_excess, digit_deficit = span("digit")
    upper_excess, upper_deficit = span("upper")
    lower_excess, lower_deficit = span("lower")
    prefix = _shared_affix(supports, False)
    suffix = _shared_affix(supports, True)
    broke_prefix = float(len(prefix) >= 2 and prefix.isalpha() and not query.startswith(prefix))
    broke_suffix = float(len(suffix) >= 2 and suffix.isalpha() and not query.endswith(suffix))
    retint_digit, retint_letter_fixed, retint_letter_varying = _retint_flags(supports, query)
    return [
        len(query) / 16.0,
        longest / 16.0,
        (len(query) - longest) / 16.0,
        puncts(query) / 4.0,
        max((puncts(text) for text in supports), default=0) / 4.0,
        run(query, str.isdigit) / 4.0,
        widest / 4.0,
        float(query_puncts <= support_puncts),
        digit_excess,
        digit_deficit,
        upper_excess,
        upper_deficit,
        lower_excess,
        lower_deficit,
        broke_prefix,
        broke_suffix,
        retint_digit,
        retint_letter_fixed,
        retint_letter_varying,
    ]


def collate_witnesses(pairs: list[tuple[list[str], str]]) -> dict[str, torch.Tensor]:
    episodes = [
        Episode(supports[:MAX_SUPPORT], query[:MAX_LEN], 1, "witness", "s1", "witness") for supports, query in pairs
    ]
    local, classes, rel, pad = featurize(episodes, max_len=MAX_LEN, max_support=MAX_SUPPORT)
    geom = torch.tensor([geometry(supports, query[:MAX_LEN]) for supports, query in pairs], dtype=torch.float32)
    return {"local": local, "classes": classes, "rel": rel, "pad": pad, "geom": geom}


class WitnessNet(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        dim = encoder.d_model
        self.n_geom = len(geometry(["0"], "0"))
        self.head = nn.Sequential(nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, 1))
        self.geom_head = nn.Linear(self.n_geom, 1)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.geom_head.weight)
        nn.init.zeros_(self.geom_head.bias)
        self.__dict__["encoder"] = encoder

    def latent(self, batch) -> torch.Tensor:
        encoder = self.encoder
        encoder.eval()
        local, classes, rel, pad = batch["local"], batch["classes"], batch["rel"], batch["pad"]
        _batch, slots, length = local.shape
        roles = torch.zeros(slots, dtype=torch.long)
        roles[-1] = 1
        which = torch.arange(slots)
        hidden = encoder.class_emb(classes) + encoder.local_emb(local) + encoder.rel_proj(rel)
        hidden = hidden + encoder.char_pos[:length][None, None, :, :]
        hidden = hidden + encoder.role(roles)[None, :, None, :] + encoder.which(which)[None, :, None, :]
        hidden = hidden * (~pad).unsqueeze(-1).float()
        with torch.no_grad():
            encoded = _attend(encoder.char_encoder, hidden, pad)
            support_pad = pad[:, :-1].reshape(pad.shape[0], -1)
            support = encoded[:, :-1].reshape(encoded.shape[0], -1, encoded.shape[-1])
            # _mean_max expects pad aligned with the flattened support characters.
            support_mean, support_max = _mean_max(support, support_pad)
            query_mean, query_max = _mean_max(encoded[:, -1], pad[:, -1])
        return torch.cat([support_mean, support_max, query_mean, query_max, batch["geom"]], dim=-1)

    def forward(self, batch) -> torch.Tensor:
        features = self.latent(batch)
        encoded = features[:, : -self.n_geom]
        geom = features[:, -self.n_geom :]
        enc_logit = self.head(encoded).squeeze(-1)
        geom_logit = self.geom_head(geom).squeeze(-1)
        decisive = torch.zeros(geom.shape[0], dtype=torch.bool, device=geom.device)
        for index in DECISIVE:
            decisive = decisive | (geom[:, index] > 0)
        # A pooled encoding treats a digit retint like any other edit. When the
        # contrast flags name the edit, the geometry head is the whole vote.
        return torch.where(decisive, geom_logit, enc_logit + geom_logit)


def witness_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weight = torch.where(target > 0.5, torch.full_like(target, 2.0), torch.ones_like(target))
    loss = F.binary_cross_entropy_with_logits(logit, target, weight=weight)
    positive = logit[target > 0.5]
    negative = logit[target < 0.5]
    if positive.numel() and negative.numel():
        loss = loss + 0.5 * F.softplus(negative[:, None] - positive[None, :]).mean()
    return loss
