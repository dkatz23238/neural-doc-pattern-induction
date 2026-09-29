"""Smallest candidate-conditioned matcher.

Each candidate factor queries one support constituent. An integer node sees
the digit range. A repeat node sees whether a run repeats. An optional node
sees which runs are missing on some supports. A literal sees the last run's
role. Nothing here receives the pooled 160-d vector.

The recognizer stays frozen. Facts are read from the run layout and from the
numeric range of digit runs. Letter renaming does not move those facts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from torch import nn

from pattern_induction.lexicon import matching_lexicons
from pattern_induction.membership import Episode
from pattern_induction.program import MAX_LEN, MAX_SUPPORT, Prog
from pattern_induction.renaming import apply_mapping, featurize
from prior.constituents import (
    field_class,
    numeric_coverage,
    repeat_summary,
    run_presence,
    structure_matrix,
    suffix_role,
)
from prior.dataset import load_split
from prior.features import INDEX, candidate_features
from prior.loss import ranking_loss

_SPLIT = re.compile(r"[^A-Za-z0-9]+")

INTERVAL = (
    "int_present",
    "int_log_span",
    "int_exact_width",
    "int_covers",
    "int_hull",
    "int_convention",
    "int_round",
    "cls_digit",
    "digit_lo",
    "digit_hi",
    "digit_exact",
    "digit_open",
)
REPEAT = ("n_repeat",)
OPTIONAL = ("n_opt",)
AFFIX = ("n_copy", "n_literal", "mean_lit_len", "max_lit_len")
FIELD = (
    "cls_digit",
    "cls_upper",
    "cls_lower",
    "n_cls",
    "n_set",
    "set_present",
    "set_family",
    "set_size",
    "set_covers",
    "set_hull",
    "set_exact_width",
    "set_open",
)
LEXICON = (
    "n_lex",
    "lex_present",
    "lex_currency",
    "lex_month",
    "lex_http",
    "lex_extension",
    "lex_mass",
    "lex_size",
)
QUERIES = {
    "interval": INTERVAL,
    "repeat": REPEAT,
    "optional": OPTIONAL,
    "affix": AFFIX,
    "field": FIELD,
    "lexicon": LEXICON,
}
# Coverage, repeat flags, run presence, last-run role, field class, lexicon bit.
SUPPORT_SLICES = {
    "interval": (0, 1),
    "repeat": (1, 3),
    "optional": (3, 19),
    "affix": (19, 75),
    "field": (75, 80),
    "lexicon": (80, 81),
}
SUPPORT_FACT_DIM = 81
HIDDEN = {"interval": 8, "repeat": 8, "optional": 8, "affix": 8, "field": 8, "lexicon": 8}
RESIDUAL_HIDDEN = 32


def lexicon_bit(supports: list[str]) -> float:
    """1 when one aligned token column is a sample of a frozen lexicon."""
    parts = [[piece for piece in _SPLIT.split(text) if piece] for text in supports if text]
    if not parts or len({len(row) for row in parts}) != 1:
        return 0.0
    for index in range(len(parts[0])):
        column = [row[index] for row in parts]
        if len(set(column)) < 2 or not all(token.isalpha() for token in column):
            continue
        if matching_lexicons(set(column)):
            return 1.0
    return 0.0


def assemble_facts(structure: torch.Tensor, coverage: torch.Tensor, groups: list[list[str]]) -> torch.Tensor:
    """Stack the constituent a query is allowed to see. Shape [N, 81]."""
    lexicon = torch.tensor([[lexicon_bit(group)] for group in groups], dtype=torch.float32)
    return torch.cat(
        [
            coverage,
            repeat_summary(structure),
            run_presence(structure),
            suffix_role(structure),
            field_class(structure),
            lexicon,
        ],
        dim=-1,
    )


def facts_from_supports(groups: list[list[str]]) -> torch.Tensor:
    """Run layout plus digit range for a few supports. Used outside the cache."""
    episodes = [
        Episode(list(group[:MAX_SUPPORT]), group[0] if group else "", 1, "prior", "prior", "prior")
        for group in groups
    ]
    _local, classes, rel, pad = featurize(episodes, max_len=MAX_LEN, max_support=MAX_SUPPORT)
    structure = structure_matrix(classes, pad, rel)
    return assemble_facts(structure, numeric_coverage(groups), groups)


def _index(names: tuple[str, ...]) -> list[int]:
    return [INDEX[name] for name in names]


class _Query(nn.Module):
    """One candidate factor attends to one support constituent."""

    def __init__(self, support_dim: int, cand_dim: int, hidden: int):
        super().__init__()
        self.support_proj = nn.Linear(support_dim, hidden)
        self.cand_proj = nn.Linear(cand_dim, hidden)
        self.score = nn.Linear(hidden * 3, 1)

    def forward(self, support: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        left = self.support_proj(support).unsqueeze(1)
        right = self.cand_proj(candidate)
        bound = left * right
        packed = torch.cat([left.expand_as(right), right, bound], dim=-1)
        return self.score(packed).squeeze(-1)


class ConstituentMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.queries = nn.ModuleDict(
            {
                name: _Query(SUPPORT_SLICES[name][1] - SUPPORT_SLICES[name][0], len(QUERIES[name]), HIDDEN[name])
                for name in QUERIES
            }
        )
        self.residual = nn.Sequential(
            nn.Linear(len(INDEX), RESIDUAL_HIDDEN),
            nn.GELU(),
            nn.Linear(RESIDUAL_HIDDEN, 1),
        )
        self.register_buffer("mean", torch.zeros(SUPPORT_FACT_DIM))
        self.register_buffer("std", torch.ones(SUPPORT_FACT_DIM))

    def set_support_scale(self, facts: torch.Tensor) -> None:
        self.mean.copy_(facts.mean(dim=0))
        self.std.copy_(facts.std(dim=0).clamp(min=1e-6))

    def forward(self, support: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        scaled = (support - self.mean) / self.std
        total = self.residual(candidate).squeeze(-1)
        for name, query in self.queries.items():
            start, stop = SUPPORT_SLICES[name]
            columns = _index(QUERIES[name])
            total = total + query(scaled[:, start:stop], candidate[:, :, columns])
        return total


def parameter_count(model: ConstituentMatcher) -> int:
    return sum(int(param.numel()) for param in model.parameters())


def load_facts(split: str, data_dir: Path, cache: Path) -> torch.Tensor:
    path = cache / f"{split}-facts.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    from prior.audit import load_rows

    rows = load_rows(data_dir, split, with_programs=False)
    saved = torch.load(cache / f"{split}.pt", weights_only=True)
    coverage_path = cache / f"{split}-numeric.pt"
    if coverage_path.exists():
        coverage = torch.load(coverage_path, weights_only=True)
    else:
        coverage = numeric_coverage(rows["supports"])
        torch.save(coverage, coverage_path)
    print(f"assembling {split}", flush=True)
    facts = assemble_facts(saved["structure"], coverage, rows["supports"])
    if facts.shape[-1] != SUPPORT_FACT_DIM:
        raise RuntimeError(f"{split} facts are {facts.shape[-1]}-d, expected {SUPPORT_FACT_DIM}")
    torch.save(facts, path)
    return facts


@torch.no_grad()
def evaluate(model: ConstituentMatcher, support: torch.Tensor, candidate: torch.Tensor, mask: torch.Tensor, gold: torch.Tensor, family: list[str], batch_size: int = 256) -> dict:
    model.eval()
    totals = {"n": 0, "top1": 0.0}
    by_family: dict[str, list[float]] = {}
    for start in range(0, support.shape[0], batch_size):
        stop = min(support.shape[0], start + batch_size)
        scores = model(support[start:stop], candidate[start:stop])
        valid = mask[start:stop]
        scores = scores.masked_fill(~valid, -1e9)
        choice = scores.argmax(dim=-1)
        rows = torch.arange(choice.shape[0])
        hit = gold[start:stop][rows, choice]
        totals["n"] += int(choice.shape[0])
        totals["top1"] += float(hit.sum())
        for name, correct in zip(family[start:stop], hit.tolist()):
            by_family.setdefault(name, []).append(float(correct))
    n = max(totals["n"], 1)
    return {
        "n": totals["n"],
        "top1": totals["top1"] / n,
        "by_family": {name: sum(rows) / len(rows) for name, rows in sorted(by_family.items())},
    }


def renaming_invariance(model: ConstituentMatcher, n: int = 32, seed: int = 2507) -> float:
    """A within-class letter rename leaves the score in place."""
    import random
    import string

    rng = random.Random(seed)
    hits = 0
    model.eval()
    for _ in range(n):
        prefix = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
        supports = [prefix + text for text in ("2024", "2025", "2026", "2024")]
        mapping = {char: char for char in prefix}
        pool = [char for char in string.ascii_uppercase if char not in prefix]
        rng.shuffle(pool)
        for char, target in zip(prefix, pool):
            mapping[char] = target
        renamed = [apply_mapping(text, mapping) for text in supports]
        program = Prog(
            "SEQ",
            children=[
                Prog("COPY", slot=0, start=0, end=3),
                Prog("INT", spec="2000:2099", lo=4, hi=4),
            ],
        )
        left = facts_from_supports([supports])
        right = facts_from_supports([renamed])
        feats = torch.tensor(
            [candidate_features(program, supports), candidate_features(program, renamed)],
            dtype=torch.float32,
        )
        scores = model(torch.cat([left, right], dim=0), feats[:, None, :])
        hits += int(torch.allclose(scores[0], scores[1], atol=1e-5))
    return hits / float(n)


def train_one(seed: int, data_dir: Path, cache: Path, out: Path, steps: int = 20000, batch_size: int = 32) -> dict:
    torch.manual_seed(seed)
    print(f"loading splits seed {seed}", flush=True)
    train = load_split(data_dir, "train", kind="mlp")
    val = load_split(data_dir, "val", kind="mlp")
    ood = load_split(data_dir, "ood", kind="mlp")
    facts = {split: load_facts(split, data_dir, cache) for split in ("train", "val", "ood")}
    model = ConstituentMatcher()
    model.set_support_scale(facts["train"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    order = torch.arange(train.support.shape[0])
    best = {"val_top1": -1.0, "step": 0, "state": None}
    history = []
    step = 0
    while step < steps:
        perm = order[torch.randperm(order.shape[0])]
        for start in range(0, perm.shape[0], batch_size):
            batch = perm[start : start + batch_size]
            if batch.shape[0] == 0:
                continue
            opt.zero_grad(set_to_none=True)
            scores = model(facts["train"][batch], train.candidate[batch])
            loss = ranking_loss(scores, train.gold[batch], train.mask[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 2000 == 0 or step >= steps:
                val_report = evaluate(model, facts["val"], val.candidate, val.mask, val.gold, val.family)
                print(f"seed {seed} step {step} val {val_report['top1']:.4f}", flush=True)
                history.append({"step": step, "val_top1": val_report["top1"]})
                if val_report["top1"] >= best["val_top1"]:
                    best = {
                        "val_top1": val_report["top1"],
                        "step": step,
                        "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    }
            if step >= steps:
                break
    model.load_state_dict(best["state"])
    val_report = evaluate(model, facts["val"], val.candidate, val.mask, val.gold, val.family)
    ood_report = evaluate(model, facts["ood"], ood.candidate, ood.mask, ood.gold, ood.family)
    invariance = renaming_invariance(model)
    report = {
        "seed": seed,
        "parameters": parameter_count(model),
        "best_step": best["step"],
        "val_top1": val_report["top1"],
        "ood_top1": ood_report["top1"],
        "val_by_family": val_report["by_family"],
        "ood_by_family": ood_report["by_family"],
        "renaming_invariance": invariance,
        "history": history,
    }
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "report": report}, out / "checkpoint.pt")
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"seed {seed} best {best['step']} val {val_report['top1']:.4f} ood {ood_report['top1']:.4f} rename {invariance:.3f}",
        flush=True,
    )
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("results/prior/shards"))
    parser.add_argument("--cache", type=Path, default=Path("/tmp/constituent-state"))
    parser.add_argument("--out", type=Path, default=Path("results/prior/runs/constituent_match"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--steps", type=int, default=20000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    reports = []
    for seed in args.seeds:
        reports.append(train_one(seed, args.data, args.cache, args.out / f"seed_{seed}", steps=args.steps))
    summary = {
        "parameters": reports[0]["parameters"],
        "seeds": reports,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    dest = Path("results/prior/constituent_match.json")
    dest.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
