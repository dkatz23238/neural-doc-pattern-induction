"""Fit needle()'s confidence to BENCH-C outcomes.

The label is that the code-length winner scores at least 0.99 on the hidden
strings. The recognizer's signals count only when it finished inside the
time limit needle() uses. Every reported number is out of fold: all tasks of
one document domain are held out together, and the ridge strength is chosen
inside the remaining domains. The final weights are fit on every task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pattern_induction.confidence import FIT_PATH, NAMES, Signals, features
from pattern_induction.needle import TIME_LIMIT, _confidence

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "results" / "needle" / "confidence_features.jsonl"
PAGE = ROOT / "results" / "needle" / "confidence.md"
LAMBDAS = (0.1, 1.0, 10.0, 100.0)
CUT = 0.99


def _signals(row: dict, supports: list[str], limit: float) -> Signals:
    answered = bool(row["learned"]) and not row["timeout"] and row["recognizer_seconds"] <= limit
    return Signals(
        regex=row["regex"],
        supports=supports,
        margin=float("inf") if row["margin"] is None else row["margin"],
        near_tie=bool(row["near_tie"]),
        n_alternatives=row["n_alternatives"],
        cost=row["cost"],
        learned=answered,
        agrees=answered and bool(row["agrees"]),
        prior_conf=row["prior_conf"] if answered else None,
        n_programs=(row["n_programs"] or 0) if answered else 0,
    )


def _fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    reg = np.full(Z.shape[1], lam)
    reg[0] = 0.0
    for _ in range(100):
        p = 1 / (1 + np.exp(-Z @ w))
        grad = Z.T @ (p - y) + reg * w
        hess = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(reg) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return mu, sd, w


def _predict(model, X):
    mu, sd, w = model
    return 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), (X - mu) / sd] @ w)))


def _log_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def _choose_lambda(X, y, groups, mask):
    best = None
    for lam in LAMBDAS:
        loss = 0.0
        for held in np.unique(groups[mask]):
            test = mask & (groups == held)
            train = mask & (groups != held)
            loss += _log_loss(_predict(_fit(X[train], y[train], lam), X[test]), y[test])
        if best is None or loss < best[0]:
            best = (loss, lam)
    return best[1]


def _auroc(score, label):
    pos, neg = score[label == 1], score[label == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def _ece(score, label, bins=10):
    total = 0.0
    index = np.minimum((score * bins).astype(int), bins - 1)
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.mean() * abs(score[mask].mean() - label[mask].mean())
    return float(total)


def _metrics(score, y) -> dict:
    return {
        "auroc": _auroc(score, y),
        "brier": float(np.mean((score - y) ** 2)),
        "log_loss": _log_loss(score, y) / len(y),
        "ece": _ece(score, y),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    args = parser.parse_args()
    visible = {}
    for line in (ROOT / "data" / "bench_c.jsonl").open():
        task = json.loads(line)
        visible[task["id"]] = [text for text in task["visible"] if text]
    rows = [json.loads(line) for line in ROWS.open()]
    signals = [_signals(row, visible[row["id"]], args.time_limit) for row in rows]
    X = np.array([features(item) for item in signals], dtype=float)
    y = np.array([row["accuracy"] >= CUT for row in rows], dtype=float)
    groups = np.array([row["domain"] for row in rows])
    v1 = np.array([
        _confidence(item.margin, (not item.learned) or item.agrees, item.prior_conf if item.agrees else None, item.near_tie)
        for item in signals
    ])

    held_out = np.zeros(len(y))
    for held in np.unique(groups):
        test = groups == held
        lam = _choose_lambda(X, y, groups, ~test)
        held_out[test] = _predict(_fit(X[~test], y[~test], lam), X[test])

    lam = _choose_lambda(X, y, groups, np.ones(len(y), dtype=bool))
    mu, sd, w = _fit(X, y, lam)
    FIT_PATH_ABS = ROOT / FIT_PATH
    FIT_PATH_ABS.write_text(json.dumps({
        "target": f"code-length winner scores >= {CUT} on BENCH-C hidden strings",
        "time_limit": args.time_limit,
        "ridge": lam,
        "names": list(NAMES),
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "w": w.tolist(),
    }, indent=2) + "\n")

    base = np.full(len(y), y.mean())
    table = {
        "always the base rate": _metrics(base, y),
        "hand-set v1 formula": _metrics(v1, y),
        "fitted, held-out domains": _metrics(held_out, y),
    }
    lines = [
        "# Fitted confidence",
        "",
        f"`needle()`'s confidence is a logistic model over signals it already computes. The label is that the code-length winner scores at least {CUT} on a BENCH-C task's hidden strings; {int(y.sum())} of {len(y)} tasks do. The recognizer's signals count only when it finished within the {args.time_limit:g}-second limit `needle()` uses, measured from the timings in `confidence_features.jsonl` (three workers sharing the CPU, so slower than a single call). Every score below is out of fold: all tasks of one domain ({len(np.unique(groups))} domains) are held out together, and the ridge strength is chosen inside the remaining domains. The weights in `confidence_fit.json` are fit on every task with ridge {lam:g}. BENCH-E is not used.",
        "",
        "| Score | AUROC | Brier | Log loss | ECE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in table.items():
        lines.append(f"| {name} | {row['auroc']:.3f} | {row['brier']:.3f} | {row['log_loss']:.3f} | {row['ece']:.3f} |")
    lines += ["", "Reliability of the held-out fitted score:", "", "| Score | Tasks | Mean score | Winner correct |", "| --- | ---: | ---: | ---: |"]
    for lo, hi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)):
        mask = (held_out >= lo) & (held_out < hi)
        if mask.any():
            lines.append(f"| {lo:.2f}–{min(hi, 1.0):.2f} | {int(mask.sum())} | {held_out[mask].mean():.3f} | {y[mask].mean():.3f} |")
    lines += ["", "Share correct among the most confident tasks:", "", "| Kept | Fitted | Hand-set |", "| --- | ---: | ---: |"]
    for share in (0.1, 0.25, 0.5, 0.75, 1.0):
        k = max(1, int(share * len(y)))
        fitted_top = y[np.argsort(-held_out, kind="stable")[:k]].mean()
        v1_top = y[np.argsort(-v1, kind="stable")[:k]].mean()
        lines.append(f"| {int(share * 100)}% | {fitted_top:.3f} | {v1_top:.3f} |")
    lines += ["", "Weights on standardized features:", "", "| Weight | Feature |", "| ---: | --- |"]
    for name, weight in sorted(zip(NAMES, w[1:]), key=lambda item: -abs(item[1])):
        lines.append(f"| {weight:+.3f} | {name.replace('|', chr(92) + '|')} |")
    PAGE.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
