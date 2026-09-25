"""Render results/exp1/exp1_report.html from the Experiment 1 JSON."""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "exp1" / "exp1_report.html"

BENCH_COPY = {
    "bench_c": (
        "BENCH-C",
        "554 curated document fields. Each task has a specified gold regex, 24 hidden positives, and 24 near-miss negatives. Held out of any future training split. Invoice-number columns are capped at 24 tasks.",
    ),
    "bench_b": (
        "BENCH-B",
        "1,500 document-style synthetic tasks from the same generator family.",
    ),
    "bench_d": (
        "BENCH-D",
        "320 tasks on the same schemas, shown with 2, 3, 5, 10, or 20 visible examples.",
    ),
    "bench_a": (
        "BENCH-A",
        "1,200-task structural split. The generator is the same function used for the 100k/10k/10k draw; this run keeps the review set small.",
    ),
}

LABELS = {
    "memorize": "Memorization",
    "heuristic": "Heuristic",
    "grex_default": "grex default",
    "grex_digits": "grex digits",
    "grex_words": "grex words",
    "grex_repetitions": "grex repetitions",
    "grex_combined": "grex combined",
    "fast": "fAST",
    "gpt-6-luna": "GPT-6 Luna",
    "gpt-5.6-terra": "GPT-5.6 Terra",
}


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def esc(value: object) -> str:
    return html.escape(str(value))


def load() -> tuple[dict, dict]:
    report = json.loads((ROOT / "results" / "exp1" / "exp1_report.json").read_text())
    frontier = json.loads((ROOT / "results" / "exp1" / "frontier_bench_c.json").read_text())
    return report, frontier


def system_rows(report: dict, frontier: dict) -> dict[str, list[dict]]:
    rows = {}
    for name, payload in report["benchmarks"].items():
        systems = list(payload["systems"])
        if name == "bench_c":
            systems.extend(frontier["systems"])
        rows[name] = systems
    return rows


def bar(value: float, kind: str) -> str:
    width = max(0.0, min(value, 1.0)) * 100
    return (
        f'<div class="bar {kind}"><span style="width:{width:.1f}%"></span>'
        f"<b>{pct(value)}</b></div>"
    )


def table(systems: list[dict], latency_note: str) -> str:
    body = []
    for row in systems:
        name = row["system"]
        klass = "frontier" if name.startswith("gpt-") else name
        lat = row["latency_ms"]
        body.append(
            "<tr class='{klass}'>"
            "<td>{name}</td>"
            "<td>{sem}</td><td>{macro}</td><td>{pos}</td><td>{neg}</td>"
            "<td>{valid}</td><td>{cat}</td><td>{p50}</td><td>{p95}</td>"
            "</tr>".format(
                klass=esc(klass),
                name=esc(LABELS.get(name, name)),
                sem=bar(row["semantic_accuracy"], "sem"),
                macro=pct(row["macro_semantic_accuracy"]),
                pos=pct(row["positive_recall"]),
                neg=pct(row["negative_rejection"]),
                valid=pct(row["valid_output_rate"]),
                cat=pct(row["catastrophic_rate"]),
                p50=f"{lat['p50']:.2f}",
                p95=f"{lat['p95']:.2f}",
            )
        )
    return f"""
    <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>System</th><th>Semantic</th><th>Macro</th><th>Positive recall</th>
          <th>Negative rejection</th><th>Valid</th><th>Catastrophic</th>
          <th>p50 ms</th><th>p95 ms</th>
        </tr>
      </thead>
      <tbody>{''.join(body)}</tbody>
    </table>
    </div>
    <p class="note">{esc(latency_note)}</p>
    """


def taxonomy_block(report: dict, frontier: dict) -> str:
    cards = []
    merged = dict(report["benchmarks"]["bench_c"]["taxonomy"])
    for system in frontier["systems"]:
        merged[system["system"]] = system["taxonomy"]
    order = [
        "heuristic",
        "grex_combined",
        "gpt-6-luna",
        "gpt-5.6-terra",
        "memorize",
        "fast",
    ]
    for name in order:
        counts = merged[name]
        chips = []
        for label, info in list(counts.items())[:6]:
            chips.append(f"<li><span>{esc(label)}</span><b>{info['n']}</b></li>")
        cards.append(
            f"<article><h3>{esc(LABELS.get(name, name))}</h3><ul>{''.join(chips)}</ul></article>"
        )
    return f'<div class="tax">{"".join(cards)}</div>'


def domain_table(report: dict, frontier: dict) -> str:
    heuristic = next(s for s in report["benchmarks"]["bench_c"]["systems"] if s["system"] == "heuristic")
    luna = frontier["systems"][0]["by_domain"]
    terra = frontier["systems"][1]["by_domain"]
    rows = []
    for domain, base in heuristic["by_domain"].items():
        rows.append((domain, base, luna[domain], terra[domain], terra[domain] - base))
    rows.sort(key=lambda item: item[4])
    body = []
    for domain, base, left, right, gap in rows:
        tone = "gap-pos" if gap >= 0.02 else "gap-neg" if gap <= -0.02 else ""
        body.append(
            "<tr class='{tone}'><td>{domain}</td><td>{base}</td><td>{left}</td><td>{right}</td><td>{gap}</td></tr>".format(
                tone=tone,
                domain=esc(domain.replace("_", " ")),
                base=pct(base),
                left=pct(left),
                right=pct(right),
                gap=f"{gap * 100:+.1f}",
            )
        )
    return f"""
    <div class="table-wrap">
    <table>
      <thead><tr><th>Domain</th><th>Heuristic</th><th>GPT-6 Luna</th><th>GPT-5.6 Terra</th><th>Terra − heuristic</th></tr></thead>
      <tbody>{''.join(body)}</tbody>
    </table>
    </div>
    <p class="note">Sorted by Terra’s gap against the heuristic. Macro accuracy is the unweighted mean of these domain scores.</p>
    """


def efficiency(report: dict) -> str:
    items = []
    for n, acc in report["sample_efficiency"]:
        items.append(
            f'<div class="eff"><em>{n}</em>{bar(acc, "sem")}</div>'
        )
    return f'<div class="effs">{"".join(items)}</div>'


def render() -> str:
    report, frontier = load()
    rows = system_rows(report, frontier)
    luna = frontier["systems"][0]
    terra = frontier["systems"][1]
    heuristic = next(s for s in report["benchmarks"]["bench_c"]["systems"] if s["system"] == "heuristic")
    h = heuristic["semantic_accuracy"]
    sections = []
    notes = {
        "bench_c": "Frontier latency is API round trip. Symbolic latency is this cloud VM CPU. fAST’s zeros are timeouts on 8 short tasks, not judged regexes.",
        "bench_b": "fAST returned no expression on 5 of 8 short tasks. A zero is a timeout.",
        "bench_d": "Overall heuristic accuracy here mixes every sample size. The curve below splits it. fAST timed out on 8 of 8 short tasks.",
        "bench_a": "grex combined is the strongest symbolic system on this structural split. fAST timed out on 8 of 8 short tasks.",
    }
    for key in ("bench_c", "bench_b", "bench_d", "bench_a"):
        title, blurb = BENCH_COPY[key]
        sections.append(
            f"<section id='{key}'><h2>{esc(title)}</h2><p>{esc(blurb)}</p>"
            f"{table(rows[key], notes[key])}</section>"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment 1 — Gate 1 report</title>
<style>
  :root {{
    --ink: #1c1915;
    --muted: #5c564c;
    --line: #e2d9c8;
    --paper: #f6f1e7;
    --card: #fffdf8;
    --no: #8d2e2a;
    --no-bg: #f8e8e2;
    --yes: #1f6b4a;
    --bar: #2c4a3e;
    --frontier: #8a5a12;
    --gold: #c4a574;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    color: var(--ink);
    background: var(--paper);
    font: 17px/1.5 "Iowan Old Style", Palatino, "Palatino Linotype", Georgia, serif;
  }}
  header, main {{ width: min(1080px, calc(100% - 32px)); margin: 0 auto; }}
  header {{ padding: 48px 0 12px; }}
  .kicker {{
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font: 12px/1.2 "Avenir Next", "Segoe UI", sans-serif;
    color: var(--muted);
  }}
  h1 {{ font-size: 46px; line-height: 1.05; font-weight: 600; margin: 8px 0 12px; }}
  h2 {{ font-size: 28px; margin: 0 0 8px; }}
  h3 {{ font-size: 16px; margin: 0 0 8px; font-family: "Avenir Next", "Segoe UI", sans-serif; }}
  p {{ margin: 0 0 12px; }}
  a {{ color: var(--no); }}
  .decision {{
    background: var(--no-bg);
    border: 1px solid #e3c2b8;
    border-radius: 16px;
    padding: 22px 24px;
    margin: 20px 0 28px;
  }}
  .decision strong {{
    display: block;
    font: 600 13px/1 "Avenir Next", "Segoe UI", sans-serif;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--no);
    margin-bottom: 8px;
  }}
  .stats {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 18px 0 8px;
  }}
  .stat {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 14px 16px;
  }}
  .stat b {{ display: block; font-size: 28px; line-height: 1.1; }}
  .stat span {{ color: var(--muted); font: 13px/1.3 "Avenir Next", "Segoe UI", sans-serif; }}
  .rule {{
    margin: 8px 0 28px;
    color: var(--muted);
    font: 14px/1.45 "Avenir Next", "Segoe UI", sans-serif;
  }}
  nav {{
    display: flex; flex-wrap: wrap; gap: 8px;
    margin: 8px 0 28px;
    font: 14px/1 "Avenir Next", "Segoe UI", sans-serif;
  }}
  nav a {{
    text-decoration: none;
    color: var(--ink);
    border: 1px solid var(--line);
    background: var(--card);
    border-radius: 999px;
    padding: 8px 12px;
  }}
  section {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 22px 22px 8px;
    margin-bottom: 16px;
  }}
  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font: 14px/1.35 "Avenir Next", "Segoe UI", sans-serif; }}
  th, td {{ text-align: left; padding: 8px 8px; border-bottom: 1px solid var(--line); vertical-align: middle; white-space: nowrap; }}
  th {{ color: var(--muted); font-weight: 600; font-size: 12px; letter-spacing: 0.04em; text-transform: uppercase; }}
  tr.heuristic td:first-child {{ color: var(--yes); font-weight: 700; }}
  tr.frontier td:first-child {{ color: var(--frontier); font-weight: 700; }}
  .bar {{ display: flex; align-items: center; gap: 8px; min-width: 140px; }}
  .bar span {{ display: block; height: 8px; background: var(--bar); border-radius: 99px; }}
  .bar b {{ font-weight: 650; }}
  .note {{ color: var(--muted); font: 13px/1.4 "Avenir Next", "Segoe UI", sans-serif; }}
  .tax {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 12px; }}
  .tax article {{ border: 1px solid var(--line); border-radius: 12px; padding: 12px; }}
  .tax ul {{ list-style: none; margin: 0; padding: 0; }}
  .tax li {{ display: flex; justify-content: space-between; gap: 8px; padding: 3px 0; border-bottom: 1px solid var(--line); font: 13px/1.3 "Avenir Next", "Segoe UI", sans-serif; }}
  .effs {{ display: grid; gap: 8px; margin-bottom: 12px; }}
  .eff {{ display: grid; grid-template-columns: 36px 1fr; align-items: center; gap: 8px; }}
  .eff em {{ font-style: normal; font: 600 14px/1 "Avenir Next", "Segoe UI", sans-serif; }}
  .gap-pos td:last-child {{ color: var(--yes); font-weight: 700; }}
  .gap-neg td:last-child {{ color: var(--no); font-weight: 700; }}
  footer {{ color: var(--muted); font: 13px/1.4 "Avenir Next", "Segoe UI", sans-serif; padding: 8px 0 40px; }}
  @media (max-width: 800px) {{
    h1 {{ font-size: 34px; }}
    .stats, .tax {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<header>
  <div class="kicker">Document pattern induction · 25 September 2026</div>
  <h1>Experiment 1 findings</h1>
  <p>Can a frontier model infer a useful regular expression from a handful of examples, or do symbolic systems already close the gap? The score is semantic accuracy on hidden strings, not how similar the regex text is to the gold pattern.</p>
  <div class="decision">
    <strong>Gate 1 · no</strong>
    <p>The rule is an 8–10 point gap over the best symbolic system on BENCH-C. The heuristic is at {pct(h)}. GPT-6 Luna is {pct(luna['semantic_accuracy'])} ({(luna['semantic_accuracy']-h)*100:+.1f} points). GPT-5.6 Terra is {pct(terra['semantic_accuracy'])} ({(terra['semantic_accuracy']-h)*100:+.1f} points). An 8-point clearance would be {pct(h+0.08)}, which is past the top of this metric. No model was trained.</p>
  </div>
  <div class="stats">
    <div class="stat"><b>{pct(h)}</b><span>Heuristic semantic accuracy · BENCH-C · n={heuristic['n']}</span></div>
    <div class="stat"><b>{pct(luna['semantic_accuracy'])}</b><span>GPT-6 Luna · macro {pct(luna['macro_semantic_accuracy'])}</span></div>
    <div class="stat"><b>{pct(terra['semantic_accuracy'])}</b><span>GPT-5.6 Terra · macro {pct(terra['macro_semantic_accuracy'])}</span></div>
    <div class="stat"><b>+4.9</b><span>Largest gap, in points. The rule asks for 8–10.</span></div>
  </div>
  <p class="rule">Semantic accuracy = (hidden positive recall + hidden negative rejection) / 2. A regex is scored by what it accepts. <code>.*</code> takes the positives and fails the negatives. An alternation of the visible strings rejects the negatives and misses the hidden positives. Both land near 50%.</p>
  <nav>
    <a href="#bench_c">BENCH-C</a>
    <a href="#domains">Domains</a>
    <a href="#failures">Failures</a>
    <a href="#bench_b">BENCH-B</a>
    <a href="#bench_d">BENCH-D</a>
    <a href="#bench_a">BENCH-A</a>
    <a href="#protocol">Protocol</a>
  </nav>
</header>
<main>
  {sections[0]}
  <section id="domains">
    <h2>Where the models move the score</h2>
    <p>Macro accuracy keeps a large easy column from deciding the headline. Terra’s largest gains are hashes (+35.4 points), UUIDs (+23.8), serials (+21.4), and transaction ids (+18.8). It stays within 2 points of the heuristic on every other domain. Luna drops on phone numbers (86.5% against the heuristic’s 99.0%), URLs (73.8% against 92.9%), and bank references (65.7% against 87.6%).</p>
    {domain_table(report, frontier)}
  </section>
  <section id="failures">
    <h2>BENCH-C failure taxonomy</h2>
    <p>A success is semantic accuracy at least 90%, with positive recall and negative rejection both at least 85%. The heuristic’s remaining misses are mostly optional regions, variable-length spans, and character classes narrower than “alphanumeric.”</p>
    {taxonomy_block(report, frontier)}
  </section>
  {sections[1]}
  <section id="bench_d">
    <h2>BENCH-D</h2>
    <p>{esc(BENCH_COPY['bench_d'][1])}</p>
    {table(rows['bench_d'], notes['bench_d'])}
    <h3>Heuristic sample efficiency</h3>
    {efficiency(report)}
  </section>
  {sections[3]}
  <section id="protocol">
    <h2>Protocol</h2>
    <p>Frontier calls used the simple prompt on <code>data/bench_c.jsonl</code>, 554 tasks, 6 visible strings each. Date 2026-09-25. The requested Terra model was <code>gpt-6-terra</code>. That id returns <code>model_not_found</code>. The Terra model on this key is <code>gpt-5.6-terra</code>. Neither model accepts temperature 0: chat completions allows only the default value 1, and the Responses API rejects the parameter, so temperature was omitted.</p>
    <p>Default grex matches the memorization floor: it rejects negatives and misses hidden positives. Combined grex (<code>-d -w -r</code>) is the useful grex setting, at 86.6% on BENCH-C. fAST (ICGI 2023, find-abstract-syntax-tree 0.2.1) is wired in, including a translation of its concatenation operator. On this suite it did not finish. Within the cap it returned no expression on the short tasks it was given. A worked example still converts correctly: <code>abc</code>, <code>abcabc</code>, <code>abcabcabc</code> becomes <code>^(?:abc)+$</code>.</p>
    <p>BENCH-C is a curated catalog of document fields. It is not a scrape of private PDFs. Hidden negatives are near-miss mutations outside the gold regex.</p>
  </section>
  <footer>Source tables: results/exp1/exp1_report.md and results/exp1/frontier_bench_c.json. No neural model was trained.</footer>
</main>
</body>
</html>
"""


def main() -> None:
    OUT.write_text(render(), encoding="utf-8")
    print(OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
