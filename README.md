# Needle

Needle turns a few positive example strings into a compact regular expression. The score is semantic: hidden positives should match, and hidden negatives should not. Two examples are the minimum. One example abstains.

`needle()` loads the frozen scope recognizer and the frozen witness head. It does not train. The learned candidate prior in `prior/` is a separate experiment; `needle()` still uses the grammar prior saved with the witness head.

## Install

[uv](https://docs.astral.sh/uv/) is the environment. From a fresh checkout:

```bash
uv sync
```

That installs this repo in editable mode with torch, NumPy, PyYAML, pyarrow, and `find-abstract-syntax-tree==0.2.1`. `uv run` does not need `PYTHONPATH`.

`pip install -r requirements.txt` installs only the fAST pin. With the packages on `PYTHONPATH`, the same commands below work as `PYTHONPATH=. python ...`.

## Use it

Run from the repo root. The call reads:

- `results/scope/p2b/seed_0/checkpoint.pt`
- `results/select/s1/seed_0/checkpoint.pt`

```bash
uv run python -c 'from pattern_induction.needle import needle; print(needle(["CUS-DE1234", "CUS-SO9281", "CUS-AM0042"]))'
```

```python
from pattern_induction.needle import needle

result = needle(["CUS-DE1234", "CUS-SO9281", "CUS-AM0042"])
result.regex
result.alternatives
result.confidence
```

Supports are clipped to four strings and 32 characters. Retraining those two checkpoints is in [DOCS.md](DOCS.md).

```bash
uv run python -m unittest discover -s tests
```

## Current experiment

Track B is a teacher-driven concept bank. A Terra call invents realistic string fields. The raw response is committed before parsing. A concept is kept only when Needle's executor accepts every positive and rejects every near miss. The large training shards are not stored; `data/factory/manifests/factory_v1.json` names the bank hash, the parser version, and the seed.

gpt-6-terra is not on this key. Calls use gpt-5.6-terra. The spend cap in this module is $4, separate from the older recipe-teacher cap.

```bash
uv run python -m pattern_induction.factory --stage generate --prompt real_world_concepts_v1.md --domain logistics
uv run python -m pattern_induction.factory --stage diversity
uv run python -m pattern_induction.factory --stage preview --limit 4
```

Prompts live in `data/factory/prompts/`. One call does not train `needle()`. The expansion prior and the packet audit are in [DOCS.md](DOCS.md).

The written record is already in the tree. Regenerating it only concatenates the result pages, including the sealed BENCH-E line:

```bash
uv run python scripts/run_needle.py --stage summary
```

| Page | What it records |
| --- | --- |
| `results/needle/summary.md` | Needle, then the prior curves |
| `results/factory/bank.md` | Accepted concept bank and its diversity |
| `results/prior/expansion.md` | Tiny prior over ambiguous expansions |
| `results/prior/packets.md` | Identifiability of the local packets |
| `results/prior/relative.md` | Candidate-relative node scores |
| `results/prior/constituents.md` | Constituent probes and the candidate-conditioned matcher |
| `results/prior/probes.md` | Probes on the frozen 160-d support vector |
| `results/prior/relations.md` | Typed field, optional, and affix relations |
| `results/prior/factor.md` | Sum of local factor scores, CPU |
| `results/prior/ast.md` | Recursive candidate encoder |
| `results/prior/diversity.md` | Compositional sampler and the flat 10k / 30k / 100k sweep |
| `results/prior/scaling.md` | Earlier menu-sampler curve |

The first bank is whatever the executor accepted from the committed Terra runs. Row count is not the diversity measure: the page reports unique programs, domain entropy, and operator-signature entropy. BENCH-E stays sealed.
