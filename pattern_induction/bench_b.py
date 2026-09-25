"""BENCH-B document-realistic synthetic tasks.

Each task is a latent document field with controlled realistic variation:
separators, widths, optional prefixes, and case. This is the large synthetic
stand-in for invoices, ids, dates, money, phones, and filenames.
"""

from __future__ import annotations

import random
import string

from pattern_induction.sample_util import build_task
from pattern_induction.tasks import Task


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))


def _upper(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(n))


def generate_bench_b(n: int = 2000, seed: int = 99) -> list[Task]:
    rng = random.Random(seed)
    prefixes = ["INV", "PO", "SO", "ORD", "BOL", "CMR", "REF", "DOC", "LOT", "BAT"]
    seps = ["-", "/", ".", ""]
    tasks: list[Task] = []
    index = 0
    while len(tasks) < n:
        kind = index % 12
        index += 1
        if kind == 0:
            prefix, width, sep = rng.choice(prefixes), rng.choice((4, 5, 6, 8)), rng.choice(seps)
            regex = rf"^{prefix}{re_sep(sep)}\d{{{width}}}$"
            sample = lambda rng, prefix=prefix, sep=sep, width=width: f"{prefix}{sep}{_digits(rng, width)}"
            domain, variant = "invoice_number", f"{prefix}{sep or 'none'}{width}"
            notes = "Document id with a fixed prefix and width."
        elif kind == 1:
            year = rng.choice((2024, 2025, 2026))
            width = rng.choice((4, 5, 6))
            regex = rf"^INV-{year}-\d{{{width}}}$"
            sample = lambda rng, year=year, width=width: f"INV-{year}-{_digits(rng, width)}"
            domain, variant = "invoice_number", f"year{year}w{width}"
            notes = "Year is constant in the latent schema."
        elif kind == 2:
            regex = r"^(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$"
            sample = lambda rng: f"{rng.randint(1995, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
            domain, variant = "date", "iso"
            notes = "ISO date."
        elif kind == 3:
            regex = r"^(?:0[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2])\.(?:19|20)\d{2}$"
            sample = lambda rng: f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(1998, 2026)}"
            domain, variant = "date", "eu_dot"
            notes = "Dotted European date."
        elif kind == 4:
            regex = r"^USD \d{1,3}(?:,\d{3})*\.\d{2}$"
            sample = lambda rng: f"USD {rng.randint(0, 25000):,}.{rng.randint(0, 99):02d}"
            domain, variant = "currency", "usd"
            notes = "USD grouped amount."
        elif kind == 5:
            code = rng.choice(("USD", "EUR", "GBP"))
            regex = rf"^\d+\.\d{{2}} {code}$"
            sample = lambda rng, code=code: f"{rng.randint(0, 4000)}.{rng.randint(0, 99):02d} {code}"
            domain, variant = "currency", code
            notes = "Trailing currency code, fixed for the column."
        elif kind == 6:
            regex = r"^\(\d{3}\) \d{3}-\d{4}$"
            sample = lambda rng: f"({_digits(rng, 3)}) {_digits(rng, 3)}-{_digits(rng, 4)}"
            domain, variant = "phone", "us"
            notes = "US phone."
        elif kind == 7:
            regex = r"^[A-Z]\d{2}_[A-Z]{2}$"
            sample = lambda rng: f"{_upper(rng, 1)}{_digits(rng, 2)}_{_upper(rng, 2)}"
            domain, variant = "sku", "country"
            notes = "Short SKU / country code."
        elif kind == 8:
            ext = rng.choice(("pdf", "csv", "xlsx", "txt"))
            regex = rf"^[a-z0-9_]+\.{ext}$"
            sample = lambda rng, ext=ext: (
                "".join(rng.choice(string.ascii_lowercase + string.digits + "_") for _ in range(rng.randint(4, 12)))
                + f".{ext}"
            )
            domain, variant = "filename", ext
            notes = "Single-extension filename column."
        elif kind == 9:
            regex = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
            sample = lambda rng: "-".join(
                "".join(rng.choice("0123456789abcdef") for _ in range(n)) for n in (8, 4, 4, 4, 12)
            )
            domain, variant = "uuid", "any"
            notes = "UUID-shaped identifier, version nibble not constrained."
        elif kind == 10:
            regex = r"^TXN-\d{6}-[A-Z]{3}$"
            sample = lambda rng: f"TXN-{_digits(rng, 6)}-{_upper(rng, 3)}"
            domain, variant = "transaction_id", "txn"
            notes = "Transaction id."
        else:
            regex = r"^\d{1,2}\.\d{1,2}\.\d{1,2}$"
            sample = lambda rng: f"{rng.randint(1, 15)}.{rng.randint(1, 20)}.{rng.randint(1, 30)}"
            domain, variant = "section_id", "dotted"
            notes = "Section number."
        # Many draws share a gold regex. That is intentional: document columns
        # repeat. Hidden strings still differ, and the id is unique.
        tasks.append(
            build_task(
                task_id=f"B-{len(tasks):05d}-{domain}-{variant}-{index}",
                benchmark="B",
                domain=domain,
                variant=variant,
                gold_regex=regex,
                sample=sample,
                rng=rng,
                latent_schema=regex,
                n_visible=rng.choice((4, 5, 6, 8)),
                n_pos=20,
                n_neg=20,
                split="test",
                notes=notes,
                meta={"family": kind},
            )
        )
    return tasks


def re_sep(sep: str) -> str:
    return {"": "", "-": r"\-", "/": "/", ".": r"\."}[sep]
