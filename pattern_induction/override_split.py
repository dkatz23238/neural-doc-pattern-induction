"""Train and validation compositions for the residual override model.

Training and validation share document families and use disjoint formats.
Easy columns shift prefixes, separators, and widths. Hard columns shift the
same phenomena (variable repeats, optional counters, sentinel mixtures) onto
a different surface, so validation is not solved by memorizing a schema.
BENCH-C is not produced here.
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable
from dataclasses import dataclass

from pattern_induction.bench_b import re_sep
from pattern_induction.sample_util import build_task
from pattern_induction.tasks import Task


@dataclass(frozen=True)
class Composition:
    family: str
    name: str
    regex: str
    sample: Callable[[random.Random], str]
    notes: str


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))


def _upper(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(n))


def _lower(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def _grouped_us(amount: int) -> str:
    return f"{amount:,}"


def _grouped_eu(amount: int) -> str:
    return f"{amount:,}".replace(",", ".")


def _sep_regex(sep: str) -> str:
    if sep == " ":
        return " "
    if sep == "_":
        return "_"
    return re_sep(sep)


def _invoice(prefixes: tuple[str, ...], seps: tuple[str, ...], widths: tuple[int, ...]) -> list[Composition]:
    rows = []
    for prefix in prefixes:
        for sep in seps:
            for width in widths:
                regex = rf"^{prefix}{_sep_regex(sep)}\d{{{width}}}$"
                rows.append(
                    Composition(
                        "invoice_number",
                        f"{prefix}{sep or 'none'}{width}",
                        regex,
                        lambda rng, prefix=prefix, sep=sep, width=width: f"{prefix}{sep}{_digits(rng, width)}",
                        "Fixed prefix and digit width.",
                    )
                )
    return rows


def _years(prefixes: tuple[str, ...], seps: tuple[str, ...], years: tuple[int, ...], widths: tuple[int, ...]) -> list[Composition]:
    rows = []
    for prefix in prefixes:
        for sep in seps:
            for year in years:
                for width in widths:
                    regex = rf"^{prefix}{_sep_regex(sep)}{year}{_sep_regex(sep)}\d{{{width}}}$"
                    rows.append(
                        Composition(
                            "invoice_year",
                            f"{prefix}{sep or 'none'}{year}w{width}",
                            regex,
                            lambda rng, prefix=prefix, sep=sep, year=year, width=width: (
                                f"{prefix}{sep}{year}{sep}{_digits(rng, width)}"
                            ),
                            "Calendar year is a literal.",
                        )
                    )
    return rows


def _dates_iso_dash() -> list[Composition]:
    regex = r"^(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$"
    return [
        Composition(
            "date",
            "iso_dash",
            regex,
            lambda rng: f"{rng.randint(1995, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "ISO date with dashes.",
        )
    ]


def _dates_iso_slash() -> list[Composition]:
    regex = r"^(?:19|20)\d{2}/(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])$"
    return [
        Composition(
            "date",
            "iso_slash",
            regex,
            lambda rng: f"{rng.randint(1995, 2026)}/{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}",
            "ISO date with slashes.",
        )
    ]


def _dates_eu_dot() -> list[Composition]:
    regex = r"^(?:0[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2])\.(?:19|20)\d{2}$"
    return [
        Composition(
            "date",
            "eu_dot",
            regex,
            lambda rng: f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(1998, 2026)}",
            "European date with dots.",
        )
    ]


def _dates_eu_slash() -> list[Composition]:
    regex = r"^(?:0[1-9]|[12]\d|3[01])/(?:0[1-9]|1[0-2])/(?:19|20)\d{2}$"
    return [
        Composition(
            "date",
            "eu_slash",
            regex,
            lambda rng: f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1998, 2026)}",
            "European date with slashes.",
        )
    ]


def _dates_us_dash() -> list[Composition]:
    regex = r"^(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])-(?:19|20)\d{2}$"
    return [
        Composition(
            "date",
            "us_dash",
            regex,
            lambda rng: f"{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}-{rng.randint(1998, 2026)}",
            "US date with dashes.",
        )
    ]


def _currency_prefix_usd() -> list[Composition]:
    regex = r"^USD \d{1,3}(?:,\d{3})*\.\d{2}$"
    return [
        Composition(
            "currency",
            "usd_prefix",
            regex,
            lambda rng: f"USD {_grouped_us(rng.randint(0, 25000))}.{rng.randint(0, 99):02d}",
            "USD prefix and comma grouping.",
        )
    ]


def _currency_suffix(codes: tuple[str, ...]) -> list[Composition]:
    rows = []
    for code in codes:
        regex = rf"^\d+\.\d{{2}} {code}$"
        rows.append(
            Composition(
                "currency",
                f"suffix_{code}",
                regex,
                lambda rng, code=code: f"{rng.randint(0, 4000)}.{rng.randint(0, 99):02d} {code}",
                "Trailing currency code.",
            )
        )
    return rows


def _currency_eu_prefix() -> list[Composition]:
    regex = r"^EUR \d{1,3}(?:\.\d{3})*,\d{2}$"
    return [
        Composition(
            "currency",
            "eur_dot_group",
            regex,
            lambda rng: f"EUR {_grouped_eu(rng.randint(0, 25000))},{rng.randint(0, 99):02d}",
            "EUR prefix, dot grouping, comma cents.",
        )
    ]


def _currency_dollar() -> list[Composition]:
    regex = r"^\$\d{1,3}(?:,\d{3})*\.\d{2}$"
    return [
        Composition(
            "currency",
            "dollar_sign",
            regex,
            lambda rng: f"${_grouped_us(rng.randint(0, 25000))}.{rng.randint(0, 99):02d}",
            "Dollar sign and comma grouping.",
        )
    ]


def _phone_us() -> list[Composition]:
    regex = r"^\(\d{3}\) \d{3}-\d{4}$"
    return [
        Composition(
            "phone",
            "us_paren",
            regex,
            lambda rng: f"({_digits(rng, 3)}) {_digits(rng, 3)}-{_digits(rng, 4)}",
            "US phone with parentheses.",
        )
    ]


def _phone_dot() -> list[Composition]:
    regex = r"^\d{3}\.\d{3}\.\d{4}$"
    return [
        Composition(
            "phone",
            "dotted",
            regex,
            lambda rng: f"{_digits(rng, 3)}.{_digits(rng, 3)}.{_digits(rng, 4)}",
            "Dotted phone.",
        )
    ]


def _phone_plus() -> list[Composition]:
    regex = r"^\+1-\d{3}-\d{3}-\d{4}$"
    return [
        Composition(
            "phone",
            "plus1",
            regex,
            lambda rng: f"+1-{_digits(rng, 3)}-{_digits(rng, 3)}-{_digits(rng, 4)}",
            "Country-code phone.",
        )
    ]


def _sku_underscore() -> list[Composition]:
    regex = r"^[A-Z]\d{2}_[A-Z]{2}$"
    return [
        Composition(
            "sku",
            "letter_underscore",
            regex,
            lambda rng: f"{_upper(rng, 1)}{_digits(rng, 2)}_{_upper(rng, 2)}",
            "Letter, two digits, underscore, two letters.",
        )
    ]


def _sku_hyphen() -> list[Composition]:
    regex = r"^[A-Z]{2}-\d{4}$"
    return [
        Composition(
            "sku",
            "two_hyphen_four",
            regex,
            lambda rng: f"{_upper(rng, 2)}-{_digits(rng, 4)}",
            "Two letters, hyphen, four digits.",
        )
    ]


def _sku_slash() -> list[Composition]:
    regex = r"^[A-Z]{2}/\d{2}/[A-Z]{2}$"
    return [
        Composition(
            "sku",
            "slash_parts",
            regex,
            lambda rng: f"{_upper(rng, 2)}/{_digits(rng, 2)}/{_upper(rng, 2)}",
            "Slash-separated SKU.",
        )
    ]


def _filenames(extensions: tuple[str, ...], upper: bool) -> list[Composition]:
    rows = []
    for ext in extensions:
        if upper:
            regex = rf"^[A-Z0-9]+(?:-[A-Z0-9]+)*\.{ext}$"
            sample = lambda rng, ext=ext: (
                "-".join(_upper(rng, rng.randint(3, 6)) for _ in range(rng.randint(1, 3))) + f".{ext}"
            )
            notes = "Uppercase filename with a held-out extension."
        else:
            regex = rf"^[a-z0-9_]+\.{ext}$"
            sample = lambda rng, ext=ext: (
                "".join(rng.choice(string.ascii_lowercase + string.digits + "_") for _ in range(rng.randint(4, 12)))
                + f".{ext}"
            )
            notes = "Lowercase filename."
        rows.append(Composition("filename", ext, regex, sample, notes))
    return rows


def _uuid(alphabet: str, name: str) -> list[Composition]:
    regex = rf"^[{alphabet}]{{8}}-[{alphabet}]{{4}}-[{alphabet}]{{4}}-[{alphabet}]{{4}}-[{alphabet}]{{12}}$"
    choices = "0123456789abcdef" if alphabet == "0-9a-f" else "0123456789ABCDEF"
    return [
        Composition(
            "uuid",
            name,
            regex,
            lambda rng, choices=choices: "-".join(
                "".join(rng.choice(choices) for _ in range(n)) for n in (8, 4, 4, 4, 12)
            ),
            "UUID-shaped identifier.",
        )
    ]


def _txn_upper() -> list[Composition]:
    regex = r"^TXN-\d{6}-[A-Z]{3}$"
    return [
        Composition(
            "transaction",
            "txn_upper",
            regex,
            lambda rng: f"TXN-{_digits(rng, 6)}-{_upper(rng, 3)}",
            "Transaction id.",
        )
    ]


def _txn_lower() -> list[Composition]:
    regex = r"^pay_\d{8}_[a-z]{2}$"
    return [
        Composition(
            "transaction",
            "pay_lower",
            regex,
            lambda rng: f"pay_{_digits(rng, 8)}_{_lower(rng, 2)}",
            "Lowercase payment id with a longer body.",
        )
    ]


def _variable_sections(sep: str, family_note: str) -> list[Composition]:
    escaped = _sep_regex(sep)
    regex = rf"^\d{{1,2}}(?:{escaped}\d{{1,2}}){{1,3}}$"
    return [
        Composition(
            "repeat_section",
            f"var_{sep or 'none'}",
            regex,
            lambda rng, sep=sep: sep.join(str(rng.randint(1, 12)) for _ in range(rng.randint(2, 4))),
            family_note,
        )
    ]


def _marked_sections() -> list[Composition]:
    regex = r"^§\d{1,2}(?:\.\d{1,2}){1,3}$"
    return [
        Composition(
            "repeat_section",
            "section_symbol",
            regex,
            lambda rng: "§" + ".".join(str(rng.randint(1, 12)) for _ in range(rng.randint(2, 4))),
            "Section mark with a variable number of dotted parts.",
        )
    ]


def _optional_files(extensions: tuple[str, ...]) -> list[Composition]:
    rows = []
    for ext in extensions:
        regex = rf"^[a-z0-9_]+(?: \(\d+\))?\.{ext}$"
        rows.append(
            Composition(
                "optional_file",
                f"opt_{ext}",
                regex,
                lambda rng, ext=ext: (
                    "".join(rng.choice(string.ascii_lowercase + string.digits + "_") for _ in range(rng.randint(4, 10)))
                    + (f" ({rng.randint(1, 20)})" if rng.random() < 0.5 else "")
                    + f".{ext}"
                ),
                "Filename with an optional parenthesized counter.",
            )
        )
    return rows


def _sentinel(sep: str, token: str, name: str) -> list[Composition]:
    escaped = _sep_regex(sep)
    regex = rf"^(?:(?:\d{{4}}{escaped}\d{{2}}{escaped}\d{{2}})|(?:{token}))$"
    return [
        Composition(
            "sentinel",
            name,
            regex,
            lambda rng, sep=sep, token=token: (
                token
                if rng.random() < 0.35
                else f"{rng.randint(1998, 2026)}{sep}{rng.randint(1, 12):02d}{sep}{rng.randint(1, 28):02d}"
            ),
            "A date column mixed with one sentinel token.",
        )
    ]


def _region(prefixes: tuple[str, ...], digits: int, letters: int) -> list[Composition]:
    rows = []
    for prefix in prefixes:
        regex = rf"^{prefix}-\d{{{digits}}}-[A-Z]{{{letters}}}$"
        rows.append(
            Composition(
                "region_code",
                f"{prefix}_d{digits}_l{letters}",
                regex,
                lambda rng, prefix=prefix, digits=digits, letters=letters: (
                    f"{prefix}-{_digits(rng, digits)}-{_upper(rng, letters)}"
                ),
                "Prefix, digits, and a letter code.",
            )
        )
    return rows


def train_compositions() -> list[Composition]:
    """Formats the residual model may memorize."""
    rows: list[Composition] = []
    rows.extend(_invoice(("INV", "PO", "SO", "ORD", "BOL", "CMR", "REF", "DOC"), ("-", "/", ".", ""), (4, 5, 6, 8)))
    rows.extend(_years(("INV",), ("-",), (2024, 2025, 2026), (4, 5, 6)))
    rows.extend(_dates_iso_dash())
    rows.extend(_dates_eu_dot())
    rows.extend(_currency_prefix_usd())
    rows.extend(_currency_suffix(("USD", "EUR", "GBP")))
    rows.extend(_phone_us())
    rows.extend(_sku_underscore())
    rows.extend(_filenames(("pdf", "csv", "xlsx", "txt"), upper=False))
    rows.extend(_uuid("0-9a-f", "lower"))
    rows.extend(_txn_upper())
    rows.extend(_variable_sections("-", "Variable-depth hyphenated sections."))
    rows.extend(_optional_files(("txt", "csv")))
    rows.extend(_sentinel("-", "N/A", "na_dash"))
    rows.extend(_region(("USR", "EMP", "CUS"), 3, 2))
    return rows


def val_compositions() -> list[Composition]:
    """Same families, held-out prefixes, separators, and widths."""
    rows: list[Composition] = []
    rows.extend(_invoice(("ACK", "SHP", "RMA", "PKG"), ("_", " "), (3, 7, 9)))
    rows.extend(_years(("PO", "DOC"), ("/",), (2018, 2019, 2023), (7,)))
    rows.extend(_dates_iso_slash())
    rows.extend(_dates_eu_slash())
    rows.extend(_dates_us_dash())
    rows.extend(_currency_eu_prefix())
    rows.extend(_currency_dollar())
    rows.extend(_phone_dot())
    rows.extend(_phone_plus())
    rows.extend(_sku_hyphen())
    rows.extend(_sku_slash())
    rows.extend(_filenames(("docx", "log"), upper=True))
    rows.extend(_uuid("0-9A-F", "upper"))
    rows.extend(_txn_lower())
    rows.extend(_variable_sections(".", "Variable-depth dotted sections."))
    rows.extend(_marked_sections())
    rows.extend(_optional_files(("log",)))
    rows.extend(_sentinel("/", "NONE", "none_slash"))
    rows.extend(_region(("WH", "BIN"), 4, 3))
    return rows


def composition_regexes(split: str) -> set[str]:
    rows = train_compositions() if split == "train" else val_compositions()
    return {row.regex for row in rows}


def _by_family(rows: list[Composition]) -> dict[str, list[Composition]]:
    grouped: dict[str, list[Composition]] = {}
    for row in rows:
        grouped.setdefault(row.family, []).append(row)
    return grouped


def generate_split(n: int, split: str, seed: int) -> list[Task]:
    """Sample families evenly so frequent invoice schemas do not crowd out the rest."""
    catalog = train_compositions() if split == "train" else val_compositions()
    families = list(_by_family(catalog).items())
    rng = random.Random(seed)
    tasks: list[Task] = []
    for index in range(n):
        family, options = families[index % len(families)]
        comp = options[rng.randrange(len(options))]
        tasks.append(
            build_task(
                task_id=f"{'OT' if split == 'train' else 'OV'}-{index:05d}-{family}-{comp.name}",
                benchmark="R",
                domain=family,
                variant=comp.name,
                gold_regex=comp.regex,
                sample=comp.sample,
                rng=rng,
                latent_schema=comp.regex,
                n_visible=rng.choice((4, 5, 6, 8)),
                n_pos=20,
                n_neg=20,
                split=split,
                notes=comp.notes,
                meta={"family": family, "composition": comp.name, "split": split},
            )
        )
    return tasks
