"""BENCH-C: curated document-pattern tasks with human-specified gold regexes.

These tasks are held out of every training split. Gold patterns are the
intended generalization for each visible sample, and hidden positives and
negatives are drawn from that decision.
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable

from pattern_induction.sample_util import build_task
from pattern_induction.tasks import Task

DIGITS = string.digits
UPPER = string.ascii_uppercase
LOWER = string.ascii_lowercase
ALNUM = string.ascii_letters + string.digits


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(DIGITS) for _ in range(n))


def _upper(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(UPPER) for _ in range(n))


def _lower(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(LOWER) for _ in range(n))


def _family(
    families: list,
    domain: str,
    name: str,
    regex: str,
    sample: Callable[[random.Random], str],
    notes: str,
    schema: str | None = None,
) -> None:
    families.append((domain, name, regex, sample, notes, schema or regex))


def _build_families() -> list:
    families: list = []
    add = lambda *args, **kwargs: _family(families, *args, **kwargs)

    for prefix in ("INV", "PO", "SO", "ORD", "CN"):
        for width in (4, 5, 6, 8):
            add(
                "invoice_number",
                f"{prefix}_w{width}",
                rf"^{prefix}-\d{{{width}}}$",
                lambda rng, p=prefix, w=width: f"{p}-{_digits(rng, w)}",
                f"Fixed prefix {prefix} and a {width}-digit body.",
            )
        add(
            "invoice_number",
            f"{prefix}_year",
            rf"^{prefix}-20(?:2[4-9]|3[0-5])-\d{{5}}$",
            lambda rng, p=prefix: f"{p}-{rng.randint(2024, 2035)}-{_digits(rng, 5)}",
            "Year varies inside a closed range; the serial is five digits.",
        )
        add(
            "invoice_number",
            f"{prefix}_fixed_year",
            rf"^{prefix}-2026-\d{{5}}$",
            lambda rng, p=prefix: f"{p}-2026-{_digits(rng, 5)}",
            "Calendar year is part of the literal prefix, not a free digit field.",
        )

    for prefix in ("USR", "EMP", "CUS"):
        add(
            "customer_id",
            f"{prefix}_state",
            rf"^{prefix}-\d{{3}}-[A-Z]{{2}}$",
            lambda rng, p=prefix: f"{p}-{_digits(rng, 3)}-{_upper(rng, 2)}",
            "Prefix, three-digit id, two-letter region.",
        )
        add(
            "customer_id",
            f"{prefix}_alnum",
            rf"^{prefix}-[A-Z]{{2}}\d{{4}}$",
            lambda rng, p=prefix: f"{p}-{_upper(rng, 2)}{_digits(rng, 4)}",
            "Two letters then four digits after the prefix.",
        )

    add(
        "order_id",
        "compact_letter_digits",
        r"^[A-Z]\d{2}_[A-Z]{2}$",
        lambda rng: f"{_upper(rng, 1)}{_digits(rng, 2)}_{_upper(rng, 2)}",
        "One letter, two digits, underscore, country code.",
    )
    add(
        "order_id",
        "warehouse_bin",
        r"^WH-[A-Z]-\d{2}-\d{3}$",
        lambda rng: f"WH-{_upper(rng, 1)}-{_digits(rng, 2)}-{_digits(rng, 3)}",
        "Warehouse, aisle letter, bay, bin.",
    )
    for width in (6, 8, 10):
        add(
            "account_reference",
            f"acct_{width}",
            rf"^ACCT-\d{{{width}}}$",
            lambda rng, w=width: f"ACCT-{_digits(rng, w)}",
            f"Account reference with a {width}-digit body.",
        )
    add(
        "account_reference",
        "iban_like",
        r"^DE\d{2}(?:\d{4}){4}\d{2}$",
        lambda rng: "DE" + _digits(rng, 20),
        "German-style IBAN shape: country, check digits, 18-digit BBAN.",
    )
    add(
        "account_reference",
        "sort_account",
        r"^\d{2}-\d{2}-\d{2} \d{8}$",
        lambda rng: f"{_digits(rng, 2)}-{_digits(rng, 2)}-{_digits(rng, 2)} {_digits(rng, 8)}",
        "UK sort code plus eight-digit account.",
    )

    add(
        "date",
        "iso",
        r"^(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$",
        lambda rng: f"{rng.randint(1990, 2026):04d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "ISO calendar date. Day is capped at 28 so every sample is a real date.",
    )
    add(
        "date",
        "us_slash",
        r"^(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/(?:19|20)\d{2}$",
        lambda rng: f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.randint(1998, 2026)}",
        "US numeric date with slashes.",
    )
    add(
        "date",
        "eu_dot",
        r"^(?:0[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2])\.(?:19|20)\d{2}$",
        lambda rng: f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(1998, 2026)}",
        "European numeric date with dots.",
    )
    add(
        "date",
        "eu_slash_short_year",
        r"^(?:0[1-9]|[12]\d|3[01])/(?:0[1-9]|1[0-2])/\d{2}$",
        lambda rng: f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(0, 99):02d}",
        "Day/month/two-digit year.",
    )
    months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    add(
        "date",
        "d_mon_yyyy",
        r"^\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (?:19|20)\d{2}$",
        lambda rng, months=months: f"{rng.randint(1, 28)} {rng.choice(months)} {rng.randint(1990, 2026)}",
        "Day, English month abbreviation, four-digit year.",
    )

    add(
        "time",
        "hhmm",
        r"^(?:[01]\d|2[0-3]):[0-5]\d$",
        lambda rng: f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}",
        "24-hour HH:MM.",
    )
    add(
        "time",
        "hhmmss",
        r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$",
        lambda rng: f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}",
        "24-hour HH:MM:SS.",
    )
    add(
        "time",
        "ampm",
        r"^(?:0?[1-9]|1[0-2]):[0-5]\d (?:AM|PM)$",
        lambda rng: f"{rng.randint(1, 12)}:{rng.randint(0, 59):02d} {rng.choice(('AM', 'PM'))}",
        "12-hour time with a required meridiem.",
    )

    def usd(rng: random.Random) -> str:
        cents = f"{rng.randint(0, 99):02d}"
        amount = rng.randint(0, 19999)
        body = f"{amount:,}"
        return f"USD {body}.{cents}"

    add(
        "currency",
        "usd_prefix",
        r"^USD \d{1,3}(?:,\d{3})*\.\d{2}$",
        usd,
        "US dollar prefix, thousands commas, two decimal digits.",
    )

    def eur(rng: random.Random) -> str:
        cents = f"{rng.randint(0, 99):02d}"
        whole = rng.randint(0, 19999)
        body = f"{whole:,}".replace(",", " ")
        return f"{body},{cents} €"

    add(
        "currency",
        "eur_suffix",
        r"^\d{1,3}(?: \d{3})*,\d{2} €$",
        eur,
        "Euro suffix, space-grouped thousands, comma decimal.",
    )
    add(
        "currency",
        "plain_symbol",
        r"^\$\d{1,3}(?:,\d{3})*\.\d{2}$",
        lambda rng: "$" + f"{rng.randint(0, 9999):,}" + f".{rng.randint(0, 99):02d}",
        "Dollar sign glued to a grouped amount.",
    )
    add(
        "percentage",
        "one_decimal",
        r"^\d{1,2}\.\d%$",
        lambda rng: f"{rng.randint(0, 99)}.{rng.randint(0, 9)}%",
        "Percent with one decimal place.",
    )
    add(
        "percentage",
        "integer",
        r"^(?:100|\d{1,2})%$",
        lambda rng: f"{rng.randint(0, 100)}%",
        "Integer percent from 0 to 100.",
    )
    add(
        "measurement",
        "kg",
        r"^\d{1,3}(?:\.\d{1,2})? kg$",
        lambda rng: (
            f"{rng.randint(1, 200)} kg"
            if rng.random() < 0.4
            else f"{rng.randint(1, 200)}.{rng.randint(0, 99):02d}".rstrip("0").rstrip(".") + " kg"
            if False
            else f"{rng.randint(0, 40)}.{rng.randint(1, 9)} kg"
        ),
        "Kilogram quantity with an optional single decimal. Sampler always emits one decimal.",
    )
    add(
        "measurement",
        "mm_int",
        r"^\d{1,4} mm$",
        lambda rng: f"{rng.randint(1, 2500)} mm",
        "Integer millimetres.",
    )
    add(
        "page_reference",
        "p_dot",
        r"^p\. \d{1,3}$",
        lambda rng: f"p. {rng.randint(1, 320)}",
        "Page marker with a literal 'p.'.",
    )
    add(
        "page_reference",
        "page_range",
        r"^pp\. \d{1,3}–\d{1,3}$",
        lambda rng: (
            lambda a, b: f"pp. {a}–{b}"
        )(rng.randint(1, 40), rng.randint(41, 120)),
        "Page range using an en dash.",
    )
    add(
        "section_id",
        "dotted_3",
        r"^\d{1,2}\.\d{1,2}\.\d{1,2}$",
        lambda rng: f"{rng.randint(1, 12)}.{rng.randint(1, 14)}.{rng.randint(1, 20)}",
        "Three-level section number.",
    )
    add(
        "section_id",
        "appendix",
        r"^Appendix [A-Z](?:\.\d{1,2})?$",
        lambda rng: (
            f"Appendix {_upper(rng, 1)}"
            if rng.random() < 0.5
            else f"Appendix {_upper(rng, 1)}.{rng.randint(1, 12)}"
        ),
        "Appendix letter with an optional numeric subsection.",
    )

    for prefix in ("SKU", "PRD", "ITEM"):
        add(
            "sku",
            f"{prefix}_split",
            rf"^{prefix}-[A-Z]{{3}}-\d{{4}}$",
            lambda rng, p=prefix: f"{p}-{_upper(rng, 3)}-{_digits(rng, 4)}",
            "Product family letters plus a four-digit item.",
        )
    add(
        "sku",
        "compact",
        r"^[A-Z]{2}\d{3}[A-Z]\d{2}$",
        lambda rng: f"{_upper(rng, 2)}{_digits(rng, 3)}{_upper(rng, 1)}{_digits(rng, 2)}",
        "Mixed SKU with a single letter embedded.",
    )

    add(
        "phone",
        "us_parens",
        r"^\(\d{3}\) \d{3}-\d{4}$",
        lambda rng: f"({_digits(rng, 3)}) {_digits(rng, 3)}-{_digits(rng, 4)}",
        "NANP phone with parentheses.",
    )
    add(
        "phone",
        "e164",
        r"^\+\d{11,13}$",
        lambda rng: "+" + _digits(rng, rng.choice((11, 12, 13))),
        "E.164-like leading plus and 11 to 13 digits.",
    )
    add(
        "phone",
        "dotted",
        r"^\d{3}\.\d{3}\.\d{4}$",
        lambda rng: f"{_digits(rng, 3)}.{_digits(rng, 3)}.{_digits(rng, 4)}",
        "Ten-digit phone with dots.",
    )
    add(
        "postal_code",
        "us_zip",
        r"^\d{5}$",
        lambda rng: _digits(rng, 5),
        "US ZIP.",
    )
    add(
        "postal_code",
        "us_zip4",
        r"^\d{5}-\d{4}$",
        lambda rng: f"{_digits(rng, 5)}-{_digits(rng, 4)}",
        "US ZIP+4.",
    )
    add(
        "postal_code",
        "uk",
        r"^[A-Z]{1,2}\d[A-Z\d]? \d[A-Z]{2}$",
        lambda rng: f"{_upper(rng, rng.choice((1, 2)))}{_digits(rng, 1)} {_digits(rng, 1)}{_upper(rng, 2)}",
        "Simplified UK outward-inward postcode.",
    )
    add(
        "postal_code",
        "ca",
        r"^[A-Z]\d[A-Z] \d[A-Z]\d$",
        lambda rng: f"{_upper(rng, 1)}{_digits(rng, 1)}{_upper(rng, 1)} {_digits(rng, 1)}{_upper(rng, 1)}{_digits(rng, 1)}",
        "Canadian postal code.",
    )
    add(
        "postal_code",
        "de",
        r"^\d{5}$",
        lambda rng: _digits(rng, 5),
        "German five-digit postal code. Same surface as US ZIP; the domain note records the source.",
    )

    exts = ("pdf", "docx", "xlsx", "csv", "txt", "png")
    add(
        "filename",
        "simple",
        r"^[a-z0-9_]+\.(?:pdf|docx|xlsx|csv|txt|png)$",
        lambda rng, exts=exts: f"{_lower(rng, rng.randint(3, 10))}{rng.choice(('', '_final', '_v2'))}.{rng.choice(exts)}",
        "Lowercase file stem and a small extension set.",
    )
    add(
        "document_filename",
        "dated",
        r"^Invoice_20\d{2}-\d{2}-\d{2}\.pdf$",
        lambda rng: f"Invoice_{rng.randint(2020, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}.pdf",
        "Dated invoice PDF filename.",
    )
    add(
        "document_filename",
        "party_doc",
        r"^[A-Z][a-z]+_[A-Z][a-z]+_(?:Invoice|Statement|Receipt)\.pdf$",
        lambda rng: (
            f"{rng.choice(('Acme', 'North', 'Blue', 'Cedar'))}_"
            f"{rng.choice(('Supply', 'Works', 'Labs', 'Goods'))}_"
            f"{rng.choice(('Invoice', 'Statement', 'Receipt'))}.pdf"
        ),
        "Party name, party name, document type, pdf.",
    )
    add(
        "url",
        "https_path",
        r"^https://[a-z0-9.-]+\.[a-z]{2,3}/[a-z0-9/_-]+$",
        lambda rng: (
            f"https://{rng.choice(('files', 'cdn', 'docs'))}.{rng.choice(('acme', 'northwind', 'example'))}."
            f"{rng.choice(('com', 'io', 'org'))}/{rng.choice(('invoices', 'orders', 'docs'))}/"
            f"{_lower(rng, 6)}"
        ),
        "HTTPS URL with one path segment pair.",
    )
    add(
        "email",
        "simple",
        r"^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,3}$",
        lambda rng: (
            f"{_lower(rng, rng.randint(3, 8))}@{rng.choice(('acme', 'northwind', 'example'))}."
            f"{rng.choice(('com', 'org', 'io'))}"
        ),
        "Ordinary email address. Local part in this generator is lowercase letters only.",
    )
    add(
        "uuid",
        "v4_shape",
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        lambda rng: (
            f"{_hex(rng, 8)}-{_hex(rng, 4)}-4{_hex(rng, 3)}-"
            f"{rng.choice('89ab')}{_hex(rng, 3)}-{_hex(rng, 12)}"
        ),
        "UUID version-4 shape.",
    )
    add(
        "hash",
        "md5",
        r"^[0-9a-f]{32}$",
        lambda rng: _hex(rng, 32),
        "MD5 hex digest.",
    )
    add(
        "hash",
        "sha256",
        r"^[0-9a-f]{64}$",
        lambda rng: _hex(rng, 64),
        "SHA-256 hex digest.",
    )
    add(
        "transaction_id",
        "txn",
        r"^TXN-\d{4}-[A-Z0-9]{6}$",
        lambda rng: f"TXN-{rng.randint(2020, 2026)}-{_alnum_upper(rng, 6)}",
        "Transaction id with a year and a six-character body.",
    )
    add(
        "bank_reference",
        "structured",
        r"^\+\+\+\d{3}/\d{4}/\d{5}\+\+\+$",
        lambda rng: f"+++{_digits(rng, 3)}/{_digits(rng, 4)}/{_digits(rng, 5)}+++",
        "Belgian structured communication.",
    )
    add(
        "legal_numbering",
        "usc_like",
        r"^\d{1,2} U\.S\.C\. § \d{1,4}(?:\([a-z]\))?$",
        lambda rng: (
            f"{rng.randint(1, 50)} U.S.C. § {rng.randint(1, 3000)}"
            + (f"({rng.choice(LOWER)})" if rng.random() < 0.5 else "")
        ),
        "US Code section with an optional letter subsection.",
    )
    add(
        "legal_numbering",
        "docket",
        r"^No\. \d{2}-cv-\d{4,5}$",
        lambda rng: f"No. {rng.randint(10, 25)}-cv-{rng.randint(0, 99999):04d}",
        "Civil docket number. Body width is at least four digits.",
    )
    add(
        "ocr_field",
        "invoice_ocr_space",
        r"^INV-\d{5}$",
        lambda rng: f"INV-{_digits(rng, 5)}",
        "Clean target behind an OCR field. Noise belongs in the hidden negatives, not the gold language.",
    )
    add(
        "ocr_field",
        "amount_ocr",
        r"^\d{1,3}(?:,\d{3})*\.\d{2}$",
        lambda rng: f"{rng.randint(0, 20000):,}.{rng.randint(0, 99):02d}",
        "Amount line a document OCR pass would isolate.",
    )
    add(
        "tracking",
        "ups_like",
        r"^1Z[A-Z0-9]{16}$",
        lambda rng: "1Z" + _alnum_upper(rng, 16),
        "UPS-like tracking number.",
    )
    add(
        "tracking",
        "postal_20",
        r"^[A-Z]{2}\d{9}[A-Z]{2}$",
        lambda rng: f"{_upper(rng, 2)}{_digits(rng, 9)}{_upper(rng, 2)}",
        "S10 postal item identifier.",
    )
    add(
        "version",
        "semver",
        r"^v\d+\.\d+\.\d+$",
        lambda rng: f"v{rng.randint(0, 12)}.{rng.randint(0, 20)}.{rng.randint(0, 40)}",
        "vMAJOR.MINOR.PATCH with unbounded component width in the gold language.",
    )
    add(
        "doi",
        "simple",
        r"^10\.\d{4,5}/[a-z0-9._-]+$",
        lambda rng: f"10.{rng.randint(1000, 99999)}/{_lower(rng, rng.randint(4, 10))}",
        "DOI prefix 10 plus registrant and suffix.",
    )
    add(
        "serial",
        "vin_like",
        r"^[A-HJ-NPR-Z0-9]{17}$",
        lambda rng: "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ0123456789") for _ in range(17)),
        "17-character VIN alphabet, excluding I, O, and Q.",
    )
    add(
        "log_line",
        "level_ts",
        r"^(?:INFO|WARN|ERROR) \d{2}:\d{2}:\d{2} [a-z0-9_]+$",
        lambda rng: (
            f"{rng.choice(('INFO', 'WARN', 'ERROR'))} "
            f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d} "
            f"{rng.choice(('boot', 'auth', 'db_connect', 'fetch_page'))}"
        ),
        "Log prefix with a level, a time, and a snake-case event.",
    )
    add(
        "form_field",
        "yes_no",
        r"^(?:YES|NO)$",
        lambda rng: rng.choice(("YES", "NO")),
        "Closed checkbox vocabulary.",
    )
    add(
        "form_field",
        "ssn_masked",
        r"^XXX-XX-\d{4}$",
        lambda rng: f"XXX-XX-{_digits(rng, 4)}",
        "Masked SSN showing only the last four digits.",
    )
    add(
        "spreadsheet_column",
        "iso_currency_code",
        r"^(?:USD|EUR|GBP|JPY|CAD|AUD)$",
        lambda rng: rng.choice(("USD", "EUR", "GBP", "JPY", "CAD", "AUD")),
        "ISO currency code column. The gold language is the observed closed set.",
    )
    add(
        "spreadsheet_column",
        "quarter",
        r"^20\d{2}Q[1-4]$",
        lambda rng: f"{rng.randint(2018, 2026)}Q{rng.randint(1, 4)}",
        "Fiscal quarter label.",
    )

    # More structural variety so the catalog is not only prefix-digit clones.
    for sep in ("-", "/", "."):
        add(
            "invoice_number",
            f"num_sep_{sep}",
            rf"^INV{re_sep(sep)}2026{re_sep(sep)}\d{{5}}$",
            lambda rng, sep=sep: f"INV{sep}2026{sep}{_digits(rng, 5)}",
            f"Invoice with separator {sep!r} and a fixed year.",
        )
    add(
        "customer_id",
        "optional_region",
        r"^CUS-\d{6}(?:-[A-Z]{2})?$",
        lambda rng: f"CUS-{_digits(rng, 6)}" + (f"-{_upper(rng, 2)}" if rng.random() < 0.5 else ""),
        "Customer id with an optional region suffix. Both forms appear in the generator.",
    )
    add(
        "order_id",
        "optional_prefix",
        r"^(?:WEB-|POS-)?\d{8}$",
        lambda rng: (rng.choice(("WEB-", "POS-", "")) + _digits(rng, 8)),
        "Channel prefix is optional.",
    )
    add(
        "measurement",
        "unit_alt",
        r"^\d{1,3}(?:\.\d)? (?:kg|lb|g|oz)$",
        lambda rng: f"{rng.randint(1, 40)}.{rng.randint(0, 9)} {rng.choice(('kg', 'lb', 'g', 'oz'))}",
        "Quantity plus a small unit alternation.",
    )
    add(
        "filename",
        "copy_suffix",
        r"^[a-z0-9]+(?: \(\d+\))?\.pdf$",
        lambda rng: _lower(rng, 6) + (f" ({rng.randint(1, 4)})" if rng.random() < 0.5 else "") + ".pdf",
        "PDF filename with an optional ' (n)' copy marker.",
    )
    add(
        "email",
        "plus_tag",
        r"^[a-z]+(?:\+[a-z0-9]+)?@[a-z]+\.[a-z]{2,3}$",
        lambda rng: (
            _lower(rng, 5)
            + (f"+{_lower(rng, 3)}" if rng.random() < 0.5 else "")
            + f"@{rng.choice(('acme', 'example'))}.{rng.choice(('com', 'org'))}"
        ),
        "Email with an optional plus tag.",
    )
    add(
        "url",
        "query",
        r"^https://[a-z]+\.[a-z]{2,3}/[a-z]+\?id=\d{4,6}$",
        lambda rng: (
            f"https://{rng.choice(('shop', 'api'))}.{rng.choice(('com', 'io'))}/"
            f"{rng.choice(('orders', 'items'))}?id={_digits(rng, rng.choice((4, 5, 6)))}"
        ),
        "URL with a numeric id query.",
    )
    add(
        "page_reference",
        "folio",
        r"^fol\. \d{1,4}[rv]$",
        lambda rng: f"fol. {rng.randint(1, 400)}{rng.choice(('r', 'v'))}",
        "Manuscript folio with recto or verso.",
    )
    add(
        "legal_numbering",
        "article",
        r"^Art\. \d{1,3}(?: bis)?$",
        lambda rng: f"Art. {rng.randint(1, 120)}" + (" bis" if rng.random() < 0.3 else ""),
        "Article number with an optional 'bis'.",
    )
    add(
        "bank_reference",
        "ach",
        r"^[0-9]{9} [0-9]{6,17}$",
        lambda rng: f"{_digits(rng, 9)} {_digits(rng, rng.randint(6, 17))}",
        "Routing number plus a variable-width account.",
    )
    add(
        "sku",
        "color_size",
        r"^[A-Z]{3}-\d{3}-(?:XS|S|M|L|XL)-\d{2}$",
        lambda rng: f"{_upper(rng, 3)}-{_digits(rng, 3)}-{rng.choice(('XS', 'S', 'M', 'L', 'XL'))}-{_digits(rng, 2)}",
        "Style, color code, size alternation, lot.",
    )
    add(
        "time",
        "duration",
        r"^\d{1,2}h \d{1,2}m$",
        lambda rng: f"{rng.randint(0, 12)}h {rng.randint(0, 59)}m",
        "Duration written as hours and minutes.",
    )
    add(
        "currency",
        "negative_paren",
        r"^\(\$\d{1,3}(?:,\d{3})*\.\d{2}\)$",
        lambda rng: f"(${rng.randint(1, 4000):,}.{rng.randint(0, 99):02d})",
        "Accounting negative amount in parentheses.",
    )
    add(
        "hash",
        "sha1",
        r"^[0-9a-f]{40}$",
        lambda rng: _hex(rng, 40),
        "SHA-1 hex digest.",
    )
    add(
        "transaction_id",
        "uuid_prefixed",
        r"^pay_[0-9a-z]{14}$",
        lambda rng: "pay_" + "".join(rng.choice("0123456789abcdefghijklmnopqrstuvwxyz") for _ in range(14)),
        "Processor-style charge id.",
    )
    add(
        "document_filename",
        "scan",
        r"^scan_\d{8}_\d{6}\.pdf$",
        lambda rng: (
            f"scan_{rng.randint(2020, 2026)}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}_"
            f"{rng.randint(0, 23):02d}{rng.randint(0, 59):02d}{rng.randint(0, 59):02d}.pdf"
        ),
        "Scanner timestamp filename.",
    )
    add(
        "section_id",
        "iso_clause",
        r"^§\s?\d{1,2}(?:\.\d{1,2}){1,3}$",
        lambda rng: "§"
        + (" " if rng.random() < 0.5 else "")
        + ".".join(str(rng.randint(1, 12)) for _ in range(rng.randint(2, 4))),
        "Section sign with two to four numeric levels and an optional space.",
    )
    add(
        "log_line",
        "request_id",
        r"^req_[0-9a-f]{8}$",
        lambda rng: "req_" + _hex(rng, 8),
        "Request id in a log column.",
    )
    add(
        "form_field",
        "date_blank",
        r"^(?:N/A|\d{4}-\d{2}-\d{2})$",
        lambda rng: (
            "N/A"
            if rng.random() < 0.25
            else f"{rng.randint(1990, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        ),
        "A date column that also contains the sentinel N/A.",
    )
    add(
        "postal_code",
        "nl",
        r"^\d{4} [A-Z]{2}$",
        lambda rng: f"{_digits(rng, 4)} {_upper(rng, 2)}",
        "Dutch postal code.",
    )
    add(
        "phone",
        "intl_spaced",
        r"^\+\d{1,3} \d{2,4} \d{3,4} \d{3,4}$",
        lambda rng: (
            f"+{rng.choice(('1', '44', '49', '33'))} {_digits(rng, rng.choice((2, 3)))} "
            f"{_digits(rng, 3)} {_digits(rng, rng.choice((3, 4)))}"
        ),
        "International phone with spaces. Widths follow the generator, not every possible spacing.",
    )
    add(
        "invoice_number",
        "leading_zeros",
        r"^INV-\d{6}$",
        lambda rng: f"INV-{rng.randint(0, 999999):06d}",
        "Six digits including leading zeros. Shorter unpadded numbers are negatives.",
    )
    add(
        "customer_id",
        "legacy_new",
        r"^(?:C\d{5}|CUST-[A-Z]{3}\d{3})$",
        lambda rng: (
            f"C{_digits(rng, 5)}" if rng.random() < 0.5 else f"CUST-{_upper(rng, 3)}{_digits(rng, 3)}"
        ),
        "Two co-existing customer id formats in one column.",
    )
    add(
        "measurement",
        "feet_inches",
        r"^\d{1,2}' \d{1,2}\"$",
        lambda rng: f"{rng.randint(0, 12)}' {rng.randint(0, 11)}\"",
        "Feet and inches.",
    )
    add(
        "currency",
        "code_after",
        r"^\d+\.\d{2} (?:USD|EUR|GBP)$",
        lambda rng: f"{rng.randint(0, 5000)}.{rng.randint(0, 99):02d} {rng.choice(('USD', 'EUR', 'GBP'))}",
        "Unpadded amount with a trailing currency code.",
    )
    add(
        "tracking",
        "fedex_like",
        r"^\d{12}$|^\d{15}$",
        lambda rng: _digits(rng, rng.choice((12, 15))),
        "Two fixed widths used by common parcel tracking numbers.",
    )
    add(
        "serial",
        "mac",
        r"^(?:[0-9A-F]{2}:){5}[0-9A-F]{2}$",
        lambda rng: ":".join(_hex(rng, 2).upper() for _ in range(6)),
        "MAC address.",
    )
    add(
        "serial",
        "ipv4",
        r"^(?:25[0-5]|2[0-4]\d|1?\d{1,2})(?:\.(?:25[0-5]|2[0-4]\d|1?\d{1,2})){3}$",
        lambda rng: ".".join(str(rng.randint(0, 255)) for _ in range(4)),
        "IPv4 address.",
    )
    add(
        "spreadsheet_column",
        "bool",
        r"^(?:true|false)$",
        lambda rng: rng.choice(("true", "false")),
        "Boolean column in lowercase.",
    )
    add(
        "ocr_field",
        "date_kept_clean",
        r"^\d{2}/\d{2}/\d{4}$",
        lambda rng: f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.randint(2018, 2026)}",
        "Date field whose gold language does not include OCR substitutions.",
    )
    add(
        "document_filename",
        "matter",
        r"^\d{4}-[A-Z]{2,4}-\d{3}_[A-Za-z0-9]+\.docx$",
        lambda rng: (
            f"{rng.randint(2016, 2026)}-{_upper(rng, rng.choice((2, 3, 4)))}-{_digits(rng, 3)}_"
            f"{rng.choice(('Draft', 'Final', 'Executed'))}{rng.randint(1, 9)}.docx"
        ),
        "Matter-number document filename.",
    )
    add(
        "section_id",
        "schedule",
        r"^Schedule [A-Z]-\d{1,2}$",
        lambda rng: f"Schedule {_upper(rng, 1)}-{rng.randint(1, 12)}",
        "Contract schedule identifier.",
    )
    add(
        "bank_reference",
        "swift",
        r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?$",
        lambda rng: _upper(rng, 6) + _alnum_upper(rng, 2) + (_alnum_upper(rng, 3) if rng.random() < 0.5 else ""),
        "SWIFT/BIC, 8 or 11 characters.",
    )
    add(
        "log_line",
        "http_status",
        r"^HTTP/1\.[01] (?:200|201|204|301|302|400|401|403|404|500) \d+ms$",
        lambda rng: (
            f"HTTP/1.{rng.choice(('0', '1'))} "
            f"{rng.choice(('200', '201', '204', '301', '302', '400', '401', '403', '404', '500'))} "
            f"{rng.randint(1, 4000)}ms"
        ),
        "Access-log status fragment.",
    )
    return families


def re_sep(sep: str) -> str:
    return re_escape_in_class(sep)


def re_escape_in_class(sep: str) -> str:
    return {"-": r"\-", "/": "/", ".": r"\."}[sep]


def _hex(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def _alnum_upper(rng: random.Random, n: int) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(n))


def _extra_families(families: list) -> None:
    """Harder document columns: classes, optionals, alternations, unicode."""
    _family(
        families,
        "hash",
        "hex_lower",
        r"^[0-9a-f]{16}$",
        lambda rng: _hex(rng, 16),
        "Short hex token. The class is hexadecimal, not the full alphanumeric alphabet.",
    )
    _family(
        families,
        "scientific_id",
        "pmid",
        r"^PMID: \d{7,8}$",
        lambda rng: f"PMID: {_digits(rng, rng.choice((7, 8)))}",
        "PubMed identifier with a variable digit width.",
    )
    _family(
        families,
        "scientific_id",
        "accession",
        r"^[A-Z]{1,2}\d{5,6}(?:\.\d+)?$",
        lambda rng: _upper(rng, rng.choice((1, 2)))
        + _digits(rng, rng.choice((5, 6)))
        + (f".{rng.randint(1, 9)}" if rng.random() < 0.5 else ""),
        "GenBank-like accession with an optional version.",
    )
    _family(
        families,
        "shipping_code",
        "sscc",
        r"^00\d{18}$",
        lambda rng: "00" + _digits(rng, 18),
        "SSCC-18 with a fixed application identifier.",
    )
    _family(
        families,
        "legal_numbering",
        "celex",
        r"^3\d{4}[A-Z]\d{4}$",
        lambda rng: f"3{rng.randint(1990, 2024)}{_upper(rng, 1)}{_digits(rng, 4)}",
        "CELEX-like legal identifier.",
    )
    _family(
        families,
        "currency",
        "unicode_yen",
        r"^¥\d{1,3}(?:,\d{3})*$",
        lambda rng: "¥" + f"{rng.randint(0, 2000000):,}",
        "Yen amounts. No decimal part.",
    )
    _family(
        families,
        "filename",
        "unicode_stem",
        r"^[\w\-]+_副本\.docx$",
        lambda rng: f"{_lower(rng, rng.randint(3, 8))}_副本.docx",
        "A copied document filename that keeps a CJK suffix.",
    )
    _family(
        families,
        "email",
        "subdomain",
        r"^[a-z]+(?:\.[a-z]+)?@[a-z]+(?:\.[a-z]+){1,2}$",
        lambda rng: (
            _lower(rng, 4)
            + (f".{_lower(rng, 3)}" if rng.random() < 0.5 else "")
            + "@"
            + rng.choice(("mail", "dept"))
            + "."
            + rng.choice(("acme", "example"))
            + "."
            + rng.choice(("com", "edu"))
        ),
        "Email that sometimes has a dotted local part and always a subdomain.",
    )
    for code in ("USD", "EUR", "GBP", "CHF"):
        _family(
            families,
            "currency",
            f"iso_{code}_space",
            rf"^{code} \d+\.\d{{2}}$",
            lambda rng, code=code: f"{code} {rng.randint(0, 5000)}.{rng.randint(0, 99):02d}",
            f"Unpadded {code} amount. Width of the whole-number part is open.",
        )


def generate_bench_c(seed: int = 2026, replicates: int = 5) -> list[Task]:
    rng = random.Random(seed)
    families = _build_families()
    _extra_families(families)
    tasks: list[Task] = []
    index = 1
    for domain, name, regex, sample, notes, schema in families:
        for replicate in range(replicates):
            tasks.append(
                build_task(
                    task_id=f"C-{index:04d}-{domain}-{name}-r{replicate}",
                    benchmark="C",
                    domain=domain,
                    variant=name,
                    gold_regex=regex,
                    sample=sample,
                    rng=rng,
                    latent_schema=schema,
                    n_visible=6,
                    n_pos=24,
                    n_neg=24,
                    split="test",
                    notes=notes,
                    meta={
                        "source": "curated-document-catalog",
                        "holdout": True,
                        "replicate": replicate,
                    },
                )
            )
            index += 1
    # Prefix-digit invoices are easy for a column heuristic. Cap them so the
    # headline score is not an average over near-duplicate document numbers.
    capped: list[Task] = []
    seen: dict[str, int] = {}
    for task in tasks:
        limit = 24 if task.domain == "invoice_number" else 30
        seen[task.domain] = seen.get(task.domain, 0) + 1
        if seen[task.domain] <= limit:
            capped.append(task)
    return capped
