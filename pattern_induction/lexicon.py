"""Frozen closed sets a few examples can complete but cannot invent.

A lexicon is a published vocabulary: currency codes, month names, status
codes, units, sizes, filename suffixes. The diagnostic benchmark is not a
source. A proposal fires only when every observed token already sits in one
of these sets and the set still has members the examples did not show.
The node stays one AST node. The alternation is compiled at the end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Lexicon:
    name: str
    members: frozenset[str]


def _lex(name: str, words: str) -> Lexicon:
    return Lexicon(name, frozenset(words.split()))


# ISO 4217 alphabetic codes. A superset of any short sample, not a short sample itself.
_ISO_4217 = """
AED AFN ALL AMD ANG AOA ARS AUD AWG AZN BAM BBD BDT BGN BHD BIF BMD BND BOB
BRL BSD BTN BWP BYN BZD CAD CDF CHF CLP CNY COP CRC CUP CVE CZK DJF DKK DOP
DZD EGP ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD GNF GTQ GYD HKD HNL HTG HUF
IDR ILS INR IQD IRR ISK JMD JOD JPY KES KGS KHR KMF KPW KRW KWD KYD KZT LAK
LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP MRU MUR MVR MWK MXN MYR MZN
NAD NGN NIO NOK NPR NZD OMR PAB PEN PGK PHP PKR PLN PYG QAR RON RSD RUB RWF
SAR SBD SCR SDG SEK SGD SHP SLE SOS SRD STN SVC SYP SZL THB TJS TMT TND TOP
TRY TTD TWD TZS UAH UGX USD UYU UZS VES VND VUV WST XAF XCD XOF XPF YER ZAR
ZMW ZWL
"""

# Gregorian month names, the three usual casings. Each casing is its own set.
_MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec"
_MONTHS_FULL = "January February March April May June July August September October November December"

# RFC 1945 HTTP/1.0 status codes, and the wider IANA registry. Both are standards.
# A sample that sits in both proposes both; the hidden set decides which survives.
_HTTP_10 = "200 201 202 204 301 302 304 400 401 403 404 500 501 502 503"
_HTTP_IANA = """
100 101 200 201 202 203 204 205 206 300 301 302 303 304 305 307 308
400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417
418 421 422 425 426 428 429 431 451 500 501 502 503 504 505 511
"""

# Popular filename suffixes from public media types. Not a document-type list.
_EXTENSIONS = """
pdf doc docx odt rtf txt csv tsv xls xlsx ods ppt pptx odp
png jpg jpeg gif svg webp bmp tif tiff zip gz tar json xml html htm
md yaml yml log sql wav mp3 mp4 avi mov
"""

# Mass units a quantity field often takes. Letter sizes are the retail scale.
_MASS = "mg g kg oz lb"
_SIZES = "XXS XS S M L XL XXL XXXL"

LEXICONS: tuple[Lexicon, ...] = (
    _lex("iso4217", _ISO_4217),
    _lex("month_abbr", _MONTHS),
    _lex("month_abbr_upper", _MONTHS.upper()),
    _lex("month_abbr_lower", _MONTHS.lower()),
    _lex("month_full", _MONTHS_FULL),
    _lex("month_full_upper", _MONTHS_FULL.upper()),
    _lex("month_full_lower", _MONTHS_FULL.lower()),
    _lex("http_1_0", _HTTP_10),
    _lex("http_iana", _HTTP_IANA),
    _lex("file_extension", _EXTENSIONS),
    _lex("mass_unit", _MASS),
    _lex("apparel_size", _SIZES),
)

_BY_NAME = {item.name: item for item in LEXICONS}
_MEMBERS: dict[str, tuple[str, ...]] = {
    item.name: tuple(sorted(item.members, key=lambda word: (-len(word), word))) for item in LEXICONS
}


def lexicon_members(name: str) -> tuple[str, ...]:
    return _MEMBERS.get(name, ())


def lexicon_regex(name: str) -> str:
    members = lexicon_members(name)
    if not members:
        return ""
    return "(?:" + "|".join(re.escape(member) for member in members) + ")"


def matching_lexicons(observed: set[str]) -> list[str]:
    """Closed sets that contain every observed token and at least one more.

    Two observed members are the minimum evidence. A single token is a
    constant, not a sample of a vocabulary. Large code lists ignore a sample
    whose tokens are shorter than the codes themselves.
    """
    if len(observed) < 2 or any(not token or len(token) > 32 for token in observed):
        return []
    found: list[str] = []
    for item in LEXICONS:
        if not observed <= item.members or len(observed) >= len(item.members):
            continue
        if len(item.members) >= 80 and min(len(token) for token in observed) < 3:
            continue
        found.append(item.name)
    return found


def closes_alternation(literals: set[str]) -> bool:
    """True when a gold alternation is a sample of a frozen lexicon.

    Used to score closed-set recall. The proposer does not call this.
    """
    if len(literals) < 3:
        return False
    return any(literals <= item.members for item in LEXICONS)
