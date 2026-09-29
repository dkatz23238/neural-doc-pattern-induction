Return one JSON object and no prose.

Domain: {domain}. Set every concept's "domain" field to exactly "{domain}". Invent 8 fields a person in this domain actually copies. The 8 fields must differ in structure.

Each field should leave the intended language underdetermined from the positives alone. Cover these decisions when the domain has them, one field each, and use the remaining fields for further realistic cases:

- the sample letters sit inside a smaller alphabet than the field (A-F observed, A-Z intended)
- every sample has the same width, and the field is open (hi 128) or strictly wider
- the numbers shown are inside a conventional window that is wider than their min and max
- the repeat counts shown are tight, and the field allows a looser bound
- the tokens shown are a subset of one frozen lexicon, not the whole character class

The gold program is the wider intended language. One neighbor is the narrower reading of the same positives. ambiguous_supports is a subset of the positives that both the gold and that neighbor accept. A near miss must be rejected by the gold.

DSL limits the executor enforces:
- seq of 2 to 6 children
- cls class digit|upper|lower, lo and hi from 1 to 32, or hi exactly 128 for an open width
- set spec such as A-Z or 0-9A-F, lo and hi in that same range
- int min and max are the numeric window, lo and hi are the digit width, hi at most 8. A fixed width (lo equal to hi) keeps leading zeros. A variable width rejects them, so an identifier that keeps zeros must use a fixed width.
- lex name, only: iso4217, month_abbr, month_abbr_upper, month_abbr_lower, month_full, month_full_upper, month_full_lower, http_1_0, http_iana, file_extension, mass_unit, apparel_size
- lit text of 1 to 8 characters that occurs inside a positive, never equal to a whole positive
- opt child
- alt of exactly two children
- repeat lo at least 1, hi at most 8, one child, not nested
- every string length 1 to 32, positives unique, no overlap with near misses

Do not emit a regex. Do not copy a worked example from another prompt.

Response shape:
{"concepts":[{"domain":"{domain}","concept":"...","description":"...","semantic_type":"...","positives":["..."],"near_misses":["..."],"ambiguous_supports":["..."],"ambiguity_types":["charset|width|integer_range|optional|repeat|lexicon"],"program":{"op":"seq","children":[]},"neighbors":[{"op":"..."}]}]}
