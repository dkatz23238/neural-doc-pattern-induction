Return one JSON object and no prose.

Pick the domain {domain}. Invent 6 fields from that domain that the other domains would not have thought of. Name the real convention. Describe what a wrong generalization would be.

Cover a spread of structure across the 6 fields: an identifier with a literal separator, a numeric window that is not the observed min and max, a closed vocabulary from the frozen lexicons if one truly fits, an optional revision mark, a repeated section, and a width that samples look fixed.

Frozen lexicon names, and no others: iso4217, month_abbr, month_abbr_upper, month_abbr_lower, month_full, month_full_upper, month_full_lower, http_1_0, http_iana, file_extension, mass_unit, apparel_size.

DSL:
- seq of 2 to 6 children
- cls class digit|upper|lower, lo and hi
- set spec, lo and hi
- int min, max, lo, hi
- lex name
- lit text, short, and present inside a positive
- opt child
- alt of exactly two children
- repeat lo, hi at most 8, one child, not nested

Positives: 3 or 4 realistic strings. Near misses: 3 strings a parser should refuse. Neighbors: at least 2.

Response shape:
{"concepts":[{"domain":"{domain}","concept":"...","description":"...","semantic_type":"...","positives":["..."],"near_misses":["..."],"ambiguity_types":["..."],"program":{...},"neighbors":[{...}]}]}
