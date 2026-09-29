Return one JSON object and no prose.

For the domain {domain}, invent 6 gold string languages and, for each, a family of neighbors that differ by exactly one decision. The neighbors are the training contrast. A neighbor that accepts the same strings as the gold is useless.

Decisions worth contrasting:
- alphabet A-Z versus A-F versus alphanumeric
- fixed width versus one wider bound versus an open width
- observed integer hull versus a conventional wider range versus any fixed-width number
- a frozen lexicon versus the character class that covers it versus the observed members only, when those members can be written as an alt of two literals
- optional mark present versus absent
- repeat count tight versus one step looser

Use this DSL only:
seq, cls, set, int, lex (frozen names only: iso4217, month_abbr, month_abbr_upper, month_abbr_lower, month_full, month_full_upper, month_full_lower, http_1_0, http_iana, file_extension, mass_unit, apparel_size), lit, opt, alt of two children, repeat with hi at most 8.

Positives must match the gold. Near misses must not. Put 3 or 4 neighbors on each concept.

Response shape:
{"concepts":[{"domain":"...","concept":"...","description":"...","semantic_type":"...","positives":["..."],"near_misses":["..."],"ambiguity_types":["..."],"program":{...},"neighbors":[{...},{...},{...}]}]}
