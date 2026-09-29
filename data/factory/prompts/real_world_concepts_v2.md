Return one JSON object and no prose.

You are a domain expert for {domain}. Set every concept's "domain" field to exactly "{domain}". Invent 8 fields this domain prints on documents or in systems. Prefer obscure conventions beside the famous ones. The 8 programs must not share one shape.

For each field, describe the meaning first, then positives a clerk might copy, near misses a tired clerk might produce, and a program. Our compiler drops the concept if a positive fails or a near miss matches.

Use a spread of structure across the 8 fields: a literal separator, a numeric window wider than the observed numbers, a frozen lexicon where one truly fits, an optional mark that appears in at least one positive, a repeated section, a character set that is not the full alphabet, and a width that the samples do not exhaust.

DSL:
- {"op":"seq","children":[...]} with 2 to 6 children
- {"op":"cls","class":"digit|upper|lower","lo":1,"hi":4} and hi may be 128 only for an open width. No other hi above 32.
- {"op":"set","spec":"A-Z","lo":1,"hi":4}
- {"op":"int","min":0,"max":9999,"lo":4,"hi":4} where min and max are the numeric window and lo and hi are the digit width. hi at most 8. Fixed width keeps leading zeros. Variable width rejects them.
- {"op":"lex","name":"iso4217"} and the only names are iso4217, month_abbr, month_abbr_upper, month_abbr_lower, month_full, month_full_upper, month_full_lower, http_1_0, http_iana, file_extension, mass_unit, apparel_size
- {"op":"lit","text":"-"} length 1 to 8, and the text must occur inside a positive
- {"op":"opt","child":{...}}
- {"op":"alt","children":[{...},{...}]} exactly two
- {"op":"repeat","lo":1,"hi":3,"child":{...}} hi at most 8, not nested

Positives: 3 or 4 strings, each length 1 to 32, all distinct. Near misses: 3 strings the program must refuse. Neighbors: 2 programs that differ by one decision. Where the positives fit a neighbor as well as the gold, put those strings in ambiguous_supports.

Do not emit a regex.

Response shape:
{"concepts":[{"domain":"{domain}","concept":"...","description":"...","semantic_type":"...","positives":["..."],"near_misses":["..."],"ambiguous_supports":["..."],"ambiguity_types":["..."],"program":{},"neighbors":[{},{}]}]}
