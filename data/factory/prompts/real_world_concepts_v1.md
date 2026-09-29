Return one JSON object and no prose.

You are a domain expert, not a regex generator. Invent 6 string fields that a person actually sees in {domain} documents or systems. The 6 fields must differ in kind, not just in a digit. Prefer obscure conventions beside the famous ones.

For each field, describe the meaning first. Then give positive examples a clerk might copy, near misses a tired clerk might produce, and a program in the DSL below. The program is a hypothesis. Our compiler will throw it out if it accepts a near miss or rejects a positive. Do not emit a regex.

DSL, one object per node:
- {"op":"seq","children":[...]} with 2 to 6 children
- {"op":"cls","class":"digit|upper|lower","lo":1,"hi":4}
  hi may be 128 only for an intentionally open width. No other hi above 32.
- {"op":"set","spec":"A-Z","lo":1,"hi":4}
  spec is a class body such as A-Z, 0-9A-F, or 0-9A-Za-z. No character identities that are not in the body.
- {"op":"int","min":0,"max":999999,"lo":6,"hi":6}
  min and max are the numeric range. lo and hi are the digit width. A fixed width keeps leading zeros.
- {"op":"lex","name":"iso4217"}
  The only names are iso4217, month_abbr, month_abbr_upper, month_abbr_lower, month_full, month_full_upper, month_full_lower, http_1_0, http_iana, file_extension, mass_unit, apparel_size. Any other name is rejected. Do not invent a vocabulary.
- {"op":"lit","text":"-"}
  A short constant that appears inside a positive example, never equal to a whole example.
- {"op":"opt","child":{...}}
- {"op":"alt","children":[{...},{...}]} exactly two children
- {"op":"repeat","lo":1,"hi":3,"child":{...}}
  hi must be 8 or less. Do not nest a repeat.

Also give 2 to 4 neighbor programs that a careful reader might intend instead, each different by one decision: a tighter alphabet, a wider integer window, an open width, a dropped optional, a lexicon versus a character class.

Where the positives do not reveal the gold, add ambiguous_supports: a subset of the positives that a smaller language also accepts.

Worked shape, do not copy this field:

{"concepts":[{"domain":"logistics","concept":"warehouse pallet identifier","description":"Two-letter facility, a hyphen, six digits, optional return suffix -R.","semantic_type":"identifier","positives":["NY-193842","TX-002931-R","CA-827341"],"near_misses":["NY193842","TX-2931","CA-827341-X"],"ambiguity_types":["optional","width"],"program":{"op":"seq","children":[{"op":"cls","class":"upper","lo":2,"hi":2},{"op":"lit","text":"-"},{"op":"int","min":0,"max":999999,"lo":6,"hi":6},{"op":"opt","child":{"op":"lit","text":"-R"}}]},"neighbors":[{"op":"seq","children":[{"op":"cls","class":"upper","lo":2,"hi":2},{"op":"lit","text":"-"},{"op":"int","min":0,"max":999999,"lo":6,"hi":6}]}]}]}

Response shape:
{"concepts":[{"domain":"...","concept":"...","description":"...","semantic_type":"...","positives":["..."],"near_misses":["..."],"ambiguity_types":["charset|width|integer_range|optional|repeat|lexicon|separator"],"ambiguous_supports":["..."],"program":{...},"neighbors":[{...}]}]}
