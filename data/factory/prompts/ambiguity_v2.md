Return one JSON object and no prose.

Domain: {domain}. Set every concept's "domain" field to exactly "{domain}". Emit 8 concepts. Each one is a real string field whose positives also fit a smaller language. The gold program is the language a careful reader would intend. At least one neighbor is that smaller language, and it must accept every ambiguous_supports string. The gold must accept them too. A near miss must fail the gold.

Situations, phrased as fields in this domain rather than as operator names:
- letters that happen to lie in A-F while the field is A-Z
- a width that is constant in the sample and open (hi 128) in the field
- years or other numbers inside a narrow band whose intended window is a conventional wider range
- a repeated section whose sample never reaches the allowed maximum
- three codes drawn from a frozen lexicon, where three uppercase letters would also fit
- an optional suffix that this sample happens to include, and the suffix text appears in a positive

DSL, same limits on every node:
seq of 2 to 6; cls digit|upper|lower with hi at most 32 or exactly 128; set spec; int with min/max as the numeric window and lo/hi as the digit width, hi at most 8; lex only iso4217, month_abbr, month_abbr_upper, month_abbr_lower, month_full, month_full_upper, month_full_lower, http_1_0, http_iana, file_extension, mass_unit, apparel_size; lit of 1 to 8 characters that occurs inside a positive; opt; alt of exactly two children; repeat with lo at least 1 and hi at most 8, not nested.

Strings are 1 to 32 characters. Positives are unique and disjoint from near misses. A variable-width integer rejects leading zeros, so keep zeros only with lo equal to hi. Do not emit a regex.

Response shape:
{"concepts":[{"domain":"{domain}","concept":"...","description":"...","semantic_type":"...","positives":["..."],"near_misses":["..."],"ambiguous_supports":["..."],"ambiguity_types":["..."],"program":{},"neighbors":[{},{}]}]}
