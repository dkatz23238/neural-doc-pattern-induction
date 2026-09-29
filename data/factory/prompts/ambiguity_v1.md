Return one JSON object and no prose.

Take latent string languages a person would mean, and show only the positive examples that leave the language underdetermined. The domain is {domain}.

Emit 6 concepts. Each concept has a gold program and at least two neighbors. The ambiguous_supports must all match the gold and must also match at least one neighbor. A near miss must match neither the gold nor that neighbor when the difference is real; if the neighbor is a wider language, a near miss of the gold may be a positive of the neighbor, and you should say so by putting that string only in near_misses.

Use the same DSL as the concept bank:
seq, cls (digit|upper|lower), set, int (min, max, lo, hi), lex (only the frozen lexicon names), lit, opt, alt of exactly two children, repeat with hi at most 8 and no nesting.

Situations to cover if the domain allows them:
- letters that happen to lie in A-F, while the intended class is A-Z
- years inside 2024-2031 whose intended window is wider
- three currency codes whose intended set is the ISO list, not any three letters
- a width that is constant in the sample and open in the language
- an optional suffix that this sample happens to include every time, or never

Do not emit a regex. Do not repeat a concept you have already used as an example.

Response shape:
{"concepts":[{"domain":"...","concept":"...","description":"...","semantic_type":"...","positives":["..."],"near_misses":["..."],"ambiguous_supports":["..."],"ambiguity_types":["..."],"program":{...},"neighbors":[{...}]}]}
