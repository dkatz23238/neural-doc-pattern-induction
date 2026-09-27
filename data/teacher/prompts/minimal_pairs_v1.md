Return JSON only. No prose.

Design 4 pairs of string languages. The two languages in a pair must differ by one structural attachment, so they can emit similar strings for different reasons.

Use this DSL. The compiler instantiates symbols and decides membership. Do not include example strings.

Node types:
- {"type":"seq","children":[...]} with 2 to 4 children
- {"type":"class","class":"digit|upper|lower","min":1,"max":4}
- {"type":"literal_slot","role":"name","class":"upper|lower|digit|punct","min":1,"max":4}
  punct width is 1
- {"type":"repeat","min":1,"max":3,"child":{...}} and do not nest repeats
- {"type":"optional","child":{...}}
- {"type":"alt","children":[{...},{...}]}

Cover these distinctions across the 6 pairs, one pair each when you can:
- PREFIX + REPEAT(group) versus REPEAT(PREFIX + group)
- REPEAT(group) + OPTIONAL(tail) versus REPEAT(group + OPTIONAL(tail))
- PREFIX + ALT(X, Y) versus ALT(PREFIX+X, PREFIX+Y)
- a required suffix versus an optional suffix
- alternation of two structures versus a sequence that merely looks like both
- a repeated separator versus a separator that occurs once

Response shape:
{"pairs":[{"id":"short_name","distinction":"one sentence","schema_a":{...},"schema_b":{...}}]}
