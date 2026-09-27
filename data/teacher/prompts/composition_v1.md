Return JSON only. No prose.

You design string languages for a tiny model that already understands single operators (a constant prefix, a repeated numeric group, an optional mark, a bare alternation) and fails when those operators are attached in new ways.

Emit 8 schemas in this DSL. Each schema is a JSON object the student's compiler will check. Do not emit strings as examples. The compiler instantiates symbols itself.

Node types:
- {"type":"seq","children":[...]} with 2 to 4 children
- {"type":"class","class":"digit|upper|lower","min":1,"max":4}
- {"type":"literal_slot","role":"name","class":"upper|lower|digit|punct","min":1,"max":4}
  punct width is always 1. A literal_slot is constant inside one task and changes across tasks.
- {"type":"repeat","min":1,"max":3,"child":{...}}
  max must be 3 or less. Do not nest a repeat inside a repeat.
- {"type":"optional","child":{...}}
- {"type":"alt","children":[{...},{...}]}

A class draws fresh characters every string. A literal_slot does not.

Do not emit these held-out shapes. The compiler rejects them:
- a constant literal prefix followed by a numeric repeat
- a numeric repeat followed by an optional tail
- a constant literal prefix followed by an alternation
- a token, an optional middle, and a token, with no other operator

Prefer attachments that are easy to confuse, and include a confusable schema that differs by one attachment.

Response shape:
{"schemas":[{"id":"short_name","schema":{...},"confusable":[{...}]}]}
