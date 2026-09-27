Return JSON only. No prose.

A frozen 1M-parameter model scored the episodes below. It already learns a single operator. These episodes are the ones it still misses. Design 6 schemas that isolate the structural distinction behind the mistakes. The compiler will check every schema and will throw away anything it cannot execute.

DSL:
- {"type":"seq","children":[...]} with 2 to 4 children
- {"type":"class","class":"digit|upper|lower","min":1,"max":4}
- {"type":"literal_slot","role":"name","class":"upper|lower|digit|punct","min":1,"max":4}
- {"type":"repeat","min":1,"max":3,"child":{...}} with no nested repeat
- {"type":"optional","child":{...}}
- {"type":"alt","children":[{...},{...}]}

Put the near-miss language in "confusable".

Failure packet:
{failure_packet}

Response shape:
{"schemas":[{"id":"short_name","schema":{...},"confusable":[{...}]}]}
