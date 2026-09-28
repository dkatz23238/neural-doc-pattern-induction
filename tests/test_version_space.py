import random
import unittest

from pattern_induction.chart import chart_agrees, gold_facts
from pattern_induction.program import agrees, describe, sample_compose_task, sample_dsl_task
from pattern_induction.recipes import d1_bank
from pattern_induction.version_space import assemble, distractors


FAMILIES = (
    "primitive",
    "repeat",
    "optional",
    "alternation",
    "prefix_repeat",
    "repeat_optional",
    "prefix_alt",
    "optional_internal",
    "repeat_extra",
)


class VersionSpaceTest(unittest.TestCase):
    def test_gold_facts_survive_two_distractors_per_head(self):
        for family in FAMILIES:
            for seed in range(4):
                task = sample_compose_task(family, random.Random(seed), probe_n=3)
                facts = gold_facts(task.program, task.supports)
                extra = distractors(task.supports, facts, 2, random.Random(100 + seed))
                built = assemble(task.supports, facts + extra)
                self.assertFalse(built.truncated, (family, seed, built.library))
                self.assertTrue(
                    chart_agrees(task.program, task.supports, task.probe),
                    (family, describe(task.program)),
                )
                hit = any(agrees(prog, task.supports, task.probe) for prog, _score in built.programs)
                self.assertTrue(hit, (family, describe(task.program), len(built.programs), built.library))

    def test_compiled_schemas_survive_distractors(self):
        for spec in d1_bank():
            task = sample_dsl_task(spec, random.Random(1), probe_n=2)
            facts = gold_facts(task.program, task.supports)
            extra = distractors(task.supports, facts, 1, random.Random(7))
            built = assemble(task.supports, facts + extra)
            self.assertFalse(built.truncated, spec.ident)
            self.assertTrue(
                any(agrees(prog, task.supports, task.probe) for prog, _score in built.programs),
                (spec.ident, describe(task.program), len(built.programs)),
            )


if __name__ == "__main__":
    unittest.main()
