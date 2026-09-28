import random
import unittest

from pattern_induction.chart import (
    chart_agrees,
    gold_facts,
    inventory_covers,
    mean_recall,
    recall,
)
from pattern_induction.program import describe, sample_compose_task, sample_dsl_task
from pattern_induction.recipes import d1_bank


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


class ChartTest(unittest.TestCase):
    def test_gold_facts_cover_the_inventory_and_rebuild_the_language(self):
        for family in FAMILIES:
            for seed in range(4):
                task = sample_compose_task(family, random.Random(seed), probe_n=3)
                facts = gold_facts(task.program, task.supports)
                self.assertTrue(facts, family)
                self.assertTrue(inventory_covers(facts, task.supports), (family, describe(task.program), facts[:6]))
                self.assertTrue(chart_agrees(task.program, task.supports, task.probe), (family, describe(task.program)))
                self.assertEqual(mean_recall(facts, facts), 1.0)

    def test_repeat_and_optional_facts_are_present(self):
        repeat = sample_compose_task("prefix_repeat", random.Random(0), 1)
        facts = gold_facts(repeat.program, repeat.supports)
        self.assertTrue(any(fact.kind == "repeat" for fact in facts))
        self.assertTrue(any(fact.kind == "copy" for fact in facts))
        self.assertTrue(any(fact.kind == "pair" for fact in facts))
        optional = sample_compose_task("optional", random.Random(0), 1)
        ofacts = gold_facts(optional.program, optional.supports)
        self.assertTrue(any(fact.kind == "optional" for fact in ofacts))
        alt = sample_compose_task("prefix_alt", random.Random(0), 1)
        afacts = gold_facts(alt.program, alt.supports)
        self.assertEqual(recall(afacts, afacts, "alt")[0], 1)

    def test_compiled_schemas_rebuild(self):
        for spec in d1_bank():
            task = sample_dsl_task(spec, random.Random(1), probe_n=2)
            self.assertTrue(inventory_covers(gold_facts(task.program, task.supports), task.supports), spec.ident)
            self.assertTrue(chart_agrees(task.program, task.supports, task.probe), spec.ident)


if __name__ == "__main__":
    unittest.main()
