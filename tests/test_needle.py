import inspect
import json
import re
import unittest
from pathlib import Path

from pattern_induction.needle import NeedleError, _confidence, needle
from pattern_induction.program import execute
from pattern_induction.symbolic import symbolic_pick
from pattern_induction.tasks import Task


ROOT = Path(__file__).resolve().parents[1]


class SymbolicPickTest(unittest.TestCase):
    def test_matches_the_recorded_bench_c_rule(self) -> None:
        tasks = {}
        with (ROOT / "data" / "bench_c.jsonl").open() as handle:
            for line in handle:
                task = Task.from_obj(json.loads(line))
                tasks[task.id] = task
                if len(tasks) == 8:
                    break
        checked = 0
        with (ROOT / "results" / "exp2" / "candidate_oracle_tasks.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["id"] not in tasks:
                    continue
                pick = symbolic_pick(tasks[row["id"]].visible)
                self.assertEqual(pick.pattern, row["rules"]["pattern"])
                checked += 1
        self.assertEqual(checked, 8)

    def test_the_winner_is_not_an_alternative(self) -> None:
        pick = symbolic_pick(["INV-10021", "INV-88310", "INV-00418"])
        self.assertNotIn(pick.pattern, pick.alternatives)
        self.assertTrue(pick.pattern.startswith("^"))


class NeedleTest(unittest.TestCase):
    def test_the_symbolic_regex_is_the_decision(self) -> None:
        examples = ["CUS-DE1234", "CUS-SO9281", "CUS-AM0042"]
        result = needle(examples, ROOT)
        pick = symbolic_pick(examples)
        self.assertEqual(result.regex, pick.pattern)
        self.assertEqual(result.method, "symbolic")
        self.assertGreater(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertLessEqual(len(result.alternatives), 4)
        self.assertNotIn(result.regex, result.alternatives)
        compiled = re.compile(result.regex)
        self.assertTrue(all(compiled.fullmatch(text) for text in examples))
        if result.ast is not None:
            self.assertTrue(all(execute(result.ast, text, examples) for text in examples))
        again = needle(examples, ROOT)
        self.assertEqual(again.regex, result.regex)
        self.assertEqual(again.alternatives, result.alternatives)
        self.assertEqual(again.ambiguous, result.ambiguous)

    def test_every_example_reaches_the_selector(self) -> None:
        examples = ["INV-1000", "INV-1001", "INV-1002", "INV-1003", "INV-1004", "INV-1005"]
        result = needle(examples, ROOT)
        self.assertEqual(result.regex, symbolic_pick(examples).pattern)

    def test_one_example_abstains(self) -> None:
        result = needle(["CUS-DE1234"], ROOT)
        self.assertEqual(result.method, "abstain")
        self.assertEqual(result.regex, "")
        self.assertEqual(result.confidence, 0.0)
        self.assertFalse(result.ambiguous)

    def test_bad_examples_raise(self) -> None:
        with self.assertRaises(NeedleError):
            needle("CUS-DE1234", ROOT)
        with self.assertRaises(NeedleError):
            needle(["CUS-DE1234", b"CUS-SO9281"], ROOT)

    def test_disagreement_lowers_confidence_without_a_new_regex(self) -> None:
        agreed = _confidence(2.0, True, 0.9)
        dissent = _confidence(2.0, False, 0.9)
        self.assertLess(dissent, agreed)
        self.assertLessEqual(agreed, 1.0)
        self.assertGreaterEqual(dissent, 0.0)

    def test_the_override_is_not_on_the_inference_path(self) -> None:
        import pattern_induction.needle as needle_mod

        source = inspect.getsource(needle_mod)
        self.assertNotIn("safe_override", source)
        self.assertNotIn("SafeOverride", source)


if __name__ == "__main__":
    unittest.main()
