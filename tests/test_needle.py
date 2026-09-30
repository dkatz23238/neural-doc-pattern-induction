import inspect
import json
import re
import tempfile
import time
import unittest
from pathlib import Path

from pattern_induction.candidates import semantic_key
from pattern_induction.confidence import NAMES, Signals, fitted_confidence, load_fit
from pattern_induction.needle import NeedleError, _confidence, needle
from pattern_induction.program import Prog, SearchTimeout, execute, set_deadline
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

    def test_long_examples_reach_the_selector_whole(self) -> None:
        examples = [
            "3f2b8c1e-9a4d-4e7b-8c2f-1a6d9e0b7c3a",
            "a1c9e7f3-5b2d-4c8e-9f1a-7d3b6e2c0f4b",
            "0e8d6c4b-2a1f-4e9d-8c7b-6a5f4e3d2c1b",
        ]
        with tempfile.TemporaryDirectory() as empty:
            result = needle(examples, Path(empty))
        self.assertEqual(result.regex, symbolic_pick(examples).pattern)
        compiled = re.compile(result.regex)
        self.assertTrue(all(compiled.fullmatch(text) for text in examples))

    def test_the_time_limit_keeps_the_symbolic_answer(self) -> None:
        examples = ["CUS-DE1234", "CUS-SO9281", "CUS-AM0042"]
        needle(examples, ROOT)
        started = time.perf_counter()
        result = needle(examples, ROOT, time_limit=0.0)
        self.assertLess(time.perf_counter() - started, 2.0)
        self.assertEqual(result.regex, symbolic_pick(examples).pattern)
        self.assertIsNone(result.ast)
        self.assertGreater(result.confidence, 0.0)
        self.assertLess(result.confidence, 1.0)

    def test_the_deadline_stops_program_construction(self) -> None:
        set_deadline(time.perf_counter() - 1.0)
        try:
            with self.assertRaises(SearchTimeout):
                for index in range(1000):
                    Prog("COPY", start=index, end=index + 1)
        finally:
            set_deadline(None)
        Prog("COPY")

    def test_the_fitted_confidence_rewards_agreement(self) -> None:
        fit = load_fit(ROOT)
        self.assertIsNotNone(fit)
        self.assertEqual(tuple(fit["names"]), NAMES)
        shared = dict(
            regex=r"^CUS-[A-Z]{2}\d{4}$",
            supports=["CUS-DE1234", "CUS-SO9281", "CUS-AM0042"],
            margin=4.0,
            near_tie=False,
            n_alternatives=5,
            cost=90.0,
            learned=True,
            prior_conf=0.9,
            n_programs=100,
        )
        agreed = fitted_confidence(Signals(agrees=True, **shared), fit)
        dissent = fitted_confidence(Signals(agrees=False, **shared), fit)
        self.assertGreater(agreed, dissent)
        self.assertTrue(0.0 < dissent < agreed < 1.0)

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
        self.assertLess(dissent, 0.8)
        self.assertLess(agreed, 1.0)

    def test_a_narrow_margin_is_not_reported_as_certain(self) -> None:
        wide = _confidence(4.0, True, 0.9, ambiguous=False)
        narrow = _confidence(0.2, True, 0.9, ambiguous=True)
        dissent = _confidence(4.0, False, 0.9, ambiguous=False)
        self.assertGreater(wide, dissent)
        self.assertGreater(dissent, narrow)
        self.assertLess(narrow, 0.7)
        self.assertLess(wide, 1.0)

    def test_alternatives_are_semantically_distinct(self) -> None:
        self.assertEqual(semantic_key(r"^PO-\d{4}$"), semantic_key(r"^PO\-[0-9]{4}$"))
        self.assertEqual(semantic_key(r"^(?:\d\.){2}\d$"), semantic_key(r"^[0-9]\.\d\.\d$"))
        self.assertNotEqual(semantic_key(r"^PO-\d{4}$"), semantic_key(r"^PO-?\d{4}$"))
        examples = ["PO-1000", "PO-1042", "PO-1108"]
        result = needle(examples, ROOT)
        self.assertEqual(result.regex, symbolic_pick(examples).pattern)
        self.assertFalse(result.ambiguous)
        self.assertIsNotNone(result.ast)
        self.assertGreater(result.confidence, 0.8)
        keys = [semantic_key(result.regex)]
        for pattern in result.alternatives:
            key = semantic_key(pattern)
            self.assertNotIn(key, keys)
            keys.append(key)

    def test_the_override_is_not_on_the_inference_path(self) -> None:
        import pattern_induction.needle as needle_mod

        source = inspect.getsource(needle_mod)
        self.assertNotIn("safe_override", source)
        self.assertNotIn("SafeOverride", source)


if __name__ == "__main__":
    unittest.main()
