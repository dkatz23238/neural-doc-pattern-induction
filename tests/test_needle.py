import unittest
from pathlib import Path

from pattern_induction.needle import needle
from pattern_induction.program import execute


class NeedleTest(unittest.TestCase):
    def test_customer_codes_return_a_regex_and_alternatives(self):
        examples = ["CUS-DE1234", "CUS-SO9281", "CUS-AM0042"]
        result = needle(examples, Path("."))
        self.assertTrue(result.regex.startswith("^"))
        self.assertTrue(result.regex.endswith("$"))
        self.assertIsNotNone(result.ast)
        self.assertIn(result.method, {"synthesis-v1", "dsl-v2"})
        self.assertGreater(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertTrue(all(execute(result.ast, text, examples) for text in examples))
        self.assertLessEqual(len(result.alternatives), 4)
        self.assertNotIn(result.regex, result.alternatives)

    def test_one_example_abstains(self):
        result = needle(["CUS-DE1234"], Path("."))
        self.assertEqual(result.method, "abstain")
        self.assertEqual(result.regex, "")
        self.assertEqual(result.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
