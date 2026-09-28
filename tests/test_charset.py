import re
import unittest

from pattern_induction.charset import (
    charset_spec,
    integer_matches,
    integer_regex,
    interval_regex,
    spec_matches,
    v2_facts,
)
from pattern_induction.program import execute, to_regex
from pattern_induction.select_train import complete_assembly
from pattern_induction.version_space import assemble


class CharsetTest(unittest.TestCase):
    def test_interval_regex_compiles_the_numeric_hull(self):
        short = interval_regex(24, 35, 2)
        self.assertEqual(short, "(?:2[4-9]|3[0-5])")
        self.assertIsNotNone(re.fullmatch(short, "24"))
        self.assertIsNotNone(re.fullmatch(short, "35"))
        self.assertIsNone(re.fullmatch(short, "23"))
        self.assertIsNone(re.fullmatch(short, "36"))
        years = interval_regex(2024, 2035, 4)
        self.assertEqual(years, "(?:202[4-9]|203[0-5])")
        self.assertIsNotNone(re.fullmatch(years, "2024"))
        self.assertIsNotNone(re.fullmatch(years, "2035"))
        self.assertIsNone(re.fullmatch(years, "2023"))
        self.assertIsNone(re.fullmatch(years, "2036"))
        months = interval_regex(1, 12, 2)
        self.assertEqual(months, "(?:0[1-9]|1[0-2])")

    def test_variable_width_integer_rejects_leading_zeros(self):
        self.assertTrue(integer_matches("0:255", "0", 1, 3))
        self.assertTrue(integer_matches("0:255", "255", 1, 3))
        self.assertFalse(integer_matches("0:255", "007", 1, 3))
        self.assertFalse(integer_matches("0:255", "256", 1, 3))
        pattern = integer_regex("0:255", 1, 3)
        self.assertIsNotNone(re.fullmatch(pattern, "255"))
        self.assertIsNone(re.fullmatch(pattern, "256"))

    def test_charset_spec_is_the_observed_alphabet(self):
        spec = charset_spec(set("0123456789abcdef"))
        self.assertEqual(spec, "0-9a-f")
        self.assertTrue(spec_matches(spec, "ab12"))
        self.assertFalse(spec_matches(spec, "ab12G"))

    def test_mixed_token_becomes_one_character_set(self):
        supports = ["TXN-2026-5WDYLK", "TXN-2023-7ZN4YL", "TXN-2020-7I4HFB", "TXN-2023-8UCL6C"]
        facts = v2_facts(supports)
        specs = {fact.cls for fact in facts if fact.kind == "set"}
        self.assertIn("0-9A-Z", specs)
        built = assemble(supports, facts, pack=True)
        regexes = [to_regex(prog, supports) for prog, _score in built.programs]
        self.assertTrue(any("[0-9A-Z]{6}" in regex for regex in regexes))
        accepting = [prog for prog, _score in built.programs if execute(prog, "TXN-9999-AB12CD", supports)]
        self.assertTrue(accepting)
        self.assertFalse(any(execute(prog, "TXN-2026-abcdef", supports) for prog in accepting))

    def test_observed_year_hull_does_not_invent_an_endpoint(self):
        supports = ["INV-2026-82047", "INV-2026-81776", "INV-2035-03997", "INV-2026-46376"]
        facts = v2_facts(supports)
        integers = [fact.cls for fact in facts if fact.kind == "int"]
        self.assertIn("2026:2035", integers)
        self.assertNotIn("2024:2035", integers)
        classes = [fact for fact in facts if fact.kind == "cls" and fact.cls == "digit" and fact.lo == 4]
        self.assertTrue(classes)

    def test_exact_digit_width_survives_v2_facts(self):
        supports = ["INV-1588", "INV-1399", "INV-8698", "INV-7973"]
        frozen = complete_assembly(supports, [])
        opened = complete_assembly(supports, [], v2=True)
        frozen_regex = {to_regex(prog, supports) for prog, _score in frozen.programs}
        opened_regex = {to_regex(prog, supports) for prog, _score in opened.programs}
        self.assertIn(r"^INV\-\d{4}$", frozen_regex)
        self.assertIn(r"^INV\-\d{4}$", opened_regex)
        self.assertTrue(any("INT" in to_regex(prog, supports) or "1399" in to_regex(prog, supports) for prog, _score in opened.programs))


if __name__ == "__main__":
    unittest.main()
