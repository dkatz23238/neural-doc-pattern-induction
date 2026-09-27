import random
import re
import unittest

from pattern_induction.candidates import (
    anchor,
    parse_pattern,
    pattern_cost,
    presented_candidates,
)
from pattern_induction.heuristic import infer_regex


def _hex_strings(n: int = 6, width: int = 32, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice("0123456789abcdef") for _ in range(width)) for _ in range(n)]


class ParserTest(unittest.TestCase):
    def test_round_trip_common_shapes(self):
        patterns = [
            r"^INV-2026-\d{5}$",
            r"^[A-Za-z0-9]{18}\d[A-Za-z0-9]{13}$",
            r"^[0-9a-f]{32}$",
            r"^0\d/\d{2}/\d{4}$",
            r"^(?:33)?\.?\d{1,3} kg$",
        ]
        for pattern in patterns:
            parsed = parse_pattern(pattern)
            self.assertIsNotNone(parsed, pattern)
            self.assertEqual(anchor(parsed), pattern)

    def test_short_digit_literal_is_not_glued_to_a_class(self):
        parsed = parse_pattern(r"^0\d$")
        rendered = anchor(parsed)
        self.assertEqual(rendered, r"^0\d$")


class CostTest(unittest.TestCase):
    def test_rules_generalize_a_single_digit_and_mdl_keeps_it(self):
        strings = ["01", "02", "03", "04", "05", "06"]
        narrow = pattern_cost(r"^0\d$", strings, digit_prior=False)
        wide = pattern_cost(r"^\d{2}$", strings, digit_prior=False)
        self.assertLess(narrow, wide)
        rules_narrow = pattern_cost(r"^0\d$", strings, digit_prior=True)
        rules_wide = pattern_cost(r"^\d{2}$", strings, digit_prior=True)
        self.assertLess(rules_wide, rules_narrow)

    def test_both_costs_keep_a_four_digit_year(self):
        strings = [
            "INV-2026-00192",
            "INV-2026-93821",
            "INV-2026-10293",
            "INV-2026-00047",
            "INV-2026-11111",
            "INV-2026-22222",
        ]
        for prior in (False, True):
            literal = pattern_cost(r"^INV-2026-\d{5}$", strings, prior)
            generalized = pattern_cost(r"^INV-\d{4}-\d{5}$", strings, prior)
            self.assertLess(literal, generalized)

    def test_mdl_prefers_hex_to_the_broad_alnum_class(self):
        strings = _hex_strings()
        hex_cost = pattern_cost(r"^[0-9a-f]{32}$", strings, False)
        alnum_cost = pattern_cost(r"^[A-Za-z0-9]{32}$", strings, False)
        loose = pattern_cost(r"^[A-Za-z0-9]{0,32}$", strings, False)
        self.assertLess(hex_cost, alnum_cost)
        self.assertLess(alnum_cost, loose)


class CandidateSetTest(unittest.TestCase):
    def test_hex_neighborhood_contains_the_restricted_class(self):
        strings = _hex_strings()
        shown, pool = presented_candidates(strings, k=8)
        self.assertIn(r"^[0-9a-f]{32}$", [item.pattern for item in shown])
        self.assertLessEqual(len(shown), 8)
        self.assertGreaterEqual(len(pool), len(shown))
        for item in pool:
            compiled = re.compile(item.pattern)
            self.assertTrue(all(compiled.fullmatch(text) for text in strings))

    def test_invoice_keeps_the_heuristic_and_the_year(self):
        strings = ["INV-2026-00192", "INV-2026-93821", "INV-2026-10293", "INV-2026-00047"]
        shown, _pool = presented_candidates(strings, k=8)
        self.assertEqual(shown[0].pattern, infer_regex(strings))
        self.assertEqual(shown[0].pattern, r"^INV-2026-\d{5}$")
        for item in shown:
            compiled = re.compile(item.pattern)
            self.assertTrue(all(compiled.fullmatch(text) for text in strings))

    def test_accidental_leading_zero_has_a_digit_class_neighbor(self):
        strings = ["01/15/2024", "02/15/2024", "03/15/2024", "04/15/2024", "05/15/2024"]
        _shown, pool = presented_candidates(strings, k=8)
        patterns = [item.pattern for item in pool]
        self.assertIn(r"^\d{2}/15/2024$", patterns)
        self.assertTrue(infer_regex(strings).startswith("^0"))


if __name__ == "__main__":
    unittest.main()
