import unittest

from pattern_induction.language import GrammarPrior, choose_family
from pattern_induction.operator_witness import (
    complete_mixed,
    integer_abstentions,
    operator_judgments,
    split_operator_families,
)
from pattern_induction.program import Prog, execute
from pattern_induction.stability import direct_stability


VIN = [
    "E74ADG2EGHV6BLC54",
    "7FH7LCGTBKM2022JX",
    "RDNXHFE8DUP0YW7FE",
    "LUXE0Y4EUDKXYRFY5",
]


class OperatorWitnessTest(unittest.TestCase):
    def test_complete_mixed_is_the_full_alphabet_only(self):
        self.assertTrue(complete_mixed("0-9A-Z"))
        self.assertFalse(complete_mixed("0-8A-Y"))
        self.assertFalse(complete_mixed("2-5"))
        self.assertFalse(complete_mixed("0-9"))

    def test_exact_fragment_families_do_not_split(self):
        digit = Prog("CLS", cls="digit", lo=4, hi=4)
        copy = Prog("COPY", slot=0, start=0, end=3)
        self.assertEqual(len(split_operator_families([[digit, copy]])), 1)

    def test_hull_and_broad_set_become_two_families(self):
        hull = Prog("SET", spec="0-8A-Y", lo=17, hi=17)
        broad = Prog("SET", spec="0-9A-Z", lo=17, hi=17)
        families = split_operator_families([[hull, broad]])
        self.assertEqual(len(families), 2)

    def test_digit_probe_prefers_the_complete_mixed_set(self):
        hull = Prog("SET", spec="0-8A-Y", lo=17, hi=17)
        broad = Prog("SET", spec="0-9A-Z", lo=17, hi=17)
        self.assertTrue(all(execute(hull, text, VIN) for text in VIN))
        self.assertTrue(all(execute(broad, text, VIN) for text in VIN))
        families = [[hull], [broad]]
        judgments = operator_judgments(families, VIN)
        self.assertGreaterEqual(len(judgments), 2)
        self.assertTrue(all(prob >= 0.8 for _text, prob in judgments))
        self.assertTrue(any(any(ch.isdigit() for ch in text) and text not in VIN for text, _prob in judgments))
        prior = GrammarPrior()
        chosen = choose_family(families, VIN, judgments, prior, {}, 0.3)
        self.assertEqual(chosen.spec, "0-9A-Z")

    def test_integer_outside_the_observed_span_abstains(self):
        supports = ["2024", "2025", "2026", "2027"]
        digit = Prog("CLS", cls="digit", lo=4, hi=4)
        integer = Prog("INT", spec="2024:2027", lo=4, hi=4)
        families = [[digit], [integer]]
        self.assertFalse(operator_judgments(families, supports))
        abstained = integer_abstentions(families, supports)
        self.assertIn(("2023", 0.5), abstained)
        self.assertIn(("2028", 0.5), abstained)
        self.assertTrue(all(0.2 < prob < 0.8 for _text, prob in abstained))
        prior = GrammarPrior()
        for _draw in range(8):
            prior.observe(Prog("CLS", cls="digit", lo=4, hi=4))
        chosen = choose_family(families, supports, abstained, prior, {}, 0.3)
        self.assertEqual(chosen.op, "CLS")

    def test_direct_stability_prefers_a_complete_class_and_does_not_resynthesize(self):
        supports = ["AB1", "CD2", "EF3", "GH4"]
        hull = Prog("SET", spec="1-4A-H", lo=3, hi=3)
        broad = Prog("SET", spec="0-9A-Z", lo=3, hi=3)
        import pattern_induction.select_train as selector

        def boom(*_args, **_kwargs):
            raise AssertionError("resynthesis")

        original = selector.synthesize
        selector.synthesize = boom
        try:
            scores = direct_stability([[hull], [broad]], supports)
        finally:
            selector.synthesize = original
        self.assertGreater(scores[1], scores[0])
        self.assertEqual(scores[1], 1.0)


if __name__ == "__main__":
    unittest.main()
