import unittest

from pattern_induction.contingent import contingent_optionals
from pattern_induction.chart import Hyp
from pattern_induction.document_facts import alignment_kind, document_facts
from pattern_induction.program import Prog, execute
from pattern_induction.select_train import complete_assembly
from pattern_induction.version_space import assemble


def _accepts(prog, text, supports) -> bool:
    return bool(execute(prog, text, supports))


class DocumentFactTest(unittest.TestCase):
    def test_varying_letter_run_is_a_class_and_the_prefix_is_a_copy(self):
        supports = ["USR-203-DE", "USR-507-SO", "USR-627-AM", "USR-316-VV"]
        self.assertEqual(alignment_kind(supports), "equal")
        hyps = document_facts(supports)
        self.assertTrue(any(hyp.kind == "copy" and hyp.slot == 0 and hyp.start == 0 and hyp.end == 3 for hyp in hyps))
        self.assertTrue(any(hyp.kind == "cls" and hyp.cls == "upper" and hyp.lo == 2 and hyp.hi == 2 for hyp in hyps))
        built = assemble(supports, hyps)
        self.assertTrue(built.programs)
        good = "USR-111-ZZ"
        bad = "EMP-111-ZZ"
        self.assertTrue(any(_accepts(prog, good, supports) and not _accepts(prog, bad, supports) for prog, _score in built.programs))

    def test_class_width_above_the_repeat_cap(self):
        supports = ["+" + "1" * 11, "+" + "2" * 13, "+" + "3" * 12, "+" + "4" * 11]
        self.assertTrue(any(hyp.kind == "cls" and hyp.cls == "digit" and hyp.lo == 11 and hyp.hi == 13 for hyp in document_facts(supports)))
        built = assemble(supports, document_facts(supports))
        self.assertTrue(
            any(
                _accepts(prog, "+" + "9" * 12, supports) and not _accepts(prog, "+" + "9" * 10, supports)
                for prog, _score in built.programs
            )
        )

    def test_constant_head_of_a_long_digit_run_stays_a_literal(self):
        body = ["00" + "1" * 18, "00" + "2" * 18, "00" + "3" * 18, "00" + "4" * 18]
        built = assemble(body, document_facts(body))
        self.assertTrue(
            any(
                _accepts(prog, "00" + "9" * 18, body) and not _accepts(prog, "99" + "9" * 18, body)
                for prog, _score in built.programs
            )
        )

    def test_two_observed_widths_stay_alternatives(self):
        supports = ["1" * 12, "2" * 12, "3" * 15, "4" * 15]
        built = assemble(supports, document_facts(supports))
        self.assertTrue(
            any(
                _accepts(prog, "9" * 12, supports)
                and _accepts(prog, "8" * 15, supports)
                and not _accepts(prog, "7" * 13, supports)
                and not _accepts(prog, "6" * 14, supports)
                for prog, _score in built.programs
            )
        )

    def test_a_short_shared_head_survives_a_longer_one(self):
        supports = ["32021M4834", "32022T1344", "32001B1479", "32016G6022"]
        built = assemble(supports, document_facts(supports))
        self.assertTrue(
            any(
                _accepts(prog, "31990A0001", supports) and not _accepts(prog, "41990A0001", supports)
                for prog, _score in built.programs
            )
        )

    def test_optional_prefix_can_be_an_alternation(self):
        supports = ["WEB-12345678", "POS-87654321", "12345678", "WEB-00001111"]
        self.assertEqual(alignment_kind(supports), "optional_prefix")
        built = assemble(supports, document_facts(supports), wrap_optional_alt=True)
        self.assertTrue(
            any(
                _accepts(prog, "WEB-22222222", supports)
                and _accepts(prog, "POS-33333333", supports)
                and _accepts(prog, "44444444", supports)
                and not _accepts(prog, "ABC-44444444", supports)
                for prog, _score in built.programs
            )
        )

    def test_two_skeletons_become_an_alternation(self):
        supports = ["N/A", "2020-01-02", "1999-12-31", "N/A"]
        self.assertEqual(alignment_kind(supports), "two_cluster")
        built = assemble(supports, document_facts(supports))
        self.assertTrue(
            any(
                _accepts(prog, "N/A", supports)
                and _accepts(prog, "2010-05-06", supports)
                and not _accepts(prog, "2010-5-6", supports)
                for prog, _score in built.programs
            )
        )

    def test_shared_hyphen_stays_out_of_the_contingent_rule(self):
        fixed = ["QZ3-5-9", "QZ7-1-4", "QZ3-5-9-CW", "QZ8-2-6-CW"]
        self.assertEqual(contingent_optionals(fixed), [])
        self.assertEqual(alignment_kind(fixed), "optional_suffix")
        varied = ["QZ3-5", "QZ3-5-9", "QZ7-1-4-8", "QZ3-5-9-CW"]
        self.assertEqual(contingent_optionals(varied), [])
        self.assertEqual(alignment_kind(varied), "none")
        self.assertEqual(document_facts(varied), [])

    def test_a_nonempty_chart_does_not_gain_facts(self):
        supports = ["AB", "CD", "EF", "GH"]
        hyps = [
            Hyp("cls", slot, 0, 2, score=3.0, cls="upper", lo=2, hi=2) for slot in range(4)
        ]
        first = assemble(supports, hyps)
        self.assertTrue(first.programs)
        done = complete_assembly(supports, hyps)
        self.assertEqual(
            [Prog("CLS", cls="upper", lo=2, hi=2)],
            [prog for prog, _score in done.programs],
        )
        self.assertEqual(len(done.programs), len(first.programs))


if __name__ == "__main__":
    unittest.main()
