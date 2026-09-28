import unittest

from pattern_induction.contingent import contingent_optionals


class ContingentOptionalTest(unittest.TestCase):
    def test_a_rare_final_delimiter_licenses_its_suffix(self):
        supports = ["16_2", "02_1_82-MZ", "57_87_4_70", "2_33_67"]
        hyps = contingent_optionals(supports)
        self.assertTrue(any(hyp.kind == "optional" and hyp.slot == 1 and hyp.start == 7 and hyp.end == 10 for hyp in hyps))
        other = ["a1a", "b2b#QQ", "c3c"]
        hyps = contingent_optionals(other)
        self.assertTrue(any(hyp.slot == 1 and hyp.start == 3 and hyp.end == 6 for hyp in hyps))
        doubled = ["aa", "bb--QQ", "cc"]
        hyps = contingent_optionals(doubled)
        self.assertTrue(any(hyp.slot == 1 and hyp.start == 2 and hyp.end == 6 for hyp in hyps))

    def test_a_shared_separator_and_an_internal_mark_stay_quiet(self):
        self.assertEqual(contingent_optionals(["16_2", "02_1_82", "57_87_4_70", "2_33_67"]), [])
        self.assertEqual(contingent_optionals(["a#b_c", "d_e", "f_g"]), [])
        self.assertEqual(contingent_optionals(["-MZ"]), [])


if __name__ == "__main__":
    unittest.main()
