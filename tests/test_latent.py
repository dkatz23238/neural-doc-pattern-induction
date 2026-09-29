import unittest

from pattern_induction.charset import v2_facts
from pattern_induction.latent import expand_program, expand_programs, factor_supports
from pattern_induction.program import Prog, execute, to_regex
from pattern_induction.version_space import assemble


class LatentExpansionTest(unittest.TestCase):
    def test_integer_hull_grows_to_a_containing_convention(self):
        supports = ["2024", "2025", "2026", "2024"]
        hull = Prog("INT", spec="2024:2026", lo=4, hi=4)
        programs = expand_program(hull, supports)
        regexes = [to_regex(prog, supports) for prog in programs]
        self.assertTrue(any(execute(prog, "2020", supports) and execute(prog, "2029", supports) for prog in programs))
        self.assertTrue(any(execute(prog, "2000", supports) and not execute(prog, "1999", supports) for prog in programs))
        self.assertTrue(all(execute(prog, text, supports) for prog in programs for text in supports))
        self.assertTrue(any(r"\d" in regex for regex in regexes))

    def test_character_hull_grows_to_a_natural_superclass(self):
        supports = ["A7", "F2", "B8", "C0"]
        hull = Prog("SET", spec="0-8A-F", lo=2, hi=2)
        programs = expand_program(hull, supports)
        self.assertTrue(any(prog.op == "SET" and prog.spec == "0-9A-F" for prog in programs))
        self.assertTrue(any(prog.op == "SET" and prog.spec == "0-9A-Z" for prog in programs))
        hex_set = next(prog for prog in programs if prog.spec == "0-9A-F")
        self.assertTrue(execute(hex_set, "F9", supports))
        self.assertFalse(execute(hex_set, "G0", supports))

    def test_date_fields_widen_together(self):
        supports = ["1995-12-10", "1996-05-24", "2008-07-21", "2023-06-17"]
        built = assemble(supports, v2_facts(supports), pack=True)
        seeds = [prog for prog, _score in built.programs]
        programs = expand_programs(seeds, supports, cap=400)
        hidden = "1990-01-31"
        self.assertTrue(any(execute(prog, hidden, supports) for prog in programs))
        self.assertTrue(any(execute(prog, "2023-06-17", supports) and not execute(prog, "2023-13-17", supports) and not execute(prog, "1899-06-17", supports) for prog in programs))

    def test_same_century_dates_widen_jointly(self):
        supports = ["1994-06-17", "1993-03-17", "1990-12-05", "1996-10-14"]
        year = Prog("INT", spec="1990:1996", lo=4, hi=4)
        month = Prog("INT", spec="3:12", lo=2, hi=2)
        day = Prog("INT", spec="5:17", lo=2, hi=2)
        dash = Prog("COPY", slot=0, start=4, end=5)
        program = Prog("SEQ", children=[year, dash, month, Prog("COPY", slot=0, start=7, end=8), day])
        programs = expand_program(program, supports, cap=200)
        self.assertTrue(
            any(
                execute(prog, "2008-07-21", supports)
                and execute(prog, "1990-01-31", supports)
                and not execute(prog, "2008-13-21", supports)
                and not execute(prog, "1899-06-17", supports)
                for prog in programs
            )
        )

    def test_thousands_groups_can_be_absent(self):
        supports = ["USD 7,646.37", "USD 13,590.47", "USD 10,728.33", "USD 14,394.97"]
        programs = expand_programs([], supports, cap=200)
        self.assertTrue(
            any(
                execute(prog, "USD 264.81", supports)
                and execute(prog, "USD 1,234,567.89", supports)
                and not execute(prog, "USDz10,641.50", supports)
                and not execute(prog, "USD 6Y486.00", supports)
                for prog in programs
            )
        )

    def test_repeated_hex_pairs_share_one_class(self):
        supports = ["71:8A:86:3F:A3:70", "2B:AB:51:8E:63:E2", "F5:EF:43:33:DA:C2", "3D:51:62:65:4C:E6"]
        programs = expand_programs([], supports, cap=200)
        self.assertTrue(
            any(
                execute(prog, "9F:AF:2F:07:FB:93", supports)
                and not execute(prog, "2B:AB51:8E:63:E2", supports)
                and not execute(prog, "71:8A:86:3F:A3:70 ", supports)
                for prog in programs
            )
        )
        self.assertTrue(factor_supports(supports))

    def test_observed_level_words_stay_while_the_clock_widens(self):
        supports = ["WARN 12:06:10 fetch_page", "ERROR 22:36:26 auth", "WARN 15:40:33 db_connect", "INFO 18:59:55 fetch_page"]
        programs = expand_programs([], supports, cap=400)
        self.assertTrue(
            any(
                execute(prog, "INFO 11:52:44 boot", supports)
                and not execute(prog, "RROR 09:59:28 auth", supports)
                for prog in programs
            )
        )

    def test_alternation_of_literals_lifts_to_a_class(self):
        supports = ["AB1234", "DE5521", "ZZ0007"]
        left = Prog("ALT", children=[Prog("COPY", slot=0, start=0, end=2), Prog("COPY", slot=1, start=0, end=2)])
        # Two branches are not enough to see Z. Include the third by nesting.
        letters = Prog(
            "ALT",
            children=[
                Prog("COPY", slot=0, start=0, end=2),
                Prog(
                    "ALT",
                    children=[Prog("COPY", slot=1, start=0, end=2), Prog("COPY", slot=2, start=0, end=2)],
                ),
            ],
        )
        digits = Prog("CLS", cls="digit", lo=4, hi=4)
        program = Prog("SEQ", children=[letters, digits])
        programs = expand_program(program, supports)
        self.assertTrue(any(execute(prog, "QX1234", supports) and not execute(prog, "Qx1234", supports) for prog in programs))
        self.assertFalse(any(prog is left for prog in programs))


if __name__ == "__main__":
    unittest.main()
