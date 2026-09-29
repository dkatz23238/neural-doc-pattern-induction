import unittest

import torch

from pattern_induction.program import OPEN, Prog
from pattern_induction.renaming import apply_mapping
from prior.relative import (
    CHARSET_DIM,
    INTEGER_DIM,
    REPEAT_DIM,
    LocalScorer,
    Observations,
    _abstain,
    _view_for,
    _matched,
    charset_features,
    integer_features,
    pack_example,
    repeat_features,
)


class RelativeEvidenceTest(unittest.TestCase):
    def test_repeat_packet_ranks_bounds_without_reading_the_child(self):
        supports = ["12-34", "56-78-90", "11-22-33-44"]
        obs = Observations(supports)
        self.assertEqual(obs.repeat_counts, [1, 2, 3])
        tight = Prog("REPEAT", lo=1, hi=3, children=[Prog("CLS", cls="digit", lo=2, hi=2)])
        opened = Prog("REPEAT", lo=1, hi=OPEN, children=[Prog("INT", spec="10:99", lo=2, hi=2)])
        missed = Prog("REPEAT", lo=3, hi=3, children=[Prog("CLS", cls="digit", lo=2, hi=2)])
        tight_row = repeat_features(obs, tight)
        open_row = repeat_features(obs, opened)
        miss_row = repeat_features(obs, missed)
        self.assertEqual(len(tight_row), REPEAT_DIM)
        self.assertEqual(tight_row[6], 1.0)
        self.assertEqual(tight_row[7], 1.0)
        self.assertEqual(tight_row[8], 1.0)
        self.assertEqual(tight_row[9], 0.0)
        self.assertEqual(open_row[8], 1.0)
        self.assertGreater(open_row[9], 0.0)
        self.assertEqual(open_row[7], 0.0)
        self.assertEqual(miss_row[8], 0.0)
        self.assertLess(miss_row[9], 0.0)
        other_child = Prog("REPEAT", lo=1, hi=3, children=[Prog("CLS", cls="upper", lo=2, hi=2)])
        self.assertEqual(tight_row, repeat_features(obs, other_child))

    def test_integer_packet_records_slack_and_the_observed_hull(self):
        supports = ["2024", "2027", "2031"]
        obs = Observations(supports)
        self.assertEqual(obs.numbers, [2024, 2027, 2031])
        hull = Prog("INT", spec="2024:2031", lo=4, hi=4)
        wider = Prog("INT", spec="2020:2035", lo=4, hi=4)
        hull_row = integer_features(obs, hull)
        wide_row = integer_features(obs, wider)
        self.assertEqual(len(hull_row), INTEGER_DIM)
        self.assertEqual(hull_row[8], 1.0)
        self.assertEqual(wide_row[8], 0.0)
        self.assertEqual(hull_row[7], 1.0)
        self.assertEqual(wide_row[7], 1.0)
        self.assertGreater(wide_row[5], 0.0)
        self.assertGreater(wide_row[6], 0.0)
        self.assertEqual(hull_row[12], 1.0)
        self.assertEqual(wide_row[12], 0.0)

    def test_charset_packet_sees_containment_and_unseen_alphabet(self):
        supports = ["AX", "BZ", "XA", "ZB"]
        obs = Observations(supports)
        self.assertEqual(obs.chars, set("ABXZ"))
        upper = Prog("SET", spec="A-Z", lo=2, hi=2)
        alnum = Prog("SET", spec="0-9A-Z", lo=2, hi=2)
        digit = Prog("CLS", cls="digit", lo=2, hi=2)
        upper_row = charset_features(obs, upper)
        alnum_row = charset_features(obs, alnum)
        digit_row = charset_features(obs, digit)
        self.assertEqual(len(upper_row), CHARSET_DIM)
        self.assertEqual(upper_row[1], 1.0)
        self.assertEqual(alnum_row[1], 1.0)
        self.assertEqual(digit_row[1], 0.0)
        self.assertGreater(alnum_row[3], upper_row[3])
        self.assertEqual(upper_row[2], 1.0)
        self.assertEqual(digit_row[2], 0.0)

    def test_a_letter_rename_leaves_the_packets_in_place(self):
        supports = ["ABC2024", "ABC2025", "ABC2026", "ABC2024"]
        mapping = {"A": "X", "B": "Y", "C": "L"}
        renamed = [apply_mapping(text, mapping) for text in supports]
        program = Prog(
            "SEQ",
            children=[
                Prog("COPY", slot=0, start=0, end=3),
                Prog("INT", spec="2024:2026", lo=4, hi=4),
            ],
        )
        left = pack_example([program], [True], [True], supports)
        right = pack_example([program], [True], [True], renamed)
        for key, value in left.items():
            self.assertTrue(torch.equal(value, right[key]), key)

    def test_pack_marks_the_gold_hypothesis_and_keeps_an_abstain(self):
        supports = ["12-34", "56-78-90", "11-22-33-44"]
        tight = Prog("REPEAT", lo=1, hi=3, children=[Prog("CLS", cls="digit", lo=2, hi=2)])
        opened = Prog("REPEAT", lo=1, hi=OPEN, children=[Prog("CLS", cls="digit", lo=2, hi=2)])
        packed = pack_example([tight, opened], [True, False], [True, True], supports)
        gold = packed["repeat_hyp_gold"]
        mask = packed["repeat_hyp_mask"]
        self.assertEqual(float(packed["repeat_hyp"][0, -1]), 1.0)
        self.assertGreater(float((gold * mask).sum()), 0.0)
        self.assertEqual(float(gold[0]), 0.0)
        chosen = int(gold.argmax())
        self.assertEqual(float(packed["repeat_hyp"][chosen, 6]), 1.0)
        self.assertEqual(float(packed["repeat_hyp"][chosen, 7]), 1.0)
        empty = pack_example([], [], [], supports)
        self.assertEqual(float(empty["repeat_hyp_gold"].sum()), 1.0)
        self.assertEqual(_abstain(REPEAT_DIM)[-1], 1.0)

    def test_a_node_is_scored_against_the_span_it_matches(self):
        supports = ["615", "617", "616", "617"]
        last = Prog(
            "SEQ",
            children=[
                Prog("COPY", slot=0, start=0, end=2),
                Prog("INT", spec="0:9", lo=1, hi=1),
            ],
        )
        whole = Prog("INT", spec="610:619", lo=3, hi=3)
        field = Prog(
            "SEQ",
            children=[
                Prog("CLS", cls="digit", lo=5, hi=5),
                Prog(
                    "REPEAT",
                    lo=2,
                    hi=4,
                    children=[
                        Prog(
                            "SEQ",
                            children=[
                                Prog("COPY", slot=0, start=5, end=6),
                                Prog("CLS", cls="digit", lo=5, hi=5),
                            ],
                        )
                    ],
                ),
            ],
        )
        base = Observations(supports)
        last_int = last.children[1]
        view = _view_for(base, last_int, _matched(last, supports))
        self.assertEqual(sorted(view.numbers), [5, 6, 7, 7])
        row = integer_features(view, last_int)
        self.assertEqual(row[7], 1.0)
        whole_view = _view_for(base, whole, _matched(whole, supports))
        self.assertEqual(whole_view.number_min, 615)
        self.assertEqual(whole_view.number_max, 617)
        prefix = ["G:9fda0", "G:949c2", "G:7e87f", "G:b2a90"]
        charset = Prog(
            "SEQ",
            children=[
                Prog("COPY", slot=0, start=0, end=1),
                Prog("COPY", slot=0, start=1, end=2),
                Prog("SET", spec="0-9a-f", lo=5, hi=5),
            ],
        )
        field_node = charset.children[2]
        chars = _view_for(Observations(prefix), field_node, _matched(charset, prefix)).chars
        self.assertNotIn("G", chars)
        self.assertTrue(chars <= set("0123456789abcdef"))
        repeated = ["07441:26962:76743", "68523:04387:82198:28927:24362", "93778:35838:77722:14167:47399", "61758:37613:58663"]
        repeat = field.children[1]
        counts = _view_for(Observations(repeated), repeat, _matched(field, repeated)).repeat_counts
        self.assertEqual(counts, [2, 4, 4, 2])

    def test_two_repeat_bounds_score_apart_at_initialization(self):
        torch.manual_seed(0)
        model = LocalScorer()
        supports = ["12-34", "56-78-90", "11-22-33-44"]
        obs = Observations(supports)
        left = torch.tensor([repeat_features(obs, Prog("REPEAT", lo=1, hi=3))])
        right = torch.tensor([repeat_features(obs, Prog("REPEAT", lo=1, hi=OPEN))])
        self.assertFalse(torch.allclose(model.heads["repeat"](left), model.heads["repeat"](right)))


if __name__ == "__main__":
    unittest.main()
