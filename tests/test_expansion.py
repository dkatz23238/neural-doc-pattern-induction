import math
import unittest

import torch

from pattern_induction.charset import charset_spec
from pattern_induction.program import OPEN, Prog
from pattern_induction.renaming import apply_mapping
from prior.expansion import (
    CHARSET_CONTAINED,
    CHARSET_CONVENTIONAL,
    CHARSET_EXPANSION,
    CHARSET_INTRODUCES_DIGIT,
    CHARSET_INTRODUCES_LETTER,
    CHARSET_RATIO,
    CHARSET_UNSEEN,
    INTEGER_CONVENTION,
    INTEGER_EXPANSION,
    INTEGER_HULL,
    INTEGER_OBS_LO,
    INTEGER_RATIO,
    INTEGER_ROUND,
    INTEGER_SLACK_LEFT,
    INTEGER_SLACK_RIGHT,
    REPEAT_EXPANSION,
    WIDTH_EXPANSION,
    WIDTH_OPEN,
    WIDTH_SLACK_ABOVE,
    WIDTH_SLACK_BELOW,
    ExpansionPrior,
    freeze_evidence,
    pack_example,
    parameter_count,
    charset_features,
    integer_features,
    repeat_features,
    width_features,
)
from prior.relative import LocalScorer, Observations, _log_pos, _matched, _view_for


class ExpansionPriorTest(unittest.TestCase):
    def test_charset_expansion_is_a_property_of_the_proposed_superclass(self):
        supports = ["AA", "BC", "ZX"]
        obs = Observations(supports)
        self.assertEqual(obs.chars, set("ABCXZ"))
        exact = Prog("SET", spec=charset_spec(set("ABCXZ")), lo=2, hi=2)
        upper = Prog("SET", spec="A-Z", lo=2, hi=2)
        alnum = Prog("SET", spec="0-9A-Z", lo=2, hi=2)
        digits = Prog("CLS", cls="digit", lo=2, hi=2)
        mixed = Prog("SET", spec="A-Za-z", lo=2, hi=2)
        exact_row = charset_features(obs, exact)
        upper_row = charset_features(obs, upper)
        alnum_row = charset_features(obs, alnum)
        digit_row = charset_features(obs, digits)
        mixed_row = charset_features(obs, mixed)
        self.assertEqual(exact_row[CHARSET_CONTAINED], 1.0)
        self.assertEqual(exact_row[CHARSET_UNSEEN], 0.0)
        self.assertEqual(exact_row[CHARSET_EXPANSION], 0.0)
        self.assertEqual(upper_row[CHARSET_CONTAINED], 1.0)
        self.assertEqual(upper_row[CHARSET_EXPANSION], 1.0)
        self.assertEqual(upper_row[CHARSET_INTRODUCES_DIGIT], 0.0)
        self.assertEqual(upper_row[CHARSET_CONVENTIONAL], 1.0)
        self.assertEqual(alnum_row[CHARSET_CONTAINED], 1.0)
        self.assertEqual(alnum_row[CHARSET_INTRODUCES_DIGIT], 1.0)
        self.assertGreater(alnum_row[CHARSET_UNSEEN], upper_row[CHARSET_UNSEEN])
        self.assertGreater(alnum_row[CHARSET_RATIO], upper_row[CHARSET_RATIO])
        self.assertEqual(digit_row[CHARSET_CONTAINED], 0.0)
        self.assertEqual(digit_row[CHARSET_EXPANSION], 0.0)
        self.assertEqual(mixed_row[CHARSET_INTRODUCES_LETTER], 1.0)
        self.assertEqual(mixed_row[CHARSET_INTRODUCES_DIGIT], 0.0)

    def test_integer_expansion_records_slack_round_endpoints_and_the_hull(self):
        supports = ["2024", "2027", "2031"]
        obs = Observations(supports)
        hull = Prog("INT", spec="2024:2031", lo=4, hi=4)
        wider = Prog("INT", spec="2020:2035", lo=4, hi=4)
        full = Prog("INT", spec="0:9999", lo=4, hi=4)
        hull_row = integer_features(obs, hull)
        wide_row = integer_features(obs, wider)
        full_row = integer_features(obs, full)
        self.assertEqual(hull_row[INTEGER_HULL], 1.0)
        self.assertEqual(hull_row[INTEGER_EXPANSION], 0.0)
        self.assertEqual(hull_row[INTEGER_SLACK_LEFT], 0.0)
        self.assertEqual(hull_row[INTEGER_SLACK_RIGHT], 0.0)
        self.assertEqual(wide_row[INTEGER_HULL], 0.0)
        self.assertEqual(wide_row[INTEGER_EXPANSION], 1.0)
        self.assertGreater(wide_row[INTEGER_SLACK_LEFT], 0.0)
        self.assertGreater(wide_row[INTEGER_SLACK_RIGHT], 0.0)
        self.assertEqual(wide_row[INTEGER_ROUND], 0.0)
        self.assertEqual(full_row[INTEGER_ROUND], 1.0)
        self.assertEqual(full_row[INTEGER_CONVENTION], 0.0)
        self.assertEqual(full_row[INTEGER_EXPANSION], 1.0)
        self.assertGreater(full_row[INTEGER_RATIO], wide_row[INTEGER_RATIO])
        small = Observations(["3", "8"])
        convention = integer_features(small, Prog("INT", spec="0:100", lo=1, hi=2))
        self.assertEqual(convention[INTEGER_CONVENTION], 1.0)
        self.assertEqual(convention[INTEGER_ROUND], 0.0)
        self.assertEqual(convention[INTEGER_EXPANSION], 1.0)

    def test_integer_slack_uses_the_matched_span(self):
        supports = ["615", "617", "616", "617"]
        program = Prog(
            "SEQ",
            children=[
                Prog("COPY", slot=0, start=0, end=2),
                Prog("INT", spec="0:9", lo=1, hi=1),
            ],
        )
        view = _view_for(Observations(supports), program.children[1], _matched(program, supports))
        row = integer_features(view, program.children[1])
        self.assertAlmostEqual(row[INTEGER_OBS_LO], _log_pos(5.0), places=5)
        self.assertGreater(row[INTEGER_SLACK_LEFT], 0.0)
        self.assertEqual(row[INTEGER_EXPANSION], 1.0)
        self.assertNotAlmostEqual(row[INTEGER_OBS_LO], _log_pos(615.0), places=3)

    def test_width_expansion_separates_an_exact_bound_from_an_open_one(self):
        supports = ["AB", "CD", "EF"]
        obs = Observations(supports)
        exact = Prog("CLS", cls="upper", lo=2, hi=2)
        opened = Prog("CLS", cls="upper", lo=2, hi=128)
        wider = Prog("CLS", cls="upper", lo=1, hi=4)
        exact_row = width_features(obs, exact)
        open_row = width_features(obs, opened)
        wide_row = width_features(obs, wider)
        self.assertEqual(exact_row[WIDTH_EXPANSION], 0.0)
        self.assertEqual(open_row[WIDTH_EXPANSION], 1.0)
        self.assertEqual(open_row[WIDTH_OPEN], 1.0)
        self.assertEqual(wide_row[WIDTH_OPEN], 0.0)
        self.assertEqual(wide_row[WIDTH_EXPANSION], 1.0)
        self.assertGreater(wide_row[WIDTH_SLACK_BELOW], 0.0)
        self.assertGreater(wide_row[WIDTH_SLACK_ABOVE], 0.0)

    def test_loose_repeat_is_independent_of_the_child(self):
        supports = ["12-34", "56-78-90", "11-22-33-44"]
        obs = Observations(supports)
        tight = Prog("REPEAT", lo=1, hi=3, children=[Prog("CLS", cls="digit", lo=2, hi=2)])
        opened = Prog("REPEAT", lo=1, hi=OPEN, children=[Prog("INT", spec="10:99", lo=2, hi=2)])
        other = Prog("REPEAT", lo=1, hi=3, children=[Prog("CLS", cls="upper", lo=2, hi=2)])
        tight_row = repeat_features(obs, tight)
        open_row = repeat_features(obs, opened)
        self.assertEqual(tight_row[REPEAT_EXPANSION], 0.0)
        self.assertEqual(open_row[REPEAT_EXPANSION], 1.0)
        self.assertEqual(tight_row, repeat_features(obs, other))
        self.assertGreater(open_row[REPEAT_EXPANSION], tight_row[REPEAT_EXPANSION])

    def test_a_letter_rename_leaves_charset_and_integer_packets_in_place(self):
        supports = ["AA", "BC", "ZX"]
        mapping = {"A": "Q", "B": "W", "C": "E", "Z": "R", "X": "T"}
        renamed = [apply_mapping(text, mapping) for text in supports]
        upper = Prog("SET", spec="A-Z", lo=2, hi=2)
        alnum = Prog("SET", spec="0-9A-Z", lo=2, hi=2)
        for node in (upper, alnum):
            left = charset_features(Observations(supports), node)
            right = charset_features(Observations(renamed), node)
            self.assertEqual(left, right)
        numbers = ["AA2024", "BC2027", "ZX2031"]
        renamed_numbers = [apply_mapping(text, mapping) for text in numbers]
        window = Prog("INT", spec="2020:2035", lo=4, hi=4)
        self.assertEqual(
            integer_features(Observations(numbers), window),
            integer_features(Observations(renamed_numbers), window),
        )

    def test_a_zero_scale_matches_the_evidence_score_and_a_hull_adds_nothing(self):
        supports = ["2024", "2027", "2031"]
        hull = Prog("INT", spec="2024:2031", lo=4, hi=4)
        wider = Prog("INT", spec="2020:2035", lo=4, hi=4)
        packed = pack_example([hull, wider], [False, True], [True, True], supports)
        self.assertEqual(float(packed["integer_hyp_expand"][0]), 0.0)
        self.assertEqual(float(packed["integer_hyp_expand"][1]), 1.0)
        self.assertTrue(bool(packed["ambiguous"].item()))
        model = ExpansionPrior()
        self.assertGreater(parameter_count(model), 1000)
        self.assertLess(parameter_count(model), 4000)
        self.assertAlmostEqual(float(model.scale().detach()), 1.0, places=5)
        feat = torch.randn(8, packed["charset_feat"].shape[-1]) * 5
        bounded = model.scores_from("charset", feat, torch.ones(8)).detach()
        self.assertGreaterEqual(float(bounded.min()), -1.0 - 1e-5)
        self.assertLessEqual(float(bounded.max()), 1.0 + 1e-5)
        hyp = packed["integer_hyp"]
        conservative = model.scores_from("integer", hyp, torch.zeros(hyp.shape[0]))
        self.assertEqual(tuple(conservative.shape), (hyp.shape[0],))
        self.assertTrue(torch.all(conservative == 0))
        batch = {key: value.unsqueeze(0) for key, value in packed.items()}
        evidence = torch.randn(1, 32, requires_grad=True)
        batch["evidence"] = evidence
        combined = model.combined_scores(batch, lam=0)
        self.assertTrue(torch.allclose(combined, evidence.detach()))
        evidence_model = freeze_evidence(LocalScorer())
        self.assertTrue(all(not param.requires_grad for param in evidence_model.parameters()))
        loss = model.loss(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNone(evidence.grad)

    def test_softplus_scale_starts_near_one(self):
        model = ExpansionPrior()
        self.assertAlmostEqual(float(model.scale().detach()), 1.0, places=5)
        self.assertAlmostEqual(math.log(math.e - 1.0), float(model.raw_lambda.detach()), places=5)


if __name__ == "__main__":
    unittest.main()
