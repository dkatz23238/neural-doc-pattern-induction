import random
import unittest

import torch

from pattern_induction.program import Prog
from pattern_induction.renaming import apply_mapping, build_mapping
from prior.explicit import candidate_pieces, match_score, score_programs
from prior.probe_labels import family_flags, probe_targets, visible_facts
from prior.probes import make_probe, train_probe


def _seq(*children: Prog) -> Prog:
    return Prog("SEQ", children=list(children))


class ProbeLabelTest(unittest.TestCase):
    def test_held_out_families_carry_the_composed_facts(self):
        repeat = family_flags("repeat_interval")
        self.assertEqual((repeat["repeat"], repeat["interval"], repeat["optional"], repeat["lexicon"]), (1, 1, 0, 0))
        self.assertEqual(repeat["affix"], 0)
        prefix = family_flags("prefix_charset_optional")
        self.assertEqual((prefix["optional"], prefix["optional_scope"], prefix["repeat"], prefix["interval"]), (1, 1, 0, 0))
        self.assertEqual(prefix["affix"], 1)
        suffix = family_flags("lexicon_suffix")
        self.assertEqual((suffix["lexicon"], suffix["affix"], suffix["optional"], suffix["repeat"]), (1, 2, 0, 0))
        trained = family_flags("prefix_optional")
        self.assertEqual(trained["optional_scope"], prefix["optional_scope"])
        self.assertNotEqual(family_flags("repeat_width")["interval"], 1)

    def test_visible_facts_separate_repeat_optional_and_affix(self):
        repeated = visible_facts(["12-34-56", "78-90", "11-22-33-44", "55-66-77"])
        self.assertTrue(repeated["repeat"])
        self.assertEqual(repeated["optional_scope"], "absent")
        self.assertEqual(repeated["charset"], "digit")
        self.assertEqual(repeated["width"], "count")
        self.assertEqual(repeated["affix"], "none")

        optional = visible_facts(["AB-1234", "5678", "AB-9012", "3456"])
        self.assertFalse(optional["repeat"])
        self.assertEqual(optional["optional_scope"], "prefix")
        self.assertEqual(optional["charset"], "digit")
        self.assertEqual(optional["affix"], "prefix")
        self.assertEqual(optional["width"], "count")

        mixed = visible_facts(["C-ABCD", "WXYZ", "C-EFGH", "IJKL"])
        self.assertFalse(mixed["repeat"])
        self.assertEqual(mixed["optional_scope"], "prefix")
        self.assertEqual(mixed["charset"], "upper")

        suffix = visible_facts(["USD-zz", "EUR-zz", "JPY-zz", "GBP-zz"])
        self.assertEqual(suffix["affix"], "suffix")
        self.assertEqual(suffix["charset"], "upper")
        self.assertTrue(suffix["lexicon"])

        opened = visible_facts(["1", "22", "333", "4444"])
        self.assertEqual(opened["width"], "open")
        self.assertEqual(opened["charset"], "digit")

    def test_a_repeated_interval_is_not_read_as_an_optional_prefix(self):
        facts = visible_facts(["147-144-147", "145-147-145-146", "144-148-143", "141-143-144-142"])
        self.assertTrue(facts["repeat"])
        self.assertEqual(facts["optional_scope"], "absent")
        self.assertEqual(facts["charset"], "digit")

    def test_letter_rename_keeps_the_structural_facts(self):
        supports = ["AB-1234", "5678", "AB-9012", "3456"]
        mapping = build_mapping(supports, random.Random(3), False)
        renamed = [apply_mapping(text, mapping) for text in supports]
        original = visible_facts(supports)
        moved = visible_facts(renamed)
        for key in ("repeat", "optional_scope", "charset", "width", "affix"):
            self.assertEqual(original[key], moved[key], key)

    def test_explicit_match_prefers_the_piece_the_strings_show(self):
        supports = ["12-34-56", "78-90", "11-22-33", "44-55-66-77"]
        facts = visible_facts(supports)
        field = Prog("CLS", cls="digit", lo=2, hi=2)
        copy = Prog("COPY", slot=0, start=2, end=3)
        repeated = _seq(field, Prog("REPEAT", lo=1, hi=3, children=[_seq(copy, field)]))
        bare = Prog("CLS", cls="digit", lo=2, hi=8)
        self.assertGreater(match_score(facts, candidate_pieces(repeated), supports), match_score(facts, candidate_pieces(bare), supports))

        optional_supports = ["AB-1234", "5678", "AB-9012", "3456"]
        optional_facts = visible_facts(optional_supports)
        prefix = _seq(
            Prog("OPT", children=[_seq(Prog("COPY", slot=0, start=0, end=2), Prog("COPY", slot=0, start=2, end=3))]),
            Prog("CLS", cls="digit", lo=4, hi=4),
        )
        plain = Prog("CLS", cls="digit", lo=4, hi=4)
        scores = score_programs([prefix, plain], optional_supports, optional_facts)
        self.assertGreater(scores[0], scores[1])

        upper_supports = ["C-ABCD", "WXYZ", "C-EFGH", "IJKL"]
        upper_facts = visible_facts(upper_supports)
        cls = _seq(
            Prog("OPT", children=[_seq(Prog("COPY", slot=0, start=0, end=1), Prog("COPY", slot=0, start=1, end=2))]),
            Prog("CLS", cls="upper", lo=4, hi=4),
        )
        completed = _seq(
            Prog("OPT", children=[_seq(Prog("COPY", slot=0, start=0, end=1), Prog("COPY", slot=0, start=1, end=2))]),
            Prog("SET", spec="0-9A-Z", lo=4, hi=4),
        )
        upper_scores = score_programs([cls, completed], upper_supports, upper_facts)
        self.assertGreater(upper_scores[0], upper_scores[1])

    def test_targets_follow_the_family_for_repeat_and_the_strings_for_charset(self):
        supports = ["12-34", "56-78-90", "11-22", "33-44-55"]
        target = probe_targets("repeat_interval", supports)
        self.assertEqual(target["repeat"], 1)
        self.assertEqual(target["interval"], 1)
        self.assertEqual(target["optional"], 0)
        self.assertEqual(target["charset"], 0)

    def test_a_linear_probe_reads_a_bit_that_is_actually_in_the_vector(self):
        generator = torch.Generator().manual_seed(0)
        features = torch.randn(400, 160, generator=generator)
        labels = (features[:, 0] > 0).long()
        model, meta = train_probe("linear", features, labels, features, labels, 2, seed=0, epochs=15, batch_size=64)
        pred = model(features).argmax(dim=-1)
        self.assertGreater(float((pred == labels).float().mean()), 0.95)
        self.assertGreater(meta["best"]["balanced"], 0.95)
        self.assertEqual(make_probe("tiny", 2).hidden, 32)


if __name__ == "__main__":
    unittest.main()
