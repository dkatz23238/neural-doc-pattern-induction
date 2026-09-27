import random
import re
import unittest

import torch

from pattern_induction.schema import (
    KIND_CLS,
    KIND_LIT,
    KIND_REP,
    STOI,
    Part,
    compile_parts,
    encode_text,
    from_regex,
    properties,
    sample_task,
)
from pattern_induction.schema_net import SchemaNet, schema_loss


class CompileTest(unittest.TestCase):
    def test_section_repeat_matches_its_samples(self):
        parts = [
            Part(KIND_LIT, chars=encode_text("§", 16), lo=1, hi=1),
            Part(KIND_LIT, optional=1, chars=encode_text(" ", 16), lo=1, hi=1),
            Part(KIND_CLS, charset=0, lo=1, hi=2),
            Part(KIND_REP, charset=0, lo=1, hi=2, rep_lo=1, rep_hi=3, delim=STOI["."]),
        ]
        pattern = compile_parts(parts)
        self.assertEqual(pattern, r"^§(?: )?\d{1,2}(?:\.\d{1,2}){1,3}$")
        compiled = re.compile(pattern)
        self.assertIsNotNone(compiled.fullmatch("§11.7.4"))
        self.assertIsNotNone(compiled.fullmatch("§ 12.7"))
        self.assertIsNotNone(compiled.fullmatch("§4.9.10.1"))
        self.assertIsNone(compiled.fullmatch("§4"))

    def test_samples_match_the_compiled_regex(self):
        rng = random.Random(1)
        for _ in range(25):
            _parts, strings, pattern = sample_task(rng, n_visible=6)
            compiled = re.compile(pattern)
            for text in strings:
                self.assertIsNotNone(compiled.fullmatch(text), pattern)

    def test_literal_alternation_round_trips(self):
        parts = from_regex(r"^(?:YES|NO)$")
        self.assertIsNotNone(parts)
        self.assertEqual(compile_parts(parts), r"^(?:YES|NO)$")

    def test_prefix_class_round_trips(self):
        parts = from_regex(r"^INV-\d{4}$")
        self.assertIsNotNone(parts)
        compiled = re.compile(compile_parts(parts))
        self.assertIsNotNone(compiled.fullmatch("INV-2044"))
        self.assertIsNone(compiled.fullmatch("INV-204"))

    def test_properties_read_a_repeated_section(self):
        facts = properties(r"^\d{1,2}(?:\.\d{1,2}){1,3}$")
        self.assertEqual(facts["repeat"], "1")
        self.assertEqual(facts["digit"], "1")
        self.assertEqual(facts["delim"], ".")
        self.assertEqual(facts["alternation"], "0")


class NetTest(unittest.TestCase):
    def test_one_step_reduces_loss(self):
        torch.manual_seed(0)
        model = SchemaNet(d=32, heads=4, layers=1)
        tokens = torch.randint(2, 20, (4, 3, 64))
        token_pad = torch.zeros(4, 3, 64, dtype=torch.bool)
        token_pad[:, :, 12:] = True
        slot_pad = torch.zeros(4, 3, dtype=torch.bool)
        target = {
            "kind": torch.zeros(4, 8, dtype=torch.long),
            "optional": torch.zeros(4, 8, dtype=torch.long),
            "charset": torch.zeros(4, 8, dtype=torch.long),
            "lo": torch.ones(4, 8, dtype=torch.long),
            "hi": torch.ones(4, 8, dtype=torch.long),
            "rep_lo": torch.zeros(4, 8, dtype=torch.long),
            "rep_hi": torch.zeros(4, 8, dtype=torch.long),
            "delim": torch.zeros(4, 8, dtype=torch.long),
            "chars": torch.zeros(4, 8, 16, dtype=torch.long),
        }
        target["kind"][:, 0] = 1
        target["chars"][:, 0, 0] = 5
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        model.train()
        logits = model(tokens, token_pad, slot_pad, target["chars"])
        first = schema_loss(logits, target)
        first.backward()
        optimizer.step()
        logits = model(tokens, token_pad, slot_pad, target["chars"])
        second = schema_loss(logits, target)
        self.assertLess(float(second.detach()), float(first.detach()))


if __name__ == "__main__":
    unittest.main()
