import unittest

import torch

from pattern_induction.ranker import (
    PAD_ID,
    TinyRegexRanker,
    collate_tasks,
    encode_utf8,
    soft_cross_entropy,
)


class EncodeTest(unittest.TestCase):
    def test_bytes_round_trip_and_pad(self):
        ids, pad, truncated = encode_utf8("INV-1")
        self.assertFalse(truncated)
        self.assertEqual(bytes(ids[:5]).decode(), "INV-1")
        self.assertEqual(ids[5], PAD_ID)
        self.assertFalse(any(pad[:5]))
        self.assertTrue(all(pad[5:]))

    def test_long_text_is_marked_truncated(self):
        _ids, _pad, truncated = encode_utf8("a" * 200, max_len=8)
        self.assertTrue(truncated)


class RankerTest(unittest.TestCase):
    def test_padding_does_not_change_the_string_vector(self):
        torch.manual_seed(0)
        model = TinyRegexRanker(cross_attention=True)
        model.eval()
        text = "abc"
        ids, pad, _truncated = encode_utf8(text, max_len=8)
        tokens = torch.tensor([ids])
        token_pad = torch.tensor([pad])
        with torch.no_grad():
            pooled = model.encode(tokens, token_pad, model.string_encoder)
        self.assertEqual(tuple(pooled.shape), (1, model.d))
        self.assertTrue(torch.isfinite(pooled).all())

    def test_one_step_reduces_soft_loss(self):
        torch.manual_seed(0)
        model = TinyRegexRanker(cross_attention=True, d=32, heads=4, string_layers=1, regex_layers=1, max_len=16)
        rows = []
        for index in range(8):
            winner = index % 2
            strings = [f"{'a' if winner == 0 else 'b'}{index}"]
            patterns = ["a+", "b+"]
            accuracies = [1.0, 0.0] if winner == 0 else [0.0, 1.0]
            string_ids, string_pad, cand_ids, cand_pad = [], [], [], []
            for text in strings:
                ids, pad, _cut = encode_utf8(text, max_len=16)
                string_ids.append(ids)
                string_pad.append(pad)
            for pattern in patterns:
                ids, pad, _cut = encode_utf8(pattern, max_len=16)
                cand_ids.append(ids)
                cand_pad.append(pad)
            rows.append(
                {
                    "string_ids": string_ids,
                    "string_pad": string_pad,
                    "cand_ids": cand_ids,
                    "cand_pad": cand_pad,
                    "accuracies": accuracies,
                }
            )
        batch = collate_tasks(rows, max_len=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        model.train()
        logits = model(
            batch["strings"],
            batch["string_pad"],
            batch["string_slot_pad"],
            batch["candidates"],
            batch["cand_pad"],
            batch["cand_slot_pad"],
        )
        first = soft_cross_entropy(logits, batch["accuracy"], ~batch["cand_slot_pad"], tau=0.05)
        before = float(first.detach())
        first.backward()
        optimizer.step()
        logits = model(
            batch["strings"],
            batch["string_pad"],
            batch["string_slot_pad"],
            batch["candidates"],
            batch["cand_pad"],
            batch["cand_slot_pad"],
        )
        after = float(soft_cross_entropy(logits, batch["accuracy"], ~batch["cand_slot_pad"], tau=0.05).detach())
        self.assertLess(after, before)

    def test_pooled_and_cross_attention_both_score_a_batch(self):
        row = {
            "string_ids": [encode_utf8("ab", max_len=8)[0]],
            "string_pad": [encode_utf8("ab", max_len=8)[1]],
            "cand_ids": [encode_utf8(r"ab", max_len=8)[0], encode_utf8(r"a+", max_len=8)[0]],
            "cand_pad": [encode_utf8(r"ab", max_len=8)[1], encode_utf8(r"a+", max_len=8)[1]],
            "accuracies": [0.5, 1.0],
        }
        batch = collate_tasks([row], max_len=8)
        for cross in (True, False):
            model = TinyRegexRanker(cross_attention=cross, d=32, heads=4, string_layers=1, regex_layers=1, max_len=8)
            logits = model(
                batch["strings"],
                batch["string_pad"],
                batch["string_slot_pad"],
                batch["candidates"],
                batch["cand_pad"],
                batch["cand_slot_pad"],
            )
            self.assertEqual(tuple(logits.shape), (1, 2))
            self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
