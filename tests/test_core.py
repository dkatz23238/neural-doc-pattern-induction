import re
import unittest

from pattern_induction.evaluate import score_regex
from pattern_induction.frontier import extract_regex, response_text
from pattern_induction.heuristic import infer_regex
from pattern_induction.tasks import Task


def _task(visible, positives, negatives, gold):
    return Task(
        id="t",
        benchmark="C",
        domain="invoice_number",
        variant="ex",
        visible=visible,
        hidden_positives=positives,
        hidden_negatives=negatives,
        gold_regex=gold,
        latent_schema=gold,
    )


class HeuristicTest(unittest.TestCase):
    def test_invoice_keeps_constant_year(self):
        strings = ["INV-2026-00192", "INV-2026-93821", "INV-2026-10293", "INV-2026-00047"]
        pattern = infer_regex(strings)
        self.assertEqual(pattern, r"^INV-2026-\d{5}$")

    def test_user_region(self):
        strings = ["USR-123-NY", "USR-829-CA", "USR-001-TX"]
        pattern = infer_regex(strings)
        self.assertEqual(pattern, r"^USR-\d{3}-[A-Z]{2}$")

    def test_sku_country(self):
        strings = ["X92_US", "K31_FR", "P08_AR"]
        pattern = infer_regex(strings)
        self.assertEqual(pattern, r"^[A-Z]\d{2}_[A-Z]{2}$")

    def test_iso_dates(self):
        strings = ["2026/09/21", "2025/04/03", "2024/12/18"]
        pattern = infer_regex(strings)
        self.assertEqual(pattern, r"^202\d/\d{2}/\d{2}$")

    def test_matches_visible(self):
        strings = ["PO-000193", "PO-002918", "PO-928123", "PO-100293"]
        pattern = infer_regex(strings)
        compiled = re.compile(pattern)
        self.assertTrue(all(compiled.fullmatch(s) for s in strings))
        self.assertIsNone(compiled.fullmatch("PO-12345"))
        self.assertIsNone(compiled.fullmatch("INV-000193"))


class EvalTest(unittest.TestCase):
    def test_semantic_accuracy_balances_recall_and_rejection(self):
        task = _task(
            ["INV-2026-00192", "INV-2026-00047"],
            ["INV-2026-12345", "INV-2026-99999"],
            ["INV-2026-1234", "ORD-2026-12345"],
            r"^INV-2026-\d{5}$",
        )
        good = score_regex(r"^INV-2026-\d{5}$", task)
        self.assertEqual(good.semantic_accuracy, 1.0)
        broad = score_regex(r"^.*$", task)
        self.assertEqual(broad.positive_recall, 1.0)
        self.assertEqual(broad.negative_rejection, 0.0)
        self.assertTrue(broad.catastrophic)
        memo = score_regex(r"^(?:INV-2026-00192|INV-2026-00047)$", task)
        self.assertEqual(memo.positive_recall, 0.0)
        self.assertEqual(memo.negative_rejection, 1.0)
        self.assertAlmostEqual(memo.semantic_accuracy, 0.5)


class FrontierParseTest(unittest.TestCase):
    def test_extract_regex_strips_fence(self):
        self.assertEqual(extract_regex("```regex\n^INV-\\d{4}$\n```"), r"^INV-\d{4}$")
        self.assertEqual(extract_regex("^INV-\\d{4}$"), r"^INV-\d{4}$")

    def test_response_text_reads_message_block(self):
        payload = {
            "output": [
                {"type": "reasoning"},
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "^A\\d+$"}],
                },
            ]
        }
        self.assertEqual(response_text(payload), r"^A\d+$")


if __name__ == "__main__":
    unittest.main()
