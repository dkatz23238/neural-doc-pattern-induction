import unittest
from pathlib import Path

from pattern_induction.dsl_cover import NODE_CAP, REPEAT_CAP, cover_regex
from pattern_induction.program import OPEN
from pattern_induction.tasks import read_jsonl
from pattern_induction.version_space import MAX_NODES


class DslCoverTest(unittest.TestCase):
    def test_limits_match_the_executor(self):
        self.assertEqual(REPEAT_CAP, OPEN)
        self.assertEqual(NODE_CAP, MAX_NODES)

    def test_literals_and_full_classes_are_exact(self):
        for pattern in (
            r"^INV-\d{4}$",
            r"^USR-\d{3}-[A-Z]{2}$",
            r"^(?:YES|NO)$",
            r"^DE\d{2}(?:\d{4}){4}\d{2}$",
            r"^\d{12}$|^\d{15}$",
            r"^(?:100|\d{1,2})%$",
            r"^(?:WEB-|POS-)?\d{8}$",
            r"^fol\. \d{1,4}[rv]$",
            r"^\+\d{11,13}$",
        ):
            cover = cover_regex(pattern)
            self.assertEqual(cover.status, "exact", pattern)

    def test_unbounded_quantifiers_stay_open(self):
        for pattern in (
            r"^v\d+\.\d+\.\d+$",
            r"^USD \d{1,3}(?:,\d{3})*\.\d{2}$",
        ):
            cover = cover_regex(pattern)
            self.assertEqual(cover.status, "open", pattern)

    def test_partial_ranges_and_mixed_classes_are_outside(self):
        cases = {
            r"^INV-20(?:2[4-9]|3[0-5])-\d{5}$": "partial_range",
            r"^(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$": "partial_range",
            r"^[a-z0-9_]+\.(?:pdf|docx|xlsx|csv|txt|png)$": "mixed_class",
            r"^1Z[A-Z0-9]{16}$": "mixed_class",
            r"^[A-Z]{1,2}\d[A-Z\d]? \d[A-Z]{2}$": "mixed_class",
            r"^[\w\-]+_副本\.docx$": "word_class",
            r"^§\s?\d{1,2}(?:\.\d{1,2}){1,3}$": "space_class",
        }
        for pattern, reason in cases.items():
            cover = cover_regex(pattern)
            self.assertEqual(cover.status, "no", pattern)
            self.assertEqual(cover.reason, reason, pattern)

    def test_every_bench_regex_classifies(self):
        tasks = read_jsonl(Path("data/bench_c.jsonl"))
        seen = {task.gold_regex for task in tasks}
        self.assertGreater(len(seen), 50)
        for pattern in seen:
            cover = cover_regex(pattern)
            self.assertIn(cover.status, {"exact", "open", "no"})


if __name__ == "__main__":
    unittest.main()
