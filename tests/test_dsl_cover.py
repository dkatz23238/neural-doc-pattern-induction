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

    def test_partial_ranges_and_mixed_classes_are_in_fragment(self):
        exact = (
            r"^INV-20(?:2[4-9]|3[0-5])-\d{5}$",
            r"^(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$",
            r"^1Z[A-Z0-9]{16}$",
            r"^[A-Z]{1,2}\d[A-Z\d]? \d[A-Z]{2}$",
        )
        for pattern in exact:
            cover = cover_regex(pattern)
            self.assertEqual(cover.status, "exact", pattern)
        opened = cover_regex(r"^[a-z0-9_]+\.(?:pdf|docx|xlsx|csv|txt|png)$")
        self.assertEqual(opened.status, "open")
        self.assertEqual(opened.reason, "unbounded")

    def test_word_and_space_classes_stay_outside(self):
        cases = {
            r"^[\w\-]+_副本\.docx$": "word_class",
            r"^§\s?\d{1,2}(?:\.\d{1,2}){1,3}$": "space_class",
        }
        for pattern, reason in cases.items():
            cover = cover_regex(pattern)
            self.assertEqual(cover.status, "no", pattern)
            self.assertEqual(cover.reason, reason, pattern)

    def test_bench_c_exact_patterns_stay_exact_and_coverage_clears_90(self):
        import json

        tasks = read_jsonl(Path("data/bench_c.jsonl"))
        previous = json.loads(Path("results/needle/bench_split.json").read_text())
        by_id = {row["id"]: row for row in previous["rows"]}
        exact = open_ = outside = 0
        for task in tasks:
            cover = cover_regex(task.gold_regex)
            if by_id[task.id]["dsl"] == "exact":
                self.assertEqual(cover.status, "exact", task.gold_regex)
            if cover.status == "exact":
                exact += 1
            elif cover.status == "open":
                open_ += 1
            else:
                outside += 1
        self.assertGreater((exact + open_) / len(tasks), 0.90)
        self.assertLess(outside, 40)

    def test_verbose_integer_and_literal_alternations_are_one_node(self):
        octet = cover_regex(r"^(?:25[0-5]|2[0-4]\d|1?\d{1,2})(?:\.(?:25[0-5]|2[0-4]\d|1?\d{1,2})){3}$")
        self.assertEqual(octet.status, "exact", octet)
        self.assertLessEqual(octet.nodes, NODE_CAP)
        months = cover_regex(
            r"^\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (?:19|20)\d{2}$"
        )
        self.assertNotEqual(months.status, "no", months)
        self.assertLessEqual(months.nodes, NODE_CAP)

    def test_every_bench_regex_classifies(self):
        tasks = read_jsonl(Path("data/bench_c.jsonl"))
        seen = {task.gold_regex for task in tasks}
        self.assertGreater(len(seen), 50)
        for pattern in seen:
            cover = cover_regex(pattern)
            self.assertIn(cover.status, {"exact", "open", "no"})


if __name__ == "__main__":
    unittest.main()
