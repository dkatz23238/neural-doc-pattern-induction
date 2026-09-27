import re
import unittest

from pattern_induction.candidates import expanded_candidates, infer_regex, parse_pattern, pattern_cost
from pattern_induction.structural import structural_candidates


def _patterns(strings: list[str]) -> list[str]:
    return [item.pattern for item in structural_candidates(strings)]


def _assert_covers(test: unittest.TestCase, strings: list[str], expected: str) -> None:
    patterns = _patterns(strings)
    test.assertIn(expected, patterns)
    compiled = re.compile(expected)
    test.assertTrue(all(compiled.fullmatch(text) for text in strings))
    test.assertIsNotNone(parse_pattern(expected))


class StructuralGeneratorTest(unittest.TestCase):
    def test_every_candidate_parses_and_matches(self):
        samples = [
            ["ABC-123", "ABC123", "ABC-999"],
            ["§4.11.8.8", "§10.2.2", "§9.3.10.10", "§11.5"],
            ["§ 11.7.4", "§ 12.7", "§ 9.9.7.2", "§ 1.3.7"],
            ["1993-01-10", "2000-06-15", "2006-04-24", "N/A"],
            ["afemql.pdf", "pbsxtt (2).pdf", "ukygxu.pdf", "akoptv (1).pdf"],
            ["($2,418.60)", "($865.24)", "($207.70)", "($2,565.51)"],
            ["YES", "NO", "NO", "YES"],
            ["1.2", "11.7.4", "9.9.7.2"],
            ["tdch@dept.example.com", "vpmf@mail.example.com", "yiwy.kgw@dept.example.edu"],
            ["AB-123-XY", "CD-928-QZ"],
            ["INV-12345", "INV-83920", "INV-00014"],
        ]
        for strings in samples:
            for item in structural_candidates(strings):
                self.assertIsNotNone(parse_pattern(item.pattern), item.pattern)
                compiled = re.compile(item.pattern)
                self.assertTrue(
                    all(compiled.fullmatch(text) for text in strings),
                    item.pattern,
                )

    def test_repeated_dotted_sections(self):
        strings = ["§4.11.8.8", "§10.2.2", "§9.3.10.10", "§11.5"]
        expected = r"^§\d{1,2}(?:\.\d{1,2}){1,3}$"
        _assert_covers(self, strings, expected)
        heuristic = infer_regex(strings)
        self.assertLess(pattern_cost(expected, strings, True), pattern_cost(heuristic, strings, True))

    def test_optional_space_in_sections(self):
        strings = ["§ 11.7.4", "§ 12.7", "§ 9.9.7.2", "§ 1.3.7"]
        patterns = _patterns(strings)
        self.assertTrue(any(r"\d{1,2}(?:\.\d{1,2})" in pattern for pattern in patterns), patterns)

    def test_sentinel_alternation(self):
        strings = ["1993-01-10", "2000-06-15", "2006-04-24", "N/A"]
        expected = r"^(?:(?:\d{4}-\d{2}-\d{2})|(?:N/A))$"
        patterns = _patterns(strings)
        self.assertTrue(
            expected in patterns or r"^(?:(?:N/A)|(?:\d{4}-\d{2}-\d{2}))$" in patterns,
            patterns,
        )
        heuristic = infer_regex(strings)
        self.assertLess(pattern_cost(expected, strings, True), pattern_cost(heuristic, strings, True))

    def test_yes_no_is_an_alternation(self):
        strings = ["YES", "NO", "NO", "YES"]
        self.assertIn(r"^(?:NO|YES)$", _patterns(strings))

    def test_optional_filename_counter(self):
        strings = ["afemql.pdf", "pbsxtt (2).pdf", "ukygxu.pdf", "akoptv (1).pdf"]
        patterns = _patterns(strings)
        self.assertTrue(any(r"(?: \(\d+\))?" in pattern and pattern.endswith(r"\.pdf$") for pattern in patterns), patterns)

    def test_currency_grouping(self):
        strings = ["($2,418.60)", "($865.24)", "($207.70)", "($2,565.51)"]
        expected = r"^\(\$\d{1,3}(?:,\d{3})*\.\d{2}\)$"
        _assert_covers(self, strings, expected)
        # The visible amounts are all either 3 digits or 2,xxx, so the
        # overfit heuristic can be shorter than the grouping pattern.
        # Presence in the pool is the generator claim; selection is measured
        # on the hidden set.

    def test_dotted_numbers_repeat(self):
        strings = ["1.2", "11.7.4", "9.9.7.2"]
        expected = r"^\d{1,2}(?:\.\d{1,2}){1,3}$"
        _assert_covers(self, strings, expected)

    def test_prefix_decomposition(self):
        strings = ["INV-12345", "INV-83920", "INV-00014"]
        patterns = _patterns(strings)
        self.assertTrue(any(pattern.startswith(r"^INV-") and r"\d" in pattern for pattern in patterns))

    def test_distinct_codes_stay_a_class(self):
        strings = ["USR-203-DE", "USR-507-SO", "USR-627-AM", "USR-316-VV", "USR-110-QZ", "USR-884-BK"]
        patterns = _patterns(strings)
        self.assertFalse(any("DE" in pattern and "SO" in pattern for pattern in patterns), patterns)
        self.assertTrue(any(r"USR-\d{3}-[A-Z]{2}" in pattern or r"USR-\d{3}-[A-Z]{2}" in pattern for pattern in patterns), patterns)

    def test_run_segmentation(self):
        strings = ["AB-123-XY", "CD-928-QZ"]
        patterns = _patterns(strings)
        self.assertIn(r"^[A-Z]{2}-\d{3}-[A-Z]{2}$", patterns)

    def test_expanded_pool_keeps_the_heuristic_first(self):
        strings = ["§4.11.8.8", "§10.2.2", "§9.3.10.10", "§11.5"]
        shown, pool = expanded_candidates(strings, k=8, pool_limit=64)
        self.assertEqual(shown[0].pattern, infer_regex(strings))
        self.assertLessEqual(len(shown), 8)
        self.assertGreaterEqual(len(pool), len(shown))
        self.assertTrue(any("(?:\\." in item.pattern for item in pool))


if __name__ == "__main__":
    unittest.main()
