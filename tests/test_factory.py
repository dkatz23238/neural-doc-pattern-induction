import json
import tempfile
import unittest
from pathlib import Path

from pattern_induction.factory import (
    FactoryError,
    bind_program,
    compile_node,
    diversity,
    extract_concepts,
    ingest_text,
    materialize,
    validate_concept,
)
from pattern_induction.program import execute


PALLET = {
    "domain": "logistics",
    "concept": "warehouse pallet identifier",
    "description": "Facility, hyphen, six digits, optional return suffix.",
    "semantic_type": "identifier",
    "positives": ["NY-193842", "TX-002931-R", "CA-827341"],
    "near_misses": ["NY193842", "TX-2931", "CA-827341-X"],
    "ambiguity_types": ["optional"],
    "program": {
        "op": "seq",
        "children": [
            {"op": "cls", "class": "upper", "lo": 2, "hi": 2},
            {"op": "lit", "text": "-"},
            {"op": "int", "min": 0, "max": 999999, "lo": 6, "hi": 6},
            {"op": "opt", "child": {"op": "lit", "text": "-R"}},
        ],
    },
    "neighbors": [
        {
            "op": "seq",
            "children": [
                {"op": "cls", "class": "upper", "lo": 2, "hi": 2},
                {"op": "lit", "text": "-"},
                {"op": "int", "min": 0, "max": 999999, "lo": 6, "hi": 6},
            ],
        }
    ],
}

YEAR = {
    "domain": "documents",
    "concept": "filing year",
    "description": "A year in the conventional window, not the observed hull.",
    "semantic_type": "year",
    "positives": ["2024", "2027", "2031", "2026"],
    "near_misses": ["2019", "2036", "24"],
    "ambiguous_supports": ["2024", "2027", "2031", "2026"],
    "ambiguity_types": ["integer_range"],
    "program": {"op": "int", "min": 2020, "max": 2035, "lo": 4, "hi": 4},
    "neighbors": [{"op": "int", "min": 2024, "max": 2031, "lo": 4, "hi": 4}],
}

CURRENCY = {
    "domain": "finance",
    "concept": "invoice currency",
    "description": "ISO 4217 alphabetic code, not any three letters.",
    "semantic_type": "currency",
    "positives": ["USD", "EUR", "JPY", "CAD"],
    "near_misses": ["ABC", "USA", "US"],
    "ambiguous_supports": ["USD", "EUR", "JPY", "CAD"],
    "ambiguity_types": ["lexicon"],
    "program": {"op": "lex", "name": "iso4217"},
    "neighbors": [{"op": "cls", "class": "upper", "lo": 3, "hi": 3}],
}


class FactoryTest(unittest.TestCase):
    def test_pallet_positives_match_and_near_misses_do_not(self):
        stored, reason = validate_concept(PALLET)
        self.assertEqual(reason, "")
        self.assertIsNotNone(stored)
        self.assertIn("opt", stored["operators"])
        self.assertEqual(stored["domain"], "logistics")
        self.assertTrue(stored["neighbors"])

    def test_a_year_window_keeps_a_tighter_neighbor_that_also_fits_the_sample(self):
        stored, reason = validate_concept(YEAR)
        self.assertEqual(reason, "")
        self.assertEqual(stored["ambiguity_status"], "witnessed")
        self.assertEqual(stored["neighbors"][0]["min"], 2024)
        self.assertEqual(stored["neighbors"][0]["max"], 2031)

    def test_a_currency_lexicon_rejects_three_letters_outside_the_list(self):
        stored, reason = validate_concept(CURRENCY)
        self.assertEqual(reason, "")
        self.assertEqual(stored["lexicons"], ["iso4217"])
        self.assertEqual(stored["ambiguity_status"], "witnessed")

    def test_a_program_that_accepts_a_near_miss_is_rejected(self):
        broken = dict(PALLET)
        broken["near_misses"] = ["NY-193842"]
        broken["positives"] = ["TX-002931-R", "CA-827341"]
        stored, reason = validate_concept(broken)
        self.assertIsNone(stored)
        self.assertEqual(reason, "near_miss_accepted")

    def test_an_unknown_lexicon_is_rejected(self):
        raw = {
            "domain": "logistics",
            "concept": "carrier code",
            "positives": ["MAEU", "MSCU"],
            "near_misses": ["maeu"],
            "program": {"op": "lex", "name": "scac"},
        }
        stored, reason = validate_concept(raw)
        self.assertIsNone(stored)
        self.assertEqual(reason, "unknown_lexicon")

    def test_repeat_above_the_executor_cap_is_rejected(self):
        with self.assertRaises(FactoryError) as caught:
            compile_node({"op": "repeat", "lo": 1, "hi": 9, "child": {"op": "cls", "class": "digit", "lo": 1, "hi": 1}})
        self.assertEqual(caught.exception.reason, "repeat_cap")

    def test_a_response_without_json_is_kept_and_not_admitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = ingest_text(
                root,
                "I think the field looks like a code, but here is no object.",
                prompt_name="real_world_concepts_v1.md",
                prompt_text="prompt",
                model="gpt-5.6-terra",
            )
            self.assertEqual(manifest["accepted"], 0)
            self.assertIn("no_json", manifest["reject_reasons"])
            raw = json.loads((root / "data" / "factory" / "runs").joinpath(manifest["run"], "response.json").read_text())
            self.assertIn("no object", raw["raw"])
            rejected = (root / "data" / "factory" / "rejected" / "concepts.jsonl").read_text()
            self.assertIn("no_json", rejected)
            self.assertFalse(list((root / "data" / "factory" / "templates").glob("*.jsonl")))

    def test_ingest_accepts_a_concept_once(self):
        payload = json.dumps({"concepts": [PALLET, YEAR]})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = ingest_text(root, payload, prompt_name="p.md", prompt_text="p", model="gpt-5.6-terra")
            second = ingest_text(root, payload, prompt_name="p.md", prompt_text="p", model="gpt-5.6-terra")
            self.assertEqual(first["accepted"], 2)
            self.assertEqual(second["accepted"], 0)
            self.assertIn("duplicate", second["reject_reasons"])
            report = diversity(
                [
                    json.loads(line)
                    for path in (root / "data" / "factory" / "templates").glob("*.jsonl")
                    for line in path.read_text().splitlines()
                ]
            )
            self.assertEqual(report["templates"], 2)
            self.assertEqual(report["domains"], 2)
            self.assertEqual(report["unique_programs"], 2)
            self.assertGreater(report["mean_positives"], 0)
            self.assertIn("ast_signature_entropy", report)
            self.assertIn("semantic_entropy", report)

    def test_materialize_builds_supports_the_gold_program_accepts(self):
        stored, _ = validate_concept(PALLET)
        tasks = materialize(stored, seed=2507, limit=6)
        self.assertTrue(tasks)
        for task in tasks:
            program = bind_program(stored["program"], task["supports"])
            for text in task["supports"]:
                self.assertTrue(execute(program, text, task["supports"]), task["configuration"])

    def test_compact_dialect_keeps_a_swapped_integer_and_a_lexicon_alias(self):
        appointment = {
            "domain": "logistics",
            "concept": "appointment year",
            "description": "The observed years sit inside a wider window.",
            "semantic_type": "year",
            "positives": ["APT-2024-01", "APT-2027-14", "APT-2031-52"],
            "near_misses": ["APT-2019-01", "APT-2036-01", "APT-202A-01"],
            "ambiguous_supports": ["APT-2024-01", "APT-2027-14", "APT-2031-52"],
            "program": {
                "seq": [
                    {"lit": "APT-"},
                    {"int": {"min": 4, "max": 4, "lo": 2020, "hi": 2035}},
                    {"lit": "-"},
                    {"int": {"min": 2, "max": 2, "lo": 1, "hi": 52}},
                ]
            },
            "neighbors": [
                {
                    "seq": [
                        {"lit": "APT-"},
                        {"int": {"min": 4, "max": 4, "lo": 2024, "hi": 2031}},
                        {"lit": "-"},
                        {"int": {"min": 2, "max": 2, "lo": 1, "hi": 52}},
                    ]
                }
            ],
        }
        stored, reason = validate_concept(appointment)
        self.assertEqual(reason, "")
        self.assertEqual(stored["program"]["children"][1]["min"], 2020)
        self.assertEqual(stored["program"]["children"][1]["lo"], 4)
        self.assertEqual(stored["ambiguity_status"], "witnessed")
        currency = {
            "domain": "logistics",
            "concept": "invoice currency code",
            "description": "ISO currency, not any three letters.",
            "semantic_type": "currency",
            "positives": ["CUR-USD", "CUR-EUR", "CUR-JPY"],
            "near_misses": ["CUR-US1", "CUR-TOOL", "CUR-XYZ"],
            "ambiguous_supports": ["CUR-USD", "CUR-EUR", "CUR-JPY"],
            "program": {"seq": [{"lit": "CUR-"}, {"lex": "iso4217_currency_codes"}]},
            "neighbors": [
                {"seq": [{"lit": "CUR-"}, {"cls": "upper"}, {"cls": "upper"}, {"cls": "upper"}]},
                {"seq": [{"lit": "CUR-"}, {"set": ["USD", "EUR", "JPY"]}]},
            ],
        }
        stored, reason = validate_concept(currency)
        self.assertEqual(reason, "")
        self.assertEqual(stored["lexicons"], ["iso4217"])
        self.assertEqual(stored["ambiguity_status"], "witnessed")
        self.assertEqual(len(stored["neighbors"]), 1)
        bay = {
            "domain": "logistics",
            "concept": "bay letter",
            "description": "Any uppercase letter, so a later letter is still in the language.",
            "semantic_type": "identifier",
            "positives": ["BAY-A1", "BAY-C2", "BAY-F3"],
            "near_misses": ["BAY-a1", "BAY-G", "BAY-AA1"],
            "ambiguous_supports": ["BAY-A1", "BAY-C2", "BAY-F3"],
            "program": {"seq": [{"lit": "BAY-"}, {"cls": "upper"}, {"cls": "digit"}]},
            "neighbors": [
                {"seq": [{"lit": "BAY-"}, {"set": ["A", "B", "C", "D", "E", "F"]}, {"cls": "digit"}]}
            ],
        }
        stored, reason = validate_concept(bay)
        self.assertEqual(reason, "")
        self.assertEqual(stored["program"]["children"][1]["class"], "upper")
        self.assertEqual(stored["neighbors"][0]["children"][1]["spec"], "A-F")
        self.assertEqual(stored["ambiguity_status"], "witnessed")
        program = bind_program(stored["program"], stored["positives"])
        self.assertTrue(execute(program, "BAY-Z9", stored["positives"]))
        self.assertFalse(execute(program, "BAY-G", stored["positives"]))

    def test_class_spec_and_width_field_compile(self):
        tracking = {
            "domain": "logistics",
            "concept": "uppercase tracking code",
            "description": "Four letters after a prefix.",
            "semantic_type": "identifier",
            "positives": ["TRK-ABCD", "TRK-WXYZ", "TRK-LOGI"],
            "near_misses": ["TRK-ABC", "TRK-ABCDE", "TRK-A1B2", "TRK-abcd"],
            "program": {"seq": [{"lit": "TRK-"}, {"repeat": {"child": {"cls": "A-Z"}, "hi": 4, "lo": 4}}]},
        }
        stored, reason = validate_concept(tracking)
        self.assertEqual(reason, "")
        self.assertEqual(stored["program"]["children"][1]["child"]["op"], "set")
        self.assertEqual(stored["program"]["children"][1]["child"]["spec"], "A-Z")
        bay = {
            "domain": "logistics",
            "concept": "loading bay number",
            "description": "Bays 12 through 18, two digits.",
            "semantic_type": "identifier",
            "positives": ["BAY-12", "BAY-15", "BAY-18"],
            "near_misses": ["BAY-11", "BAY-19", "BAY-09", "BAY-120"],
            "program": {"seq": [{"lit": "BAY-"}, {"int": {"hi": 18, "lo": 12, "width": 2}}]},
        }
        stored, reason = validate_concept(bay)
        self.assertEqual(reason, "")
        self.assertEqual(stored["program"]["children"][1]["min"], 12)
        self.assertEqual(stored["program"]["children"][1]["max"], 18)
        self.assertEqual(stored["program"]["children"][1]["lo"], 2)

    def test_type_key_is_an_op(self):
        citation = {
            "domain": "legal",
            "concept": "westlaw citation",
            "description": "Year, WL, serial.",
            "semantic_type": "citation",
            "positives": ["2024 WL 1234567", "2023 WL 58291", "2019 WL 87654321"],
            "near_misses": ["2024WL1234567", "2024 W.L. 1234567", "24 WL 1234567"],
            "program": {
                "type": "seq",
                "children": [
                    {"type": "int", "hi": 4, "lo": 4, "max": 2099, "min": 1800},
                    {"type": "lit", "text": " WL ", "short": True},
                    {"type": "int", "hi": 8, "lo": 1, "max": 99999999, "min": 1},
                ],
            },
        }
        stored, reason = validate_concept(citation)
        self.assertEqual(reason, "", stored)
        self.assertEqual(stored["program"]["op"], "seq")
        self.assertEqual(stored["parser_version"], "factory-3")

    def test_extract_reads_a_fenced_concept_list(self):
        text = "```json\n" + json.dumps({"concepts": [CURRENCY]}) + "\n```"
        rows = extract_concepts(text)
        self.assertEqual(rows[0]["concept"], "invoice currency")


if __name__ == "__main__":
    unittest.main()
