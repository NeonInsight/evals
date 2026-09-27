from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import conditional_logit
import phase0
import study


class FrozenStudyTests(unittest.TestCase):
    def test_canonical_artifact_and_results(self):
        study.verify_canonical_copy()
        self.assertEqual(study.file_hash(study.CANONICAL_LOCAL_COPY), study.CANONICAL_SOURCE_SHA256)

    def test_parser(self):
        self.assertEqual(study.parse_choice("A"), "A")
        self.assertEqual(study.parse_choice("answer: b"), "B")
        self.assertIsNone(study.parse_choice("A or B"))
        self.assertIsNone(study.parse_choice("none"))

    def test_phase0_hashes_and_counts(self):
        phase0.verify()
        manifest = json.loads((HERE / "frozen_manifest.json").read_text())
        fixtures = study.read_jsonl(HERE / "fixtures.jsonl")
        rows = study.read_jsonl(HERE / "rendered_prompts.jsonl")
        self.assertEqual(len(fixtures), manifest["scientific_n"])
        self.assertEqual(len(rows), 5 * len(fixtures))
        self.assertEqual({row["condition"] for row in rows}, set(study.CONDITIONS))

    def test_fixture_decisions_and_complexity(self):
        fixtures = study.read_jsonl(HERE / "fixtures.jsonl")
        rows = study.read_jsonl(HERE / "rendered_prompts.jsonl")
        for fixture in fixtures:
            study.validate_fixture_invariants(fixture)
            group = [row for row in rows if row["fixture_id"] == fixture["fixture_id"]]
            self.assertEqual(len(group), 5)
            self.assertEqual({row["expected"] for row in group}, {fixture["correct_code"]})
            low = next(row for row in group if row["condition"] == "NEUTRAL_LOW")
            high = next(row for row in group if row["condition"] == "NEUTRAL_HIGH")
            for metric in study.STRUCTURAL_METRICS:
                self.assertGreater(high[metric], low[metric])

    def test_control_matching_and_attempt_limit(self):
        fixtures = study.read_jsonl(HERE / "fixtures.jsonl")
        for fixture in fixtures:
            match = fixture["control_matching"]
            self.assertLessEqual(match["attempts_used"], 2)
            if match["valid"]:
                self.assertLessEqual(
                    abs(match["control_delta_tokens"] - match["relational_delta_tokens"]),
                    match["token_tolerance"],
                )
                self.assertLessEqual(
                    abs(match["control_delta_characters"] - match["relational_delta_characters"]),
                    match["character_tolerance"],
                )

    def test_conditional_interaction_sign(self):
        # A fixed synthetic nonseparated sample has a positive Neutral×HIGH effect.
        rng = np.random.default_rng(0)
        probabilities = np.asarray([0.50, 0.50, 0.40, 0.65])
        pattern = (rng.random((300, 4)) < probabilities).astype(int)
        result = conditional_logit.fit_conditional_logit(pattern)
        self.assertEqual(result["status"], "FIT_CONVERGED")
        self.assertGreater(result["terms"]["neutral_by_high"]["log_odds"], 0)


if __name__ == "__main__":
    unittest.main()
