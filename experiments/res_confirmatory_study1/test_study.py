"""Pre-outcome checks for the intended causal comparison and scoring parity."""
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol as p


class StudyDesignTests(unittest.TestCase):
    def test_four_matched_views_preserve_decision(self):
        fixtures = {f["fixture_id"]: f for f in p.fixtures()}
        rows = p.rendered()
        self.assertEqual((len(fixtures), len(rows)), (64, 256))
        for fid, fixture in fixtures.items():
            group = [r for r in rows if r["fixture_id"] == fid]
            self.assertEqual({r["condition"] for r in group}, set(p.CONDITIONS))
            self.assertEqual({r["expected"] for r in group}, {fixture["correct_code"]})
            self.assertEqual(len({r["messages"][0]["content"] for r in group}), 1)
            for row in group:
                self.assertEqual(row["messages_sha256"], p.digest(row["messages"]))
                self.assertEqual(row["messages"][1]["content"].count("SUPPORTS"),
                                 sum(fixture["executor_bits"]) + sum(fixture["peer_bits"]))
                self.assertEqual(row["messages"][1]["content"].count("BLOCKS"),
                                 10 - sum(fixture["executor_bits"]) - sum(fixture["peer_bits"]))
            self.assertEqual(group[0]["structural_features"]["decision_signals"], 5)

    def test_balanced_outcomes_and_positions(self):
        rows = p.rendered()
        for condition in p.CONDITIONS:
            selected = [r for r in rows if r["condition"] == condition]
            self.assertEqual(Counter(r["expected"] for r in selected), {"A": 32, "B": 32})
            self.assertEqual(Counter(r["position_within_fixture"] for r in selected),
                             {0: 16, 1: 16, 2: 16, 3: 16})
        self.assertEqual(p.rendered(), p.rendered())

    def test_frozen_parser(self):
        self.assertEqual([p.parse_choice(s) for s in ("A", "b", "A.", "A or B", "AB", "unknown")],
                         ["A", "B", "A", None, None, None])


if __name__ == "__main__":
    unittest.main()
