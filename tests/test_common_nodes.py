from __future__ import annotations

import runpy
from pathlib import Path
import unittest

from backend.schema import normalize_solve_info


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate-common-nodes.py"
MODULE = runpy.run_path(str(SCRIPT), run_name="common_nodes_test_module")


class CommonNodeGenerationTests(unittest.TestCase):
    def test_representatives_cover_all_169_preflop_classes(self):
        hands = MODULE["representative_hands"]()

        self.assertEqual(len(hands), 169)
        self.assertEqual(len({label for label, _ in hands}), 169)
        self.assertTrue(all(len(set(cards)) == 2 for _, cards in hands))

    def test_ranked_nodes_are_valid_solver_information_sets(self):
        states = MODULE["common_public_states"](12)
        self.assertEqual(len(states), 12)
        self.assertEqual(states[0].probability, 1.0)
        self.assertTrue(
            all(left.probability >= right.probability for left, right in zip(states, states[1:]))
        )
        label, hole = MODULE["representative_hands"]()[0]
        for ranked in states:
            info = MODULE["state_info"](ranked, label, hole)["info"]
            normalized = normalize_solve_info(info)
            self.assertEqual(normalized["street"], "preflop")
            self.assertEqual(normalized["pending"][0], normalized["hero"]["id"])
            self.assertTrue(normalized["legalActions"])

    def test_every_hand_for_one_public_state_shares_a_validation_group(self):
        ranked = MODULE["common_public_states"](1)[0]
        groups = {
            MODULE["state_info"](ranked, label, hole)["validationGroup"]
            for label, hole in MODULE["representative_hands"]()
        }

        self.assertEqual(len(groups), 1)


if __name__ == "__main__":
    unittest.main()
