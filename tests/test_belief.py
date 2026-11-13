from __future__ import annotations

import math
import random
import unittest

from backend.belief import ALL_COMBOS, BeliefState, ComboRange


class ComboRangeTests(unittest.TestCase):
    def test_uniform_range_removes_blocked_combos_and_renormalizes(self):
        blocked = {0, 1, 2}
        hand_range = ComboRange.uniform(blocked)

        self.assertAlmostEqual(sum(hand_range.probabilities), 1.0)
        for combo, probability in zip(ALL_COMBOS, hand_range.probabilities):
            if blocked.intersection(combo):
                self.assertEqual(probability, 0.0)

        diagnostics = hand_range.diagnostics()
        expected = math.comb(52 - len(blocked), 2)
        self.assertEqual(diagnostics.supported_combos, expected)
        self.assertAlmostEqual(diagnostics.effective_sample_size, expected)
        self.assertAlmostEqual(diagnostics.entropy, math.log(expected))

    def test_action_likelihood_is_a_normalized_bayes_update(self):
        prior = ComboRange.uniform()

        def contains_ace(combo: tuple[int, int]) -> bool:
            return any(card // 4 == 12 for card in combo)

        prior_mass = sum(
            probability
            for combo, probability in zip(ALL_COMBOS, prior.probabilities)
            if contains_ace(combo)
        )
        posterior = prior.update_action(
            lambda combo: 4.0 if contains_ace(combo) else 1.0
        )
        posterior_mass = sum(
            probability
            for combo, probability in zip(ALL_COMBOS, posterior.probabilities)
            if contains_ace(combo)
        )
        expected = (4 * prior_mass) / (4 * prior_mass + 1 - prior_mass)

        self.assertAlmostEqual(posterior_mass, expected)
        self.assertGreater(posterior_mass, prior_mass)
        self.assertAlmostEqual(sum(posterior.probabilities), 1.0)

    def test_new_public_card_conditions_an_existing_range(self):
        prior = ComboRange.uniform()
        conditioned = prior.condition_cards([51])

        self.assertAlmostEqual(sum(conditioned.probabilities), 1.0)
        self.assertTrue(
            all(
                probability == 0.0
                for combo, probability in zip(ALL_COMBOS, conditioned.probabilities)
                if 51 in combo
            )
        )

    def test_folded_player_range_is_retained_for_hidden_card_blocking(self):
        active_range = ComboRange.uniform()
        folded_range = active_range.update_action(
            lambda combo: 8.0 if 48 in combo else 1.0
        )
        belief = BeliefState(
            {"active": active_range, "folded": folded_range},
            public_cards=(),
        )

        updated = belief.update_action(
            "active",
            lambda combo: 3.0 if 49 in combo else 1.0,
        )
        self.assertIs(updated.ranges["folded"], folded_range)
        self.assertIsNot(updated.ranges["active"], active_range)

        conditioned = updated.condition_public_cards([12])
        for hand_range in conditioned.ranges.values():
            self.assertTrue(
                all(
                    probability == 0.0
                    for combo, probability in zip(ALL_COMBOS, hand_range.probabilities)
                    if 12 in combo
                )
            )

        # A folded hand is not a showdown participant, but retaining its range
        # lets a full hidden deal remove its two physical cards from the deck.
        deal = conditioned.sample_deal(
            ["active", "folded"],
            random.Random(20260731),
        )
        self.assertFalse(set(deal["active"]) & set(deal["folded"]))
        self.assertNotIn(12, deal["active"])
        self.assertNotIn(12, deal["folded"])
        self.assertEqual(len(conditioned.flattened(["active", "folded"])), 2 * 1326)


if __name__ == "__main__":
    unittest.main()
