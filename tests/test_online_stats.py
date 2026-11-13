import math
import unittest

from backend.online_stats import OnlineMoments, PairedActionStats


class OnlineStatisticsTests(unittest.TestCase):
    def test_welford_matches_known_mean_and_standard_error(self):
        moments = OnlineMoments()
        for value in (1.0, 2.0, 3.0, 4.0):
            moments.update(value)

        self.assertEqual(moments.count, 4)
        self.assertAlmostEqual(moments.mean, 2.5)
        self.assertAlmostEqual(moments.variance, 5 / 3)
        self.assertAlmostEqual(moments.stderr, math.sqrt((5 / 3) / 4))

    def test_paired_stopping_uses_difference_variance(self):
        stats = PairedActionStats(("check", "raise"))
        # Both actions share a very noisy deal effect, while their difference
        # is constant. Independent confidence intervals would miss this.
        for index in range(32):
            common = 100.0 if index % 2 else -100.0
            stats.update({"check": common + 1.0, "raise": common})

        difference, stderr = stats.difference("check", "raise")
        self.assertAlmostEqual(difference, 1.0)
        self.assertEqual(stderr, 0.0)
        self.assertTrue(
            stats.can_stop(
                min_samples=32,
                confidence_z=3.9,
                target_difference_ci=0.02,
                material_edge=0.02,
            )
        )
        self.assertFalse(
            stats.can_stop(
                min_samples=32,
                confidence_z=3.9,
                target_difference_ci=0.02,
                material_edge=0.02,
                additive_model_error=2.0,
            )
        )

    def test_stopping_waits_for_minimum_samples(self):
        stats = PairedActionStats(("fold", "call", "raise"))
        for _ in range(7):
            stats.update({"fold": 0.0, "call": 1.0, "raise": -1.0})
        self.assertFalse(
            stats.can_stop(
                min_samples=8,
                confidence_z=3.9,
                target_difference_ci=0.02,
                material_edge=0.02,
            )
        )

    def test_stopping_also_respects_absolute_action_error_budget(self):
        stats = PairedActionStats(("call", "raise"))
        for index in range(128):
            common_noise = 1_000.0 if index % 2 else -1_000.0
            stats.update({"call": common_noise, "raise": common_noise - 0.1})

        self.assertFalse(
            stats.can_stop(
                min_samples=128,
                confidence_z=2.58,
                target_difference_ci=0.05,
                material_edge=0.01,
                target_action_ci=5.0,
            )
        )

    def test_bounded_radius_prevents_zero_variance_rare_event_stop(self):
        stats = PairedActionStats(("call", "raise"))
        for _ in range(128):
            stats.update({"call": 0.0, "raise": -0.1})

        self.assertFalse(
            stats.can_stop(
                min_samples=128,
                confidence_z=2.58,
                target_difference_ci=0.05,
                material_edge=0.01,
                target_action_ci=5.0,
                utility_range_width=1_000.0,
                confidence_alpha=0.001,
            )
        )

    def test_pair_radius_sums_action_specific_utility_widths(self):
        stats = PairedActionStats(("fold", "call"))
        for _ in range(128):
            stats.update({"fold": 0.0, "call": 1.0})

        common = {
            "min_samples": 128,
            "confidence_z": 2.58,
            "target_difference_ci": 0.05,
            "material_edge": 0.85,
            "confidence_alpha": 0.05,
        }
        self.assertTrue(
            stats.can_stop(
                **common,
                utility_range_widths={"fold": 0.0, "call": 1.0},
            )
        )
        self.assertFalse(
            stats.can_stop(
                **common,
                utility_range_widths={"fold": 1.0, "call": 1.0},
            )
        )

    def test_pair_model_calibration_is_sum_of_both_actions(self):
        stats = PairedActionStats(("fold", "call"))
        for _ in range(32):
            stats.update({"fold": 0.0, "call": 1.0})

        common = {
            "min_samples": 32,
            "confidence_z": 2.58,
            "target_difference_ci": 0.05,
            "material_edge": 0.85,
            "utility_range_widths": {"fold": 0.0, "call": 0.0},
        }
        self.assertTrue(
            stats.can_stop(
                **common,
                action_model_errors={"fold": 0.0, "call": 0.1},
            )
        )
        self.assertFalse(
            stats.can_stop(
                **common,
                action_model_errors={"fold": 0.2, "call": 0.1},
            )
        )


if __name__ == "__main__":
    unittest.main()
