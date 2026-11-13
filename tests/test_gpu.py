import importlib.util
import random
import unittest

from backend.gpu import BatchEvaluator, pack_score, settle_terminal_records, showdown_shares_batch
from backend.poker import evaluate


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional for the CPU-only install")
class TensorEvaluatorTests(unittest.TestCase):
    def test_tensor_evaluator_matches_canonical_python_evaluator(self):
        rng = random.Random(90210)
        hands = [rng.sample(range(52), 7) for _ in range(2_000)]
        expected = [pack_score(evaluate(hand)) for hand in hands]
        actual = BatchEvaluator("cpu").evaluate_tensor_for_test(hands)
        self.assertEqual(actual, expected)

    def test_batched_showdown_and_side_pot_settlement(self):
        board = (25, 12, 26, 5, 0)  # 8h 5s 8d 3h 2s
        showdowns = [(board, ((31, 46), (13, 6)))]  # K9 loses to 53
        shares, evaluated = showdown_shares_batch(showdowns)
        self.assertEqual(shares, [0.0])
        self.assertEqual(evaluated, 2)

        rows = (
            ("hero", 90.0, 10.0, False, 31, 46),
            ("villain", 90.0, 10.0, False, 13, 6),
        )
        utilities, evaluated = settle_terminal_records(
            [("showdown", board, rows, "villain", 90.0), ("direct", 7.5)]
        )
        self.assertEqual(utilities, [20.0, 7.5])
        self.assertEqual(evaluated, 2)


if __name__ == "__main__":
    unittest.main()
