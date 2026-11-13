from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch

from backend.belief import ALL_COMBOS
from backend.solver import SubgameCFR, build_range, make_state


POSITIONS = ("UTG", "HJ", "CO", "BTN", "SB", "BB")


def preflop_info():
    players = []
    for index, position in enumerate(POSITIONS):
        committed = 0.5 if position == "SB" else 1.0 if position == "BB" else 0.0
        players.append(
            {
                "id": f"p{index}",
                "position": position,
                "stack": 100.0 - committed,
                "streetCommitted": committed,
                "totalCommitted": committed,
                "folded": False,
                "allIn": False,
            }
        )
    return {
        "street": "preflop",
        "board": [],
        "pot": 1.5,
        "currentBet": 1.0,
        "aggressiveActions": 0,
        "hero": {
            "id": "p0",
            "position": "UTG",
            "stack": 100.0,
            "streetCommitted": 0.0,
            "hole": [0, 4],
        },
        "players": players,
        "pending": [f"p{index}" for index in range(6)],
        "history": [],
        "legalActions": ["fold", "call", "raise"],
        "raiseTo": 2.5,
    }


class UniformBlueprint:
    has_valid_policy = True
    fingerprint = "blueprint-test-v1"

    def policy(self, features, legal_actions):
        return tuple(1 / len(legal_actions) for _ in legal_actions)

    def policy_many(self, feature_rows, legal_actions):
        policy = tuple(1 / len(legal_actions) for _ in legal_actions)
        return [policy] * len(feature_rows)


class BiasedBlueprint(UniformBlueprint):
    def policy(self, features, legal_actions):
        weights = {"fold": 0.8, "call": 0.1, "raise": 0.1, "check": 0.9}
        total = sum(weights[action] for action in legal_actions)
        return tuple(weights[action] / total for action in legal_actions)

    def policy_many(self, feature_rows, legal_actions):
        return [self.policy(None, legal_actions)] * len(feature_rows)


class ApprovedLeafRuntime:
    fingerprint = "leaf-test-v1"

    def __init__(self):
        self.cutoff_gate_calls = 0
        self.predict_calls = 0
        self.model = SimpleNamespace(
            payload={
                "training": {
                    "blueprintFingerprint": "blueprint-test-v1",
                    "datasetHeaders": [
                        {"blueprintVersion": "blueprint-test-v1"},
                    ]
                }
            },
            predict=self._model_predict,
        )
        self.support = {"maxInferenceUncertaintyBb": 1.0}

    def cutoff_gate(self, model_input):
        self.cutoff_gate_calls += 1
        return SimpleNamespace(allowed=True, reasons=(), model_error_bb=0.1)

    def predict(self, model_input, seat_index, hand_index):
        return self._model_predict(model_input, seat_index, hand_index)

    def _model_predict(self, model_input, seat_index, hand_index):
        self.predict_calls += 1
        return SimpleNamespace(final_stack=101.0, uncertainty=0.01)


class RecoveringLeafRuntime(ApprovedLeafRuntime):
    def cutoff_gate(self, model_input):
        self.cutoff_gate_calls += 1
        if self.cutoff_gate_calls == 1:
            return SimpleNamespace(
                allowed=False,
                reasons=("temporary-gate-rejection",),
                model_error_bb=0.1,
            )
        return SimpleNamespace(allowed=True, reasons=(), model_error_bb=0.1)

    def _model_predict(self, model_input, seat_index, hand_index):
        self.predict_calls += 1
        if self.predict_calls == 1:
            raise RuntimeError("temporary inference failure")
        return SimpleNamespace(final_stack=101.0, uncertainty=0.01)


class DepthLimitedSearchTests(unittest.TestCase):
    def test_prediction_cache_is_reach_epoch_scoped_and_preserves_accounting(self):
        info = preflop_info()
        root = make_state(info)
        ranges = {
            player.id: build_range(
                info,
                next(row for row in info["players"] if row["id"] == player.id),
                list(range(52)),
            )
            for player in root.players
        }
        leaf = ApprovedLeafRuntime()
        with patch.dict(os.environ, {"TEXAS_LAB_LEAF_DEPTH": "1"}), patch(
            "backend.leaf_runtime.get_leaf_runtime",
            return_value=leaf,
        ):
            engine = SubgameCFR(
                root,
                {"p0": (0, 4)},
                ranges,
                seed=24,
                root_info=info,
                blueprint_runtime=UniformBlueprint(),
            )
            child = engine.action_state(root, "call")
            evaluator = engine.depth_evaluator
            self.assertIsNotNone(evaluator)

            first = evaluator.maybe_value(child, {"p0": (0, 4)}, "p0", 100.0)
            evaluator.begin_evaluation()
            cached = evaluator.maybe_value(child, {"p0": (0, 4)}, "p0", 99.0)
            other_hand = evaluator.maybe_value(child, {"p1": (8, 12)}, "p1", 100.0)
            evaluator.refresh_beliefs()
            refreshed = evaluator.maybe_value(child, {"p0": (0, 4)}, "p0", 100.0)

        self.assertEqual(first.utility, 1.0)
        # baseline_stack is deliberately absent from the cache key: the raw
        # final-stack prediction must be converted on every caller invocation.
        self.assertEqual(cached.utility, 2.0)
        self.assertEqual(other_hand.utility, 1.0)
        self.assertEqual(refreshed.utility, first.utility)
        self.assertEqual(leaf.predict_calls, 3)
        self.assertEqual(leaf.cutoff_gate_calls, 2)
        diagnostics = evaluator.diagnostics()
        self.assertEqual(diagnostics["leafCutoffHits"], 4)
        self.assertEqual(diagnostics["leafEvaluationCutoffHits"], 3)
        self.assertEqual(diagnostics["leafModelErrorBound"], 0.1)
        self.assertEqual(diagnostics["leafModelMaxUncertainty"], 0.01)
        self.assertEqual(diagnostics["leafPredictionCacheHits"], 1)
        self.assertEqual(diagnostics["leafPredictionCacheMisses"], 3)
        self.assertEqual(diagnostics["leafCutoffGateCacheHits"], 1)
        self.assertEqual(diagnostics["leafCutoffGateCacheMisses"], 2)
        self.assertEqual(diagnostics["leafReachEpoch"], 1)

    def test_rejections_and_prediction_errors_are_not_cached(self):
        info = preflop_info()
        root = make_state(info)
        ranges = {
            player.id: build_range(
                info,
                next(row for row in info["players"] if row["id"] == player.id),
                list(range(52)),
            )
            for player in root.players
        }
        leaf = RecoveringLeafRuntime()
        with patch.dict(os.environ, {"TEXAS_LAB_LEAF_DEPTH": "1"}), patch(
            "backend.leaf_runtime.get_leaf_runtime",
            return_value=leaf,
        ):
            engine = SubgameCFR(
                root,
                {"p0": (0, 4)},
                ranges,
                seed=25,
                root_info=info,
                blueprint_runtime=UniformBlueprint(),
            )
            child = engine.action_state(root, "call")
            evaluator = engine.depth_evaluator
            self.assertIsNotNone(evaluator)
            first = evaluator.maybe_value(child, {"p0": (0, 4)}, "p0", 100.0)
            second = evaluator.maybe_value(child, {"p0": (0, 4)}, "p0", 100.0)
            recovered = evaluator.maybe_value(child, {"p0": (0, 4)}, "p0", 100.0)
            cached = evaluator.maybe_value(child, {"p0": (0, 4)}, "p0", 100.0)

        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertEqual(recovered.utility, 1.0)
        self.assertEqual(cached.utility, recovered.utility)
        self.assertEqual(leaf.cutoff_gate_calls, 3)
        self.assertEqual(leaf.predict_calls, 2)
        diagnostics = evaluator.diagnostics()
        self.assertEqual(diagnostics["leafPredictionCacheHits"], 1)
        self.assertEqual(diagnostics["leafPredictionCacheMisses"], 3)
        self.assertEqual(diagnostics["leafCutoffGateCacheHits"], 0)
        self.assertEqual(diagnostics["leafCutoffGateCacheMisses"], 3)
        self.assertEqual(
            diagnostics["leafCutoffRejections"]["temporary-gate-rejection"],
            1,
        )
        self.assertEqual(
            diagnostics["leafCutoffRejections"]["leaf-evaluation-error"],
            1,
        )

    def test_approved_model_updates_public_belief_and_cuts_tree(self):
        info = preflop_info()
        root = make_state(info)
        ranges = {
            player.id: build_range(
                info,
                next(row for row in info["players"] if row["id"] == player.id),
                list(range(52)),
            )
            for player in root.players
        }
        blueprint = UniformBlueprint()
        with patch.dict(os.environ, {"TEXAS_LAB_LEAF_DEPTH": "1"}), patch(
            "backend.leaf_runtime.get_leaf_runtime",
            return_value=ApprovedLeafRuntime(),
        ):
            engine = SubgameCFR(
                root,
                {"p0": (0, 4)},
                ranges,
                seed=19,
                root_info=info,
                blueprint_runtime=blueprint,
            )
            child = engine.action_state(root, "call")
            engine.begin_evaluation()
            value = engine.depth_limited_utility(child, {"p0": (0, 4)}, "p0")

        self.assertEqual(value, 1.0)
        diagnostics = engine.depth_evaluator.diagnostics()
        self.assertTrue(diagnostics["leafCutoffActive"])
        self.assertEqual(diagnostics["leafCutoffHits"], 1)
        self.assertEqual(diagnostics["leafEvaluationCutoffHits"], 1)
        self.assertEqual(diagnostics["leafModelErrorBound"], 0.1)
        self.assertEqual(diagnostics["leafBeliefUpdates"], 1)
        self.assertGreater(diagnostics["leafBlueprintRangePolicyQueries"], 1_000)

    def test_blueprint_version_mismatch_disables_cutoff(self):
        info = preflop_info()
        root = make_state(info)
        ranges = {
            player.id: build_range(
                info,
                next(row for row in info["players"] if row["id"] == player.id),
                list(range(52)),
            )
            for player in root.players
        }
        leaf = ApprovedLeafRuntime()
        leaf.model.payload["training"]["datasetHeaders"][0]["blueprintVersion"] = "other"
        with patch("backend.leaf_runtime.get_leaf_runtime", return_value=leaf):
            engine = SubgameCFR(
                root,
                {"p0": (0, 4)},
                ranges,
                seed=20,
                root_info=info,
                blueprint_runtime=BiasedBlueprint(),
            )

        self.assertFalse(engine.depth_evaluator.active)
        self.assertIn(
            "blueprint-version-mismatch",
            engine.depth_evaluator.diagnostics()["leafCutoffRejections"],
        )

    def test_folded_target_uses_exact_known_stack_instead_of_leaf_model(self):
        info = preflop_info()
        root = make_state(info)
        ranges = {
            player.id: build_range(
                info,
                next(row for row in info["players"] if row["id"] == player.id),
                list(range(52)),
            )
            for player in root.players
        }
        with patch.dict(os.environ, {"TEXAS_LAB_LEAF_DEPTH": "1"}), patch(
            "backend.leaf_runtime.get_leaf_runtime",
            return_value=ApprovedLeafRuntime(),
        ):
            engine = SubgameCFR(
                root,
                {"p0": (0, 4)},
                ranges,
                seed=22,
                root_info=info,
                blueprint_runtime=UniformBlueprint(),
            )
            folded = engine.action_state(root, "fold")
            value = engine.depth_limited_utility(folded, {"p0": (0, 4)}, "p0")

        self.assertEqual(value, 0.0)
        diagnostics = engine.depth_evaluator.diagnostics()
        self.assertEqual(diagnostics["leafCutoffHits"], 0)
        self.assertEqual(diagnostics["leafExactFoldShortcuts"], 1)

    def test_leaf_reach_uses_current_cfr_policy_and_refreshes_between_epochs(self):
        info = preflop_info()
        root = make_state(info)
        ranges = {
            player.id: build_range(
                info,
                next(row for row in info["players"] if row["id"] == player.id),
                list(range(52)),
            )
            for player in root.players
        }
        with patch("backend.leaf_runtime.get_leaf_runtime", return_value=ApprovedLeafRuntime()):
            engine = SubgameCFR(
                root,
                {"p0": (0, 4)},
                ranges,
                seed=21,
                root_info=info,
                blueprint_runtime=UniformBlueprint(),
            )
            root_node = engine.get_infoset(root, {"p0": (0, 4)})
            call_index = root_node.actions.index("call")
            root_node.regrets = [0.0] * len(root_node.actions)
            root_node.regrets[call_index] = 100.0
            child = engine.action_state(root, "call")
            engine.refresh_leaf_reach()

            first = engine.depth_evaluator._belief(child).ranges["p0"]
            selected_index = next(
                index
                for index, combo in enumerate(ALL_COMBOS)
                if combo == (0, 4)
            )
            other_index = next(
                index
                for index, combo in enumerate(ALL_COMBOS)
                if combo == (48, 49)
            )
            self.assertGreater(
                first.probabilities[selected_index],
                2.5 * first.probabilities[other_index],
            )

            root_node.regrets = [100.0 if action == "fold" else 0.0 for action in root_node.actions]
            engine.refresh_leaf_reach()
            second = engine.depth_evaluator._belief(child).ranges["p0"]

        self.assertEqual(second.probabilities[selected_index], 0.0)
        diagnostics = engine.depth_evaluator.diagnostics()
        self.assertGreater(diagnostics["leafCurrentPolicyRangeQueries"], 0)
        self.assertGreater(diagnostics["leafBlueprintRangePolicyQueries"], 0)
        self.assertGreaterEqual(diagnostics["leafBeliefRefreshes"], 1)

    def test_disabled_blueprint_prior_uses_uniform_policy_for_unseen_reach_bucket(self):
        info = preflop_info()
        root = make_state(info)
        ranges = {
            player.id: build_range(
                info,
                next(row for row in info["players"] if row["id"] == player.id),
                list(range(52)),
            )
            for player in root.players
        }
        actions = ("fold", "call", "raise")
        with patch.dict(os.environ, {"TEXAS_LAB_BLUEPRINT_PRIOR_WEIGHT": "0"}), patch(
            "backend.leaf_runtime.get_leaf_runtime",
            return_value=ApprovedLeafRuntime(),
        ):
            engine = SubgameCFR(
                root,
                {"p0": (0, 4)},
                ranges,
                seed=23,
                root_info=info,
                blueprint_runtime=BiasedBlueprint(),
            )
            policies, existing = engine.peek_range_policies(
                root,
                "p0",
                [0],
                [[0.0] * 139],
                actions,
            )

        self.assertEqual(existing, 0)
        self.assertEqual(policies[0], (1 / 3, 1 / 3, 1 / 3))


if __name__ == "__main__":
    unittest.main()
