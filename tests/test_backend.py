import unittest
from itertools import combinations
import os
from unittest.mock import patch

import backend.solver as solver_module
from backend.poker import evaluate
from backend.solver import (
    SubgameCFR,
    active_players,
    advance_street,
    apply_action,
    board_texture,
    board_texture_code,
    build_range,
    hand_bucket_code,
    is_terminal,
    hand_bucket,
    make_state,
    policy_bucket_groups,
    solve_decision,
)


POSITIONS = ("UTG", "HJ", "CO", "BTN", "SB", "BB")


def river_info(*, actor="p5", pending=("p5", "p3"), current_bet=0, pot=20):
    board = [25, 12, 26, 5, 0]
    players = []
    for index, position in enumerate(POSITIONS):
        active = index in (3, 5)
        committed = 10.0 if active else 0.0
        players.append(
            {
                "id": f"p{index}",
                "position": position,
                "stack": 90.0 if active else 100.0,
                "streetCommitted": 0.0,
                "totalCommitted": committed,
                "folded": not active,
                "allIn": False,
            }
        )
    hole = [13, 6] if actor == "p5" else [31, 46]
    position = POSITIONS[int(actor[1:])]
    legal = ["check", "raise"] if current_bet == 0 else ["fold", "call", "raise"]
    return {
        "street": "river",
        "board": board,
        "pot": pot,
        "currentBet": current_bet,
        "aggressiveActions": 0 if current_bet == 0 else 1,
        "hero": {"id": actor, "position": position, "stack": 90, "streetCommitted": 0, "hole": hole},
        "players": players,
        "pending": list(pending),
        "history": [],
        "legalActions": legal,
        "raiseTo": 10 if current_bet == 0 else 30,
    }


class BackendSolverTests(unittest.TestCase):
    def test_training_convergence_requires_consecutive_fresh_ev_snapshots(self):
        convergence = solver_module.TrainingConvergence(
            ("check", "raise"),
            minimum_iterations=100,
            tolerance=0.1,
            patience=2,
        )

        self.assertFalse(
            convergence.inspect(50, 1, {"check": 1.0, "raise": 1.5})
        )
        self.assertFalse(
            convergence.inspect(100, 2, {"check": 1.05, "raise": 1.44})
        )
        self.assertFalse(
            convergence.inspect(125, 2, {"check": 1.05, "raise": 1.44})
        )
        self.assertTrue(
            convergence.inspect(150, 3, {"check": 1.08, "raise": 1.40})
        )
        self.assertEqual(convergence.checks, 3)
        self.assertEqual(convergence.stable_checks, 2)
        self.assertAlmostEqual(convergence.maximum_change, 0.04)

    def test_preflop_training_can_stop_when_one_action_stably_dominates(self):
        convergence = solver_module.TrainingConvergence(
            ("fold", "call", "raise"),
            minimum_iterations=100,
            tolerance=0.001,
            patience=3,
            dominance_margin=0.25,
            dominance_patience=2,
            dominance_drift=1.0,
            minimum_best_frequency=0.70,
        )

        self.assertFalse(
            convergence.inspect(
                50,
                1,
                {"fold": 0.0, "call": -1.0, "raise": -2.0},
                {"fold": 0.80, "call": 0.10, "raise": 0.10},
            )
        )
        self.assertFalse(
            convergence.inspect(
                100,
                2,
                {"fold": 0.0, "call": -1.1, "raise": -2.1},
                {"fold": 0.82, "call": 0.09, "raise": 0.09},
            )
        )
        self.assertTrue(
            convergence.inspect(
                150,
                3,
                {"fold": 0.0, "call": -1.15, "raise": -2.15},
                {"fold": 0.84, "call": 0.08, "raise": 0.08},
            )
        )
        self.assertEqual(convergence.stop_reason, "action-dominated")
        self.assertEqual(convergence.best_action, "fold")
        self.assertEqual(convergence.decision_stable_checks, 2)
        self.assertAlmostEqual(convergence.minimum_gap, 1.15)

    def test_dominance_stop_requires_strategy_mass_on_the_best_action(self):
        convergence = solver_module.TrainingConvergence(
            ("fold", "call"),
            minimum_iterations=2,
            tolerance=0.001,
            patience=3,
            dominance_margin=0.1,
            dominance_patience=1,
            dominance_drift=0.0,
            minimum_best_frequency=0.70,
        )
        convergence.inspect(
            1,
            1,
            {"fold": 0.0, "call": -2.0},
            {"fold": 0.69, "call": 0.31},
        )

        self.assertFalse(
            convergence.inspect(
                2,
                2,
                {"fold": 0.0, "call": -2.0},
                {"fold": 0.69, "call": 0.31},
            )
        )
        self.assertIsNone(convergence.stop_reason)

    def test_replica_prunes_remaining_training_when_root_evs_stabilize(self):
        with patch.dict(
            os.environ,
            {
                "TEXAS_LAB_ADAPTIVE_TRAINING": "true",
                "TEXAS_LAB_TRAIN_MIN_ITERATIONS": "2",
                "TEXAS_LAB_TRAIN_CHUNK_SIZE": "2",
                "TEXAS_LAB_TRAIN_EV_TOLERANCE": "999",
                "TEXAS_LAB_TRAIN_STABLE_CHECKS": "1",
                "TEXAS_LAB_ADAPTIVE_EVALUATION": "off",
            },
        ):
            result = solver_module.run_replica(
                river_info(actor="p3", pending=("p3",)),
                training_iterations=10,
                evaluation_iterations=1,
                seed_nonce=92,
            )

        self.assertTrue(result["adaptiveTrainingEnabled"])
        self.assertTrue(result["adaptiveTrainingStopped"])
        self.assertEqual(result["trainingIterations"], 4)
        self.assertEqual(result["trainingRequestedIterations"], 10)
        self.assertEqual(result["trainingPrunedIterations"], 6)
        self.assertEqual(result["trainingEvChecks"], 2)
        self.assertGreater(result["trainingRootEvSamples"], 0)

    def test_execution_semantics_are_part_of_solver_configuration(self):
        cpu = solver_module.solver_execution_semantics(
            uses_cuda=False,
            leaf_runtime_loaded=False,
        )
        cuda = solver_module.solver_execution_semantics(
            uses_cuda=True,
            leaf_runtime_loaded=False,
        )
        with patch.dict(
            os.environ,
            {"TEXAS_LAB_LEAF_BELIEF_REFRESH": "16"},
        ):
            cuda_with_leaf = solver_module.solver_execution_semantics(
                uses_cuda=True,
                leaf_runtime_loaded=True,
            )

        self.assertIn("cpu-sequential:batch=1", cpu)
        self.assertIn("cuda-minibatch:batch=64", cuda)
        self.assertIn("cuda-minibatch:batch=16", cuda_with_leaf)
        self.assertNotEqual(
            solver_module.solver_config_signature(cpu),
            solver_module.solver_config_signature(cuda),
        )
        with patch.dict(
            os.environ,
            {"TEXAS_LAB_ADAPTIVE_TRAINING": "off"},
        ):
            fixed_training = solver_module.solver_config_signature(cpu)
        with patch.dict(
            os.environ,
            {"TEXAS_LAB_ADAPTIVE_TRAINING": "true"},
        ):
            adaptive_training = solver_module.solver_config_signature(cpu)
        self.assertNotEqual(fixed_training, adaptive_training)

    def test_parallel_replica_uses_the_full_fixed_evaluation_budget(self):
        with patch.dict(
            os.environ,
            {"TEXAS_LAB_ADAPTIVE_EVALUATION": "true"},
        ):
            result = solver_module.run_replica(
                river_info(actor="p3", pending=("p3",)),
                training_iterations=1,
                evaluation_iterations=7,
                seed_nonce=91,
                parallel_fixed_evaluation=True,
            )

        self.assertTrue(result["adaptiveEvaluationRequested"])
        self.assertFalse(result["adaptiveEvaluationEnabled"])
        self.assertTrue(result["parallelFixedEvaluation"])
        self.assertFalse(result["adaptiveEvaluationStopped"])
        self.assertEqual(result["evaluationIterations"], 7)
        self.assertEqual(
            result["confidenceMethod"],
            "pooled-fixed-sample-empirical-bernstein",
        )
        self.assertTrue(
            all(len(samples) == 7 for samples in result["actionOutcomes"].values())
        )

    def test_evaluator_orders_straight_flush_above_quads(self):
        straight_flush = [28, 24, 20, 16, 12, 1, 2]
        quads = [48, 49, 50, 51, 44, 1, 2]
        self.assertGreater(evaluate(straight_flush), evaluate(quads))

    def test_check_keeps_the_remaining_player_in_the_action_tree(self):
        state = make_state(river_info())
        checked = apply_action(state, "check")
        self.assertEqual(checked.pending, ("p3",))
        self.assertFalse(is_terminal(checked))
        self.assertEqual(checked.pot, 20)

    def test_specialized_street_transition_preserves_public_state(self):
        info = river_info(actor="p3", pending=("p3",))
        info["street"] = "flop"
        info["board"] = info["board"][:3]
        info["aggressiveActions"] = 2
        info["players"][3]["streetCommitted"] = 4
        state = make_state(info)

        advanced = advance_street(state, (5,))

        self.assertEqual(advanced.street, "turn")
        self.assertEqual(advanced.board, (*state.board, 5))
        self.assertEqual(advanced.pot, state.pot)
        self.assertEqual(advanced.current_bet, 0)
        self.assertEqual(advanced.aggressive_actions, 0)
        self.assertEqual(advanced.pending, ("p5", "p3"))
        self.assertTrue(all(player.street_committed == 0 for player in advanced.players))
        self.assertEqual(
            [player.total_committed for player in advanced.players],
            [player.total_committed for player in state.players],
        )
        self.assertEqual(state.players[3].street_committed, 4)

    def test_solver_caches_compact_public_action_transitions(self):
        info = river_info()
        root = make_state(info)
        root_hand = tuple(info["hero"]["hole"])
        available = [card for card in range(52) if card not in root.board]
        ranges = {
            player.id: build_range(
                info,
                next(row for row in info["players"] if row["id"] == player.id),
                available,
            )
            for player in active_players(root)
        }
        solver = SubgameCFR(root, {info["hero"]["id"]: root_hand}, ranges, seed=440)

        first = solver.action_state(root, "check")
        second = solver.action_state(root, "check")
        raised = solver.action_state(root, "raise")

        self.assertIs(first, second)
        self.assertEqual(first, apply_action(root, "check"))
        self.assertNotEqual(first.node_id, raised.node_id)
        self.assertEqual(solver.action_transition_hits, 1)
        self.assertEqual(solver.action_transition_misses, 2)
        self.assertEqual(len(solver.infoset_key(root, info["hero"]["id"], root_hand)), 4)

    def test_bucket_codes_are_stable_and_collision_free_for_seen_features(self):
        info = river_info(actor="p3", pending=("p3",))
        info["street"] = "flop"
        info["board"] = info["board"][:3]
        root = make_state(info)
        root_hand = tuple(info["hero"]["hole"])
        available = [card for card in range(52) if card not in root.board]
        ranges = {
            player.id: build_range(
                info,
                next(row for row in info["players"] if row["id"] == player.id),
                available,
            )
            for player in active_players(root)
        }
        hands_by_feature = {}
        for hand in combinations(available, 2):
            hands_by_feature.setdefault(hand_bucket(hand, root.board), hand)
            if len(hands_by_feature) >= 12:
                break
        hands = list(hands_by_feature.values())
        forward = SubgameCFR(root, {"p3": root_hand}, ranges, seed=446)
        reverse = SubgameCFR(root, {"p3": root_hand}, ranges, seed=446)

        forward_codes = {
            hand: forward.infoset_key(root, "p3", hand)[2] for hand in hands
        }
        for hand in reversed(hands):
            reverse.infoset_key(root, "p3", hand)
        reverse_codes = {
            hand: reverse.infoset_key(root, "p3", hand)[2] for hand in hands
        }

        features = {hand_bucket(hand, root.board) for hand in hands}
        self.assertEqual(forward_codes, reverse_codes)
        self.assertEqual(len({hand_bucket_code(row) for row in features}), len(features))
        self.assertEqual(forward.hand_bucket_features, reverse.hand_bucket_features)

        texture_features = {
            board_texture(tuple(board))
            for board in (
                (0, 5, 10),
                (0, 1, 18),
                (3, 15, 27, 39),
                (8, 9, 10, 11, 12),
                (48, 44, 40, 36, 32),
            )
        }
        self.assertEqual(
            len({board_texture_code(row) for row in texture_features}),
            len(texture_features),
        )

    def test_evaluation_uses_hand_aware_policy_backoff_before_uniform(self):
        info = river_info(actor="p3", pending=("p3",))
        root = make_state(info)
        source_hand = tuple(info["hero"]["hole"])
        available = [card for card in range(52) if card not in root.board]
        ranges = {
            player.id: build_range(
                info,
                next(row for row in info["players"] if row["id"] == player.id),
                available,
            )
            for player in active_players(root)
        }
        solver = SubgameCFR(root, {info["hero"]["id"]: source_hand}, ranges, seed=443)
        source_features = hand_bucket(source_hand, root.board)
        source_group = policy_bucket_groups(source_features)[0]
        target_hand = next(
            combo
            for combo in combinations(available, 2)
            if hand_bucket(combo, root.board) != source_features
            and policy_bucket_groups(hand_bucket(combo, root.board))[0] == source_group
        )

        source_node = solver.get_infoset(root, {"p3": source_hand})
        source_node.strategy_sum[:] = [9.0, 1.0]
        solver.begin_evaluation()
        target_node = solver.get_infoset(root, {"p3": target_hand})
        strategy = solver.evaluation_strategy(root, {"p3": target_hand}, target_node)
        cached_strategy = solver.evaluation_strategy(root, {"p3": target_hand}, target_node)

        self.assertEqual(strategy, (0.9, 0.1))
        self.assertIs(cached_strategy, strategy)
        self.assertEqual(solver.evaluation_policy_backoff_lookups, 2)
        self.assertEqual(solver.evaluation_uniform_fallback_lookups, 0)
        self.assertEqual(len(solver.evaluation_strategy_cache), 1)

    def test_validated_blueprint_policy_warm_starts_new_infosets(self):
        class FixedBlueprint:
            has_valid_policy = True

            def policy(self, features, actions):
                self.feature_size = len(features)
                self.actions = actions
                return (0.8, 0.2)

        info = river_info(actor="p3", pending=("p3",))
        root = make_state(info)
        root_hand = tuple(info["hero"]["hole"])
        available = [card for card in range(52) if card not in root.board]
        ranges = {
            player.id: build_range(
                info,
                next(row for row in info["players"] if row["id"] == player.id),
                available,
            )
            for player in root.players
        }
        blueprint = FixedBlueprint()
        engine = SubgameCFR(
            root,
            {"p3": root_hand},
            ranges,
            seed=444,
            root_info=info,
            blueprint_runtime=blueprint,
        )

        node = engine.get_infoset(root, {"p3": root_hand})

        self.assertTrue(node.blueprint_prior)
        self.assertEqual(node.strategy(), [0.8, 0.2])
        self.assertEqual(node.average_strategy(), [0.8, 0.2])
        self.assertEqual(engine.blueprint_policy_seeds, 1)
        self.assertEqual(blueprint.feature_size, 139)
        self.assertEqual(blueprint.actions, ("check", "raise"))

    def test_gpu_batched_training_matches_sequential_cfr_at_batch_size_one(self):
        info = river_info(actor="p3", pending=("p3",))
        root = make_state(info)
        root_hand = tuple(info["hero"]["hole"])
        available = [card for card in range(52) if card not in root.board]
        ranges = {
            player.id: build_range(
                info,
                next(row for row in info["players"] if row["id"] == player.id),
                available,
            )
            for player in active_players(root)
        }
        sequential = SubgameCFR(root, {"p3": root_hand}, ranges, seed=441)
        batched = SubgameCFR(root, {"p3": root_hand}, ranges, seed=441)

        sequential.train(40)
        metrics = batched.train_gpu_batched(40, batch_size=1)

        sequential_root = sequential.get_infoset(root, {"p3": root_hand})
        batched_root = batched.get_infoset(root, {"p3": root_hand})
        self.assertEqual(sequential_root.visits, batched_root.visits)
        self.assertEqual(sequential_root.regret_updates, batched_root.regret_updates)
        self.assertEqual(sequential_root.regrets, batched_root.regrets)
        self.assertEqual(sequential_root.strategy_sum, batched_root.strategy_sum)
        self.assertEqual(sequential.root_strategy_sum, batched.root_strategy_sum)
        self.assertEqual(sequential.root_strategy_weight, batched.root_strategy_weight)
        self.assertGreater(metrics["handEvaluations"], 0)

    def test_signed_linear_cfr_matches_deferred_batch_size_one(self):
        info = river_info(actor="p3", pending=("p3",))
        root = make_state(info)
        root_hand = tuple(info["hero"]["hole"])
        available = [card for card in range(52) if card not in root.board]
        ranges = {
            player.id: build_range(
                info,
                next(row for row in info["players"] if row["id"] == player.id),
                available,
            )
            for player in active_players(root)
        }
        sequential = SubgameCFR(
            root,
            {"p3": root_hand},
            ranges,
            seed=445,
            regret_mode="linear",
        )
        batched = SubgameCFR(
            root,
            {"p3": root_hand},
            ranges,
            seed=445,
            regret_mode="linear",
        )

        sequential.train(40)
        batched.train_gpu_batched(40, batch_size=1)

        self.assertEqual(sequential.infosets, batched.infosets)
        self.assertEqual(sequential.root_strategy_sum, batched.root_strategy_sum)
        self.assertEqual(sequential.training_iteration, 40)
        self.assertEqual(sequential.training_diagnostics()["regretMode"], "linear")

    def test_training_chunks_continue_the_global_mccfr_schedule(self):
        info = river_info(actor="p3", pending=("p3",))
        root = make_state(info)
        root_hand = tuple(info["hero"]["hole"])
        available = [card for card in range(52) if card not in root.board]
        ranges = {
            player.id: build_range(
                info,
                next(row for row in info["players"] if row["id"] == player.id),
                available,
            )
            for player in active_players(root)
        }
        continuous = SubgameCFR(root, {"p3": root_hand}, ranges, seed=444)
        chunked = SubgameCFR(root, {"p3": root_hand}, ranges, seed=444)

        continuous.train(41)
        first = chunked.train_chunk(13)
        final = chunked.train_chunk(28)

        self.assertEqual(first["trainingIterations"], 13)
        self.assertEqual(first["chunkIterations"], 13)
        self.assertEqual(final["trainingIterations"], 41)
        self.assertEqual(final["chunkIterations"], 28)
        self.assertEqual(final["completedTraverserCycles"], 20)
        self.assertEqual(final["nextTraverser"], "p5")
        self.assertEqual(final["nextAveragePlayer"], "p3")
        self.assertEqual(final["nextAverageWeight"], 21)
        self.assertEqual(final["batches"], 0)
        self.assertEqual(final["handEvaluations"], 0)
        self.assertEqual(chunked.training_iteration, continuous.training_iteration)
        self.assertEqual(chunked.infosets, continuous.infosets)
        self.assertEqual(chunked.hand_bucket_ids, continuous.hand_bucket_ids)
        self.assertEqual(chunked.board_texture_ids, continuous.board_texture_ids)
        self.assertEqual(chunked.root_strategy_sum, continuous.root_strategy_sum)
        self.assertEqual(chunked.root_strategy_weight, continuous.root_strategy_weight)
        self.assertEqual(
            chunked.root_action_value_sums,
            continuous.root_action_value_sums,
        )
        self.assertEqual(
            chunked.root_action_value_samples,
            continuous.root_action_value_samples,
        )
        self.assertEqual(chunked.rng.getstate(), continuous.rng.getstate())

    def test_training_after_evaluation_restores_training_semantics(self):
        info = river_info(actor="p3", pending=("p3",))
        root = make_state(info)
        root_hand = tuple(info["hero"]["hole"])
        available = [card for card in range(52) if card not in root.board]
        ranges = {
            player.id: build_range(
                info,
                next(row for row in info["players"] if row["id"] == player.id),
                available,
            )
            for player in active_players(root)
        }
        solver = SubgameCFR(root, {"p3": root_hand}, ranges, seed=446)
        solver.train(8)
        solver.begin_evaluation()
        evaluation_lookups = solver.evaluation_infoset_lookups

        solver.train_chunk(4)

        self.assertFalse(solver.evaluation_mode)
        self.assertEqual(solver.evaluation_infoset_lookups, evaluation_lookups)

    def test_gpu_training_chunks_continue_the_global_batched_schedule(self):
        info = river_info(actor="p3", pending=("p3",))
        root = make_state(info)
        root_hand = tuple(info["hero"]["hole"])
        available = [card for card in range(52) if card not in root.board]
        ranges = {
            player.id: build_range(
                info,
                next(row for row in info["players"] if row["id"] == player.id),
                available,
            )
            for player in active_players(root)
        }
        continuous = SubgameCFR(root, {"p3": root_hand}, ranges, seed=445)
        chunked = SubgameCFR(root, {"p3": root_hand}, ranges, seed=445)

        continuous.train_gpu_batched(24, batch_size=4)
        first = chunked.train_chunk(8, use_gpu=True, batch_size=4)
        final = chunked.train_chunk(16, use_gpu=True, batch_size=4)

        self.assertEqual(first["trainingIterations"], 8)
        self.assertEqual(final["trainingIterations"], 24)
        self.assertGreater(first["batches"], 0)
        self.assertGreater(first["handEvaluations"], 0)
        self.assertEqual(chunked.training_iteration, continuous.training_iteration)
        self.assertEqual(chunked.infosets, continuous.infosets)
        self.assertEqual(chunked.root_strategy_sum, continuous.root_strategy_sum)
        self.assertEqual(chunked.root_strategy_weight, continuous.root_strategy_weight)
        self.assertEqual(
            chunked.root_action_value_sums,
            continuous.root_action_value_sums,
        )
        self.assertEqual(
            chunked.root_action_value_samples,
            continuous.root_action_value_samples,
        )
        self.assertEqual(chunked.rng.getstate(), continuous.rng.getstate())

    def test_deal_samples_one_shared_public_runout(self):
        info = river_info(actor="p3", pending=("p3",))
        info["street"] = "flop"
        info["board"] = info["board"][:3]
        root = make_state(info)
        root_hand = tuple(info["hero"]["hole"])
        available = [card for card in range(52) if card not in root.board]
        ranges = {
            player.id: build_range(
                info,
                next(row for row in info["players"] if row["id"] == player.id),
                available,
            )
            for player in active_players(root)
        }
        solver = SubgameCFR(root, {"p3": root_hand}, ranges, seed=442)

        hands, runout = solver.sample_deal()
        self.assertEqual(len(runout), 2)
        self.assertFalse(set(runout) & set(root.board))
        self.assertFalse(set(runout) & {card for hand in hands.values() for card in hand})

        turn, river_runout = solver.chance_step(root, runout)
        self.assertEqual(turn.board[-1], runout[0])
        self.assertEqual(river_runout, runout[1:])

    def test_solver_uses_monetary_ev_without_preflop_policy_penalty(self):
        self.assertFalse(hasattr(solver_module, "preflop_policy_penalty"))
        result = solve_decision(river_info(actor="p3", pending=("p3",)), iterations=500)
        self.assertEqual({row["action"] for row in result["actions"]}, {"check", "raise"})
        self.assertAlmostEqual(sum(row["frequency"] for row in result["actions"]), 1.0)
        self.assertIn("MCCFR", result["model"])
        self.assertGreater(result["trainingIterations"], 0)
        self.assertGreater(result["evaluationIterations"], 0)
        self.assertGreaterEqual(result["rootRegret"], 0)
        self.assertGreater(result["evaluationInfosetLookups"], 0)
        self.assertGreater(
            result["publicTransitionHits"] + result["publicTransitionMisses"],
            0,
        )
        self.assertGreaterEqual(result["evaluationUntrainedRate"], 0)
        self.assertLessEqual(result["evaluationUntrainedRate"], 1)
        self.assertGreaterEqual(result["evaluationUncoveredRate"], 0)
        self.assertLessEqual(result["evaluationUncoveredRate"], 1)
        self.assertGreaterEqual(result["evaluationPolicyBackoffRate"], 0)
        self.assertLessEqual(result["evaluationPolicyBackoffRate"], 1)
        self.assertGreaterEqual(result["evaluationUniformFallbackRate"], 0)
        self.assertLessEqual(result["evaluationUniformFallbackRate"], 1)
        self.assertIn(result["strategyReliable"], (True, False))
        self.assertTrue(result["statisticallyTied"])
        best = max(result["actions"], key=lambda row: row["ev"])
        self.assertEqual(best["evLossEstimate"], 0)
        self.assertEqual(best["differenceCi95"], 0)
        self.assertTrue(all(row["differenceCi95"] >= 0 for row in result["actions"]))
        if result["cudaAvailable"]:
            self.assertEqual(result["gpuTrainingWorkers"], 1)
            self.assertGreater(result["gpuTrainingBatches"], 0)
            self.assertGreater(result["gpuTrainingHandEvaluations"], 0)
            self.assertGreater(result["gpuEvaluationHandEvaluations"], 0)

    def test_fold_is_zero_from_the_current_decision_point(self):
        info = river_info(actor="p5", pending=("p5",), current_bet=10, pot=30)
        info["players"][3]["streetCommitted"] = 10
        info["players"][3]["totalCommitted"] = 20
        info["players"][3]["stack"] = 80
        result = solve_decision(info, iterations=500)
        actions = {row["action"]: row for row in result["actions"]}
        self.assertEqual(actions["fold"]["ev"], 0)
        self.assertEqual(actions["fold"]["ci95"], 0)
        self.assertAlmostEqual(sum(row["frequency"] for row in result["actions"]), 1.0)


if __name__ == "__main__":
    unittest.main()
