import unittest
from concurrent.futures import Future, ThreadPoolExecutor
import threading
from unittest.mock import patch

import server
from tests.test_schema import permute_suits, sample_info


class SolveCacheTests(unittest.TestCase):
    def setUp(self):
        server.clear_solve_cache()

    def test_exact_request_is_cached_without_sharing_mutable_results(self):
        info = {"street": "river", "board": [1, 2, 3, 4, 5], "hero": {"id": "p1"}}
        solved = {"actions": [{"action": "check", "ev": 1.25}]}

        with patch.object(server, "SOLVE_CACHE_SIZE", 64), patch.object(
            server,
            "solve_decision",
            return_value=solved,
        ) as solve:
            first = server.solve_decision_cached(info, 500)
            first["actions"][0]["ev"] = 99
            reordered = {"hero": {"id": "p1"}, "board": [1, 2, 3, 4, 5], "street": "river"}
            second = server.solve_decision_cached(reordered, 500)

        self.assertEqual(solve.call_count, 1)
        self.assertFalse(first["cacheHit"])
        self.assertTrue(second["cacheHit"])
        self.assertEqual(first["solveSource"], "live-solve")
        self.assertEqual(second["solveSource"], "memory-cache")
        self.assertGreaterEqual(first["solveElapsedMs"], 0)
        self.assertEqual(second["actions"][0]["ev"], 1.25)

    def test_suit_isomorphic_requests_share_the_same_cache_entry(self):
        info = sample_info()
        permuted = permute_suits(info, (1, 2, 3, 0))
        solved = {"actions": [{"action": "check", "ev": 1.25}]}

        with patch.object(server, "SOLVE_CACHE_SIZE", 64), patch.object(
            server,
            "solve_decision",
            return_value=solved,
        ) as solve:
            first = server.solve_decision_cached(info, 500)
            second = server.solve_decision_cached(permuted, 500)

        self.assertEqual(solve.call_count, 1)
        self.assertFalse(first["cacheHit"])
        self.assertTrue(second["cacheHit"])

    def test_iterations_and_information_set_are_part_of_cache_key(self):
        info = {"street": "river", "board": [1, 2, 3, 4, 5]}

        with patch.object(server, "SOLVE_CACHE_SIZE", 64), patch.object(
            server,
            "solve_decision",
            side_effect=lambda request, iterations: {"iterations": iterations},
        ) as solve:
            server.solve_decision_cached(info, 500)
            server.solve_decision_cached(info, 501)
            server.solve_decision_cached({**info, "pot": 10}, 500)

        self.assertEqual(solve.call_count, 3)

    def test_artifact_fingerprints_are_part_of_cache_key(self):
        info = {"street": "river", "board": [1, 2, 3, 4, 5]}
        leaf_fingerprint = {"value": "leaf-a"}

        with patch.object(server, "SOLVE_CACHE_SIZE", 64), patch.object(
            server,
            "runtime_signature",
            return_value="blueprint-a",
        ), patch.object(
            server,
            "leaf_runtime_signature",
            side_effect=lambda: leaf_fingerprint["value"],
        ), patch.object(
            server,
            "solve_decision",
            side_effect=lambda request, iterations: {"iterations": iterations},
        ) as solve:
            server.solve_decision_cached(info, 500)
            server.solve_decision_cached(info, 500)
            leaf_fingerprint["value"] = "leaf-b"
            server.solve_decision_cached(info, 500)

        self.assertEqual(solve.call_count, 2)

    def test_solver_configuration_is_part_of_cache_key(self):
        info = {"street": "river", "board": [1, 2, 3, 4, 5]}
        config = {"value": "config-a"}
        with patch.object(server, "SOLVE_CACHE_SIZE", 64), patch.object(
            server,
            "solver_config_signature",
            side_effect=lambda: config["value"],
        ), patch.object(
            server,
            "solve_decision",
            side_effect=lambda request, iterations: {"iterations": iterations},
        ) as solve:
            server.solve_decision_cached(info, 500)
            server.solve_decision_cached(info, 500)
            config["value"] = "config-b"
            server.solve_decision_cached(info, 500)

        self.assertEqual(solve.call_count, 2)

    def test_result_from_a_different_runtime_is_never_cached_under_old_key(self):
        info = {"street": "river", "board": [1, 2, 3, 4, 5]}
        runtime = {
            "blueprint": "blueprint-a",
            "leaf": "leaf-a",
            "solverConfig": "config-a",
        }
        stale_result = {
            "blueprintFingerprint": "blueprint-b",
            "leafModelFingerprint": "leaf-a",
            "solverConfigFingerprint": "config-a",
        }

        with patch.object(server, "SOLVE_CACHE_SIZE", 64), patch.object(
            server,
            "runtime_identity",
            return_value=runtime,
        ), patch.object(
            server,
            "solve_decision",
            return_value=stale_result,
        ) as solve:
            first = server.solve_decision_cached(info, 500)
            second = server.solve_decision_cached(info, 500)

        self.assertEqual(solve.call_count, 2)
        self.assertFalse(first["cacheHit"])
        self.assertFalse(second["cacheHit"])

    def test_runtime_publication_during_solve_does_not_poison_old_cache_key(self):
        info = {"street": "turn", "board": [1, 2, 3, 4]}
        active = {"blueprint": "blueprint-a"}

        def identity():
            return {
                "blueprint": active["blueprint"],
                "leaf": "leaf-a",
                "solverConfig": "config-a",
            }

        def solve_and_publish(_request, _iterations):
            active["blueprint"] = "blueprint-b"
            return {
                "blueprintFingerprint": "blueprint-a",
                "leafModelFingerprint": "leaf-a",
                "solverConfigFingerprint": "config-a",
            }

        with patch.object(server, "SOLVE_CACHE_SIZE", 64), patch.object(
            server,
            "runtime_identity",
            side_effect=identity,
        ), patch.object(
            server,
            "solve_decision",
            side_effect=solve_and_publish,
        ) as solve:
            first = server.solve_decision_cached(info, 500)
            second = server.solve_decision_cached(info, 500)

        self.assertEqual(solve.call_count, 2)
        self.assertFalse(first["cacheHit"])
        self.assertFalse(second["cacheHit"])

    def test_concurrent_identical_requests_share_one_solve(self):
        solve_started = threading.Event()
        release_solve = threading.Event()
        waiter_started = threading.Event()

        class TrackingFuture(Future):
            def result(self, timeout=None):
                waiter_started.set()
                return super().result(timeout)

        def slow_solve(info, iterations):
            solve_started.set()
            self.assertTrue(release_solve.wait(2))
            return {"iterations": iterations}

        info = {"street": "turn", "board": [1, 2, 3, 4]}
        with patch.object(server, "SOLVE_CACHE_SIZE", 64), patch.object(
            server,
            "Future",
            TrackingFuture,
        ), patch.object(server, "solve_decision", side_effect=slow_solve) as solve:
            with ThreadPoolExecutor(max_workers=2) as pool:
                owner = pool.submit(server.solve_decision_cached, info, 500)
                self.assertTrue(solve_started.wait(2))
                waiter = pool.submit(server.solve_decision_cached, info, 500)
                self.assertTrue(waiter_started.wait(2))
                release_solve.set()
                owner_result = owner.result(2)
                waiter_result = waiter.result(2)

        self.assertEqual(solve.call_count, 1)
        self.assertFalse(owner_result["cacheHit"])
        self.assertTrue(waiter_result["cacheHit"])
        self.assertEqual(waiter_result["solveSource"], "shared-inflight")

    def test_exact_blueprint_result_is_stamped_for_the_active_runtime(self):
        info = {"street": "preflop", "board": [], "hero": {"id": "p1"}}
        runtime = {
            "blueprint": "blueprint-production",
            "leaf": "none",
            "solverConfig": "online-rmplus",
        }
        exact = {
            "actions": [{"action": "fold", "ev": 0.0}],
            "blueprintExactHit": True,
            "blueprintFingerprint": "teacher-runtime",
            "solverConfigFingerprint": "teacher-linear",
        }
        blueprint = type(
            "ExactBlueprint",
            (),
            {"lookup_exact": staticmethod(lambda request, iterations: exact)},
        )()
        with patch.object(server, "SOLVE_CACHE_SIZE", 64), patch.object(
            server,
            "runtime_identity",
            return_value=runtime,
        ), patch.object(
            server,
            "get_blueprint_runtime",
            return_value=blueprint,
        ), patch.object(server, "solve_decision") as solve:
            first = server.solve_decision_cached(info, 20_000)
            second = server.solve_decision_cached(info, 20_000)

        solve.assert_not_called()
        self.assertEqual(first["blueprintFingerprint"], runtime["blueprint"])
        self.assertEqual(first["solverConfigFingerprint"], runtime["solverConfig"])
        self.assertFalse(first["cacheHit"])
        self.assertTrue(second["cacheHit"])
        self.assertEqual(first["solveSource"], "blueprint-exact")
        self.assertEqual(second["solveSource"], "memory-cache")


if __name__ == "__main__":
    unittest.main()
