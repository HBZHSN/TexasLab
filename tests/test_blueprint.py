from __future__ import annotations

from copy import deepcopy
import gzip
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.blueprint import (
    FEATURE_SIZE,
    BlueprintError,
    DenseNetwork,
    bundle_fingerprint,
    encode_info_features,
    encode_state_features,
    encode_state_features_many_hands,
    get_blueprint_runtime,
    reset_blueprint_runtime,
    validate_bundle,
    write_bundle,
)
from backend.solver import make_state, run_replica


POSITIONS = ("UTG", "HJ", "CO", "BTN", "SB", "BB")


def sample_info() -> dict:
    players = []
    for index, position in enumerate(POSITIONS):
        active = index in (3, 5)
        players.append(
            {
                "id": f"p{index}",
                "position": position,
                "stack": 90.0 if active else 100.0,
                "streetCommitted": 0.0,
                "totalCommitted": 10.0 if active else 0.0,
                "folded": not active,
                "allIn": False,
            }
        )
    return {
        "street": "river",
        "board": [25, 12, 26, 5, 0],
        "pot": 20.0,
        "currentBet": 0.0,
        "aggressiveActions": 0,
        "hero": {
            "id": "p3",
            "position": "BTN",
            "stack": 90.0,
            "streetCommitted": 0.0,
            "hole": [31, 46],
        },
        "players": players,
        "pending": ["p3", "p5"],
        "history": [],
        "legalActions": ["check", "raise"],
        "raiseTo": 10.0,
    }


def minimal_bundle(marker: str = "first") -> dict:
    return {
        "validated": False,
        "metrics": {"marker": marker},
        "exactSolutions": {},
        "policyModel": None,
        "valueModel": None,
        "leafModel": None,
    }


def read_bundle(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    if opener is gzip.open:
        with opener(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class BlueprintFeatureTests(unittest.TestCase):
    def test_information_set_features_are_139_finite_values(self):
        features = encode_info_features(sample_info())

        self.assertEqual(FEATURE_SIZE, 139)
        self.assertEqual(len(features), FEATURE_SIZE)
        self.assertTrue(all(math.isfinite(value) for value in features))

    def test_many_hand_encoder_matches_scalar_encoder(self):
        info = sample_info()
        state = make_state(info)
        hands = [(31, 46), (32, 47), (33, 48)]
        batched = encode_state_features_many_hands(
            state,
            "p3",
            hands,
            root_history=info["history"],
            root_street=info["street"],
        )
        scalar = [
            encode_state_features(
                state,
                {"p3": hand},
                "p3",
                root_history=info["history"],
                root_street=info["street"],
            )
            for hand in hands
        ]

        self.assertEqual(batched, scalar)

    def test_dense_network_runs_portably_without_torch(self):
        network = DenseNetwork(
            {
                "inputSize": 2,
                "outputSize": 1,
                "normalization": {"mean": [1.0, 2.0], "scale": [2.0, 0.0]},
                "layers": [
                    {
                        "weights": [[2.0, 1.0], [-1.0, 1.0]],
                        "bias": [-1.0, 2.0],
                        "activation": "relu",
                    },
                    {
                        "weights": [[1.0, -2.0]],
                        "bias": [0.5],
                        "activation": "tanh",
                    },
                ],
            }
        )

        self.assertAlmostEqual(network.predict([3.0, 2.0])[0], math.tanh(-0.5))
        batched = network.predict_many([[3.0, 2.0], [1.0, 2.0]])
        self.assertAlmostEqual(batched[0][0], network.predict([3.0, 2.0])[0])
        self.assertAlmostEqual(batched[1][0], network.predict([1.0, 2.0])[0])
        with self.assertRaises(BlueprintError):
            network.predict([3.0])

    def test_dense_network_rejects_non_finite_weights(self):
        with self.assertRaises(BlueprintError):
            DenseNetwork(
                {
                    "inputSize": 1,
                    "outputSize": 1,
                    "normalization": {"mean": [0.0], "scale": [1.0]},
                    "layers": [
                        {
                            "weights": [[float("nan")]],
                            "bias": [0.0],
                            "activation": "linear",
                        }
                    ],
                }
            )


class BlueprintBundleTests(unittest.TestCase):
    def tearDown(self):
        reset_blueprint_runtime()

    def test_bundle_write_is_fingerprinted_and_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blueprint.json.gz"
            first_fingerprint = write_bundle(path, minimal_bundle("first"))
            first_payload = read_bundle(path)

            self.assertEqual(first_payload["fingerprint"], first_fingerprint)
            self.assertEqual(bundle_fingerprint(first_payload), first_fingerprint)
            validate_bundle(first_payload)

            # A failed publication must leave the prior complete artifact in place.
            with patch("backend.blueprint.os.replace", side_effect=OSError("publish failed")):
                with self.assertRaises(OSError):
                    write_bundle(path, minimal_bundle("second"))

            self.assertEqual(read_bundle(path), first_payload)
            self.assertFalse(
                any(item.name.startswith(".blueprint-") for item in path.parent.iterdir())
            )

            second_fingerprint = write_bundle(path, minimal_bundle("second"))
            self.assertNotEqual(second_fingerprint, first_fingerprint)
            self.assertEqual(read_bundle(path)["fingerprint"], second_fingerprint)

    def test_tampered_fingerprint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blueprint.json"
            write_bundle(path, minimal_bundle())
            payload = read_bundle(path)
            tampered = deepcopy(payload)
            tampered["metrics"]["marker"] = "tampered"

            with self.assertRaises(BlueprintError):
                validate_bundle(tampered)

    def test_scalar_leaf_model_cannot_be_smuggled_into_blueprint_bundle(self):
        payload = minimal_bundle()
        payload["leafModel"] = {"network": {"outputSize": 1}}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(BlueprintError, "单标量|信念条件"):
                write_bundle(Path(directory) / "unsafe.json", payload)

    def test_validation_flags_must_be_real_booleans(self):
        payload = minimal_bundle()
        payload["validated"] = "false"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(BlueprintError, "validated"):
                write_bundle(Path(directory) / "invalid.json.gz", payload)

    def test_corrupt_bundle_falls_back_in_auto_mode_and_fails_when_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text("{not-json", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "TEXAS_LAB_BLUEPRINT_PATH": str(path),
                    "TEXAS_LAB_BLUEPRINT_MODE": "auto",
                },
            ):
                reset_blueprint_runtime()
                runtime = get_blueprint_runtime()
                self.assertFalse(runtime.loaded)
                self.assertEqual(runtime.fingerprint, "none")

            with patch.dict(
                os.environ,
                {
                    "TEXAS_LAB_BLUEPRINT_PATH": str(path),
                    "TEXAS_LAB_BLUEPRINT_MODE": "required",
                },
            ):
                reset_blueprint_runtime()
                with self.assertRaises(BlueprintError):
                    get_blueprint_runtime()

    def test_validated_policy_bundle_warm_starts_real_solver(self):
        network = {
            "inputSize": FEATURE_SIZE,
            "outputSize": 4,
            "normalization": {
                "mean": [0.0] * FEATURE_SIZE,
                "scale": [1.0] * FEATURE_SIZE,
            },
            "layers": [
                {
                    "weights": [[0.0] * FEATURE_SIZE for _ in range(4)],
                    "bias": [0.0, 0.0, 0.0, 1.0],
                    "activation": "linear",
                }
            ],
        }
        payload = minimal_bundle()
        payload["validated"] = True
        payload["policyModel"] = {
            "kind": "acting-player-policy-v1",
            "validated": True,
            "trainingSamples": 1_000,
            "network": network,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blueprint.json.gz"
            fingerprint = write_bundle(path, payload)
            with patch.dict(
                os.environ,
                {
                    "TEXAS_LAB_BLUEPRINT_PATH": str(path),
                    "TEXAS_LAB_BLUEPRINT_MODE": "required",
                    "TEXAS_LAB_LEAF_MODEL_MODE": "off",
                },
            ):
                reset_blueprint_runtime()
                result = run_replica(
                    sample_info(),
                    training_iterations=20,
                    evaluation_iterations=20,
                    seed_nonce=7,
                )

        self.assertEqual(result["blueprintFingerprint"], fingerprint)
        self.assertGreater(result["blueprintPolicySeeds"], 0)
        self.assertEqual(result["blueprintPolicyFailures"], 0)

    def test_street_scoped_policy_does_not_seed_unsupported_streets(self):
        network = {
            "inputSize": FEATURE_SIZE,
            "outputSize": 4,
            "normalization": {
                "mean": [0.0] * FEATURE_SIZE,
                "scale": [1.0] * FEATURE_SIZE,
            },
            "layers": [
                {
                    "weights": [[0.0] * FEATURE_SIZE for _ in range(4)],
                    "bias": [0.0] * 4,
                    "activation": "linear",
                }
            ],
        }
        payload = minimal_bundle()
        payload["validated"] = True
        payload["policyModel"] = {
            "kind": "acting-player-policy-v1",
            "validated": True,
            "trainingSamples": 1_000,
            "supportedStreets": ["preflop"],
            "network": network,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blueprint.json.gz"
            write_bundle(path, payload)
            with patch.dict(
                os.environ,
                {
                    "TEXAS_LAB_BLUEPRINT_PATH": str(path),
                    "TEXAS_LAB_BLUEPRINT_MODE": "required",
                },
            ):
                reset_blueprint_runtime()
                runtime = get_blueprint_runtime()
                river_features = encode_info_features(sample_info())
                preflop_features = list(river_features)
                preflop_features[:4] = [1.0, 0.0, 0.0, 0.0]

                self.assertIsNone(runtime.policy(river_features, ("check", "raise")))
                self.assertIsNotNone(runtime.policy(preflop_features, ("fold", "call", "raise")))


if __name__ == "__main__":
    unittest.main()
