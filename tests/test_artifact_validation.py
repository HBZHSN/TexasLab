from __future__ import annotations

from pathlib import Path
import runpy
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from backend.value_model import (
    PRODUCTION_SOLVER_CONFIG,
    LeafArtifactNotApproved,
    LeafModelRuntime,
    PortableLeafValueModel,
    load_manifest,
    write_leaf_artifact,
)
from tests.test_value_model import (
    network_payload,
    passing_metrics,
    support_payload,
    training_payload,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate-artifacts.py"


def validator_namespace() -> dict:
    return runpy.run_path(str(SCRIPT))


def candidate_artifact(directory: str) -> Path:
    metrics = passing_metrics()
    metrics.update(
        {
            "decisiveNodes": 0,
            "actionAgreement": 0.0,
            "strategyL1Median": 1.0,
            "strategyL1P95": 1.0,
            "actionEvRelativeErrorP95": 1.0,
            "oodFallbackRecall": 0.0,
            "leafActivatedNodes": 0,
            "leafActivationRate": 0.0,
        }
    )
    return write_leaf_artifact(
        directory,
        network_payload(),
        metrics,
        training_payload(),
        support=support_payload(),
    )


def passing_integration() -> dict:
    return {
        "decisiveNodes": 2_000,
        "actionAgreement": 0.97,
        "strategyL1Median": 0.08,
        "strategyL1P95": 0.10,
        "actionEvRelativeErrorP95": 0.04,
        "oodFallbackRecall": 0.97,
        "iidNodes": 775,
        "leafActivatedNodes": 750,
        "leafActivationRate": 750 / 775,
        "oodNodes": 225,
    }


class ArtifactValidationTests(unittest.TestCase):
    def test_unapproved_artifact_gets_token_bound_temporary_ab_candidate(self):
        prepare = validator_namespace()["validation_manifest"]
        with tempfile.TemporaryDirectory() as source_directory, tempfile.TemporaryDirectory() as candidate_directory:
            source = candidate_artifact(source_directory)
            candidate = prepare(
                source,
                Path(candidate_directory),
                blueprint_fingerprint="blueprint-test-v1",
            )
            with self.assertRaises(LeafArtifactNotApproved):
                LeafModelRuntime.load(candidate.manifest)
            runtime = LeafModelRuntime.load(
                candidate.manifest,
                ab_candidate_token=candidate.token,
            )

        self.assertFalse(runtime.model.approved)
        self.assertTrue(runtime.model.temporary_candidate)
        self.assertEqual(runtime.fingerprint, candidate.artifact_fingerprint)

    def test_integration_metrics_use_evaluation_hits_for_iid_and_ood(self):
        compute = validator_namespace()["compute_integration_metrics"]
        baseline = [
            {
                "actions": [
                    {"action": "fold", "ev": 0.0, "frequency": 0.8},
                    {"action": "call", "ev": -1.0, "frequency": 0.2},
                ]
            },
            {
                "actions": [
                    {"action": "check", "ev": 0.0, "frequency": 0.5},
                    {"action": "raise", "ev": 0.02, "frequency": 0.5},
                ]
            },
        ]
        accelerated = [
            {
                "leafCutoffHits": 100,
                "leafEvaluationCutoffHits": 3,
                "actions": [
                    {"action": "fold", "ev": 0.0, "frequency": 0.7},
                    {"action": "call", "ev": -0.8, "frequency": 0.3},
                ],
            },
            {
                # Training may have hit the leaf, but OOD safety is measured
                # only on the frozen evaluation cutoff path.
                "leafCutoffHits": 100,
                "leafEvaluationCutoffHits": 0,
                "actions": [
                    {"action": "check", "ev": 0.01, "frequency": 0.5},
                    {"action": "raise", "ev": 0.0, "frequency": 0.5},
                ],
            },
        ]
        rows = [
            {"expectLeafFallback": False},
            {"expectLeafFallback": True},
        ]

        metrics = compute(baseline, accelerated, rows, decisive_gap=0.1)

        self.assertEqual(metrics["decisiveNodes"], 1)
        self.assertEqual(metrics["actionAgreement"], 1.0)
        self.assertAlmostEqual(metrics["strategyL1Median"], 0.1)
        self.assertEqual(metrics["oodFallbackRecall"], 1.0)
        self.assertEqual(metrics["leafActivatedNodes"], 1)
        self.assertEqual(metrics["leafActivationRate"], 1.0)

    def test_iid_zero_activation_can_never_publish(self):
        publish = validator_namespace()["publish_measured_artifact"]
        integration = passing_integration()
        integration["leafActivatedNodes"] = 0
        integration["leafActivationRate"] = 0.0
        with tempfile.TemporaryDirectory() as source_directory, tempfile.TemporaryDirectory() as output_directory:
            source = candidate_artifact(source_directory)
            manifest, approved, failures = publish(
                source,
                Path(output_directory),
                integration,
                blueprint_fingerprint="blueprint-test-v1",
                holdout_sha256="b" * 64,
            )

        self.assertIsNone(manifest)
        self.assertFalse(approved)
        self.assertTrue(any("leafActivatedNodes" in failure or "零命中" in failure for failure in failures))

    def test_final_publication_contains_bound_attestation_and_loads_normally(self):
        publish = validator_namespace()["publish_measured_artifact"]
        with tempfile.TemporaryDirectory() as source_directory, tempfile.TemporaryDirectory() as output_directory:
            source = candidate_artifact(source_directory)
            manifest_path, approved, failures = publish(
                source,
                Path(output_directory),
                passing_integration(),
                blueprint_fingerprint="blueprint-test-v1",
                holdout_sha256="b" * 64,
            )
            self.assertIsNotNone(manifest_path)
            manifest, _ = load_manifest(manifest_path)
            model = PortableLeafValueModel.load(manifest_path)

        self.assertTrue(approved)
        self.assertEqual(failures, [])
        self.assertTrue(model.approved)
        self.assertEqual(
            manifest["approvalAttestation"]["payloadFingerprint"],
            manifest["payloadFingerprint"],
        )
        self.assertEqual(
            manifest["approvalAttestation"]["blueprintFingerprint"],
            "blueprint-test-v1",
        )
        self.assertEqual(manifest["approvalAttestation"]["holdoutSha256"], "b" * 64)

    def test_trainer_rejects_mixed_blueprint_or_teacher_versions(self):
        unique = runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "scripts" / "train-leaf-model.py")
        )["unique_training_versions"]
        with self.assertRaisesRegex(ValueError, "blueprintVersion"):
            unique(
                [
                    {"blueprintVersion": "one", "teacherVersion": "teacher"},
                    {"blueprintVersion": "two", "teacherVersion": "teacher"},
                ]
            )
        with self.assertRaisesRegex(ValueError, "teacherVersion"):
            unique(
                [
                    {"blueprintVersion": "one", "teacherVersion": "first"},
                    {"blueprintVersion": "one", "teacherVersion": "second"},
                ]
            )

    def test_production_solve_pass_checks_leaf_active_and_fingerprint(self):
        solve_pass = validator_namespace()["solve_pass"]
        fake_blueprint = SimpleNamespace(
            fingerprint="blueprint-test-v1",
            has_valid_policy=True,
        )
        wrong_result = {
            "blueprintFingerprint": "blueprint-test-v1",
            "leafModelFingerprint": "wrong-leaf",
            "leafCutoffWorkers": 1,
            "actions": [{"action": "check", "ev": 0.0, "frequency": 1.0}],
        }
        solve = Mock(return_value=wrong_result)
        globals_patch = {
            "get_blueprint_runtime": lambda: fake_blueprint,
            "reset_blueprint_runtime": lambda: None,
            "reset_leaf_runtime": lambda: None,
            "shutdown_process_pool": lambda: None,
            "solve_decision": solve,
        }
        with patch.dict(solve_pass.__globals__, globals_patch):
            with self.assertRaisesRegex(ValueError, "different leaf artifact fingerprint"):
                solve_pass(
                    [{"info": {}}],
                    blueprint=Path("blueprint.json"),
                    blueprint_fingerprint="blueprint-test-v1",
                    leaf_manifest=Path("manifest.json"),
                    expected_leaf_fingerprint="expected-leaf",
                    ab_token="x" * 32,
                )

        solve.assert_called_once_with(
            {},
            iterations=int(PRODUCTION_SOLVER_CONFIG["iterations"]),
        )


if __name__ == "__main__":
    unittest.main()
