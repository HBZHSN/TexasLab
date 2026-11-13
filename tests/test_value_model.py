from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from backend.belief import ALL_COMBOS, COMBO_INDEX, ComboRange
from backend.value_model import (
    FEATURE_SIZE,
    HAND_COUNT,
    MODEL_TYPE,
    POSITIONS,
    LeafArtifactError,
    LeafArtifactNotApproved,
    LeafModelRuntime,
    LeafModelInput,
    LeafSchemaError,
    LeafTrainingSample,
    PortableLeafValueModel,
    build_approval_attestation,
    build_leaf_features,
    dataset_header,
    load_manifest,
    metrics_fingerprint,
    read_leaf_dataset,
    strict_json_loads,
    write_leaf_artifact,
    write_leaf_dataset,
)


def public_state(*, street: str = "preflop", board: list[int] | None = None) -> dict:
    current_board = list(board or [])
    players = []
    for index, position in enumerate(POSITIONS):
        committed = 0.5 if position == "SB" else 1.0 if position == "BB" else 0.0
        players.append(
            {
                "id": f"p{index}",
                "position": position,
                "stack": 100.0 - committed,
                "streetCommitted": committed if street == "preflop" else 0.0,
                "totalCommitted": committed,
                "folded": False,
                "allIn": False,
            }
        )
    return {
        "street": street,
        "board": current_board,
        "pot": 1.5,
        "currentBet": 1.0 if street == "preflop" else 0.0,
        "aggressiveActions": 0,
        "players": players,
        "pending": [f"p{index}" for index in range(6)],
        "history": [],
    }


def model_input(*, street: str = "preflop", board: list[int] | None = None) -> LeafModelInput:
    state = public_state(street=street, board=board)
    probabilities = list(ComboRange.uniform(state["board"]).probabilities)
    return LeafModelInput.from_dict(
        {
            "publicState": state,
            "ranges": [probabilities for _ in range(6)],
            "continuationIds": [0, 1, 2, 3, 0, 1],
        }
    )


def sample_payload() -> dict:
    query = model_input()
    values = [[0.0] * HAND_COUNT for _ in range(6)]
    stderr = [[0.0] * HAND_COUNT for _ in range(6)]
    counts = [[0] * HAND_COUNT for _ in range(6)]
    for seat in range(6):
        values[seat][seat] = 100.0
        stderr[seat][seat] = 0.01
        counts[seat][seat] = 100
    return {
        "sampleId": "teacher-node-1",
        **query.to_dict(),
        "finalStackValues": values,
        "targetStdErr": stderr,
        "targetCounts": counts,
    }


def network_payload() -> dict:
    return {
        "modelType": MODEL_TYPE,
        "inputSize": FEATURE_SIZE,
        "outputSize": 2,
        "normalization": {
            "mean": [0.0] * FEATURE_SIZE,
            "scale": [1.0] * FEATURE_SIZE,
        },
        "layers": [
            {
                "weights": [[0.0] * FEATURE_SIZE, [0.0] * FEATURE_SIZE],
                "bias": [0.0, -10.0],
                "activation": "linear",
            }
        ],
    }


def passing_metrics() -> dict:
    return {
        "heldoutExamples": 20_000,
        "heldoutPublicStates": 1_000,
        "maeBb": 0.02,
        "p99AbsoluteErrorBb": 0.10,
        "coverage95": 0.95,
        "meanUncertaintyBb": 0.04,
        "maxGroupMaeBb": 0.04,
        "decisiveNodes": 2_000,
        "actionAgreement": 0.97,
        "strategyL1Median": 0.08,
        "strategyL1P95": 0.10,
        "actionEvRelativeErrorP95": 0.04,
        "oodFallbackRecall": 0.97,
        "leafActivatedNodes": 750,
        "leafActivationRate": 0.97,
    }


def support_payload() -> dict:
    return {
        "streets": ["preflop"],
        "activeCounts": [6],
        "continuationProfiles": [[0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 0, 1]],
        "minimumSupportedCombos": 1_000,
        "maxInferenceUncertaintyBb": 0.20,
        "potFraction": [0.0, 0.01],
        "currentBetFraction": [0.0, 0.01],
        "totalChipsBb": [599.0, 601.0],
        "stackFraction": [0.15, 0.18],
        "rangeEntropyNormalized": [0.99, 1.0],
        "rangeEssFraction": [0.99, 1.0],
    }


def training_payload(
    *,
    blueprint_fingerprint: str = "blueprint-test-v1",
    teacher_version: str = "teacher-test-v1",
) -> dict:
    return {
        "teacherVersion": teacher_version,
        "blueprintFingerprint": blueprint_fingerprint,
        "datasetHeaders": [
            {
                "teacherVersion": teacher_version,
                "blueprintVersion": blueprint_fingerprint,
            }
        ],
    }


def write_approved_artifact(
    directory: str,
    *,
    network: dict | None = None,
    metrics: dict | None = None,
    support: dict | None = None,
    blueprint_fingerprint: str = "blueprint-test-v1",
    holdout_sha256: str = "a" * 64,
) -> Path:
    selected_network = network or network_payload()
    selected_metrics = metrics or passing_metrics()
    selected_support = support or support_payload()
    training = training_payload(blueprint_fingerprint=blueprint_fingerprint)
    candidate = write_leaf_artifact(
        directory,
        selected_network,
        selected_metrics,
        training,
        support=selected_support,
    )
    candidate_manifest, _ = load_manifest(candidate)
    attestation = build_approval_attestation(
        payload_fingerprint=candidate_manifest["payloadFingerprint"],
        blueprint_fingerprint=blueprint_fingerprint,
        holdout_sha256=holdout_sha256,
        metrics=selected_metrics,
        teacher_version=training["teacherVersion"],
        continuation_profiles=selected_support["continuationProfiles"],
    )
    return write_leaf_artifact(
        directory,
        selected_network,
        selected_metrics,
        training,
        support=selected_support,
        approval_attestation=attestation,
    )


class LeafValueModelTests(unittest.TestCase):
    def test_manifest_publish_failure_preserves_previous_complete_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            first_manifest = write_approved_artifact(temporary)
            before_manifest = first_manifest.read_bytes()
            before_files = {path.name: path.read_bytes() for path in Path(temporary).iterdir()}
            changed_network = network_payload()
            changed_network["layers"][0]["bias"][0] = 1.0
            original_replace = os.replace

            def publish(source, destination):
                if Path(destination).name == "manifest.json":
                    raise OSError("manifest publish failed")
                return original_replace(source, destination)

            with patch("backend.value_model.os.replace", side_effect=publish):
                with self.assertRaisesRegex(OSError, "manifest publish failed"):
                    write_leaf_artifact(
                        temporary,
                        changed_network,
                        passing_metrics(),
                        training_payload(teacher_version="second"),
                        support=support_payload(),
                    )

            after_files = {path.name: path.read_bytes() for path in Path(temporary).iterdir()}
            after_manifest = first_manifest.read_bytes()
            PortableLeafValueModel.load(first_manifest)

        self.assertEqual(after_manifest, before_manifest)
        self.assertEqual(after_files, before_files)

    def test_generation_cli_does_not_destroy_existing_output_on_empty_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "empty.jsonl"
            output = Path(temporary) / "leaf-shard.jsonl.gz"
            source.write_text("", encoding="utf-8")
            output.write_bytes(b"previous-artifact")
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/generate-leaf-data.py",
                    str(source),
                    "--output",
                    str(output),
                    "--teacher-version",
                    "teacher-test-v1",
                    "--blueprint-version",
                    "blueprint-test-v1",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            preserved = output.read_bytes()

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(preserved, b"previous-artifact")

    def test_generation_cli_validates_plain_teacher_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "teacher.jsonl"
            output = Path(temporary) / "leaf-shard.jsonl.gz"
            source.write_text(json.dumps(sample_payload(), separators=(",", ":")) + "\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/generate-leaf-data.py",
                    str(source),
                    "--output",
                    str(output),
                    "--teacher-version",
                    "teacher-test-v1",
                    "--blueprint-version",
                    "blueprint-test-v1",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            )
            header, samples = read_leaf_dataset(output)
            loaded = list(samples)

        self.assertIn('"samples": 1', result.stdout)
        self.assertEqual(header["metadata"]["targetSemantics"], "conditional-final-stack-by-seat-and-hand")
        self.assertEqual(len(loaded), 1)

    def test_sample_schema_round_trips_safe_jsonl_gzip(self):
        sample = LeafTrainingSample.from_dict(sample_payload())
        header = dataset_header(
            teacher_version="teacher-test-v1",
            blueprint_version="blueprint-test-v1",
            metadata={"seed": 17},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "leaf-shard-000.jsonl.gz"
            write_leaf_dataset(path, header, [sample])
            loaded_header, samples = read_leaf_dataset(path)
            loaded = list(samples)

        self.assertEqual(loaded_header["teacherVersion"], "teacher-test-v1")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].sample_id, sample.sample_id)
        self.assertEqual(loaded[0].target_counts[3][3], 100)
        self.assertEqual(loaded[0].model_input.continuation_ids, (0, 1, 2, 3, 0, 1))

    def test_schema_rejects_private_history_and_board_conflicting_ranges(self):
        private = sample_payload()
        private["publicState"]["history"].append(
            {
                "street": "preflop",
                "playerId": "p0",
                "type": "call",
                "heroCards": [1, 2],
            }
        )
        with self.assertRaisesRegex(LeafSchemaError, "私有牌"):
            LeafTrainingSample.from_dict(private)

        state = public_state(street="flop", board=[0, 5, 10])
        invalid_range = list(ComboRange.uniform(state["board"]).probabilities)
        invalid_range[COMBO_INDEX[(0, 1)]] = 0.0001
        invalid_range[COMBO_INDEX[(1, 2)]] -= 0.0001
        with self.assertRaisesRegex(LeafSchemaError, "公共牌冲突"):
            LeafModelInput.from_dict(
                {
                    "publicState": state,
                    "ranges": [invalid_range] * 6,
                    "continuationIds": [0] * 6,
                }
            )

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(self):
        with self.assertRaisesRegex(LeafSchemaError, "重复字段"):
            strict_json_loads('{"sampleId":"a","sampleId":"b"}')
        with self.assertRaisesRegex(LeafSchemaError, "非法常量"):
            strict_json_loads('{"value":NaN}')

    def test_features_are_stable_and_preflop_suit_isomorphic(self):
        query = model_input()
        first_index = COMBO_INDEX[tuple(sorted((0, 4)))]
        second_index = COMBO_INDEX[tuple(sorted((1, 5)))]
        first = build_leaf_features(query, 0, first_index)
        second = build_leaf_features(query, 0, second_index)

        self.assertEqual(len(first), FEATURE_SIZE)
        self.assertEqual(first, second)

    def test_approved_artifact_loads_and_predicts_by_seat_and_hand(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = write_approved_artifact(temporary)
            model = PortableLeafValueModel.load(manifest_path)
            prediction = model.predict(model_input(), 0, 0)
            prediction_by_hand = model.predict_hand(model_input(), 0, ALL_COMBOS[0])

        self.assertTrue(model.approved)
        self.assertAlmostEqual(prediction.final_stack, 300.0)
        self.assertAlmostEqual(prediction.final_stack, prediction_by_hand.final_stack)
        self.assertGreater(prediction.uncertainty, 0)
        self.assertLess(prediction.uncertainty, 0.1)

    def test_runtime_exposes_fingerprint_and_cutoff_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = write_approved_artifact(temporary)
            runtime = LeafModelRuntime.load(manifest_path)
            gate = runtime.cutoff_gate(model_input())
            prediction = runtime.predict(model_input(), 0, 0)

        self.assertTrue(gate.allowed)
        self.assertEqual(gate.fingerprint, runtime.fingerprint)
        self.assertEqual(gate.model_error_bb, 0.10)
        self.assertAlmostEqual(prediction.final_stack, 300.0)

    def test_training_artifact_with_naked_passing_metrics_cannot_be_enabled(self):
        metrics = passing_metrics()
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = write_leaf_artifact(
                temporary,
                network_payload(),
                metrics,
                training_payload(),
                support=support_payload(),
            )
            with self.assertRaisesRegex(LeafArtifactNotApproved, "未达到启用门槛"):
                PortableLeafValueModel.load(manifest_path)
            diagnostic_model = PortableLeafValueModel.load(manifest_path, require_approved=False)

        self.assertFalse(diagnostic_model.approved)
        self.assertEqual(
            diagnostic_model.manifest["gateFailures"],
            ["production-approval-attestation-missing"],
        )

    def test_continuation_profile_must_be_inside_training_support(self):
        restricted_support = support_payload()
        restricted_support["continuationProfiles"] = [[0, 0, 0, 0, 0, 0]]
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = write_approved_artifact(
                temporary,
                support=restricted_support,
            )
            runtime = LeafModelRuntime.load(manifest_path)
            gate = runtime.cutoff_gate(model_input())

        self.assertFalse(gate.allowed)
        self.assertIn("continuation-profile-out-of-distribution", gate.reasons)

    def test_approval_binds_payload_blueprint_holdout_config_and_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = write_approved_artifact(temporary)
            manifest, _ = load_manifest(manifest_path)
            attestation = manifest["approvalAttestation"]

        self.assertEqual(attestation["payloadFingerprint"], manifest["payloadFingerprint"])
        self.assertEqual(attestation["blueprintFingerprint"], "blueprint-test-v1")
        self.assertEqual(attestation["teacherVersion"], "teacher-test-v1")
        self.assertEqual(attestation["holdoutSha256"], "a" * 64)
        self.assertEqual(attestation["metricsSha256"], metrics_fingerprint(manifest["metrics"]))
        self.assertIn("solverConfig", attestation)
        self.assertIn("cutoffConfig", attestation)
        self.assertIn("continuationConfig", attestation)

    def test_same_weights_with_different_approved_error_metrics_change_runtime_fingerprint(self):
        first_metrics = passing_metrics()
        second_metrics = passing_metrics()
        second_metrics["p99AbsoluteErrorBb"] = 0.20
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_runtime = LeafModelRuntime.load(
                write_approved_artifact(first, metrics=first_metrics)
            )
            second_runtime = LeafModelRuntime.load(
                write_approved_artifact(second, metrics=second_metrics)
            )

        self.assertNotEqual(first_runtime.fingerprint, second_runtime.fingerprint)
        self.assertEqual(first_runtime.model.payload_fingerprint, second_runtime.model.payload_fingerprint)
        self.assertEqual(first_runtime.model_error_bb, 0.10)
        self.assertEqual(second_runtime.model_error_bb, 0.20)

    def test_approval_attestation_cannot_be_reused_with_changed_metrics(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as output:
            approved = write_approved_artifact(source)
            manifest, _ = load_manifest(approved)
            changed = passing_metrics()
            changed["p99AbsoluteErrorBb"] = 0.20

            with self.assertRaisesRegex(LeafArtifactError, "metricsSha256"):
                write_leaf_artifact(
                    output,
                    network_payload(),
                    changed,
                    training_payload(),
                    support=support_payload(),
                    approval_attestation=manifest["approvalAttestation"],
                )

    def test_artifact_sha256_detects_model_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = write_approved_artifact(temporary)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            model_path = Path(temporary) / manifest["modelFile"]
            model_path.write_bytes(model_path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(LeafArtifactError, "大小|SHA-256"):
                PortableLeafValueModel.load(manifest_path)


if __name__ == "__main__":
    unittest.main()
