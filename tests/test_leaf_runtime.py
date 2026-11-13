import os
import json
from pathlib import Path
import secrets
import tempfile
import unittest
from unittest.mock import patch

from backend.leaf_runtime import (
    get_leaf_runtime,
    leaf_runtime_signature,
    reset_leaf_runtime,
)
from backend.value_model import LeafArtifactError, write_leaf_artifact
from tests.test_value_model import (
    network_payload,
    passing_metrics,
    support_payload,
    training_payload,
    write_approved_artifact,
)


class LeafRuntimeLoaderTests(unittest.TestCase):
    def tearDown(self):
        reset_leaf_runtime()

    def test_approved_artifact_hot_loads_and_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_approved_artifact(temporary)
            environment = {
                "TEXAS_LAB_LEAF_MODEL_MANIFEST": str(manifest),
                "TEXAS_LAB_LEAF_MODEL_MODE": "auto",
            }
            with patch.dict(os.environ, environment, clear=False):
                reset_leaf_runtime()
                runtime = get_leaf_runtime()
                self.assertIsNotNone(runtime)
                self.assertEqual(leaf_runtime_signature(), runtime.fingerprint)

                manifest_payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
                model_path = Path(temporary) / manifest_payload["modelFile"]
                model_path.write_bytes(model_path.read_bytes() + b"tampered")
                self.assertIsNone(get_leaf_runtime())
                self.assertEqual(leaf_runtime_signature(), "none")

    def test_ab_candidate_requires_ab_mode_and_matching_random_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            token = secrets.token_urlsafe(32)
            manifest = write_leaf_artifact(
                temporary,
                network_payload(),
                passing_metrics(),
                training_payload(),
                support=support_payload(),
                ab_candidate_token=token,
            )
            base_environment = {
                "TEXAS_LAB_LEAF_MODEL_MANIFEST": str(manifest),
            }
            with patch.dict(
                os.environ,
                {**base_environment, "TEXAS_LAB_LEAF_MODEL_MODE": "auto"},
                clear=False,
            ):
                reset_leaf_runtime()
                self.assertIsNone(get_leaf_runtime())

            with patch.dict(
                os.environ,
                {**base_environment, "TEXAS_LAB_LEAF_MODEL_MODE": "required"},
                clear=False,
            ):
                reset_leaf_runtime()
                with self.assertRaises(LeafArtifactError):
                    get_leaf_runtime()

            with patch.dict(
                os.environ,
                {
                    **base_environment,
                    "TEXAS_LAB_LEAF_MODEL_MODE": "ab",
                    "TEXAS_LAB_LEAF_AB_TOKEN": secrets.token_urlsafe(32),
                },
                clear=False,
            ):
                reset_leaf_runtime()
                with self.assertRaises(LeafArtifactError):
                    get_leaf_runtime()

            with patch.dict(
                os.environ,
                {
                    **base_environment,
                    "TEXAS_LAB_LEAF_MODEL_MODE": "ab",
                    "TEXAS_LAB_LEAF_AB_TOKEN": token,
                },
                clear=False,
            ):
                reset_leaf_runtime()
                runtime = get_leaf_runtime()
                self.assertIsNotNone(runtime)
                self.assertTrue(runtime.model.temporary_candidate)
                self.assertFalse(runtime.model.approved)
                # Switching to a production mode must not reuse the cached
                # token-authorized runtime from the same process.
                os.environ["TEXAS_LAB_LEAF_MODEL_MODE"] = "required"
                os.environ.pop("TEXAS_LAB_LEAF_AB_TOKEN", None)
                with self.assertRaises(LeafArtifactError):
                    get_leaf_runtime()

    def test_manifest_metric_change_reloads_runtime_even_when_weights_match(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {
                "TEXAS_LAB_LEAF_MODEL_MANIFEST": str(Path(temporary) / "manifest.json"),
                "TEXAS_LAB_LEAF_MODEL_MODE": "required",
            },
            clear=False,
        ):
            write_approved_artifact(temporary)
            reset_leaf_runtime()
            first_signature = leaf_runtime_signature()
            changed = passing_metrics()
            changed["p99AbsoluteErrorBb"] = 0.20
            write_approved_artifact(temporary, metrics=changed)
            second_signature = leaf_runtime_signature()

        self.assertNotEqual(first_signature, second_signature)

    def test_ab_candidate_cannot_be_published_outside_configured_temp_root(self):
        with tempfile.TemporaryDirectory() as destination, tempfile.TemporaryDirectory() as other_root:
            with patch("backend.value_model.tempfile.gettempdir", return_value=other_root):
                with self.assertRaisesRegex(LeafArtifactError, "系统临时目录"):
                    write_leaf_artifact(
                        destination,
                        network_payload(),
                        passing_metrics(),
                        training_payload(),
                        support=support_payload(),
                        ab_candidate_token=secrets.token_urlsafe(32),
                    )

    def test_required_mode_rejects_missing_manifest(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {
                "TEXAS_LAB_LEAF_MODEL_MANIFEST": str(Path(temporary) / "missing.json"),
                "TEXAS_LAB_LEAF_MODEL_MODE": "required",
            },
            clear=False,
        ):
            reset_leaf_runtime()
            with self.assertRaises(LeafArtifactError):
                get_leaf_runtime()


if __name__ == "__main__":
    unittest.main()
