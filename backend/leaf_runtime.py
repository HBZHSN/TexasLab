from __future__ import annotations

import os
import hashlib
from pathlib import Path
import threading

from .value_model import LeafArtifactError, LeafModelRuntime


_LOCK = threading.Lock()
_RUNTIME: LeafModelRuntime | None = None
_STAMP: tuple[object, ...] | None = None
_MODEL_PATH: Path | None = None
_LAST_ERROR: str | None = None


def _configured_manifest() -> Path | None:
    raw = os.environ.get("TEXAS_LAB_LEAF_MODEL_MANIFEST", "").strip()
    return Path(raw).resolve() if raw else None


def _path_stamp(path: Path | None) -> tuple[str, int, int] | None:
    if path is None or not path.exists():
        return None
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def get_leaf_runtime() -> LeafModelRuntime | None:
    """Hot-load an approved leaf artifact; auto mode fails closed."""
    global _RUNTIME, _STAMP, _MODEL_PATH, _LAST_ERROR
    mode = os.environ.get("TEXAS_LAB_LEAF_MODEL_MODE", "auto").strip().lower()
    if mode not in {"auto", "off", "required", "ab"}:
        raise LeafArtifactError("TEXAS_LAB_LEAF_MODEL_MODE must be auto, off, required or ab")
    if mode == "off":
        return None
    manifest_path = _configured_manifest()
    if manifest_path is None or not manifest_path.exists():
        if mode in {"required", "ab"}:
            raise LeafArtifactError("required leaf model manifest does not exist")
        return None

    ab_token = os.environ.get("TEXAS_LAB_LEAF_AB_TOKEN") if mode == "ab" else None
    if mode == "ab" and (not isinstance(ab_token, str) or len(ab_token) < 32):
        raise LeafArtifactError("A/B leaf model mode requires a random candidate token")
    credential_stamp = (
        hashlib.sha256(ab_token.encode("utf-8")).hexdigest()
        if ab_token is not None
        else None
    )

    stamp: tuple[object, ...] = (
        mode,
        credential_stamp,
        _path_stamp(manifest_path),
        _path_stamp(_MODEL_PATH),
    )
    with _LOCK:
        if _STAMP == stamp:
            if _RUNTIME is None and mode in {"required", "ab"}:
                raise LeafArtifactError(_LAST_ERROR or "leaf model failed to load")
            return _RUNTIME
        try:
            runtime = LeafModelRuntime.load(
                manifest_path,
                ab_candidate_token=ab_token,
            )
            model_file = runtime.model.manifest["modelFile"]
            model_path = (manifest_path.parent / model_file).resolve()
            runtime_stamp = (
                mode,
                credential_stamp,
                _path_stamp(manifest_path),
                _path_stamp(model_path),
            )
        except Exception as error:
            _RUNTIME = None
            _STAMP = stamp
            _LAST_ERROR = str(error)
            if mode in {"required", "ab"}:
                raise LeafArtifactError(f"leaf model load failed: {error}") from error
            return None
        _RUNTIME = runtime
        _MODEL_PATH = model_path
        _STAMP = runtime_stamp
        _LAST_ERROR = None
        return runtime


def leaf_runtime_signature() -> str:
    runtime = get_leaf_runtime()
    return "none" if runtime is None else runtime.fingerprint


def leaf_runtime_diagnostics() -> dict[str, object]:
    runtime = get_leaf_runtime()
    return {
        "leafModelLoaded": runtime is not None,
        "leafModelFingerprint": "none" if runtime is None else runtime.fingerprint,
        "leafModelLoadError": _LAST_ERROR,
    }


def reset_leaf_runtime() -> None:
    global _RUNTIME, _STAMP, _MODEL_PATH, _LAST_ERROR
    with _LOCK:
        _RUNTIME = None
        _STAMP = None
        _MODEL_PATH = None
        _LAST_ERROR = None
