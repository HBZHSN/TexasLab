from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from typing import Any

from .abstraction import ABSTRACTION_VERSION, RULES_HASH, stable_public_key
from .schema import normalize_solve_info, public_history


BLUEPRINT_FORMAT = "texas-lab-blueprint"
SCHEMA_VERSION = 1
FEATURE_VERSION = "public-acting-player-v1"
ACTION_ORDER = ("fold", "check", "call", "raise")
POSITIONS = ("UTG", "HJ", "CO", "BTN", "SB", "BB")
STREETS = ("preflop", "flop", "turn", "river")
ACTION_CODES = {"fold": 0, "check": 1, "call": 2, "raise": 3}
PATH_ACTIONS = {1: "check", 2: "call", 3: "raise", 4: "fold"}
FEATURE_SIZE = 139


class BlueprintError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_info(info: dict[str, Any]) -> str:
    return stable_public_key(info)[1]


def stable_info_key(info: dict[str, Any]) -> tuple[str, str]:
    return stable_public_key(info)


def _bundle_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in bundle.items() if key != "fingerprint"}


def bundle_fingerprint(bundle: dict[str, Any]) -> str:
    payload = _canonical_json(_bundle_payload(bundle)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _open_text(path: Path, mode: str):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def write_bundle(path: str | Path, bundle: dict[str, Any]) -> str:
    """Validate, fingerprint and atomically publish a portable JSON bundle."""
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(bundle)
    payload["format"] = BLUEPRINT_FORMAT
    payload["schemaVersion"] = SCHEMA_VERSION
    payload["featureVersion"] = FEATURE_VERSION
    payload["abstractionVersion"] = ABSTRACTION_VERSION
    payload["rulesHash"] = RULES_HASH
    payload["fingerprint"] = bundle_fingerprint(payload)
    validate_bundle(payload, require_fingerprint=True)

    suffix = ".json.gz" if destination.suffix.lower() == ".gz" else ".json"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-",
        suffix=suffix,
        dir=destination.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with _open_text(temporary, "wt") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload["fingerprint"]


def _finite_vector(values: Any, length: int, label: str) -> list[float]:
    if not isinstance(values, list) or len(values) != length:
        raise BlueprintError(f"{label} 维度错误：期望 {length}")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise BlueprintError(f"{label} 包含非有限数值")
    return result


class DenseNetwork:
    """Small portable dense network; training may use Torch, inference does not."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.input_size = int(payload["inputSize"])
        self.output_size = int(payload["outputSize"])
        self.mean = _finite_vector(payload["normalization"]["mean"], self.input_size, "mean")
        self.scale = _finite_vector(payload["normalization"]["scale"], self.input_size, "scale")
        self.layers: list[tuple[list[list[float]], list[float], str]] = []
        width = self.input_size
        for layer_index, layer in enumerate(payload["layers"]):
            weights = layer["weights"]
            if not isinstance(weights, list) or not weights:
                raise BlueprintError(f"layer {layer_index} 没有权重")
            matrix = [
                _finite_vector(row, width, f"layer {layer_index} weights")
                for row in weights
            ]
            bias = _finite_vector(layer["bias"], len(matrix), f"layer {layer_index} bias")
            activation = layer.get("activation", "linear")
            if activation not in {"linear", "relu", "tanh"}:
                raise BlueprintError(f"不支持的激活函数：{activation}")
            self.layers.append((matrix, bias, activation))
            width = len(matrix)
        if width != self.output_size:
            raise BlueprintError("模型输出维度与最后一层不一致")
        self._numpy = None
        self._numpy_mean = None
        self._numpy_scale = None
        self._numpy_layers = None
        try:
            import numpy as np

            self._numpy = np
            # float32 is ample under the configured 5% decision-error budget
            # and halves model bandwidth versus float64 during hot inference.
            self._numpy_mean = np.asarray(self.mean, dtype=np.float32)
            self._numpy_scale = np.maximum(
                np.abs(np.asarray(self.scale, dtype=np.float32)),
                1e-6,
            )
            self._numpy_layers = [
                (
                    np.asarray(weights, dtype=np.float32),
                    np.asarray(bias, dtype=np.float32),
                    activation,
                )
                for weights, bias, activation in self.layers
            ]
        except ImportError:
            pass

    def predict(self, features: list[float]) -> list[float]:
        if len(features) != self.input_size:
            raise BlueprintError(f"特征维度错误：期望 {self.input_size}")
        if self._numpy_layers is not None:
            return self.predict_many([features])[0]
        values = [
            (float(value) - mean) / max(abs(scale), 1e-6)
            for value, mean, scale in zip(features, self.mean, self.scale)
        ]
        for weights, bias, activation in self.layers:
            output = [
                offset + sum(weight * value for weight, value in zip(row, values))
                for row, offset in zip(weights, bias)
            ]
            if activation == "relu":
                output = [max(0.0, value) for value in output]
            elif activation == "tanh":
                output = [math.tanh(value) for value in output]
            values = output
        if not all(math.isfinite(value) for value in values):
            raise BlueprintError("模型产生了非有限输出")
        return values

    def predict_many(self, rows: list[list[float]]) -> list[list[float]]:
        """Vectorized inference when NumPy is present, with a Python fallback."""
        if not rows:
            return []
        if any(len(row) != self.input_size for row in rows):
            raise BlueprintError(f"feature width must be {self.input_size}")
        if self._numpy_layers is None:
            return [self.predict(row) for row in rows]
        np = self._numpy
        values = np.asarray(rows, dtype=np.float32)
        values = (values - self._numpy_mean) / self._numpy_scale
        for weights, bias, activation in self._numpy_layers:
            values = values @ weights.T
            values += bias
            if activation == "relu":
                values = np.maximum(values, 0.0)
            elif activation == "tanh":
                values = np.tanh(values)
        if not np.isfinite(values).all():
            raise BlueprintError("model produced a non-finite output")
        return values.tolist()


def validate_bundle(bundle: dict[str, Any], *, require_fingerprint: bool = True) -> None:
    if bundle.get("format") != BLUEPRINT_FORMAT:
        raise BlueprintError("不是 Texas Lab Blueprint 文件")
    if bundle.get("schemaVersion") != SCHEMA_VERSION:
        raise BlueprintError("Blueprint schema 版本不兼容")
    if bundle.get("featureVersion") != FEATURE_VERSION:
        raise BlueprintError("Blueprint 特征版本不兼容")
    if bundle.get("abstractionVersion") != ABSTRACTION_VERSION:
        raise BlueprintError("Blueprint 抽象版本不兼容")
    if bundle.get("rulesHash") != RULES_HASH:
        raise BlueprintError("Blueprint 规则版本不兼容")
    if not isinstance(bundle.get("validated"), bool):
        raise BlueprintError("Blueprint validated 必须是布尔值")
    if require_fingerprint:
        fingerprint = bundle.get("fingerprint")
        if not isinstance(fingerprint, str) or fingerprint != bundle_fingerprint(bundle):
            raise BlueprintError("Blueprint 指纹校验失败")
    exact = bundle.get("exactSolutions", {})
    if not isinstance(exact, dict):
        raise BlueprintError("exactSolutions 必须是对象")
    for model_name in ("policyModel", "valueModel"):
        model = bundle.get(model_name)
        if model is not None:
            if not isinstance(model, dict) or not isinstance(model.get("network"), dict):
                raise BlueprintError(f"{model_name} 缺少可移植 network")
            if not isinstance(model.get("validated"), bool):
                raise BlueprintError(f"{model_name}.validated 必须是布尔值")
            DenseNetwork(model["network"])
    leaf = bundle.get("leafModel")
    if leaf is not None:
        raise BlueprintError(
            "Blueprint 内的单标量 leafModel 不安全；请使用独立的六席信念条件叶模型 artifact"
        )
    policy = bundle.get("policyModel")
    if policy is not None and policy["network"].get("outputSize") != len(ACTION_ORDER):
        raise BlueprintError("Blueprint 策略模型输出维度错误")
    value = bundle.get("valueModel")
    if value is not None and value["network"].get("outputSize") != 1:
        raise BlueprintError("Blueprint 状态价值模型必须输出一个标量")


def _rank(card: int) -> int:
    return 2 + int(card) // 4


def _board_features(board: tuple[int, ...]) -> list[float]:
    if not board:
        return [0.0] * 10
    ranks = [_rank(card) for card in board]
    suits = [card % 4 for card in board]
    unique_ranks = set(ranks)
    connected = set(unique_ranks)
    if 14 in connected:
        connected.add(1)
    connectivity = max(
        sum(rank in connected for rank in range(high - 4, high + 1))
        for high in range(5, 15)
    )
    return [
        len(board) / 5,
        max(ranks) / 14,
        min(ranks) / 14,
        len(unique_ranks) / 5,
        max(ranks.count(rank) for rank in unique_ranks) / 4,
        sum(ranks.count(rank) >= 2 for rank in unique_ranks) / 2,
        max(suits.count(suit) for suit in set(suits)) / 5,
        len(set(suits)) / 4,
        connectivity / 5,
        sum(rank >= 10 for rank in ranks) / 5,
    ]


def _hand_features(hand: tuple[int, int], board: tuple[int, ...]) -> list[float]:
    from .solver import hand_bucket

    ranks = sorted((_rank(hand[0]), _rank(hand[1])), reverse=True)
    values = [
        ranks[0] / 14,
        ranks[1] / 14,
        float(ranks[0] == ranks[1]),
        float(hand[0] % 4 == hand[1] % 4),
        (ranks[0] - ranks[1]) / 12,
    ]
    if not board:
        return [*values, *([0.0] * 8)]
    features = hand_bucket(hand, board)
    return [
        *values,
        features[1] / 8,
        features[2] / 2,
        features[3] / 2,
        float(features[5]),
        features[6] / 5,
        features[7] / 2,
        features[8] / 5,
        features[9] / 4,
    ]


def _decode_path(node_id: int, root_street: str) -> list[tuple[str, str]]:
    digits: list[int] = []
    current = int(node_id)
    while current > 1:
        digits.append(current % 6)
        current //= 6
    street_index = STREETS.index(root_street)
    result: list[tuple[str, str]] = []
    for digit in reversed(digits):
        if digit == 5:
            street_index = min(len(STREETS) - 1, street_index + 1)
        elif digit in PATH_ACTIONS:
            result.append((STREETS[street_index], PATH_ACTIONS[digit]))
    return result


def _history_features(history: list[dict[str, Any]], suffix: list[tuple[str, str]]) -> list[float]:
    actions = [
        (str(entry.get("street", "preflop")), str(entry.get("type", "")))
        for entry in public_history(history)
        if entry.get("type") in ACTION_CODES
    ]
    actions.extend(suffix)
    counts = [sum(action == name for _, action in actions) / 12 for name in ACTION_ORDER]
    current_street = actions[-1][0] if actions else "preflop"
    street_counts = [
        sum(street == current_street and action == name for street, action in actions) / 6
        for name in ACTION_ORDER
    ]
    recent: list[float] = []
    for _, action in actions[-8:]:
        recent.extend(float(index == ACTION_CODES[action]) for index in range(4))
    recent = [0.0] * (32 - len(recent)) + recent
    return [*counts, *street_counts, *recent, min(1.0, len(actions) / 32)]


def encode_state_features(
    state: Any,
    hands: dict[str, tuple[int, int]],
    target_id: str,
    *,
    root_history: list[dict[str, Any]] | None = None,
    root_street: str | None = None,
) -> list[float]:
    """Encode a public state and the acting player's private hand only."""
    players = {player.position: player for player in state.players}
    target = next(player for player in state.players if player.id == target_id)
    board = tuple(state.board)
    active_count = sum(not player.folded for player in state.players)
    actable_count = sum(not player.folded and not player.all_in for player in state.players)
    legal = ()
    if state.pending and state.pending[0] == target_id:
        from .solver import legal_actions

        legal = legal_actions(state)
    values: list[float] = [float(state.street == street) for street in STREETS]
    values.extend(
        [
            state.pot / 100,
            state.current_bet / 100,
            state.aggressive_actions / 3,
            active_count / 6,
            actable_count / 6,
            len(state.pending) / 6,
        ]
    )
    values.extend(float(action in legal) for action in ACTION_ORDER)
    values.extend(_board_features(board))
    values.extend(float(target.position == position) for position in POSITIONS)
    to_call = max(0.0, state.current_bet - target.street_committed)
    values.extend(
        [
            target.stack / 100,
            target.street_committed / 100,
            target.total_committed / 100,
            to_call / 100,
            target.stack / max(1.0, state.pot),
            float(target.folded),
            float(target.all_in),
        ]
    )
    values.extend(_hand_features(hands[target_id], board))
    pending_index = {player_id: index for index, player_id in enumerate(state.pending)}
    for position in POSITIONS:
        player = players[position]
        values.extend(
            [
                1.0,
                player.stack / 100,
                player.street_committed / 100,
                player.total_committed / 100,
                float(player.folded),
                float(player.all_in),
                float(player.id in pending_index),
                pending_index.get(player.id, 6) / 6,
            ]
        )
    suffix = _decode_path(state.node_id, root_street or state.street)
    values.extend(_history_features(root_history or [], suffix))
    if len(values) != FEATURE_SIZE or not all(math.isfinite(value) for value in values):
        raise BlueprintError(f"生成了无效特征：维度 {len(values)}，期望 {FEATURE_SIZE}")
    return values


def encode_state_features_many_hands(
    state: Any,
    target_id: str,
    hands: list[tuple[int, int]],
    *,
    root_history: list[dict[str, Any]] | None = None,
    root_street: str | None = None,
) -> list[list[float]]:
    """Encode many private hands while reusing all public-state features."""
    if not hands:
        return []
    template = encode_state_features(
        state,
        {target_id: hands[0]},
        target_id,
        root_history=root_history,
        root_street=root_street,
    )
    # The 13 acting-hand features occupy a stable slice in FEATURE_SIZE v1.
    hand_start = 37
    hand_end = hand_start + 13
    prefix = template[:hand_start]
    suffix = template[hand_end:]
    board = tuple(state.board)
    rows = [
        [*prefix, *_hand_features(hand, board), *suffix]
        for hand in hands
    ]
    if any(len(row) != FEATURE_SIZE for row in rows):
        raise BlueprintError("批量手牌特征维度漂移")
    return rows


def encode_info_features(info: dict[str, Any]) -> list[float]:
    from .solver import make_state

    normalized = normalize_solve_info(info)
    state = make_state(normalized)
    hero_id = normalized["hero"]["id"]
    return encode_state_features(
        state,
        {hero_id: tuple(normalized["hero"]["hole"])},
        hero_id,
        root_history=normalized["history"],
        root_street=normalized["street"],
    )


class BlueprintRuntime:
    def __init__(self, bundle: dict[str, Any] | None, path: str | None = None) -> None:
        self.bundle = bundle
        self.path = path
        self.fingerprint = "none" if bundle is None else str(bundle["fingerprint"])
        self.validated = bool(bundle and bundle.get("validated") is True)
        self.exact_solutions = {} if bundle is None else bundle.get("exactSolutions", {})
        policy = None if bundle is None else bundle.get("policyModel")
        value = None if bundle is None else bundle.get("valueModel")
        self.policy_payload = policy
        self.value_payload = value
        self.policy_network = None if policy is None else DenseNetwork(policy["network"])
        self.value_network = None if value is None else DenseNetwork(value["network"])
        self.exact_hits = 0
        self.policy_lookups = 0
        self.value_evaluations = 0
        self._lock = threading.Lock()

    @staticmethod
    def _supports_features(payload: dict[str, Any] | None, features: list[float]) -> bool:
        if not payload:
            return False
        supported = payload.get("supportedStreets")
        if supported is None:
            # Bundles created before support metadata covered every street.
            return True
        if not isinstance(supported, list) or not supported:
            return False
        street_values = features[: len(STREETS)]
        if len(street_values) != len(STREETS):
            return False
        street = STREETS[max(range(len(STREETS)), key=street_values.__getitem__)]
        return street in supported

    @property
    def loaded(self) -> bool:
        return self.bundle is not None

    @property
    def has_valid_policy(self) -> bool:
        return bool(
            self.policy_network is not None
            and self.policy_payload
            and self.policy_payload.get("validated") is True
        )

    def lookup_exact(self, info: dict[str, Any], iterations: int) -> dict[str, Any] | None:
        if not self.exact_solutions:
            return None
        digest, canonical = stable_info_key(info)
        entry = self.exact_solutions.get(digest)
        if not isinstance(entry, dict):
            return None
        if entry.get("canonical") != canonical or int(entry.get("iterations", 0)) < iterations:
            return None
        result = json.loads(json.dumps(entry["result"]))
        result["blueprintExactHit"] = True
        result["blueprintFingerprint"] = self.fingerprint
        with self._lock:
            self.exact_hits += 1
        return result

    def policy(self, features: list[float], legal_actions: tuple[str, ...]) -> tuple[float, ...] | None:
        if (
            not legal_actions
            or not self.has_valid_policy
            or not self._supports_features(self.policy_payload, features)
        ):
            return None
        logits = self.policy_network.predict(features)
        maximum = max(logits[ACTION_CODES[action]] for action in legal_actions)
        weights = [math.exp(logits[ACTION_CODES[action]] - maximum) for action in legal_actions]
        total = sum(weights)
        with self._lock:
            self.policy_lookups += 1
        return tuple(weight / total for weight in weights)

    def policy_many(
        self,
        feature_rows: list[list[float]],
        legal_actions: tuple[str, ...],
    ) -> list[tuple[float, ...]] | None:
        if (
            not feature_rows
            or not legal_actions
            or not self.has_valid_policy
            or not self._supports_features(self.policy_payload, feature_rows[0])
        ):
            return None
        logits_rows = self.policy_network.predict_many(feature_rows)
        action_indices = [ACTION_CODES[action] for action in legal_actions]
        policies: list[tuple[float, ...]] = []
        for logits in logits_rows:
            maximum = max(logits[index] for index in action_indices)
            weights = [math.exp(logits[index] - maximum) for index in action_indices]
            total = sum(weights)
            policies.append(tuple(weight / total for weight in weights))
        with self._lock:
            self.policy_lookups += len(policies)
        return policies

    def state_value(self, features: list[float]) -> float | None:
        """Return the acting player's root value for diagnostics/control variates.

        This model is deliberately separate from ``leafModel``: a scalar root
        value is not sufficient to replace a range-conditioned subgame leaf.
        """
        if (
            self.value_network is None
            or not self.value_payload
            or self.value_payload.get("validated") is not True
            or not self._supports_features(self.value_payload, features)
        ):
            return None
        raw = self.value_network.predict(features)[0]
        target_mean = float(self.value_payload.get("targetMean", 0.0))
        target_scale = max(1e-6, abs(float(self.value_payload.get("targetScale", 1.0))))
        with self._lock:
            self.value_evaluations += 1
        return raw * target_scale + target_mean

    def diagnostics(self) -> dict[str, Any]:
        return {
            "blueprintLoaded": self.loaded,
            "blueprintValidated": self.validated,
            "blueprintFingerprint": self.fingerprint,
            "blueprintExactHits": self.exact_hits,
            "blueprintPolicyLookups": self.policy_lookups,
            "blueprintValueEvaluations": self.value_evaluations,
        }


_RUNTIME: BlueprintRuntime | None = None
_RUNTIME_STAMP: tuple[str, int, int] | None = None
_RUNTIME_LOCK = threading.Lock()


def _configured_path() -> Path | None:
    raw = os.environ.get("TEXAS_LAB_BLUEPRINT_PATH", "").strip()
    return Path(raw).resolve() if raw else None


def get_blueprint_runtime() -> BlueprintRuntime:
    global _RUNTIME, _RUNTIME_STAMP
    mode = os.environ.get("TEXAS_LAB_BLUEPRINT_MODE", "auto").lower()
    if mode not in {"auto", "off", "required"}:
        raise BlueprintError("TEXAS_LAB_BLUEPRINT_MODE 必须是 auto、off 或 required")
    if mode == "off":
        return BlueprintRuntime(None)
    path = _configured_path()
    if path is None or not path.exists():
        if mode == "required":
            raise BlueprintError("已要求 Blueprint，但 TEXAS_LAB_BLUEPRINT_PATH 不存在")
        return BlueprintRuntime(None)
    stat = path.stat()
    stamp = (str(path), stat.st_mtime_ns, stat.st_size)
    with _RUNTIME_LOCK:
        if _RUNTIME is not None and _RUNTIME_STAMP == stamp:
            return _RUNTIME
        try:
            with _open_text(path, "rt") as handle:
                bundle = json.load(handle)
            validate_bundle(bundle, require_fingerprint=True)
            runtime = BlueprintRuntime(bundle, str(path))
        except Exception as error:
            if mode == "required":
                raise BlueprintError(f"Blueprint 加载失败：{error}") from error
            runtime = BlueprintRuntime(None, str(path))
        _RUNTIME = runtime
        _RUNTIME_STAMP = stamp
        return runtime


def reset_blueprint_runtime() -> None:
    global _RUNTIME, _RUNTIME_STAMP
    with _RUNTIME_LOCK:
        _RUNTIME = None
        _RUNTIME_STAMP = None


def runtime_signature() -> str:
    return get_blueprint_runtime().fingerprint
