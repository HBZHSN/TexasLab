from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import hmac
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator, Sequence

from .abstraction import ABSTRACTION_VERSION, RULES_HASH
from .belief import ALL_COMBOS, COMBO_INDEX, COMBO_MASKS
from .poker import evaluate
from .schema import PUBLIC_HISTORY_FIELDS, history_contains_private_cards, public_history


LEAF_DATASET_FORMAT = "texas-lab-leaf-dataset"
LEAF_MODEL_FORMAT = "texas-lab-leaf-value-model"
LEAF_MANIFEST_FORMAT = "texas-lab-leaf-value-manifest"
LEAF_APPROVAL_FORMAT = "texas-lab-leaf-production-approval"
LEAF_AB_CANDIDATE_FORMAT = "texas-lab-leaf-ab-candidate"
APPROVAL_VALIDATOR_VERSION = "leaf-production-validator-v4"
SCHEMA_VERSION = 1
FEATURE_VERSION = "public-belief-six-seat-range169-v2"
MODEL_TYPE = "portable-dense-seat-hand-v2"
POSITIONS = ("UTG", "HJ", "CO", "BTN", "SB", "BB")
STREETS = ("preflop", "flop", "turn", "river")
ACTIONS = ("fold", "check", "call", "raise")
CONTINUATIONS = ("blueprint", "fold-x5", "call-x5", "raise-x5")
HAND_COUNT = len(ALL_COMBOS)
SEAT_COUNT = len(POSITIONS)
COMBO_ORDER_HASH = hashlib.sha256(
    b"".join(bytes((first, second)) for first, second in ALL_COMBOS)
).hexdigest()[:16]

MAX_HEADER_BYTES = 256_000
MAX_SAMPLE_BYTES = 8_000_000
MAX_MANIFEST_BYTES = 1_000_000
MAX_MODEL_COMPRESSED_BYTES = 128_000_000
MAX_MODEL_UNCOMPRESSED_BYTES = 512_000_000
MAX_MODEL_PARAMETERS = 10_000_000

QUALITY_GATES: dict[str, float] = {
    "minHeldoutExamples": 5_000,
    "minHeldoutPublicStates": 250,
    "maxMaeBb": 0.50,
    "maxP99AbsoluteErrorBb": 2.50,
    "minCoverage95": 0.90,
    "maxMeanUncertaintyBb": 1.00,
    "maxGroupMaeBb": 1.00,
    "minDecisiveNodes": 500,
    "minActionAgreement": 0.95,
    "maxStrategyL1Median": 0.10,
    "maxStrategyL1P95": 0.15,
    "maxActionEvRelativeErrorP95": 0.05,
    "minOodFallbackRecall": 0.95,
    "minLeafActivatedNodes": 250,
    "minLeafActivationRate": 0.95,
}

INTEGRATION_METRIC_NAMES = frozenset(
    {
        "decisiveNodes",
        "actionAgreement",
        "strategyL1Median",
        "strategyL1P95",
        "actionEvRelativeErrorP95",
        "oodFallbackRecall",
        "leafActivatedNodes",
        "leafActivationRate",
    }
)
PRODUCTION_SOLVER_CONFIG = {
    "iterations": 20_000,
    "regretMode": "rmplus",
    "adaptiveTraining": False,
    "adaptiveEvaluation": False,
    "errorBudgetFraction": 0.05,
    "blueprintPriorWeight": 16.0,
    "leafBeliefRefresh": 64,
}
PRODUCTION_CUTOFF_CONFIG = {
    "mode": "auto",
    "depth": None,
}
PRODUCTION_CONTINUATION_CONFIG = {
    "positions": list(POSITIONS),
    "continuations": ["blueprint"] * len(POSITIONS),
}
PRODUCTION_VALIDATION_CONFIG = {
    "decisiveGapFraction": 0.05,
    "holdoutSelection": "all-normalized-rows-v1",
}


class LeafValueError(RuntimeError):
    pass


class LeafSchemaError(LeafValueError):
    pass


class LeafArtifactError(LeafValueError):
    pass


class LeafArtifactNotApproved(LeafArtifactError):
    pass


class LeafCutoffRejected(LeafValueError):
    pass


def _reject_constant(value: str) -> None:
    raise LeafSchemaError(f"JSON 包含非法常量：{value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LeafSchemaError(f"JSON 包含重复字段：{key}")
        result[key] = value
    return result


def strict_json_loads(data: str | bytes) -> Any:
    try:
        return json.loads(
            data,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except LeafSchemaError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise LeafSchemaError(f"JSON 格式错误：{error}") from error


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise LeafSchemaError(f"{label} 必须是有限数字")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise LeafSchemaError(f"{label} 必须是有限数字") from error
    if not math.isfinite(number):
        raise LeafSchemaError(f"{label} 必须是有限数字")
    return number


def _integer(value: Any, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise LeafSchemaError(f"{label} 必须是 {low}..{high} 的整数")
    return value


def _require_keys(value: Any, required: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LeafSchemaError(f"{label} 必须是对象")
    missing = required.difference(value)
    unknown = set(value).difference(required)
    if missing:
        raise LeafSchemaError(f"{label} 缺少字段：{sorted(missing)}")
    if unknown:
        raise LeafSchemaError(f"{label} 包含未知字段：{sorted(unknown)}")
    return value


PUBLIC_STATE_FIELDS = {
    "street",
    "board",
    "pot",
    "currentBet",
    "aggressiveActions",
    "players",
    "pending",
    "history",
}
PLAYER_FIELDS = {
    "id",
    "position",
    "stack",
    "streetCommitted",
    "totalCommitted",
    "folded",
    "allIn",
}


def public_state_from_info(info: dict[str, Any]) -> dict[str, Any]:
    """Project a live solve request onto the leaf model's public-only contract."""
    return {
        "street": info["street"],
        "board": list(info["board"]),
        "pot": info["pot"],
        "currentBet": info["currentBet"],
        "aggressiveActions": info.get("aggressiveActions", 0),
        "players": [
            {
                field: player[field]
                for field in PLAYER_FIELDS
            }
            for player in info["players"]
        ],
        "pending": list(info["pending"]),
        "history": public_history(info["history"]),
    }


def validate_public_state(value: Any) -> dict[str, Any]:
    state = _require_keys(value, PUBLIC_STATE_FIELDS, "publicState")
    if history_contains_private_cards(state):
        raise LeafSchemaError("publicState 包含私有牌字段")
    street = state["street"]
    if street not in STREETS:
        raise LeafSchemaError("publicState.street 非法")
    if not isinstance(state["board"], list):
        raise LeafSchemaError("publicState.board 必须是数组")
    board = [
        _integer(card, f"publicState.board[{index}]", 0, 51)
        for index, card in enumerate(state["board"])
    ]
    expected_board = (0, 3, 4, 5)[STREETS.index(street)]
    if len(board) != expected_board or len(set(board)) != len(board):
        raise LeafSchemaError("公共牌数量、街道或唯一性不一致")

    players = state["players"]
    if not isinstance(players, list) or len(players) != SEAT_COUNT:
        raise LeafSchemaError("publicState.players 必须包含六个座位")
    normalized_players: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_positions: set[str] = set()
    for index, raw_player in enumerate(players):
        player = _require_keys(raw_player, PLAYER_FIELDS, f"players[{index}]")
        player_id = player["id"]
        position = player["position"]
        if not isinstance(player_id, str) or not player_id or len(player_id) > 64:
            raise LeafSchemaError(f"players[{index}].id 非法")
        if position not in POSITIONS:
            raise LeafSchemaError(f"players[{index}].position 非法")
        if player_id in seen_ids or position in seen_positions:
            raise LeafSchemaError("玩家 ID 或位置重复")
        seen_ids.add(player_id)
        seen_positions.add(position)
        amounts = {
            field: _finite_number(player[field], f"players[{index}].{field}")
            for field in ("stack", "streetCommitted", "totalCommitted")
        }
        if any(amount < 0 for amount in amounts.values()):
            raise LeafSchemaError("玩家筹码或投入不能为负")
        if amounts["streetCommitted"] > amounts["totalCommitted"] + 1e-6:
            raise LeafSchemaError("本街投入不能大于累计投入")
        if not isinstance(player["folded"], bool) or not isinstance(player["allIn"], bool):
            raise LeafSchemaError("folded/allIn 必须是布尔值")
        normalized_players.append(
            {
                "id": player_id,
                "position": position,
                **amounts,
                "folded": player["folded"],
                "allIn": player["allIn"],
            }
        )
    if seen_positions != set(POSITIONS):
        raise LeafSchemaError("六个标准位置必须各出现一次")

    pot = _finite_number(state["pot"], "publicState.pot")
    current_bet = _finite_number(state["currentBet"], "publicState.currentBet")
    if pot < 0 or current_bet < 0:
        raise LeafSchemaError("底池和当前下注不能为负")
    committed = sum(player["totalCommitted"] for player in normalized_players)
    if abs(pot - committed) > 0.051:
        raise LeafSchemaError("底池与玩家累计投入不一致")
    aggressive = _integer(state["aggressiveActions"], "aggressiveActions", 0, 3)

    pending = state["pending"]
    if not isinstance(pending, list) or any(player_id not in seen_ids for player_id in pending):
        raise LeafSchemaError("pending 包含未知玩家")
    if len(pending) != len(set(pending)):
        raise LeafSchemaError("pending 玩家重复")

    history = state["history"]
    if not isinstance(history, list) or len(history) > 128:
        raise LeafSchemaError("history 必须是不超过 128 项的数组")
    allowed_history = set(PUBLIC_HISTORY_FIELDS)
    normalized_history: list[dict[str, Any]] = []
    for index, entry in enumerate(history):
        if not isinstance(entry, dict) or not set(entry).issubset(allowed_history):
            raise LeafSchemaError(f"history[{index}] 包含非公共字段")
        if entry.get("street") not in STREETS:
            raise LeafSchemaError(f"history[{index}].street 非法")
        if entry.get("type") not in ACTIONS:
            raise LeafSchemaError(f"history[{index}].type 非法")
        if entry.get("playerId") not in seen_ids:
            raise LeafSchemaError(f"history[{index}].playerId 非法")
        history_board = entry.get("board", [])
        if not isinstance(history_board, list) or any(
            isinstance(card, bool) or not isinstance(card, int) or not 0 <= card <= 51
            for card in history_board
        ):
            raise LeafSchemaError(f"history[{index}].board 非法")
        normalized_history.append(dict(entry))

    ordered_players = sorted(normalized_players, key=lambda row: POSITIONS.index(row["position"]))
    return {
        "street": street,
        "board": board,
        "pot": pot,
        "currentBet": current_bet,
        "aggressiveActions": aggressive,
        "players": ordered_players,
        "pending": list(pending),
        "history": normalized_history,
    }


def _validate_ranges(value: Any, board: Sequence[int]) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or len(value) != SEAT_COUNT:
        raise LeafSchemaError("ranges 必须是六个 1326 组合范围")
    board_mask = sum(1 << card for card in board)
    result: list[tuple[float, ...]] = []
    for seat, raw_range in enumerate(value):
        if not isinstance(raw_range, list) or len(raw_range) != HAND_COUNT:
            raise LeafSchemaError(f"ranges[{seat}] 必须包含 {HAND_COUNT} 项")
        probabilities = tuple(
            _finite_number(probability, f"ranges[{seat}][{index}]")
            for index, probability in enumerate(raw_range)
        )
        if any(probability < 0 for probability in probabilities):
            raise LeafSchemaError("range 概率不能为负")
        if abs(sum(probabilities) - 1.0) > 1e-5:
            raise LeafSchemaError(f"ranges[{seat}] 概率之和必须为 1")
        if any(
            probability > 1e-9 and combo_mask & board_mask
            for probability, combo_mask in zip(probabilities, COMBO_MASKS)
        ):
            raise LeafSchemaError(f"ranges[{seat}] 对公共牌冲突组合赋予了概率")
        result.append(probabilities)
    return tuple(result)


def _validate_continuations(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != SEAT_COUNT:
        raise LeafSchemaError("continuationIds 必须包含六项")
    return tuple(
        _integer(item, f"continuationIds[{index}]", 0, len(CONTINUATIONS) - 1)
        for index, item in enumerate(value)
    )


@dataclass(frozen=True)
class LeafModelInput:
    public_state: dict[str, Any]
    ranges: tuple[tuple[float, ...], ...]
    continuation_ids: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: Any) -> "LeafModelInput":
        payload = _require_keys(
            value,
            {"publicState", "ranges", "continuationIds"},
            "modelInput",
        )
        state = validate_public_state(payload["publicState"])
        ranges = _validate_ranges(payload["ranges"], state["board"])
        continuations = _validate_continuations(payload["continuationIds"])
        return cls(state, ranges, continuations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "publicState": self.public_state,
            "ranges": [list(hand_range) for hand_range in self.ranges],
            "continuationIds": list(self.continuation_ids),
        }


def _matrix(
    value: Any,
    label: str,
    converter,
) -> tuple[tuple[Any, ...], ...]:
    if not isinstance(value, list) or len(value) != SEAT_COUNT:
        raise LeafSchemaError(f"{label} 必须有六行")
    result: list[tuple[Any, ...]] = []
    for seat, row in enumerate(value):
        if not isinstance(row, list) or len(row) != HAND_COUNT:
            raise LeafSchemaError(f"{label}[{seat}] 必须包含 {HAND_COUNT} 项")
        result.append(tuple(converter(item, f"{label}[{seat}][{index}]") for index, item in enumerate(row)))
    return tuple(result)


@dataclass(frozen=True)
class LeafTrainingSample:
    sample_id: str
    model_input: LeafModelInput
    final_stack_values: tuple[tuple[float, ...], ...]
    target_stderr: tuple[tuple[float, ...], ...]
    target_counts: tuple[tuple[int, ...], ...]

    @classmethod
    def from_dict(cls, value: Any) -> "LeafTrainingSample":
        payload = _require_keys(
            value,
            {
                "sampleId",
                "publicState",
                "ranges",
                "continuationIds",
                "finalStackValues",
                "targetStdErr",
                "targetCounts",
            },
            "sample",
        )
        sample_id = payload["sampleId"]
        if not isinstance(sample_id, str) or not sample_id or len(sample_id) > 128:
            raise LeafSchemaError("sampleId 必须是 1..128 字符的字符串")
        model_input = LeafModelInput.from_dict(
            {
                "publicState": payload["publicState"],
                "ranges": payload["ranges"],
                "continuationIds": payload["continuationIds"],
            }
        )
        values = _matrix(payload["finalStackValues"], "finalStackValues", _finite_number)
        stderr = _matrix(payload["targetStdErr"], "targetStdErr", _finite_number)

        def count_converter(item: Any, label: str) -> int:
            return _integer(item, label, 0, 2_147_483_647)

        counts = _matrix(payload["targetCounts"], "targetCounts", count_converter)
        total_chips = sum(
            player["stack"] + player["totalCommitted"]
            for player in model_input.public_state["players"]
        )
        supported = 0
        board_mask = sum(1 << card for card in model_input.public_state["board"])
        for seat in range(SEAT_COUNT):
            for hand_index in range(HAND_COUNT):
                value_item = values[seat][hand_index]
                stderr_item = stderr[seat][hand_index]
                count_item = counts[seat][hand_index]
                if stderr_item < 0:
                    raise LeafSchemaError("targetStdErr 不能为负")
                invalid = bool(COMBO_MASKS[hand_index] & board_mask)
                if invalid and count_item:
                    raise LeafSchemaError("公共牌冲突手牌不能含训练目标")
                if count_item:
                    supported += 1
                    if not -1e-6 <= value_item <= total_chips + 1e-6:
                        raise LeafSchemaError("finalStackValues 超出总筹码范围")
        if supported == 0:
            raise LeafSchemaError("样本没有任何有效训练目标")
        return cls(sample_id, model_input, values, stderr, counts)

    def to_dict(self) -> dict[str, Any]:
        model_input = self.model_input.to_dict()
        return {
            "sampleId": self.sample_id,
            **model_input,
            "finalStackValues": [list(row) for row in self.final_stack_values],
            "targetStdErr": [list(row) for row in self.target_stderr],
            "targetCounts": [list(row) for row in self.target_counts],
        }


def dataset_header(*, teacher_version: str, blueprint_version: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    if not teacher_version or not blueprint_version:
        raise LeafSchemaError("teacherVersion 和 blueprintVersion 不能为空")
    return {
        "format": LEAF_DATASET_FORMAT,
        "schemaVersion": SCHEMA_VERSION,
        "featureVersion": FEATURE_VERSION,
        "abstractionVersion": ABSTRACTION_VERSION,
        "rulesHash": RULES_HASH,
        "comboOrderHash": COMBO_ORDER_HASH,
        "seatCount": SEAT_COUNT,
        "handCount": HAND_COUNT,
        "continuationCount": len(CONTINUATIONS),
        "teacherVersion": teacher_version,
        "blueprintVersion": blueprint_version,
        "metadata": dict(metadata or {}),
    }


def validate_dataset_header(value: Any) -> dict[str, Any]:
    required = {
        "format",
        "schemaVersion",
        "featureVersion",
        "abstractionVersion",
        "rulesHash",
        "comboOrderHash",
        "seatCount",
        "handCount",
        "continuationCount",
        "teacherVersion",
        "blueprintVersion",
        "metadata",
    }
    header = _require_keys(value, required, "dataset header")
    expected = {
        "format": LEAF_DATASET_FORMAT,
        "schemaVersion": SCHEMA_VERSION,
        "featureVersion": FEATURE_VERSION,
        "abstractionVersion": ABSTRACTION_VERSION,
        "rulesHash": RULES_HASH,
        "comboOrderHash": COMBO_ORDER_HASH,
        "seatCount": SEAT_COUNT,
        "handCount": HAND_COUNT,
        "continuationCount": len(CONTINUATIONS),
    }
    for key, expected_value in expected.items():
        if header.get(key) != expected_value:
            raise LeafSchemaError(f"数据集 {key} 不兼容")
    if not isinstance(header["teacherVersion"], str) or not header["teacherVersion"]:
        raise LeafSchemaError("teacherVersion 非法")
    if not isinstance(header["blueprintVersion"], str) or not header["blueprintVersion"]:
        raise LeafSchemaError("blueprintVersion 非法")
    if not isinstance(header["metadata"], dict):
        raise LeafSchemaError("metadata 必须是对象")
    if history_contains_private_cards(header["metadata"]):
        raise LeafSchemaError("metadata 包含私有牌字段")
    return header


def _atomic_gzip_json_lines(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                for row in rows:
                    compressed.write(canonical_json_bytes(row) + b"\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_leaf_dataset(
    path: str | Path,
    header: dict[str, Any],
    samples: Iterable[LeafTrainingSample | dict[str, Any]],
) -> None:
    destination = Path(path)
    if not destination.name.endswith(".jsonl.gz"):
        raise LeafSchemaError("叶节点样本文件必须使用 .jsonl.gz")
    normalized_header = validate_dataset_header(header)

    def rows() -> Iterator[dict[str, Any]]:
        yield normalized_header
        for sample in samples:
            normalized = sample if isinstance(sample, LeafTrainingSample) else LeafTrainingSample.from_dict(sample)
            yield normalized.to_dict()

    _atomic_gzip_json_lines(destination, rows())


def _read_bounded_line(handle: gzip.GzipFile, limit: int, label: str) -> bytes:
    line = handle.readline(limit + 1)
    if len(line) > limit:
        raise LeafSchemaError(f"{label} 超过安全大小限制")
    return line


def iter_leaf_dataset(path: str | Path) -> Iterator[tuple[dict[str, Any], LeafTrainingSample | None]]:
    source = Path(path)
    if not source.name.endswith(".jsonl.gz"):
        raise LeafSchemaError("叶节点样本文件必须使用 .jsonl.gz")
    with source.open("rb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="rb") as compressed:
            first = _read_bounded_line(compressed, MAX_HEADER_BYTES, "数据集头")
            if not first:
                raise LeafSchemaError("叶节点数据集为空")
            header = validate_dataset_header(strict_json_loads(first))
            yield header, None
            line_number = 1
            while True:
                line = _read_bounded_line(compressed, MAX_SAMPLE_BYTES, f"样本行 {line_number + 1}")
                if not line:
                    break
                line_number += 1
                if not line.strip():
                    raise LeafSchemaError(f"样本行 {line_number} 为空")
                yield header, LeafTrainingSample.from_dict(strict_json_loads(line))


def read_leaf_dataset(path: str | Path) -> tuple[dict[str, Any], Iterator[LeafTrainingSample]]:
    iterator = iter_leaf_dataset(path)
    header, marker = next(iterator)
    assert marker is None

    def samples() -> Iterator[LeafTrainingSample]:
        for current_header, sample in iterator:
            if current_header is not header:
                raise AssertionError("数据集迭代器头发生变化")
            assert sample is not None
            yield sample

    return header, samples()


def _canonical_suits(board: Sequence[int], hand: tuple[int, int]) -> tuple[list[int], tuple[int, int], dict[int, int]]:
    suit_map: dict[int, int] = {}
    for card in [*board, *sorted(hand)]:
        suit = card % 4
        if suit not in suit_map:
            suit_map[suit] = len(suit_map)
    for suit in range(4):
        if suit not in suit_map:
            suit_map[suit] = len(suit_map)

    def transform(card: int) -> int:
        return (card // 4) * 4 + suit_map[card % 4]

    return [transform(card) for card in board], tuple(sorted(transform(card) for card in hand)), suit_map


def _range_summary(probabilities: Sequence[float], suit_map: dict[int, int]) -> list[float]:
    entropy = 0.0
    squared = 0.0
    support = 0
    rank_mass = [0.0] * 13
    suit_mass = [0.0] * 4
    pair_mass = 0.0
    suited_mass = 0.0
    mean_high = 0.0
    mean_gap = 0.0
    # Preserve the complete standard 169 starting-hand class distribution.
    # The previous rank/suit marginals mapped materially different ranges to
    # identical features, an information loss no larger network could repair.
    hand_class_mass = [0.0] * 169
    for probability, (first, second) in zip(probabilities, ALL_COMBOS):
        if probability <= 0:
            continue
        entropy -= probability * math.log(probability)
        squared += probability * probability
        support += 1
        first_rank, second_rank = first // 4, second // 4
        rank_mass[first_rank] += probability
        rank_mass[second_rank] += probability
        suit_mass[suit_map[first % 4]] += probability
        suit_mass[suit_map[second % 4]] += probability
        pair_mass += probability * float(first_rank == second_rank)
        suited_mass += probability * float(first % 4 == second % 4)
        mean_high += probability * max(first_rank, second_rank) / 12
        mean_gap += probability * abs(first_rank - second_rank) / 12
        high_rank, low_rank = sorted((first_rank, second_rank), reverse=True)
        if high_rank == low_rank:
            class_index = high_rank
        else:
            triangular = high_rank * (high_rank - 1) // 2 + low_rank
            class_index = (13 if first % 4 == second % 4 else 91) + triangular
        hand_class_mass[class_index] += probability
    ess = 0.0 if squared <= 0 else 1 / squared
    return [
        entropy / math.log(HAND_COUNT),
        ess / HAND_COUNT,
        support / HAND_COUNT,
        *[mass / 2 for mass in rank_mass],
        *[mass / 2 for mass in suit_mass],
        pair_mass,
        suited_mass,
        mean_high,
        mean_gap,
        *hand_class_mass,
    ]


def _total_chips(state: dict[str, Any]) -> float:
    return sum(player["stack"] + player["totalCommitted"] for player in state["players"])


def build_leaf_features(model_input: LeafModelInput, seat_index: int, hand_index: int) -> list[float]:
    if not 0 <= seat_index < SEAT_COUNT:
        raise LeafValueError("seat_index 超出范围")
    if not 0 <= hand_index < HAND_COUNT:
        raise LeafValueError("hand_index 超出范围")
    state = model_input.public_state
    board = state["board"]
    if COMBO_MASKS[hand_index] & sum(1 << card for card in board):
        raise LeafValueError("查询手牌与公共牌冲突")
    original_hand = ALL_COMBOS[hand_index]
    canonical_board, canonical_hand, suit_map = _canonical_suits(board, original_hand)
    total_chips = max(_total_chips(state), 1e-6)
    player_index = {player["id"]: index for index, player in enumerate(state["players"])}
    features: list[float] = []

    street_index = STREETS.index(state["street"])
    features.extend(float(index == street_index) for index in range(len(STREETS)))
    for slot in range(5):
        if slot < len(canonical_board):
            card = canonical_board[slot]
            rank, suit = card // 4, card % 4
            features.extend(float(index == rank) for index in range(13))
            features.extend(float(index == suit) for index in range(4))
            features.append(1.0)
        else:
            features.extend([0.0] * 18)
    features.extend(
        [
            state["pot"] / total_chips,
            state["currentBet"] / total_chips,
            state["aggressiveActions"] / 3,
            len(board) / 5,
        ]
    )

    pending = state["pending"]
    actor = player_index[pending[0]] if pending else -1
    features.extend(float(index == actor) for index in range(SEAT_COUNT))
    features.extend(float(player["id"] in pending) for player in state["players"])
    pending_order = {player_id: index for index, player_id in enumerate(pending)}
    features.extend(
        pending_order.get(player["id"], -1) / max(1, SEAT_COUNT - 1)
        for player in state["players"]
    )

    for player in state["players"]:
        features.extend(
            [
                player["stack"] / total_chips,
                player["streetCommitted"] / total_chips,
                player["totalCommitted"] / total_chips,
                float(player["folded"]),
                float(player["allIn"]),
            ]
        )

    street_action_counts = [[0.0] * len(ACTIONS) for _ in STREETS]
    seat_action_counts = [[0.0] * len(ACTIONS) for _ in POSITIONS]
    for entry in state["history"]:
        action = ACTIONS.index(entry["type"])
        street_action_counts[STREETS.index(entry["street"])][action] += 1 / 6
        seat = player_index[entry["playerId"]]
        seat_action_counts[seat][action] += 1 / 6
    features.extend(value for row in street_action_counts for value in row)
    features.extend(value for row in seat_action_counts for value in row)
    features.append(len(state["history"]) / 48)

    recent = state["history"][-12:]
    padding = 12 - len(recent)
    features.extend([0.0] * (padding * 6))
    for entry in recent:
        amount = _finite_number(entry.get("amount", 0), "history.amount")
        pot_after = _finite_number(entry.get("potAfter", 0), "history.potAfter")
        features.extend(
            [
                1.0,
                STREETS.index(entry["street"]) / 3,
                player_index[entry["playerId"]] / 5,
                ACTIONS.index(entry["type"]) / 3,
                amount / total_chips,
                pot_after / total_chips,
            ]
        )

    for continuation in model_input.continuation_ids:
        features.extend(float(index == continuation) for index in range(len(CONTINUATIONS)))
    for hand_range in model_input.ranges:
        features.extend(_range_summary(hand_range, suit_map))

    features.extend(float(index == seat_index) for index in range(SEAT_COUNT))
    high_card, low_card = sorted(canonical_hand, key=lambda card: card // 4, reverse=True)
    high_rank, low_rank = high_card // 4, low_card // 4
    features.extend(float(index == high_rank) for index in range(13))
    features.extend(float(index == low_rank) for index in range(13))
    features.extend(float(index == high_card % 4) for index in range(4))
    features.extend(float(index == low_card % 4) for index in range(4))
    features.extend(
        [
            float(high_rank == low_rank),
            float(high_card % 4 == low_card % 4),
            abs(high_rank - low_rank) / 12,
        ]
    )
    if len(canonical_board) >= 3:
        category = evaluate([high_card, low_card, *canonical_board])[0]
        features.extend(float(index == category) for index in range(9))
    else:
        features.extend([0.0] * 9)
    board_ranks = [card // 4 for card in canonical_board]
    board_suits = [card % 4 for card in canonical_board]
    ranks = {high_rank, low_rank, *board_ranks}
    if 12 in ranks:
        ranks.add(-1)
    straight_density = max(
        (sum(rank in ranks for rank in range(high - 4, high + 1)) for high in range(3, 13)),
        default=0,
    )
    features.extend(
        [
            sum(rank in board_ranks for rank in (high_rank, low_rank)) / 2,
            sum(rank > max(board_ranks, default=-1) for rank in (high_rank, low_rank)) / 2,
            max((board_suits + [high_card % 4, low_card % 4]).count(suit) for suit in range(4)) / 5,
            straight_density / 5,
        ]
    )
    features.extend(hand_range[hand_index] for hand_range in model_input.ranges)
    for hand_range in model_input.ranges:
        for target_card in original_hand:
            features.append(
                sum(
                    probability
                    for probability, combo in zip(hand_range, ALL_COMBOS)
                    if target_card in combo
                )
            )

    if len(features) != FEATURE_SIZE:
        raise AssertionError(f"叶节点特征维度漂移：{len(features)} != {FEATURE_SIZE}")
    if not all(math.isfinite(value) for value in features):
        raise LeafValueError("叶节点特征包含非有限值")
    return features


# See build_leaf_features. Keeping this explicit makes artifact incompatibility fail fast.
FEATURE_SIZE = 1515


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise LeafArtifactError(f"{label} 维度错误，期望 {length}")
    result = []
    for index, item in enumerate(value):
        try:
            number = float(item)
        except (TypeError, ValueError) as error:
            raise LeafArtifactError(f"{label}[{index}] 非法") from error
        if not math.isfinite(number):
            raise LeafArtifactError(f"{label}[{index}] 非有限")
        result.append(number)
    return result


class PortableDenseNetwork:
    def __init__(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict) or set(payload) != {
            "modelType",
            "inputSize",
            "outputSize",
            "normalization",
            "layers",
        }:
            raise LeafArtifactError("叶模型 network schema 非法")
        if payload.get("modelType") != MODEL_TYPE:
            raise LeafArtifactError("不支持的叶模型类型")
        self.input_size = int(payload.get("inputSize", -1))
        self.output_size = int(payload.get("outputSize", -1))
        if self.input_size != FEATURE_SIZE or self.output_size != 2:
            raise LeafArtifactError("叶模型输入或输出维度不兼容")
        normalization = payload.get("normalization")
        if not isinstance(normalization, dict) or set(normalization) != {"mean", "scale"}:
            raise LeafArtifactError("叶模型缺少 normalization")
        self.mean = _finite_vector(normalization.get("mean"), FEATURE_SIZE, "normalization.mean")
        self.scale = _finite_vector(normalization.get("scale"), FEATURE_SIZE, "normalization.scale")
        if any(scale <= 0 for scale in self.scale):
            raise LeafArtifactError("normalization.scale 必须为正")
        layers = payload.get("layers")
        if not isinstance(layers, list) or not layers:
            raise LeafArtifactError("叶模型没有网络层")
        self.layers: list[tuple[list[list[float]], list[float], str]] = []
        width = FEATURE_SIZE
        parameter_count = 0
        for layer_index, layer in enumerate(layers):
            if not isinstance(layer, dict) or set(layer) != {"weights", "bias", "activation"}:
                raise LeafArtifactError(f"layer {layer_index} schema 非法")
            weights = layer["weights"]
            if not isinstance(weights, list) or not weights or len(weights) > 4096:
                raise LeafArtifactError(f"layer {layer_index} 权重行数非法")
            matrix = [
                _finite_vector(row, width, f"layer {layer_index}.weights")
                for row in weights
            ]
            bias = _finite_vector(layer["bias"], len(matrix), f"layer {layer_index}.bias")
            activation = layer["activation"]
            if activation not in {"linear", "relu", "tanh"}:
                raise LeafArtifactError(f"layer {layer_index} 激活函数非法")
            parameter_count += len(matrix) * width + len(bias)
            if parameter_count > MAX_MODEL_PARAMETERS:
                raise LeafArtifactError("叶模型参数超过安全限制")
            self.layers.append((matrix, bias, activation))
            width = len(matrix)
        if width != 2 or self.layers[-1][2] != "linear":
            raise LeafArtifactError("叶模型末层必须是两个线性输出")
        self._numpy = None
        self._numpy_mean = None
        self._numpy_scale = None
        self._numpy_layers = None
        try:
            import numpy as np

            self._numpy = np
            self._numpy_mean = np.asarray(self.mean, dtype=np.float32)
            self._numpy_scale = np.asarray(self.scale, dtype=np.float32)
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

    def predict_raw(self, features: Sequence[float]) -> tuple[float, float]:
        if len(features) != FEATURE_SIZE:
            raise LeafValueError(f"模型特征必须包含 {FEATURE_SIZE} 项")
        if self._numpy_layers is not None:
            values = self._numpy.asarray(features, dtype=self._numpy.float32)
            values = (values - self._numpy_mean) / self._numpy_scale
            for weights, bias, activation in self._numpy_layers:
                values = weights @ values + bias
                if activation == "relu":
                    values = self._numpy.maximum(values, 0.0)
                elif activation == "tanh":
                    values = self._numpy.tanh(values)
            if not self._numpy.isfinite(values).all():
                raise LeafValueError("叶模型产生非有限输出")
            return float(values[0]), float(values[1])
        values = [
            (float(value) - mean) / scale
            for value, mean, scale in zip(features, self.mean, self.scale)
        ]
        if not all(math.isfinite(value) for value in values):
            raise LeafValueError("模型输入包含非有限值")
        for weights, bias, activation in self.layers:
            output = [
                offset + math.fsum(weight * value for weight, value in zip(row, values))
                for row, offset in zip(weights, bias)
            ]
            if activation == "relu":
                output = [max(0.0, value) for value in output]
            elif activation == "tanh":
                output = [math.tanh(value) for value in output]
            values = output
        if not all(math.isfinite(value) for value in values):
            raise LeafValueError("叶模型产生非有限输出")
        return values[0], values[1]


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1 / (1 + inverse)
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def _softplus(value: float) -> float:
    if value > 30:
        return value
    if value < -30:
        return math.exp(value)
    return math.log1p(math.exp(value))


def quality_gate_failures(metrics: Any) -> list[str]:
    if not isinstance(metrics, dict):
        return ["quality.metrics 缺失"]
    failures: list[str] = []

    def number(name: str) -> float | None:
        value = metrics.get(name)
        if isinstance(value, bool):
            failures.append(f"{name} 缺失或非有限")
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            failures.append(f"{name} 缺失或非有限")
            return None
        if not math.isfinite(result):
            failures.append(f"{name} 缺失或非有限")
            return None
        return result

    checks = [
        ("heldoutExamples", ">=", QUALITY_GATES["minHeldoutExamples"]),
        ("heldoutPublicStates", ">=", QUALITY_GATES["minHeldoutPublicStates"]),
        ("maeBb", "<=", QUALITY_GATES["maxMaeBb"]),
        ("p99AbsoluteErrorBb", "<=", QUALITY_GATES["maxP99AbsoluteErrorBb"]),
        ("coverage95", ">=", QUALITY_GATES["minCoverage95"]),
        ("meanUncertaintyBb", "<=", QUALITY_GATES["maxMeanUncertaintyBb"]),
        ("maxGroupMaeBb", "<=", QUALITY_GATES["maxGroupMaeBb"]),
        ("decisiveNodes", ">=", QUALITY_GATES["minDecisiveNodes"]),
        ("actionAgreement", ">=", QUALITY_GATES["minActionAgreement"]),
        ("strategyL1Median", "<=", QUALITY_GATES["maxStrategyL1Median"]),
        ("strategyL1P95", "<=", QUALITY_GATES["maxStrategyL1P95"]),
        (
            "actionEvRelativeErrorP95",
            "<=",
            QUALITY_GATES["maxActionEvRelativeErrorP95"],
        ),
        ("oodFallbackRecall", ">=", QUALITY_GATES["minOodFallbackRecall"]),
        ("leafActivatedNodes", ">=", QUALITY_GATES["minLeafActivatedNodes"]),
        ("leafActivationRate", ">=", QUALITY_GATES["minLeafActivationRate"]),
    ]
    for name, operator, threshold in checks:
        current = number(name)
        if current is None:
            continue
        failed = current < threshold if operator == ">=" else current > threshold
        if failed:
            failures.append(f"{name}={current:g} 未达到 {operator}{threshold:g}")
    return failures


def offline_quality_gate_failures(metrics: Any) -> list[str]:
    """Return failures unrelated to the production solver A/B run."""
    return [
        failure
        for failure in quality_gate_failures(metrics)
        if failure.split("=", 1)[0].split(" ", 1)[0] not in INTEGRATION_METRIC_NAMES
    ]


def _payload_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def metrics_fingerprint(metrics: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(metrics)).hexdigest()


def build_approval_attestation(
    *,
    payload_fingerprint: str,
    blueprint_fingerprint: str,
    holdout_sha256: str,
    metrics: dict[str, Any],
    teacher_version: str,
    continuation_profiles: list[list[int]],
) -> dict[str, Any]:
    """Bind a production approval to exact code, data, model and configuration."""
    return {
        "format": LEAF_APPROVAL_FORMAT,
        "payloadFingerprint": payload_fingerprint,
        "blueprintFingerprint": blueprint_fingerprint,
        "teacherVersion": teacher_version,
        "holdoutSha256": holdout_sha256,
        "validatorVersion": APPROVAL_VALIDATOR_VERSION,
        "solverConfig": dict(PRODUCTION_SOLVER_CONFIG),
        "cutoffConfig": dict(PRODUCTION_CUTOFF_CONFIG),
        "continuationConfig": {
            "positions": list(PRODUCTION_CONTINUATION_CONFIG["positions"]),
            "continuations": list(PRODUCTION_CONTINUATION_CONFIG["continuations"]),
        },
        "validationConfig": dict(PRODUCTION_VALIDATION_CONFIG),
        "continuationProfilesSha256": hashlib.sha256(
            canonical_json_bytes(continuation_profiles)
        ).hexdigest(),
        "metricsSha256": metrics_fingerprint(metrics),
    }


def validate_approval_attestation(
    value: Any,
    *,
    payload_fingerprint: str,
    metrics: dict[str, Any],
    training: dict[str, Any],
    support: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "format",
        "payloadFingerprint",
        "blueprintFingerprint",
        "teacherVersion",
        "holdoutSha256",
        "validatorVersion",
        "solverConfig",
        "cutoffConfig",
        "continuationConfig",
        "validationConfig",
        "continuationProfilesSha256",
        "metricsSha256",
    }
    try:
        attestation = _require_keys(value, required, "approval attestation")
    except LeafSchemaError as error:
        raise LeafArtifactError(str(error)) from error
    expected = {
        "format": LEAF_APPROVAL_FORMAT,
        "payloadFingerprint": payload_fingerprint,
        "validatorVersion": APPROVAL_VALIDATOR_VERSION,
        "solverConfig": PRODUCTION_SOLVER_CONFIG,
        "cutoffConfig": PRODUCTION_CUTOFF_CONFIG,
        "continuationConfig": PRODUCTION_CONTINUATION_CONFIG,
        "validationConfig": PRODUCTION_VALIDATION_CONFIG,
        "metricsSha256": metrics_fingerprint(metrics),
    }
    for key, expected_value in expected.items():
        if attestation.get(key) != expected_value:
            raise LeafArtifactError(f"approval attestation {key} 不匹配")
    blueprint_fingerprint = attestation.get("blueprintFingerprint")
    if (
        not isinstance(blueprint_fingerprint, str)
        or not 1 <= len(blueprint_fingerprint) <= 128
    ):
        raise LeafArtifactError("approval attestation blueprintFingerprint 非法")
    holdout_sha256 = attestation.get("holdoutSha256")
    if (
        not isinstance(holdout_sha256, str)
        or len(holdout_sha256) != 64
        or any(character not in "0123456789abcdef" for character in holdout_sha256)
    ):
        raise LeafArtifactError("approval attestation holdoutSha256 非法")
    if blueprint_fingerprint != training.get("blueprintFingerprint"):
        raise LeafArtifactError("approval Blueprint 与训练数据版本不一致")
    if attestation.get("teacherVersion") != training.get("teacherVersion"):
        raise LeafArtifactError("approval Teacher 与训练数据版本不一致")
    expected_profiles_hash = hashlib.sha256(
        canonical_json_bytes(support.get("continuationProfiles"))
    ).hexdigest()
    if attestation.get("continuationProfilesSha256") != expected_profiles_hash:
        raise LeafArtifactError("approval continuation support 绑定不匹配")
    production_profile = [
        CONTINUATIONS.index(name)
        for name in PRODUCTION_CONTINUATION_CONFIG["continuations"]
    ]
    if production_profile not in support.get("continuationProfiles", []):
        raise LeafArtifactError("模型 support 不包含生产 continuation profile")
    return attestation


def _manifest_fingerprint(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifestFingerprint"}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


SUPPORT_INTERVALS = (
    "potFraction",
    "currentBetFraction",
    "totalChipsBb",
    "stackFraction",
    "rangeEntropyNormalized",
    "rangeEssFraction",
)


def validate_model_support(value: Any) -> dict[str, Any]:
    required = {
        "streets",
        "activeCounts",
        "continuationProfiles",
        "minimumSupportedCombos",
        "maxInferenceUncertaintyBb",
        *SUPPORT_INTERVALS,
    }
    support = _require_keys(value, required, "leaf model support")
    streets = support["streets"]
    if (
        not isinstance(streets, list)
        or not streets
        or len(streets) != len(set(streets))
        or any(street not in STREETS for street in streets)
    ):
        raise LeafArtifactError("support.streets 非法")
    active_counts = support["activeCounts"]
    if (
        not isinstance(active_counts, list)
        or not active_counts
        or any(isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 6 for count in active_counts)
        or len(active_counts) != len(set(active_counts))
    ):
        raise LeafArtifactError("support.activeCounts 非法")
    continuation_profiles = support["continuationProfiles"]
    if not isinstance(continuation_profiles, list) or not continuation_profiles:
        raise LeafArtifactError("support.continuationProfiles 非法")
    normalized_profiles: set[tuple[int, ...]] = set()
    for profile in continuation_profiles:
        if (
            not isinstance(profile, list)
            or len(profile) != SEAT_COUNT
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or not 0 <= item < len(CONTINUATIONS)
                for item in profile
            )
        ):
            raise LeafArtifactError("support.continuationProfiles 非法")
        normalized_profiles.add(tuple(profile))
    if len(normalized_profiles) != len(continuation_profiles):
        raise LeafArtifactError("support.continuationProfiles 包含重复项")
    minimum_supported = support["minimumSupportedCombos"]
    if isinstance(minimum_supported, bool) or not isinstance(minimum_supported, int) or not 1 <= minimum_supported <= HAND_COUNT:
        raise LeafArtifactError("support.minimumSupportedCombos 非法")
    max_uncertainty = support["maxInferenceUncertaintyBb"]
    if isinstance(max_uncertainty, bool) or not isinstance(max_uncertainty, (int, float)) or not math.isfinite(float(max_uncertainty)) or not 0 < float(max_uncertainty) <= QUALITY_GATES["maxMeanUncertaintyBb"]:
        raise LeafArtifactError("support.maxInferenceUncertaintyBb 非法")
    for name in SUPPORT_INTERVALS:
        interval = support[name]
        if not isinstance(interval, list) or len(interval) != 2:
            raise LeafArtifactError(f"support.{name} 必须是 [min,max]")
        low, high = interval
        if (
            isinstance(low, bool)
            or isinstance(high, bool)
            or not isinstance(low, (int, float))
            or not isinstance(high, (int, float))
            or not math.isfinite(float(low))
            or not math.isfinite(float(high))
            or float(low) > float(high)
        ):
            raise LeafArtifactError(f"support.{name} 区间非法")
    return support


def validate_model_payload(payload: Any) -> dict[str, Any]:
    required = {
        "format",
        "schemaVersion",
        "featureVersion",
        "abstractionVersion",
        "rulesHash",
        "comboOrderHash",
        "seatCount",
        "handCount",
        "continuations",
        "network",
        "support",
        "training",
    }
    model = _require_keys(payload, required, "leaf model payload")
    expected = {
        "format": LEAF_MODEL_FORMAT,
        "schemaVersion": SCHEMA_VERSION,
        "featureVersion": FEATURE_VERSION,
        "abstractionVersion": ABSTRACTION_VERSION,
        "rulesHash": RULES_HASH,
        "comboOrderHash": COMBO_ORDER_HASH,
        "seatCount": SEAT_COUNT,
        "handCount": HAND_COUNT,
        "continuations": list(CONTINUATIONS),
    }
    for key, expected_value in expected.items():
        if model.get(key) != expected_value:
            raise LeafArtifactError(f"叶模型 {key} 不兼容")
    if not isinstance(model["training"], dict):
        raise LeafArtifactError("叶模型 training 元数据非法")
    training = model["training"]
    blueprint_fingerprint = training.get("blueprintFingerprint")
    teacher_version = training.get("teacherVersion")
    headers = training.get("datasetHeaders")
    if not isinstance(blueprint_fingerprint, str) or not blueprint_fingerprint:
        raise LeafArtifactError("叶模型 training.blueprintFingerprint 缺失")
    if not isinstance(teacher_version, str) or not teacher_version:
        raise LeafArtifactError("叶模型 training.teacherVersion 缺失")
    if not isinstance(headers, list) or not headers:
        raise LeafArtifactError("叶模型 training.datasetHeaders 缺失")
    blueprint_versions = {
        str(header.get("blueprintVersion"))
        for header in headers
        if isinstance(header, dict) and header.get("blueprintVersion")
    }
    teacher_versions = {
        str(header.get("teacherVersion"))
        for header in headers
        if isinstance(header, dict) and header.get("teacherVersion")
    }
    if blueprint_versions != {blueprint_fingerprint}:
        raise LeafArtifactError("训练数据包含混合或不匹配的 Blueprint 版本")
    if teacher_versions != {teacher_version}:
        raise LeafArtifactError("训练数据包含混合或不匹配的 Teacher 版本")
    validate_model_support(model["support"])
    PortableDenseNetwork(model["network"])
    return model


def _write_gzip_payload(path: Path, payload: dict[str, Any]) -> None:
    raw_payload = canonical_json_bytes(payload)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write(raw_payload)


def _ab_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _ab_candidate_descriptor(directory: Path, token: str) -> dict[str, Any]:
    if not isinstance(token, str) or len(token) < 32:
        raise LeafArtifactError("A/B candidate token 太短")
    temporary_root = Path(tempfile.gettempdir()).resolve()
    try:
        directory.relative_to(temporary_root)
    except ValueError as error:
        raise LeafArtifactError("A/B candidate 只能发布到系统临时目录") from error
    if directory == temporary_root:
        raise LeafArtifactError("A/B candidate 必须位于独立临时子目录")
    return {
        "format": LEAF_AB_CANDIDATE_FORMAT,
        "directory": str(directory),
        "tokenSha256": _ab_token_hash(token),
    }


def _validate_ab_candidate_descriptor(
    value: Any,
    manifest_path: Path,
) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        descriptor = _require_keys(
            value,
            {"format", "directory", "tokenSha256"},
            "A/B candidate",
        )
    except LeafSchemaError as error:
        raise LeafArtifactError(str(error)) from error
    if descriptor.get("format") != LEAF_AB_CANDIDATE_FORMAT:
        raise LeafArtifactError("A/B candidate format 不兼容")
    directory = descriptor.get("directory")
    if not isinstance(directory, str) or Path(directory).resolve() != manifest_path.parent:
        raise LeafArtifactError("A/B candidate 目录绑定不匹配")
    temporary_root = Path(tempfile.gettempdir()).resolve()
    try:
        manifest_path.parent.relative_to(temporary_root)
    except ValueError as error:
        raise LeafArtifactError("A/B candidate 已离开系统临时目录") from error
    if manifest_path.parent == temporary_root:
        raise LeafArtifactError("A/B candidate 不在独立临时子目录")
    token_hash = descriptor.get("tokenSha256")
    if (
        not isinstance(token_hash, str)
        or len(token_hash) != 64
        or any(character not in "0123456789abcdef" for character in token_hash)
    ):
        raise LeafArtifactError("A/B candidate token hash 非法")
    return descriptor


def _authorize_ab_candidate(
    manifest: dict[str, Any],
    manifest_path: Path,
    token: str | None,
) -> bool:
    descriptor = _validate_ab_candidate_descriptor(
        manifest.get("abCandidate"),
        manifest_path,
    )
    if descriptor is None or token is None:
        return False
    if offline_quality_gate_failures(manifest.get("metrics")):
        return False
    if not isinstance(token, str) or len(token) < 32:
        return False
    return hmac.compare_digest(descriptor["tokenSha256"], _ab_token_hash(token))


def write_leaf_artifact(
    output_directory: str | Path,
    network: dict[str, Any],
    metrics: dict[str, Any],
    training: dict[str, Any],
    *,
    support: dict[str, Any],
    model_filename: str | None = None,
    approval_attestation: dict[str, Any] | None = None,
    ab_candidate_token: str | None = None,
) -> Path:
    base_filename = model_filename or "leaf-value-model.json.gz"
    if Path(base_filename).name != base_filename or not base_filename.endswith(".json.gz"):
        raise LeafArtifactError("model_filename 必须是目录内的 .json.gz 文件名")
    directory = Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": LEAF_MODEL_FORMAT,
        "schemaVersion": SCHEMA_VERSION,
        "featureVersion": FEATURE_VERSION,
        "abstractionVersion": ABSTRACTION_VERSION,
        "rulesHash": RULES_HASH,
        "comboOrderHash": COMBO_ORDER_HASH,
        "seatCount": SEAT_COUNT,
        "handCount": HAND_COUNT,
        "continuations": list(CONTINUATIONS),
        "network": network,
        "support": support,
        "training": dict(training),
    }
    validate_model_payload(payload)
    payload_fingerprint = _payload_fingerprint(payload)
    model_filename = (
        f"{base_filename[:-len('.json.gz')]}-{payload_fingerprint[:16]}.json.gz"
    )
    failures = quality_gate_failures(metrics)
    if approval_attestation is not None and ab_candidate_token is not None:
        raise LeafArtifactError("正式审批和 A/B candidate token 不能同时存在")
    if approval_attestation is not None:
        validate_approval_attestation(
            approval_attestation,
            payload_fingerprint=payload_fingerprint,
            metrics=metrics,
            training=payload["training"],
            support=payload["support"],
        )
        if failures:
            raise LeafArtifactError("未通过质量门槛的叶模型不能附加正式审批")
    ab_candidate = (
        _ab_candidate_descriptor(directory, ab_candidate_token)
        if ab_candidate_token is not None
        else None
    )
    approved = approval_attestation is not None and not failures
    gate_failures = list(failures)
    if approval_attestation is None:
        gate_failures.append("production-approval-attestation-missing")
    model_path = directory / model_filename
    manifest_path = directory / "manifest.json"
    descriptor, temporary_model_name = tempfile.mkstemp(prefix=".leaf-model-", suffix=".tmp", dir=directory)
    os.close(descriptor)
    temporary_model = Path(temporary_model_name)
    descriptor, temporary_manifest_name = tempfile.mkstemp(prefix=".leaf-manifest-", suffix=".tmp", dir=directory)
    os.close(descriptor)
    temporary_manifest = Path(temporary_manifest_name)
    model_existed = model_path.exists()
    model_published = False
    manifest_published = False
    try:
        _write_gzip_payload(temporary_model, payload)
        model_bytes = temporary_model.read_bytes()
        manifest = {
            "format": LEAF_MANIFEST_FORMAT,
            "schemaVersion": SCHEMA_VERSION,
            "featureVersion": FEATURE_VERSION,
            "abstractionVersion": ABSTRACTION_VERSION,
            "rulesHash": RULES_HASH,
            "comboOrderHash": COMBO_ORDER_HASH,
            "modelFile": model_filename,
            "modelSha256": hashlib.sha256(model_bytes).hexdigest(),
            "modelBytes": len(model_bytes),
            "payloadFingerprint": payload_fingerprint,
            "qualityGates": QUALITY_GATES,
            "metrics": dict(metrics),
            "approvalAttestation": approval_attestation,
            "abCandidate": ab_candidate,
            "approved": approved,
            "gateFailures": gate_failures,
        }
        manifest["manifestFingerprint"] = _manifest_fingerprint(manifest)
        temporary_manifest.write_bytes(canonical_json_bytes(manifest))
        os.replace(temporary_model, model_path)
        model_published = True
        os.replace(temporary_manifest, manifest_path)
        manifest_published = True
    finally:
        if temporary_model.exists():
            temporary_model.unlink()
        if temporary_manifest.exists():
            temporary_manifest.unlink()
        if model_published and not manifest_published and not model_existed:
            model_path.unlink(missing_ok=True)
    return manifest_path


def _read_limited(path: Path, maximum: int, label: str) -> bytes:
    size = path.stat().st_size
    if size > maximum:
        raise LeafArtifactError(f"{label} 超过安全大小限制")
    return path.read_bytes()


def _decompress_limited(data: bytes) -> bytes:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as compressed:
            payload = compressed.read(MAX_MODEL_UNCOMPRESSED_BYTES + 1)
    except (gzip.BadGzipFile, EOFError, OSError) as error:
        raise LeafArtifactError("叶模型 gzip 损坏") from error
    if len(payload) > MAX_MODEL_UNCOMPRESSED_BYTES:
        raise LeafArtifactError("叶模型解压后超过安全大小限制")
    return payload


def load_manifest(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = Path(path).resolve()
    try:
        manifest_raw = _read_limited(manifest_path, MAX_MANIFEST_BYTES, "manifest")
        manifest = strict_json_loads(manifest_raw)
    except LeafSchemaError as error:
        raise LeafArtifactError(str(error)) from error
    if not isinstance(manifest, dict):
        raise LeafArtifactError("manifest 必须是对象")
    required_manifest = {
        "format",
        "schemaVersion",
        "featureVersion",
        "abstractionVersion",
        "rulesHash",
        "comboOrderHash",
        "modelFile",
        "modelSha256",
        "modelBytes",
        "payloadFingerprint",
        "qualityGates",
        "metrics",
        "approvalAttestation",
        "abCandidate",
        "approved",
        "gateFailures",
        "manifestFingerprint",
    }
    if set(manifest) != required_manifest:
        raise LeafArtifactError("manifest schema 非法")
    fingerprint = manifest.get("manifestFingerprint")
    if not isinstance(fingerprint, str) or fingerprint != _manifest_fingerprint(manifest):
        raise LeafArtifactError("manifest 指纹校验失败")
    expected = {
        "format": LEAF_MANIFEST_FORMAT,
        "schemaVersion": SCHEMA_VERSION,
        "featureVersion": FEATURE_VERSION,
        "abstractionVersion": ABSTRACTION_VERSION,
        "rulesHash": RULES_HASH,
        "comboOrderHash": COMBO_ORDER_HASH,
        "qualityGates": QUALITY_GATES,
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise LeafArtifactError(f"manifest {key} 不兼容")
    model_filename = manifest.get("modelFile")
    if not isinstance(model_filename, str) or Path(model_filename).name != model_filename:
        raise LeafArtifactError("manifest modelFile 非法")
    model_path = (manifest_path.parent / model_filename).resolve()
    if model_path.parent != manifest_path.parent:
        raise LeafArtifactError("模型文件不能越出 artifact 目录")
    model_raw = _read_limited(model_path, MAX_MODEL_COMPRESSED_BYTES, "叶模型")
    if manifest.get("modelBytes") != len(model_raw):
        raise LeafArtifactError("叶模型大小与 manifest 不一致")
    if manifest.get("modelSha256") != hashlib.sha256(model_raw).hexdigest():
        raise LeafArtifactError("叶模型 SHA-256 校验失败")
    try:
        payload = strict_json_loads(_decompress_limited(model_raw))
    except LeafSchemaError as error:
        raise LeafArtifactError(str(error)) from error
    model = validate_model_payload(payload)
    if manifest.get("payloadFingerprint") != _payload_fingerprint(model):
        raise LeafArtifactError("叶模型 payload 指纹校验失败")
    failures = quality_gate_failures(manifest.get("metrics"))
    attestation = manifest.get("approvalAttestation")
    ab_candidate = _validate_ab_candidate_descriptor(
        manifest.get("abCandidate"),
        manifest_path,
    )
    if attestation is not None and ab_candidate is not None:
        raise LeafArtifactError("正式审批 artifact 不能同时是 A/B candidate")
    if attestation is not None:
        validate_approval_attestation(
            attestation,
            payload_fingerprint=manifest["payloadFingerprint"],
            metrics=manifest["metrics"],
            training=model["training"],
            support=model["support"],
        )
    expected_approved = attestation is not None and not failures
    expected_failures = list(failures)
    if attestation is None:
        expected_failures.append("production-approval-attestation-missing")
    if (
        manifest.get("gateFailures") != expected_failures
        or manifest.get("approved") is not expected_approved
    ):
        raise LeafArtifactError("manifest 质量门槛状态与指标不一致")
    return manifest, model


@dataclass(frozen=True)
class LeafPrediction:
    final_stack: float
    uncertainty: float


class PortableLeafValueModel:
    """Validated leaf model with pure-Python, dependency-free inference."""

    def __init__(
        self,
        manifest: dict[str, Any],
        payload: dict[str, Any],
        *,
        temporary_candidate: bool = False,
    ) -> None:
        self.manifest = manifest
        self.payload = payload
        self.network = PortableDenseNetwork(payload["network"])
        self.approved = bool(manifest["approved"])
        self.temporary_candidate = temporary_candidate
        self.payload_fingerprint = str(manifest["payloadFingerprint"])
        # Approval metrics and attestation affect value error bounds and cache
        # safety even when the weight payload is byte-identical.
        self.version = str(manifest["manifestFingerprint"])

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        *,
        require_approved: bool = True,
        ab_candidate_token: str | None = None,
    ) -> "PortableLeafValueModel":
        manifest, payload = load_manifest(manifest_path)
        temporary_candidate = _authorize_ab_candidate(
            manifest,
            Path(manifest_path).resolve(),
            ab_candidate_token,
        )
        if require_approved and not manifest["approved"] and not temporary_candidate:
            reasons = "; ".join(manifest["gateFailures"])
            raise LeafArtifactNotApproved(f"叶模型未达到启用门槛：{reasons}")
        return cls(
            manifest,
            payload,
            temporary_candidate=temporary_candidate,
        )

    def predict(self, model_input: LeafModelInput, seat_index: int, hand_index: int) -> LeafPrediction:
        features = build_leaf_features(model_input, seat_index, hand_index)
        raw_mean, raw_uncertainty = self.network.predict_raw(features)
        total_chips = _total_chips(model_input.public_state)
        final_stack = _sigmoid(raw_mean) * total_chips
        uncertainty = (_softplus(raw_uncertainty) + 1e-6) * total_chips
        return LeafPrediction(final_stack, uncertainty)

    def predict_hand(self, model_input: LeafModelInput, seat_index: int, hand: tuple[int, int]) -> LeafPrediction:
        ordered = tuple(sorted(hand))
        try:
            hand_index = COMBO_INDEX[ordered]
        except (KeyError, TypeError) as error:
            raise LeafValueError("查询手牌非法") from error
        return self.predict(model_input, seat_index, hand_index)


@dataclass(frozen=True)
class LeafCutoffGate:
    allowed: bool
    reasons: tuple[str, ...]
    model_error_bb: float
    fingerprint: str


def _normalized_entropy(probabilities: Sequence[float]) -> float:
    return -math.fsum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0
    ) / math.log(HAND_COUNT)


def _ess_fraction(probabilities: Sequence[float]) -> float:
    squared = math.fsum(probability * probability for probability in probabilities)
    return 0.0 if squared <= 0 else 1 / squared / HAND_COUNT


class LeafModelRuntime:
    """Online-facing loader and conservative pre-inference cutoff gate."""

    def __init__(self, model: PortableLeafValueModel) -> None:
        self.model = model
        self.fingerprint = model.version
        self.support = validate_model_support(model.payload["support"])
        self.model_error_bb = float(model.manifest["metrics"]["p99AbsoluteErrorBb"])

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        *,
        ab_candidate_token: str | None = None,
    ) -> "LeafModelRuntime":
        return cls(
            PortableLeafValueModel.load(
                manifest_path,
                require_approved=True,
                ab_candidate_token=ab_candidate_token,
            )
        )

    def cutoff_gate(self, model_input: LeafModelInput) -> LeafCutoffGate:
        state = model_input.public_state
        reasons: list[str] = []
        if not (self.model.approved or self.model.temporary_candidate):
            reasons.append("artifact-not-approved")
        if state["street"] not in self.support["streets"]:
            reasons.append("street-out-of-distribution")
        active_count = sum(not player["folded"] for player in state["players"])
        if active_count not in self.support["activeCounts"]:
            reasons.append("active-count-out-of-distribution")
        if list(model_input.continuation_ids) not in self.support["continuationProfiles"]:
            reasons.append("continuation-profile-out-of-distribution")
        total_chips = max(_total_chips(state), 1e-9)
        diagnostics: dict[str, list[float]] = {
            "potFraction": [state["pot"] / total_chips],
            "currentBetFraction": [state["currentBet"] / total_chips],
            "totalChipsBb": [total_chips],
            "stackFraction": [player["stack"] / total_chips for player in state["players"]],
            "rangeEntropyNormalized": [_normalized_entropy(hand_range) for hand_range in model_input.ranges],
            "rangeEssFraction": [_ess_fraction(hand_range) for hand_range in model_input.ranges],
        }
        for name, values in diagnostics.items():
            low, high = (float(bound) for bound in self.support[name])
            tolerance = max(1e-9, (high - low) * 1e-6)
            if any(value < low - tolerance or value > high + tolerance for value in values):
                reasons.append(f"{name}-out-of-distribution")
        minimum_supported = int(self.support["minimumSupportedCombos"])
        if any(sum(probability > 0 for probability in hand_range) < minimum_supported for hand_range in model_input.ranges):
            reasons.append("range-support-out-of-distribution")
        return LeafCutoffGate(not reasons, tuple(reasons), self.model_error_bb, self.fingerprint)

    def predict(self, model_input: LeafModelInput, seat_index: int, hand_index: int) -> LeafPrediction:
        gate = self.cutoff_gate(model_input)
        if not gate.allowed:
            raise LeafCutoffRejected("叶模型 cutoff 被拒绝：" + ", ".join(gate.reasons))
        prediction = self.model.predict(model_input, seat_index, hand_index)
        maximum = float(self.support["maxInferenceUncertaintyBb"])
        if prediction.uncertainty > maximum:
            raise LeafCutoffRejected(
                f"叶模型 uncertainty={prediction.uncertainty:.6f}BB 超过 {maximum:.6f}BB"
            )
        return prediction
