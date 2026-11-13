from __future__ import annotations

import bisect
import hashlib
import math
import os
import random
import threading
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from multiprocessing import get_context
from typing import Any

from .abstraction import RULES
from .online_stats import PairedActionStats, empirical_bernstein_radius
from .poker import evaluate


ACTION_NAMES = {
    "check": "Check",
    "call": "Call",
    "raise": "Raise",
    "fold": "Fold",
}
ACTION_PATH_CODES = {
    "check": 1,
    "call": 2,
    "raise": 3,
    "fold": 4,
}
PATH_RADIX = 6
CHANCE_PATH_CODE = 5
POSITIONS = ("UTG", "HJ", "CO", "BTN", "SB", "BB")
POSTFLOP_ORDER = ("SB", "BB", "UTG", "HJ", "CO", "BTN")
STREETS = ("preflop", "flop", "turn", "river")
EPSILON = 1e-9
_PROCESS_POOL: ProcessPoolExecutor | None = None
_PROCESS_POOL_WORKERS = 0
_PROCESS_POOL_RUNTIME_KEY: tuple[str, str, str] | None = None
_PROCESS_POOL_LOCK = threading.Lock()
try:
    GPU_TRAINING_WORKERS = max(
        1,
        int(os.environ.get("TEXAS_LAB_GPU_TRAINING_WORKERS", "4")),
    )
except ValueError:
    GPU_TRAINING_WORKERS = 4
try:
    PREFLOP_GPU_TRAINING_WORKERS = max(
        1,
        int(os.environ.get("TEXAS_LAB_PREFLOP_GPU_TRAINING_WORKERS", "2")),
    )
except ValueError:
    PREFLOP_GPU_TRAINING_WORKERS = 2
try:
    CPU_TRAINING_WORKERS = max(
        1,
        int(os.environ.get("TEXAS_LAB_CPU_TRAINING_WORKERS", "4")),
    )
except ValueError:
    CPU_TRAINING_WORKERS = 4
GPU_TRAINING_BATCH_SIZE = 64
SOLVER_EXECUTION_VERSION = "mccfr-execution-v2"


def solver_execution_semantics(
    *,
    uses_cuda: bool | None = None,
    leaf_runtime_loaded: bool | None = None,
) -> str:
    """Describe the execution path that changes regret-update semantics."""
    if uses_cuda is None:
        from .gpu import get_batch_evaluator

        uses_cuda = get_batch_evaluator().uses_cuda
    if leaf_runtime_loaded is None:
        from .leaf_runtime import get_leaf_runtime

        leaf_runtime_loaded = get_leaf_runtime() is not None
    if not uses_cuda:
        return f"{SOLVER_EXECUTION_VERSION}:cpu-sequential:batch=1"
    try:
        refresh_interval = max(
            1,
            int(
                os.environ.get(
                    "TEXAS_LAB_LEAF_BELIEF_REFRESH",
                    str(GPU_TRAINING_BATCH_SIZE),
                )
            ),
        )
    except ValueError:
        refresh_interval = GPU_TRAINING_BATCH_SIZE
    effective_batch = (
        min(GPU_TRAINING_BATCH_SIZE, refresh_interval)
        if leaf_runtime_loaded
        else GPU_TRAINING_BATCH_SIZE
    )
    return (
        f"{SOLVER_EXECUTION_VERSION}:cuda-minibatch:"
        f"batch={effective_batch}"
    )


def solver_config_signature(execution_semantics: str | None = None) -> str:
    """Fingerprint every runtime option that can change a solve result."""
    defaults = {
        "TEXAS_LAB_REGRET_MODE": "rmplus",
        "TEXAS_LAB_BLUEPRINT_PRIOR_WEIGHT": "16",
        "TEXAS_LAB_LEAF_CUTOFF_MODE": "auto",
        "TEXAS_LAB_LEAF_DEPTH": "",
        "TEXAS_LAB_LEAF_CONTINUATIONS": "",
        "TEXAS_LAB_LEAF_BELIEF_REFRESH": str(GPU_TRAINING_BATCH_SIZE),
        "TEXAS_LAB_ERROR_BUDGET": "0.05",
        "TEXAS_LAB_ADAPTIVE_TRAINING": "true",
        "TEXAS_LAB_TRAIN_CHUNK_SIZE": "256",
        "TEXAS_LAB_TRAIN_MIN_ITERATIONS": "1500",
        "TEXAS_LAB_PREFLOP_TRAIN_MIN_ITERATIONS": "1024",
        "TEXAS_LAB_TRAIN_EV_TOLERANCE": "",
        "TEXAS_LAB_TRAIN_STABLE_CHECKS": "2",
        "TEXAS_LAB_TRAIN_DOMINANCE": "true",
        "TEXAS_LAB_TRAIN_DOMINANCE_MARGIN": "",
        "TEXAS_LAB_TRAIN_DOMINANCE_CHECKS": "1",
        "TEXAS_LAB_TRAIN_DOMINANCE_DRIFT": "2",
        "TEXAS_LAB_TRAIN_BEST_FREQUENCY": "0.70",
        "TEXAS_LAB_PREFLOP_DOMINATED_EVAL_TOTAL": "2000",
        "TEXAS_LAB_ADAPTIVE_EVALUATION": "true",
        "TEXAS_LAB_EVAL_CHUNK_SIZE": "128",
        "TEXAS_LAB_EVAL_MIN_SAMPLES": "128",
        "TEXAS_LAB_EVAL_SEQUENTIAL_Z": "2.58",
        "TEXAS_LAB_EVAL_ALPHA": "0.05",
        "TEXAS_LAB_EVAL_TARGET_CI": "",
        "TEXAS_LAB_EVAL_MATERIAL_EDGE": "",
    }
    values = tuple(
        (name, os.environ.get(name, default).strip().lower())
        for name, default in sorted(defaults.items())
    )
    semantics = execution_semantics or solver_execution_semantics()
    payload = repr(
        (
            values,
            CPU_TRAINING_WORKERS,
            GPU_TRAINING_WORKERS,
            PREFLOP_GPU_TRAINING_WORKERS,
            semantics,
        )
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=12).hexdigest()


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except ValueError:
        return max(minimum, default)


def env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, str(default))))
    except ValueError:
        return max(minimum, default)
BUCKET_ENCODING_VERSION = 1


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def preflop_strength(cards: tuple[int, int]) -> float:
    """A range-prior feature, never an EV adjustment."""
    ranks = sorted((2 + card // 4 for card in cards), reverse=True)
    pair = ranks[0] == ranks[1]
    suited = cards[0] % 4 == cards[1] % 4
    gap = ranks[0] - ranks[1]
    score = (ranks[0] + ranks[1]) / 28
    if pair:
        score += 0.24 + ranks[0] / 70
    if suited:
        score += 0.055
    if gap <= 2:
        score += 0.045
    if ranks[0] >= 11 and ranks[1] >= 10:
        score += 0.07
    return clamp(score, 0.04, 0.99)


def visible_strength(cards: tuple[int, int], board: tuple[int, ...] | list[int]) -> float:
    if len(board) < 3:
        return preflop_strength(cards)
    category = evaluate([*cards, *board])[0]
    bases = (0.12, 0.32, 0.50, 0.68, 0.79, 0.86, 0.93, 0.975, 0.995)
    high = max(2 + card // 4 for card in cards) / 14
    return clamp(bases[category] + high * 0.035, 0.03, 0.999)


@dataclass
class WeightedRange:
    combos: list[tuple[int, int]]
    cumulative: list[float]
    total: float


def range_weight(info: dict[str, Any], opponent: dict[str, Any], cards: tuple[int, int]) -> float:
    """Return the public-action likelihood used as the subgame range prior.

    This prior only establishes beliefs at the root of the current subgame. All
    future actions are chosen by the same regret-matching strategy, rather than
    by a separate hard-coded fold/call response model.
    """
    weight = 1.0
    for action in info["history"]:
        if action["playerId"] != opponent["id"]:
            continue
        board = tuple(action.get("board", info["board"]))
        strength = visible_strength(cards, board)
        if action["type"] == "raise":
            weight *= 0.06 + strength**2.4 * 3.5
        elif action["type"] == "call":
            weight *= 0.20 + strength * 1.30
        elif action["type"] == "check":
            weight *= 0.74 + (1 - strength) * 0.28
        elif action["type"] == "fold":
            # Folded cards remain physically unavailable for the runout. Keep
            # a weak-hand-weighted posterior so card removal is represented.
            weight *= 0.08 + (1 - strength) ** 1.8 * 1.65
    return max(0.00001, weight)


def build_range(info: dict[str, Any], opponent: dict[str, Any], available: list[int]) -> WeightedRange:
    combos_list: list[tuple[int, int]] = []
    cumulative: list[float] = []
    total = 0.0
    for combo in combinations(available, 2):
        weight = range_weight(info, opponent, combo)
        total += weight
        combos_list.append(combo)
        cumulative.append(total)
    return WeightedRange(combos_list, cumulative, total)


def sample_combo(hand_range: WeightedRange, used: set[int], rng: random.Random) -> tuple[int, int]:
    for _ in range(64):
        index = bisect.bisect_left(hand_range.cumulative, rng.random() * hand_range.total)
        cards = hand_range.combos[min(index, len(hand_range.combos) - 1)]
        if cards[0] not in used and cards[1] not in used:
            return cards
    valid = [cards for cards in hand_range.combos if cards[0] not in used and cards[1] not in used]
    if not valid:
        raise RuntimeError("没有可用的对手底牌组合")
    return valid[rng.randrange(len(valid))]


def showdown_share(hero: tuple[int, int] | list[int], opponents: list[tuple[int, int]], board: list[int]) -> float:
    hero_score = evaluate([*hero, *board])
    scores = [evaluate([*cards, *board]) for cards in opponents]
    best = max([hero_score, *scores])
    if hero_score < best:
        return 0.0
    return 1 / (1 + sum(score == best for score in scores))


def preflop_bucket(hand: tuple[int, int]) -> tuple[Any, ...]:
    ranks = sorted((2 + card // 4 for card in hand), reverse=True)
    return (ranks[0], ranks[1], hand[0] % 4 == hand[1] % 4)


@lru_cache(maxsize=65_536)
def board_texture(board: tuple[int, ...]) -> tuple[Any, ...]:
    if not board:
        return ()
    ranks = [2 + card // 4 for card in board]
    suits = [card % 4 for card in board]
    rank_pattern = tuple(sorted((ranks.count(rank) for rank in set(ranks)), reverse=True))
    suit_pattern = tuple(sorted((suits.count(suit) for suit in set(suits)), reverse=True))
    unique = set(ranks)
    if 14 in unique:
        unique.add(1)
    connectivity = max(
        sum(rank in unique for rank in range(high - 4, high + 1))
        for high in range(5, 15)
    )
    high_band = max(ranks) // 3
    broadways = sum(rank >= 10 for rank in ranks)
    return (len(board), rank_pattern, suit_pattern, connectivity, high_band, broadways)


@lru_cache(maxsize=262_144)
def hand_bucket(hand: tuple[int, int], board: tuple[int, ...]) -> tuple[Any, ...]:
    """Card abstraction shared by strategically equivalent sampled states.

    Exact card identities at every sampled runout are too sparse for a real-time
    tabular solver. The bucket retains made-hand class, board interaction, draw
    density, overcards and blocker-relevant suit participation.
    """
    if not board:
        return ("preflop", *preflop_bucket(hand))
    hole_ranks = [2 + card // 4 for card in hand]
    board_ranks = [2 + card // 4 for card in board]
    score = evaluate([*hand, *board])
    matched = sum(rank in board_ranks for rank in hole_ranks)
    overcards = sum(rank > max(board_ranks) for rank in hole_ranks)
    pocket_pair = hole_ranks[0] == hole_ranks[1]
    overpair = pocket_pair and hole_ranks[0] > max(board_ranks)

    all_cards = [*hand, *board]
    suit_counts = [sum(card % 4 == suit for card in all_cards) for suit in range(4)]
    flush_count = max(suit_counts)
    dominant_suit = max(range(4), key=lambda suit: suit_counts[suit])
    hole_flush_cards = sum(card % 4 == dominant_suit for card in hand)

    ranks = {2 + card // 4 for card in all_cards}
    if 14 in ranks:
        ranks.add(1)
    straight_count = max(
        sum(rank in ranks for rank in range(high - 4, high + 1))
        for high in range(5, 15)
    )
    kicker_band = max(hole_ranks) // 3
    return (
        "postflop",
        score[0],
        matched,
        overcards,
        pocket_pair,
        overpair,
        min(flush_count, 5),
        hole_flush_cards,
        straight_count,
        kicker_band,
    )


def policy_bucket_groups(hand_features: tuple[Any, ...]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Return two conservative backoff groups for an unseen exact card bucket.

    The first group retains the strategically important hand/draw features. The
    second is used only when that detailed group was not sampled at the same
    public action node. Exact trained information sets always take precedence.
    """
    if hand_features[0] == "preflop":
        _, high, low, suited = hand_features
        pair = high == low
        gap = high - low
        strength = (high + low) / 28
        if pair:
            strength += 0.24 + high / 70
        if suited:
            strength += 0.055
        if gap <= 2:
            strength += 0.045
        if high >= 11 and low >= 10:
            strength += 0.07
        strength = clamp(strength, 0.04, 0.99)
        return (
            ("preflop", min(7, int(strength * 8)), pair, suited, min(2, gap)),
            ("preflop", min(4, int(strength * 5)), pair),
        )

    (
        _,
        category,
        matched,
        overcards,
        _pocket_pair,
        overpair,
        flush_count,
        hole_flush_cards,
        straight_count,
        _kicker_band,
    ) = hand_features
    flush_draw = flush_count >= 4 and hole_flush_cards > 0
    straight_draw = straight_count >= 4
    return (
        (
            "postflop",
            category,
            matched > 0,
            overpair,
            flush_draw,
            straight_draw,
            overcards > 0,
        ),
        ("postflop", category, flush_draw, straight_draw),
    )


def hand_bucket_code(features: tuple[Any, ...]) -> int:
    """Encode a hand abstraction deterministically across solvers/processes."""
    if features[0] == "preflop":
        _, high, low, suited = features
        return 1 | int(high) << 2 | int(low) << 6 | int(bool(suited)) << 10

    (
        _,
        category,
        matched,
        overcards,
        pocket_pair,
        overpair,
        flush_count,
        hole_flush_cards,
        straight_count,
        kicker_band,
    ) = features
    return (
        2
        | int(category) << 2
        | int(matched) << 6
        | int(overcards) << 8
        | int(bool(pocket_pair)) << 10
        | int(bool(overpair)) << 11
        | int(flush_count) << 12
        | int(hole_flush_cards) << 15
        | int(straight_count) << 17
        | int(kicker_band) << 20
    )


def _count_pattern_code(pattern: tuple[int, ...]) -> int:
    code = 0
    for index, count in enumerate(pattern):
        code |= int(count) << (index * 3)
    return code


def board_texture_code(features: tuple[Any, ...]) -> int:
    """Encode a public-card abstraction without discovery-order identifiers."""
    if not features:
        return 0
    length, rank_pattern, suit_pattern, connectivity, high_band, broadways = features
    return (
        int(length)
        | _count_pattern_code(rank_pattern) << 3
        | _count_pattern_code(suit_pattern) << 18
        | int(connectivity) << 33
        | int(high_band) << 36
        | int(broadways) << 39
    )


@dataclass(frozen=True)
class TreePlayer:
    id: str
    position: str
    stack: float
    street_committed: float
    total_committed: float
    folded: bool
    all_in: bool


@dataclass(frozen=True)
class PublicState:
    node_id: int
    street: str
    board: tuple[int, ...]
    pot: float
    current_bet: float
    aggressive_actions: int
    players: tuple[TreePlayer, ...]
    pending: tuple[str, ...]


def clean_amount(value: float) -> float:
    return round(float(value), 2)


def make_state(info: dict[str, Any]) -> PublicState:
    return PublicState(
        node_id=1,
        street=info["street"],
        board=tuple(info["board"]),
        pot=clean_amount(info["pot"]),
        current_bet=clean_amount(info["currentBet"]),
        aggressive_actions=int(info.get("aggressiveActions", 0)),
        players=tuple(
            TreePlayer(
                id=player["id"],
                position=player["position"],
                stack=clean_amount(player["stack"]),
                street_committed=clean_amount(player["streetCommitted"]),
                total_committed=clean_amount(player["totalCommitted"]),
                folded=bool(player["folded"]),
                all_in=bool(player["allIn"]),
            )
            for player in info["players"]
        ),
        pending=tuple(info["pending"]),
    )


def player_index(state: PublicState, player_id: str) -> int:
    return next(index for index, player in enumerate(state.players) if player.id == player_id)


def can_act(player: TreePlayer) -> bool:
    return not player.folded and not player.all_in


def active_players(state: PublicState) -> list[TreePlayer]:
    return [player for player in state.players if not player.folded]


def ordered_after(
    state: PublicState,
    actor_id: str | None = None,
    players: tuple[TreePlayer, ...] | list[TreePlayer] | None = None,
) -> list[TreePlayer]:
    current_players = state.players if players is None else players
    order = POSITIONS if state.street == "preflop" else POSTFLOP_ORDER
    ordered = [next(player for player in current_players if player.position == position) for position in order]
    if actor_id is None:
        return ordered
    index = next(index for index, player in enumerate(ordered) if player.id == actor_id)
    return ordered[index + 1 :] + ordered[: index + 1]


def raise_target(state: PublicState, player: TreePlayer) -> float:
    total_available = player.street_committed + player.stack
    sizing = RULES["betSizing"]
    if state.street == "preflop":
        if state.aggressive_actions == 0:
            target = sizing["preflopOpenBb"]
        elif state.aggressive_actions == 1:
            target = state.current_bet * sizing["preflopFirstReraiseMultiplier"]
        else:
            target = state.current_bet * sizing["preflopLaterReraiseMultiplier"]
    elif state.aggressive_actions == 0:
        target = max(
            sizing["postflopMinimumBetBb"],
            state.pot * sizing["postflopOpenPotFraction"],
        )
    else:
        target = state.current_bet * sizing["postflopRaiseMultiplier"]
    return clean_amount(min(total_available, target))


def legal_actions(state: PublicState) -> tuple[str, ...]:
    if not state.pending:
        return ()
    player = state.players[player_index(state, state.pending[0])]
    to_call = max(0.0, state.current_bet - player.street_committed)
    actions: list[str] = ["check"] if to_call <= EPSILON else ["fold", "call"]
    target = raise_target(state, player)
    if (
        state.aggressive_actions < 3
        and player.stack > to_call + EPSILON
        and target > state.current_bet + EPSILON
    ):
        actions.append("raise")
    return tuple(actions)


def commit(player: TreePlayer, chips: float) -> tuple[TreePlayer, float]:
    paid = min(player.stack, max(0.0, chips))
    stack = clean_amount(player.stack - paid)
    return (
        TreePlayer(
            id=player.id,
            position=player.position,
            stack=stack,
            street_committed=clean_amount(player.street_committed + paid),
            total_committed=clean_amount(player.total_committed + paid),
            folded=player.folded,
            all_in=stack <= EPSILON,
        ),
        clean_amount(paid),
    )


def apply_action(state: PublicState, action: str) -> PublicState:
    actions = legal_actions(state)
    if action not in actions:
        raise ValueError(f"非法树动作：{action}，合法动作为 {actions}")
    return _apply_action_unchecked(state, action)


def _apply_action_unchecked(state: PublicState, action: str) -> PublicState:
    """Apply an action already obtained from this state's legal action set."""
    actor_id = state.pending[0]
    index = player_index(state, actor_id)
    actor = state.players[index]
    players = list(state.players)
    pending = list(state.pending[1:])
    pot = state.pot
    current_bet = state.current_bet
    aggressive_actions = state.aggressive_actions

    if action == "fold":
        players[index] = TreePlayer(
            id=actor.id,
            position=actor.position,
            stack=actor.stack,
            street_committed=actor.street_committed,
            total_committed=actor.total_committed,
            folded=True,
            all_in=actor.all_in,
        )
    elif action == "call":
        players[index], paid = commit(actor, current_bet - actor.street_committed)
        pot = clean_amount(pot + paid)
    elif action == "raise":
        target = raise_target(state, actor)
        players[index], paid = commit(actor, target - actor.street_committed)
        pot = clean_amount(pot + paid)
        current_bet = players[index].street_committed
        aggressive_actions += 1
        pending = [
            player.id
            for player in ordered_after(state, actor_id, players)
            if player.id != actor_id and can_act(player)
        ]

    return PublicState(
        node_id=state.node_id * PATH_RADIX + ACTION_PATH_CODES[action],
        street=state.street,
        board=state.board,
        pot=pot,
        current_bet=current_bet,
        aggressive_actions=aggressive_actions,
        players=tuple(players),
        pending=tuple(pending),
    )


def cards_for_next_street(state: PublicState) -> int:
    return 3 if state.street == "preflop" else 1


def advance_street(state: PublicState, cards: tuple[int, ...]) -> PublicState:
    next_street = STREETS[STREETS.index(state.street) + 1]
    players = tuple(
        TreePlayer(
            id=player.id,
            position=player.position,
            stack=player.stack,
            street_committed=0.0,
            total_committed=player.total_committed,
            folded=player.folded,
            all_in=player.all_in,
        )
        for player in state.players
    )
    order = POSITIONS if next_street == "preflop" else POSTFLOP_ORDER
    pending = tuple(
        player.id
        for position in order
        for player in players
        if player.position == position and can_act(player)
    )
    return PublicState(
        node_id=state.node_id * PATH_RADIX + CHANCE_PATH_CODE,
        street=next_street,
        board=(*state.board, *cards),
        pot=state.pot,
        current_bet=0.0,
        aggressive_actions=0,
        players=players,
        pending=pending,
    )


def is_terminal(state: PublicState) -> bool:
    if len(active_players(state)) == 1:
        return True
    return state.street == "river" and not state.pending


def pot_awards(state: PublicState, hands: dict[str, tuple[int, int]]) -> dict[str, float]:
    active = active_players(state)
    if len(active) == 1:
        return {active[0].id: state.pot}

    levels = sorted({player.total_committed for player in state.players if player.total_committed > EPSILON})
    awards: dict[str, float] = {}
    previous = 0.0
    for level in levels:
        contributors = [player for player in state.players if player.total_committed + EPSILON >= level]
        amount = (level - previous) * len(contributors)
        previous = level
        eligible = [player for player in contributors if not player.folded]
        if not eligible:
            continue
        scores = {player.id: evaluate([*hands[player.id], *state.board]) for player in eligible}
        best = max(scores.values())
        winners = [player_id for player_id, score in scores.items() if score == best]
        share = amount / len(winners)
        for winner in winners:
            awards[winner] = awards.get(winner, 0.0) + share
    return awards


@dataclass
class InfoSet:
    actions: tuple[str, ...]
    regrets: list[float]
    strategy_sum: list[float]
    visits: int = 0
    regret_updates: int = 0
    blueprint_prior: bool = False
    regret_weight_sum: float = 0.0

    def strategy(self) -> list[float]:
        positive = [max(0.0, regret) for regret in self.regrets]
        total = sum(positive)
        if total <= EPSILON:
            return [1 / len(self.actions)] * len(self.actions)
        return [value / total for value in positive]

    def average_strategy(self) -> list[float]:
        total = sum(self.strategy_sum)
        if total <= EPSILON:
            return self.strategy()
        return [value / total for value in self.strategy_sum]


@dataclass
class DeferredDecision:
    """A traverser decision whose terminal utilities are resolved as one batch."""

    node: InfoSet
    strategy: tuple[float, ...]
    children: tuple[Any, ...]
    regret_weight: float = 1.0
    record_root_action_values: bool = False


@dataclass
class TrainingConvergence:
    """Stop a training budget after root action EVs remain stable.

    The snapshots are cumulative counterfactual values collected by normal
    Hero-root MCCFR traversals, so convergence checks add no rollout tree walk.
    Requiring consecutive stable checks avoids pruning the remaining budget on
    one lucky chunk.
    """

    actions: tuple[str, ...]
    minimum_iterations: int
    tolerance: float
    patience: int
    dominance_margin: float | None = None
    dominance_patience: int = 1
    dominance_drift: float = 2.0
    minimum_best_frequency: float = 0.70
    previous: tuple[float, ...] | None = None
    previous_samples: int = 0
    checks: int = 0
    stable_checks: int = 0
    decision_stable_checks: int = 0
    maximum_change: float | None = None
    best_action: str | None = None
    best_frequency: float | None = None
    minimum_gap: float | None = None
    stop_reason: str | None = None

    def inspect(
        self,
        iterations: int,
        samples: int,
        values: dict[str, float],
        frequencies: dict[str, float] | None = None,
    ) -> bool:
        """Return true when all EVs stabilize or one decision safely separates."""
        if samples <= self.previous_samples:
            return False
        if set(values) != set(self.actions):
            raise ValueError("training EV snapshot must contain every root action")
        current = tuple(float(values[action]) for action in self.actions)
        if not all(math.isfinite(value) for value in current):
            raise ValueError("training EV snapshot contains a non-finite value")

        self.checks += 1
        self.previous_samples = samples
        if self.previous is None:
            self.previous = current
            return False

        previous = self.previous
        changes = tuple(
            abs(current_value - previous_value)
            for current_value, previous_value in zip(current, previous)
        )
        self.maximum_change = max(changes)
        self.previous = current
        if (
            iterations >= self.minimum_iterations
            and self.maximum_change <= self.tolerance
        ):
            self.stable_checks += 1
        else:
            self.stable_checks = 0
        if self.stable_checks >= self.patience:
            self.stop_reason = "ev-stable"
            return True

        best_index = max(range(len(current)), key=current.__getitem__)
        best_action = self.actions[best_index]
        previous_best_index = max(range(len(previous)), key=previous.__getitem__)
        best_frequency = (
            float(frequencies[best_action])
            if frequencies is not None and best_action in frequencies
            else None
        )
        gaps = [
            current[best_index] - current[index]
            for index in range(len(current))
            if index != best_index
        ]
        self.best_action = best_action
        self.best_frequency = best_frequency
        self.minimum_gap = min(gaps, default=math.inf)
        dominance_ready = bool(
            self.dominance_margin is not None
            and iterations >= self.minimum_iterations
            and previous_best_index == best_index
            and best_frequency is not None
            and best_frequency >= self.minimum_best_frequency
            and all(
                current[best_index] - current[index]
                > self.dominance_margin
                + self.dominance_drift * (changes[best_index] + changes[index])
                for index in range(len(current))
                if index != best_index
            )
        )
        if dominance_ready:
            self.decision_stable_checks += 1
        else:
            self.decision_stable_checks = 0
        if self.decision_stable_checks >= self.dominance_patience:
            self.stop_reason = "action-dominated"
            return True
        return False


def sample_index(
    probabilities: list[float] | tuple[float, ...],
    rng: random.Random,
) -> int:
    roll = rng.random()
    for index, probability in enumerate(probabilities):
        roll -= probability
        if roll <= 0:
            return index
    return len(probabilities) - 1


class SubgameCFR:
    def __init__(
        self,
        root: PublicState,
        root_hands: dict[str, tuple[int, int]],
        ranges: dict[str, WeightedRange],
        seed: int,
        *,
        root_info: dict[str, Any] | None = None,
        blueprint_runtime: Any | None = None,
        regret_mode: str | None = None,
    ) -> None:
        self.root = root
        self.root_hands = root_hands
        self.root_actor = next(iter(root_hands))
        self.ranges = ranges
        self.rng = random.Random(seed)
        configured_regret_mode = (
            regret_mode
            if regret_mode is not None
            else os.environ.get("TEXAS_LAB_REGRET_MODE", "rmplus")
        ).strip().lower()
        if configured_regret_mode not in {"rmplus", "linear"}:
            raise ValueError("TEXAS_LAB_REGRET_MODE must be rmplus or linear")
        self.regret_mode = configured_regret_mode
        self.root_info = root_info
        if blueprint_runtime is None and root_info is not None:
            from .blueprint import get_blueprint_runtime

            blueprint_runtime = get_blueprint_runtime()
        self.blueprint_runtime = blueprint_runtime
        try:
            self.blueprint_prior_weight = max(
                0.0,
                float(os.environ.get("TEXAS_LAB_BLUEPRINT_PRIOR_WEIGHT", "16")),
            )
        except ValueError:
            self.blueprint_prior_weight = 16.0
        self.blueprint_policy_seeds = 0
        self.blueprint_policy_failures = 0
        self.infosets: dict[tuple[Any, ...], InfoSet] = {}
        self.reach_policy_snapshot: dict[tuple[Any, ...], tuple[float, ...]] = {}
        self.depth_evaluator = None
        self.leaf_belief_refresh_interval = env_int(
            "TEXAS_LAB_LEAF_BELIEF_REFRESH",
            GPU_TRAINING_BATCH_SIZE,
        )
        if root_info is not None:
            from .leaf_runtime import get_leaf_runtime

            leaf_runtime = get_leaf_runtime()
            if leaf_runtime is not None:
                from .depth_limit import DepthLimitedEvaluator

                self.depth_evaluator = DepthLimitedEvaluator(
                    root,
                    root_info,
                    ranges,
                    blueprint_runtime,
                    leaf_runtime,
                    policy_provider=self.peek_range_policies,
                )
        self.baseline_stacks = {player.id: player.stack for player in root.players}
        self.root_strategy_sum = [0.0] * len(legal_actions(root))
        self.root_strategy_weight = 0.0
        self.root_action_value_sums = [0.0] * len(legal_actions(root))
        self.root_action_value_samples = 0
        self.training_traversers = tuple(
            player.id for player in active_players(root) if not player.all_in
        )
        self.training_iteration = 0
        self.evaluation_mode = False
        self.evaluation_infoset_lookups = 0
        self.evaluation_untrained_lookups = 0
        self.evaluation_policy_backoff_lookups = 0
        self.evaluation_blueprint_policy_lookups = 0
        self.evaluation_uniform_fallback_lookups = 0
        self.evaluation_policy_fallbacks: dict[tuple[Any, ...], tuple[float, ...]] = {}
        self.evaluation_strategy_cache: dict[
            int,
            tuple[tuple[float, ...], str],
        ] = {}
        self.action_transitions: dict[tuple[int, str], PublicState] = {}
        self.action_transition_hits = 0
        self.action_transition_misses = 0
        self.hand_bucket_ids: dict[tuple[Any, ...], int] = {}
        self.hand_bucket_features: dict[int, tuple[Any, ...]] = {}
        self.board_texture_ids: dict[tuple[Any, ...], int] = {}

    def blueprint_prior(
        self,
        state: PublicState,
        hands: dict[str, tuple[int, int]],
        actions: tuple[str, ...],
    ) -> tuple[float, ...] | None:
        """Return a validated offline policy prior for this information set."""
        runtime = self.blueprint_runtime
        if (
            runtime is None
            or self.root_info is None
            or self.blueprint_prior_weight <= EPSILON
            or not getattr(runtime, "has_valid_policy", False)
        ):
            return None
        try:
            from .blueprint import encode_state_features

            actor_id = state.pending[0]
            features = encode_state_features(
                state,
                hands,
                actor_id,
                root_history=self.root_info["history"],
                root_street=self.root.street,
            )
            prior = runtime.policy(features, actions)
        except Exception:
            # A feature/model mismatch must never corrupt or abort an online
            # solve. Required-mode artifact loading itself still fails loudly.
            self.blueprint_policy_failures += 1
            return None
        if prior is not None:
            self.blueprint_policy_seeds += 1
        return prior

    def peek_range_policies(
        self,
        state: PublicState,
        actor_id: str,
        hand_indices: list[int],
        feature_rows: list[list[float]],
        actions: tuple[str, ...],
    ) -> tuple[list[tuple[float, ...]], int] | None:
        """Read a frozen hand-by-action policy matrix without creating nodes.

        Reach propagation must follow the policy currently being resolved.
        Existing abstract infosets therefore take precedence; only unseen
        buckets use the validated Blueprint prior.  This method is deliberately
        read-only so evaluating 1326 hands cannot change CFR visit counts,
        regrets, or average-strategy mass.
        """
        if actor_id != state.pending[0] or len(hand_indices) != len(feature_rows):
            return None
        from .belief import ALL_COMBOS

        policies: list[tuple[float, ...] | None] = [None] * len(hand_indices)
        missing_positions: list[int] = []
        existing = 0
        for position, hand_index in enumerate(hand_indices):
            try:
                hand = ALL_COMBOS[hand_index]
                key = self.infoset_key(state, actor_id, hand)
            except (IndexError, KeyError, ValueError):
                return None
            policy = self.reach_policy_snapshot.get(key)
            if policy is None:
                missing_positions.append(position)
                continue
            node = self.infosets.get(key)
            if node is None:
                return None
            if node.actions != actions:
                return None
            policies[position] = policy
            existing += 1

        if missing_positions:
            runtime = self.blueprint_runtime
            priors = None
            if (
                self.blueprint_prior_weight > EPSILON
                and runtime is not None
                and getattr(runtime, "has_valid_policy", False)
            ):
                try:
                    priors = runtime.policy_many(
                        [feature_rows[position] for position in missing_positions],
                        actions,
                    )
                except Exception:
                    self.blueprint_policy_failures += 1
            if priors is None or len(priors) != len(missing_positions):
                # This exactly matches get_infoset(): when the prior is
                # disabled or cannot be evaluated, a new node starts uniform.
                uniform = tuple(1.0 / len(actions) for _ in actions)
                priors = [uniform] * len(missing_positions)
            for position, prior in zip(missing_positions, priors):
                policies[position] = tuple(float(value) for value in prior)

        if any(policy is None for policy in policies):
            return None
        return [policy for policy in policies if policy is not None], existing

    def refresh_leaf_reach(self) -> None:
        """Freeze all existing short policy vectors for one reach epoch."""
        snapshot: dict[tuple[Any, ...], tuple[float, ...]] = {}
        for key, node in self.infosets.items():
            if self.evaluation_mode:
                if node.regret_updates > 0 or sum(node.strategy_sum) > EPSILON:
                    policy = node.average_strategy()
                else:
                    player_id, node_id, hand_id, _texture_id = key
                    policy = None
                    for tier, group in enumerate(
                        policy_bucket_groups(self.hand_bucket_features[hand_id])
                    ):
                        policy = self.evaluation_policy_fallbacks.get(
                            (player_id, node_id, tier, group)
                        )
                        if policy is not None:
                            break
                    if policy is None:
                        policy = node.strategy()
            else:
                policy = node.strategy()
            snapshot[key] = tuple(float(value) for value in policy)
        self.reach_policy_snapshot = snapshot
        if self.depth_evaluator is not None:
            self.depth_evaluator.refresh_beliefs()

    def sample_deal(self, *, fix_root_hand: bool = True) -> tuple[dict[str, tuple[int, int]], tuple[int, ...]]:
        """Sample private hands and one shared public runout.

        Sampling the complete public runout once makes every counterfactual
        action in a traversal see the same chance outcome. This preserves the
        correct marginal card distribution while substantially reducing the
        variance of action differences.
        """
        hands = dict(self.root_hands) if fix_root_hand else {}
        used = {*self.root.board}
        for cards in hands.values():
            used.update(cards)
        # Folded players' private cards are still out of the deck. Sampling all
        # seats fixes the common but material error of returning mucked cards
        # to future public runouts.
        for player in self.root.players:
            if player.id in hands:
                continue
            if player.id not in self.ranges:
                continue
            cards = sample_combo(self.ranges[player.id], used, self.rng)
            hands[player.id] = cards
            used.update(cards)
        deck = tuple(card for card in range(52) if card not in used)
        missing = 5 - len(self.root.board)
        return hands, tuple(self.rng.sample(deck, missing))

    def infoset_key(self, state: PublicState, player_id: str, hand: tuple[int, int]) -> tuple[Any, ...]:
        hand_features = hand_bucket(hand, state.board)
        hand_id = self.hand_bucket_ids.get(hand_features)
        if hand_id is None:
            hand_id = hand_bucket_code(hand_features)
            self.hand_bucket_ids[hand_features] = hand_id
            previous = self.hand_bucket_features.setdefault(hand_id, hand_features)
            if previous != hand_features:
                raise RuntimeError("hand bucket encoding collision")

        texture_features = board_texture(state.board)
        texture_id = self.board_texture_ids.get(texture_features)
        if texture_id is None:
            texture_id = board_texture_code(texture_features)
            self.board_texture_ids[texture_features] = texture_id

        return (
            player_id,
            state.node_id,
            hand_id,
            texture_id,
        )

    def action_state(self, state: PublicState, action: str) -> PublicState:
        """Return a deterministic transition from a board-independent template."""
        key = (state.node_id, action)
        cached = self.action_transitions.get(key)
        if cached is not None:
            self.action_transition_hits += 1
            if cached.board == state.board:
                child = cached
            else:
                child = PublicState(
                    node_id=cached.node_id,
                    street=cached.street,
                    board=state.board,
                    pot=cached.pot,
                    current_bet=cached.current_bet,
                    aggressive_actions=cached.aggressive_actions,
                    players=cached.players,
                    pending=cached.pending,
                )
        else:
            child = _apply_action_unchecked(state, action)
            self.action_transitions[key] = child
            self.action_transition_misses += 1
        if self.depth_evaluator is not None:
            self.depth_evaluator.register_action(
                state,
                child,
                action,
                legal_actions(state),
            )
        return child

    def get_infoset(self, state: PublicState, hands: dict[str, tuple[int, int]]) -> InfoSet:
        player_id = state.pending[0]
        key = self.infoset_key(state, player_id, hands[player_id])
        node = self.infosets.get(key)
        if self.evaluation_mode:
            self.evaluation_infoset_lookups += 1
            if node is None or node.regret_updates == 0:
                self.evaluation_untrained_lookups += 1
        if node is None:
            actions = legal_actions(state)
            prior = self.blueprint_prior(state, hands, actions)
            if prior is None:
                regrets = [0.0] * len(actions)
                strategy_sum = [0.0] * len(actions)
            else:
                regrets = [self.blueprint_prior_weight * value for value in prior]
                strategy_sum = [self.blueprint_prior_weight * value for value in prior]
            node = InfoSet(
                actions,
                regrets,
                strategy_sum,
                blueprint_prior=prior is not None,
            )
            self.infosets[key] = node
        return node

    def begin_evaluation(self) -> None:
        fallback_masses: dict[tuple[Any, ...], list[float]] = {}
        for (player_id, node_id, hand_id, _texture_id), node in self.infosets.items():
            if sum(node.strategy_sum) <= EPSILON:
                continue
            groups = policy_bucket_groups(self.hand_bucket_features[hand_id])
            for tier, group in enumerate(groups):
                key = (player_id, node_id, tier, group)
                mass = fallback_masses.get(key)
                if mass is None:
                    mass = [0.0] * len(node.actions)
                    fallback_masses[key] = mass
                for index, value in enumerate(node.strategy_sum):
                    mass[index] += value
        self.evaluation_policy_fallbacks = {
            key: tuple(value / sum(mass) for value in mass)
            for key, mass in fallback_masses.items()
            if sum(mass) > EPSILON
        }
        self.evaluation_mode = True
        self.evaluation_infoset_lookups = 0
        self.evaluation_untrained_lookups = 0
        self.evaluation_policy_backoff_lookups = 0
        self.evaluation_blueprint_policy_lookups = 0
        self.evaluation_uniform_fallback_lookups = 0
        self.evaluation_strategy_cache.clear()
        if self.depth_evaluator is not None:
            self.refresh_leaf_reach()
            self.depth_evaluator.begin_evaluation()

    def end_evaluation(self) -> None:
        """Return an incremental solver to training semantics."""
        if not self.evaluation_mode:
            return
        self.evaluation_mode = False
        self.evaluation_strategy_cache.clear()
        if self.depth_evaluator is not None:
            self.depth_evaluator.end_evaluation()
            self.refresh_leaf_reach()

    def evaluation_strategy(
        self,
        state: PublicState,
        hands: dict[str, tuple[int, int]],
        node: InfoSet,
        *,
        record_lookup: bool = True,
    ) -> list[float] | tuple[float, ...]:
        """Use an exact policy when available, then hand-aware trained backoffs."""
        cached = self.evaluation_strategy_cache.get(id(node))
        if cached is None:
            if node.regret_updates > 0 or sum(node.strategy_sum) > EPSILON:
                source = "blueprint" if node.blueprint_prior and node.regret_updates == 0 else "exact"
                cached = (tuple(node.average_strategy()), source)
            else:
                player_id = state.pending[0]
                features = hand_bucket(hands[player_id], state.board)
                for tier, group in enumerate(policy_bucket_groups(features)):
                    strategy = self.evaluation_policy_fallbacks.get(
                        (player_id, state.node_id, tier, group)
                    )
                    if strategy is not None:
                        cached = (strategy, "backoff")
                        break
                else:
                    cached = (tuple(node.strategy()), "uniform")
            self.evaluation_strategy_cache[id(node)] = cached

        strategy, source = cached
        if record_lookup:
            if source == "backoff":
                self.evaluation_policy_backoff_lookups += 1
            elif source == "blueprint":
                self.evaluation_blueprint_policy_lookups += 1
            elif source == "uniform":
                self.evaluation_uniform_fallback_lookups += 1
        return strategy

    def root_average_strategy(self, root_node: InfoSet) -> list[float]:
        if self.root_strategy_weight <= EPSILON:
            return root_node.strategy()
        return [value / self.root_strategy_weight for value in self.root_strategy_sum]

    def record_root_action_values(self, values: list[float]) -> None:
        if len(values) != len(self.root_action_value_sums):
            raise ValueError("root action values do not match the root action count")
        for index, value in enumerate(values):
            self.root_action_value_sums[index] += float(value)
        self.root_action_value_samples += 1

    def root_action_ev_snapshot(self) -> tuple[int, dict[str, float]]:
        """Return cumulative root counterfactual EVs gathered during training."""
        actions = legal_actions(self.root)
        if self.root_action_value_samples <= 0:
            return 0, {}
        return self.root_action_value_samples, {
            action: self.root_action_value_sums[index]
            / self.root_action_value_samples
            for index, action in enumerate(actions)
        }

    def accumulate_average_strategy(
        self,
        state: PublicState,
        actor_id: str,
        traverser: str,
        average_player: str,
        average_weight: float,
        node: InfoSet,
        strategy: list[float] | tuple[float, ...],
    ) -> None:
        """Accumulate the sampled average policy without a full tree pass.

        The multiplayer rule mirrors OpenSpiel's simple external-sampling
        average: player i's regret pass updates player i+1's average policy.
        The exact root hand is a special query state, so its strategy is also
        accumulated on the root actor's own pass with CFR+ style linear weight.
        """
        if actor_id == average_player:
            for index, probability in enumerate(strategy):
                node.strategy_sum[index] += average_weight * probability
        if state == self.root and actor_id == self.root_actor and traverser == self.root_actor:
            for index, probability in enumerate(strategy):
                self.root_strategy_sum[index] += average_weight * probability
            self.root_strategy_weight += average_weight

    def terminal_utility(self, state: PublicState, hands: dict[str, tuple[int, int]], traverser: str) -> float:
        awards = pot_awards(state, hands)
        player = state.players[player_index(state, traverser)]
        final_stack = player.stack + awards.get(traverser, 0.0)
        return final_stack - self.baseline_stacks[traverser]

    def terminal_record(
        self,
        state: PublicState,
        hands: dict[str, tuple[int, int]],
        target: str,
    ) -> tuple[Any, ...]:
        """Encode a terminal without evaluating showdown hands on the CPU."""
        if len(active_players(state)) == 1:
            return ("direct", self.terminal_utility(state, hands, target))
        rows = tuple(
            (
                player.id,
                player.stack,
                player.total_committed,
                player.folded,
                hands[player.id][0] if player.id in hands else -1,
                hands[player.id][1] if player.id in hands else -1,
            )
            for player in state.players
        )
        return ("showdown", state.board, rows, target, self.baseline_stacks[target])

    def chance_step(self, state: PublicState, runout: tuple[int, ...]) -> tuple[PublicState, tuple[int, ...]]:
        count = cards_for_next_street(state)
        cards = runout[:count]
        if len(cards) != count:
            raise RuntimeError("公共牌 runout 长度不足")
        child = advance_street(state, cards)
        if self.depth_evaluator is not None:
            self.depth_evaluator.register_chance(state, child)
        return child, runout[count:]

    def depth_limited_utility(
        self,
        state: PublicState,
        hands: dict[str, tuple[int, int]],
        target: str,
    ) -> float | None:
        if self.depth_evaluator is None:
            return None
        result = self.depth_evaluator.maybe_value(
            state,
            hands,
            target,
            self.baseline_stacks[target],
        )
        return None if result is None else result.utility

    def traverse(
        self,
        state: PublicState,
        hands: dict[str, tuple[int, int]],
        runout: tuple[int, ...],
        traverser: str,
        average_player: str,
        average_weight: float,
    ) -> float:
        if is_terminal(state):
            return self.terminal_utility(state, hands, traverser)
        leaf_utility = self.depth_limited_utility(state, hands, traverser)
        if leaf_utility is not None:
            return leaf_utility
        if not state.pending:
            next_state, next_runout = self.chance_step(state, runout)
            return self.traverse(
                next_state,
                hands,
                next_runout,
                traverser,
                average_player,
                average_weight,
            )

        actor_id = state.pending[0]
        node = self.get_infoset(state, hands)
        strategy = node.strategy()
        node.visits += 1
        self.accumulate_average_strategy(
            state,
            actor_id,
            traverser,
            average_player,
            average_weight,
            node,
            strategy,
        )

        if actor_id != traverser:
            action_index = sample_index(strategy, self.rng)
            return self.traverse(
                self.action_state(state, node.actions[action_index]),
                hands,
                runout,
                traverser,
                average_player,
                average_weight,
            )

        action_utilities = [
            self.traverse(
                self.action_state(state, action),
                hands,
                runout,
                traverser,
                average_player,
                average_weight,
            )
            for action in node.actions
        ]
        if state == self.root and traverser == self.root_actor:
            self.record_root_action_values(action_utilities)
        node_utility = sum(probability * utility for probability, utility in zip(strategy, action_utilities))
        node.regret_updates += 1
        node.regret_weight_sum += average_weight if self.regret_mode == "linear" else 1.0
        for index, utility in enumerate(action_utilities):
            delta = utility - node_utility
            if self.regret_mode == "linear":
                node.regrets[index] += average_weight * delta
            else:
                # RM+ keeps cumulative regrets non-negative and is more stable
                # at the small real-time budget.
                node.regrets[index] = max(0.0, node.regrets[index] + delta)
        return node_utility

    def train(self, iterations: int) -> None:
        self.end_evaluation()
        traversers = self.training_traversers
        if not traversers:
            return
        start_iteration = self.training_iteration
        for offset in range(iterations):
            iteration = start_iteration + offset
            if (
                self.depth_evaluator is not None
                and iteration % self.leaf_belief_refresh_interval == 0
            ):
                self.refresh_leaf_reach()
            traverser_index = iteration % len(traversers)
            traverser = traversers[traverser_index]
            average_player = traversers[(traverser_index + 1) % len(traversers)]
            average_weight = 1 + iteration // len(traversers)
            # The acting player's exact hand is fixed only while updating that
            # player's regrets. Other players train against the actor's full
            # root range, so their policy cannot implicitly see the hole cards.
            hands, runout = self.sample_deal(fix_root_hand=traverser == self.root_actor)
            self.traverse(
                self.root,
                hands,
                runout,
                traverser,
                average_player,
                average_weight,
            )
            self.training_iteration += 1

    def training_diagnostics(self) -> dict[str, Any]:
        """Return scheduling state needed by incremental training controllers."""
        traverser_count = len(self.training_traversers)
        if traverser_count == 0:
            next_traverser = None
            next_average_player = None
            completed_cycles = 0
            next_average_weight = 0
        else:
            next_index = self.training_iteration % traverser_count
            next_traverser = self.training_traversers[next_index]
            next_average_player = self.training_traversers[
                (next_index + 1) % traverser_count
            ]
            completed_cycles = self.training_iteration // traverser_count
            next_average_weight = 1 + completed_cycles
        return {
            "regretMode": self.regret_mode,
            "trainingIterations": self.training_iteration,
            "traverserCount": traverser_count,
            "completedTraverserCycles": completed_cycles,
            "nextTraverser": next_traverser,
            "nextAveragePlayer": next_average_player,
            "nextAverageWeight": next_average_weight,
            "infosetCount": len(self.infosets),
            "rootStrategyWeight": self.root_strategy_weight,
        }

    def train_chunk(
        self,
        iterations: int,
        *,
        use_gpu: bool = False,
        batch_size: int = GPU_TRAINING_BATCH_SIZE,
    ) -> dict[str, Any]:
        """Continue training without resetting the global MCCFR schedule."""
        start_iteration = self.training_iteration
        gpu_metrics = {"batches": 0, "handEvaluations": 0}
        if use_gpu:
            gpu_metrics = self.train_gpu_batched(iterations, batch_size=batch_size)
        else:
            self.train(iterations)
        diagnostics = self.training_diagnostics()
        diagnostics.update(
            {
                "chunkIterations": self.training_iteration - start_iteration,
                "batches": gpu_metrics["batches"],
                "handEvaluations": gpu_metrics["handEvaluations"],
            }
        )
        return diagnostics

    def build_deferred_traversal(
        self,
        state: PublicState,
        hands: dict[str, tuple[int, int]],
        runout: tuple[int, ...],
        traverser: str,
        average_player: str,
        average_weight: float,
        terminals: list[tuple[Any, ...]],
    ) -> int | DeferredDecision:
        """Build one MCCFR traversal while deferring every terminal utility.

        Regrets are intentionally frozen while a mini-batch is constructed.
        This is mini-batch external-sampling RM+: the action tree and chip
        utility are unchanged, while same-shaped showdown work is large enough
        for a GPU kernel to be useful.
        """
        if is_terminal(state):
            terminals.append(self.terminal_record(state, hands, traverser))
            return len(terminals) - 1
        leaf_utility = self.depth_limited_utility(state, hands, traverser)
        if leaf_utility is not None:
            terminals.append(("direct", leaf_utility))
            return len(terminals) - 1
        if not state.pending:
            next_state, next_runout = self.chance_step(state, runout)
            return self.build_deferred_traversal(
                next_state,
                hands,
                next_runout,
                traverser,
                average_player,
                average_weight,
                terminals,
            )

        actor_id = state.pending[0]
        node = self.get_infoset(state, hands)
        strategy = tuple(node.strategy())
        node.visits += 1
        self.accumulate_average_strategy(
            state,
            actor_id,
            traverser,
            average_player,
            average_weight,
            node,
            strategy,
        )

        if actor_id != traverser:
            action_index = sample_index(strategy, self.rng)
            return self.build_deferred_traversal(
                self.action_state(state, node.actions[action_index]),
                hands,
                runout,
                traverser,
                average_player,
                average_weight,
                terminals,
            )

        children = tuple(
            self.build_deferred_traversal(
                self.action_state(state, action),
                hands,
                runout,
                traverser,
                average_player,
                average_weight,
                terminals,
            )
            for action in node.actions
        )
        node.regret_updates += 1
        regret_weight = average_weight if self.regret_mode == "linear" else 1.0
        node.regret_weight_sum += regret_weight
        return DeferredDecision(
            node,
            strategy,
            children,
            regret_weight,
            state == self.root and traverser == self.root_actor,
        )

    def resolve_deferred_traversal(
        self,
        traversal: int | DeferredDecision,
        terminal_utilities: list[float],
        regret_updates: dict[int, tuple[InfoSet, list[float]]],
    ) -> float:
        if isinstance(traversal, int):
            return terminal_utilities[traversal]
        action_utilities = [
            self.resolve_deferred_traversal(
                child,
                terminal_utilities,
                regret_updates,
            )
            for child in traversal.children
        ]
        if traversal.record_root_action_values:
            self.record_root_action_values(action_utilities)
        node_utility = sum(
            probability * utility
            for probability, utility in zip(traversal.strategy, action_utilities)
        )
        update = regret_updates.get(id(traversal.node))
        if update is None:
            update = (traversal.node, [0.0] * len(traversal.node.actions))
            regret_updates[id(traversal.node)] = update
        for index, utility in enumerate(action_utilities):
            update[1][index] += traversal.regret_weight * (utility - node_utility)
        return node_utility

    def train_gpu_batched(self, iterations: int, batch_size: int = GPU_TRAINING_BATCH_SIZE) -> dict[str, int]:
        """Train RM+ with terminal showdown evaluation batched on CUDA."""
        from .gpu import settle_terminal_records

        self.end_evaluation()
        traversers = self.training_traversers
        if not traversers:
            return {"batches": 0, "handEvaluations": 0}
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        gpu_batches = 0
        gpu_hand_evaluations = 0
        start_iteration = self.training_iteration
        effective_batch_size = (
            min(batch_size, self.leaf_belief_refresh_interval)
            if self.depth_evaluator is not None
            else batch_size
        )
        for start in range(0, iterations, effective_batch_size):
            if self.depth_evaluator is not None:
                # Regrets are frozen while this deferred batch is built, so a
                # single reach snapshot is both coherent and amortized.
                self.refresh_leaf_reach()
            traversals: list[int | DeferredDecision] = []
            terminals: list[tuple[Any, ...]] = []
            batch_iterations = min(iterations, start + effective_batch_size) - start
            for offset in range(batch_iterations):
                iteration = start_iteration + start + offset
                traverser_index = iteration % len(traversers)
                traverser = traversers[traverser_index]
                average_player = traversers[(traverser_index + 1) % len(traversers)]
                average_weight = 1 + iteration // len(traversers)
                hands, runout = self.sample_deal(fix_root_hand=traverser == self.root_actor)
                traversals.append(
                    self.build_deferred_traversal(
                        self.root,
                        hands,
                        runout,
                        traverser,
                        average_player,
                        average_weight,
                        terminals,
                    )
                )

            terminal_utilities, evaluated = settle_terminal_records(terminals)
            gpu_hand_evaluations += evaluated
            gpu_batches += int(evaluated > 0)
            regret_updates: dict[int, tuple[InfoSet, list[float]]] = {}
            for traversal in traversals:
                self.resolve_deferred_traversal(
                    traversal,
                    terminal_utilities,
                    regret_updates,
                )
            for node, deltas in regret_updates.values():
                for index, delta in enumerate(deltas):
                    if self.regret_mode == "linear":
                        node.regrets[index] += delta
                    else:
                        node.regrets[index] = max(0.0, node.regrets[index] + delta)
            self.training_iteration += batch_iterations

        return {
            "batches": gpu_batches,
            "handEvaluations": gpu_hand_evaluations,
        }

    def rollout(
        self,
        state: PublicState,
        hands: dict[str, tuple[int, int]],
        runout: tuple[int, ...],
        target: str,
    ) -> float:
        if is_terminal(state):
            return self.terminal_utility(state, hands, target)
        leaf_utility = self.depth_limited_utility(state, hands, target)
        if leaf_utility is not None:
            return leaf_utility
        if not state.pending:
            next_state, next_runout = self.chance_step(state, runout)
            return self.rollout(next_state, hands, next_runout, target)
        node = self.get_infoset(state, hands)
        strategy = (
            self.evaluation_strategy(state, hands, node)
            if self.evaluation_mode
            else node.average_strategy()
        )
        action_index = sample_index(strategy, self.rng)
        return self.rollout(
            self.action_state(state, node.actions[action_index]),
            hands,
            runout,
            target,
        )

    def rollout_terminal(
        self,
        state: PublicState,
        hands: dict[str, tuple[int, int]],
        runout: tuple[int, ...],
        target: str,
    ) -> tuple[Any, ...]:
        """Sample a policy path and return a compact terminal for batch GPU settlement."""
        if is_terminal(state):
            return self.terminal_record(state, hands, target)
        leaf_utility = self.depth_limited_utility(state, hands, target)
        if leaf_utility is not None:
            return ("direct", leaf_utility)
        if not state.pending:
            next_state, next_runout = self.chance_step(state, runout)
            return self.rollout_terminal(next_state, hands, next_runout, target)
        node = self.get_infoset(state, hands)
        strategy = (
            self.evaluation_strategy(state, hands, node)
            if self.evaluation_mode
            else node.average_strategy()
        )
        action_index = sample_index(strategy, self.rng)
        return self.rollout_terminal(
            self.action_state(state, node.actions[action_index]), hands, runout, target
        )


def sample_statistics(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    return mean, math.sqrt(variance / len(values))


def build_explanation(
    info: dict[str, Any],
    actions: list[dict[str, Any]],
    equity: float,
    root_regret: float,
    evaluation_untrained_rate: float,
    evaluation_policy_backoff_rate: float,
    evaluation_uniform_fallback_rate: float,
    leaf_cutoff_hits: int = 0,
    leaf_model_error_calibration: float = 0.0,
) -> dict[str, Any]:
    best = max(actions, key=lambda row: row["ev"])
    compared = min((row for row in actions if row is not best), key=lambda row: row["ev"], default=best)
    edge = best["ev"] - compared["ev"]
    return {
        "bestAction": best["action"],
        "comparedWith": compared["action"],
        "edge": round(edge, 3),
        "summary": (
            f"在当前离散下注树的平均策略下，{ACTION_NAMES[best['action']]} 的估计 EV 为 "
            f"{best['ev']:.3f}BB；范围摊牌 equity 为 {equity * 100:.1f}%。"
        ),
        "reasons": [
            {
                "label": "后续树与安全叶值",
                "text": (
                    f"搜索在 {leaf_cutoff_hits} 个经范围、分布和不确定性门槛批准的叶节点使用了信念条件终局筹码模型；"
                    f"其预声明 P99/最大不确定性校准 {leaf_model_error_calibration:.3f}BB 已并入动作置信半径。其余分支完整计算到终局。"
                    if leaf_cutoff_hits
                    else "每个动作都会继续计算本街剩余行动、后续公共牌、后续下注、弃牌和摊牌，不再把 Check/Call 直接当作摊牌。"
                ),
            },
            {
                "label": "策略来源",
                "text": "所有玩家在同一子博弈内通过 external-sampling 遗憾匹配更新策略，并使用多人 simple averaging；频率不是 EV softmax，也没有人为最低频率。",
            },
            {
                "label": "收敛诊断",
                "text": f"并行副本的根信息集平均正遗憾诊断值为 {root_regret:.4f}BB；越接近 0，当前抽象下越稳定。该值不是多人博弈 exploitability 上界。",
            },
            {
                "label": "评估覆盖",
                "text": (
                    f"独立 EV rollout 中有 {evaluation_untrained_rate * 100:.1f}% 的精确信息集尚未完成遗憾更新；"
                    f"其中 {evaluation_policy_backoff_rate * 100:.1f}% 使用了同一下注节点的手牌/听牌分层策略，"
                    f"仍有 {evaluation_uniform_fallback_rate * 100:.1f}% 只能使用均匀策略。"
                ),
            },
        ],
        "caveat": "这是固定下注尺度、最多三次加注的子博弈 MCCFR 近似解；历史范围先验和有限采样仍会产生模型误差，不等同于无抽象六人桌精确 GTO。",
    }


def get_process_pool(
    worker_count: int,
    runtime_key: tuple[str, str, str] = ("none", "none", "default"),
) -> ProcessPoolExecutor:
    global _PROCESS_POOL, _PROCESS_POOL_WORKERS, _PROCESS_POOL_RUNTIME_KEY
    with _PROCESS_POOL_LOCK:
        if _PROCESS_POOL is not None and (
            _PROCESS_POOL_WORKERS != worker_count
            or _PROCESS_POOL_RUNTIME_KEY != runtime_key
        ):
            _PROCESS_POOL.shutdown(wait=True, cancel_futures=True)
            _PROCESS_POOL = None
            _PROCESS_POOL_WORKERS = 0
            _PROCESS_POOL_RUNTIME_KEY = None
        if _PROCESS_POOL is None:
            # Spawn is required after the parent probes CUDA; forking an
            # initialized CUDA runtime is unsupported on POSIX and can hang.
            _PROCESS_POOL = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=get_context("spawn"),
            )
            _PROCESS_POOL_WORKERS = worker_count
            _PROCESS_POOL_RUNTIME_KEY = runtime_key
        return _PROCESS_POOL


def shutdown_process_pool() -> None:
    global _PROCESS_POOL, _PROCESS_POOL_WORKERS, _PROCESS_POOL_RUNTIME_KEY
    with _PROCESS_POOL_LOCK:
        if _PROCESS_POOL is not None:
            _PROCESS_POOL.shutdown(wait=True, cancel_futures=True)
            _PROCESS_POOL = None
            _PROCESS_POOL_WORKERS = 0
            _PROCESS_POOL_RUNTIME_KEY = None


def root_action_model_error_calibrations(
    solver: SubgameCFR,
    root: PublicState,
    actions: tuple[str, ...],
) -> dict[str, float]:
    """Return predeclared leaf-error calibration by forced root action.

    The artifact P99 is a calibration statistic rather than a hard bound.  The
    support-domain uncertainty ceiling prevents adaptive stopping from seeing
    only low-uncertainty leaves and understating the configured model risk.
    """
    evaluator = solver.depth_evaluator
    if evaluator is None or not evaluator.active:
        return {action: 0.0 for action in actions}
    cutoff_can_be_reached = bool(
        evaluator.explicit_depth is not None
        or evaluator.mode == "aggressive"
        or root.street == "preflop"
        or (
            root.street == "flop"
            and sum(not player.folded for player in root.players) > 2
        )
    )
    if not cutoff_can_be_reached:
        return {action: 0.0 for action in actions}
    maximum_uncertainty = float(
        evaluator.leaf_runtime.support["maxInferenceUncertaintyBb"]
    )
    calibration = max(
        float(evaluator.leaf_runtime.model_error_bb),
        1.96 * maximum_uncertainty,
    )
    calibrations: dict[str, float] = {}
    for action in actions:
        child = solver.action_state(root, action)
        can_reach_model = bool(
            action != "fold"
            and not is_terminal(child)
            and sum(
                not player.folded and not player.all_in
                for player in child.players
            )
            >= 2
        )
        calibrations[action] = calibration if can_reach_model else 0.0
    return calibrations


def train_to_ev_convergence(
    solver: SubgameCFR,
    requested_iterations: int,
    *,
    use_gpu: bool,
    decision_scale: float,
    error_budget_fraction: float,
) -> dict[str, Any]:
    """Train in chunks and prune the unused budget after root EV convergence."""
    requested = max(0, int(requested_iterations))
    adaptive_requested = env_flag("TEXAS_LAB_ADAPTIVE_TRAINING", True)
    default_minimum = (
        env_int("TEXAS_LAB_PREFLOP_TRAIN_MIN_ITERATIONS", 1_024)
        if solver.root.street == "preflop"
        else env_int("TEXAS_LAB_TRAIN_MIN_ITERATIONS", 1_500)
    )
    minimum_iterations = min(
        requested,
        default_minimum,
    )
    chunk_size = env_int("TEXAS_LAB_TRAIN_CHUNK_SIZE", 256)
    tolerance = env_float(
        "TEXAS_LAB_TRAIN_EV_TOLERANCE",
        max(0.01, error_budget_fraction * decision_scale),
    )
    patience = env_int("TEXAS_LAB_TRAIN_STABLE_CHECKS", 2)
    adaptive_enabled = adaptive_requested and requested > minimum_iterations
    actions = legal_actions(solver.root)
    dominance_enabled = bool(
        adaptive_enabled
        and solver.root.street == "preflop"
        and len(actions) > 1
        and env_flag("TEXAS_LAB_TRAIN_DOMINANCE", True)
    )
    dominance_margin = env_float(
        "TEXAS_LAB_TRAIN_DOMINANCE_MARGIN",
        tolerance,
    )
    dominance_patience = env_int("TEXAS_LAB_TRAIN_DOMINANCE_CHECKS", 1)
    dominance_drift = env_float("TEXAS_LAB_TRAIN_DOMINANCE_DRIFT", 2.0)
    minimum_best_frequency = clamp(
        env_float("TEXAS_LAB_TRAIN_BEST_FREQUENCY", 0.70),
        0.0,
        1.0,
    )
    convergence = TrainingConvergence(
        actions,
        minimum_iterations,
        tolerance,
        patience,
        dominance_margin if dominance_enabled else None,
        dominance_patience,
        dominance_drift,
        minimum_best_frequency,
    )
    gpu_batches = 0
    gpu_hand_evaluations = 0
    stopped = False

    while solver.training_iteration < requested:
        remaining = requested - solver.training_iteration
        iterations = min(chunk_size, remaining) if adaptive_enabled else remaining
        diagnostics = solver.train_chunk(iterations, use_gpu=use_gpu)
        gpu_batches += diagnostics["batches"]
        gpu_hand_evaluations += diagnostics["handEvaluations"]
        if not adaptive_enabled:
            break

        samples, values = solver.root_action_ev_snapshot()
        root_strategy = (
            {
                action: solver.root_strategy_sum[index]
                / solver.root_strategy_weight
                for index, action in enumerate(actions)
            }
            if solver.root_strategy_weight > EPSILON
            else None
        )
        if values and convergence.inspect(
            solver.training_iteration,
            samples,
            values,
            root_strategy,
        ):
            stopped = solver.training_iteration < requested
            break

    samples, values = solver.root_action_ev_snapshot()
    return {
        "requestedIterations": requested,
        "iterations": solver.training_iteration,
        "adaptiveRequested": adaptive_requested,
        "adaptiveEnabled": adaptive_enabled,
        "stopped": stopped,
        "prunedIterations": requested - solver.training_iteration,
        "minimumIterations": minimum_iterations,
        "chunkSize": chunk_size,
        "tolerance": tolerance,
        "patience": patience,
        "checks": convergence.checks,
        "stableChecks": convergence.stable_checks,
        "decisionStableChecks": convergence.decision_stable_checks,
        "maximumChange": convergence.maximum_change,
        "stopReason": convergence.stop_reason,
        "bestAction": convergence.best_action,
        "bestFrequency": convergence.best_frequency,
        "minimumGap": convergence.minimum_gap,
        "dominanceEnabled": dominance_enabled,
        "dominanceMargin": dominance_margin,
        "dominancePatience": dominance_patience,
        "dominanceDrift": dominance_drift,
        "minimumBestFrequency": minimum_best_frequency,
        "rootEvSamples": samples,
        "rootActionEvs": values,
        "batches": gpu_batches,
        "handEvaluations": gpu_hand_evaluations,
    }


def run_replica(
    info: dict[str, Any],
    training_iterations: int,
    evaluation_iterations: int,
    seed_nonce: int,
    use_gpu_training: bool = False,
    parallel_fixed_evaluation: bool = False,
    decision_evaluation_iterations: int | None = None,
) -> dict[str, Any]:
    root = make_state(info)
    root_actor = info["hero"]["id"]
    root_hand = tuple(info["hero"]["hole"])
    available = [card for card in range(52) if card not in root.board]
    ranges = {
        player.id: build_range(info, next(row for row in info["players"] if row["id"] == player.id), available)
        for player in root.players
    }

    seed_payload = repr(
        (
            root_actor,
            tuple(sorted(root_hand)),
            root.street,
            root.board,
            root.pot,
            root.current_bet,
            root.aggressive_actions,
            root.players,
            root.pending,
            tuple(
                (
                    entry.get("street"),
                    entry.get("playerId"),
                    entry.get("type"),
                    clean_amount(entry.get("amount", 0)),
                )
                for entry in info["history"]
            ),
            seed_nonce,
        )
    ).encode()
    seed = int.from_bytes(hashlib.blake2b(seed_payload, digest_size=8).digest(), "big")
    solver = SubgameCFR(
        root,
        {root_actor: root_hand},
        ranges,
        seed,
        root_info=info,
    )

    training_gpu = False
    if use_gpu_training:
        from .gpu import get_batch_evaluator

        training_gpu = get_batch_evaluator().uses_cuda

    error_budget_fraction = clamp(
        env_float("TEXAS_LAB_ERROR_BUDGET", 0.05),
        0.001,
        0.25,
    )
    root_target = root.players[player_index(root, root_actor)]
    decision_scale = max(1.0, root.pot)
    absolute_error_scale = max(
        1.0,
        root.pot,
        root_target.stack + root_target.total_committed,
    )
    training_run = train_to_ev_convergence(
        solver,
        training_iterations,
        use_gpu=training_gpu,
        decision_scale=decision_scale,
        error_budget_fraction=error_budget_fraction,
    )
    training_gpu_metrics = {
        "batches": training_run["batches"],
        "handEvaluations": training_run["handEvaluations"],
    }

    evaluation_requested_iterations = evaluation_iterations
    decision_evaluation_stopped = bool(
        training_run["stopReason"] == "action-dominated"
        and decision_evaluation_iterations is not None
        and decision_evaluation_iterations < evaluation_iterations
    )
    if decision_evaluation_stopped:
        evaluation_iterations = max(1, int(decision_evaluation_iterations))

    root_node = solver.get_infoset(root, {root_actor: root_hand})
    root_strategy = solver.root_average_strategy(root_node)
    action_outcomes = {action: [] for action in root_node.actions}
    action_leaf_cutoff_hits = {action: 0 for action in root_node.actions}
    equity_shares: list[float] = []
    paired_stats = PairedActionStats(root_node.actions)
    adaptive_requested = env_flag("TEXAS_LAB_ADAPTIVE_EVALUATION", True)
    adaptive_enabled = adaptive_requested and not parallel_fixed_evaluation
    evaluation_chunk = env_int("TEXAS_LAB_EVAL_CHUNK_SIZE", 128)
    minimum_evaluation = min(
        evaluation_iterations,
        env_int("TEXAS_LAB_EVAL_MIN_SAMPLES", 128),
    )
    confidence_z = env_float("TEXAS_LAB_EVAL_SEQUENTIAL_Z", 2.58)
    target_difference_ci = env_float(
        "TEXAS_LAB_EVAL_TARGET_CI",
        error_budget_fraction * decision_scale,
    )
    material_edge = env_float(
        "TEXAS_LAB_EVAL_MATERIAL_EDGE",
        error_budget_fraction * decision_scale * 0.5,
    )
    confidence_alpha = clamp(env_float("TEXAS_LAB_EVAL_ALPHA", 0.05), 1e-6, 0.25)
    maximum_inspections = max(
        1,
        math.ceil(evaluation_requested_iterations / evaluation_chunk),
    )
    reported_intervals = len(root_node.actions) + (
        len(root_node.actions) * (len(root_node.actions) - 1) // 2
    )
    utility_range_width = sum(
        player.stack + player.total_committed for player in root.players
    )
    utility_range_widths = {
        action: 0.0 if action == "fold" else utility_range_width
        for action in root_node.actions
    }
    action_model_error_calibrations = root_action_model_error_calibrations(
        solver,
        root,
        root_node.actions,
    )
    confidence_method = (
        "sequential-empirical-bernstein"
        if adaptive_enabled
        else "pooled-fixed-sample-empirical-bernstein"
        if parallel_fixed_evaluation
        else "fixed-sample-normal"
    )
    alpha_per_interval = confidence_alpha / max(
        1,
        reported_intervals
        * (maximum_inspections if adaptive_enabled else 1),
    )
    adaptive_stopped = False
    gpu_evaluation_batches = 0
    gpu_evaluation_hand_evaluations = 0

    from .gpu import settle_terminal_records, showdown_shares_batch

    solver.begin_evaluation()
    while paired_stats.count < evaluation_iterations:
        chunk_count = min(evaluation_chunk, evaluation_iterations - paired_stats.count)
        action_terminals = {action: [] for action in root_node.actions}
        equity_showdowns: list[tuple[tuple[int, ...], tuple[tuple[int, int], ...]]] = []
        for _ in range(chunk_count):
            hands, runout = solver.sample_deal()
            final_board = (*root.board, *runout)
            showdown_hands = (root_hand, *(
                hands[player.id]
                for player in active_players(root)
                if player.id != root_actor
            ))
            equity_showdowns.append((final_board, showdown_hands))
            for action in root_node.actions:
                evaluation_hits_before = (
                    solver.depth_evaluator.evaluation_cutoff_hits
                    if solver.depth_evaluator is not None
                    else 0
                )
                action_terminals[action].append(
                    solver.rollout_terminal(
                        solver.action_state(root, action),
                        hands,
                        runout,
                        root_actor,
                    )
                )
                if solver.depth_evaluator is not None:
                    action_leaf_cutoff_hits[action] += (
                        solver.depth_evaluator.evaluation_cutoff_hits
                        - evaluation_hits_before
                    )

        flattened_terminals = [
            terminal
            for action in root_node.actions
            for terminal in action_terminals[action]
        ]
        flattened_outcomes, evaluated = settle_terminal_records(flattened_terminals)
        gpu_evaluation_hand_evaluations += evaluated
        gpu_evaluation_batches += int(evaluated > 0)
        chunk_outcomes: dict[str, list[float]] = {}
        for action_index, action in enumerate(root_node.actions):
            start = action_index * chunk_count
            outcomes = flattened_outcomes[start : start + chunk_count]
            chunk_outcomes[action] = outcomes
            action_outcomes[action].extend(outcomes)
        shares, evaluated = showdown_shares_batch(equity_showdowns)
        equity_shares.extend(shares)
        gpu_evaluation_hand_evaluations += evaluated
        gpu_evaluation_batches += int(evaluated > 0)

        for sample_index_in_chunk in range(chunk_count):
            paired_stats.update(
                {
                    action: chunk_outcomes[action][sample_index_in_chunk]
                    for action in root_node.actions
                }
            )
        if adaptive_enabled and paired_stats.can_stop(
            min_samples=minimum_evaluation,
            confidence_z=confidence_z,
            target_difference_ci=target_difference_ci,
            material_edge=material_edge,
            action_model_errors=action_model_error_calibrations,
            target_action_ci=error_budget_fraction * absolute_error_scale,
            utility_range_widths=utility_range_widths,
            confidence_alpha=alpha_per_interval,
        ):
            adaptive_stopped = paired_stats.count < evaluation_iterations
            break

    leaf_diagnostics = (
        solver.depth_evaluator.diagnostics()
        if solver.depth_evaluator is not None
        else {
            "leafCutoffActive": False,
            "leafCutoffMode": "off",
            "leafModelFingerprint": "none",
            "leafCutoffAttempts": 0,
            "leafCutoffHits": 0,
            "leafExactFoldShortcuts": 0,
            "leafEvaluationCutoffHits": 0,
            "leafModelErrorBound": 0.0,
            "leafModelMaxUncertainty": 0.0,
            "leafBeliefUpdates": 0,
            "leafBeliefRefreshes": 0,
            "leafReachEpoch": 0,
            "leafCurrentPolicyRangeQueries": 0,
            "leafBlueprintRangePolicyQueries": 0,
            "leafCutoffGateCacheHits": 0,
            "leafCutoffGateCacheMisses": 0,
            "leafPredictionCacheHits": 0,
            "leafPredictionCacheMisses": 0,
            "leafCutoffRejections": {},
        }
    )
    execution_semantics = solver_execution_semantics(
        uses_cuda=training_gpu,
        leaf_runtime_loaded=solver.depth_evaluator is not None,
    )
    return {
        "actionOrder": root_node.actions,
        "actionOutcomes": action_outcomes,
        "strategyMass": [
            probability * solver.root_strategy_weight for probability in root_strategy
        ],
        "strategyWeight": solver.root_strategy_weight,
        "rootRegretUpdates": root_node.regret_updates,
        "rootRegretWeight": root_node.regret_weight_sum,
        "rootRegret": (
            max(root_node.regrets, default=0.0) / max(1.0, root_node.regret_weight_sum)
        ),
        "equityShares": equity_shares,
        "evaluationIterations": paired_stats.count,
        "evaluationRequestedIterations": evaluation_requested_iterations,
        "decisionEvaluationStopped": decision_evaluation_stopped,
        "evaluationPrunedIterations": (
            evaluation_requested_iterations - evaluation_iterations
        ),
        "adaptiveEvaluationRequested": adaptive_requested,
        "adaptiveEvaluationEnabled": adaptive_enabled,
        "adaptiveEvaluationStopped": adaptive_stopped,
        "parallelFixedEvaluation": parallel_fixed_evaluation,
        "errorBudgetFraction": error_budget_fraction,
        "decisionErrorScale": decision_scale,
        "decisionGapErrorBudget": error_budget_fraction * decision_scale,
        "absoluteErrorScale": absolute_error_scale,
        "absoluteErrorBudget": error_budget_fraction * absolute_error_scale,
        "confidenceMethod": confidence_method,
        "samplingConfidenceAlpha": confidence_alpha,
        "samplingConfidenceZ": 1.96 if confidence_method == "fixed-sample-normal" else None,
        "samplingAlphaPerInterval": alpha_per_interval,
        "samplingReportedIntervals": reported_intervals,
        "samplingMaximumInspections": maximum_inspections if adaptive_enabled else 1,
        "utilityRangeWidth": utility_range_width,
        "utilityRangeWidths": utility_range_widths,
        "actionModelErrorCalibrations": action_model_error_calibrations,
        "actionLeafCutoffHits": action_leaf_cutoff_hits,
        "leafModelErrorCalibration": max(
            action_model_error_calibrations.values(),
            default=0.0,
        ),
        "trainingIterations": training_run["iterations"],
        "trainingRequestedIterations": training_run["requestedIterations"],
        "adaptiveTrainingRequested": training_run["adaptiveRequested"],
        "adaptiveTrainingEnabled": training_run["adaptiveEnabled"],
        "adaptiveTrainingStopped": training_run["stopped"],
        "trainingPrunedIterations": training_run["prunedIterations"],
        "trainingMinimumIterations": training_run["minimumIterations"],
        "trainingChunkSize": training_run["chunkSize"],
        "trainingEvTolerance": training_run["tolerance"],
        "trainingStableChecksRequired": training_run["patience"],
        "trainingEvChecks": training_run["checks"],
        "trainingStableChecks": training_run["stableChecks"],
        "trainingDecisionStableChecks": training_run["decisionStableChecks"],
        "trainingEvMaxChange": training_run["maximumChange"],
        "trainingStopReason": training_run["stopReason"],
        "trainingBestAction": training_run["bestAction"],
        "trainingBestFrequency": training_run["bestFrequency"],
        "trainingMinimumGap": training_run["minimumGap"],
        "trainingDominanceEnabled": training_run["dominanceEnabled"],
        "trainingDominanceMargin": training_run["dominanceMargin"],
        "trainingDominanceChecksRequired": training_run["dominancePatience"],
        "trainingDominanceDrift": training_run["dominanceDrift"],
        "trainingMinimumBestFrequency": training_run["minimumBestFrequency"],
        "trainingRootEvSamples": training_run["rootEvSamples"],
        "trainingRootActionEvs": training_run["rootActionEvs"],
        "regretMode": solver.regret_mode,
        "regretUpdateSemantics": (
            f"minibatch-{solver.regret_mode}" if training_gpu else solver.regret_mode
        ),
        "executionSemantics": execution_semantics,
        "solverConfigFingerprint": solver_config_signature(execution_semantics),
        "evaluationInfosetLookups": solver.evaluation_infoset_lookups,
        "evaluationUntrainedLookups": solver.evaluation_untrained_lookups,
        "evaluationPolicyBackoffLookups": solver.evaluation_policy_backoff_lookups,
        "evaluationBlueprintPolicyLookups": solver.evaluation_blueprint_policy_lookups,
        "evaluationUniformFallbackLookups": solver.evaluation_uniform_fallback_lookups,
        "blueprintPolicySeeds": solver.blueprint_policy_seeds,
        "blueprintPolicyFailures": solver.blueprint_policy_failures,
        "blueprintFingerprint": getattr(solver.blueprint_runtime, "fingerprint", "none"),
        "publicTransitionHits": solver.action_transition_hits,
        "publicTransitionMisses": solver.action_transition_misses,
        "gpuTraining": training_gpu,
        "gpuTrainingBatches": training_gpu_metrics["batches"],
        "gpuTrainingHandEvaluations": training_gpu_metrics["handEvaluations"],
        "gpuEvaluationBatches": gpu_evaluation_batches,
        "gpuEvaluationHandEvaluations": gpu_evaluation_hand_evaluations,
        "leafBeliefRefreshInterval": solver.leaf_belief_refresh_interval,
        "leafReachMaxLagIterations": max(0, solver.leaf_belief_refresh_interval - 1),
        **leaf_diagnostics,
    }


def solve_decision(info: dict[str, Any], iterations: int = 20_000) -> dict[str, Any]:
    """Resolve the complete remaining fixed-size betting tree with MCCFR.

    CUDA requests use a small number of independent MCCFR producer processes;
    each process batches training leaves on the GPU. Their average strategies
    and independent EV samples are pooled. CPU-only hosts retain the original
    all-core fallback.
    EV is always measured in chips from the current decision point; no policy
    penalty or hand-strength score is added to chip utility.
    """
    total_budget = max(500, int(iterations))
    cpu_count = max(1, os.cpu_count() or 1)
    from .gpu import get_batch_evaluator
    from .blueprint import get_blueprint_runtime, runtime_signature
    from .leaf_runtime import leaf_runtime_signature

    evaluator = get_batch_evaluator()
    artifact_runtime_key = (
        runtime_signature(),
        leaf_runtime_signature(),
        solver_config_signature(),
    )
    blueprint_policy_available = get_blueprint_runtime().has_valid_policy
    gpu_training = evaluator.uses_cuda
    gpu_worker_limit = (
        min(GPU_TRAINING_WORKERS, PREFLOP_GPU_TRAINING_WORKERS)
        if info["street"] == "preflop"
        else GPU_TRAINING_WORKERS
    )
    worker_count = (
        min(gpu_worker_limit, cpu_count)
        if gpu_training and total_budget >= 5_000
        else min(CPU_TRAINING_WORKERS, cpu_count) if total_budget >= 5_000 else 1
    )
    decision_evaluation_per_worker = (
        math.ceil(
            env_int("TEXAS_LAB_PREFLOP_DOMINATED_EVAL_TOTAL", 2_000)
            / worker_count
        )
        if info["street"] == "preflop"
        else None
    )

    if worker_count == 1:
        training_fraction = 0.45 if blueprint_policy_available else 0.60
        training_per_worker = max(300, int(total_budget * training_fraction))
        evaluation_per_worker = max(200, total_budget - training_per_worker)
        replicas = [
            run_replica(
                info,
                training_per_worker,
                evaluation_per_worker,
                0,
                gpu_training,
                False,
                decision_evaluation_per_worker,
            )
        ]
    else:
        # A bounded minimum keeps each independent replica useful.  The global
        # 5% engineering error budget permits substantially smaller shards;
        # a validated Blueprint prior lowers the floor again because replicas
        # no longer start from a uniform policy.
        training_fraction = 0.45 if blueprint_policy_available else 0.60
        minimum_training_shard = 1_500 if blueprint_policy_available else 3_000
        training_per_worker = max(
            minimum_training_shard,
            math.ceil(total_budget * training_fraction / worker_count),
        )
        evaluation_per_worker = max(
            500,
            math.ceil(total_budget * 0.20 / worker_count),
        )
        pool = get_process_pool(worker_count, artifact_runtime_key)
        futures = [
            pool.submit(
                run_replica,
                info,
                training_per_worker,
                evaluation_per_worker,
                nonce,
                gpu_training,
                True,
                decision_evaluation_per_worker,
            )
            for nonce in range(worker_count)
        ]
        replicas = [future.result() for future in futures]

    pooled_evaluation_per_replica = min(
        replica["evaluationIterations"] for replica in replicas
    )

    action_order = replicas[0]["actionOrder"]
    regret_mode = replicas[0]["regretMode"]
    regret_update_semantics = replicas[0]["regretUpdateSemantics"]
    blueprint_fingerprint = replicas[0]["blueprintFingerprint"]
    (
        expected_blueprint_fingerprint,
        expected_leaf_fingerprint,
        expected_solver_config_fingerprint,
    ) = artifact_runtime_key
    error_budget_fraction = replicas[0]["errorBudgetFraction"]
    decision_error_scale = replicas[0]["decisionErrorScale"]
    decision_gap_error_budget = replicas[0]["decisionGapErrorBudget"]
    absolute_error_scale = replicas[0]["absoluteErrorScale"]
    absolute_error_budget = replicas[0]["absoluteErrorBudget"]
    adaptive_evaluation_requested = replicas[0]["adaptiveEvaluationRequested"]
    adaptive_evaluation_enabled = replicas[0]["adaptiveEvaluationEnabled"]
    parallel_fixed_evaluation = replicas[0]["parallelFixedEvaluation"]
    adaptive_training_requested = replicas[0]["adaptiveTrainingRequested"]
    adaptive_training_enabled = replicas[0]["adaptiveTrainingEnabled"]
    training_minimum_iterations = replicas[0]["trainingMinimumIterations"]
    training_chunk_size = replicas[0]["trainingChunkSize"]
    training_ev_tolerance = replicas[0]["trainingEvTolerance"]
    training_stable_checks_required = replicas[0]["trainingStableChecksRequired"]
    training_dominance_enabled = replicas[0]["trainingDominanceEnabled"]
    training_dominance_margin = replicas[0]["trainingDominanceMargin"]
    training_dominance_checks_required = replicas[0][
        "trainingDominanceChecksRequired"
    ]
    training_dominance_drift = replicas[0]["trainingDominanceDrift"]
    training_minimum_best_frequency = replicas[0][
        "trainingMinimumBestFrequency"
    ]
    confidence_method = replicas[0]["confidenceMethod"]
    sampling_confidence_alpha = replicas[0]["samplingConfidenceAlpha"]
    sampling_confidence_z = replicas[0]["samplingConfidenceZ"]
    sampling_alpha_per_interval = replicas[0]["samplingAlphaPerInterval"]
    sampling_reported_intervals = replicas[0]["samplingReportedIntervals"]
    sampling_maximum_inspections = replicas[0]["samplingMaximumInspections"]
    utility_range_width = replicas[0]["utilityRangeWidth"]
    utility_range_widths = dict(replicas[0]["utilityRangeWidths"])
    action_model_error_calibrations = dict(
        replicas[0]["actionModelErrorCalibrations"]
    )
    execution_semantics = replicas[0]["executionSemantics"]
    action_outcomes = {action: [] for action in action_order}
    action_leaf_cutoff_hits = {action: 0 for action in action_order}
    equity_shares: list[float] = []
    strategy_mass = [0.0] * len(action_order)
    strategy_weight = 0
    root_regret_updates_total = 0
    root_regret_weight_total = 0.0
    evaluation_iterations = 0
    training_iterations = 0
    training_requested_iterations = 0
    training_pruned_iterations = 0
    training_ev_checks = 0
    training_decision_stable_checks = 0
    training_root_ev_samples = 0
    adaptive_training_workers = 0
    adaptive_training_dominance_workers = 0
    training_stop_reasons: Counter[str] = Counter()
    training_ev_max_changes: list[float] = []
    training_best_frequencies: list[float] = []
    training_minimum_gaps: list[float] = []
    weighted_root_regret = 0.0
    evaluation_infoset_lookups = 0
    evaluation_untrained_lookups = 0
    evaluation_policy_backoff_lookups = 0
    evaluation_blueprint_policy_lookups = 0
    evaluation_uniform_fallback_lookups = 0
    blueprint_policy_seeds = 0
    blueprint_policy_failures = 0
    public_transition_hits = 0
    public_transition_misses = 0
    gpu_training_batches = 0
    gpu_training_hand_evaluations = 0
    gpu_evaluation_batches = 0
    gpu_evaluation_hand_evaluations = 0
    gpu_training_workers = 0
    adaptive_evaluation_workers = 0
    decision_evaluation_workers = 0
    evaluation_pruned_iterations = 0
    evaluation_requested_iterations = 0
    leaf_cutoff_workers = 0
    leaf_model_fingerprints: set[str] = set()
    leaf_cutoff_attempts = 0
    leaf_cutoff_hits = 0
    leaf_exact_fold_shortcuts = 0
    leaf_evaluation_cutoff_hits = 0
    leaf_model_error_bound = 0.0
    leaf_model_error_calibration = 0.0
    leaf_model_max_uncertainty = 0.0
    leaf_belief_updates = 0
    leaf_belief_refreshes = 0
    leaf_reach_epochs = 0
    leaf_current_policy_range_queries = 0
    leaf_blueprint_range_policy_queries = 0
    leaf_cutoff_gate_cache_hits = 0
    leaf_cutoff_gate_cache_misses = 0
    leaf_prediction_cache_hits = 0
    leaf_prediction_cache_misses = 0
    leaf_belief_refresh_interval = replicas[0]["leafBeliefRefreshInterval"]
    leaf_reach_max_lag_iterations = replicas[0]["leafReachMaxLagIterations"]
    leaf_cutoff_rejections: Counter[str] = Counter()

    for replica in replicas:
        if replica["actionOrder"] != action_order:
            raise RuntimeError("并行求解副本返回了不一致的根动作集合")
        if replica["regretMode"] != regret_mode:
            raise RuntimeError("并行求解副本返回了不一致的遗憾更新模式")
        if replica["regretUpdateSemantics"] != regret_update_semantics:
            raise RuntimeError("并行求解副本返回了不一致的遗憾批处理语义")
        if replica["executionSemantics"] != execution_semantics:
            raise RuntimeError("并行求解副本返回了不一致的执行语义")
        if (
            replica["errorBudgetFraction"] != error_budget_fraction
            or replica["decisionErrorScale"] != decision_error_scale
            or replica["decisionGapErrorBudget"] != decision_gap_error_budget
            or replica["absoluteErrorScale"] != absolute_error_scale
            or replica["absoluteErrorBudget"] != absolute_error_budget
            or replica["adaptiveEvaluationRequested"] != adaptive_evaluation_requested
            or replica["adaptiveEvaluationEnabled"] != adaptive_evaluation_enabled
            or replica["parallelFixedEvaluation"] != parallel_fixed_evaluation
            or replica["adaptiveTrainingRequested"] != adaptive_training_requested
            or replica["adaptiveTrainingEnabled"] != adaptive_training_enabled
            or replica["trainingMinimumIterations"] != training_minimum_iterations
            or replica["trainingChunkSize"] != training_chunk_size
            or replica["trainingEvTolerance"] != training_ev_tolerance
            or replica["trainingStableChecksRequired"]
            != training_stable_checks_required
            or replica["trainingDominanceEnabled"] != training_dominance_enabled
            or replica["trainingDominanceMargin"] != training_dominance_margin
            or replica["trainingDominanceChecksRequired"]
            != training_dominance_checks_required
            or replica["trainingDominanceDrift"] != training_dominance_drift
            or replica["trainingMinimumBestFrequency"]
            != training_minimum_best_frequency
            or replica["confidenceMethod"] != confidence_method
            or replica["samplingConfidenceAlpha"] != sampling_confidence_alpha
            or replica["samplingConfidenceZ"] != sampling_confidence_z
            or replica["samplingAlphaPerInterval"] != sampling_alpha_per_interval
            or replica["samplingReportedIntervals"] != sampling_reported_intervals
            or replica["samplingMaximumInspections"] != sampling_maximum_inspections
            or replica["utilityRangeWidth"] != utility_range_width
            or replica["utilityRangeWidths"] != utility_range_widths
            or replica["actionModelErrorCalibrations"]
            != action_model_error_calibrations
        ):
            raise RuntimeError("并行求解副本使用了不一致的误差预算")
        if set(replica["actionOutcomes"]) != set(action_order):
            raise RuntimeError("并行求解副本返回了不一致的动作样本")
        if set(replica["actionLeafCutoffHits"]) != set(action_order):
            raise RuntimeError("并行求解副本返回了不一致的逐动作叶模型诊断")
        if worker_count > 1 and (
            replica["adaptiveEvaluationEnabled"]
            or not replica["parallelFixedEvaluation"]
            or replica["adaptiveEvaluationStopped"]
            or replica["evaluationIterations"] > evaluation_per_worker
        ):
            raise RuntimeError("并行求解副本未使用有界固定样本评估")
        if replica["blueprintFingerprint"] != blueprint_fingerprint:
            raise RuntimeError("并行求解副本加载了不一致的 Blueprint")
        if replica["blueprintFingerprint"] != expected_blueprint_fingerprint:
            raise RuntimeError("求解进程加载的 Blueprint 与请求缓存键不一致")
        if replica["leafModelFingerprint"] != expected_leaf_fingerprint:
            raise RuntimeError("求解进程加载的叶模型与请求缓存键不一致")
        if replica["solverConfigFingerprint"] != expected_solver_config_fingerprint:
            raise RuntimeError("求解进程加载的配置与请求缓存键不一致")
        if (
            replica["leafBeliefRefreshInterval"] != leaf_belief_refresh_interval
            or replica["leafReachMaxLagIterations"] != leaf_reach_max_lag_iterations
        ):
            raise RuntimeError("并行求解副本使用了不一致的叶范围刷新配置")
        for action in action_order:
            if len(replica["actionOutcomes"][action]) != replica["evaluationIterations"]:
                raise RuntimeError("并行求解副本的逐动作样本数不一致")
            action_outcomes[action].extend(
                replica["actionOutcomes"][action][
                    :pooled_evaluation_per_replica
                ]
            )
            action_leaf_cutoff_hits[action] += replica["actionLeafCutoffHits"][action]
        for index, mass in enumerate(replica["strategyMass"]):
            strategy_mass[index] += mass
        weight = replica["strategyWeight"]
        strategy_weight += weight
        root_regret_updates = replica["rootRegretUpdates"]
        root_regret_weight = replica["rootRegretWeight"]
        root_regret_updates_total += root_regret_updates
        root_regret_weight_total += root_regret_weight
        weighted_root_regret += replica["rootRegret"] * root_regret_weight
        equity_shares.extend(
            replica["equityShares"][:pooled_evaluation_per_replica]
        )
        evaluation_iterations += pooled_evaluation_per_replica
        evaluation_requested_iterations += replica["evaluationRequestedIterations"]
        adaptive_evaluation_workers += int(replica["adaptiveEvaluationStopped"])
        decision_evaluation_workers += int(replica["decisionEvaluationStopped"])
        evaluation_pruned_iterations += replica["evaluationPrunedIterations"]
        training_iterations += replica["trainingIterations"]
        training_requested_iterations += replica["trainingRequestedIterations"]
        training_pruned_iterations += replica["trainingPrunedIterations"]
        training_ev_checks += replica["trainingEvChecks"]
        training_decision_stable_checks += replica[
            "trainingDecisionStableChecks"
        ]
        training_root_ev_samples += replica["trainingRootEvSamples"]
        adaptive_training_workers += int(replica["adaptiveTrainingStopped"])
        adaptive_training_dominance_workers += int(
            replica["trainingStopReason"] == "action-dominated"
        )
        if replica["trainingStopReason"] is not None:
            training_stop_reasons.update((replica["trainingStopReason"],))
        if replica["trainingEvMaxChange"] is not None:
            training_ev_max_changes.append(replica["trainingEvMaxChange"])
        if replica["trainingBestFrequency"] is not None:
            training_best_frequencies.append(replica["trainingBestFrequency"])
        if replica["trainingMinimumGap"] is not None:
            training_minimum_gaps.append(replica["trainingMinimumGap"])
        evaluation_infoset_lookups += replica["evaluationInfosetLookups"]
        evaluation_untrained_lookups += replica["evaluationUntrainedLookups"]
        evaluation_policy_backoff_lookups += replica["evaluationPolicyBackoffLookups"]
        evaluation_blueprint_policy_lookups += replica["evaluationBlueprintPolicyLookups"]
        evaluation_uniform_fallback_lookups += replica["evaluationUniformFallbackLookups"]
        blueprint_policy_seeds += replica["blueprintPolicySeeds"]
        blueprint_policy_failures += replica["blueprintPolicyFailures"]
        public_transition_hits += replica["publicTransitionHits"]
        public_transition_misses += replica["publicTransitionMisses"]
        gpu_training_batches += replica["gpuTrainingBatches"]
        gpu_training_hand_evaluations += replica["gpuTrainingHandEvaluations"]
        gpu_evaluation_batches += replica["gpuEvaluationBatches"]
        gpu_evaluation_hand_evaluations += replica["gpuEvaluationHandEvaluations"]
        gpu_training_workers += int(replica["gpuTraining"])
        leaf_cutoff_workers += int(replica["leafCutoffActive"])
        if replica["leafModelFingerprint"] != "none":
            leaf_model_fingerprints.add(replica["leafModelFingerprint"])
        leaf_cutoff_attempts += replica["leafCutoffAttempts"]
        leaf_cutoff_hits += replica["leafCutoffHits"]
        leaf_exact_fold_shortcuts += replica["leafExactFoldShortcuts"]
        leaf_evaluation_cutoff_hits += replica["leafEvaluationCutoffHits"]
        leaf_model_error_bound = max(
            leaf_model_error_bound,
            replica["leafModelErrorBound"],
        )
        leaf_model_error_calibration = max(
            leaf_model_error_calibration,
            replica["leafModelErrorCalibration"],
        )
        leaf_model_max_uncertainty = max(
            leaf_model_max_uncertainty,
            replica["leafModelMaxUncertainty"],
        )
        leaf_belief_updates += replica["leafBeliefUpdates"]
        leaf_belief_refreshes += replica["leafBeliefRefreshes"]
        leaf_reach_epochs += replica["leafReachEpoch"]
        leaf_current_policy_range_queries += replica["leafCurrentPolicyRangeQueries"]
        leaf_blueprint_range_policy_queries += replica["leafBlueprintRangePolicyQueries"]
        leaf_cutoff_gate_cache_hits += replica["leafCutoffGateCacheHits"]
        leaf_cutoff_gate_cache_misses += replica["leafCutoffGateCacheMisses"]
        leaf_prediction_cache_hits += replica["leafPredictionCacheHits"]
        leaf_prediction_cache_misses += replica["leafPredictionCacheMisses"]
        leaf_cutoff_rejections.update(replica["leafCutoffRejections"])

    gpu_hand_evaluations = (
        gpu_training_hand_evaluations + gpu_evaluation_hand_evaluations
    )
    gpu_batches = gpu_training_batches + gpu_evaluation_batches

    root_strategy = [mass / max(1, strategy_weight) for mass in strategy_mass]
    strategy_total = sum(root_strategy)
    root_strategy = [probability / strategy_total for probability in root_strategy]
    action_statistics = {
        action: sample_statistics(action_outcomes[action]) for action in action_order
    }
    best_action = max(action_order, key=lambda action: action_statistics[action][0])
    best_mean = action_statistics[best_action][0]
    actions: list[dict[str, Any]] = []
    for index, action in enumerate(action_order):
        mean, stderr = action_statistics[action]
        differences = [
            best - candidate
            for best, candidate in zip(
                action_outcomes[best_action],
                action_outcomes[action],
            )
        ]
        difference_mean, difference_stderr = sample_statistics(differences)
        if confidence_method in {
            "sequential-empirical-bernstein",
            "pooled-fixed-sample-empirical-bernstein",
        }:
            sampling_ci95 = empirical_bernstein_radius(
                len(action_outcomes[action]),
                stderr * stderr * len(action_outcomes[action]),
                utility_range_widths[action],
                sampling_alpha_per_interval,
            )
            difference_sampling_ci95 = (
                empirical_bernstein_radius(
                    len(differences),
                    difference_stderr * difference_stderr * len(differences),
                    utility_range_widths[best_action]
                    + utility_range_widths[action],
                    sampling_alpha_per_interval,
                )
                if action != best_action
                else 0.0
            )
        else:
            sampling_ci95 = stderr * 1.96
            difference_sampling_ci95 = (
                difference_stderr * 1.96 if action != best_action else 0.0
            )
        action_model_error_calibration = action_model_error_calibrations[action]
        difference_model_error_calibration = (
            action_model_error_calibrations[best_action]
            + action_model_error_calibrations[action]
            if action != best_action
            else 0.0
        )
        actions.append(
            {
                "action": action,
                "ev": round(mean, 3),
                "stderr": stderr,
                "samplingCi95": sampling_ci95,
                "modelErrorCalibration": action_model_error_calibration,
                "ci95": sampling_ci95 + action_model_error_calibration,
                "evLossEstimate": max(0.0, difference_mean),
                "differenceStderr": difference_stderr,
                "differenceSamplingCi95": difference_sampling_ci95,
                "differenceModelErrorCalibration": (
                    difference_model_error_calibration
                ),
                "differenceCi95": (
                    difference_sampling_ci95
                    + difference_model_error_calibration
                    if action != best_action
                    else 0.0
                ),
                "leafModelPossible": action_model_error_calibration > 0,
                "leafModelCutoffHits": action_leaf_cutoff_hits[action],
                "frequency": root_strategy[index],
            }
        )

    best_ev = round(best_mean, 3)
    best_row = next(row for row in actions if row["action"] == best_action)
    statistically_tied = []
    for row in actions:
        if row["evLossEstimate"] <= max(0.02, row["differenceCi95"]):
            statistically_tied.append(row["action"])

    root_regret = weighted_root_regret / max(1.0, root_regret_weight_total)
    evaluation_untrained_rate = (
        evaluation_untrained_lookups / max(1, evaluation_infoset_lookups)
    )
    evaluation_policy_backoff_rate = (
        evaluation_policy_backoff_lookups / max(1, evaluation_infoset_lookups)
    )
    evaluation_blueprint_policy_rate = (
        evaluation_blueprint_policy_lookups / max(1, evaluation_infoset_lookups)
    )
    evaluation_uncovered_lookups = max(
        0,
        evaluation_untrained_lookups - evaluation_blueprint_policy_lookups,
    )
    evaluation_uncovered_rate = (
        evaluation_uncovered_lookups / max(1, evaluation_infoset_lookups)
    )
    evaluation_uniform_fallback_rate = (
        evaluation_uniform_fallback_lookups / max(1, evaluation_infoset_lookups)
    )
    regret_tolerance = decision_gap_error_budget
    strategy_reliable = (
        root_regret <= regret_tolerance
        and evaluation_uncovered_rate <= error_budget_fraction
    )
    recommended = (
        [row["action"] for row in actions if row["frequency"] >= 0.01 and row["action"] in statistically_tied]
        if strategy_reliable
        else [best_row["action"]]
    )
    if not recommended:
        recommended = [best_row["action"]]
    equity = sum(equity_shares) / evaluation_iterations
    compute_runtime = evaluator.runtime()
    return {
        "equity": equity,
        "actions": actions,
        "bestEV": best_ev,
        "recommended": recommended,
        "statisticallyTied": statistically_tied,
        "strategyReliable": strategy_reliable,
        "regretTolerance": regret_tolerance,
        "uncoveredRateTolerance": error_budget_fraction,
        "iterations": total_budget,
        "trainingIterations": training_iterations,
        "trainingRequestedIterations": training_requested_iterations,
        "trainingPrunedIterations": training_pruned_iterations,
        "adaptiveTraining": adaptive_training_enabled,
        "adaptiveTrainingRequested": adaptive_training_requested,
        "adaptiveTrainingWorkersStopped": adaptive_training_workers,
        "adaptiveTrainingDominanceWorkersStopped": (
            adaptive_training_dominance_workers
        ),
        "trainingStopReasons": dict(training_stop_reasons),
        "trainingMinimumIterations": training_minimum_iterations,
        "trainingChunkSize": training_chunk_size,
        "trainingEvTolerance": training_ev_tolerance,
        "trainingStableChecksRequired": training_stable_checks_required,
        "trainingEvChecks": training_ev_checks,
        "trainingDecisionStableChecks": training_decision_stable_checks,
        "trainingRootEvSamples": training_root_ev_samples,
        "trainingEvMaxChange": (
            max(training_ev_max_changes) if training_ev_max_changes else None
        ),
        "trainingBestFrequency": (
            min(training_best_frequencies) if training_best_frequencies else None
        ),
        "trainingMinimumGap": (
            min(training_minimum_gaps) if training_minimum_gaps else None
        ),
        "trainingDominanceEnabled": training_dominance_enabled,
        "trainingDominanceMargin": training_dominance_margin,
        "trainingDominanceChecksRequired": training_dominance_checks_required,
        "trainingDominanceDrift": training_dominance_drift,
        "trainingMinimumBestFrequency": training_minimum_best_frequency,
        "regretMode": regret_mode,
        "regretUpdateSemantics": regret_update_semantics,
        "evaluationIterations": evaluation_iterations,
        "evaluationRequestedIterations": evaluation_requested_iterations,
        "evaluationPerWorkerActual": pooled_evaluation_per_replica,
        "decisionEvaluationWorkersStopped": decision_evaluation_workers,
        "evaluationPrunedIterations": evaluation_pruned_iterations,
        "adaptiveEvaluation": adaptive_evaluation_enabled,
        "adaptiveEvaluationRequested": adaptive_evaluation_requested,
        "adaptiveEvaluationDisabledForParallel": (
            adaptive_evaluation_requested and parallel_fixed_evaluation
        ),
        "adaptiveEvaluationWorkersStopped": adaptive_evaluation_workers,
        "errorBudgetFraction": error_budget_fraction,
        "decisionErrorScale": decision_error_scale,
        "decisionGapErrorBudget": decision_gap_error_budget,
        "absoluteErrorScale": absolute_error_scale,
        "absoluteErrorBudget": absolute_error_budget,
        "confidenceMethod": confidence_method,
        "samplingConfidenceAlpha": sampling_confidence_alpha,
        "samplingConfidenceZ": sampling_confidence_z,
        "samplingAlphaPerInterval": sampling_alpha_per_interval,
        "samplingReportedIntervals": sampling_reported_intervals,
        "samplingMaximumInspections": sampling_maximum_inspections,
        "utilityRangeWidth": utility_range_width,
        "utilityRangeWidths": utility_range_widths,
        "actionModelErrorCalibrations": action_model_error_calibrations,
        "actionLeafCutoffHits": action_leaf_cutoff_hits,
        "executionSemantics": execution_semantics,
        "solverConfigFingerprint": expected_solver_config_fingerprint,
        "evaluationInfosetLookups": evaluation_infoset_lookups,
        "evaluationUntrainedLookups": evaluation_untrained_lookups,
        "evaluationUntrainedRate": evaluation_untrained_rate,
        "evaluationUncoveredLookups": evaluation_uncovered_lookups,
        "evaluationUncoveredRate": evaluation_uncovered_rate,
        "evaluationPolicyBackoffLookups": evaluation_policy_backoff_lookups,
        "evaluationPolicyBackoffRate": evaluation_policy_backoff_rate,
        "evaluationBlueprintPolicyLookups": evaluation_blueprint_policy_lookups,
        "evaluationBlueprintPolicyRate": evaluation_blueprint_policy_rate,
        "evaluationUniformFallbackLookups": evaluation_uniform_fallback_lookups,
        "evaluationUniformFallbackRate": evaluation_uniform_fallback_rate,
        "blueprintPolicySeeds": blueprint_policy_seeds,
        "blueprintPolicyFailures": blueprint_policy_failures,
        "blueprintFingerprint": blueprint_fingerprint,
        "leafCutoffWorkers": leaf_cutoff_workers,
        "leafModelFingerprint": (
            next(iter(leaf_model_fingerprints))
            if len(leaf_model_fingerprints) == 1
            else "none" if not leaf_model_fingerprints else "mixed"
        ),
        "leafCutoffAttempts": leaf_cutoff_attempts,
        "leafCutoffHits": leaf_cutoff_hits,
        "leafExactFoldShortcuts": leaf_exact_fold_shortcuts,
        "leafEvaluationCutoffHits": leaf_evaluation_cutoff_hits,
        "leafModelErrorBound": leaf_model_error_bound,
        "leafObservedModelErrorRadius": leaf_model_error_bound,
        "leafModelErrorCalibration": leaf_model_error_calibration,
        "leafModelErrorCalibrationIsHardBound": False,
        "leafModelMaxUncertainty": leaf_model_max_uncertainty,
        "leafBeliefUpdates": leaf_belief_updates,
        "leafBeliefRefreshes": leaf_belief_refreshes,
        "leafReachEpochs": leaf_reach_epochs,
        "leafBeliefRefreshInterval": leaf_belief_refresh_interval,
        "leafReachMaxLagIterations": leaf_reach_max_lag_iterations,
        "leafCurrentPolicyRangeQueries": leaf_current_policy_range_queries,
        "leafBlueprintRangePolicyQueries": leaf_blueprint_range_policy_queries,
        "leafCutoffGateCacheHits": leaf_cutoff_gate_cache_hits,
        "leafCutoffGateCacheMisses": leaf_cutoff_gate_cache_misses,
        "leafPredictionCacheHits": leaf_prediction_cache_hits,
        "leafPredictionCacheMisses": leaf_prediction_cache_misses,
        "leafCutoffRejections": dict(leaf_cutoff_rejections),
        "publicTransitionHits": public_transition_hits,
        "publicTransitionMisses": public_transition_misses,
        "aggregateIterations": training_iterations + evaluation_iterations,
        "workerCount": worker_count,
        "trainingPerWorker": training_per_worker,
        "evaluationPerWorker": evaluation_per_worker,
        "rootRegret": root_regret,
        "rootRegretUpdates": root_regret_updates_total,
        "rootRegretWeight": root_regret_weight_total,
        "computeBackend": compute_runtime.backend,
        "computeDevice": compute_runtime.device,
        "cudaAvailable": compute_runtime.cuda_available,
        "torchVersion": compute_runtime.torch_version,
        "cudaVersion": compute_runtime.cuda_version,
        "gpuBatches": gpu_batches if compute_runtime.cuda_available else 0,
        "gpuHandEvaluations": gpu_hand_evaluations if compute_runtime.cuda_available else 0,
        "gpuTrainingWorkers": gpu_training_workers,
        "gpuTrainingBatches": gpu_training_batches if compute_runtime.cuda_available else 0,
        "gpuTrainingHandEvaluations": (
            gpu_training_hand_evaluations if compute_runtime.cuda_available else 0
        ),
        "gpuEvaluationBatches": gpu_evaluation_batches if compute_runtime.cuda_available else 0,
        "gpuEvaluationHandEvaluations": (
            gpu_evaluation_hand_evaluations if compute_runtime.cuda_available else 0
        ),
        "model": (
            "离线 Blueprint 热启动 + 信念条件叶价值的深度限制 MCCFR/"
            + ("Linear CFR" if regret_mode == "linear" else "RM+")
            if leaf_evaluation_cutoff_hits
            else "固定下注尺度完整后续树的 external-sampling MCCFR/"
            + ("Linear CFR" if regret_mode == "linear" else "RM+")
            + " 近似策略"
        ),
        "method": (
            "紧凑公共节点 · 手牌感知分层策略回退 · 共享公共牌采样 · GPU 批量叶节点及配对 EV 评估"
            if compute_runtime.cuda_available
            else "紧凑公共节点 · 手牌感知分层策略回退 · 共享公共牌采样 · CPU 精确终局回退"
        ),
        "explanation": build_explanation(
            info,
            actions,
            equity,
            root_regret,
            evaluation_untrained_rate,
            evaluation_policy_backoff_rate,
            evaluation_uniform_fallback_rate,
            leaf_evaluation_cutoff_hits,
            leaf_model_error_calibration,
        ),
    }
