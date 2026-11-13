from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.solver import (  # noqa: E402
    PublicState,
    active_players,
    apply_action,
    legal_actions,
    make_state,
    player_index,
    raise_target,
)


POSITIONS = ("UTG", "HJ", "CO", "BTN", "SB", "BB")


def representative_hands() -> list[tuple[str, tuple[int, int]]]:
    """Return one suit-canonical concrete hand for every 169 preflop class."""
    result: list[tuple[str, tuple[int, int]]] = []
    labels = "23456789TJQKA"
    for high in range(14, 1, -1):
        for low in range(high, 1, -1):
            high_card = (high - 2) * 4
            low_card = (low - 2) * 4
            if high == low:
                result.append((labels[high - 2] * 2, (high_card, low_card + 1)))
            else:
                result.append(
                    (f"{labels[high - 2]}{labels[low - 2]}s", (high_card, low_card))
                )
                result.append(
                    (f"{labels[high - 2]}{labels[low - 2]}o", (high_card, low_card + 1))
                )
    if len(result) != 169:
        raise RuntimeError(f"expected 169 hand classes, got {len(result)}")
    return result


def initial_info() -> dict[str, Any]:
    players = []
    for index, position in enumerate(POSITIONS):
        committed = 0.5 if position == "SB" else 1.0 if position == "BB" else 0.0
        players.append(
            {
                "id": f"p{index}",
                "position": position,
                "stack": 100.0 - committed,
                "streetCommitted": committed,
                "totalCommitted": committed,
                "folded": False,
                "allIn": False,
            }
        )
    return {
        "street": "preflop",
        "board": [],
        "pot": 1.5,
        "currentBet": 1.0,
        "aggressiveActions": 0,
        "hero": {
            "id": "p0",
            "position": "UTG",
            "stack": 100.0,
            "streetCommitted": 0.0,
            "hole": [0, 1],
        },
        "players": players,
        "pending": [f"p{index}" for index in range(6)],
        "history": [],
        "legalActions": ["fold", "call", "raise"],
        "raiseTo": 2.5,
    }


def action_probabilities(state: PublicState) -> dict[str, float]:
    """A deterministic traffic prior used only to rank exact coverage nodes."""
    actions = legal_actions(state)
    if "check" in actions:
        weights = {"check": 0.82, "raise": 0.18}
    elif state.aggressive_actions == 0:
        weights = {"fold": 0.48, "call": 0.23, "raise": 0.29}
    elif state.aggressive_actions == 1:
        weights = {"fold": 0.66, "call": 0.25, "raise": 0.09}
    elif state.aggressive_actions == 2:
        weights = {"fold": 0.72, "call": 0.23, "raise": 0.05}
    else:
        weights = {"fold": 0.78, "call": 0.22}
    selected = {action: weights[action] for action in actions}
    total = sum(selected.values())
    return {action: value / total for action, value in selected.items()}


def public_action(parent: PublicState, child: PublicState, action: str) -> dict[str, Any]:
    actor_id = parent.pending[0]
    before = parent.players[player_index(parent, actor_id)]
    after = child.players[player_index(child, actor_id)]
    paid = round(after.total_committed - before.total_committed, 2)
    amount = after.street_committed if action == "raise" else paid
    return {
        "street": parent.street,
        "board": list(parent.board),
        "playerId": actor_id,
        "position": before.position,
        "type": action,
        "amount": amount,
        "potAfter": child.pot,
    }


def state_key(state: PublicState) -> tuple[Any, ...]:
    return (
        state.node_id,
        state.street,
        state.board,
        state.pot,
        state.current_bet,
        state.aggressive_actions,
        state.players,
        state.pending,
    )


@dataclass(frozen=True)
class RankedState:
    probability: float
    order: int
    state: PublicState
    history: tuple[dict[str, Any], ...]


def common_public_states(limit: int) -> list[RankedState]:
    root = make_state(initial_info())
    heap: list[tuple[float, int, PublicState, tuple[dict[str, Any], ...]]] = [
        (-1.0, 0, root, ()),
    ]
    queued = {state_key(root)}
    result: list[RankedState] = []
    order = 0
    while heap and len(result) < limit:
        negative_probability, _, state, history = heapq.heappop(heap)
        probability = -negative_probability
        if not state.pending or len(active_players(state)) <= 1:
            continue
        result.append(RankedState(probability, len(result), state, history))
        for action, action_probability in action_probabilities(state).items():
            child = apply_action(state, action)
            if child.street != "preflop" or not child.pending or len(active_players(child)) <= 1:
                continue
            key = state_key(child)
            if key in queued:
                continue
            queued.add(key)
            order += 1
            child_history = (*history, public_action(state, child, action))
            heapq.heappush(
                heap,
                (-probability * action_probability, order, child, child_history),
            )
    return result


def state_info(
    ranked: RankedState,
    hand_label: str,
    hole: tuple[int, int],
) -> dict[str, Any]:
    state = ranked.state
    actor_id = state.pending[0]
    actor = state.players[player_index(state, actor_id)]
    return {
        "profile": "common-preflop-100bb-v1",
        "validationGroup": f"common-preflop-100bb-v1:{ranked.order + 1}",
        "coverageRank": ranked.order + 1,
        "pathProbability": ranked.probability,
        "handClass": hand_label,
        "info": {
            "street": state.street,
            "board": list(state.board),
            "pot": state.pot,
            "currentBet": state.current_bet,
            "aggressiveActions": state.aggressive_actions,
            "hero": {
                "id": actor.id,
                "position": actor.position,
                "stack": actor.stack,
                "streetCommitted": actor.street_committed,
                "hole": list(hole),
            },
            "players": [
                {
                    "id": player.id,
                    "position": player.position,
                    "stack": player.stack,
                    "streetCommitted": player.street_committed,
                    "totalCommitted": player.total_committed,
                    "folded": player.folded,
                    "allIn": player.all_in,
                }
                for player in state.players
            ],
            "pending": list(state.pending),
            "history": list(ranked.history),
            "legalActions": list(legal_actions(state)),
            "raiseTo": raise_target(state, actor),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate suit-canonical, traffic-ranked common preflop solve nodes.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-states", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.public_states < 1:
        raise SystemExit("--public-states must be positive")
    states = common_public_states(args.public_states)
    hands = representative_hands()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for ranked in states:
            for label, hole in hands:
                handle.write(
                    json.dumps(
                        state_info(ranked, label, hole),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
    summary = {
        "output": str(output),
        "profile": "common-preflop-100bb-v1",
        "publicStates": len(states),
        "handClasses": len(hands),
        "solveNodes": len(states) * len(hands),
        "minimumPathProbability": states[-1].probability,
    }
    all_states = common_public_states(100_000)
    expected_visits = sum(state.probability for state in all_states)
    summary["estimatedDecisionVisitCoverage"] = (
        sum(state.probability for state in states) / expected_visits
        if expected_visits
        else 0.0
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
