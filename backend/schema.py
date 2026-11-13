from __future__ import annotations

from copy import deepcopy
from typing import Any


PRIVATE_HISTORY_FIELDS = frozenset({"deck", "heroCards", "hole", "holes"})
PUBLIC_HISTORY_FIELDS = (
    "street",
    "board",
    "playerId",
    "position",
    "actor",
    "type",
    "amount",
    "potAfter",
)


def history_contains_private_cards(value: Any) -> bool:
    if isinstance(value, dict):
        if PRIVATE_HISTORY_FIELDS.intersection(value):
            return True
        return any(history_contains_private_cards(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(history_contains_private_cards(item) for item in value)
    return False


def public_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project replay-enriched history onto fields visible to every player."""
    return [
        {
            field: deepcopy(entry[field])
            for field in PUBLIC_HISTORY_FIELDS
            if field in entry
        }
        for entry in history
    ]


def normalize_solve_info(info: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical solver request without UI or replay-only fields."""
    hero = info["hero"]
    return {
        "street": info["street"],
        "board": list(info["board"]),
        "pot": info["pot"],
        "currentBet": info["currentBet"],
        "aggressiveActions": info.get("aggressiveActions", 0),
        "hero": {
            "id": hero["id"],
            "position": hero["position"],
            "stack": hero["stack"],
            "streetCommitted": hero["streetCommitted"],
            "hole": list(hero["hole"]),
        },
        "players": [
            {
                "id": player["id"],
                "position": player["position"],
                "stack": player["stack"],
                "streetCommitted": player["streetCommitted"],
                "totalCommitted": player["totalCommitted"],
                "folded": player["folded"],
                "allIn": player["allIn"],
            }
            for player in info["players"]
        ],
        "pending": list(info["pending"]),
        "history": public_history(info["history"]),
        "legalActions": list(info["legalActions"]),
        "raiseTo": info["raiseTo"],
    }
