from __future__ import annotations

import hashlib
from itertools import permutations
import json
from typing import Any

from .schema import normalize_solve_info


ABSTRACTION_VERSION = "public-state-v2"
RULES = {
    "players": 6,
    "positions": ["UTG", "HJ", "CO", "BTN", "SB", "BB"],
    "streets": ["preflop", "flop", "turn", "river"],
    "actions": ["fold", "check", "call", "raise"],
    "blinds": {"small": 0.5, "big": 1.0},
    "betSizing": {
        "preflopOpenBb": 2.5,
        "preflopFirstReraiseMultiplier": 3.0,
        "preflopLaterReraiseMultiplier": 2.5,
        "postflopOpenPotFraction": 0.5,
        "postflopMinimumBetBb": 0.5,
        "postflopRaiseMultiplier": 3.0,
    },
    "maxAggressiveActionsPerStreet": 3,
    "amountPrecision": 0.01,
}
RULES_HASH = hashlib.sha256(
    json.dumps(RULES, separators=(",", ":"), sort_keys=True).encode("utf-8")
).hexdigest()[:16]


def _transform_card(card: int, suit_permutation: tuple[int, ...]) -> int:
    return (int(card) // 4) * 4 + suit_permutation[int(card) % 4]


def _transform_info(info: dict[str, Any], suit_permutation: tuple[int, ...]) -> dict[str, Any]:
    transformed = json.loads(json.dumps(info))
    transformed["board"] = [
        _transform_card(card, suit_permutation) for card in transformed["board"]
    ]
    transformed["hero"]["hole"] = sorted(
        _transform_card(card, suit_permutation) for card in transformed["hero"]["hole"]
    )
    for entry in transformed["history"]:
        if "board" in entry:
            entry["board"] = [
                _transform_card(card, suit_permutation) for card in entry["board"]
            ]
    return transformed


def canonical_suit_info(info: dict[str, Any]) -> dict[str, Any]:
    """Choose the lexicographically minimal of all 24 suit-isomorphic states."""
    normalized = normalize_solve_info(info)
    # ``actor`` is a presentation label (for example ``Hero`` or ``Bot 3``),
    # not poker information.  Keeping it in an exact key caused the same
    # position/action state to be trained and cached once per UI identity.
    for entry in normalized["history"]:
        entry.pop("actor", None)
    candidates = (
        _transform_info(normalized, permutation)
        for permutation in permutations(range(4))
    )
    return min(
        candidates,
        key=lambda candidate: json.dumps(
            candidate,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def stable_public_key(info: dict[str, Any]) -> tuple[str, str]:
    payload = {
        "abstractionVersion": ABSTRACTION_VERSION,
        "rulesHash": RULES_HASH,
        "state": canonical_suit_info(info),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()
    return digest, canonical
