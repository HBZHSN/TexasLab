from __future__ import annotations

from copy import deepcopy
import unittest

from backend.abstraction import stable_public_key
from backend.schema import (
    PUBLIC_HISTORY_FIELDS,
    history_contains_private_cards,
    normalize_solve_info,
    public_history,
)


POSITIONS = ("UTG", "HJ", "CO", "BTN", "SB", "BB")


def sample_info() -> dict:
    players = []
    for index, position in enumerate(POSITIONS):
        active = index in (3, 5)
        players.append(
            {
                "id": f"p{index}",
                "position": position,
                "stack": 90.0 if active else 100.0,
                "streetCommitted": 0.0,
                "totalCommitted": 10.0 if active else 0.0,
                "folded": not active,
                "allIn": False,
            }
        )
    return {
        "street": "river",
        "board": [25, 12, 26, 5, 0],
        "pot": 20.0,
        "currentBet": 0.0,
        "aggressiveActions": 0,
        "hero": {
            "id": "p3",
            "position": "BTN",
            "stack": 90.0,
            "streetCommitted": 0.0,
            "hole": [31, 46],
        },
        "players": players,
        "pending": ["p3", "p5"],
        "history": [
            {
                "street": "preflop",
                "board": [],
                "playerId": "p3",
                "position": "BTN",
                "actor": "Hero",
                "type": "raise",
                "amount": 2.5,
                "potAfter": 4.0,
                "review": {"bestEV": 1.25},
                "context": {"uiOnly": True},
            }
        ],
        "legalActions": ["check", "raise"],
        "raiseTo": 10.0,
        "uiOnly": {"selectedTab": "review"},
    }


def permute_suits(info: dict, permutation: tuple[int, int, int, int]) -> dict:
    result = deepcopy(info)

    def transform(card: int) -> int:
        return (card // 4) * 4 + permutation[card % 4]

    result["board"] = [transform(card) for card in result["board"]]
    result["hero"]["hole"] = [transform(card) for card in result["hero"]["hole"]]
    for entry in result["history"]:
        if "board" in entry:
            entry["board"] = [transform(card) for card in entry["board"]]
    return result


class PublicSchemaTests(unittest.TestCase):
    def test_history_projection_is_an_explicit_public_field_whitelist(self):
        source = sample_info()["history"]
        projected = public_history(source)

        self.assertEqual(set(projected[0]), set(PUBLIC_HISTORY_FIELDS))
        self.assertNotIn("review", projected[0])
        self.assertNotIn("context", projected[0])

        # The projection owns its nested values and cannot mutate replay state.
        projected[0]["board"].append(51)
        self.assertEqual(source[0]["board"], [])

    def test_private_cards_are_detected_recursively_in_replay_metadata(self):
        private_examples = (
            [{"heroCards": [1, 2]}],
            [{"review": {"hole": [1, 2]}}],
            [{"context": [{"deck": [1, 2, 3]}]}],
            {"nested": {"holes": {"p1": [1, 2]}}},
        )
        for value in private_examples:
            with self.subTest(value=value):
                self.assertTrue(history_contains_private_cards(value))

        self.assertFalse(
            history_contains_private_cards(
                [{"type": "raise", "amount": 2.5, "holeCardsShown": False}]
            )
        )

    def test_normalized_request_discards_ui_and_replay_only_fields(self):
        normalized = normalize_solve_info(sample_info())

        self.assertNotIn("uiOnly", normalized)
        self.assertEqual(set(normalized["history"][0]), set(PUBLIC_HISTORY_FIELDS))
        self.assertNotIn("review", normalized["history"][0])
        self.assertNotIn("context", normalized["history"][0])
        self.assertEqual(
            set(normalized["hero"]),
            {"id", "position", "stack", "streetCommitted", "hole"},
        )

    def test_stable_public_key_is_invariant_under_global_suit_renaming(self):
        original = sample_info()
        isomorphic = permute_suits(original, (2, 0, 3, 1))

        original_digest, original_canonical = stable_public_key(original)
        isomorphic_digest, isomorphic_canonical = stable_public_key(isomorphic)

        self.assertEqual(isomorphic_digest, original_digest)
        self.assertEqual(isomorphic_canonical, original_canonical)

        different_rank = deepcopy(original)
        different_rank["hero"]["hole"][0] = 35
        self.assertNotEqual(stable_public_key(different_rank)[0], original_digest)

    def test_stable_public_key_ignores_ui_actor_label(self):
        original = sample_info()
        relabeled = deepcopy(original)
        relabeled["history"][0]["actor"] = "Bot 6"

        self.assertEqual(stable_public_key(original), stable_public_key(relabeled))


if __name__ == "__main__":
    unittest.main()
