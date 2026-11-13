from __future__ import annotations

import bisect
from dataclasses import dataclass
from itertools import combinations
import math
import random
from typing import Callable, Iterable


ALL_COMBOS: tuple[tuple[int, int], ...] = tuple(combinations(range(52), 2))
COMBO_INDEX = {combo: index for index, combo in enumerate(ALL_COMBOS)}
COMBO_MASKS: tuple[int, ...] = tuple((1 << first) | (1 << second) for first, second in ALL_COMBOS)


def cards_mask(cards: Iterable[int]) -> int:
    mask = 0
    for card in cards:
        mask |= 1 << int(card)
    return mask


@dataclass(frozen=True)
class RangeDiagnostics:
    entropy: float
    effective_sample_size: float
    supported_combos: int


@dataclass(frozen=True)
class ComboRange:
    """Stable 1326-combo marginal range used by Blueprint/value artifacts."""

    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.probabilities) != len(ALL_COMBOS):
            raise ValueError("范围必须包含全部 1326 个两张牌组合")
        if any(not math.isfinite(value) or value < 0 for value in self.probabilities):
            raise ValueError("范围包含非法概率")
        if abs(sum(self.probabilities) - 1.0) > 1e-6:
            raise ValueError("范围概率之和必须为 1")

    @classmethod
    def uniform(cls, blocked_cards: Iterable[int] = ()) -> "ComboRange":
        blocked = cards_mask(blocked_cards)
        valid = [not (mask & blocked) for mask in COMBO_MASKS]
        total = sum(valid)
        if not total:
            raise ValueError("没有可用手牌组合")
        return cls(tuple(float(value) / total for value in valid))

    @classmethod
    def from_weighted(
        cls,
        combos: list[tuple[int, int]],
        cumulative: list[float],
        blocked_cards: Iterable[int] = (),
    ) -> "ComboRange":
        blocked = cards_mask(blocked_cards)
        weights = [0.0] * len(ALL_COMBOS)
        previous = 0.0
        for combo, cumulative_weight in zip(combos, cumulative):
            weight = max(0.0, float(cumulative_weight) - previous)
            previous = float(cumulative_weight)
            ordered = tuple(sorted(combo))
            index = COMBO_INDEX[ordered]
            if not (COMBO_MASKS[index] & blocked):
                weights[index] = weight
        return cls._normalized(weights)

    @classmethod
    def _normalized(cls, weights: list[float]) -> "ComboRange":
        total = sum(weights)
        if total <= 0:
            raise ValueError("范围归一化后没有有效组合")
        return cls(tuple(weight / total for weight in weights))

    def condition_cards(self, blocked_cards: Iterable[int]) -> "ComboRange":
        blocked = cards_mask(blocked_cards)
        return self._normalized(
            [
                probability if not (mask & blocked) else 0.0
                for probability, mask in zip(self.probabilities, COMBO_MASKS)
            ]
        )

    def update_action(
        self,
        likelihood: Callable[[tuple[int, int]], float] | list[float] | tuple[float, ...],
    ) -> "ComboRange":
        if callable(likelihood):
            factors = [max(0.0, float(likelihood(combo))) for combo in ALL_COMBOS]
        else:
            if len(likelihood) != len(ALL_COMBOS):
                raise ValueError("行动似然必须覆盖 1326 个组合")
            factors = [max(0.0, float(value)) for value in likelihood]
        return self._normalized(
            [probability * factor for probability, factor in zip(self.probabilities, factors)]
        )

    def diagnostics(self) -> RangeDiagnostics:
        positive = [value for value in self.probabilities if value > 0]
        entropy = -sum(value * math.log(value) for value in positive)
        ess = 1 / sum(value * value for value in positive)
        return RangeDiagnostics(entropy, ess, len(positive))

    def sample(self, used_cards: Iterable[int], rng: random.Random) -> tuple[int, int]:
        blocked = cards_mask(used_cards)
        cumulative: list[float] = []
        choices: list[tuple[int, int]] = []
        total = 0.0
        for combo, mask, probability in zip(ALL_COMBOS, COMBO_MASKS, self.probabilities):
            if probability <= 0 or mask & blocked:
                continue
            total += probability
            cumulative.append(total)
            choices.append(combo)
        if not choices:
            raise RuntimeError("范围中没有不冲突的可采样手牌")
        index = bisect.bisect_left(cumulative, rng.random() * total)
        return choices[min(index, len(choices) - 1)]


@dataclass(frozen=True)
class BeliefState:
    ranges: dict[str, ComboRange]
    public_cards: tuple[int, ...]

    def condition_public_cards(self, cards: Iterable[int]) -> "BeliefState":
        board = tuple(cards)
        return BeliefState(
            {
                player_id: hand_range.condition_cards(board)
                for player_id, hand_range in self.ranges.items()
            },
            board,
        )

    def update_action(
        self,
        player_id: str,
        likelihood: Callable[[tuple[int, int]], float] | list[float] | tuple[float, ...],
    ) -> "BeliefState":
        updated = dict(self.ranges)
        updated[player_id] = updated[player_id].update_action(likelihood)
        return BeliefState(updated, self.public_cards)

    def sample_deal(
        self,
        player_order: Iterable[str],
        rng: random.Random,
        fixed_hands: dict[str, tuple[int, int]] | None = None,
    ) -> dict[str, tuple[int, int]]:
        hands = dict(fixed_hands or {})
        used = set(self.public_cards)
        for hand in hands.values():
            used.update(hand)
        for player_id in player_order:
            if player_id in hands:
                continue
            hand = self.ranges[player_id].sample(used, rng)
            hands[player_id] = hand
            used.update(hand)
        return hands

    def flattened(self, player_order: Iterable[str]) -> list[float]:
        return [
            probability
            for player_id in player_order
            for probability in self.ranges[player_id].probabilities
        ]
