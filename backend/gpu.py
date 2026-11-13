from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .poker import evaluate


SCORE_BASE = 15


def pack_score(score: tuple[int, ...]) -> int:
    packed = 0
    for value in score[:6]:
        packed = packed * SCORE_BASE + int(value)
    return packed


@dataclass(frozen=True)
class ComputeRuntime:
    backend: str
    device: str
    cuda_available: bool
    torch_version: str | None
    cuda_version: str | None


class BatchEvaluator:
    """Exact seven-card evaluator with a CUDA tensor implementation.

    CUDA receives large, same-shaped batches. The Python fallback intentionally
    uses the canonical evaluator so machines without CUDA retain exact results.
    """

    def __init__(self, device: str | None = None) -> None:
        self.torch = None
        self.device = "cpu"
        try:
            import torch

            self.torch = torch
            if device is not None:
                self.device = device
            elif torch.cuda.is_available():
                self.device = "cuda"
        except ImportError:
            pass

    @property
    def uses_cuda(self) -> bool:
        return self.torch is not None and self.device.startswith("cuda")

    def runtime(self) -> ComputeRuntime:
        if self.torch is None:
            return ComputeRuntime("python", "cpu", False, None, None)
        return ComputeRuntime(
            "torch-cuda" if self.uses_cuda else "python",
            str(self.device),
            self.uses_cuda,
            self.torch.__version__,
            self.torch.version.cuda,
        )

    def evaluate(self, hands: list[list[int]], batch_size: int = 131_072) -> list[int]:
        if not hands:
            return []
        if not self.uses_cuda:
            return [pack_score(evaluate(cards)) for cards in hands]
        scores: list[int] = []
        for start in range(0, len(hands), batch_size):
            scores.extend(self._evaluate_tensor(hands[start : start + batch_size]))
        return scores

    def evaluate_tensor_for_test(self, hands: list[list[int]]) -> list[int]:
        if self.torch is None:
            raise RuntimeError("PyTorch 未安装")
        return self._evaluate_tensor(hands)

    def _evaluate_tensor(self, hands: list[list[int]]) -> list[int]:
        torch = self.torch
        assert torch is not None
        with torch.inference_mode():
            cards = torch.as_tensor(hands, dtype=torch.int64, device=self.device)
            if cards.ndim != 2 or cards.shape[1] != 7:
                raise ValueError("GPU 牌型评估要求每行正好七张牌")
            ranks = cards.div(4, rounding_mode="floor") + 2
            suits = cards.remainder(4)
            rank_axis = torch.arange(2, 15, device=self.device)
            rank_counts = (ranks.unsqueeze(-1) == rank_axis).sum(dim=1)
            rank_present = rank_counts > 0
            suit_axis = torch.arange(4, device=self.device)
            suit_counts = (suits.unsqueeze(-1) == suit_axis).sum(dim=1)

            def highest(mask: Any) -> Any:
                values = torch.where(mask, rank_axis, torch.zeros_like(rank_axis))
                return values.max(dim=1).values

            def top(mask: Any, count: int) -> Any:
                values = torch.where(mask, rank_axis, torch.zeros_like(rank_axis))
                return torch.topk(values, count, dim=1).values

            straight_high = torch.zeros(cards.shape[0], dtype=torch.int64, device=self.device)
            wheel = rank_present[:, 12] & rank_present[:, 0] & rank_present[:, 1] & rank_present[:, 2] & rank_present[:, 3]
            straight_high = torch.where(wheel, torch.full_like(straight_high, 5), straight_high)
            for high in range(6, 15):
                start = high - 6
                made = rank_present[:, start : start + 5].all(dim=1)
                straight_high = torch.where(made, torch.full_like(straight_high, high), straight_high)

            flush_suit = torch.where(
                suit_counts >= 5,
                suit_axis,
                torch.full_like(suit_axis, -1),
            ).max(dim=1).values
            has_flush = flush_suit >= 0
            flush_mask = suits == flush_suit.unsqueeze(1)
            flush_ranks = torch.where(flush_mask, ranks, torch.zeros_like(ranks))
            flush_top = torch.topk(flush_ranks, 5, dim=1).values

            straight_flush_high = torch.zeros_like(straight_high)
            for suit in range(4):
                suited_present = ((ranks.unsqueeze(-1) == rank_axis) & (suits.unsqueeze(-1) == suit)).any(dim=1)
                suited_wheel = suited_present[:, 12] & suited_present[:, 0] & suited_present[:, 1] & suited_present[:, 2] & suited_present[:, 3]
                suited_high = torch.where(suited_wheel, torch.full_like(straight_high, 5), torch.zeros_like(straight_high))
                for high in range(6, 15):
                    start = high - 6
                    made = suited_present[:, start : start + 5].all(dim=1)
                    suited_high = torch.where(made, torch.full_like(suited_high, high), suited_high)
                straight_flush_high = torch.maximum(straight_flush_high, suited_high)

            quad_rank = highest(rank_counts == 4)
            trip_ranks = top(rank_counts >= 3, 2)
            pair_ranks = top(rank_counts >= 2, 3)
            unique_ranks = top(rank_present, 5)

            quad_kicker = highest(rank_present & (rank_axis != quad_rank.unsqueeze(1)))
            full_house_pair = highest(
                (rank_counts >= 2) & (rank_axis != trip_ranks[:, 0].unsqueeze(1))
            )
            trip_kickers = top(rank_present & (rank_axis != trip_ranks[:, 0].unsqueeze(1)), 2)
            two_pair_kicker = highest(
                rank_present
                & (rank_axis != pair_ranks[:, 0].unsqueeze(1))
                & (rank_axis != pair_ranks[:, 1].unsqueeze(1))
            )
            pair_kickers = top(rank_present & (rank_axis != pair_ranks[:, 0].unsqueeze(1)), 3)

            category = torch.zeros_like(straight_high)
            values = torch.zeros((cards.shape[0], 5), dtype=torch.int64, device=self.device)

            def assign(mask: Any, new_category: int, fields: list[Any]) -> None:
                nonlocal category, values
                category = torch.where(mask, torch.full_like(category, new_category), category)
                replacement = torch.zeros_like(values)
                for index, field in enumerate(fields):
                    replacement[:, index] = field
                values = torch.where(mask.unsqueeze(1), replacement, values)

            assign(torch.ones_like(category, dtype=torch.bool), 0, [*unique_ranks.unbind(dim=1)])
            has_pair = pair_ranks[:, 0] > 0
            assign(has_pair, 1, [pair_ranks[:, 0], *pair_kickers.unbind(dim=1)])
            has_two_pair = pair_ranks[:, 1] > 0
            assign(has_two_pair, 2, [pair_ranks[:, 0], pair_ranks[:, 1], two_pair_kicker])
            has_trips = trip_ranks[:, 0] > 0
            assign(has_trips, 3, [trip_ranks[:, 0], *trip_kickers.unbind(dim=1)])
            assign(straight_high > 0, 4, [straight_high])
            assign(has_flush, 5, [*flush_top.unbind(dim=1)])
            has_full_house = (trip_ranks[:, 0] > 0) & (full_house_pair > 0)
            assign(has_full_house, 6, [trip_ranks[:, 0], full_house_pair])
            assign(quad_rank > 0, 7, [quad_rank, quad_kicker])
            assign(straight_flush_high > 0, 8, [straight_flush_high])

            packed = category
            for index in range(5):
                packed = packed * SCORE_BASE + values[:, index]
            return packed.cpu().tolist()


_EVALUATOR: BatchEvaluator | None = None


def get_batch_evaluator() -> BatchEvaluator:
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = BatchEvaluator()
    return _EVALUATOR


def showdown_shares_batch(
    showdowns: list[tuple[tuple[int, ...], tuple[tuple[int, int], ...]]],
) -> tuple[list[float], int]:
    evaluator = get_batch_evaluator()
    seven_card_hands: list[list[int]] = []
    widths: list[int] = []
    for board, hands in showdowns:
        widths.append(len(hands))
        seven_card_hands.extend([[*hand, *board] for hand in hands])
    scores = evaluator.evaluate(seven_card_hands)
    shares: list[float] = []
    offset = 0
    for width in widths:
        row = scores[offset : offset + width]
        offset += width
        best = max(row)
        shares.append(0.0 if row[0] < best else 1 / sum(score == best for score in row))
    return shares, len(seven_card_hands)


def settle_terminal_records(records: list[tuple[Any, ...]]) -> tuple[list[float], int]:
    evaluator = get_batch_evaluator()
    hands_to_evaluate: list[list[int]] = []
    record_offsets: list[tuple[int, int] | None] = []
    for record in records:
        if record[0] == "direct":
            record_offsets.append(None)
            continue
        _, board, rows, _, _ = record
        start = len(hands_to_evaluate)
        eligible = [row for row in rows if not row[3]]
        hands_to_evaluate.extend([[row[4], row[5], *board] for row in eligible])
        record_offsets.append((start, len(eligible)))

    scores = evaluator.evaluate(hands_to_evaluate)
    utilities: list[float] = []
    for record, offset in zip(records, record_offsets):
        if record[0] == "direct":
            utilities.append(float(record[1]))
            continue
        _, _, rows, target, baseline = record
        assert offset is not None
        start, width = offset
        eligible_rows = [row for row in rows if not row[3]]
        score_by_id = {
            row[0]: score for row, score in zip(eligible_rows, scores[start : start + width])
        }
        levels = sorted({row[2] for row in rows if row[2] > 1e-9})
        previous = 0.0
        target_award = 0.0
        for level in levels:
            contributors = [row for row in rows if row[2] + 1e-9 >= level]
            amount = (level - previous) * len(contributors)
            previous = level
            eligible_ids = [row[0] for row in contributors if not row[3]]
            best = max(score_by_id[player_id] for player_id in eligible_ids)
            winners = [player_id for player_id in eligible_ids if score_by_id[player_id] == best]
            if target in winners:
                target_award += amount / len(winners)
        target_stack = next(row[1] for row in rows if row[0] == target)
        utilities.append(target_stack + target_award - baseline)
    return utilities, len(hands_to_evaluate)
