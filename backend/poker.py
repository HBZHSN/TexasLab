from __future__ import annotations


def _straight_high(counts: list[int]) -> int:
    for high in range(14, 4, -1):
        ranks = [14 if rank == 1 else rank for rank in range(high - 4, high + 1)]
        if all(counts[rank] for rank in ranks):
            return high
    return 0


def _pack(category: int, *values: int) -> tuple[int, ...]:
    return (category, *values, *(0 for _ in range(5 - len(values))))


def evaluate(cards: list[int]) -> tuple[int, ...]:
    """Return a directly comparable 5-card hand score from 5-7 encoded cards."""
    counts = [0] * 15
    suits: list[list[int]] = [[], [], [], []]
    for card in cards:
        rank = 2 + card // 4
        counts[rank] += 1
        suits[card % 4].append(rank)

    flush = next((sorted(ranks, reverse=True) for ranks in suits if len(ranks) >= 5), None)
    if flush:
        flush_counts = [0] * 15
        for rank in flush:
            flush_counts[rank] += 1
        high = _straight_high(flush_counts)
        if high:
            return _pack(8, high)

    quads = next((rank for rank in range(14, 1, -1) if counts[rank] == 4), 0)
    if quads:
        kicker = next(rank for rank in range(14, 1, -1) if rank != quads and counts[rank])
        return _pack(7, quads, kicker)

    trips = [rank for rank in range(14, 1, -1) if counts[rank] >= 3]
    pairs = [rank for rank in range(14, 1, -1) if counts[rank] >= 2]
    if trips:
        pair = next((rank for rank in pairs if rank != trips[0]), 0)
        if pair:
            return _pack(6, trips[0], pair)
    if flush:
        return _pack(5, *flush[:5])
    straight = _straight_high(counts)
    if straight:
        return _pack(4, straight)
    if trips:
        kickers = [rank for rank in range(14, 1, -1) if rank != trips[0] and counts[rank]][:2]
        return _pack(3, trips[0], *kickers)
    if len(pairs) >= 2:
        kicker = next(rank for rank in range(14, 1, -1) if rank not in pairs[:2] and counts[rank])
        return _pack(2, pairs[0], pairs[1], kicker)
    if pairs:
        kickers = [rank for rank in range(14, 1, -1) if rank != pairs[0] and counts[rank]][:3]
        return _pack(1, pairs[0], *kickers)
    return _pack(0, *[rank for rank in range(14, 1, -1) if counts[rank]][:5])
