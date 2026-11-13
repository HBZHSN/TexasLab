import { HAND_CATEGORIES } from "./constants.js";

function straightHigh(rankCounts) {
  for (let high = 14; high >= 5; high -= 1) {
    let complete = true;
    for (let rank = high - 4; rank <= high; rank += 1) {
      const normalizedRank = rank === 1 ? 14 : rank;
      if (!rankCounts[normalizedRank]) {
        complete = false;
        break;
      }
    }
    if (complete) return high;
  }
  return 0;
}

function packScore(category, a = 0, b = 0, c = 0, d = 0, e = 0) {
  return (((((category * 15 + a) * 15 + b) * 15 + c) * 15 + d) * 15 + e);
}

export function evaluateHand(codes) {
  const rankCounts = Array(15).fill(0);
  const suitRanks = [[], [], [], []];

  for (const code of codes) {
    const rank = 2 + Math.floor(code / 4);
    const suit = code % 4;
    rankCounts[rank] += 1;
    suitRanks[suit].push(rank);
  }

  const flushRanks = suitRanks
    .find((ranks) => ranks.length >= 5)
    ?.slice()
    .sort((left, right) => right - left);

  if (flushRanks) {
    const flushCounts = Array(15).fill(0);
    flushRanks.forEach((rank) => {
      flushCounts[rank] += 1;
    });
    const high = straightHigh(flushCounts);
    if (high) return { score: packScore(8, high), category: 8 };
  }

  for (let rank = 14; rank >= 2; rank -= 1) {
    if (rankCounts[rank] === 4) {
      let kicker = 14;
      while (kicker === rank || !rankCounts[kicker]) kicker -= 1;
      return { score: packScore(7, rank, kicker), category: 7 };
    }
  }

  const trips = [];
  const pairs = [];
  for (let rank = 14; rank >= 2; rank -= 1) {
    if (rankCounts[rank] >= 3) trips.push(rank);
    if (rankCounts[rank] >= 2) pairs.push(rank);
  }

  if (trips.length) {
    const pair = pairs.find((rank) => rank !== trips[0]);
    if (pair) return { score: packScore(6, trips[0], pair), category: 6 };
  }

  if (flushRanks) {
    return { score: packScore(5, ...flushRanks.slice(0, 5)), category: 5 };
  }

  const highStraight = straightHigh(rankCounts);
  if (highStraight) return { score: packScore(4, highStraight), category: 4 };

  if (trips.length) {
    const kickers = [];
    for (let rank = 14; rank >= 2 && kickers.length < 2; rank -= 1) {
      if (rank !== trips[0] && rankCounts[rank]) kickers.push(rank);
    }
    return { score: packScore(3, trips[0], ...kickers), category: 3 };
  }

  if (pairs.length >= 2) {
    let kicker = 14;
    while (pairs.slice(0, 2).includes(kicker) || !rankCounts[kicker]) kicker -= 1;
    return { score: packScore(2, pairs[0], pairs[1], kicker), category: 2 };
  }

  if (pairs.length === 1) {
    const kickers = [];
    for (let rank = 14; rank >= 2 && kickers.length < 3; rank -= 1) {
      if (rank !== pairs[0] && rankCounts[rank]) kickers.push(rank);
    }
    return { score: packScore(1, pairs[0], ...kickers), category: 1 };
  }

  const highCards = [];
  for (let rank = 14; rank >= 2 && highCards.length < 5; rank -= 1) {
    if (rankCounts[rank]) highCards.push(rank);
  }
  return { score: packScore(0, ...highCards), category: 0 };
}

export function currentHandName(codes) {
  if (codes.length >= 5) return HAND_CATEGORIES[evaluateHand(codes).category];

  const counts = new Map();
  for (const code of codes) {
    const rank = 2 + Math.floor(code / 4);
    counts.set(rank, (counts.get(rank) ?? 0) + 1);
  }
  const maxCount = Math.max(1, ...counts.values());
  if (maxCount >= 3) return "三条";
  if (maxCount === 2) return "一对";
  return "高牌";
}
