import { DECK } from "./cards.js";

export function findDrawOuts(codes) {
  const ranks = new Set(codes.map((code) => 2 + Math.floor(code / 4)));
  if (ranks.has(14)) ranks.add(1);

  const openEndedRanks = new Set();
  const gutshotRanks = new Set();

  for (let start = 2; start <= 10; start += 1) {
    const fourRanks = [start, start + 1, start + 2, start + 3];
    if (fourRanks.every((rank) => ranks.has(rank))) {
      openEndedRanks.add(start - 1 === 1 ? 14 : start - 1);
      openEndedRanks.add(start + 4);
    }
  }

  for (let start = 1; start <= 10; start += 1) {
    const window = [start, start + 1, start + 2, start + 3, start + 4];
    const missing = window.filter((rank) => !ranks.has(rank));
    if (missing.length === 1) {
      const rank = missing[0] === 1 ? 14 : missing[0];
      if (!openEndedRanks.has(rank)) gutshotRanks.add(rank);
    }
  }

  const suitCounts = [0, 0, 0, 0];
  codes.forEach((code) => {
    suitCounts[code % 4] += 1;
  });
  const flushSuits = suitCounts
    .map((count, suit) => (count === 4 ? suit : -1))
    .filter((suit) => suit >= 0);
  const usedCodes = new Set(codes);

  return {
    gut: DECK.filter(
      (card) => gutshotRanks.has(card.rank) && !usedCodes.has(card.code),
    ).map((card) => card.code),
    open: DECK.filter(
      (card) => openEndedRanks.has(card.rank) && !usedCodes.has(card.code),
    ).map((card) => card.code),
    flush: DECK.filter(
      (card) => flushSuits.includes(card.suit) && !usedCodes.has(card.code),
    ).map((card) => card.code),
  };
}
