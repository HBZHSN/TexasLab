import { RANKS, RANK_TEXT, SUITS } from "./constants.js";

export function cardCode(rank, suit) {
  return (rank - 2) * 4 + suit;
}

export const DECK = RANKS.flatMap((rank) =>
  SUITS.map((suit) => ({
    rank,
    suit: suit.id,
    text: RANK_TEXT[rank] + suit.symbol,
    red: suit.red,
    code: cardCode(rank, suit.id),
  })),
);

export const CARDS_BY_CODE = new Map(DECK.map((card) => [card.code, card]));

export function remainingCardCodes(knownCodes) {
  const known = new Set(knownCodes);
  return Array.from({ length: 52 }, (_, code) => code).filter(
    (code) => !known.has(code),
  );
}

export function createStartingHands() {
  const hands = [];

  for (let first = 0; first < RANKS.length; first += 1) {
    for (let second = first; second < RANKS.length; second += 1) {
      const high = RANKS[first];
      const low = RANKS[second];
      const label = RANK_TEXT[high] + RANK_TEXT[low];

      if (high === low) {
        hands.push({ key: label, r1: high, r2: low, kind: "p", type: "口袋对子" });
      } else {
        hands.push({ key: `${label}s`, r1: high, r2: low, kind: "s", type: "同花" });
        hands.push({ key: `${label}o`, r1: high, r2: low, kind: "o", type: "非同花" });
      }
    }
  }

  return hands.sort((left, right) => {
    if (left.r1 !== right.r1) return right.r1 - left.r1;
    if (left.r2 !== right.r2) return right.r2 - left.r2;
    return left.kind.localeCompare(right.kind);
  });
}

export function startingHandCodes(hand) {
  if (hand.r1 === hand.r2) {
    return [cardCode(hand.r1, 0), cardCode(hand.r2, 1)];
  }
  if (hand.kind === "s") {
    return [cardCode(hand.r1, 0), cardCode(hand.r2, 0)];
  }
  return [cardCode(hand.r1, 0), cardCode(hand.r2, 1)];
}
