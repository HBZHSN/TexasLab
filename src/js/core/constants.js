export const SUITS = [
  { id: 0, symbol: "♠", red: false },
  { id: 1, symbol: "♥", red: true },
  { id: 2, symbol: "♦", red: true },
  { id: 3, symbol: "♣", red: false },
];

export const RANKS = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2];

export const RANK_TEXT = {
  14: "A",
  13: "K",
  12: "Q",
  11: "J",
  10: "T",
  9: "9",
  8: "8",
  7: "7",
  6: "6",
  5: "5",
  4: "4",
  3: "3",
  2: "2",
};

export const HAND_CATEGORIES = [
  "高牌",
  "一对",
  "两对",
  "三条",
  "顺子",
  "同花",
  "葫芦",
  "四条",
  "同花顺",
];

export const EVENT_LABELS = {
  pair: "至少一对",
  twoPair: "两对或更好",
  set: "手对变三条",
  trips: "三条或更好",
  straight: "顺子或更好",
  flush: "同花或更好",
  fullHouse: "葫芦或更好",
  quads: "四条",
  straightFlush: "同花顺",
  gutshot: "卡顺补中",
  openEnded: "两头顺补中",
  flushDraw: "同花听牌补中",
};

export const STREET_BOARD_COUNTS = {
  preflop: 0,
  flop: 3,
  turn: 4,
  river: 5,
};

export const TARGET_BOARD_COUNTS = {
  flop: 3,
  turn: 4,
  river: 5,
  showdown: 5,
};
