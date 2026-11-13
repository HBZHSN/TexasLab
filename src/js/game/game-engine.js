import { CARDS_BY_CODE, remainingCardCodes } from "../core/cards.js";
import { evaluateHand } from "../core/evaluator.js";

export const POSITIONS = ["UTG", "HJ", "CO", "BTN", "SB", "BB"];
export const STREETS = ["preflop", "flop", "turn", "river"];
export const STREET_NAMES = {
  preflop: "翻牌前",
  flop: "翻牌",
  turn: "转牌",
  river: "河牌",
};

const EPSILON = 1e-9;

export function createSeededRandom(seed) {
  let value = Number(seed) >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

function newSeed() {
  if (globalThis.crypto?.getRandomValues) {
    return globalThis.crypto.getRandomValues(new Uint32Array(1))[0];
  }
  return Math.floor(Math.random() * 4294967296) >>> 0;
}

function shuffledDeck(random) {
  const deck = Array.from({ length: 52 }, (_, code) => code);
  for (let index = deck.length - 1; index > 0; index -= 1) {
    const other = Math.floor(random() * (index + 1));
    [deck[index], deck[other]] = [deck[other], deck[index]];
  }
  return deck;
}

function nextOrder(game, actorId = null) {
  const base = game.street === "preflop"
    ? ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
    : ["SB", "BB", "UTG", "HJ", "CO", "BTN"];
  const ordered = base.map((position) => game.players.find((player) => player.position === position));
  if (actorId === null) return ordered;
  const actorIndex = ordered.findIndex((player) => player.id === actorId);
  return ordered.slice(actorIndex + 1).concat(ordered.slice(0, actorIndex + 1));
}

function canAct(player) {
  return !player.folded && !player.allIn;
}

function commit(player, chips) {
  const paid = Math.min(player.stack, Math.max(0, chips));
  player.stack -= paid;
  player.streetCommitted += paid;
  player.totalCommitted += paid;
  if (player.stack <= EPSILON) {
    player.stack = 0;
    player.allIn = true;
  }
  return paid;
}

function potSize(game) {
  return game.players.reduce((sum, player) => sum + player.totalCommitted, 0);
}

function formatBB(value) {
  return Number(value.toFixed(2));
}

function publicAction(game, player, type, amount = 0) {
  const entry = {
    street: game.street,
    board: game.board.slice(),
    playerId: player.id,
    position: player.position,
    actor: player.isHero ? "Hero" : player.name,
    type,
    amount: formatBB(amount),
    potAfter: formatBB(potSize(game)),
  };
  game.history.push(entry);
  return entry;
}

function dealNextStreet(game) {
  const nextIndex = STREETS.indexOf(game.street) + 1;
  if (nextIndex >= STREETS.length) return false;
  game.street = STREETS[nextIndex];
  const count = game.street === "flop" ? 3 : 1;
  game.board.push(...game.deck.splice(0, count));
  game.players.forEach((player) => {
    player.streetCommitted = 0;
  });
  game.currentBet = 0;
  game.aggressiveActions = 0;
  game.pending = nextOrder(game).filter(canAct).map((player) => player.id);
  return true;
}

function buildSidePots(game) {
  const levels = [...new Set(game.players.map((player) => player.totalCommitted).filter(Boolean))]
    .sort((left, right) => left - right);
  let previous = 0;
  return levels.map((level) => {
    const contributors = game.players.filter((player) => player.totalCommitted + EPSILON >= level);
    const amount = (level - previous) * contributors.length;
    previous = level;
    return {
      amount,
      eligible: contributors.filter((player) => !player.folded),
    };
  }).filter((pot) => pot.amount > EPSILON);
}

function finishByFold(game) {
  const winner = game.players.find((player) => !player.folded);
  const amount = potSize(game);
  winner.stack += amount;
  game.complete = true;
  game.result = {
    reason: "fold",
    winners: [{ playerId: winner.id, amount: formatBB(amount) }],
    pots: [{ amount: formatBB(amount), winners: [winner.id] }],
  };
}

function finishShowdown(game) {
  while (game.board.length < 5) game.board.push(...game.deck.splice(0, game.board.length ? 1 : 3));
  const pots = buildSidePots(game);
  const awards = new Map();
  const settledPots = pots.map((pot) => {
    const scores = pot.eligible.map((player) => ({
      player,
      score: evaluateHand(player.hole.concat(game.board)).score,
    }));
    const best = Math.max(...scores.map(({ score }) => score));
    const winners = scores.filter(({ score }) => score === best).map(({ player }) => player);
    const share = pot.amount / winners.length;
    winners.forEach((winner) => awards.set(winner.id, (awards.get(winner.id) ?? 0) + share));
    return { amount: formatBB(pot.amount), winners: winners.map((winner) => winner.id) };
  });
  awards.forEach((amount, playerId) => {
    game.players.find((player) => player.id === playerId).stack += amount;
  });
  game.complete = true;
  game.result = {
    reason: "showdown",
    winners: [...awards].map(([playerId, amount]) => ({ playerId, amount: formatBB(amount) })),
    pots: settledPots,
  };
}

function finishRound(game) {
  if (game.players.filter((player) => !player.folded).length === 1) {
    finishByFold(game);
    return;
  }
  if (game.street === "river") {
    finishShowdown(game);
    return;
  }
  dealNextStreet(game);
  if (!game.pending.length) finishRound(game);
}

export function createGame({ random, heroPosition, seed } = {}) {
  const startingStack = 100;
  const handSeed = seed === undefined ? newSeed() : Number(seed) >>> 0;
  const gameRandom = random ?? createSeededRandom(handSeed);
  const deck = shuffledDeck(gameRandom);
  const selectedHeroPosition = heroPosition ?? POSITIONS[Math.floor(gameRandom() * POSITIONS.length)];
  const heroPositionIndex = POSITIONS.indexOf(selectedHeroPosition);
  const players = POSITIONS.map((position, index) => ({
    id: `p${index}`,
    name: `Bot ${index + 1}`,
    position,
    seatIndex: (index - heroPositionIndex + 4 + POSITIONS.length) % POSITIONS.length,
    isHero: position === selectedHeroPosition,
    stack: startingStack,
    hole: [],
    folded: false,
    allIn: false,
    streetCommitted: 0,
    totalCommitted: 0,
  }));
  players.find((player) => player.isHero).name = "Hero";
  for (let round = 0; round < 2; round += 1) {
    players.forEach((player) => player.hole.push(deck.shift()));
  }
  const game = {
    seed: handSeed,
    startingStack,
    players,
    deck,
    board: [],
    street: "preflop",
    currentBet: 1,
    aggressiveActions: 0,
    pending: [],
    history: [],
    complete: false,
    result: null,
  };
  commit(players.find((player) => player.position === "SB"), 0.5);
  commit(players.find((player) => player.position === "BB"), 1);
  game.pending = nextOrder(game).filter(canAct).map((player) => player.id);
  return game;
}

export function getPot(game) {
  return formatBB(potSize(game));
}

export function currentPlayer(game) {
  return game.players.find((player) => player.id === game.pending[0]) ?? null;
}

export function raiseTarget(game, player) {
  const totalAvailable = player.streetCommitted + player.stack;
  let target;
  if (game.street === "preflop") {
    if (game.aggressiveActions === 0) target = 2.5;
    else if (game.aggressiveActions === 1) target = game.currentBet * 3;
    else target = game.currentBet * 2.5;
  } else if (game.aggressiveActions === 0) {
    target = Math.max(0.5, potSize(game) * 0.5);
  } else {
    target = game.currentBet * 3;
  }
  return formatBB(Math.min(totalAvailable, target));
}

export function legalActions(game, playerId = game.pending[0]) {
  const player = game.players.find((item) => item.id === playerId);
  if (!player || game.complete || player.id !== game.pending[0]) return [];
  const toCall = Math.max(0, game.currentBet - player.streetCommitted);
  const actions = toCall <= EPSILON ? ["check"] : ["fold", "call"];
  const target = raiseTarget(game, player);
  if (game.aggressiveActions < 3 && player.stack > toCall + EPSILON && target > game.currentBet + EPSILON) {
    actions.push("raise");
  }
  return actions;
}

export function act(game, playerId, action, { deferAdvance = false } = {}) {
  const player = currentPlayer(game);
  if (!player || player.id !== playerId) throw new Error("当前不是该玩家行动");
  if (!legalActions(game, playerId).includes(action)) throw new Error(`非法操作：${action}`);
  const toCall = Math.max(0, game.currentBet - player.streetCommitted);
  game.pending = game.pending.filter((id) => id !== player.id);

  if (action === "fold") {
    player.folded = true;
    publicAction(game, player, "fold");
  } else if (action === "check") {
    publicAction(game, player, "check");
  } else if (action === "call") {
    const paid = commit(player, toCall);
    publicAction(game, player, "call", paid);
  } else {
    const target = raiseTarget(game, player);
    commit(player, target - player.streetCommitted);
    game.currentBet = player.streetCommitted;
    game.aggressiveActions += 1;
    game.pending = nextOrder(game, player.id)
      .filter((other) => other.id !== player.id && canAct(other))
      .map((other) => other.id);
    publicAction(game, player, "raise", player.streetCommitted);
  }

  if (game.players.filter((item) => !item.folded).length === 1) {
    finishByFold(game);
  } else if (!game.pending.length) {
    if (deferAdvance) game.roundComplete = true;
    else finishRound(game);
  }
  return game;
}

export function advanceRound(game) {
  if (!game.roundComplete || game.complete) return game;
  game.roundComplete = false;
  finishRound(game);
  return game;
}

export function createInformationSet(game, playerId) {
  const player = game.players.find((item) => item.id === playerId);
  return {
    street: game.street,
    board: game.board.slice(),
    pot: getPot(game),
    currentBet: game.currentBet,
    aggressiveActions: game.aggressiveActions,
    hero: {
      id: player.id,
      position: player.position,
      stack: player.stack,
      streetCommitted: player.streetCommitted,
      hole: player.hole.slice(),
    },
    players: game.players.map((item) => ({
      id: item.id,
      position: item.position,
      stack: item.stack,
      streetCommitted: item.streetCommitted,
      totalCommitted: item.totalCommitted,
      folded: item.folded,
      allIn: item.allIn,
    })),
    pending: game.pending.slice(),
    history: game.history.map((entry) => ({
      street: entry.street,
      board: entry.board?.slice() ?? [],
      playerId: entry.playerId,
      position: entry.position,
      actor: entry.actor,
      type: entry.type,
      amount: entry.amount,
      potAfter: entry.potAfter,
    })),
    legalActions: legalActions(game, playerId),
    raiseTo: raiseTarget(game, player),
  };
}

export function cardText(code) {
  return CARDS_BY_CODE.get(code).text;
}

export function verifyInformationIsolation(info) {
  const known = [...info.hero.hole, ...info.board];
  return remainingCardCodes(known).length === 52 - known.length
    && info.players.every((player) => !("hole" in player))
    && !("deck" in info);
}
