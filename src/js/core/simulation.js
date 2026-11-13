import { remainingCardCodes, startingHandCodes } from "./cards.js";
import { evaluateHand } from "./evaluator.js";

function createRandom(seed) {
  let value = seed | 0;
  return () => {
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    return (value >>> 0) / 4294967296;
  };
}

export function simulateEquity({ hero, board, players, iterations, seed }) {
  const deck = remainingCardCodes([...hero, ...board]);
  const missingBoardCards = 5 - board.length;
  const cardsToDeal = missingBoardCards + 2 * (players - 1);
  const random = createRandom(seed);
  let equity = 0;
  let wins = 0;
  let ties = 0;
  let losses = 0;

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const shuffled = deck.slice();
    for (let index = 0; index < cardsToDeal; index += 1) {
      const swapIndex = index + Math.floor(random() * (shuffled.length - index));
      [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
    }

    let cursor = 0;
    const finalBoard = board.concat(shuffled.slice(cursor, cursor + missingBoardCards));
    cursor += missingBoardCards;
    const heroScore = evaluateHand(hero.concat(finalBoard)).score;
    let tiedOpponents = 0;
    let beaten = false;

    for (let opponent = 0; opponent < players - 1; opponent += 1) {
      const opponentScore = evaluateHand(
        [shuffled[cursor++], shuffled[cursor++]].concat(finalBoard),
      ).score;
      if (opponentScore > heroScore) {
        beaten = true;
        break;
      }
      if (opponentScore === heroScore) tiedOpponents += 1;
    }

    if (beaten) {
      losses += 1;
    } else if (tiedOpponents) {
      ties += 1;
      equity += 1 / (tiedOpponents + 1);
    } else {
      wins += 1;
      equity += 1;
    }
  }

  return { equity, wins, ties, losses, iterations };
}

function classifyEvents(finalCards, hero, run, selected, outs) {
  const evaluation = evaluateHand(finalCards);
  const events = {
    pair: evaluation.category >= 1,
    twoPair: evaluation.category >= 2,
    trips: evaluation.category >= 3,
    straight: evaluation.category >= 4,
    flush: evaluation.category >= 5,
    fullHouse: evaluation.category >= 6,
    quads: evaluation.category === 7,
    straightFlush: evaluation.category === 8,
  };

  const heroRank = 2 + Math.floor(hero[0] / 4);
  events.set =
    hero.length === 2 &&
    heroRank === 2 + Math.floor(hero[1] / 4) &&
    finalCards.filter((code) => 2 + Math.floor(code / 4) === heroRank).length >= 3;

  const runIntersects = (codes) => run.some((code) => codes.includes(code));
  events.gutshot = runIntersects(outs.gut) && evaluation.category >= 4;
  events.openEnded = runIntersects(outs.open) && evaluation.category >= 4;
  events.flushDraw = runIntersects(outs.flush) && evaluation.category >= 5;

  return {
    evaluation,
    hits: selected.filter((event) => events[event]),
  };
}

export function enumerateEvents({
  hero,
  board,
  remaining,
  need,
  selected,
  outs,
  starts,
  sampleCap,
}) {
  const counts = Object.fromEntries(selected.map((event) => [event, 0]));
  const rows = [];
  let total = 0;

  const handleRun = (run) => {
    total += 1;
    const { evaluation, hits } = classifyEvents(
      hero.concat(board, run),
      hero,
      run,
      selected,
      outs,
    );
    hits.forEach((event) => {
      counts[event] += 1;
    });
    if (hits.length && rows.length < sampleCap) {
      rows.push({ run, category: evaluation.category, hits });
    }
  };

  if (need === 0) {
    handleRun([]);
  } else {
    const visit = (start, left, picked) => {
      if (left === 0) {
        handleRun(picked.slice());
        return;
      }
      for (let index = start; index <= remaining.length - left; index += 1) {
        picked.push(remaining[index]);
        visit(index + 1, left - 1, picked);
        picked.pop();
      }
    };

    for (const start of starts) {
      const picked = [remaining[start]];
      if (need === 1) handleRun(picked);
      else visit(start + 1, need - 1, picked);
    }
  }

  return { counts, total, rows };
}

export function simulateStartingHands({ hands, players, iterations, seed }) {
  return Object.fromEntries(
    hands.map((hand) => {
      const result = simulateEquity({
        hero: startingHandCodes(hand),
        board: [],
        players,
        iterations,
        seed: (seed + hand.key.length * 7919 + hand.r1 * 97 + hand.r2 * 193) | 0,
      });
      return [hand.key, result.equity / result.iterations];
    }),
  );
}
