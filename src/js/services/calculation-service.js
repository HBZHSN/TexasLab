import { remainingCardCodes } from "../core/cards.js";
import { findDrawOuts } from "../core/draws.js";
import { choose } from "../core/probability.js";
import { TARGET_BOARD_COUNTS } from "../core/constants.js";
import { runWorkerTask } from "./worker-client.js";

const ROW_LIMIT = 1500;

function partition(items, partCount) {
  const parts = Array.from({ length: Math.min(partCount, items.length) }, () => []);
  items.forEach((item, index) => parts[index % parts.length].push(item));
  return parts;
}

export async function calculateScenario({
  hero,
  board,
  target,
  players,
  iterations,
  selected,
  workerCount,
  onProgress = () => {},
}) {
  const remaining = remainingCardCodes([...hero, ...board]);
  const need = Math.max(0, TARGET_BOARD_COUNTS[target] - board.length);
  const total = choose(remaining.length, need);
  const outs = findDrawOuts([...hero, ...board]);

  const enumerationTasks = [];
  if (need === 0) {
    enumerationTasks.push(
      runWorkerTask({
        type: "enumerate",
        id: "enumerate-0",
        hero,
        board,
        remaining,
        need,
        selected,
        outs,
        starts: [],
        sampleCap: ROW_LIMIT,
      }),
    );
  } else {
    const starts = Array.from(
      { length: remaining.length - need + 1 },
      (_, index) => index,
    );
    const parts = partition(starts, workerCount);
    parts.forEach((startPart, index) => {
      enumerationTasks.push(
        runWorkerTask({
          type: "enumerate",
          id: `enumerate-${index}`,
          hero,
          board,
          remaining,
          need,
          selected,
          outs,
          starts: startPart,
          sampleCap: Math.ceil(ROW_LIMIT / parts.length),
        }),
      );
    });
  }

  const iterationParts = Array.from(
    { length: workerCount },
    (_, index) =>
      Math.floor(iterations / workerCount) + (index < iterations % workerCount ? 1 : 0),
  );
  const equityTasks = iterationParts
    .filter(Boolean)
    .map((count, index) =>
      runWorkerTask({
        type: "equity",
        id: `equity-${index}`,
        hero,
        board,
        players,
        iterations: count,
        seed: (Date.now() + index * 100003) | 0,
      }),
    );

  onProgress(15);
  const [enumerations, equities] = await Promise.all([
    Promise.all(enumerationTasks),
    Promise.all(equityTasks),
  ]);
  onProgress(92);

  const counts = Object.fromEntries(selected.map((event) => [event, 0]));
  let processed = 0;
  let rows = [];
  for (const result of enumerations) {
    processed += result.total;
    selected.forEach((event) => {
      counts[event] += result.counts[event];
    });
    rows = rows.concat(result.rows);
  }

  const equityTotals = equities.reduce(
    (totals, result) => ({
      equity: totals.equity + result.equity,
      iterations: totals.iterations + result.iterations,
      wins: totals.wins + result.wins,
      ties: totals.ties + result.ties,
      losses: totals.losses + result.losses,
    }),
    { equity: 0, iterations: 0, wins: 0, ties: 0, losses: 0 },
  );

  return {
    hero,
    board,
    remaining,
    need,
    total,
    processed,
    selected,
    counts,
    rows: rows.slice(0, ROW_LIMIT),
    outs,
    equity: equityTotals.equity / equityTotals.iterations,
    ...equityTotals,
  };
}
