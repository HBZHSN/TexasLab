import test from "node:test";
import assert from "node:assert/strict";

import {
  cardCode,
  createStartingHands,
  remainingCardCodes,
} from "../src/js/core/cards.js";
import { findDrawOuts } from "../src/js/core/draws.js";
import { evaluateHand } from "../src/js/core/evaluator.js";
import { choose } from "../src/js/core/probability.js";
import {
  enumerateEvents,
  simulateEquity,
} from "../src/js/core/simulation.js";
import { PRECOMPUTED_EQUITY } from "../src/js/data/precomputed-equity.js";
import {
  COMMON_GROUPS,
  COMMON_SCENARIOS,
  hypergeometricAtLeast,
  probabilityAtLeastOne,
  scenarioMetrics,
} from "../src/js/data/common-probabilities.js";
import {
  startingHandKey,
  startingHandMatrixKey,
} from "../src/js/ui/reference-tables.js";

const c = cardCode;

test("牌型评估覆盖全部九类牌型", () => {
  const hands = [
    [c(14, 0), c(13, 1), c(9, 2), c(7, 3), c(4, 0), c(3, 1), c(2, 2)],
    [c(14, 0), c(14, 1), c(9, 2), c(7, 3), c(4, 0), c(3, 1), c(2, 2)],
    [c(14, 0), c(14, 1), c(9, 2), c(9, 3), c(4, 0), c(3, 1), c(2, 2)],
    [c(14, 0), c(14, 1), c(14, 2), c(9, 3), c(4, 0), c(3, 1), c(2, 2)],
    [c(9, 0), c(8, 1), c(7, 2), c(6, 3), c(5, 0), c(3, 1), c(2, 2)],
    [c(14, 0), c(11, 0), c(9, 0), c(6, 0), c(3, 0), c(2, 1), c(2, 2)],
    [c(14, 0), c(14, 1), c(14, 2), c(9, 0), c(9, 1), c(3, 1), c(2, 2)],
    [c(14, 0), c(14, 1), c(14, 2), c(14, 3), c(9, 0), c(3, 1), c(2, 2)],
    [c(9, 0), c(8, 0), c(7, 0), c(6, 0), c(5, 0), c(3, 1), c(2, 2)],
  ];

  assert.deepEqual(
    hands.map((hand) => evaluateHand(hand).category),
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
  );
});

test("评估分数可正确比较同类牌型", () => {
  const aceHighFlush = [c(14, 0), c(11, 0), c(9, 0), c(6, 0), c(3, 0), c(2, 1), c(2, 2)];
  const kingHighFlush = [c(13, 1), c(11, 1), c(9, 1), c(6, 1), c(3, 1), c(2, 0), c(2, 2)];
  assert.ok(evaluateHand(aceHighFlush).score > evaluateHand(kingHighFlush).score);
});

test("组合数计算处理边界和常用值", () => {
  assert.equal(choose(52, 2), 1326);
  assert.equal(choose(47, 2), 1081);
  assert.equal(choose(10, 0), 1);
  assert.equal(choose(3, 4), 0);
});

test("听牌识别返回干净的牌码集合", () => {
  const straightDraw = [c(9, 0), c(8, 1), c(7, 2), c(6, 3)];
  assert.equal(findDrawOuts(straightDraw).open.length, 8);

  const flushDraw = [c(14, 0), c(12, 0), c(7, 0), c(2, 0)];
  assert.equal(findDrawOuts(flushDraw).flush.length, 9);
});

test("精确枚举能计算口袋对子在下一张成三条的分子", () => {
  const hero = [c(8, 0), c(8, 1)];
  const board = [c(14, 0), c(13, 1), c(2, 2)];
  const remaining = remainingCardCodes([...hero, ...board]);
  const result = enumerateEvents({
    hero,
    board,
    remaining,
    need: 1,
    selected: ["set"],
    outs: findDrawOuts([...hero, ...board]),
    starts: remaining.map((_, index) => index),
    sampleCap: 10,
  });

  assert.equal(result.total, 47);
  assert.equal(result.counts.set, 2);
});

test("Monte Carlo 结果保持计数守恒", () => {
  const result = simulateEquity({
    hero: [c(14, 0), c(14, 1)],
    board: [],
    players: 2,
    iterations: 200,
    seed: 12345,
  });
  assert.equal(result.wins + result.ties + result.losses, result.iterations);
  assert.ok(result.equity >= 0 && result.equity <= result.iterations);
});

test("起手牌与预计算数据维度完整", () => {
  assert.equal(createStartingHands().length, 169);
  assert.deepEqual(Object.keys(PRECOMPUTED_EQUITY), ["2", "3", "4", "5", "6", "7", "8", "9"]);
  Object.values(PRECOMPUTED_EQUITY).forEach((table) => {
    assert.equal(Object.keys(table).length, 169);
  });
});

test("起手牌矩阵按右上同花、左下非同花映射", () => {
  assert.equal(startingHandMatrixKey(14, 14), "AA");
  assert.equal(startingHandMatrixKey(14, 13), "AKs");
  assert.equal(startingHandMatrixKey(13, 14), "AKo");
  assert.equal(startingHandMatrixKey(2, 2), "22");
});

test("两张实体牌映射到对应的起手牌矩阵格子", () => {
  assert.equal(startingHandKey({ rank: 14, suit: 0 }, { rank: 13, suit: 0 }), "AKs");
  assert.equal(startingHandKey({ rank: 13, suit: 1 }, { rank: 14, suit: 0 }), "AKo");
  assert.equal(startingHandKey({ rank: 8, suit: 2 }, { rank: 8, suit: 3 }), "88");
});

test("常见 outs 概率使用无放回组合公式", () => {
  assert.ok(Math.abs(probabilityAtLeastOne(9, 1) - 9 / 47) < 1e-12);
  assert.ok(Math.abs(probabilityAtLeastOne(9, 2) - 0.3496762257169288) < 1e-12);
  assert.ok(Math.abs(probabilityAtLeastOne(9, 1, 46) - 9 / 46) < 1e-12);
});

test("同花起手牌到河牌成花使用超几何分布", () => {
  const probability = hypergeometricAtLeast({
    successes: 11,
    failures: 39,
    draws: 5,
    minimum: 3,
  });
  assert.ok(Math.abs(probability - 0.06399828201400819) < 1e-12);
});

test("常见概率场景覆盖全部牌型分组且标识唯一", () => {
  assert.equal(COMMON_SCENARIOS.length, 26);
  assert.equal(new Set(COMMON_SCENARIOS.map((scenario) => scenario.id)).size, 26);
  COMMON_GROUPS.forEach((group) => {
    assert.ok(COMMON_SCENARIOS.some((scenario) => scenario.group === group.id));
  });

  const flushDraw = COMMON_SCENARIOS.find((scenario) => scenario.id === "flush-draw");
  assert.equal(scenarioMetrics(flushDraw)[1].value, probabilityAtLeastOne(9, 2));
});
