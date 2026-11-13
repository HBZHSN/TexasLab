import test from "node:test";
import assert from "node:assert/strict";

import { buildAiHandText } from "../src/js/game/game-export.js";
import {
  act,
  createGame,
  currentPlayer,
  getPot,
} from "../src/js/game/game-engine.js";

test("完整分析记录包含决策上下文、求解详情、全部底牌和净输赢", () => {
  const game = {
    seed: 8675309,
    startingStack: 100,
    board: [48, 29, 2, 19, 34],
    complete: true,
    players: [
      { id: "p0", name: "Bot 1", position: "UTG", isHero: false, hole: [40, 13], folded: false, totalCommitted: 10, stack: 90 },
      { id: "p1", name: "Bot 2", position: "HJ", isHero: false, hole: [44, 45], folded: true, totalCommitted: 0, stack: 100 },
      { id: "p4", name: "Hero", position: "SB", isHero: true, hole: [48, 49], folded: false, totalCommitted: 10, stack: 110 },
    ],
    history: [
      {
        street: "preflop",
        board: [],
        playerId: "p0",
        position: "UTG",
        actor: "Bot 1",
        type: "raise",
        amount: 2.5,
        context: { pot: 1.5, stack: 100, effectiveStack: 99, currentBet: 1, toCall: 1 },
        review: {
          rating: "优秀",
          chosen: "raise",
          loss: 0,
          equity: 0.6123,
          iterations: 20_000,
          trainingIterations: 12_000,
          trainingRequestedIterations: 20_000,
          trainingPrunedIterations: 8_000,
          adaptiveTrainingDominanceWorkersStopped: 4,
          evaluationIterations: 2_000,
          evaluationRequestedIterations: 4_000,
          decisionEvaluationWorkersStopped: 2,
          evaluationPrunedIterations: 2_000,
          errorBudgetFraction: 0.05,
          confidenceMethod: "sequential-empirical-bernstein",
          cacheHit: true,
          evaluationPolicyBackoffRate: 0.251,
          evaluationUniformFallbackRate: 0.486,
          recommended: ["raise"],
          actions: [
            { action: "fold", ev: 0, frequency: 0.01, ci95: 0 },
            { action: "call", ev: 0.45, frequency: 0.09, ci95: 0.021 },
            { action: "raise", ev: 0.82, frequency: 0.9, ci95: 0.028 },
          ],
          randomRoll: 0.314159,
        },
      },
      {
        street: "preflop",
        board: [],
        playerId: "p4",
        position: "SB",
        actor: "Hero",
        type: "raise",
        amount: 7.5,
        context: { pot: 4, stack: 99.5, effectiveStack: 97.5, currentBet: 2.5, toCall: 2 },
        review: {
          rating: "优秀",
          chosen: "raise",
          loss: 0,
          equity: 0.71,
          iterations: 20_000,
          recommended: ["raise"],
          actions: [
            { action: "fold", ev: 0, frequency: 0.01, ci95: 0 },
            { action: "raise", ev: 2.125, frequency: 0.99, ci95: 0.035 },
          ],
        },
      },
      {
        street: "flop",
        board: [48, 29, 2],
        playerId: "p0",
        position: "UTG",
        actor: "Bot 1",
        type: "raise",
        amount: 5,
        context: { pot: 10, stack: 92.5, effectiveStack: 92.5, currentBet: 0, toCall: 0 },
      },
    ],
    result: { reason: "showdown", winners: [{ playerId: "p4", amount: 20 }] },
  };
  const text = buildAiHandText(game, 7);
  assert.match(text, /Texas Lab #7 \| 3-max \| 100BB/);
  assert.match(text, /Seed 8675309/);
  assert.match(text, /UTG Bot 1: Raise to 2\.5BB \| 行动前 Pot 1\.5BB/);
  assert.match(text, /评级 优秀 \| 推荐 Raise \| 所选 EV 0\.820BB \| EV 损失 0\.000BB/);
  assert.match(text, /范围 Equity 61\.2% \| 采样 20,000 次/);
  assert.match(text, /训练 12,000 次 \| 训练预算 20,000 次（EV 收敛后剪枝 8,000 次）/);
  assert.match(text, /翻牌前决策剪枝 4 个训练副本/);
  assert.match(text, /评估预算 4,000 次（翻牌前决策已确定，剪枝 2,000 次）/);
  assert.match(text, /目标误差预算 5\.0% \| 序贯经验 Bernstein 界/);
  assert.match(text, /精确缓存命中 \| Random 0\.314159/);
  assert.match(text, /分层策略回退 25\.1%/);
  assert.match(text, /均匀策略回退 48\.6%/);
  assert.match(text, /Raise 0\.820BB · 策略 90\.0% · 95% ±0\.028BB/);
  assert.match(text, /UTG Bot 1: Bet 5BB/);
  assert.match(text, /UTG Bot 1 \[Q♠ 5♥\] · 未弃牌/);
  assert.match(text, /HJ Bot 2 \[K♠ K♥\] · 已弃牌/);
  assert.match(text, /底池奖励 SB Hero 20BB/);
  assert.match(text, /净输赢 UTG Bot 1 -10BB（最终 90BB） \| HJ Bot 2 0BB（最终 100BB） \| SB Hero \+10BB（最终 110BB）/);
});

test("Seed 1604460729 的 Fold 牌局可导出全部 Bot 手牌、Bet 语义和真实净盈利", () => {
  const game = createGame({ seed: 1604460729 });
  const actions = [
    "fold", "call", "fold", "fold", "raise", "fold", "call",
    "check", "check", "raise", "call", "check", "raise", "fold",
  ];
  actions.forEach((action) => {
    const actor = currentPlayer(game);
    const activeOpponentStacks = game.players
      .filter((player) => !player.folded && player.id !== actor.id)
      .map((player) => player.stack);
    const context = {
      pot: getPot(game),
      stack: actor.stack,
      effectiveStack: Math.min(actor.stack, Math.max(...activeOpponentStacks)),
      currentBet: game.currentBet,
      toCall: Math.max(0, game.currentBet - actor.streetCommitted),
    };
    act(game, actor.id, action);
    game.history.at(-1).context = context;
  });

  const text = buildAiHandText(game, 4);
  assert.match(text, /SB Bot 5: Bet 3BB \| 行动前 Pot 6BB/);
  assert.match(text, /HJ Hero: Bet 6BB \| 行动前 Pot 12BB/);
  assert.match(text, /UTG Bot 1 \[K♣ 7♥\] · 已弃牌/);
  assert.match(text, /CO Bot 3 \[4♦ K♠\] · 已弃牌/);
  assert.match(text, /BTN Bot 4 \[4♣ 2♠\] · 已弃牌/);
  assert.match(text, /SB Bot 5 \[8♣ A♣\] · 已弃牌/);
  assert.match(text, /BB Bot 6 \[8♠ 5♣\] · 已弃牌/);
  assert.match(text, /结果 Fold/);
  assert.match(text, /底池奖励 HJ Hero 18BB/);
  assert.match(text, /HJ Hero \+6\.5BB（最终 106\.5BB）/);
  assert.doesNotMatch(text, /HJ Hero \+18BB/);
});
