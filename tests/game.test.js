import test from "node:test";
import assert from "node:assert/strict";

import {
  act,
  advanceRound,
  createGame,
  createSeededRandom,
  createInformationSet,
  currentPlayer,
  getPot,
  legalActions,
  verifyInformationIsolation,
} from "../src/js/game/game-engine.js";
import { GameUI, cardMarkup, chooseMixedAction, solveLogPresentation } from "../src/js/game/game-ui.js";

function seededRandom(seed = 123456) {
  let value = seed;
  return () => {
    value = (value * 1664525 + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

test("六人桌初始化盲注、位置和行动顺序正确", () => {
  const game = createGame({ random: seededRandom(), heroPosition: "BTN" });
  assert.equal(game.players.length, 6);
  assert.equal(getPot(game), 1.5);
  assert.equal(currentPlayer(game).position, "UTG");
  assert.deepEqual(legalActions(game), ["fold", "call", "raise"]);
  assert.equal(game.players.find((player) => player.isHero).seatIndex, 4);
  assert.equal(new Set(game.players.map((player) => player.seatIndex)).size, 6);
});

test("求解信息集不包含牌堆和任何对手底牌", () => {
  const game = createGame({ random: seededRandom(), heroPosition: "UTG" });
  const info = createInformationSet(game, currentPlayer(game).id);
  assert.equal(verifyInformationIsolation(info), true);
  assert.equal("deck" in info, false);
  assert.ok(info.players.every((player) => !("hole" in player)));
  assert.deepEqual(info.pending, game.pending);
  assert.ok(info.players.every((player) => Number.isFinite(player.totalCommitted)));
});

test("固定尺度翻牌前开池到 2.5BB 并重置待行动玩家", () => {
  const game = createGame({ random: seededRandom(), heroPosition: "UTG" });
  const utg = currentPlayer(game);
  act(game, utg.id, "raise");
  assert.equal(game.currentBet, 2.5);
  assert.equal(utg.streetCommitted, 2.5);
  assert.equal(game.aggressiveActions, 1);
  assert.equal(currentPlayer(game).position, "HJ");
});

test("所有其他玩家弃牌后立即向唯一玩家结算", () => {
  const game = createGame({ random: seededRandom(), heroPosition: "UTG" });
  while (!game.complete) act(game, currentPlayer(game).id, "fold");
  assert.equal(game.result.reason, "fold");
  assert.equal(game.result.winners.length, 1);
  assert.equal(game.players.reduce((sum, player) => sum + player.stack, 0), 600);
});

test("Hero 结束下注轮时公共牌延迟到确认后发出", () => {
  const game = createGame({ random: seededRandom(), heroPosition: "BB" });
  for (const action of ["fold", "fold", "fold", "fold", "call"]) {
    act(game, currentPlayer(game).id, action);
  }
  assert.equal(currentPlayer(game).position, "BB");
  act(game, currentPlayer(game).id, "check", { deferAdvance: true });
  assert.equal(game.board.length, 0);
  assert.equal(game.roundComplete, true);
  advanceRound(game);
  assert.equal(game.board.length, 3);
  assert.equal(game.street, "flop");
});

test("无人下注时只提供 Check 和 Raise，不提供 Fold", () => {
  const game = createGame({ random: seededRandom(), heroPosition: "BB" });
  for (const action of ["fold", "fold", "fold", "fold", "call", "check"]) {
    act(game, currentPlayer(game).id, action);
  }
  assert.equal(game.street, "flop");
  assert.equal(game.currentBet, 0);
  assert.deepEqual(legalActions(game), ["check", "raise"]);
});

test("相同种子复现位置、底牌、牌堆和 Bot 随机序列", () => {
  const first = createGame({ seed: 246813579 });
  const second = createGame({ seed: 246813579 });
  assert.equal(first.seed, 246813579);
  assert.deepEqual(
    first.players.map(({ position, isHero, hole }) => ({ position, isHero, hole })),
    second.players.map(({ position, isHero, hole }) => ({ position, isHero, hole })),
  );
  assert.deepEqual(first.deck, second.deck);
  const firstBotRandom = createSeededRandom(first.seed ^ 0x9e3779b9);
  const secondBotRandom = createSeededRandom(second.seed ^ 0x9e3779b9);
  assert.deepEqual(
    [firstBotRandom(), firstBotRandom(), firstBotRandom()],
    [secondBotRandom(), secondBotRandom(), secondBotRandom()],
  );
});

test("未收敛策略不会按残留频率随机选择负 EV 动作", () => {
  const analysis = {
    strategyReliable: false,
    actions: [
      { action: "fold", ev: 0, frequency: 0.65 },
      { action: "call", ev: -4.665, frequency: 0.169 },
      { action: "raise", ev: -2.881, frequency: 0.181 },
    ],
  };
  const selection = chooseMixedAction(analysis, () => 0.99);
  assert.equal(selection.action, "fold");
  assert.equal(selection.mode, "highest-ev-fallback");
  assert.equal(selection.roll, 0.99);
});

test("牌桌上的十点显示为 10 而不是 T", () => {
  assert.match(cardMarkup(32), /<b>10<\/b>/);
  assert.doesNotMatch(cardMarkup(32), /<b>T<\/b>/);
});

test("求解日志区分缓存、Blueprint 与现场训练", () => {
  const base = { status: "complete", requestedIterations: 20_000, elapsedMs: 1250 };
  const live = solveLogPresentation({
    ...base,
    result: {
      solveSource: "live-solve",
      solveElapsedMs: 1200,
      trainingIterations: 12_000,
      trainingRequestedIterations: 12_000,
      evaluationIterations: 8_000,
      aggregateIterations: 20_000,
      workerCount: 1,
      computeDevice: "cpu",
      rootRegret: 0.02,
      strategyReliable: true,
    },
  });
  const cached = solveLogPresentation({
    ...base,
    result: { solveSource: "memory-cache", blueprintExactHit: true, solveElapsedMs: 1 },
  });
  const blueprint = solveLogPresentation({
    ...base,
    result: { solveSource: "blueprint-exact", iterations: 20_000, solveElapsedMs: 3 },
  });

  assert.equal(live.source, "现场训练/求解");
  assert.match(live.lines.join(" "), /现场训练 12,000\/12,000 次/);
  assert.match(live.foot, /后端 1.20 秒/);
  assert.equal(cached.source, "内存缓存命中");
  assert.match(cached.lines[0], /原始结果来自 Blueprint/);
  assert.equal(blueprint.source, "Blueprint 精确结果");
  assert.match(blueprint.lines[0], /本次无需训练/);
});

test("Hero 选择后完成 EV 计算会直接推进，不等待二次确认", async () => {
  const previousStorage = globalThis.localStorage;
  globalThis.localStorage = {
    getItem: () => null,
    setItem: () => {},
  };

  class FlowTestUI extends GameUI {
    showSolving() {}
    renderSeats() {}
    renderActions() {}
    render() {}
    recordDecision() {}
    showAnalysis(review) { this.shownReview = review; }
    advanceBots() { this.advancedBots = true; }
    async solve() {
      return {
        actions: [
          {
            action: "call",
            ev: 0.4,
            stderr: 1,
            evLossEstimate: 0.2,
            differenceCi95: 0.03,
            frequency: 0,
          },
          {
            action: "raise",
            ev: 0.6,
            stderr: 1,
            evLossEstimate: 0,
            differenceCi95: 0,
            frequency: 1,
          },
        ],
        bestEV: 0.6,
        strategyReliable: true,
        rootRegret: 0,
        evaluationUntrainedRate: 0,
      };
    }
  }

  try {
    const ui = new FlowTestUI();
    ui.game = createGame({ seed: 0, heroPosition: "UTG" });
    await ui.heroAction("call");

    assert.equal(ui.game.history.at(-1).actor, "Hero");
    assert.equal(ui.game.history.at(-1).type, "call");
    assert.equal(currentPlayer(ui.game).position, "HJ");
    assert.equal(ui.busy, false);
    assert.equal(ui.solvingPlayerId, null);
    assert.equal(ui.advancedBots, true);
    assert.equal(ui.shownReview.chosen, "call");
    assert.equal(ui.shownReview.loss, 0.2);
    assert.equal(ui.shownReview.differenceCi95, 0.03);
    assert.equal(ui.shownReview.rating, "轻微错误");
    assert.equal("awaitingContinue" in ui, false);
  } finally {
    if (previousStorage === undefined) delete globalThis.localStorage;
    else globalThis.localStorage = previousStorage;
  }
});
