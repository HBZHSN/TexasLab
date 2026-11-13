import { CARDS_BY_CODE } from "../core/cards.js";
import { STREET_NAMES, getPot } from "./game-engine.js";

const ACTION_NAMES = { check: "Check", call: "Call", raise: "Raise", fold: "Fold" };
const EPSILON = 1e-9;

function cardsText(cards) {
  return cards.length ? cards.map((code) => CARDS_BY_CODE.get(code).text).join(" ") : "-";
}

function compactNumber(value) {
  return Number(value).toFixed(2).replace(/\.00$/, "").replace(/(\.\d)0$/, "$1");
}

function signedNumber(value) {
  const normalized = Math.abs(value) <= EPSILON ? 0 : value;
  return `${normalized > 0 ? "+" : ""}${compactNumber(normalized)}`;
}

function isOpeningBet(entry) {
  return entry.type === "raise"
    && entry.street !== "preflop"
    && Number(entry.context?.currentBet ?? 1) <= EPSILON;
}

function actionName(action, entry) {
  if (action === "raise" && isOpeningBet(entry)) return "Bet";
  return ACTION_NAMES[action];
}

function actionSummary(entry) {
  const amount = entry.type === "raise"
    ? isOpeningBet(entry)
      ? ` ${compactNumber(entry.amount)}BB`
      : ` to ${compactNumber(entry.amount)}BB`
    : entry.type === "call" ? ` ${compactNumber(entry.amount)}BB` : "";
  const context = entry.context;
  const details = context
    ? ` | 行动前 Pot ${compactNumber(context.pot)}BB · 剩余筹码 ${compactNumber(context.stack)}BB · 有效筹码 ${compactNumber(context.effectiveStack)}BB · To call ${compactNumber(context.toCall)}BB`
    : "";
  return `${entry.position} ${entry.actor}: ${actionName(entry.type, entry)}${amount}${details}`;
}

function actionEvText(row, entry) {
  let text = `${actionName(row.action, entry)} ${row.ev.toFixed(3)}BB`;
  if (Number.isFinite(row.frequency)) {
    const label = entry.review?.strategyReliable === false ? "未收敛参考" : "策略";
    text += ` · ${label} ${(row.frequency * 100).toFixed(1)}%`;
  }
  if (Number.isFinite(row.ci95)) text += ` · 95% ±${row.ci95.toFixed(3)}BB`;
  if (Number.isFinite(row.differenceCi95)) {
    text += ` · 相对最佳95% ±${row.differenceCi95.toFixed(3)}BB`;
  }
  return text;
}

function reviewLines(entry) {
  if (!entry.review) return [];
  const review = entry.review;
  const selected = review.actions.find((row) => row.action === review.chosen)
    ?? review.actions.find((row) => row.action === entry.type);
  const recommended = review.recommended
    .map((action) => actionName(action, entry))
    .join("/");
  const summary = [
    `评级 ${review.rating}`,
    `推荐 ${recommended}`,
  ];
  if (selected) summary.push(`所选 EV ${selected.ev.toFixed(3)}BB`);
  if (Number.isFinite(review.loss)) summary.push(`EV 损失 ${review.loss.toFixed(3)}BB`);
  if (Number.isFinite(review.equity)) summary.push(`范围 Equity ${(review.equity * 100).toFixed(1)}%`);
  if (Number.isFinite(review.iterations)) summary.push(`采样 ${review.iterations.toLocaleString("zh-CN")} 次`);
  if (Number.isFinite(review.trainingIterations)) summary.push(`训练 ${review.trainingIterations.toLocaleString("zh-CN")} 次`);
  if (Number.isFinite(review.trainingRequestedIterations) && review.trainingRequestedIterations !== review.trainingIterations) {
    summary.push(`训练预算 ${review.trainingRequestedIterations.toLocaleString("zh-CN")} 次（EV 收敛后剪枝 ${review.trainingPrunedIterations.toLocaleString("zh-CN")} 次）`);
  }
  if (review.adaptiveTrainingDominanceWorkersStopped > 0) {
    summary.push(`翻牌前决策剪枝 ${review.adaptiveTrainingDominanceWorkersStopped} 个训练副本`);
  }
  if (Number.isFinite(review.evaluationIterations)) summary.push(`评估 ${review.evaluationIterations.toLocaleString("zh-CN")} 次`);
  if (Number.isFinite(review.errorBudgetFraction)) {
    summary.push(`目标误差预算 ${(review.errorBudgetFraction * 100).toFixed(1)}%`);
  }
  if (review.confidenceMethod) {
    const labels = {
      "sequential-empirical-bernstein": "序贯经验 Bernstein 界",
      "pooled-fixed-sample-empirical-bernstein": "多副本等样本经验 Bernstein 界",
      "fixed-sample-normal": "固定样本正态区间",
    };
    summary.push(labels[review.confidenceMethod] ?? review.confidenceMethod);
  }
  if (Number.isFinite(review.evaluationRequestedIterations) && review.evaluationRequestedIterations !== review.evaluationIterations) {
    const reason = review.decisionEvaluationWorkersStopped > 0
      ? `翻牌前决策已确定，剪枝 ${review.evaluationPrunedIterations.toLocaleString("zh-CN")} 次`
      : "自适应提前结束";
    summary.push(`评估预算 ${review.evaluationRequestedIterations.toLocaleString("zh-CN")} 次（${reason}）`);
  }
  if (review.gpuTrainingWorkers > 0) summary.push(`GPU 训练进程 ${review.gpuTrainingWorkers} 个`);
  else if (Number.isFinite(review.workerCount)) summary.push(`并行 ${review.workerCount} 核`);
  if (Number.isFinite(review.aggregateIterations)) summary.push(`合并 ${review.aggregateIterations.toLocaleString("zh-CN")} 次`);
  if (review.cudaAvailable) {
    const trainingHands = review.gpuTrainingHandEvaluations ?? 0;
    const trainingBatches = review.gpuTrainingBatches ?? 0;
    const evaluationHands = review.gpuEvaluationHandEvaluations ?? review.gpuHandEvaluations ?? 0;
    const evaluationBatches = review.gpuEvaluationBatches ?? review.gpuBatches ?? 0;
    summary.push(`GPU ${review.computeDevice} · 训练叶节点 ${trainingHands.toLocaleString("zh-CN")} 手牌/${trainingBatches} 批 · EV评估 ${evaluationHands.toLocaleString("zh-CN")} 手牌/${evaluationBatches} 批`);
  }
  else if (review.computeDevice) summary.push(`计算设备 ${review.computeDevice}`);
  if (Number.isFinite(review.rootRegret)) summary.push(`根遗憾 ${review.rootRegret.toFixed(4)}BB`);
  if (Number.isFinite(review.evaluationUntrainedRate)) {
    summary.push(`评估未训练访问 ${(review.evaluationUntrainedRate * 100).toFixed(1)}%`);
  }
  if (Number.isFinite(review.evaluationPolicyBackoffRate)) {
    summary.push(`分层策略回退 ${(review.evaluationPolicyBackoffRate * 100).toFixed(1)}%`);
  }
  if (Number.isFinite(review.evaluationBlueprintPolicyRate)) {
    summary.push(`Blueprint 策略回退 ${(review.evaluationBlueprintPolicyRate * 100).toFixed(1)}%`);
  }
  if (Number.isFinite(review.evaluationUniformFallbackRate)) {
    summary.push(`均匀策略回退 ${(review.evaluationUniformFallbackRate * 100).toFixed(1)}%`);
  }
  if (review.blueprintExactHit) summary.push("Blueprint 精确节点命中");
  if (review.blueprintPolicySeeds > 0) summary.push(`Blueprint 热启动 ${review.blueprintPolicySeeds} 桶`);
  if (review.leafCutoffHits > 0) {
    summary.push(`叶模型截断 ${review.leafCutoffHits} 次 · 误差校准半径 ${(review.leafModelErrorCalibration ?? review.leafModelErrorBound ?? 0).toFixed(3)}BB`);
  }
  if (review.cacheHit) summary.push("精确缓存命中");
  if (review.selectionMode === "highest-ev-fallback") summary.push("Bot 模式 未收敛/最高EV兜底");
  if (Number.isFinite(review.randomRoll)) summary.push(`Random ${review.randomRoll.toFixed(6)}`);
  return [
    `  ${summary.join(" | ")}`,
    `  EV ${review.actions.map((row) => actionEvText(row, entry)).join(" / ")}`,
  ];
}

function actionLines(entry) {
  return [actionSummary(entry), ...reviewLines(entry)];
}

function playerLabel(player) {
  return `${player.position} ${player.isHero ? "Hero" : player.name}`;
}

export function buildAiHandText(game, handNumber) {
  const hero = game.players.find((player) => player.isHero);
  const startingStack = game.startingStack ?? 100;
  const lines = [
    `Texas Lab #${handNumber} | ${game.players.length}-max | ${compactNumber(startingStack)}BB | Seed ${game.seed ?? "-"}`,
    `Hero ${hero.position}: [${cardsText(hero.hole)}]`,
  ];
  let street = null;
  game.history.forEach((entry) => {
    if (entry.street !== street) {
      street = entry.street;
      lines.push("", `${STREET_NAMES[street]} [${cardsText(entry.board)}]`);
    }
    lines.push(...actionLines(entry));
  });
  lines.push("", `最终牌面 [${cardsText(game.board)}] | 最终底池 ${compactNumber(getPot(game))}BB`);

  if (game.complete) {
    lines.push("", "全部底牌（牌局结束后，仅供复盘）");
    game.players.forEach((player) => {
      lines.push(`${playerLabel(player)} [${cardsText(player.hole)}] · ${player.folded ? "已弃牌" : "未弃牌"}`);
    });
    const awards = game.result.winners.map(({ playerId, amount }) => {
      const player = game.players.find((item) => item.id === playerId);
      return `${playerLabel(player)} ${compactNumber(amount)}BB`;
    });
    lines.push("", `结果 ${game.result.reason === "showdown" ? "Showdown" : "Fold"}`);
    lines.push(`底池奖励 ${awards.join(" | ")}`);
    const netResults = game.players
      .filter((player) => Number.isFinite(player.stack))
      .map((player) => `${playerLabel(player)} ${signedNumber(player.stack - startingStack)}BB（最终 ${compactNumber(player.stack)}BB）`);
    if (netResults.length) lines.push(`净输赢 ${netResults.join(" | ")}`);
  }
  return lines.join("\n");
}

export async function copyAiHandText(game, handNumber) {
  const text = buildAiHandText(game, handNumber);
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return text;
    } catch {
      // Fall through for browsers that expose Clipboard API but deny it.
    }
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.append(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  if (!copied) throw new Error("无法访问剪贴板");
  return text;
}
