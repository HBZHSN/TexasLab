import { CARDS_BY_CODE } from "../core/cards.js";
import { currentHandName } from "../core/evaluator.js";
import { byId, selectAll, showToast } from "../ui/dom.js";
import { GameSocketClient } from "../services/game-socket.js";
import { copyAiHandText } from "./game-export.js";
import {
  POSITIONS,
  STREET_NAMES,
  act,
  createGame,
  createSeededRandom,
  createInformationSet,
  currentPlayer,
  getPot,
  legalActions,
  raiseTarget,
  verifyInformationIsolation,
} from "./game-engine.js";

const ACTION_NAMES = { check: "Check", call: "Call", raise: "Raise", fold: "Fold" };
const STATS_KEY = "texas-lab-game-stats-v1";

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatInteger(value) {
  return Number.isFinite(value) ? value.toLocaleString("zh-CN") : "—";
}

function formatElapsed(milliseconds) {
  if (!Number.isFinite(milliseconds)) return "—";
  if (milliseconds < 1000) return `${Math.max(0, milliseconds).toFixed(0)} ms`;
  return `${(milliseconds / 1000).toFixed(milliseconds < 10_000 ? 2 : 1)} 秒`;
}

export function solveLogPresentation(entry) {
  const result = entry.result ?? {};
  if (entry.status === "error") {
    return {
      tone: "error",
      source: "求解失败",
      lines: [entry.error ?? "未知错误"],
      foot: `客户端等待 ${formatElapsed(entry.elapsedMs)}`,
    };
  }
  if (entry.status !== "complete") {
    return {
      tone: "pending",
      source: entry.status === "running" ? "正在现场检查/求解" : "等待后端接收",
      lines: [`请求预算 ${formatInteger(entry.requestedIterations)} 次`],
      foot: "结果返回后将显示缓存来源和实际训练信息",
    };
  }

  const source = result.solveSource
    ?? (result.cacheHit ? "memory-cache" : result.blueprintExactHit ? "blueprint-exact" : "live-solve");
  const sourceLabels = {
    "memory-cache": "内存缓存命中",
    "shared-inflight": "复用正在求解的同节点",
    "blueprint-exact": "Blueprint 精确结果",
    "live-solve": "现场训练/求解",
  };
  const cacheSource = source !== "live-solve";
  const tone = source === "live-solve" ? "live" : "cache";
  const lines = [];
  if (source === "blueprint-exact") {
    lines.push(`本次无需训练 · 直接读取预训练节点 · 请求预算 ${formatInteger(result.iterations ?? entry.requestedIterations)} 次`);
  } else if (source === "memory-cache") {
    const underlying = result.blueprintExactHit ? " · 原始结果来自 Blueprint" : "";
    lines.push(`本次无需训练 · 读取进程内精确缓存${underlying}`);
  } else if (source === "shared-inflight") {
    lines.push("本次无需重复训练 · 等待并复用相同节点的在途结果");
  } else {
    lines.push(`现场训练 ${formatInteger(result.trainingIterations)}/${formatInteger(result.trainingRequestedIterations ?? result.trainingIterations)} 次 · EV 评估 ${formatInteger(result.evaluationIterations)} 次`);
  }
  if (Number.isFinite(result.aggregateIterations) || Number.isFinite(result.workerCount)) {
    const workers = result.gpuTrainingWorkers > 0
      ? `${result.gpuTrainingWorkers} 个 GPU 进程`
      : `${formatInteger(result.workerCount)} 个 CPU 进程`;
    lines.push(`合并预算 ${formatInteger(result.aggregateIterations)} 次 · ${workers} · ${result.computeDevice ?? "cpu"}`);
  }
  if (Number.isFinite(result.rootRegret)) {
    const reliability = result.strategyReliable === false ? "未收敛" : "策略可靠";
    lines.push(`根遗憾 ${result.rootRegret.toFixed(4)} BB · ${reliability}`);
  }
  const backEnd = formatElapsed(result.solveElapsedMs);
  const frontEnd = formatElapsed(entry.elapsedMs);
  return {
    tone,
    source: sourceLabels[source] ?? (cacheSource ? "缓存结果" : "现场求解"),
    lines,
    foot: `后端 ${backEnd} · 请求总耗时 ${frontEnd}`,
  };
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

export function cardMarkup(code, hidden = false) {
  if (hidden) return '<span class="table-card card-back"><i></i></span>';
  const card = CARDS_BY_CODE.get(code);
  const rank = card.rank === 10 ? "10" : card.text.slice(0, -1);
  return `<span class="table-card ${card.red ? "red" : ""}"><b>${rank}</b><i>${card.text.slice(-1)}</i></span>`;
}

function loadStats() {
  try {
    return JSON.parse(localStorage.getItem(STATS_KEY)) ?? { hands: 0, decisions: [] };
  } catch {
    return { hands: 0, decisions: [] };
  }
}

export function chooseMixedAction(analysis, random = Math.random) {
  const sampledRoll = random();
  if (analysis.strategyReliable === false) {
    const best = analysis.actions.reduce((current, row) => row.ev > current.ev ? row : current);
    return { action: best.action, roll: sampledRoll, mode: "highest-ev-fallback" };
  }
  let roll = sampledRoll;
  for (const row of analysis.actions) {
    roll -= row.frequency;
    if (roll <= 0) return { action: row.action, roll: sampledRoll, mode: "equilibrium-mix" };
  }
  return { action: analysis.actions.at(-1).action, roll: sampledRoll, mode: "equilibrium-mix" };
}

function evaluateChoice(analysis, chosen) {
  const selected = analysis.actions.find((row) => row.action === chosen);
  const loss = Number.isFinite(selected.evLossEstimate)
    ? selected.evLossEstimate
    : Math.max(0, analysis.bestEV - selected.ev);
  const best = analysis.actions.reduce((current, row) => row.ev > current.ev ? row : current);
  const differenceCi95 = Number.isFinite(selected.differenceCi95)
    ? selected.differenceCi95
    : 1.96 * Math.hypot(selected.stderr ?? 0, best.stderr ?? 0);
  const unresolved = analysis.strategyReliable === false
    || (analysis.rootRegret ?? 0) > 0.05
    || (analysis.evaluationUntrainedRate ?? 0) > 0.01;
  const rating = unresolved
    ? "未收敛"
    : loss <= 0.02 ? "优秀" : loss <= differenceCi95 ? "合理" : loss <= 0.3 ? "轻微错误" : "明显错误";
  return { ...analysis, chosen, loss, rating, differenceCi95, unresolved };
}

function decisionContext(info) {
  const activeOpponentStacks = info.players
    .filter((player) => !player.folded && player.id !== info.hero.id)
    .map((player) => player.stack);
  const deepestOpponentStack = Math.max(0, ...activeOpponentStacks);
  return {
    pot: info.pot,
    board: info.board,
    stack: info.hero.stack,
    effectiveStack: Math.min(info.hero.stack, deepestOpponentStack),
    currentBet: info.currentBet,
    toCall: Math.max(0, info.currentBet - info.hero.streetCommitted),
    raiseTo: info.raiseTo,
    legalActions: info.legalActions,
  };
}

function spotType(info) {
  if (info.street !== "preflop") return `${STREET_NAMES[info.street]} · ${currentHandName(info.hero.hole.concat(info.board))}`;
  const ranks = info.hero.hole.map((code) => 2 + Math.floor(code / 4));
  if (ranks[0] === ranks[1]) return "翻牌前 · 口袋对子";
  if (ranks.every((rank) => rank >= 10)) return "翻牌前 · Broadway";
  if (info.hero.hole[0] % 4 === info.hero.hole[1] % 4) return "翻牌前 · 同花非对子";
  return "翻牌前 · 非同花非对子";
}

export class GameUI {
  constructor() {
    this.client = new GameSocketClient();
    this.game = null;
    this.busy = false;
    this.solvingPlayerId = null;
    this.solveIterations = null;
    this.revealBotCards = false;
    this.handNumber = loadStats().hands + 1;
    this.solverLogs = [];
    this.solveLogSequence = 0;
    this.solverLogCollapsed = false;
  }

  initialize() {
    byId("gameBtn").onclick = () => this.enter();
    byId("gameExitBtn").onclick = () => document.querySelector('[data-tab="hands"]').click();
    byId("startGameBtn").onclick = () => this.startGame();
    byId("newHandBtn").onclick = () => this.startHand();
    byId("revealBotCardsBtn").onclick = () => {
      if (!this.game) return;
      this.revealBotCards = !this.revealBotCards;
      this.renderSeats();
      this.renderRevealButton();
    };
    byId("loadSeedBtn").onclick = () => this.loadSeed();
    byId("handSeedInput").onkeydown = (event) => {
      if (event.key === "Enter") this.loadSeed();
    };
    byId("exportGameHistoryBtn").onclick = async () => {
      if (!this.game?.history.length) {
        showToast("当前没有可导出的行动记录");
        return;
      }
      try {
        await copyAiHandText(this.game, this.handNumber);
        showToast("行动记录已复制到剪贴板");
      } catch (error) {
        showToast(`复制失败：${error.message}`);
      }
    };
    byId("resetStatsBtn").onclick = () => {
      localStorage.removeItem(STATS_KEY);
      this.handNumber = 1;
      this.renderStats();
      showToast("训练统计已重置");
    };
    byId("solverLogToggle").onclick = () => {
      this.solverLogCollapsed = !this.solverLogCollapsed;
      this.renderSolverLog();
    };
    selectAll("#gameActions [data-action]").forEach((button) => {
      button.onclick = () => this.heroAction(button.dataset.action);
    });
    this.renderStats();
    this.renderSolverLog();
  }

  enter() {
    selectAll(".tab-btn").forEach((button) => button.classList.remove("active"));
    selectAll(".tab").forEach((tab) => tab.classList.remove("active"));
    byId("tab-game").classList.add("active");
    byId("gameBtn").classList.add("active");
    byId("calculatorActions").classList.add("hidden");
  }

  leave() {
    byId("gameBtn").classList.remove("active");
  }

  startGame() {
    const raw = byId("handSeedInput").value.trim();
    if (!raw) {
      this.startHand();
      return;
    }
    if (!/^\d+$/.test(raw) || Number(raw) > 0xffffffff) {
      showToast("种子必须是 0 到 4294967295 之间的整数");
      return;
    }
    this.startHand(Number(raw));
  }

  startHand(seed) {
    this.handToken = (this.handToken ?? 0) + 1;
    this.handNumber = loadStats().hands + 1;
    this.game = createGame(seed === undefined ? {} : { seed });
    this.botRandom = createSeededRandom(this.game.seed ^ 0x9e3779b9);
    this.busy = false;
    this.solvingPlayerId = null;
    this.solveIterations = null;
    byId("startGameBtn").classList.add("hidden");
    byId("newHandBtn").classList.remove("hidden");
    byId("revealBotCardsBtn").disabled = false;
    byId("handReview").classList.add("hidden");
    this.lockAnalysis();
    this.render();
    this.advanceBots();
  }

  loadSeed() {
    const raw = byId("handSeedInput").value.trim();
    if (!/^\d+$/.test(raw) || Number(raw) > 0xffffffff) {
      showToast("种子必须是 0 到 4294967295 之间的整数");
      return;
    }
    this.startHand(Number(raw));
    showToast(`已载入种子 ${this.game.seed}`);
  }

  informationFor(player) {
    const info = createInformationSet(this.game, player.id);
    if (!verifyInformationIsolation(info)) throw new Error("信息隔离检查失败");
    return info;
  }

  async solve(info, quality) {
    const started = performance.now();
    const entry = {
      id: ++this.solveLogSequence,
      status: "pending",
      requestedIterations: 20_000,
      quality,
      position: info.hero.position,
      street: info.street,
      timestamp: new Date(),
    };
    this.solverLogs.push(entry);
    if (this.solverLogs.length > 80) this.solverLogs.shift();
    this.renderSolverLog();
    try {
      const result = await this.client.solve(info, {
        quality,
        iterations: entry.requestedIterations,
        onStarted: (iterations) => {
          entry.status = "running";
          entry.requestedIterations = iterations;
          this.solveIterations = iterations;
          this.renderSeats();
          this.renderSolverLog();
        },
      });
      entry.status = "complete";
      entry.elapsedMs = performance.now() - started;
      entry.result = result;
      this.renderSolverLog();
      return result;
    } catch (error) {
      entry.status = "error";
      entry.elapsedMs = performance.now() - started;
      entry.error = error.message;
      this.renderSolverLog();
      throw error;
    }
  }

  renderSolverLog() {
    const panel = byId("solverLogPanel");
    const toggle = byId("solverLogToggle");
    const list = byId("solverLog");
    panel.classList.toggle("collapsed", this.solverLogCollapsed);
    toggle.textContent = this.solverLogCollapsed ? "展开" : "收起";
    toggle.setAttribute("aria-expanded", String(!this.solverLogCollapsed));
    if (!this.solverLogs.length) {
      list.innerHTML = '<div class="solver-log-empty">开始牌局后显示缓存命中、现场训练和性能信息</div>';
      return;
    }
    list.innerHTML = [...this.solverLogs].reverse().map((entry) => {
      const view = solveLogPresentation(entry);
      const actor = entry.quality === "hero" ? "Hero" : "Bot";
      const street = STREET_NAMES[entry.street] ?? entry.street;
      const time = entry.timestamp.toLocaleTimeString("zh-CN", { hour12: false });
      return `<article class="solver-log-entry ${view.tone}">
        <div class="solver-log-entry-head"><b>${actor} · ${escapeHtml(entry.position)} · ${escapeHtml(street)}</b><time>${escapeHtml(time)}</time></div>
        <span class="solver-log-source">${escapeHtml(view.source)}</span>
        <div class="solver-log-detail">${view.lines.map((line) => `<span>${escapeHtml(line)}</span>`).join("")}</div>
        <div class="solver-log-foot">${escapeHtml(view.foot)}</div>
      </article>`;
    }).join("");
    list.scrollTop = 0;
  }

  async advanceBots() {
    if (!this.game || this.game.complete) {
      if (this.game?.complete) this.finishHand();
      return;
    }
    const actor = currentPlayer(this.game);
    if (!actor || actor.isHero) {
      this.busy = false;
      this.render();
      return;
    }
    this.busy = true;
    this.solvingPlayerId = actor.id;
    this.solveIterations = null;
    const gameAtRequest = this.game;
    this.render();
    try {
      const info = this.informationFor(actor);
      const analysis = await this.solve(info, "bot");
      if (this.game !== gameAtRequest || currentPlayer(this.game)?.id !== actor.id) return;
      this.solvingPlayerId = null;
      this.solveIterations = null;
      const selection = chooseMixedAction(analysis, this.botRandom);
      const review = evaluateChoice(analysis, selection.action);
      review.randomRoll = selection.roll;
      review.selectionMode = selection.mode;
      act(this.game, actor.id, selection.action);
      const historyEntry = this.game.history.at(-1);
      historyEntry.review = review;
      historyEntry.context = decisionContext(info);
      this.render();
      await delay(220);
      this.advanceBots();
    } catch (error) {
      if (this.game !== gameAtRequest) return;
      this.busy = false;
      this.solvingPlayerId = null;
      this.solveIterations = null;
      this.render();
      showToast(error.message);
    }
  }

  async heroAction(action) {
    const actor = currentPlayer(this.game);
    if (this.busy || !actor?.isHero || !legalActions(this.game).includes(action)) return;
    const info = this.informationFor(actor);
    const gameAtRequest = this.game;
    this.busy = true;
    this.solvingPlayerId = actor.id;
    this.solveIterations = null;
    this.showSolving();
    this.renderSeats();
    this.renderActions();
    try {
      const analysis = await this.solve(info, "hero");
      if (this.game !== gameAtRequest || currentPlayer(this.game)?.id !== actor.id) return;
      this.solvingPlayerId = null;
      this.solveIterations = null;
      const review = evaluateChoice(analysis, action);
      act(this.game, actor.id, action);
      const historyEntry = this.game.history.at(-1);
      historyEntry.review = review;
      historyEntry.heroCards = info.hero.hole.slice();
      historyEntry.context = decisionContext(info);
      this.recordDecision(review, info);
      this.busy = false;
      this.showAnalysis(review);
      this.render();
      if (this.game.complete) this.finishHand();
      else this.advanceBots();
    } catch (error) {
      if (this.game !== gameAtRequest) return;
      this.busy = false;
      this.solvingPlayerId = null;
      this.solveIterations = null;
      this.lockAnalysis();
      this.render();
      showToast(error.message);
    }
  }

  finishHand() {
    if (!this.game.handCounted) {
      const stats = loadStats();
      stats.hands += 1;
      localStorage.setItem(STATS_KEY, JSON.stringify(stats));
      this.game.handCounted = true;
    }
    this.busy = false;
    this.solvingPlayerId = null;
    this.solveIterations = null;
    this.render();
    this.renderReview();
    this.renderStats();
  }

  recordDecision(review, info) {
    const stats = loadStats();
    stats.decisions.push({
      street: info.street,
      position: info.hero.position,
      action: review.chosen,
      loss: review.loss,
      rating: review.rating,
      cards: info.hero.hole,
      board: info.board,
      pot: info.pot,
      spotType: spotType(info),
    });
    localStorage.setItem(STATS_KEY, JSON.stringify(stats));
    this.renderStats();
  }

  render() {
    if (!this.game) return;
    byId("gamePot").textContent = `${getPot(this.game)} BB`;
    byId("gameStreet").textContent = STREET_NAMES[this.game.street];
    byId("gameCurrentBet").textContent = `${this.game.currentBet.toFixed(2).replace(/\.00$/, "")} BB`;
    byId("gameRaiseCount").textContent = `${this.game.aggressiveActions} / 3`;
    byId("handNumber").textContent = `#${this.handNumber}`;
    byId("handSeedInput").value = String(this.game.seed);
    byId("gameBoard").innerHTML = Array.from({ length: 5 }, (_, index) =>
      this.game.board[index] === undefined
        ? '<span class="table-card card-placeholder"></span>'
        : cardMarkup(this.game.board[index]),
    ).join("");
    this.renderSeats();
    this.renderRevealButton();
    this.renderActions();
    this.renderHistory();
  }

  renderSeats() {
    const actor = currentPlayer(this.game);
    byId("gameSeats").innerHTML = this.game.players.map((player) => {
      const reveal = player.isHero || this.revealBotCards || (this.game.complete && this.game.result?.reason === "showdown" && !player.folded);
      const thinking = this.solvingPlayerId === player.id;
      const progressLabel = player.isHero ? "正在计算 EV" : "正在思考";
      const progressDetail = this.solveIterations ? ` · 上限 ${this.solveIterations.toLocaleString("zh-CN")} 次` : "";
      return `<article class="game-seat seat-${player.seatIndex} ${player.isHero ? "hero-seat" : ""} ${player.folded ? "folded" : ""} ${player.streetCommitted ? "has-bet" : ""} ${actor?.id === player.id ? "acting" : ""} ${thinking ? "thinking" : ""}">
        <div class="seat-cards">${player.hole.map((code) => cardMarkup(code, !reveal)).join("")}</div>
        <div class="seat-info"><div><b>${player.isHero ? "Hero" : player.name}</b><span>${player.position}</span></div><strong>${player.stack.toFixed(1)} BB</strong></div>
        ${player.position === "BTN" ? '<span class="dealer-button">D</span>' : ""}
        ${player.streetCommitted ? `<small class="seat-bet">${player.streetCommitted.toFixed(1)} BB</small>` : ""}
        ${thinking ? `<div class="seat-progress" role="progressbar" aria-label="${progressLabel}"><span>${progressLabel}${progressDetail}</span><i></i></div>` : ""}
      </article>`;
    }).join("");
  }

  renderRevealButton() {
    const button = byId("revealBotCardsBtn");
    button.textContent = this.revealBotCards ? "隐藏 Bot 手牌" : "查看 Bot 手牌";
    button.setAttribute("aria-pressed", String(this.revealBotCards));
  }

  renderActions() {
    const actor = currentPlayer(this.game);
    const canChoose = actor?.isHero && !this.busy && !this.game.complete;
    const legal = canChoose ? legalActions(this.game) : [];
    byId("decisionTitle").textContent = this.game.complete
      ? "本手牌已结束"
      : actor?.isHero ? `${actor.position} · 请选择行动` : `${actor?.position ?? ""} · 电脑思考中`;
    selectAll("#gameActions [data-action]").forEach((button) => {
      const action = button.dataset.action;
      button.classList.toggle("hidden", !legal.includes(action));
      button.disabled = !canChoose;
      if (action === "call" && actor) {
        button.textContent = `Call ${(this.game.currentBet - actor.streetCommitted).toFixed(1)}BB`;
      } else if (action === "raise" && actor) {
        button.textContent = `Raise to ${raiseTarget(this.game, actor)}BB`;
      } else button.textContent = ACTION_NAMES[action];
    });
    byId("decisionWait").textContent = actor?.isHero && this.busy ? "正在计算所选行动的 EV…" : "电脑玩家行动中…";
    byId("decisionWait").classList.toggle("hidden", Boolean(canChoose) || this.game.complete);
  }

  renderHistory() {
    if (!this.game.history.length) {
      byId("gameHistory").innerHTML = '<div class="history-empty">牌局刚刚开始</div>';
      return;
    }
    let street = null;
    byId("gameHistory").innerHTML = this.game.history.map((entry) => {
      const heading = entry.street !== street ? `<div class="history-street">${STREET_NAMES[entry.street]}</div>` : "";
      street = entry.street;
      const amount = entry.type === "raise" ? ` ${entry.amount.toFixed(1)}BB` : entry.type === "call" ? ` ${entry.amount.toFixed(1)}BB` : "";
      const evTitle = entry.review
        ? `EV ${entry.review.actions.map((row) => `${ACTION_NAMES[row.action]} ${row.ev.toFixed(3)}`).join(" / ")}`
        : "";
      return `${heading}<div class="history-row ${entry.actor === "Hero" ? "hero-history" : ""}"><span>${entry.actor} <small>${entry.position}</small></span><b>${ACTION_NAMES[entry.type]}${amount}</b>${entry.review ? `<em title="${evTitle}">${entry.review.rating}</em>` : ""}</div>`;
    }).join("");
    byId("gameHistory").scrollTop = byId("gameHistory").scrollHeight;
  }

  lockAnalysis() {
    byId("gameAnalysis").classList.add("locked");
    byId("analysisLock").classList.remove("hidden");
    byId("analysisLock").querySelector("b").textContent = "策略分析已锁定";
    byId("analysisLock").querySelector("small").textContent = "完成本次行动后显示策略频率、EV 与评分";
    byId("analysisContent").classList.add("hidden");
  }

  showSolving() {
    byId("gameAnalysis").classList.remove("locked");
    byId("analysisLock").classList.remove("hidden");
    byId("analysisLock").querySelector("b").textContent = "正在计算策略分析";
    byId("analysisLock").querySelector("small").textContent = "计算进度显示在 Hero 席位下方";
    byId("analysisContent").classList.add("hidden");
  }

  showAnalysis(review) {
    byId("gameAnalysis").classList.remove("locked");
    byId("analysisLock").classList.add("hidden");
    byId("analysisContent").classList.remove("hidden");
    byId("analysisVerdict").textContent = `${review.rating} · 你选择了 ${ACTION_NAMES[review.chosen]}`;
    byId("analysisLoss").textContent = `EV 损失 ${review.loss.toFixed(3)} BB`;
    byId("analysisSummary").innerHTML = `<span>推荐 <b>${review.recommended.map((action) => ACTION_NAMES[action]).join(" / ")}</b></span><span>范围 Equity <b>${(review.equity * 100).toFixed(1)}%</b></span>`;
    const explanation = review.explanation;
    const bestName = ACTION_NAMES[explanation.bestAction];
    const chosenName = ACTION_NAMES[review.chosen];
    const explanationTitle = explanation.bestAction === review.chosen
      ? `为什么 ${chosenName} 是合理选择`
      : `为什么应当 ${bestName}，而不是 ${chosenName}`;
    byId("analysisExplanation").innerHTML = `<h4>${explanationTitle}</h4>
      <p>${explanation.summary}</p>
      <div>${explanation.reasons.map((reason) => `<p><b>${reason.label}</b>${reason.text}</p>`).join("")}</div>
      <small>${explanation.caveat}</small>`;
    const actionOrder = ["raise", "fold", "call", "check"];
    byId("analysisActions").innerHTML = [...review.actions].sort((a, b) => actionOrder.indexOf(a.action) - actionOrder.indexOf(b.action)).map((row) => `<div class="ev-row ${row.action === review.chosen ? "chosen" : ""} ${review.recommended.includes(row.action) ? "recommended" : ""}">
      <div><b>${ACTION_NAMES[row.action]}</b><small>${(row.frequency * 100).toFixed(1)}% ${review.strategyReliable === false ? "未收敛参考频率" : "均衡策略频率"}</small></div>
      <strong>${row.ev.toFixed(3)} BB <small>±${row.ci95.toFixed(3)}</small></strong>
    </div>`).join("");
    const workerLabel = review.gpuTrainingWorkers > 0
      ? `${review.gpuTrainingWorkers} 个 GPU 训练进程`
      : `${review.workerCount} 核并行`;
    const coverage = Number.isFinite(review.evaluationUntrainedRate)
      ? ` · 未训练访问 ${(review.evaluationUntrainedRate * 100).toFixed(1)}%`
      : "";
    const fallback = Number.isFinite(review.evaluationUniformFallbackRate)
      ? ` · Blueprint 回退 ${((review.evaluationBlueprintPolicyRate ?? 0) * 100).toFixed(1)}% · 分层回退 ${((review.evaluationPolicyBackoffRate ?? 0) * 100).toFixed(1)}% · 均匀回退 ${(review.evaluationUniformFallbackRate * 100).toFixed(1)}%`
      : "";
    const trainingProgress = Number.isFinite(review.trainingRequestedIterations)
      ? ` · 实际训练 ${review.trainingIterations.toLocaleString("zh-CN")}/${review.trainingRequestedIterations.toLocaleString("zh-CN")}`
      : "";
    const budget = Number.isFinite(review.workerCount)
      ? `${workerLabel} · 每进程训练上限 ${review.trainingPerWorker.toLocaleString("zh-CN")}${trainingProgress} · 每进程评估预算 ${review.evaluationPerWorker.toLocaleString("zh-CN")} · 实际评估 ${review.evaluationIterations.toLocaleString("zh-CN")}/${(review.evaluationRequestedIterations ?? review.evaluationIterations).toLocaleString("zh-CN")} · 合并 ${review.aggregateIterations.toLocaleString("zh-CN")} · 根遗憾 ${review.rootRegret.toFixed(4)}BB${coverage}${fallback}`
      : Number.isFinite(review.trainingIterations)
      ? `训练 ${review.trainingIterations.toLocaleString("zh-CN")} · 评估 ${review.evaluationIterations.toLocaleString("zh-CN")} · 根遗憾 ${review.rootRegret.toFixed(4)}BB${coverage}${fallback}`
      : `${review.iterations.toLocaleString("zh-CN")} 次采样`;
    const gpuTrainingHands = review.gpuTrainingHandEvaluations ?? 0;
    const gpuTrainingBatches = review.gpuTrainingBatches ?? 0;
    const gpuEvaluationHands = review.gpuEvaluationHandEvaluations ?? review.gpuHandEvaluations ?? 0;
    const gpuEvaluationBatches = review.gpuEvaluationBatches ?? review.gpuBatches ?? 0;
    const compute = review.cudaAvailable
      ? `GPU ${review.computeDevice} · 训练叶节点 ${gpuTrainingHands.toLocaleString("zh-CN")} 手牌 / ${gpuTrainingBatches} 批 · EV评估 ${gpuEvaluationHands.toLocaleString("zh-CN")} 手牌 / ${gpuEvaluationBatches} 批`
      : `计算设备 ${review.computeDevice ?? "cpu"}`;
    const cache = review.cacheHit ? " · 精确缓存命中" : "";
    const acceleration = [
      review.blueprintExactHit ? "Blueprint 精确命中" : "",
      review.blueprintPolicySeeds > 0 ? `Blueprint 热启动 ${review.blueprintPolicySeeds.toLocaleString("zh-CN")} 桶` : "",
      review.leafCutoffHits > 0 ? `叶模型截断 ${review.leafCutoffHits.toLocaleString("zh-CN")} 次 · 误差校准半径 ${(review.leafModelErrorCalibration ?? review.leafModelErrorBound ?? 0).toFixed(3)}BB` : "",
      review.adaptiveTrainingDominanceWorkersStopped > 0 ? `翻牌前决策剪枝 ${review.trainingPrunedIterations.toLocaleString("zh-CN")} 次训练` : "",
      review.adaptiveTrainingWorkersStopped > (review.adaptiveTrainingDominanceWorkersStopped ?? 0) ? `EV 收敛剪枝 ${review.adaptiveTrainingWorkersStopped - (review.adaptiveTrainingDominanceWorkersStopped ?? 0)} 个副本` : "",
      review.decisionEvaluationWorkersStopped > 0 ? `决策已定，剪枝 ${review.evaluationPrunedIterations.toLocaleString("zh-CN")} 次评估` : "",
      review.adaptiveEvaluationWorkersStopped > 0 ? `自适应提前停止 ${review.adaptiveEvaluationWorkersStopped} 个评估分片` : "",
    ].filter(Boolean).join(" · ");
    const accelerationText = acceleration ? ` · ${acceleration}` : "";
    const precision = Number.isFinite(review.errorBudgetFraction)
      ? ` · 目标误差预算 ${(review.errorBudgetFraction * 100).toFixed(1)}%`
      : "";
    const confidenceLabels = {
      "sequential-empirical-bernstein": "序贯经验 Bernstein 界",
      "pooled-fixed-sample-empirical-bernstein": "多副本等样本经验 Bernstein 界",
      "fixed-sample-normal": "固定样本正态区间",
    };
    const confidence = review.confidenceMethod
      ? ` · ${confidenceLabels[review.confidenceMethod] ?? review.confidenceMethod}`
      : "";
    byId("analysisMethod").textContent = `${review.model} · ${review.method} · ${compute} · ${budget}${cache}${accelerationText}${precision}${confidence} · EV 置信半径包含 Monte Carlo 采样误差与已启用模型误差`;
  }

  renderStats() {
    const stats = loadStats();
    const losses = stats.decisions.map((decision) => decision.loss);
    const totalLoss = losses.reduce((sum, loss) => sum + loss, 0);
    byId("statHands").textContent = stats.hands;
    byId("statDecisions").textContent = stats.decisions.length;
    byId("statAverage").textContent = `${(totalLoss / Math.max(1, losses.length)).toFixed(3)} BB`;
    byId("statPer100").textContent = `${(totalLoss / Math.max(1, stats.hands) * 100).toFixed(2)} BB`;
    byId("streetStats").innerHTML = Object.entries(STREET_NAMES).map(([street, name]) => {
      const rows = stats.decisions.filter((decision) => decision.street === street);
      const average = rows.reduce((sum, row) => sum + row.loss, 0) / Math.max(1, rows.length);
      return `<div><span>${name}</span><i><b style="width:${Math.min(100, average * 180)}%"></b></i><strong>${average.toFixed(3)}</strong></div>`;
    }).join("");
    const actionCounts = Object.keys(ACTION_NAMES).map((action) => {
      const count = stats.decisions.filter((decision) => decision.action === action).length;
      return `${ACTION_NAMES[action]} ${stats.decisions.length ? (count / stats.decisions.length * 100).toFixed(0) : 0}%`;
    }).join(" · ");
    const positionAverages = POSITIONS.map((position) => {
      const rows = stats.decisions.filter((decision) => decision.position === position);
      const average = rows.reduce((sum, row) => sum + row.loss, 0) / Math.max(1, rows.length);
      return `${position} ${average.toFixed(2)}`;
    }).join(" · ");
    const maximum = stats.decisions.reduce((worst, row) => !worst || row.loss > worst.loss ? row : worst, null);
    const errorGroups = new Map();
    stats.decisions.filter((row) => row.loss > 0.1).forEach((row) => {
      errorGroups.set(row.spotType, (errorGroups.get(row.spotType) ?? 0) + 1);
    });
    const commonError = [...errorGroups].sort((left, right) => right[1] - left[1])[0];
    byId("statsBreakdown").innerHTML = `<div><span>行动频率</span><b>${actionCounts}</b></div>
      <div><span>各位置平均损失</span><b>${positionAverages}</b></div>
      <div><span>最大 EV 损失</span><b>${maximum ? `${maximum.loss.toFixed(3)}BB · ${maximum.position} ${ACTION_NAMES[maximum.action]}` : "—"}</b></div>
      <div><span>高频错误牌局</span><b>${commonError ? `${commonError[0]} · ${commonError[1]} 次` : "尚无样本"}</b></div>`;
  }

  renderReview() {
    const result = this.game.result;
    const winnerText = result.winners.map(({ playerId, amount }) => {
      const player = this.game.players.find((item) => item.id === playerId);
      return `${player.isHero ? "Hero" : player.name} +${amount.toFixed(1)}BB`;
    }).join(" · ");
    byId("reviewResult").textContent = `${result.reason === "showdown" ? "摊牌" : "其他玩家弃牌"} · ${winnerText}`;
    byId("reviewTimeline").innerHTML = this.game.history.map((entry) => `<div class="review-row">
      <span>${STREET_NAMES[entry.street]}</span><b>${entry.actor} ${ACTION_NAMES[entry.type]}${entry.amount ? ` ${entry.amount.toFixed(1)}BB` : ""}</b>
      ${entry.review ? `<em>${entry.review.rating} · 损失 ${entry.review.loss.toFixed(3)}BB</em>` : ""}
      ${entry.review ? `<div class="review-decision">推荐 ${entry.review.recommended.map((action) => ACTION_NAMES[action]).join("/")} · ${entry.review.actions.map((row) => `${ACTION_NAMES[row.action]} ${row.ev.toFixed(3)}BB (${(row.frequency * 100).toFixed(0)}%)`).join(" · ")}<p>${entry.review.explanation.summary}</p></div>` : ""}
    </div>`).join("");
    byId("handReview").classList.remove("hidden");
    byId("handReview").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}
