import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const DEFAULT_CHROME_PATHS = [
  process.env.CHROME_PATH,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
].filter(Boolean);

const chromePath = DEFAULT_CHROME_PATHS.find(existsSync);
if (!chromePath) {
  throw new Error("未找到 Chrome/Edge；可通过 CHROME_PATH 指定浏览器路径");
}

const appUrl = process.argv[2] ?? "http://0.0.0.0:8000/";
const debugPort = 9223;
const profileDirectory = await mkdtemp(join(tmpdir(), "texas-lab-chrome-"));
const chromeArguments = [
  "--headless=new",
  "--disable-gpu",
  "--no-first-run",
  "--no-default-browser-check",
  `--remote-debugging-port=${debugPort}`,
  `--user-data-dir=${profileDirectory}`,
  appUrl,
];
// Some isolated Windows CI sandboxes deny creation of Chrome's own sandboxed
// GPU process (0xC0000022). Keep Chrome's sandbox enabled by default and make
// the weaker mode an explicit opt-in for an already isolated smoke runner.
if (process.env.TEXAS_LAB_CHROME_NO_SANDBOX === "1") {
  chromeArguments.splice(2, 0, "--no-sandbox");
}
const browser = spawn(
  chromePath,
  chromeArguments,
  { stdio: "ignore" },
);

const delay = (milliseconds) =>
  new Promise((resolve) => setTimeout(resolve, milliseconds));

async function findPage() {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    try {
      const pages = await fetch(`http://0.0.0.0:${debugPort}/json/list`).then((response) =>
        response.json(),
      );
      const page = pages.find((entry) => entry.type === "page");
      if (page) return page;
    } catch {
      // Browser startup can take a few polling intervals.
    }
    await delay(100);
  }
  throw new Error("无法连接 Chrome DevTools Protocol");
}

function connect(webSocketUrl) {
  const socket = new WebSocket(webSocketUrl);
  const pending = new Map();
  let commandId = 0;

  socket.addEventListener("message", ({ data }) => {
    const message = JSON.parse(data);
    const request = pending.get(message.id);
    if (!request) return;
    pending.delete(message.id);
    if (message.error) request.reject(new Error(message.error.message));
    else request.resolve(message.result);
  });

  return new Promise((resolve, reject) => {
    socket.addEventListener("error", reject, { once: true });
    socket.addEventListener(
      "open",
      () =>
        resolve({
          close: () => socket.close(),
          send(method, params = {}) {
            commandId += 1;
            return new Promise((commandResolve, commandReject) => {
              pending.set(commandId, {
                resolve: commandResolve,
                reject: commandReject,
              });
              socket.send(JSON.stringify({ id: commandId, method, params }));
            });
          },
        }),
      { once: true },
    );
  });
}

async function evaluate(client, expression) {
  const result = await client.send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.text || "浏览器表达式执行失败");
  }
  return result.result.value;
}

let client;
try {
  const page = await findPage();
  client = await connect(page.webSocketDebuggerUrl);
  await client.send("Runtime.enable");

  for (let attempt = 0; attempt < 50; attempt += 1) {
    const ready = await evaluate(
      client,
      'document.readyState === "complete" && document.querySelectorAll(".deck-card").length === 52',
    );
    if (ready) break;
    await delay(100);
  }

  await client.send("Emulation.setDeviceMetricsOverride", {
    width: 1440,
    height: 1000,
    deviceScaleFactor: 1,
    mobile: false,
  });

  const navigation = await evaluate(client, `({
    tabOrder: [...document.querySelectorAll(".tab-btn")].map((button) => button.dataset.tab).join(","),
    activeTab: document.querySelector(".tab-btn.active").dataset.tab,
    activePanel: document.querySelector(".tab.active").id,
    actionsHidden: getComputedStyle(document.getElementById("calculatorActions")).display === "none",
  })`);
  if (
    navigation.tabOrder !== "hands,common,calculator" ||
    navigation.activeTab !== "hands" ||
    navigation.activePanel !== "tab-hands" ||
    !navigation.actionsHidden
  ) {
    throw new Error(`顶部导航异常：${JSON.stringify(navigation)}`);
  }

  await evaluate(client, 'document.querySelector("[data-tab=hands]").click()');

  const initial = await evaluate(client, `({
    deckCards: document.querySelectorAll(".deck-card").length,
    handPickerCards: document.querySelectorAll("#handCardPicker .hand-picker-card").length,
    handPickerSuits: document.querySelectorAll("#handCardPicker .hand-picker-suit").length,
    handRows: document.querySelectorAll("#handsBody tr").length,
    handCells: document.querySelectorAll("#handsBody .hand-matrix-cell").length,
    columnRanks: [...document.querySelectorAll("#handsHead .matrix-axis:not(.matrix-corner)")].map((cell) => cell.textContent).join(""),
    rowRanks: [...document.querySelectorAll("#handsBody .matrix-row-axis")].map((cell) => cell.textContent).join(""),
    cornerWidth: document.querySelector("#handsHead .matrix-corner").getBoundingClientRect().width,
    aceWidth: document.querySelector("#handsHead .matrix-axis:not(.matrix-corner)").getBoundingClientRect().width,
    aaBackground: getComputedStyle(document.querySelector("#handsBody .hand-matrix-cell")).backgroundColor,
    lowHandBackground: getComputedStyle(document.querySelector("#handsBody tr:last-child .hand-matrix-cell:nth-of-type(8)")).backgroundColor,
    equityTiers: [...new Set([...document.querySelectorAll("#handsBody .hand-matrix-cell")].map((cell) => cell.dataset.equityTier))].sort().join(""),
    legendTiers: document.querySelectorAll(".heat-legend i").length,
    sixPlayerLabel: document.querySelector('#handPlayersChoices [data-value="6"]').textContent,
    pairOutlineWidth: getComputedStyle(document.querySelector("#handsBody .hand-pair")).outlineWidth,
    pairNameSize: getComputedStyle(document.querySelector("#handsBody .hand-pair .matrix-hand-name")).fontSize,
    matrixVerticalScroll: document.querySelector(".hands-table-wrap").scrollHeight > document.querySelector(".hands-table-wrap").clientHeight,
    bodyOverflow: document.documentElement.scrollWidth > window.innerWidth
  })`);
  if (
    initial.deckCards !== 52 ||
    initial.handPickerCards !== 52 ||
    initial.handPickerSuits !== 4 ||
    initial.handRows !== 13 ||
    initial.handCells !== 169 ||
    initial.columnRanks !== "AKQJT98765432" ||
    initial.rowRanks !== "AKQJT98765432" ||
    Math.abs(initial.cornerWidth - initial.aceWidth) > 1 ||
    initial.aaBackground === initial.lowHandBackground ||
    initial.equityTiers !== "012345" ||
    initial.legendTiers !== 6 ||
    initial.sixPlayerLabel !== "6" ||
    initial.pairOutlineWidth !== "2px" ||
    initial.pairNameSize !== "15px" ||
    initial.matrixVerticalScroll ||
    initial.bodyOverflow
  ) {
    throw new Error(`初始渲染异常：${JSON.stringify(initial)}`);
  }

  const handSelection = await evaluate(client, `(() => {
    document.querySelector('#handCardPicker [data-card="48"]').click();
    document.querySelector('#handCardPicker [data-card="44"]').click();
    const selectedCell = document.querySelector('#handsBody .selected-hand');
    return {
      selectedCards: document.querySelectorAll('#handCardPicker .hand-picker-card.selected').length,
      count: document.getElementById('handPickerCount').textContent,
      status: document.getElementById('handPickerStatus').textContent,
      handKey: selectedCell?.dataset.handKey ?? '',
      current: selectedCell?.getAttribute('aria-current') ?? '',
    };
  })()`);
  if (
    handSelection.selectedCards !== 2 ||
    handSelection.count !== "2 / 2" ||
    !handSelection.status.includes("A♠ K♠") ||
    handSelection.handKey !== "AKs" ||
    handSelection.current !== "true"
  ) {
    throw new Error(`起手牌选择异常：${JSON.stringify(handSelection)}`);
  }

  const playerSwitch = await evaluate(client, `(() => {
    document.querySelector('#handPlayersChoices [data-value="2"]').click();
    return {
      active: document.querySelector('#handPlayersChoices .active').dataset.value,
      aa: document.querySelector('#handsBody .hand-matrix-cell .matrix-equity').textContent,
      selectedHand: document.querySelector('#handsBody .selected-hand')?.dataset.handKey ?? '',
    };
  })()`);
  if (playerSwitch.active !== "2" || playerSwitch.aa !== "85.2%" || playerSwitch.selectedHand !== "AKs") {
    throw new Error(`人数切换异常：${JSON.stringify(playerSwitch)}`);
  }

  await evaluate(client, 'document.querySelector("[data-tab=common]").click()');
  const commonProbabilityPage = await evaluate(client, `(() => {
    document.querySelector('[data-group="straight"]').click();
    const range = document.getElementById('commonOutsRange');
    range.value = '15';
    range.dispatchEvent(new Event('input', { bubbles: true }));
    return {
      situationChoices: document.querySelectorAll('.situation-choice').length,
      activeSituation: document.querySelector('.situation-choice.active b').textContent,
      scenarioCount: document.querySelectorAll('#commonScenarioList .probability-scenario').length,
      metricFontSize: getComputedStyle(document.querySelector('#commonScenarioList .probability-metric b')).fontSize,
      customOuts: document.getElementById('commonOutsValue').textContent,
      customRiver: document.querySelector('#commonOutsResults .probability-metric.primary b').textContent,
      bodyOverflow: document.documentElement.scrollWidth > window.innerWidth,
    };
  })()`);
  if (
    commonProbabilityPage.situationChoices !== 8 ||
    commonProbabilityPage.activeSituation !== "顺子听牌" ||
    commonProbabilityPage.scenarioCount !== 3 ||
    commonProbabilityPage.metricFontSize !== "22px" ||
    commonProbabilityPage.customOuts !== "15" ||
    commonProbabilityPage.customRiver !== "54.12%" ||
    commonProbabilityPage.bodyOverflow
  ) {
    throw new Error(`常见概率页面异常：${JSON.stringify(commonProbabilityPage)}`);
  }

  const gamePage = await evaluate(client, `(async () => {
    document.getElementById('handSeedInput').value = '0';
    document.getElementById('gameBtn').click();
    const waitsForExplicitStart = document.querySelectorAll('.game-seat').length === 0
      && getComputedStyle(document.getElementById('startGameBtn')).display !== 'none'
      && document.getElementById('decisionTitle').textContent === '准备开始';
    document.getElementById('startGameBtn').click();
    let heroProgressVisible = false;
    let heroThinkingHighlighted = false;
    for (let attempt = 0; attempt < 1200; attempt += 1) {
      const visible = [...document.querySelectorAll('#gameActions [data-action]')]
        .filter((button) => getComputedStyle(button).display !== 'none' && !button.disabled);
      if (visible.length) {
        visible[0].click();
        const heroSeat = document.querySelector('.hero-seat');
        heroProgressVisible = Boolean(heroSeat?.querySelector('.seat-progress'));
        heroThinkingHighlighted = heroSeat?.classList.contains('thinking') ?? false;
        break;
      }
      if (!document.getElementById('handReview').classList.contains('hidden')) {
        document.getElementById('newHandBtn').click();
      }
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    for (let attempt = 0; attempt < 1200; attempt += 1) {
      if (!document.getElementById('analysisContent').classList.contains('hidden')) break;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    const hiddenBotCards = document.querySelectorAll('.game-seat:not(.hero-seat) .card-back').length;
    document.getElementById('revealBotCardsBtn').click();
    const revealedBotCards = document.querySelectorAll('.game-seat:not(.hero-seat) .seat-cards .table-card:not(.card-back)').length;
    const tableWrapTop = document.querySelector('.poker-table-wrap').getBoundingClientRect().top;
    const topCardTop = Math.min(...[...document.querySelectorAll('.seat-0 .table-card, .seat-1 .table-card, .seat-2 .table-card')].map((card) => card.getBoundingClientRect().top));
    const heroCard = document.querySelector('.hero-seat .seat-cards .table-card');
    return {
      activePanel: document.querySelector('.tab.active').id,
      waitsForExplicitStart,
      startButtonHidden: getComputedStyle(document.getElementById('startGameBtn')).display === 'none',
      seats: document.querySelectorAll('.game-seat').length,
      heroCards: document.querySelectorAll('.hero-seat .seat-cards .table-card').length,
      analysisVisible: !document.getElementById('analysisContent').classList.contains('hidden'),
      explanationTitle: document.querySelector('#analysisExplanation h4')?.textContent ?? '',
      explanationReasons: document.querySelectorAll('#analysisExplanation > div p').length,
      exportButton: document.getElementById('exportGameHistoryBtn')?.textContent ?? '',
      seed: document.getElementById('handSeedInput')?.value ?? '',
      hiddenBotCards,
      revealedBotCards,
      heroProgressVisible,
      heroThinkingHighlighted,
      topCardsFullyVisible: topCardTop >= tableWrapTop,
      heroRankSize: getComputedStyle(heroCard.querySelector('b')).fontSize,
      heroSuitSize: getComputedStyle(heroCard.querySelector('i')).fontSize,
      evRows: document.querySelectorAll('#analysisActions .ev-row').length,
      continueButtonExists: Boolean(document.getElementById('continueHandBtn')),
      modelLabel: document.getElementById('analysisMethod').textContent,
      bodyOverflow: document.documentElement.scrollWidth > window.innerWidth,
    };
  })()`);
  const gpuBudgetLabel = gamePage.modelLabel.includes("GPU 训练进程")
    && gamePage.modelLabel.includes("每进程训练")
    && gamePage.modelLabel.includes("每进程评估");
  const cpuBudgetLabel = gamePage.modelLabel.includes("核并行")
    && gamePage.modelLabel.includes("每进程训练")
    && gamePage.modelLabel.includes("每进程评估");
  if (
    gamePage.activePanel !== "tab-game" ||
    !gamePage.waitsForExplicitStart ||
    !gamePage.startButtonHidden ||
    gamePage.seats !== 6 ||
    gamePage.heroCards !== 2 ||
    !gamePage.analysisVisible ||
    !gamePage.explanationTitle.startsWith("为什么") ||
    gamePage.explanationReasons < 2 ||
    gamePage.exportButton !== "复制完整记录" ||
    gamePage.seed !== "0" ||
    gamePage.hiddenBotCards !== 10 ||
    gamePage.revealedBotCards !== 10 ||
    !gamePage.heroProgressVisible ||
    !gamePage.heroThinkingHighlighted ||
    !gamePage.topCardsFullyVisible ||
    gamePage.heroRankSize !== "17px" ||
    gamePage.heroSuitSize !== "15px" ||
    gamePage.evRows < 2 ||
    gamePage.continueButtonExists ||
    (!gpuBudgetLabel && !cpuBudgetLabel) ||
    !gamePage.modelLabel.includes("合并") ||
    !gamePage.modelLabel.includes("根遗憾") ||
    !gamePage.modelLabel.includes("Monte Carlo") ||
    gamePage.bodyOverflow
  ) {
    throw new Error(`实时游戏页面异常：${JSON.stringify(gamePage)}`);
  }

  const calculatorActionsVisible = await evaluate(client, `(() => {
    document.querySelector("[data-tab=calculator]").click();
    return getComputedStyle(document.getElementById("calculatorActions")).display !== "none";
  })()`);
  if (!calculatorActionsVisible) throw new Error("自定义计算器操作按钮未显示");

  await evaluate(client, 'document.getElementById("calcBtn").click()');
  let status = "";
  for (let attempt = 0; attempt < 300; attempt += 1) {
    status = await evaluate(client, 'document.getElementById("status").textContent');
    if (status === "精确枚举完成" || status === "计算失败") break;
    await delay(100);
  }
  if (status !== "精确枚举完成") throw new Error(`计算未完成，当前状态：${status}`);

  await evaluate(client, 'document.querySelector("[data-tab=hands]").click()');

  await client.send("Emulation.setDeviceMetricsOverride", {
    width: 390,
    height: 844,
    deviceScaleFactor: 1,
    mobile: true,
  });
  await delay(50);
  const mobileLayout = await evaluate(client, `({
    bodyOverflow: document.documentElement.scrollWidth > window.innerWidth,
    bodyVerticalScroll: document.documentElement.scrollHeight > window.innerHeight,
    matrixScrollsInside: document.querySelector('.hands-table-wrap').scrollWidth > document.querySelector('.hands-table-wrap').clientWidth,
    axesHidden: getComputedStyle(document.querySelector('.hands-table thead')).display === 'none' && getComputedStyle(document.querySelector('.matrix-row-axis')).display === 'none',
    matrixColumns: document.querySelectorAll('#handsBody tr:first-child .hand-matrix-cell').length,
    cellWidth: document.querySelector('.hand-matrix-cell').getBoundingClientRect().width,
    cellHeight: document.querySelector('.hand-matrix-cell').getBoundingClientRect().height,
    cellContentFits: [...document.querySelectorAll('.hand-matrix-cell')].every((cell) => cell.scrollWidth <= cell.clientWidth && cell.scrollHeight <= cell.clientHeight),
    playerButtonRows: new Set([...document.querySelectorAll('#handPlayersChoices .choice')].map((button) => Math.round(button.getBoundingClientRect().top))).size,
    playerChoicesOverflow: document.getElementById('handPlayersChoices').scrollWidth > document.getElementById('handPlayersChoices').clientWidth,
    handPickerColumns: getComputedStyle(document.getElementById('handCardPicker')).gridTemplateColumns.split(' ').length,
    handPickerOverflow: document.getElementById('handCardPicker').scrollWidth > document.getElementById('handCardPicker').clientWidth,
    tabButtonRows: new Set([...document.querySelectorAll('.tab-btn')].map((button) => Math.round(button.getBoundingClientRect().top))).size,
    tabsOverflow: document.querySelector('.tabs').scrollWidth > document.querySelector('.tabs').clientWidth,
    equityLegendAfterTable: document.querySelector('.hands-table-wrap').nextElementSibling.classList.contains('hand-data-scope'),
  })`);
  if (
    mobileLayout.bodyOverflow ||
    mobileLayout.matrixScrollsInside ||
    !mobileLayout.axesHidden ||
    mobileLayout.matrixColumns !== 13 ||
    mobileLayout.cellWidth > 30 ||
    mobileLayout.cellHeight > 32 ||
    !mobileLayout.cellContentFits ||
    mobileLayout.playerButtonRows !== 1 ||
    mobileLayout.playerChoicesOverflow ||
    mobileLayout.handPickerColumns !== 14 ||
    mobileLayout.handPickerOverflow ||
    mobileLayout.tabButtonRows !== 1 ||
    mobileLayout.tabsOverflow ||
    !mobileLayout.equityLegendAfterTable
  ) {
    throw new Error(`390px 起手牌矩阵响应式异常：${JSON.stringify(mobileLayout)}`);
  }

  await evaluate(client, 'document.querySelector("[data-tab=common]").click()');
  const commonMobileLayout = await evaluate(client, `({
    bodyOverflow: document.documentElement.scrollWidth > window.innerWidth,
    situationColumns: getComputedStyle(document.querySelector('.situation-choices')).gridTemplateColumns.split(' ').length,
    scenarioColumns: getComputedStyle(document.querySelector('.probability-scenario')).gridTemplateColumns.split(' ').length,
  })`);
  if (
    commonMobileLayout.bodyOverflow ||
    commonMobileLayout.situationColumns !== 2 ||
    commonMobileLayout.scenarioColumns !== 1
  ) {
    throw new Error(`390px 常见概率页面响应式异常：${JSON.stringify(commonMobileLayout)}`);
  }

  console.log("Browser smoke passed: modules, Worker calculation, probability reference, desktop and mobile layout");
} finally {
  client?.close();
  browser.kill();
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      await rm(profileDirectory, { recursive: true, force: true });
      break;
    } catch (error) {
      if (attempt === 4) {
        console.warn(`浏览器临时目录将在系统清理时删除：${error.message}`);
      } else {
        await delay(200 * (attempt + 1));
      }
    }
  }
}
