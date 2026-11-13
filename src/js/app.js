import { cardCode } from "./core/cards.js";
import { STREET_BOARD_COUNTS } from "./core/constants.js";
import { calculateScenario } from "./services/calculation-service.js";
import { getWorkerCount } from "./services/worker-client.js";
import { createInitialState } from "./state.js";
import { GameUI } from "./game/game-ui.js";
import { CardPicker } from "./ui/card-picker.js";
import {
  exportRowsAsCsv,
  finishProgress,
  renderCalculation,
  setCalculationBusy,
  showCalculationError,
} from "./ui/calculator-view.js";
import {
  byId,
  renderChoiceGroup,
  selectAll,
  setProgress,
  showToast,
} from "./ui/dom.js";
import {
  renderCommonProbabilities,
  renderStartingHands,
} from "./ui/reference-tables.js?v=20260829-hand-picker";

const state = createInitialState();
const cardPicker = new CardPicker(state);
const workerCount = getWorkerCount();
const gameUI = new GameUI();

const STARTING_HAND_TEMPLATES = {
  sc: [cardCode(9, 0), cardCode(8, 0)],
  oc: [cardCode(9, 0), cardCode(8, 1)],
  suited: [cardCode(14, 0), cardCode(7, 0)],
  pair: [cardCode(8, 0), cardCode(8, 1)],
  aks: [cardCode(14, 0), cardCode(13, 0)],
  ako: [cardCode(14, 0), cardCode(13, 1)],
  gap: [cardCode(9, 0), cardCode(7, 0)],
  broadway: [cardCode(12, 0), cardCode(11, 1)],
};

function setDefaultTarget() {
  if (state.street === "preflop") state.target = "flop";
  else if (state.street === "flop") state.target = "river";
  else state.target = "showdown";
}

function renderControls() {
  renderChoiceGroup(
    byId("playersChoices"),
    [2, 3, 4, 5, 6, 7, 8, 9].map((players) => ({
      value: String(players),
      label: players === 6 ? "6-max" : `${players} 人`,
    })),
    String(state.players),
    (value) => {
      state.players = Number(value);
      renderControls();
    },
  );

  renderChoiceGroup(
    byId("streetChoices"),
    [
      ["preflop", "翻牌前"],
      ["flop", "翻牌"],
      ["turn", "转牌"],
      ["river", "河牌"],
    ].map(([value, label]) => ({ value, label })),
    state.street,
    (value) => {
      state.street = value;
      cardPicker.trimBoard();
      setDefaultTarget();
      renderControls();
      cardPicker.render();
    },
  );

  const targetOptions = {
    preflop: [
      ["flop", "看到翻牌"],
      ["turn", "看到转牌"],
      ["river", "看到河牌"],
      ["showdown", "到摊牌"],
    ],
    flop: [
      ["turn", "下一张转牌"],
      ["river", "转牌 + 河牌"],
      ["showdown", "到摊牌"],
    ],
    turn: [
      ["river", "下一张河牌"],
      ["showdown", "到摊牌"],
    ],
    river: [["showdown", "当前摊牌"]],
  }[state.street];
  renderChoiceGroup(
    byId("targetChoices"),
    targetOptions.map(([value, label]) => ({ value, label })),
    state.target,
    (value) => {
      state.target = value;
      renderControls();
    },
  );

  renderChoiceGroup(
    byId("iterationChoices"),
    [2000, 10000, 50000].map((iterations) => ({
      value: String(iterations),
      label: new Intl.NumberFormat("zh-CN").format(iterations),
    })),
    String(state.iterations),
    (value) => {
      state.iterations = Number(value);
      renderControls();
    },
  );
}

function validateScenario() {
  const known = cardPicker.knownCards();
  if (known.hero.length !== 2) {
    showToast("请先选择两张手牌");
    return false;
  }
  if (known.board.length !== STREET_BOARD_COUNTS[state.street]) {
    showToast(`当前街道需要 ${STREET_BOARD_COUNTS[state.street]} 张公共牌`);
    return false;
  }
  if (!selectAll("#goals input:checked").length) {
    showToast("至少选择一个期望事件");
    return false;
  }
  return true;
}

async function calculate() {
  if (state.busy || !validateScenario()) return;
  state.busy = true;
  setCalculationBusy(true);

  try {
    const { hero, board } = cardPicker.knownCards();
    const result = await calculateScenario({
      hero,
      board,
      target: state.target,
      players: state.players,
      iterations: state.iterations,
      selected: selectAll("#goals input:checked").map((input) => input.value),
      workerCount,
      onProgress: setProgress,
    });
    state.lastRows = result.rows;
    renderCalculation(result, state, workerCount);
    finishProgress();
  } catch (error) {
    console.error(error);
    showToast(`计算失败：${error.message}`);
    showCalculationError();
  } finally {
    state.busy = false;
    setCalculationBusy(false);
  }
}

function bindEvents() {
  selectAll(".tab-btn").forEach((button) => {
    button.onclick = () => {
      gameUI.leave();
      selectAll(".tab-btn").forEach((item) => {
        item.classList.toggle("active", item === button);
      });
      selectAll(".tab").forEach((tab) => tab.classList.remove("active"));
      byId(`tab-${button.dataset.tab}`).classList.add("active");
      byId("calculatorActions").classList.toggle(
        "hidden",
        button.dataset.tab !== "calculator",
      );
    };
  });

  selectAll(".template").forEach((button) => {
    button.onclick = () => {
      cardPicker.setCards(STARTING_HAND_TEMPLATES[button.dataset.template]);
      showToast("已设置起手牌，仍可手动替换任意一张");
    };
  });

  byId("clearBtn").onclick = () => {
    cardPicker.clear();
    showToast("牌面已清空");
  };
  byId("sampleBtn").onclick = loadSample;
  byId("calcBtn").onclick = calculate;
  byId("exportBtn").onclick = () => {
    if (!state.lastRows.length) showToast("当前没有可导出明细");
    else exportRowsAsCsv(state.lastRows);
  };
}

function bindResponsiveLayout() {
  const mobileQuery = window.matchMedia("(max-width: 600px)");
  const dataScope = document.querySelector(".hand-data-scope");
  const toolbar = document.querySelector(".hand-toolbar");
  const table = document.querySelector(".hands-table-wrap");
  const updateDataScopePosition = () => {
    if (mobileQuery.matches) table.after(dataScope);
    else toolbar.append(dataScope);
  };

  mobileQuery.addEventListener("change", updateDataScopePosition);
  updateDataScopePosition();
}

function loadSample() {
  state.street = "flop";
  state.target = "river";
  cardPicker.setCards(
    [cardCode(14, 0), cardCode(13, 0)],
    [cardCode(12, 0), cardCode(11, 1), cardCode(2, 0)],
  );
  renderControls();
  showToast("已载入 A♠K♠ / Q♠J♥2♠");
}

renderControls();
renderCommonProbabilities();
renderStartingHands();
bindEvents();
bindResponsiveLayout();
gameUI.initialize();
loadSample();
byId("cores").textContent = workerCount;
