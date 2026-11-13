import { DECK } from "../core/cards.js";
import { RANKS, RANK_TEXT, SUITS } from "../core/constants.js";
import {
  COMMON_GROUPS,
  COMMON_SCENARIOS,
  probabilityAtLeastOne,
  scenarioMetrics,
} from "../data/common-probabilities.js";
import { PRECOMPUTED_EQUITY } from "../data/precomputed-equity.js";
import {
  byId,
  formatPercent,
  renderChoiceGroup,
} from "./dom.js";

const PLAYER_COUNTS = [2, 3, 4, 5, 6, 7, 8, 9];
const selectedStartingCards = [];

export function renderCommonProbabilities(groupId = "pair") {
  const selectedGroup = COMMON_GROUPS.find((group) => group.id === groupId) ?? COMMON_GROUPS[0];
  const scenarios = COMMON_SCENARIOS.filter((scenario) => scenario.group === selectedGroup.id);

  byId("commonSituationChoices").innerHTML = COMMON_GROUPS.map((group) => {
    const count = COMMON_SCENARIOS.filter((scenario) => scenario.group === group.id).length;
    const active = group.id === selectedGroup.id;
    return `<button class="situation-choice${active ? " active" : ""}" data-group="${group.id}" aria-pressed="${active}"><span class="situation-short">${group.short}</span><span class="situation-copy"><b>${group.label}</b><small>${count} 种情况</small></span><span class="situation-arrow">›</span></button>`;
  }).join("");
  [...byId("commonSituationChoices").children].forEach((button) => {
    button.onclick = () => renderCommonProbabilities(button.dataset.group);
  });

  byId("commonGroupTitle").textContent = selectedGroup.label;
  byId("commonGroupDescription").textContent = selectedGroup.description;
  byId("commonScenarioCount").textContent = `${scenarios.length} 种精确场景`;
  byId("commonScenarioList").innerHTML = scenarios.map(renderScenario).join("");

  const outsRange = byId("commonOutsRange");
  outsRange.oninput = () => renderCustomOuts(Number(outsRange.value));
  renderCustomOuts(Number(outsRange.value));
}

function renderScenario(scenario) {
  const outsBadge = scenario.outs
    ? `<span class="scenario-outs"><b>${scenario.outs}</b> outs</span>`
    : `<span class="scenario-outs formula-badge">组合公式</span>`;
  return `<article class="probability-scenario">
    <div class="scenario-copy">
      <div class="scenario-meta"><span>${scenario.stage}</span>${outsBadge}</div>
      <h3>${scenario.title}</h3>
      <p>${scenario.description}</p>
      ${scenario.target ? `<div class="scenario-target">目标：${scenario.target}</div>` : ""}
    </div>
    <div class="scenario-metrics">${scenarioMetrics(scenario).map(renderProbabilityMetric).join("")}</div>
    ${scenario.note ? `<div class="scenario-note">${scenario.note}</div>` : ""}
  </article>`;
}

function renderProbabilityMetric(metric) {
  return `<div class="probability-metric${metric.primary ? " primary" : ""}">
    <span>${metric.label}</span>
    <b>${formatPercent(metric.value, 2)}</b>
    <small>${metric.detail ?? "精确组合概率"}</small>
  </div>`;
}

function renderCustomOuts(outs) {
  byId("commonOutsValue").textContent = String(outs);
  const metrics = [
    { label: "转牌命中率", value: probabilityAtLeastOne(outs, 1, 47), detail: `${outs} / 47` },
    { label: "截至河牌的命中率", value: probabilityAtLeastOne(outs, 2, 47), detail: "转牌或河牌，任意一次命中", primary: true },
    { label: "若转牌没中：河牌命中率", value: probabilityAtLeastOne(outs, 1, 46), detail: `${outs} / 46` },
  ];
  byId("commonOutsResults").innerHTML = metrics.map(renderProbabilityMetric).join("");
}

export function startingHandMatrixKey(rowRank, columnRank) {
  if (rowRank === columnRank) return `${RANK_TEXT[rowRank]}${RANK_TEXT[columnRank]}`;

  const highRank = Math.max(rowRank, columnRank);
  const lowRank = Math.min(rowRank, columnRank);
  const kind = rowRank > columnRank ? "s" : "o";
  return `${RANK_TEXT[highRank]}${RANK_TEXT[lowRank]}${kind}`;
}

export function startingHandKey(firstCard, secondCard) {
  const highRank = Math.max(firstCard.rank, secondCard.rank);
  const lowRank = Math.min(firstCard.rank, secondCard.rank);
  const ranks = `${RANK_TEXT[highRank]}${RANK_TEXT[lowRank]}`;

  if (highRank === lowRank) return ranks;
  return `${ranks}${firstCard.suit === secondCard.suit ? "s" : "o"}`;
}

export function renderStartingHands(players = 6) {
  const selectedPlayers = PLAYER_COUNTS.includes(players) ? players : 6;
  const equityTable = PRECOMPUTED_EQUITY[String(selectedPlayers)];
  const equityValues = Object.values(equityTable);
  const minimumEquity = Math.min(...equityValues);
  const maximumEquity = Math.max(...equityValues);

  renderChoiceGroup(
    byId("handPlayersChoices"),
    PLAYER_COUNTS.map((playerCount) => ({
      value: String(playerCount),
      label: playerCount === 6 ? "6" : `${playerCount} 人`,
    })),
    String(selectedPlayers),
    (value) => renderStartingHands(Number(value)),
  );

  byId("handsHead").innerHTML = `<tr><th class="matrix-axis matrix-corner" scope="col">手牌</th>${RANKS.map((rank) => `<th class="matrix-axis" scope="col">${RANK_TEXT[rank]}</th>`).join("")}</tr>`;
  byId("handsBody").innerHTML = RANKS.map(
    (rowRank) =>
      `<tr><th class="matrix-axis matrix-row-axis" scope="row">${RANK_TEXT[rowRank]}</th>${RANKS.map((columnRank) => {
        const handKey = startingHandMatrixKey(rowRank, columnRank);
        const kind = rowRank === columnRank ? "pair" : rowRank > columnRank ? "suited" : "offsuit";
        const kindLabel = kind === "pair" ? "口袋对子" : kind === "suited" ? "同花" : "非同花";
        const equity = equityTable[handKey];
        const heatTier = equityHeatTier(equity, minimumEquity, maximumEquity);
        return `<td class="hand-matrix-cell hand-${kind}" data-hand-key="${handKey}" data-equity-tier="${heatTier}" title="${handKey} · ${kindLabel} · ${formatPercent(equity)}"><span class="matrix-hand-name">${handKey}</span>${renderEquityCell(equity)}</td>`;
      }).join("")}</tr>`,
  ).join("");
  renderStartingHandPicker();
  updateStartingHandHighlight();
}

function renderStartingHandPicker() {
  const selectedCodes = new Set(selectedStartingCards.map((card) => card.code));
  byId("handCardPicker").innerHTML = SUITS.map((suit) => {
    const suitCards = DECK.filter((card) => card.suit === suit.id);
    return `<span class="hand-picker-suit${suit.red ? " red" : ""}" aria-hidden="true">${suit.symbol}</span>${suitCards.map((card) => {
      const selected = selectedCodes.has(card.code);
      const action = selected ? "取消选择" : "选择";
      return `<button type="button" class="hand-picker-card${card.red ? " red" : ""}${selected ? " selected" : ""}" data-card="${card.code}" aria-pressed="${selected}" aria-label="${action} ${card.text}"><b>${RANK_TEXT[card.rank]}</b><span>${suit.symbol}</span></button>`;
    }).join("")}`;
  }).join("");

  [...byId("handCardPicker").querySelectorAll(".hand-picker-card")].forEach((button) => {
    button.onclick = () => selectStartingCard(Number(button.dataset.card));
  });
  updateStartingHandStatus();
}

function selectStartingCard(code) {
  const selectedIndex = selectedStartingCards.findIndex((card) => card.code === code);
  if (selectedIndex >= 0) {
    selectedStartingCards.splice(selectedIndex, 1);
  } else {
    const card = DECK.find((item) => item.code === code);
    if (selectedStartingCards.length === 2) selectedStartingCards.splice(0, 2);
    selectedStartingCards.push(card);
  }

  renderStartingHandPicker();
  updateStartingHandHighlight(selectedStartingCards.length === 2);
}

function updateStartingHandStatus() {
  const status = byId("handPickerStatus");
  byId("handPickerCount").textContent = `${selectedStartingCards.length} / 2`;

  if (selectedStartingCards.length === 0) {
    status.textContent = "点击任意两张牌，选完后自动高亮对应胜率";
  } else if (selectedStartingCards.length === 1) {
    status.textContent = `已选 ${selectedStartingCards[0].text}，再选择一张牌`;
  } else {
    const handKey = startingHandKey(...selectedStartingCards);
    status.textContent = `已选 ${selectedStartingCards.map((card) => card.text).join(" ")} · 对应 ${handKey}`;
  }
}

function updateStartingHandHighlight(scrollIntoView = false) {
  const cells = [...byId("handsBody").querySelectorAll(".hand-matrix-cell")];
  const handKey = selectedStartingCards.length === 2
    ? startingHandKey(...selectedStartingCards)
    : null;
  let selectedCell = null;

  cells.forEach((cell) => {
    const selected = cell.dataset.handKey === handKey;
    cell.classList.toggle("selected-hand", selected);
    if (selected) {
      cell.setAttribute("aria-current", "true");
      selectedCell = cell;
    } else {
      cell.removeAttribute("aria-current");
    }
  });

  if (scrollIntoView && selectedCell) {
    selectedCell.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
  }
}

function equityHeatTier(value, minimum, maximum) {
  const normalized = (value - minimum) / (maximum - minimum);
  return Math.min(5, Math.floor(normalized * 6));
}

function renderEquityCell(value) {
  return `<span class="matrix-equity">${formatPercent(value)}</span>`;
}
