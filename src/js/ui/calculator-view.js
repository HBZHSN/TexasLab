import { CARDS_BY_CODE } from "../core/cards.js";
import { EVENT_LABELS, HAND_CATEGORIES } from "../core/constants.js";
import { currentHandName } from "../core/evaluator.js";
import { byId, formatNumber, formatPercent, setProgress } from "./dom.js";

const STREET_LABELS = {
  preflop: "翻牌前",
  flop: "翻牌",
  turn: "转牌",
  river: "河牌",
};

const TARGET_LABELS = {
  flop: "翻牌",
  turn: "转牌",
  river: "河牌",
  showdown: "摊牌",
};

export function setCalculationBusy(isBusy) {
  byId("calcBtn").disabled = isBusy;
  if (isBusy) {
    const status = byId("status");
    status.textContent = "多核计算中";
    status.style.background = "#fff7e5";
    status.style.color = "#8f6500";
    setProgress(3);
  }
}

export function showCalculationError() {
  byId("status").textContent = "计算失败";
}

export function renderCalculation(result, state, workerCount) {
  const equityHoeffding95 = Math.sqrt(Math.log(40) / (2 * result.iterations));
  byId("made").textContent = currentHandName([...result.hero, ...result.board]);
  byId("equity").textContent = formatPercent(result.equity);
  byId("combos").textContent = formatNumber(result.total);
  byId("cores").textContent = workerCount;
  byId("equityFoot").textContent = `${formatNumber(result.iterations)} 次模拟 · 95% 分布无关误差 ≤ ${(equityHoeffding95 * 100).toFixed(1)} 个百分点 · 胜 ${result.wins} / 平 ${result.ties} / 负 ${result.losses}`;
  byId("comboFoot").textContent = `已完整枚举 ${formatNumber(result.processed)} 组`;
  byId("summary").textContent = `${state.players} 人桌 · ${STREET_LABELS[state.street]} → ${TARGET_LABELS[state.target]}`;

  const status = byId("status");
  status.textContent = "精确枚举完成";
  status.style.background = "#f0fbf6";
  status.style.color = "#16734e";

  byId("probList").innerHTML = result.selected
    .map((event) => {
      const probability = result.counts[event] / result.total;
      return `<div class="prob"><div class="prob-name">${EVENT_LABELS[event]}</div><div class="bar"><i style="width:${probability * 100}%"></i></div><div class="prob-v">${formatPercent(probability, probability < 0.01 ? 2 : 1)}</div></div>`;
    })
    .join("");

  byId("formulaArea").innerHTML = [
    `<div class="formula"><div class="formula-label">公共牌组合空间</div><div class="formula-main">C(${result.remaining.length}, ${result.need}) = ${formatNumber(result.total)}</div><div class="formula-detail">所有组合均已枚举，不使用抽样近似。</div></div>`,
    `<div class="formula"><div class="formula-label">事件分子</div>${result.selected.map((event) => `<div class="formula-detail">${EVENT_LABELS[event]}：${formatNumber(result.counts[event])} ÷ ${formatNumber(result.total)} = ${formatPercent(result.counts[event] / result.total, result.counts[event] / result.total < 0.01 ? 2 : 1)}</div>`).join("")}</div>`,
    `<div class="formula"><div class="formula-label">随机对手 Equity</div><div class="formula-main">${formatPercent(result.equity)}</div><div class="formula-detail">Monte Carlo ${formatNumber(result.iterations)} 次，由 ${workerCount} 个 Web Worker 并行完成；95% Hoeffding 误差半径不超过 ${(equityHoeffding95 * 100).toFixed(1)} 个百分点，平局按底池份额计入。</div></div>`,
  ].join("");

  const outTags = [];
  if (result.outs.gut.length) outTags.push(`卡顺 outs ${result.outs.gut.length} 张`);
  if (result.outs.open.length) outTags.push(`两头顺 outs ${result.outs.open.length} 张`);
  if (result.outs.flush.length) outTags.push(`同花 outs ${result.outs.flush.length} 张`);
  if (!outTags.length) outTags.push("未识别到标准一张牌听牌");
  byId("outs").innerHTML = outTags
    .map((label, index) => `<span class="tag ${index ? "" : "blue"}">${label}</span>`)
    .join("");

  renderDetailRows(result.rows);
  byId("detailNote").textContent = `完整命中数量按上方分子统计；界面展示前 ${formatNumber(result.rows.length)} 条组合。`;
}

function renderDetailRows(rows) {
  byId("detailBody").innerHTML = rows.length
    ? rows
        .map(
          (row, index) =>
            `<tr><td>${index + 1}</td><td><div class="mini">${
              row.run.length
                ? row.run
                    .map((code) => {
                      const card = CARDS_BY_CODE.get(code);
                      return `<span class="mini-card ${card.red ? "red" : ""}">${card.text}</span>`;
                    })
                    .join("")
                : '<span class="tag">无需补牌</span>'
            }</div></td><td>${HAND_CATEGORIES[row.category]}</td><td><div class="mini">${row.hits.map((event) => `<span class="match">${EVENT_LABELS[event]}</span>`).join("")}</div></td></tr>`,
        )
        .join("")
    : '<tr><td colspan="4" class="empty">没有组合命中当前条件</td></tr>';
}

export function finishProgress() {
  setProgress(100);
  setTimeout(() => {
    byId("progress").style.display = "none";
  }, 600);
}

export function exportRowsAsCsv(rows) {
  const csvRows = [["序号", "后续公共牌", "最终牌型", "命中目标"]];
  rows.forEach((row, index) => {
    csvRows.push([
      index + 1,
      row.run.map((code) => CARDS_BY_CODE.get(code).text).join(" "),
      HAND_CATEGORIES[row.category],
      row.hits.map((event) => EVENT_LABELS[event]).join("、"),
    ]);
  });

  const csv =
    "\ufeff" +
    csvRows
      .map((row) =>
        row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(","),
      )
      .join("\r\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "texas-lab-combinations.csv";
  link.click();
  URL.revokeObjectURL(url);
}
