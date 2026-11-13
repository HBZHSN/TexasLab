export const byId = (id) => document.getElementById(id);
export const selectAll = (selector) => [...document.querySelectorAll(selector)];

export function renderChoiceGroup(element, items, value, onChange) {
  element.innerHTML = items
    .map(
      (item) =>
        `<button class="choice ${item.value === value ? "active" : ""}" data-value="${item.value}">${item.label}</button>`,
    )
    .join("");
  [...element.children].forEach((button) => {
    button.onclick = () => onChange(button.dataset.value);
  });
}

export function formatPercent(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatNumber(value) {
  return new Intl.NumberFormat("zh-CN").format(value);
}

export function showToast(message) {
  const element = byId("toast");
  element.textContent = message;
  element.classList.add("show");
  clearTimeout(showToast.timeout);
  showToast.timeout = setTimeout(() => element.classList.remove("show"), 1800);
}

export function setProgress(percent) {
  const progress = byId("progress");
  progress.style.display = "block";
  progress.firstElementChild.style.width = `${Math.max(0, Math.min(100, percent))}%`;
}
