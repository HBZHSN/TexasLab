import {
  enumerateEvents,
  simulateEquity,
  simulateStartingHands,
} from "../core/simulation.js";

self.onmessage = ({ data: message }) => {
  try {
    let result;
    if (message.type === "equity") result = simulateEquity(message);
    if (message.type === "enumerate") result = enumerateEvents(message);
    if (message.type === "hands") result = simulateStartingHands(message);

    if (!result) throw new Error(`未知的计算任务：${message.type}`);
    self.postMessage({ id: message.id, type: message.type, result });
  } catch (error) {
    self.postMessage({
      id: message.id,
      error: String(error?.stack ?? error),
    });
  }
};
