const WORKER_URL = new URL("../workers/calculation.worker.js", import.meta.url);

export function getWorkerCount() {
  return Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1));
}

export function runWorkerTask(payload) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(WORKER_URL, { type: "module" });

    worker.onmessage = ({ data }) => {
      worker.terminate();
      if (data.error) reject(new Error(data.error));
      else resolve(data.result);
    };

    worker.onerror = (event) => {
      worker.terminate();
      reject(event.error || new Error(event.message));
    };

    worker.postMessage(payload);
  });
}
