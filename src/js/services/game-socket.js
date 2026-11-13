export class GameSocketClient {
  constructor() {
    this.socket = null;
    this.connecting = null;
    this.pending = new Map();
    this.sequence = 0;
  }

  async connect() {
    if (this.socket?.readyState === WebSocket.OPEN) return;
    if (this.connecting) return this.connecting;
    this.connecting = new Promise((resolve, reject) => {
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      const socket = new WebSocket(`${protocol}//${location.host}/ws/game`);
      socket.onopen = () => {
        this.socket = socket;
        this.connecting = null;
        resolve();
      };
      socket.onerror = () => {
        this.connecting = null;
        reject(new Error("无法连接 Python 求解服务，请使用 python server.py 启动项目"));
      };
      socket.onclose = () => {
        if (this.socket === socket) this.socket = null;
        this.pending.forEach(({ reject: rejectRequest }) => rejectRequest(new Error("求解连接已断开")));
        this.pending.clear();
      };
      socket.onmessage = ({ data }) => this.handle(JSON.parse(data));
    });
    return this.connecting;
  }

  handle(message) {
    const request = this.pending.get(message.requestId);
    if (!request) return;
    if (message.type === "started") request.onStarted?.(message.iterations);
    if (message.type === "result") {
      this.pending.delete(message.requestId);
      request.resolve(message.result);
    } else if (message.type === "error") {
      this.pending.delete(message.requestId);
      request.reject(new Error(message.message));
    }
  }

  async solve(info, { quality = "hero", iterations, onStarted } = {}) {
    await this.connect();
    this.sequence += 1;
    const requestId = `decision-${this.sequence}`;
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject, onStarted });
      this.socket.send(JSON.stringify({ type: "solve", requestId, info, quality, iterations }));
    });
  }
}
