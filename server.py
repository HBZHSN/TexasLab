from __future__ import annotations

import asyncio
from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from backend.abstraction import stable_public_key
from backend.blueprint import get_blueprint_runtime, runtime_signature
from backend.leaf_runtime import leaf_runtime_signature
from backend.schema import history_contains_private_cards, normalize_solve_info
from backend.solver import solve_decision, solver_config_signature

ROOT = Path(__file__).resolve().parent
# A validated policy or exact-node bundle published by the bundled training
# pipeline is picked up automatically.  ``auto`` preserves the safe full-tree
# fallback while the artifact is absent, incomplete, corrupt or incompatible.
os.environ.setdefault(
    "TEXAS_LAB_BLUEPRINT_PATH",
    str(ROOT / "artifacts" / "blueprint-production.json.gz"),
)
os.environ.setdefault("TEXAS_LAB_BLUEPRINT_MODE", "auto")
app = FastAPI(title="Texas Lab", version="1.1.0")
try:
    SOLVE_CACHE_SIZE = max(0, int(os.environ.get("TEXAS_LAB_SOLVE_CACHE_SIZE", "64")))
except ValueError:
    SOLVE_CACHE_SIZE = 64

_SOLVE_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_SOLVE_INFLIGHT: dict[str, Future[dict[str, Any]]] = {}
_SOLVE_CACHE_LOCK = threading.Lock()


def runtime_identity() -> dict[str, str]:
    return {
        "blueprint": runtime_signature(),
        "leaf": leaf_runtime_signature(),
        "solverConfig": solver_config_signature(),
    }


def solve_cache_key(
    info: dict[str, Any],
    iterations: int,
    runtime: dict[str, str] | None = None,
) -> str:
    """Canonical, collision-free key, shared by suit-isomorphic requests."""
    try:
        canonical_info: Any = stable_public_key(info)[1]
    except (KeyError, TypeError, ValueError):
        # Unit-level callers and diagnostics may provide a partial request.
        # Production WebSocket requests are normalized before reaching here.
        canonical_info = info
    return json.dumps(
        {
            "info": canonical_info,
            "iterations": iterations,
            "runtime": runtime or runtime_identity(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def clear_solve_cache() -> None:
    """Clear completed entries; primarily useful for tests and benchmarks."""
    with _SOLVE_CACHE_LOCK:
        _SOLVE_CACHE.clear()


def _solve_response(
    result: dict[str, Any],
    *,
    cache_hit: bool,
    source: str,
    started: float,
) -> dict[str, Any]:
    response = deepcopy(result)
    response["cacheHit"] = cache_hit
    response["solveSource"] = source
    response["solveElapsedMs"] = round((time.perf_counter() - started) * 1000, 3)
    return response


def solve_decision_cached(info: dict[str, Any], iterations: int) -> dict[str, Any]:
    """Reuse exact solves and coalesce concurrent requests for the same node."""
    started = time.perf_counter()
    requested_runtime = runtime_identity()
    key = solve_cache_key(info, iterations, requested_runtime)
    with _SOLVE_CACHE_LOCK:
        cached = _SOLVE_CACHE.get(key)
        if cached is not None:
            _SOLVE_CACHE.move_to_end(key)
            return _solve_response(
                cached,
                cache_hit=True,
                source="memory-cache",
                started=started,
            )

        future = _SOLVE_INFLIGHT.get(key)
        owner = future is None
        if owner:
            future = Future()
            _SOLVE_INFLIGHT[key] = future

    if not owner:
        return _solve_response(
            future.result(),
            cache_hit=True,
            source="shared-inflight",
            started=started,
        )

    try:
        try:
            blueprint_result = get_blueprint_runtime().lookup_exact(info, iterations)
        except (KeyError, TypeError, ValueError):
            # Production WebSocket inputs are normalized and validated.  Keep
            # lower-level diagnostic callers with partial payloads usable.
            blueprint_result = None
        canonical = deepcopy(
            blueprint_result
            if blueprint_result is not None
            else solve_decision(info, iterations)
        )
        if blueprint_result is not None:
            # Exact artifacts are intentionally produced under fixed, more
            # conservative Teacher settings.  Stamp the active artifact/config
            # identity so the reusable result belongs to the current cache key.
            canonical["blueprintFingerprint"] = requested_runtime["blueprint"]
            canonical["leafModelFingerprint"] = requested_runtime["leaf"]
            canonical["solverConfigFingerprint"] = requested_runtime["solverConfig"]
    except BaseException as error:
        with _SOLVE_CACHE_LOCK:
            _SOLVE_INFLIGHT.pop(key, None)
            future.set_exception(error)
        raise

    result_runtime_matches = all(
        canonical.get(field, expected) == expected
        for field, expected in (
            ("blueprintFingerprint", requested_runtime["blueprint"]),
            ("leafModelFingerprint", requested_runtime["leaf"]),
            ("solverConfigFingerprint", requested_runtime["solverConfig"]),
        )
    )
    runtime_stable = runtime_identity() == requested_runtime
    with _SOLVE_CACHE_LOCK:
        # Hot publication can occur between key creation and solve completion.
        # Never write a result produced by a different artifact/config snapshot
        # under the old cache key; callers may retry naturally on the next node.
        if SOLVE_CACHE_SIZE and result_runtime_matches and runtime_stable:
            _SOLVE_CACHE[key] = canonical
            _SOLVE_CACHE.move_to_end(key)
            while len(_SOLVE_CACHE) > SOLVE_CACHE_SIZE:
                _SOLVE_CACHE.popitem(last=False)
        _SOLVE_INFLIGHT.pop(key, None)
        future.set_result(canonical)

    return _solve_response(
        canonical,
        cache_hit=False,
        source="blueprint-exact" if blueprint_result is not None else "live-solve",
        started=started,
    )


def validate_information_set(info: dict[str, Any]) -> None:
    required = {
        "street", "board", "pot", "currentBet", "hero", "players",
        "pending", "history", "legalActions", "raiseTo",
    }
    if not required.issubset(info):
        raise ValueError("信息集字段不完整")
    if "deck" in info or any("hole" in player for player in info["players"]):
        raise ValueError("求解请求包含不可见牌信息")
    if history_contains_private_cards(info["history"]):
        raise ValueError("公开行动历史包含私有牌信息")
    known = [*info["hero"]["hole"], *info["board"]]
    if len(known) != len(set(known)) or any(not isinstance(card, int) or card < 0 or card > 51 for card in known):
        raise ValueError("已知牌数据无效")
    if not set(info["legalActions"]).issubset({"check", "call", "raise", "fold"}):
        raise ValueError("存在非法动作")
    if not info["pending"] or info["pending"][0] != info["hero"]["id"]:
        raise ValueError("待行动队列与当前行动者不一致")
    if any("totalCommitted" not in player for player in info["players"]):
        raise ValueError("缺少边池求解所需的累计投入")


@app.websocket("/ws/game")
async def game_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            request_id = message.get("requestId")
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong", "requestId": request_id})
                continue
            if message.get("type") != "solve":
                await websocket.send_json({"type": "error", "requestId": request_id, "message": "未知请求类型"})
                continue
            try:
                raw_info = message["info"]
                validate_information_set(raw_info)
                info = normalize_solve_info(raw_info)
                requested = int(message.get("iterations", 20_000))
                limit = 100_000
                iterations = max(500, min(requested, limit))
                await websocket.send_json({"type": "started", "requestId": request_id, "iterations": iterations})
                result = await asyncio.to_thread(solve_decision_cached, info, iterations)
                await websocket.send_json({"type": "result", "requestId": request_id, "result": result})
            except Exception as error:  # Keep the socket usable after a bad node.
                await websocket.send_json({"type": "error", "requestId": request_id, "message": str(error)})
    except WebSocketDisconnect:
        return


app.mount("/", StaticFiles(directory=ROOT, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=os.environ.get("TEXAS_LAB_HOST", "0.0.0.0"),
        port=int(os.environ.get("TEXAS_LAB_PORT", "8000")),
        reload=False,
    )
