from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
sys.path.insert(0, str(ROOT))


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def solved_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row["key"])
        for row in read_rows(path)
        if row.get("kind") == "root-teacher-v1" and row.get("key")
    }


def run_one(
    index: int,
    row: dict[str, Any],
    directory: Path,
    iterations: int,
    seed: int,
    device: str,
) -> tuple[int, dict[str, Any], float]:
    input_path = directory / f"node-{index:06d}.jsonl"
    output_path = directory / f"teacher-{index:06d}.jsonl"
    input_path.write_text(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "TEXAS_LAB_GPU_TRAINING_WORKERS": "1",
            "TEXAS_LAB_PREFLOP_GPU_TRAINING_WORKERS": "1",
            "TEXAS_LAB_CPU_TRAINING_WORKERS": "1",
        }
    )
    if device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    started = time.perf_counter()
    completed = subprocess.run(
        [
            PYTHON,
            "scripts/train-blueprint.py",
            "collect",
            "--input",
            str(input_path),
            "--dataset",
            str(output_path),
            "--iterations",
            str(iterations),
            "--seed",
            str(seed),
            "--regret-mode",
            "rmplus",
            "--single-replica",
            "--max-new-samples",
            "1",
        ],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="backslashreplace",
    )
    if completed.returncode:
        raise RuntimeError(
            f"node {index} failed with {completed.returncode}:\n{completed.stdout[-4000:]}"
        )
    output_rows = read_rows(output_path)
    if len(output_rows) != 1:
        raise RuntimeError(f"node {index} produced {len(output_rows)} Teacher rows")
    return index, output_rows[0], time.perf_counter() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect independent Blueprint nodes concurrently on CPU or one CUDA device.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-new-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 500 or args.workers < 1 or args.max_new_samples < 0:
        raise SystemExit("invalid parallel collection configuration")
    source = read_rows(args.input.resolve())
    dataset = args.dataset.resolve()
    dataset.parent.mkdir(parents=True, exist_ok=True)
    solved = solved_keys(dataset)
    from backend.blueprint import stable_info_key

    pending = [
        (index, row)
        for index, row in enumerate(source)
        if stable_info_key(row.get("info", row))[0] not in solved
    ]
    if args.max_new_samples:
        pending = pending[: args.max_new_samples]
    if not pending:
        print(json.dumps({"completed": 0, "message": "no pending nodes"}), flush=True)
        return

    completed_count = 0
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="texas-lab-parallel-", dir=dataset.parent) as temporary:
        directory = Path(temporary)
        with ThreadPoolExecutor(max_workers=min(args.workers, len(pending))) as pool:
            futures: dict[Future[tuple[int, dict[str, Any], float]], int] = {
                pool.submit(
                    run_one,
                    index,
                    row,
                    directory,
                    args.iterations,
                    args.seed,
                    args.device,
                ): index
                for index, row in pending
            }
            with dataset.open("a", encoding="utf-8", newline="\n") as output:
                for future in as_completed(futures):
                    index, teacher_row, seconds = future.result()
                    output.write(
                        json.dumps(teacher_row, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                    output.flush()
                    os.fsync(output.fileno())
                    completed_count += 1
                    elapsed = time.perf_counter() - started
                    rate = completed_count / elapsed * 60
                    print(
                        f"[{completed_count}/{len(pending)}] source={index} "
                        f"node={seconds:.1f}s throughput={rate:.2f}/min",
                        flush=True,
                    )

    print(
        json.dumps(
            {
                "completed": completed_count,
                "workers": min(args.workers, len(pending)),
                "elapsedSeconds": time.perf_counter() - started,
                "nodesPerMinute": completed_count / (time.perf_counter() - started) * 60,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
