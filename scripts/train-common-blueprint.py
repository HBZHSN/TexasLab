from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
sys.path.insert(0, str(ROOT))


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    opener = gzip.open if path.suffix.lower() == ".gz" else Path.open
    if opener is gzip.open:
        handle = opener(path, "rt", encoding="utf-8")
    else:
        handle = path.open("r", encoding="utf-8")
    with handle:
        return sum(bool(line.strip()) for line in handle)


def run(command: list[str], log) -> None:
    print("$ " + " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="backslashreplace",
        env=environment,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.buffer.write(line.encode(sys.stdout.encoding or "utf-8", errors="backslashreplace"))
        sys.stdout.buffer.flush()
        log.write(line)
        log.flush()
    status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def priority_source(source: Path, teacher: Path, destination: Path) -> Path:
    """Place unsolved nodes first so capped collect batches never rescan 50MB."""
    if not teacher.exists():
        return source
    solved: set[str] = set()
    with teacher.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("key")
            if isinstance(key, str) and key:
                solved.add(key)
    from backend.blueprint import stable_info_key

    remaining: list[str] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            digest = stable_info_key(row.get("info", row))[0]
            if digest not in solved:
                remaining.append(line if line.endswith("\n") else line + "\n")
    destination.write_text("".join(remaining), encoding="utf-8", newline="\n")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume common-node Teacher collection and publish Blueprint batches.",
    )
    parser.add_argument("--public-states", type=int, default=192)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--batch-nodes", type=int, default=507)
    parser.add_argument("--initial-batch-nodes", type=int, default=169)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--max-total-nodes", type=int, default=0)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--log", type=Path, default=ROOT / "logs" / "common-blueprint.log")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.public_states < 1
        or args.iterations < 500
        or args.batch_nodes < 1
        or args.initial_batch_nodes < 1
        or args.workers < 1
    ):
        raise SystemExit("invalid positive training configuration")
    artifacts = args.artifacts_dir.resolve()
    data = args.data_dir.resolve()
    log_path = args.log.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    nodes = data / "common-preflop-nodes.jsonl"
    teacher = artifacts / "common-preflop-teacher.jsonl"
    production = artifacts / "blueprint-production.json.gz"
    pending_nodes = data / ".common-preflop-pending.jsonl"

    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(f"\n=== run {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        run(
            [
                PYTHON,
                "scripts/generate-common-nodes.py",
                "--output",
                str(nodes),
                "--public-states",
                str(args.public_states),
            ],
            log,
        )
        total = count_rows(nodes)
        target = min(total, args.max_total_nodes) if args.max_total_nodes else total
        while count_rows(teacher) < target:
            before = count_rows(teacher)
            # If an interrupted first run already persisted a few rows, finish
            # the small initial publication instead of waiting for a full batch.
            if not production.exists() and before < args.initial_batch_nodes:
                batch_size = args.initial_batch_nodes - before
            else:
                batch_size = args.batch_nodes
            batch = min(batch_size, target - before)
            collect_source = priority_source(nodes, teacher, pending_nodes)
            run(
                [
                    PYTHON,
                    "scripts/collect-blueprint-parallel.py",
                    "--input",
                    str(collect_source),
                    "--dataset",
                    str(teacher),
                    "--iterations",
                    str(args.iterations),
                    "--seed",
                    str(args.seed),
                    "--workers",
                    str(args.workers),
                    "--device",
                    args.device,
                    "--max-new-samples",
                    str(batch),
                ],
                log,
            )
            after = count_rows(teacher)
            if after <= before:
                raise RuntimeError("Teacher collection made no progress")
            run(
                [
                    PYTHON,
                    "scripts/train-blueprint.py",
                    "train",
                    "--dataset",
                    str(teacher),
                    "--output",
                    str(production),
                    "--engine",
                    "auto",
                    "--device",
                    "cpu",
                    "--epochs",
                    str(args.epochs),
                    "--seed",
                    str(args.seed),
                ],
                log,
            )
            print(json.dumps({"completed": after, "target": target, "artifact": str(production)}), flush=True)


if __name__ == "__main__":
    main()
