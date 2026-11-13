from __future__ import annotations

import argparse
import hashlib
from itertools import chain
import json
from pathlib import Path
import sys
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.value_model import (  # noqa: E402
    MAX_SAMPLE_BYTES,
    LeafSchemaError,
    LeafTrainingSample,
    dataset_header,
    strict_json_loads,
    write_leaf_dataset,
)
from backend.schema import history_contains_private_cards  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate public-belief teacher JSONL and atomically publish a safe JSONL.GZ leaf dataset. "
            "This command never converts a single-Hero EV solve into counterfactual value targets."
        ),
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Plain UTF-8 JSONL files, one dense teacher sample per line")
    parser.add_argument("--output", type=Path, required=True, help="Destination ending in .jsonl.gz")
    parser.add_argument("--teacher-version", required=True)
    parser.add_argument("--blueprint-version", required=True)
    parser.add_argument("--metadata", type=Path, help="Optional public-only JSON object")
    parser.add_argument("--max-samples", type=int, default=0, help="0 keeps all samples")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    raw = path.read_bytes()
    if len(raw) > 256_000:
        raise LeafSchemaError("metadata 超过安全大小限制")
    value = strict_json_loads(raw)
    if not isinstance(value, dict):
        raise LeafSchemaError("metadata 必须是 JSON 对象")
    if history_contains_private_cards(value):
        raise LeafSchemaError("metadata 包含私有牌字段")
    return value


def source_rows(paths: list[Path], maximum: int, counter: list[int]) -> Iterator[LeafTrainingSample]:
    sample_ids: set[str] = set()
    for path in paths:
        if path.suffix.lower() != ".jsonl":
            raise LeafSchemaError(f"teacher 输入必须是未压缩 .jsonl：{path}")
        with path.open("rb") as handle:
            line_number = 0
            while True:
                line = handle.readline(MAX_SAMPLE_BYTES + 1)
                if not line:
                    break
                line_number += 1
                if len(line) > MAX_SAMPLE_BYTES:
                    raise LeafSchemaError(f"{path.name}:{line_number} 超过安全行大小")
                if not line.strip():
                    raise LeafSchemaError(f"{path.name}:{line_number} 是空行")
                try:
                    sample = LeafTrainingSample.from_dict(strict_json_loads(line))
                except LeafSchemaError as error:
                    raise LeafSchemaError(f"{path.name}:{line_number}: {error}") from error
                if sample.sample_id in sample_ids:
                    raise LeafSchemaError(f"重复 sampleId：{sample.sample_id}")
                sample_ids.add(sample.sample_id)
                counter[0] += 1
                yield sample
                if maximum and counter[0] >= maximum:
                    return


def main() -> int:
    args = parse_args()
    if args.max_samples < 0:
        raise SystemExit("--max-samples cannot be negative")
    output = args.output.resolve()
    inputs = [path.resolve() for path in args.inputs]
    if output in inputs:
        raise SystemExit("output must not overwrite a teacher input")
    metadata = read_metadata(args.metadata)
    metadata.update(
        {
            "sources": [
                {
                    "name": path.name,
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in inputs
            ],
            "targetSemantics": "conditional-final-stack-by-seat-and-hand",
            "privateCardsAllowed": False,
        }
    )
    header = dataset_header(
        teacher_version=args.teacher_version,
        blueprint_version=args.blueprint_version,
        metadata=metadata,
    )
    counter = [0]
    samples = source_rows(inputs, args.max_samples, counter)
    try:
        first = next(samples)
    except StopIteration:
        raise SystemExit("teacher input did not contain any samples")
    write_leaf_dataset(output, header, chain((first,), samples))
    print(
        json.dumps(
            {
                "output": str(output),
                "samples": counter[0],
                "teacherVersion": args.teacher_version,
                "blueprintVersion": args.blueprint_version,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
