from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.abstraction import ABSTRACTION_VERSION, RULES_HASH
from backend.blueprint import (
    ACTION_ORDER,
    BLUEPRINT_FORMAT,
    FEATURE_SIZE,
    FEATURE_VERSION,
    SCHEMA_VERSION,
    encode_info_features,
    stable_info_key,
    validate_bundle,
    write_bundle,
)
from backend.schema import normalize_solve_info


def open_text(path: Path, mode: str):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open_text(path, "rt") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: JSON 无效") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: 每行必须是对象")
            rows.append(value)
    return rows


def teacher_dataset_index(path: Path) -> tuple[int, set[str]]:
    if not path.exists():
        return 0, set()
    count = 0
    canonical: set[str] = set()
    with open_text(path, "rt") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: JSON 无效") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: 每行必须是对象")
            if row.get("kind") == "root-teacher-v1" and row.get("canonical"):
                count += 1
                canonical.add(str(row["canonical"]))
    return count, canonical


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".jsonl.gz" if path.suffix.lower() == ".gz" else ".jsonl"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=suffix, dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with open_text(temporary, "wt") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def policy_target(result: dict[str, Any]) -> list[float]:
    by_action = {row["action"]: float(row["frequency"]) for row in result["actions"]}
    target = [max(0.0, by_action.get(action, 0.0)) for action in ACTION_ORDER]
    total = sum(target)
    if total <= 0:
        raise ValueError("Teacher 返回了空策略")
    return [value / total for value in target]


def collect_teacher(args: argparse.Namespace) -> None:
    if args.max_new_samples < 0 or args.checkpoint_every < 1:
        raise ValueError("max-new-samples must be non-negative and checkpoint-every must be positive")
    # A teacher must never consume the artifact it is currently rebuilding.
    os.environ["TEXAS_LAB_BLUEPRINT_MODE"] = "off"
    os.environ["TEXAS_LAB_LEAF_MODEL_MODE"] = "off"
    os.environ["TEXAS_LAB_ADAPTIVE_TRAINING"] = "off"
    os.environ["TEXAS_LAB_TRAIN_CHUNK_SIZE"] = "256"
    os.environ["TEXAS_LAB_TRAIN_MIN_ITERATIONS"] = "1500"
    os.environ["TEXAS_LAB_PREFLOP_TRAIN_MIN_ITERATIONS"] = "1024"
    os.environ.pop("TEXAS_LAB_TRAIN_EV_TOLERANCE", None)
    os.environ["TEXAS_LAB_TRAIN_STABLE_CHECKS"] = "2"
    os.environ["TEXAS_LAB_TRAIN_DOMINANCE"] = "true"
    os.environ.pop("TEXAS_LAB_TRAIN_DOMINANCE_MARGIN", None)
    os.environ["TEXAS_LAB_TRAIN_DOMINANCE_CHECKS"] = "1"
    os.environ["TEXAS_LAB_TRAIN_DOMINANCE_DRIFT"] = "2"
    os.environ["TEXAS_LAB_TRAIN_BEST_FREQUENCY"] = "0.70"
    os.environ["TEXAS_LAB_PREFLOP_DOMINATED_EVAL_TOTAL"] = "2000"
    os.environ["TEXAS_LAB_ADAPTIVE_EVALUATION"] = "off"
    os.environ["TEXAS_LAB_REGRET_MODE"] = args.regret_mode
    if args.single_replica:
        # Outer node parallelism keeps independent deep trees alive across
        # nodes and avoids nesting another process pool.  A single 20k solve
        # keeps the full 12k traversal budget and uses a larger 8k EV sample.
        os.environ["TEXAS_LAB_GPU_TRAINING_WORKERS"] = "1"
        os.environ["TEXAS_LAB_PREFLOP_GPU_TRAINING_WORKERS"] = "1"
        os.environ["TEXAS_LAB_CPU_TRAINING_WORKERS"] = "1"
    from backend.solver import shutdown_process_pool, solve_decision

    source = read_jsonl(Path(args.input))
    dataset_path = Path(args.dataset)
    initial_count, seen = (
        teacher_dataset_index(dataset_path)
        if args.append
        else (0, set())
    )
    dataset_path = dataset_path.resolve()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    # Long Teacher runs take days.  Append every completed solve immediately so
    # a reboot or interruption loses at most the currently running node.
    if not args.append:
        atomic_write_jsonl(dataset_path, [])
    output = open_text(dataset_path, "at")
    completed = 0
    try:
        for index, row in enumerate(source):
            if args.max_new_samples and completed >= args.max_new_samples:
                break
            info = normalize_solve_info(row.get("info", row))
            digest, canonical = stable_info_key(info)
            if canonical in seen:
                continue
            seen.add(canonical)
            result = solve_decision(info, iterations=args.iterations)
            target = policy_target(result)
            value = sum(
                probability * next(item["ev"] for item in result["actions"] if item["action"] == action)
                for action, probability in zip(ACTION_ORDER, target)
                if probability > 0
            )
            teacher_row = {
                    "kind": "root-teacher-v1",
                    "sampleId": f"{args.seed}-{digest}",
                    "teacherVersion": f"mccfr-{args.regret_mode}-v1",
                    "teacherIterations": result["trainingIterations"],
                    "teacherRootRegret": result["rootRegret"],
                    "info": info,
                    "canonical": canonical,
                    "key": digest,
                    "features": encode_info_features(info),
                    "policyTarget": target,
                    "valueTarget": value,
                    "result": result,
                }
            validation_group = row.get("validationGroup")
            if isinstance(validation_group, str) and validation_group:
                teacher_row["validationGroup"] = validation_group
            output.write(
                json.dumps(teacher_row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            output.flush()
            completed += 1
            if completed % args.checkpoint_every == 0:
                os.fsync(output.fileno())
            print(
                f"[{completed}/{len(source)}] {digest} {info['street']} value={value:.3f}",
                flush=True,
            )
    finally:
        output.flush()
        os.fsync(output.fileno())
        output.close()
        shutdown_process_pool()
    print(
        f"Teacher 数据已写入 {dataset_path}，本次新增 {completed}，"
        f"累计 {initial_count + completed} 个唯一节点",
        flush=True,
    )


def split_indices(count: int, validation_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, min(count - 1, round(count * validation_fraction))) if count > 1 else 1
    return indices[validation_count:], indices[:validation_count]


def split_row_indices(
    rows: list[dict[str, Any]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    groups = [row.get("validationGroup") for row in rows]
    if not all(isinstance(group, str) and group for group in groups):
        return split_indices(len(rows), validation_fraction, seed)
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        return split_indices(len(rows), validation_fraction, seed)
    random.Random(seed).shuffle(unique_groups)
    validation_group_count = max(
        1,
        min(len(unique_groups) - 1, round(len(unique_groups) * validation_fraction)),
    )
    validation_groups = set(unique_groups[:validation_group_count])
    train = [index for index, group in enumerate(groups) if group not in validation_groups]
    validation = [index for index, group in enumerate(groups) if group in validation_groups]
    return train, validation


def normalization(features: list[list[float]], train_indices: list[int]) -> tuple[list[float], list[float]]:
    source = train_indices or list(range(len(features)))
    mean = [sum(features[index][column] for index in source) / len(source) for column in range(FEATURE_SIZE)]
    scale = []
    for column in range(FEATURE_SIZE):
        variance = sum((features[index][column] - mean[column]) ** 2 for index in source) / max(1, len(source) - 1)
        scale.append(max(1e-4, math.sqrt(variance)))
    return mean, scale


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [value / total for value in weights]


def train_linear(
    features: list[list[float]],
    targets: list[list[float]],
    train_indices: list[int],
    mean: list[float],
    scale: list[float],
    *,
    epochs: int,
    learning_rate: float,
    policy: bool,
    seed: int,
) -> dict[str, Any]:
    output_size = len(targets[0])
    rng = random.Random(seed)
    weights = [[rng.uniform(-0.005, 0.005) for _ in range(FEATURE_SIZE)] for _ in range(output_size)]
    bias = [0.0] * output_size
    source = train_indices or list(range(len(features)))
    for epoch in range(epochs):
        order = source[:]
        rng.shuffle(order)
        rate = learning_rate / math.sqrt(1 + epoch * 0.02)
        for index in order:
            row = [(value - offset) / divisor for value, offset, divisor in zip(features[index], mean, scale)]
            output = [sum(weight * value for weight, value in zip(line, row)) + offset for line, offset in zip(weights, bias)]
            if policy:
                predicted = softmax(output)
                gradient = [predicted_value - target for predicted_value, target in zip(predicted, targets[index])]
            else:
                gradient = [2 * (predicted_value - target) for predicted_value, target in zip(output, targets[index])]
            for output_index in range(output_size):
                bias[output_index] -= rate * gradient[output_index]
                for column, value in enumerate(row):
                    weights[output_index][column] -= rate * gradient[output_index] * value
    return {
        "inputSize": FEATURE_SIZE,
        "outputSize": output_size,
        "normalization": {"mean": mean, "scale": scale},
        "layers": [{"weights": weights, "bias": bias, "activation": "linear"}],
    }


def train_torch(
    features: list[list[float]],
    targets: list[list[float]],
    train_indices: list[int],
    mean: list[float],
    scale: list[float],
    *,
    epochs: int,
    learning_rate: float,
    hidden: int,
    batch_size: int,
    policy: bool,
    seed: int,
    device: str,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is unavailable")
    if device == "cpu":
        torch.set_num_threads(1)
    x = torch.tensor(features, dtype=torch.float32)
    x = (x - torch.tensor(mean)) / torch.tensor(scale)
    y = torch.tensor(targets, dtype=torch.float32)
    source = train_indices or list(range(len(features)))
    dataset = torch.utils.data.TensorDataset(x[source], y[source])
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device == "cuda",
    )
    layers: list[Any] = []
    if hidden > 0:
        layers.extend([torch.nn.Linear(FEATURE_SIZE, hidden), torch.nn.ReLU()])
        layers.append(torch.nn.Linear(hidden, y.shape[1]))
    else:
        layers.append(torch.nn.Linear(FEATURE_SIZE, y.shape[1]))
    model = torch.nn.Sequential(*layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = (
                -(batch_y * torch.log_softmax(prediction, dim=1)).sum(dim=1).mean()
                if policy
                else torch.nn.functional.smooth_l1_loss(prediction, batch_y)
            )
            loss.backward()
            optimizer.step()
    exported: list[dict[str, Any]] = []
    modules = list(model)
    for index, module in enumerate(modules):
        if not isinstance(module, torch.nn.Linear):
            continue
        activation = "relu" if index + 1 < len(modules) and isinstance(modules[index + 1], torch.nn.ReLU) else "linear"
        exported.append(
            {
                "weights": module.weight.detach().cpu().tolist(),
                "bias": module.bias.detach().cpu().tolist(),
                "activation": activation,
            }
        )
    return {
        "inputSize": FEATURE_SIZE,
        "outputSize": len(targets[0]),
        "normalization": {"mean": mean, "scale": scale},
        "layers": exported,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def evaluate_models(
    policy_network: dict[str, Any],
    value_network: dict[str, Any],
    features: list[list[float]],
    policy_targets: list[list[float]],
    value_targets: list[float],
    validation_indices: list[int],
    target_mean: float,
    target_scale: float,
) -> dict[str, float]:
    from backend.blueprint import DenseNetwork

    correct = 0
    l1_values: list[float] = []
    errors: list[float] = []
    policy_model = DenseNetwork(policy_network)
    value_model = DenseNetwork(value_network)
    validation_features = [features[index] for index in validation_indices]
    policy_outputs = policy_model.predict_many(validation_features)
    value_outputs = value_model.predict_many(validation_features)
    for index, policy_output, value_output in zip(
        validation_indices,
        policy_outputs,
        value_outputs,
    ):
        predicted_policy = softmax(policy_output)
        correct += int(max(range(4), key=predicted_policy.__getitem__) == max(range(4), key=policy_targets[index].__getitem__))
        l1_values.append(sum(abs(left - right) for left, right in zip(predicted_policy, policy_targets[index])))
        predicted_value = value_output[0] * target_scale + target_mean
        errors.append(abs(predicted_value - value_targets[index]))
    return {
        "validationSamples": len(validation_indices),
        "policyTopActionAccuracy": correct / len(validation_indices),
        "policyMedianL1": percentile(l1_values, 0.5),
        "valueMae": sum(errors) / len(errors),
        "valueP99": percentile(errors, 0.99),
        "valueRmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
    }


def train_bundle(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.hidden < 0:
        raise ValueError("epochs/batch-size must be positive and hidden cannot be negative")
    rows = [row for row in read_jsonl(Path(args.dataset)) if row.get("kind") == "root-teacher-v1"]
    if not rows:
        raise ValueError("数据集中没有 root-teacher-v1 样本")
    features = [list(map(float, row["features"])) for row in rows]
    if any(len(row) != FEATURE_SIZE for row in features):
        raise ValueError(f"训练特征必须为 {FEATURE_SIZE} 维")
    policies = [list(map(float, row["policyTarget"])) for row in rows]
    values = [float(row["valueTarget"]) for row in rows]
    supported_streets = sorted(
        {str(row["info"]["street"]) for row in rows},
        key=("preflop", "flop", "turn", "river").index,
    )
    train_indices, validation_indices = split_row_indices(
        rows,
        args.validation_fraction,
        args.seed,
    )
    mean, scale = normalization(features, train_indices)
    target_source = train_indices or list(range(len(rows)))
    target_mean = sum(values[index] for index in target_source) / len(target_source)
    target_scale = max(
        1e-4,
        math.sqrt(sum((values[index] - target_mean) ** 2 for index in target_source) / max(1, len(target_source) - 1)),
    )
    normalized_values = [[(value - target_mean) / target_scale] for value in values]

    engine = args.engine
    if engine == "auto":
        try:
            import torch  # noqa: F401

            engine = "torch"
        except ImportError:
            engine = "linear"
    trainer = train_torch if engine == "torch" else train_linear
    common = {
        "features": features,
        "train_indices": train_indices,
        "mean": mean,
        "scale": scale,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }
    if trainer is train_torch:
        policy_network = trainer(targets=policies, policy=True, hidden=args.hidden, batch_size=args.batch_size, device=args.device, **common)
        value_network = trainer(targets=normalized_values, policy=False, hidden=args.hidden, batch_size=args.batch_size, device=args.device, **common)
    else:
        policy_network = trainer(targets=policies, policy=True, **common)
        value_network = trainer(targets=normalized_values, policy=False, **common)
    metrics = evaluate_models(
        policy_network,
        value_network,
        features,
        policies,
        values,
        validation_indices,
        target_mean,
        target_scale,
    )
    policy_valid = (
        metrics["validationSamples"] >= args.min_validation_samples
        and metrics["policyTopActionAccuracy"] >= args.min_policy_accuracy
        and metrics["policyMedianL1"] <= args.max_policy_l1
    )
    value_valid = (
        metrics["validationSamples"] >= args.min_validation_samples
        and metrics["valueMae"] <= args.max_value_mae
        and metrics["valueP99"] <= args.max_value_p99
    )
    exact: dict[str, Any] = {}
    if args.base_bundle:
        base_path = Path(args.base_bundle)
        with open_text(base_path, "rt") as handle:
            base_bundle = json.load(handle)
        validate_bundle(base_bundle)
        exact.update(base_bundle.get("exactSolutions", {}))
    for row in rows:
        existing = exact.get(row["key"])
        iterations = int(row["result"].get("iterations", row["teacherIterations"]))
        if existing is None or int(existing["iterations"]) < iterations:
            exact[row["key"]] = {
                "canonical": row["canonical"],
                "iterations": iterations,
                "result": row["result"],
            }
    bundle = {
        "format": BLUEPRINT_FORMAT,
        "schemaVersion": SCHEMA_VERSION,
        "featureVersion": FEATURE_VERSION,
        "abstractionVersion": ABSTRACTION_VERSION,
        "rulesHash": RULES_HASH,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "baseFingerprint": (
            base_bundle["fingerprint"] if args.base_bundle else None
        ),
        "seed": args.seed,
        "validated": policy_valid and value_valid,
        "metrics": metrics,
        "exactSolutions": exact,
        "policyModel": {
            "kind": "acting-player-policy-v1",
            "validated": policy_valid,
            "trainingSamples": len(train_indices),
            "supportedStreets": supported_streets,
            "network": policy_network,
        },
        # This predicts a root acting-player value and is intentionally not
        # advertised as a safe depth-cut leaf model.
        "valueModel": {
            "kind": "acting-player-state-value-v1",
            "validated": value_valid,
            "targetMean": target_mean,
            "targetScale": target_scale,
            "trainingSamples": len(train_indices),
            "supportedStreets": supported_streets,
            "network": value_network,
        },
        "leafModel": None,
    }
    fingerprint = write_bundle(args.output, bundle)
    print(json.dumps({"output": str(Path(args.output).resolve()), "fingerprint": fingerprint, "validated": bundle["validated"], "metrics": metrics}, ensure_ascii=False, indent=2))


def inspect_bundle(args: argparse.Namespace) -> None:
    path = Path(args.path)
    with open_text(path, "rt") as handle:
        bundle = json.load(handle)
    validate_bundle(bundle)
    summary = {
        "path": str(path.resolve()),
        "fingerprint": bundle["fingerprint"],
        "validated": bundle.get("validated", False),
        "exactSolutions": len(bundle.get("exactSolutions", {})),
        "policyModel": bundle.get("policyModel", {}).get("validated", False),
        "valueModel": bundle.get("valueModel", {}).get("validated", False),
        "leafModel": bool(bundle.get("leafModel")),
        "metrics": bundle.get("metrics", {}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Texas Lab 离线 Blueprint 构建工具")
    commands = root.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="使用当前完整 MCCFR 求解器生成 Teacher 数据")
    collect.add_argument("--input", required=True, help="每行一个 info 或 {info: ...} 的 JSONL/JSONL.GZ")
    collect.add_argument("--dataset", required=True)
    collect.add_argument(
        "--iterations",
        type=int,
        default=20_000,
        help="每个 Teacher 节点的求解预算（默认 20000，与 5%% 工程误差配置一致）",
    )
    collect.add_argument("--seed", type=int, default=20260731)
    collect.add_argument("--append", action="store_true", help="去重后增量合并到现有 Teacher 数据集")
    collect.add_argument(
        "--max-new-samples",
        type=int,
        default=0,
        help="本次最多新增多少个节点；0 表示处理全部，用于分批发布和断点续训",
    )
    collect.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="每多少个完成节点执行一次磁盘同步（默认 1）",
    )
    collect.add_argument(
        "--single-replica",
        action="store_true",
        help="用于外层节点并行：单进程执行与默认双副本相同的总训练/评估量",
    )
    collect.add_argument(
        "--regret-mode",
        choices=("linear", "rmplus"),
        default="rmplus",
        help="Teacher 默认匹配在线 RM+；linear 保留给显式实验",
    )
    collect.set_defaults(run=collect_teacher)

    train = commands.add_parser("train", help="训练 portable policy/value 模型并发布 bundle")
    train.add_argument("--dataset", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--base-bundle", help="合并已有 bundle 的精确解条目")
    train.add_argument("--engine", choices=("auto", "torch", "linear"), default="auto")
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--epochs", type=int, default=600)
    train.add_argument("--hidden", type=int, default=64)
    train.add_argument("--batch-size", type=int, default=2048)
    train.add_argument("--learning-rate", type=float, default=0.003)
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=20260731)
    train.add_argument("--min-validation-samples", type=int, default=100)
    train.add_argument("--min-policy-accuracy", type=float, default=0.95)
    # L1=0.10 corresponds to 5% total-variation error.  The value head is a
    # warm-start diagnostic rather than a safe tree cutoff, so its absolute
    # gates intentionally spend the project's 5% engineering error budget.
    train.add_argument("--max-policy-l1", type=float, default=0.10)
    train.add_argument("--max-value-mae", type=float, default=0.50)
    train.add_argument("--max-value-p99", type=float, default=2.50)
    train.set_defaults(run=train_bundle)

    inspect = commands.add_parser("inspect", help="校验并显示 bundle manifest")
    inspect.add_argument("path")
    inspect.set_defaults(run=inspect_bundle)
    return root


def main() -> None:
    args = parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
