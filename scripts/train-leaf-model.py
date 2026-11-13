from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.value_model import (  # noqa: E402
    FEATURE_SIZE,
    HAND_COUNT,
    MODEL_TYPE,
    SEAT_COUNT,
    STREETS,
    LeafTrainingSample,
    build_leaf_features,
    quality_gate_failures,
    read_leaf_dataset,
    write_leaf_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export the belief-conditioned Texas Lab leaf value model.",
    )
    parser.add_argument("datasets", nargs="+", type=Path, help="Validated .jsonl.gz teacher shards")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden", default="128,64", help="Comma-separated hidden widths")
    parser.add_argument("--hands-per-seat", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=100_000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def stable_validation_split(sample_id: str, fraction: float) -> bool:
    digest = hashlib.blake2b(sample_id.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big") / (1 << 64)
    return value < fraction


def selected_targets(sample: LeafTrainingSample, rng: random.Random, hands_per_seat: int) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    for seat in range(SEAT_COUNT):
        supported = [
            hand_index
            for hand_index in range(HAND_COUNT)
            if sample.target_counts[seat][hand_index] > 0
        ]
        if len(supported) <= hands_per_seat:
            chosen = supported
        else:
            # Preserve one high-confidence target and randomly cover the rest.
            best = max(supported, key=lambda index: sample.target_counts[seat][index])
            remainder = [index for index in supported if index != best]
            chosen = [best, *rng.sample(remainder, hands_per_seat - 1)]
        selected.extend((seat, hand_index) for hand_index in chosen)
    return selected


def support_observation(sample: LeafTrainingSample) -> dict[str, Any]:
    state = sample.model_input.public_state
    total_chips = sum(player["stack"] + player["totalCommitted"] for player in state["players"])
    entropies = []
    ess_values = []
    supports = []
    for hand_range in sample.model_input.ranges:
        positive = [probability for probability in hand_range if probability > 0]
        entropies.append(-sum(probability * math.log(probability) for probability in positive) / math.log(HAND_COUNT))
        ess_values.append(1 / sum(probability * probability for probability in positive) / HAND_COUNT)
        supports.append(len(positive))
    return {
        "street": state["street"],
        "activeCount": sum(not player["folded"] for player in state["players"]),
        "potFraction": state["pot"] / total_chips,
        "currentBetFraction": state["currentBet"] / total_chips,
        "totalChipsBb": total_chips,
        "stackFraction": [player["stack"] / total_chips for player in state["players"]],
        "rangeEntropyNormalized": entropies,
        "rangeEssFraction": ess_values,
        "supportedCombos": supports,
        "continuationProfile": list(sample.model_input.continuation_ids),
    }


def unique_training_versions(headers: list[dict[str, Any]]) -> tuple[str, str]:
    blueprint_versions = {str(header.get("blueprintVersion", "")) for header in headers}
    teacher_versions = {str(header.get("teacherVersion", "")) for header in headers}
    if len(blueprint_versions) != 1 or "" in blueprint_versions:
        raise ValueError("datasets must contain exactly one blueprintVersion")
    if len(teacher_versions) != 1 or "" in teacher_versions:
        raise ValueError("datasets must contain exactly one teacherVersion")
    return next(iter(blueprint_versions)), next(iter(teacher_versions))


def load_examples(args: argparse.Namespace) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    rng = random.Random(args.seed)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    headers: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    seen_headers: set[str] = set()
    sources = []
    for dataset_path in args.datasets:
        header, samples = read_leaf_dataset(dataset_path)
        sources.append(samples)
        header_key = json.dumps(header, ensure_ascii=False, sort_keys=True)
        if header_key not in seen_headers:
            headers.append(header)
            seen_headers.add(header_key)
    unique_training_versions(headers)
    for samples in sources:
        for sample in samples:
            support_rows.append(support_observation(sample))
            validation_sample = stable_validation_split(sample.sample_id, args.validation_fraction)
            destination = validation if validation_sample else train
            total_chips = sum(
                player["stack"] + player["totalCommitted"]
                for player in sample.model_input.public_state["players"]
            )
            active_count = sum(not player["folded"] for player in sample.model_input.public_state["players"])
            group = f"{sample.model_input.public_state['street']}:{active_count}"
            for seat, hand_index in selected_targets(sample, rng, args.hands_per_seat):
                destination.append(
                    {
                        "features": build_leaf_features(sample.model_input, seat, hand_index),
                        "target": sample.final_stack_values[seat][hand_index] / total_chips,
                        "teacherStderr": sample.target_stderr[seat][hand_index] / total_chips,
                        "totalChips": total_chips,
                        "sampleId": sample.sample_id,
                        "group": group,
                    }
                )
                if len(train) + len(validation) >= args.max_examples:
                    return train, validation, headers, support_rows
    return train, validation, headers, support_rows


def quantile_interval(values: list[float], low_fraction: float = 0.005, high_fraction: float = 0.995) -> list[float]:
    if not values:
        raise ValueError("cannot construct support from no observations")
    ordered = sorted(values)
    low_index = min(len(ordered) - 1, int((len(ordered) - 1) * low_fraction))
    high_index = min(len(ordered) - 1, math.ceil((len(ordered) - 1) * high_fraction))
    return [ordered[low_index], ordered[high_index]]


def model_support(observations: list[dict[str, Any]]) -> dict[str, Any]:
    def flattened(name: str) -> list[float]:
        result: list[float] = []
        for observation in observations:
            value = observation[name]
            result.extend(float(item) for item in value) if isinstance(value, list) else result.append(float(value))
        return result

    supported_counts = [
        int(count)
        for observation in observations
        for count in observation["supportedCombos"]
    ]
    return {
        "streets": [street for street in STREETS if any(row["street"] == street for row in observations)],
        "activeCounts": sorted({int(row["activeCount"]) for row in observations}),
        "continuationProfiles": [
            list(profile)
            for profile in sorted(
                {tuple(row["continuationProfile"]) for row in observations}
            )
        ],
        "minimumSupportedCombos": min(supported_counts),
        "maxInferenceUncertaintyBb": 1.00,
        "potFraction": quantile_interval(flattened("potFraction")),
        "currentBetFraction": quantile_interval(flattened("currentBetFraction")),
        "totalChipsBb": quantile_interval(flattened("totalChipsBb")),
        "stackFraction": quantile_interval(flattened("stackFraction")),
        "rangeEntropyNormalized": quantile_interval(flattened("rangeEntropyNormalized")),
        "rangeEssFraction": quantile_interval(flattened("rangeEssFraction")),
    }


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.hands_per_seat < 1 or args.max_examples < 1:
        raise SystemExit("epochs, batch-size, hands-per-seat and max-examples must be positive")
    if not 0.05 <= args.validation_fraction <= 0.5:
        raise SystemExit("validation-fraction must be between 0.05 and 0.5")
    hidden = [int(value) for value in args.hidden.split(",") if value.strip()]
    if not hidden or any(width < 2 or width > 4096 for width in hidden):
        raise SystemExit("hidden widths must be between 2 and 4096")

    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise SystemExit(
            "PyTorch is required only for offline training. Install requirements-gpu.txt or a CPU PyTorch build."
        ) from error

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_rows, validation_rows, headers, support_rows = load_examples(args)
    blueprint_fingerprint, teacher_version = unique_training_versions(headers)
    if not train_rows or not validation_rows:
        raise SystemExit("dataset split produced no training or validation examples")
    train_x = torch.tensor([row["features"] for row in train_rows], dtype=torch.float32)
    train_y = torch.tensor([row["target"] for row in train_rows], dtype=torch.float32)
    train_teacher_sigma = torch.tensor([row["teacherStderr"] for row in train_rows], dtype=torch.float32)
    validation_x = torch.tensor([row["features"] for row in validation_rows], dtype=torch.float32)
    validation_y = torch.tensor([row["target"] for row in validation_rows], dtype=torch.float32)
    validation_teacher_sigma = torch.tensor(
        [row["teacherStderr"] for row in validation_rows], dtype=torch.float32
    )
    validation_total = torch.tensor([row["totalChips"] for row in validation_rows], dtype=torch.float32)

    feature_mean = train_x.mean(dim=0)
    feature_scale = train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_x = (train_x - feature_mean) / feature_scale
    validation_x = (validation_x - feature_mean) / feature_scale

    modules: list[nn.Module] = []
    width = FEATURE_SIZE
    for next_width in hidden:
        modules.extend((nn.Linear(width, next_width), nn.ReLU()))
        width = next_width
    modules.append(nn.Linear(width, 2))
    model = nn.Sequential(*modules).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y, train_teacher_sigma),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device == "cuda",
    )

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        examples = 0
        for batch_x, batch_y, teacher_sigma in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            teacher_sigma = teacher_sigma.to(device, non_blocking=True)
            raw = model(batch_x)
            mean = torch.sigmoid(raw[:, 0])
            predicted_sigma = functional.softplus(raw[:, 1]) + 1e-6
            variance = predicted_sigma.square() + teacher_sigma.square() + 1e-8
            residual = batch_y - mean
            gaussian_nll = 0.5 * (residual.square() / variance + variance.log())
            robust_mean = functional.smooth_l1_loss(mean, batch_y, reduction="none", beta=0.01)
            loss = (gaussian_nll + robust_mean).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_x)
            examples += len(batch_x)
        print(f"epoch={epoch + 1} loss={loss_sum / max(1, examples):.6f}", flush=True)

    model.eval()
    with torch.inference_mode():
        raw = model(validation_x.to(device)).cpu()
        predicted_fraction = torch.sigmoid(raw[:, 0])
        predicted_sigma_fraction = functional.softplus(raw[:, 1]) + 1e-6
        absolute_error_bb = (predicted_fraction - validation_y).abs() * validation_total
        uncertainty_bb = predicted_sigma_fraction * validation_total
        combined_sigma_bb = (
            predicted_sigma_fraction.square() + validation_teacher_sigma.square()
        ).sqrt() * validation_total
        coverage = (absolute_error_bb <= 1.96 * combined_sigma_bb).float().mean().item()

    group_errors: dict[str, list[float]] = {}
    for row, error in zip(validation_rows, absolute_error_bb.tolist()):
        group_errors.setdefault(row["group"], []).append(error)
    group_mae = {
        group: sum(errors) / len(errors)
        for group, errors in group_errors.items()
    }
    metrics: dict[str, Any] = {
        "trainingExamples": len(train_rows),
        "trainingPublicStates": len({row["sampleId"] for row in train_rows}),
        "heldoutExamples": len(validation_rows),
        "heldoutPublicStates": len({row["sampleId"] for row in validation_rows}),
        "maeBb": float(absolute_error_bb.mean()),
        "p99AbsoluteErrorBb": float(torch.quantile(absolute_error_bb, 0.99)),
        "coverage95": float(coverage),
        "meanUncertaintyBb": float(uncertainty_bb.mean()),
        "maxGroupMaeBb": max(group_mae.values()),
        "groupMaeBb": group_mae,
        # Only the production validator may replace these placeholders and
        # attach an approval attestation. Training always emits a candidate.
        "decisiveNodes": 0,
        "actionAgreement": 0.0,
        "strategyL1Median": 1.0,
        "strategyL1P95": 1.0,
        "actionEvRelativeErrorP95": 1.0,
        "oodFallbackRecall": 0.0,
        "leafActivatedNodes": 0,
        "leafActivationRate": 0.0,
    }

    exported_layers: list[dict[str, Any]] = []
    for index, module in enumerate(model):
        if not isinstance(module, nn.Linear):
            continue
        next_module = model[index + 1] if index + 1 < len(model) else None
        activation = "relu" if isinstance(next_module, nn.ReLU) else "linear"
        exported_layers.append(
            {
                "weights": module.weight.detach().cpu().tolist(),
                "bias": module.bias.detach().cpu().tolist(),
                "activation": activation,
            }
        )
    network = {
        "modelType": MODEL_TYPE,
        "inputSize": FEATURE_SIZE,
        "outputSize": 2,
        "normalization": {
            "mean": feature_mean.tolist(),
            "scale": feature_scale.tolist(),
        },
        "layers": exported_layers,
    }
    training = {
        "algorithm": "heteroscedastic-dense-v1",
        "seed": args.seed,
        "epochs": args.epochs,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "hidden": hidden,
        "datasetHeaders": headers,
        "blueprintFingerprint": blueprint_fingerprint,
        "teacherVersion": teacher_version,
        "target": "conditional-final-stack-by-seat-and-hand",
    }
    manifest_path = write_leaf_artifact(
        args.output_dir,
        network,
        metrics,
        training,
        support=model_support(support_rows),
    )
    failures = quality_gate_failures(metrics)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "approved": False,
                "candidate": True,
                "metrics": metrics,
                "gateFailures": [
                    *failures,
                    "production-approval-attestation-missing",
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
