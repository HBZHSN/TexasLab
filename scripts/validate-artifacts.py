from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import statistics
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.blueprint import get_blueprint_runtime, reset_blueprint_runtime  # noqa: E402
from backend.leaf_runtime import reset_leaf_runtime  # noqa: E402
from backend.schema import normalize_solve_info  # noqa: E402
from backend.solver import shutdown_process_pool, solve_decision  # noqa: E402
from backend.value_model import (  # noqa: E402
    PRODUCTION_CONTINUATION_CONFIG,
    PRODUCTION_CUTOFF_CONFIG,
    PRODUCTION_SOLVER_CONFIG,
    PRODUCTION_VALIDATION_CONFIG,
    build_approval_attestation,
    load_manifest,
    offline_quality_gate_failures,
    quality_gate_failures,
    write_leaf_artifact,
)

VALIDATION_ENV_KEYS = (
    "TEXAS_LAB_BLUEPRINT_PATH",
    "TEXAS_LAB_BLUEPRINT_MODE",
    "TEXAS_LAB_REGRET_MODE",
    "TEXAS_LAB_ADAPTIVE_TRAINING",
    "TEXAS_LAB_TRAIN_CHUNK_SIZE",
    "TEXAS_LAB_TRAIN_MIN_ITERATIONS",
    "TEXAS_LAB_PREFLOP_TRAIN_MIN_ITERATIONS",
    "TEXAS_LAB_TRAIN_EV_TOLERANCE",
    "TEXAS_LAB_TRAIN_STABLE_CHECKS",
    "TEXAS_LAB_TRAIN_DOMINANCE",
    "TEXAS_LAB_TRAIN_DOMINANCE_MARGIN",
    "TEXAS_LAB_TRAIN_DOMINANCE_CHECKS",
    "TEXAS_LAB_TRAIN_DOMINANCE_DRIFT",
    "TEXAS_LAB_TRAIN_BEST_FREQUENCY",
    "TEXAS_LAB_PREFLOP_DOMINATED_EVAL_TOTAL",
    "TEXAS_LAB_ADAPTIVE_EVALUATION",
    "TEXAS_LAB_ERROR_BUDGET",
    "TEXAS_LAB_BLUEPRINT_PRIOR_WEIGHT",
    "TEXAS_LAB_LEAF_BELIEF_REFRESH",
    "TEXAS_LAB_LEAF_CUTOFF_MODE",
    "TEXAS_LAB_LEAF_DEPTH",
    "TEXAS_LAB_LEAF_CONTINUATIONS",
    "TEXAS_LAB_LEAF_MODEL_MODE",
    "TEXAS_LAB_LEAF_MODEL_MANIFEST",
    "TEXAS_LAB_LEAF_AB_TOKEN",
)


@dataclass(frozen=True)
class ValidationCandidate:
    manifest: Path
    token: str
    artifact_fingerprint: str
    payload_fingerprint: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed production A/B validation and attest a leaf artifact.",
    )
    parser.add_argument("--holdout", type=Path, required=True, help="JSONL/JSONL.GZ info states")
    parser.add_argument("--blueprint", type=Path, required=True)
    parser.add_argument("--leaf-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--publish-output-dir",
        type=Path,
        help="Atomically publish an attested artifact only when every gate passes",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    handle = (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix.lower() == ".gz"
        else path.open("r", encoding="utf-8")
    )
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(
                {
                    "info": normalize_solve_info(row.get("info", row)),
                    "expectLeafFallback": bool(row.get("expectLeafFallback", False)),
                }
            )
    if not rows:
        raise ValueError("holdout dataset is empty")
    return rows


def action_map(result: dict[str, Any], field: str) -> dict[str, float]:
    return {row["action"]: float(row[field]) for row in result["actions"]}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(0, index)]


def compute_integration_metrics(
    baseline: list[dict[str, Any]],
    accelerated: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    decisive_gap: float,
) -> dict[str, float | int]:
    if not baseline or len(baseline) != len(accelerated) or len(rows) != len(baseline):
        raise ValueError("baseline, accelerated and rows must have equal non-zero lengths")
    decisive = 0
    agreements = 0
    l1_values: list[float] = []
    relative_ev_errors: list[float] = []
    iid_nodes = 0
    leaf_activated_nodes = 0
    ood_total = 0
    ood_fallbacks = 0
    for reference, candidate, row in zip(baseline, accelerated, rows):
        reference_ev = action_map(reference, "ev")
        candidate_ev = action_map(candidate, "ev")
        common = sorted(set(reference_ev) & set(candidate_ev))
        if not common:
            raise ValueError("A/B result has no common legal action")
        ordered = sorted((reference_ev[action], action) for action in common)
        reference_best = ordered[-1][1]
        gap = ordered[-1][0] - ordered[-2][0] if len(ordered) > 1 else float("inf")
        info = row.get("info", {})
        hero = info.get("hero", {}) if isinstance(info, dict) else {}
        decision_scale = max(
            1.0,
            float(info.get("pot", 0.0)) if isinstance(info, dict) else 0.0,
        )
        absolute_scale = max(
            decision_scale,
            float(hero.get("stack", 0.0)) + float(hero.get("totalCommitted", 0.0))
            if isinstance(hero, dict)
            else 0.0,
        )
        relative_ev_errors.extend(
            abs(candidate_ev[action] - reference_ev[action]) / absolute_scale
            for action in common
        )
        candidate_best = max(common, key=candidate_ev.__getitem__)
        if gap > decisive_gap * decision_scale:
            decisive += 1
            agreements += int(candidate_best == reference_best)

        reference_policy = action_map(reference, "frequency")
        candidate_policy = action_map(candidate, "frequency")
        all_actions = set(reference_policy) | set(candidate_policy)
        l1_values.append(
            sum(
                abs(reference_policy.get(action, 0.0) - candidate_policy.get(action, 0.0))
                for action in all_actions
            )
        )
        evaluation_hits = int(candidate.get("leafEvaluationCutoffHits", 0))
        if row["expectLeafFallback"]:
            ood_total += 1
            ood_fallbacks += int(evaluation_hits == 0)
        else:
            iid_nodes += 1
            leaf_activated_nodes += int(evaluation_hits > 0)
    return {
        "decisiveNodes": decisive,
        "actionAgreement": agreements / decisive if decisive else 0.0,
        "strategyL1Median": statistics.median(l1_values),
        "strategyL1P95": percentile(l1_values, 0.95),
        "actionEvRelativeErrorP95": percentile(relative_ev_errors, 0.95),
        "oodFallbackRecall": ood_fallbacks / ood_total if ood_total else 0.0,
        "iidNodes": iid_nodes,
        "leafActivatedNodes": leaf_activated_nodes,
        "leafActivationRate": leaf_activated_nodes / iid_nodes if iid_nodes else 0.0,
        "oodNodes": ood_total,
    }


def _configure_production_environment(
    *,
    blueprint: Path,
    leaf_manifest: Path | None,
    ab_token: str | None,
) -> None:
    os.environ["TEXAS_LAB_BLUEPRINT_PATH"] = str(blueprint.resolve())
    os.environ["TEXAS_LAB_BLUEPRINT_MODE"] = "required"
    os.environ["TEXAS_LAB_REGRET_MODE"] = str(PRODUCTION_SOLVER_CONFIG["regretMode"])
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
    os.environ["TEXAS_LAB_ERROR_BUDGET"] = str(
        PRODUCTION_SOLVER_CONFIG["errorBudgetFraction"]
    )
    os.environ["TEXAS_LAB_BLUEPRINT_PRIOR_WEIGHT"] = str(
        PRODUCTION_SOLVER_CONFIG["blueprintPriorWeight"]
    )
    os.environ["TEXAS_LAB_LEAF_BELIEF_REFRESH"] = str(
        PRODUCTION_SOLVER_CONFIG["leafBeliefRefresh"]
    )
    os.environ["TEXAS_LAB_LEAF_CUTOFF_MODE"] = str(PRODUCTION_CUTOFF_CONFIG["mode"])
    os.environ.pop("TEXAS_LAB_LEAF_DEPTH", None)
    os.environ["TEXAS_LAB_LEAF_CONTINUATIONS"] = ",".join(
        PRODUCTION_CONTINUATION_CONFIG["continuations"]
    )
    if leaf_manifest is None:
        os.environ["TEXAS_LAB_LEAF_MODEL_MODE"] = "off"
        os.environ.pop("TEXAS_LAB_LEAF_MODEL_MANIFEST", None)
        os.environ.pop("TEXAS_LAB_LEAF_AB_TOKEN", None)
    else:
        os.environ["TEXAS_LAB_LEAF_MODEL_MODE"] = "ab" if ab_token else "required"
        os.environ["TEXAS_LAB_LEAF_MODEL_MANIFEST"] = str(leaf_manifest.resolve())
        if ab_token is None:
            os.environ.pop("TEXAS_LAB_LEAF_AB_TOKEN", None)
        else:
            os.environ["TEXAS_LAB_LEAF_AB_TOKEN"] = ab_token


def solve_pass(
    rows: list[dict[str, Any]],
    *,
    blueprint: Path,
    blueprint_fingerprint: str,
    leaf_manifest: Path | None,
    expected_leaf_fingerprint: str | None = None,
    ab_token: str | None = None,
) -> list[dict[str, Any]]:
    previous_environment = {name: os.environ.get(name) for name in VALIDATION_ENV_KEYS}
    try:
        _configure_production_environment(
            blueprint=blueprint,
            leaf_manifest=leaf_manifest,
            ab_token=ab_token,
        )
        reset_blueprint_runtime()
        reset_leaf_runtime()
        shutdown_process_pool()
        blueprint_runtime = get_blueprint_runtime()
        if (
            blueprint_runtime.fingerprint != blueprint_fingerprint
            or not blueprint_runtime.has_valid_policy
        ):
            raise ValueError("production Blueprint is inactive or has the wrong fingerprint")
        results: list[dict[str, Any]] = []
        for index, row in enumerate(rows, 1):
            result = solve_decision(
                row["info"],
                iterations=int(PRODUCTION_SOLVER_CONFIG["iterations"]),
            )
            if result.get("blueprintFingerprint") != blueprint_fingerprint:
                raise ValueError("solver used a different Blueprint fingerprint")
            if leaf_manifest is not None:
                if result.get("leafModelFingerprint") != expected_leaf_fingerprint:
                    raise ValueError("solver used a different leaf artifact fingerprint")
                if int(result.get("leafCutoffWorkers", 0)) <= 0:
                    raise ValueError("leaf runtime was not active during production validation")
            results.append(result)
            print(
                f"[{index}/{len(rows)}] leaf={'off' if leaf_manifest is None else 'on'} "
                f"best={max(result['actions'], key=lambda action: action['ev'])['action']}",
                file=sys.stderr,
                flush=True,
            )
        return results
    finally:
        shutdown_process_pool()
        for name, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
        reset_blueprint_runtime()
        reset_leaf_runtime()


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def validation_manifest(
    source: Path,
    temporary_directory: Path,
    *,
    blueprint_fingerprint: str | None = None,
) -> ValidationCandidate:
    """Create a token-bound, temporary candidate without granting approval."""
    manifest, payload = load_manifest(source)
    expected_blueprint = blueprint_fingerprint or payload["training"]["blueprintFingerprint"]
    if payload["training"]["blueprintFingerprint"] != expected_blueprint:
        raise ValueError("candidate was trained against a different Blueprint")
    remaining = offline_quality_gate_failures(manifest["metrics"])
    if remaining:
        raise ValueError(
            "leaf artifact fails offline quality gates and cannot enter A/B: "
            + "; ".join(remaining)
        )
    token = secrets.token_urlsafe(32)
    candidate_manifest = write_leaf_artifact(
        temporary_directory,
        payload["network"],
        manifest["metrics"],
        payload["training"],
        support=payload["support"],
        ab_candidate_token=token,
    )
    candidate, _ = load_manifest(candidate_manifest)
    if candidate["approved"]:
        raise RuntimeError("temporary A/B candidate unexpectedly became approved")
    return ValidationCandidate(
        candidate_manifest,
        token,
        candidate["manifestFingerprint"],
        candidate["payloadFingerprint"],
    )


def publish_measured_artifact(
    source: Path,
    output_directory: Path,
    integration: dict[str, Any],
    *,
    blueprint_fingerprint: str,
    holdout_sha256: str,
) -> tuple[Path | None, bool, list[str]]:
    manifest, payload = load_manifest(source)
    if payload["training"]["blueprintFingerprint"] != blueprint_fingerprint:
        return None, False, ["approval Blueprint 与训练数据版本不一致"]
    metrics = dict(manifest["metrics"])
    metrics.update(integration)
    failures = quality_gate_failures(metrics)
    if int(metrics.get("leafActivatedNodes", 0)) <= 0:
        failures.append("IID leaf evaluation cutoff 零命中")
    if failures:
        return None, False, failures
    attestation = build_approval_attestation(
        payload_fingerprint=manifest["payloadFingerprint"],
        blueprint_fingerprint=blueprint_fingerprint,
        holdout_sha256=holdout_sha256,
        metrics=metrics,
        teacher_version=payload["training"]["teacherVersion"],
        continuation_profiles=payload["support"]["continuationProfiles"],
    )
    published = write_leaf_artifact(
        output_directory,
        payload["network"],
        metrics,
        payload["training"],
        support=payload["support"],
        approval_attestation=attestation,
    )
    return published, True, []


def main() -> int:
    args = parse_args()
    holdout = args.holdout.resolve()
    holdout_sha256 = hashlib.sha256(holdout.read_bytes()).hexdigest()
    rows = read_rows(holdout)
    blueprint_path = args.blueprint.resolve()
    os.environ["TEXAS_LAB_BLUEPRINT_PATH"] = str(blueprint_path)
    os.environ["TEXAS_LAB_BLUEPRINT_MODE"] = "required"
    reset_blueprint_runtime()
    blueprint_runtime = get_blueprint_runtime()
    if not blueprint_runtime.has_valid_policy:
        raise SystemExit("production Blueprint policy is not validated")
    blueprint_fingerprint = blueprint_runtime.fingerprint

    with tempfile.TemporaryDirectory(prefix="texas-lab-leaf-validation-") as temporary:
        candidate = validation_manifest(
            args.leaf_manifest.resolve(),
            Path(temporary),
            blueprint_fingerprint=blueprint_fingerprint,
        )
        try:
            baseline = solve_pass(
                rows,
                blueprint=blueprint_path,
                blueprint_fingerprint=blueprint_fingerprint,
                leaf_manifest=None,
            )
            accelerated = solve_pass(
                rows,
                blueprint=blueprint_path,
                blueprint_fingerprint=blueprint_fingerprint,
                leaf_manifest=candidate.manifest,
                expected_leaf_fingerprint=candidate.artifact_fingerprint,
                ab_token=candidate.token,
            )
        finally:
            shutdown_process_pool()
    metrics = compute_integration_metrics(
        baseline,
        accelerated,
        rows,
        float(PRODUCTION_VALIDATION_CONFIG["decisiveGapFraction"]),
    )
    atomic_write(args.output, metrics)
    report: dict[str, Any] = {
        "metrics": metrics,
        "output": str(args.output.resolve()),
        "validatorConfig": {
            "solver": PRODUCTION_SOLVER_CONFIG,
            "cutoff": PRODUCTION_CUTOFF_CONFIG,
            "continuation": PRODUCTION_CONTINUATION_CONFIG,
            "validation": PRODUCTION_VALIDATION_CONFIG,
        },
        "blueprintFingerprint": blueprint_fingerprint,
        "holdoutSha256": holdout_sha256,
    }
    if args.publish_output_dir is not None:
        published, approved, failures = publish_measured_artifact(
            args.leaf_manifest.resolve(),
            args.publish_output_dir.resolve(),
            metrics,
            blueprint_fingerprint=blueprint_fingerprint,
            holdout_sha256=holdout_sha256,
        )
        report.update(
            {
                "publishedManifest": None if published is None else str(published),
                "approved": approved,
                "gateFailures": failures,
            }
        )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
