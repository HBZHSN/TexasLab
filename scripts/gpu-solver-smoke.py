from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.solver import shutdown_process_pool, solve_decision


def seed_zero_preflop_info() -> dict:
    positions = ("UTG", "HJ", "CO", "BTN", "SB", "BB")
    players = []
    for index, position in enumerate(positions):
        committed = 0.5 if position == "SB" else 1.0 if position == "BB" else 0.0
        players.append(
            {
                "id": f"p{index}",
                "position": position,
                "stack": 100 - committed,
                "streetCommitted": committed,
                "totalCommitted": committed,
                "folded": False,
                "allIn": False,
            }
        )
    return {
        "street": "preflop",
        "board": [],
        "pot": 1.5,
        "currentBet": 1,
        "aggressiveActions": 0,
        "hero": {"id": "p0", "position": "UTG", "stack": 100, "streetCommitted": 0, "hole": [4, 1]},
        "players": players,
        "pending": [f"p{index}" for index in range(6)],
        "history": [],
        "legalActions": ["fold", "call", "raise"],
        "raiseTo": 2.5,
    }


if __name__ == "__main__":
    started = time.perf_counter()
    try:
        result = solve_decision(seed_zero_preflop_info(), iterations=20_000)
        summary = {
            "seconds": round(time.perf_counter() - started, 3),
            "backend": result["computeBackend"],
            "device": result["computeDevice"],
            "cuda": result["cudaAvailable"],
            "gpuBatches": result["gpuBatches"],
            "gpuHandEvaluations": result["gpuHandEvaluations"],
            "gpuTrainingBatches": result["gpuTrainingBatches"],
            "gpuTrainingHandEvaluations": result["gpuTrainingHandEvaluations"],
            "gpuEvaluationHandEvaluations": result["gpuEvaluationHandEvaluations"],
            "workers": result["workerCount"],
            "trainingIterations": result["trainingIterations"],
            "trainingRequestedIterations": result["trainingRequestedIterations"],
            "trainingPrunedIterations": result["trainingPrunedIterations"],
            "adaptiveTrainingWorkersStopped": result["adaptiveTrainingWorkersStopped"],
            "adaptiveTrainingDominanceWorkersStopped": result[
                "adaptiveTrainingDominanceWorkersStopped"
            ],
            "trainingStopReasons": result["trainingStopReasons"],
            "evaluationIterations": result["evaluationIterations"],
            "evaluationRequestedIterations": result["evaluationRequestedIterations"],
            "decisionEvaluationWorkersStopped": result[
                "decisionEvaluationWorkersStopped"
            ],
            "evaluationPrunedIterations": result["evaluationPrunedIterations"],
            "adaptiveEvaluation": result["adaptiveEvaluation"],
            "adaptiveEvaluationRequested": result["adaptiveEvaluationRequested"],
            "adaptiveEvaluationDisabledForParallel": result["adaptiveEvaluationDisabledForParallel"],
            "adaptiveEvaluationWorkersStopped": result["adaptiveEvaluationWorkersStopped"],
            "confidenceMethod": result["confidenceMethod"],
            "errorBudgetFraction": result["errorBudgetFraction"],
            "executionSemantics": result["executionSemantics"],
            "regretMode": result["regretMode"],
            "blueprintPolicySeeds": result["blueprintPolicySeeds"],
            "leafCutoffHits": result["leafCutoffHits"],
            "leafPredictionCacheHits": result["leafPredictionCacheHits"],
            "actions": result["actions"],
            "rootRegret": result["rootRegret"],
            "evaluationInfosetLookups": result["evaluationInfosetLookups"],
            "evaluationUntrainedLookups": result["evaluationUntrainedLookups"],
            "evaluationUntrainedRate": result["evaluationUntrainedRate"],
            "evaluationPolicyBackoffLookups": result["evaluationPolicyBackoffLookups"],
            "evaluationPolicyBackoffRate": result["evaluationPolicyBackoffRate"],
            "evaluationUniformFallbackLookups": result["evaluationUniformFallbackLookups"],
            "evaluationUniformFallbackRate": result["evaluationUniformFallbackRate"],
            "publicTransitionHits": result["publicTransitionHits"],
            "publicTransitionMisses": result["publicTransitionMisses"],
        }
        print(json.dumps(summary, ensure_ascii=False))
        if not result["cudaAvailable"] or result["gpuTrainingHandEvaluations"] <= 0:
            raise SystemExit("GPU solver smoke did not execute CUDA training-leaf evaluation")
        if result["gpuEvaluationHandEvaluations"] <= 0:
            raise SystemExit("GPU solver smoke did not execute CUDA EV evaluation")
        if "cuda-minibatch" not in result["executionSemantics"]:
            raise SystemExit("GPU solver smoke did not report CUDA minibatch semantics")
        if abs(result["errorBudgetFraction"] - 0.05) > 1e-12:
            raise SystemExit("GPU solver smoke did not use the default 5% error budget")
        if result["workerCount"] > 1 and (
            result["adaptiveEvaluation"]
            or not result["adaptiveEvaluationDisabledForParallel"]
            or result["confidenceMethod"] != "pooled-fixed-sample-empirical-bernstein"
            or result["evaluationIterations"]
            != result["evaluationPerWorkerActual"] * result["workerCount"]
            or (
                result["decisionEvaluationWorkersStopped"] == 0
                and result["evaluationIterations"]
                != result["evaluationRequestedIterations"]
            )
        ):
            raise SystemExit("parallel GPU evaluation was not fixed-length and equally pooled")
        fold = next((row for row in result["actions"] if row["action"] == "fold"), None)
        if fold is not None and (
            fold["ci95"] != 0
            or fold["modelErrorCalibration"] != 0
        ):
            raise SystemExit("exact Fold action received sampling or model error")
    finally:
        shutdown_process_pool()
