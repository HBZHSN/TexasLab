from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


def empirical_bernstein_radius(
    count: int,
    variance: float,
    range_width: float,
    alpha: float,
) -> float:
    """Finite-sample bounded-utility radius that survives zero variance."""
    if count <= 1:
        return math.inf
    width = max(0.0, float(range_width))
    failure_probability = min(0.5, max(1e-12, float(alpha)))
    log_term = math.log(3.0 / failure_probability)
    return (
        math.sqrt(2.0 * max(0.0, float(variance)) * log_term / count)
        + 3.0 * width * log_term / count
    )


@dataclass
class OnlineMoments:
    """Numerically stable streaming mean and sampling variance."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("online statistic received a non-finite value")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / max(1, self.count - 1)

    @property
    def stderr(self) -> float:
        if self.count <= 1:
            return math.inf
        return math.sqrt(max(0.0, self.variance) / self.count)


class PairedActionStats:
    """Streaming action EVs and common-random-number pair differences.

    Every update must contain one outcome for every action from the same deal
    and public runout.  Tracking the paired differences directly is much more
    efficient than comparing independent action confidence intervals.
    """

    def __init__(self, actions: Sequence[str]) -> None:
        self.actions = tuple(actions)
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be a non-empty unique sequence")
        self.action = {name: OnlineMoments() for name in self.actions}
        self.pairs = {
            (left, right): OnlineMoments()
            for left_index, left in enumerate(self.actions)
            for right in self.actions[left_index + 1 :]
        }

    @property
    def count(self) -> int:
        return self.action[self.actions[0]].count

    def update(self, outcomes: Mapping[str, float]) -> None:
        if set(outcomes) != set(self.actions):
            raise ValueError("paired update must contain every action exactly once")
        values = {name: float(outcomes[name]) for name in self.actions}
        for name, value in values.items():
            self.action[name].update(value)
        for (left, right), moments in self.pairs.items():
            moments.update(values[left] - values[right])

    def best_action(self) -> str:
        return max(self.actions, key=lambda name: self.action[name].mean)

    def difference(self, left: str, right: str) -> tuple[float, float]:
        if left == right:
            return 0.0, 0.0
        if (left, right) in self.pairs:
            moments = self.pairs[(left, right)]
            return moments.mean, moments.stderr
        moments = self.pairs[(right, left)]
        return -moments.mean, moments.stderr

    def can_stop(
        self,
        *,
        min_samples: int,
        confidence_z: float,
        target_difference_ci: float,
        material_edge: float,
        additive_model_error: float = 0.0,
        action_model_errors: Mapping[str, float] | None = None,
        target_action_ci: float | None = None,
        utility_range_width: float | None = None,
        utility_range_widths: Mapping[str, float] | None = None,
        confidence_alpha: float = 0.05,
    ) -> bool:
        """Conservatively stop when every alternative is resolved.

        An alternative is resolved when it is confidently materially worse,
        or when its paired confidence radius is already below the requested
        decision precision.  A high ``confidence_z`` is intentional because
        this predicate is inspected repeatedly during sequential sampling.
        """
        if self.count < max(2, int(min_samples)):
            return False

        if utility_range_widths is not None:
            if set(utility_range_widths) != set(self.actions):
                raise ValueError("utility range widths must contain every action exactly once")
            range_widths = {
                action: max(0.0, float(utility_range_widths[action]))
                for action in self.actions
            }
        elif utility_range_width is not None:
            width = max(0.0, float(utility_range_width))
            range_widths = {action: width for action in self.actions}
        else:
            range_widths = None

        if action_model_errors is not None:
            if set(action_model_errors) != set(self.actions):
                raise ValueError("model errors must contain every action exactly once")
            model_errors = {
                action: max(0.0, float(action_model_errors[action]))
                for action in self.actions
            }
        else:
            shared_error = max(0.0, float(additive_model_error)) / 2
            model_errors = {action: shared_error for action in self.actions}

        def sampling_radius(moments: OnlineMoments, width: float) -> float:
            if range_widths is None:
                return float(confidence_z) * moments.stderr
            return empirical_bernstein_radius(
                moments.count,
                moments.variance,
                width,
                confidence_alpha,
            )

        if target_action_ci is not None:
            if any(
                sampling_radius(
                    moments,
                    0.0 if range_widths is None else range_widths[action],
                )
                + model_errors[action]
                > float(target_action_ci)
                for action, moments in self.action.items()
            ):
                return False
        best = self.best_action()
        for action in self.actions:
            if action == best:
                continue
            difference, _stderr = self.difference(best, action)
            pair = self.pairs.get((best, action)) or self.pairs[(action, best)]
            radius = sampling_radius(
                pair,
                0.0
                if range_widths is None
                else range_widths[best] + range_widths[action],
            ) + model_errors[best] + model_errors[action]
            confidently_worse = difference - radius > float(material_edge)
            precisely_tied = radius <= float(target_difference_ci)
            if not confidently_worse and not precisely_tied:
                return False
        return True

    def snapshot(self) -> dict[str, object]:
        best = self.best_action()
        return {
            "samples": self.count,
            "bestAction": best,
            "actions": {
                action: {
                    "mean": self.action[action].mean,
                    "stderr": self.action[action].stderr,
                    "differenceFromBest": self.difference(best, action)[0],
                    "differenceStderr": self.difference(best, action)[1],
                }
                for action in self.actions
            },
        }
