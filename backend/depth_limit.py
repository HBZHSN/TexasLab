from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import os
from typing import Any, Callable

from .belief import ALL_COMBOS, BeliefState, ComboRange
from .blueprint import BlueprintRuntime, encode_state_features_many_hands
from .schema import public_history
from .value_model import (
    COMBO_INDEX,
    CONTINUATIONS,
    POSITIONS,
    LeafCutoffGate,
    LeafCutoffRejected,
    LeafModelInput,
    LeafModelRuntime,
    LeafPrediction,
)


StateKey = tuple[int, tuple[int, ...]]
GateCacheKey = tuple[int, StateKey]
PredictionCacheKey = tuple[int, StateKey, str, int, int]
RangePolicyProvider = Callable[
    [Any, str, list[int], list[list[float]], tuple[str, ...]],
    tuple[list[tuple[float, ...]], int] | None,
]


@dataclass(frozen=True)
class _Transition:
    parent: StateKey
    kind: str
    parent_state: Any
    action: str | None = None
    legal_actions: tuple[str, ...] = ()
    history_entry: dict[str, Any] | None = None


@dataclass(frozen=True)
class DepthLimitedValue:
    utility: float
    model_error_bb: float


def _state_key(state: Any) -> StateKey:
    return int(state.node_id), tuple(state.board)


def _path_depth(node_id: int) -> int:
    depth = 0
    current = int(node_id)
    while current > 1:
        depth += 1
        current //= 6
    return depth


class DepthLimitedEvaluator:
    """Belief propagation and approved value-model cutoffs for one subgame."""

    def __init__(
        self,
        root: Any,
        root_info: dict[str, Any],
        ranges: dict[str, Any],
        blueprint: BlueprintRuntime,
        leaf_runtime: LeafModelRuntime,
        policy_provider: RangePolicyProvider | None = None,
    ) -> None:
        self.root = root
        self.root_info = root_info
        self.blueprint = blueprint
        self.leaf_runtime = leaf_runtime
        self.policy_provider = policy_provider
        self.root_key = _state_key(root)
        self.transitions: dict[StateKey, _Transition] = {}
        self.history_cache: dict[StateKey, list[dict[str, Any]]] = {
            self.root_key: public_history(root_info["history"]),
        }
        self.belief_cache: dict[StateKey, BeliefState] = {}
        self.input_cache: dict[StateKey, LeafModelInput] = {}
        self.gate_cache: dict[GateCacheKey, LeafCutoffGate] = {}
        self.prediction_cache: dict[
            PredictionCacheKey,
            tuple[LeafCutoffGate, LeafPrediction],
        ] = {}
        self.rejection_reasons: Counter[str] = Counter()
        self.cutoff_attempts = 0
        self.cutoff_hits = 0
        self.exact_fold_shortcuts = 0
        self.belief_updates = 0
        self.belief_refreshes = 0
        self.reach_epoch = 0
        self.current_policy_queries = 0
        self.blueprint_range_policy_queries = 0
        self.gate_cache_hits = 0
        self.gate_cache_misses = 0
        self.prediction_cache_hits = 0
        self.prediction_cache_misses = 0
        self.evaluation_cutoff_hits = 0
        self.evaluation_model_error_bb = 0.0
        self.evaluation_uncertainty_bb = 0.0
        self._evaluation_mode = False

        try:
            self.mode = os.environ.get("TEXAS_LAB_LEAF_CUTOFF_MODE", "auto").strip().lower()
            if self.mode not in {"auto", "off", "aggressive"}:
                raise ValueError
        except ValueError:
            self.mode = "auto"
        try:
            self.explicit_depth = max(1, int(os.environ["TEXAS_LAB_LEAF_DEPTH"]))
        except (KeyError, ValueError):
            self.explicit_depth = None
        self.continuation_ids = self._continuation_ids()

        root_ranges: dict[str, ComboRange] = {}
        for player in root.players:
            source = ranges.get(player.id)
            if source is None:
                root_ranges[player.id] = ComboRange.uniform(root.board)
            else:
                root_ranges[player.id] = ComboRange.from_weighted(
                    source.combos,
                    source.cumulative,
                    root.board,
                )
        self.belief_cache[self.root_key] = BeliefState(root_ranges, tuple(root.board))

        self.compatible_blueprint = self._blueprint_compatible()
        if not getattr(blueprint, "has_valid_policy", False):
            self.rejection_reasons["blueprint-policy-unavailable"] += 1
        elif not self.compatible_blueprint:
            self.rejection_reasons["blueprint-version-mismatch"] += 1

    def _continuation_ids(self) -> tuple[int, ...]:
        raw = os.environ.get("TEXAS_LAB_LEAF_CONTINUATIONS", "").strip()
        if not raw:
            return (0,) * len(POSITIONS)
        names = {name: index for index, name in enumerate(CONTINUATIONS)}
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != len(POSITIONS):
            self.rejection_reasons["invalid-continuation-config"] += 1
            return (0,) * len(POSITIONS)
        try:
            values = tuple(
                names[part] if part in names else int(part)
                for part in parts
            )
        except (KeyError, ValueError):
            self.rejection_reasons["invalid-continuation-config"] += 1
            return (0,) * len(POSITIONS)
        if any(not 0 <= value < len(CONTINUATIONS) for value in values):
            self.rejection_reasons["invalid-continuation-config"] += 1
            return (0,) * len(POSITIONS)
        return values

    def _blueprint_compatible(self) -> bool:
        training = self.leaf_runtime.model.payload.get("training", {})
        if training.get("blueprintFingerprint") != self.blueprint.fingerprint:
            return False
        headers = training.get("datasetHeaders", [])
        versions = {
            str(header.get("blueprintVersion"))
            for header in headers
            if isinstance(header, dict) and header.get("blueprintVersion")
        }
        return versions == {self.blueprint.fingerprint}

    @property
    def active(self) -> bool:
        return bool(
            self.mode != "off"
            and getattr(self.blueprint, "has_valid_policy", False)
            and self.compatible_blueprint
        )

    def begin_evaluation(self) -> None:
        self._evaluation_mode = True
        self.evaluation_cutoff_hits = 0
        self.evaluation_model_error_bb = 0.0
        self.evaluation_uncertainty_bb = 0.0

    def end_evaluation(self) -> None:
        self._evaluation_mode = False

    def refresh_beliefs(self) -> None:
        """Start a new frozen reach epoch after CFR policies have changed.

        Public transitions and histories are immutable, but every posterior
        below an action depends on the current resolving policy.  Keeping
        those posteriors across regret updates silently turns a range-aware
        leaf model into a fixed Blueprint evaluator.  One epoch freezes a
        coherent policy snapshot; the solver advances epochs at bounded
        mini-batch boundaries so the cost remains amortized.
        """
        root_belief = self.belief_cache[self.root_key]
        self.belief_cache = {self.root_key: root_belief}
        self.input_cache.clear()
        self.gate_cache.clear()
        self.prediction_cache.clear()
        self.reach_epoch += 1
        self.belief_refreshes += 1

    def register_action(
        self,
        parent: Any,
        child: Any,
        action: str,
        legal_actions: tuple[str, ...],
    ) -> None:
        if not self.active:
            return
        actor_id = parent.pending[0]
        actor = next(player for player in parent.players if player.id == actor_id)
        child_actor = next(player for player in child.players if player.id == actor_id)
        paid = max(0.0, child.pot - parent.pot)
        amount = child_actor.street_committed if action == "raise" else paid
        entry = {
            "street": parent.street,
            "playerId": actor_id,
            "position": actor.position,
            "type": action,
            "amount": round(amount, 2),
            "potAfter": round(child.pot, 2),
        }
        self.transitions.setdefault(
            _state_key(child),
            _Transition(
                _state_key(parent),
                "action",
                parent,
                action,
                tuple(legal_actions),
                entry,
            ),
        )

    def register_chance(self, parent: Any, child: Any) -> None:
        if not self.active:
            return
        self.transitions.setdefault(
            _state_key(child),
            _Transition(_state_key(parent), "chance", parent),
        )

    def _history(self, key: StateKey) -> list[dict[str, Any]]:
        cached = self.history_cache.get(key)
        if cached is not None:
            return cached
        transition = self.transitions[key]
        history = list(self._history(transition.parent))
        if transition.history_entry is not None:
            history.append(dict(transition.history_entry))
        self.history_cache[key] = history
        return history

    def _posterior(self, belief: BeliefState, transition: _Transition) -> BeliefState:
        assert transition.action is not None
        state = transition.parent_state
        actor_id = state.pending[0]
        actor_range = belief.ranges[actor_id]
        supported_indices = [
            index
            for index, probability in enumerate(actor_range.probabilities)
            if probability > 0
        ]
        feature_rows = encode_state_features_many_hands(
            state,
            actor_id,
            [ALL_COMBOS[index] for index in supported_indices],
            root_history=self.root_info["history"],
            root_street=self.root.street,
        )
        policies = None
        if self.policy_provider is not None:
            provided = self.policy_provider(
                state,
                actor_id,
                supported_indices,
                feature_rows,
                transition.legal_actions,
            )
            if provided is not None:
                policies, current_count = provided
                if not 0 <= current_count <= len(supported_indices):
                    raise RuntimeError("range policy provider returned an invalid source count")
                self.current_policy_queries += current_count
                self.blueprint_range_policy_queries += len(supported_indices) - current_count
        if policies is None:
            # A validated Blueprint remains a conservative fallback only for
            # policy buckets that the current subgame has never reached.
            policies = self.blueprint.policy_many(feature_rows, transition.legal_actions)
            if policies is not None:
                self.blueprint_range_policy_queries += len(supported_indices)
        if policies is None:
            raise RuntimeError("current and Blueprint policies unavailable for belief update")
        if len(policies) != len(supported_indices):
            raise RuntimeError("range policy provider returned the wrong row count")
        action_index = transition.legal_actions.index(transition.action)
        likelihood = [0.0] * len(ALL_COMBOS)
        for hand_index, policy in zip(supported_indices, policies):
            if len(policy) != len(transition.legal_actions):
                raise RuntimeError("range policy provider returned the wrong action count")
            likelihood[hand_index] = policy[action_index]
        self.belief_updates += 1
        return belief.update_action(actor_id, likelihood)

    def _belief(self, state: Any) -> BeliefState:
        key = _state_key(state)
        cached = self.belief_cache.get(key)
        if cached is not None:
            return cached
        transition = self.transitions[key]
        parent = self._belief(transition.parent_state)
        if transition.kind == "chance":
            belief = parent.condition_public_cards(state.board)
        else:
            belief = self._posterior(parent, transition)
        self.belief_cache[key] = belief
        return belief

    def _public_state(self, state: Any) -> dict[str, Any]:
        return {
            "street": state.street,
            "board": list(state.board),
            "pot": state.pot,
            "currentBet": state.current_bet,
            "aggressiveActions": state.aggressive_actions,
            "players": [
                {
                    "id": player.id,
                    "position": player.position,
                    "stack": player.stack,
                    "streetCommitted": player.street_committed,
                    "totalCommitted": player.total_committed,
                    "folded": player.folded,
                    "allIn": player.all_in,
                }
                for player in state.players
            ],
            "pending": list(state.pending),
            "history": self._history(_state_key(state)),
        }

    def _model_input(self, state: Any) -> LeafModelInput:
        key = _state_key(state)
        cached = self.input_cache.get(key)
        if cached is not None:
            return cached
        belief = self._belief(state)
        by_position = {player.position: player.id for player in state.players}
        model_input = LeafModelInput.from_dict(
            {
                "publicState": self._public_state(state),
                "ranges": [
                    list(belief.ranges[by_position[position]].probabilities)
                    for position in POSITIONS
                ],
                "continuationIds": list(self.continuation_ids),
            }
        )
        self.input_cache[key] = model_input
        return model_input

    def _candidate(self, state: Any) -> bool:
        if not self.active or _state_key(state) == self.root_key:
            return False
        actable = sum(
            not player.folded and not player.all_in
            for player in state.players
        )
        if actable < 2:
            # With fewer than two decision-makers the remaining runout is an
            # exact batch showdown and is cheaper/safer than model inference.
            return False
        depth = _path_depth(state.node_id) - _path_depth(self.root.node_id)
        if self.explicit_depth is not None:
            return depth >= self.explicit_depth
        if self.mode == "aggressive":
            return depth >= 3 and state.street != "river"
        if self.root.street == "preflop":
            return state.street == "flop"
        root_active = sum(not player.folded for player in self.root.players)
        if self.root.street == "flop" and root_active > 2:
            return state.street == "turn" or (
                state.street == "flop" and state.aggressive_actions >= 2
            )
        # Turn, river and heads-up flop trees remain exact by default.
        return False

    def _predict_after_approved_gate(
        self,
        model_input: LeafModelInput,
        seat_index: int,
        hand_index: int,
    ) -> LeafPrediction:
        """Run only inference after this epoch's state gate has passed.

        LeafModelRuntime.predict() intentionally gates standalone callers, but
        doing that here would repeat the six-range OOD/entropy scan for every
        target hand.  The evaluator owns an epoch-scoped approved gate, so it
        can invoke the portable model directly and retain the runtime's
        uncertainty rejection.  Lightweight test/dummy runtimes without the
        production model/support surface keep using their public method.
        """
        try:
            predict = self.leaf_runtime.model.predict
            maximum = float(self.leaf_runtime.support["maxInferenceUncertaintyBb"])
        except (AttributeError, KeyError, TypeError):
            return self.leaf_runtime.predict(model_input, seat_index, hand_index)
        prediction = predict(model_input, seat_index, hand_index)
        if prediction.uncertainty > maximum:
            raise LeafCutoffRejected(
                f"叶模型 uncertainty={prediction.uncertainty:.6f}BB 超过 {maximum:.6f}BB"
            )
        return prediction

    def maybe_value(
        self,
        state: Any,
        hands: dict[str, tuple[int, int]],
        target_id: str,
        baseline_stack: float,
    ) -> DepthLimitedValue | None:
        if not self._candidate(state):
            return None
        player = next(player for player in state.players if player.id == target_id)
        if player.folded:
            # A folded player's final stack is already known exactly.  Asking
            # a value model here can turn the root Fold EV away from zero and
            # corrupt a decision even though no uncertainty remains.
            self.exact_fold_shortcuts += 1
            return DepthLimitedValue(player.stack - baseline_stack, 0.0)
        self.cutoff_attempts += 1
        try:
            cache_prediction = False
            cache_gate = False
            seat_index = POSITIONS.index(player.position)
            hand_index = COMBO_INDEX[tuple(sorted(hands[target_id]))]
            state_key = _state_key(state)
            gate_key = (self.reach_epoch, state_key)
            prediction_key = (
                self.reach_epoch,
                state_key,
                target_id,
                seat_index,
                hand_index,
            )
            cached = self.prediction_cache.get(prediction_key)
            if cached is not None:
                self.prediction_cache_hits += 1
                gate, prediction = cached
            else:
                self.prediction_cache_misses += 1
                model_input = self._model_input(state)
                gate = self.gate_cache.get(gate_key)
                gate_was_cached = gate is not None
                if gate_was_cached:
                    self.gate_cache_hits += 1
                else:
                    self.gate_cache_misses += 1
                    gate = self.leaf_runtime.cutoff_gate(model_input)
                    if not gate.allowed:
                        for reason in gate.reasons:
                            self.rejection_reasons[reason] += 1
                        return None
                prediction = self._predict_after_approved_gate(
                    model_input,
                    seat_index,
                    hand_index,
                )
                if not math.isfinite(prediction.final_stack) or not math.isfinite(
                    prediction.uncertainty
                ):
                    self.rejection_reasons["non-finite-prediction"] += 1
                    return None
                # Only successful, finite predictions make either cache.  A
                # rejection or transient exception must be observed again on
                # the next call rather than hidden behind stale state.
                cache_prediction = True
                cache_gate = not gate_was_cached
            error_bound = max(gate.model_error_bb, 1.96 * prediction.uncertainty)
            # Delay mutation until every operation needed for a usable value
            # has succeeded, so even malformed gate data cannot poison a
            # later attempt in the same epoch.
            if cache_prediction:
                if cache_gate:
                    self.gate_cache[gate_key] = gate
                self.prediction_cache[prediction_key] = (gate, prediction)
        except LeafCutoffRejected:
            self.rejection_reasons["prediction-uncertainty"] += 1
            return None
        except Exception:
            self.rejection_reasons["leaf-evaluation-error"] += 1
            return None
        self.cutoff_hits += 1
        if self._evaluation_mode:
            self.evaluation_cutoff_hits += 1
            self.evaluation_model_error_bb = max(self.evaluation_model_error_bb, error_bound)
            self.evaluation_uncertainty_bb = max(
                self.evaluation_uncertainty_bb,
                prediction.uncertainty,
            )
        return DepthLimitedValue(
            prediction.final_stack - baseline_stack,
            error_bound,
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "leafCutoffActive": self.active,
            "leafCutoffMode": self.mode,
            "leafModelFingerprint": self.leaf_runtime.fingerprint,
            "leafCutoffAttempts": self.cutoff_attempts,
            "leafCutoffHits": self.cutoff_hits,
            "leafExactFoldShortcuts": self.exact_fold_shortcuts,
            "leafEvaluationCutoffHits": self.evaluation_cutoff_hits,
            "leafModelErrorBound": self.evaluation_model_error_bb,
            "leafModelMaxUncertainty": self.evaluation_uncertainty_bb,
            "leafBeliefUpdates": self.belief_updates,
            "leafBeliefRefreshes": self.belief_refreshes,
            "leafReachEpoch": self.reach_epoch,
            "leafCurrentPolicyRangeQueries": self.current_policy_queries,
            "leafBlueprintRangePolicyQueries": self.blueprint_range_policy_queries,
            "leafCutoffGateCacheHits": self.gate_cache_hits,
            "leafCutoffGateCacheMisses": self.gate_cache_misses,
            "leafPredictionCacheHits": self.prediction_cache_hits,
            "leafPredictionCacheMisses": self.prediction_cache_misses,
            "leafCutoffRejections": dict(self.rejection_reasons),
        }
