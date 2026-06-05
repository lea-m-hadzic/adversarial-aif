"""Simplified expected free energy objective.

    J(a) = alpha * P_coll(a, p) + beta * T_cross(a) + gamma * H(p_{t+1} | a)

gamma defaults to 0: under passive observation the entropy term is
approximately invariant in `a` and contributes nothing useful to action
selection. The field stays in the config so callers that introduce an
active-signaling action (where Q(s' | a) varies with a) can switch it on.

Action selection is argmin over a small discrete action set.
"""

from dataclasses import dataclass
from typing import List

from src.agent import Agent


@dataclass
class EFEWeights:
    alpha: float = 1.0    # collision risk
    beta: float = 0.1     # time to cross
    gamma: float = 0.0    # entropy / uncertainty


@dataclass
class CollisionModel:
    """Parameters used to predict collision probability under a candidate action."""
    intersection_width: float = 4.0
    horizon_seconds: float = 4.0     # how far ahead to look when predicting
    dt: float = 0.1
    safety_margin: float = 1.0       # seconds of required separation between
                                     # one agent leaving the intersection and
                                     # the other entering it
    margin_softness: float = 0.5     # seconds; controls how sharply the soft
                                     # collision probability transitions to 0
                                     # as the gap exceeds the safety margin


def _time_inside_intersection(d: float, v: float, action: float, model: CollisionModel) -> tuple[float, float]:
    """Predict when the agent enters and exits the intersection box, given a
    constant acceleration `action` from current state (d, v).

    Returns (t_enter, t_exit) in seconds. If the agent never enters within
    horizon, returns (inf, inf). If the agent is already inside, t_enter = 0.

    Uses simple integration with velocity clipped at zero (no reversing).
    """
    horizon_steps = int(model.horizon_seconds / model.dt)
    t_enter = float("inf")
    t_exit = float("inf")
    cur_d = d
    cur_v = v
    inside = -model.intersection_width <= cur_d <= 0.0
    if inside:
        t_enter = 0.0

    for k in range(1, horizon_steps + 1):
        cur_v = max(0.0, cur_v + action * model.dt)
        cur_d = cur_d - cur_v * model.dt
        now_inside = -model.intersection_width <= cur_d <= 0.0
        if now_inside and not inside:
            t_enter = k * model.dt
            inside = True
        elif not now_inside and inside:
            t_exit = k * model.dt
            return t_enter, t_exit
    return t_enter, t_exit


def _soft_collision_prob(gap: float, model: CollisionModel) -> float:
    """Map a temporal gap (seconds) between two intersection-occupancy
    intervals to a soft collision probability.

    gap > 0: one agent finishes its occupancy `gap` seconds before the other
    starts. Larger gap => safer.

    gap < 0: the intervals overlap. Definitely a collision.

    Piecewise linear ramp:
        gap <= 0                           -> 1.0
        gap >= safety_margin + softness    -> 0.0
        in between                         -> linear ramp
    """
    safe_threshold = model.safety_margin + model.margin_softness
    if gap <= 0.0:
        return 1.0
    if gap >= safe_threshold:
        return 0.0
    return 1.0 - gap / safe_threshold


def predicted_collision_prob(
    self_state: Agent,
    other_state: Agent,
    action_self: float,
    belief_other_yields: float,
    model: CollisionModel,
) -> float:
    """Probability of collision under candidate action.

    Two-component model:
        - If other is a Yielder (prob = p): assume they decelerate hard.
          Yielders are assumed to successfully avoid us, so P(coll | Y) = 0.
        - If other is a Non-yielder (prob = 1 - p): assume they continue at
          current velocity. Compute the temporal gap between our and their
          intersection-occupancy intervals and convert to a soft collision
          probability with a safety margin.

    P(coll) = (1 - p) * P(coll | N).
    """
    p = belief_other_yields

    t_enter_other, t_exit_other = _time_inside_intersection(
        other_state.d, other_state.v, action=0.0, model=model
    )
    if t_enter_other == float("inf"):
        return 0.0

    t_enter_self, t_exit_self = _time_inside_intersection(
        self_state.d, self_state.v, action=action_self, model=model
    )
    if t_enter_self == float("inf"):
        return 0.0

    # Temporal gap: positive iff the two occupancy intervals are disjoint.
    # We take the smaller of (other_enter - self_exit) and (self_enter - other_exit).
    # If both are positive, that's the actual gap between intervals.
    # If either is negative, the intervals overlap, so gap <= 0 and the soft
    # function returns 1.
    gap = min(t_enter_other - t_exit_self, t_enter_self - t_exit_other)
    p_coll_given_no_yield = _soft_collision_prob(gap, model)

    return (1.0 - p) * p_coll_given_no_yield


def time_to_cross(self_state: Agent, action_self: float, model: CollisionModel) -> float:
    """Predicted time (seconds) until self exits the intersection.

    If self never exits within horizon, return horizon (cap; not infinity, so
    the objective stays well-scaled).
    """
    _, t_exit = _time_inside_intersection(
        self_state.d, self_state.v, action=action_self, model=model
    )
    if t_exit == float("inf"):
        return model.horizon_seconds
    return t_exit


def efe(
    self_state: Agent,
    other_state: Agent,
    action_self: float,
    belief_other_yields: float,
    weights: EFEWeights,
    model: CollisionModel,
) -> float:
    """Scalar objective for one candidate action."""
    p_coll = predicted_collision_prob(
        self_state, other_state, action_self, belief_other_yields, model
    )
    t_cross = time_to_cross(self_state, action_self, model)
    # gamma * H(...) omitted while gamma == 0; placeholder for later.
    return weights.alpha * p_coll + weights.beta * t_cross


def select_action(
    self_state: Agent,
    other_state: Agent,
    belief_other_yields: float,
    action_set: List[float],
    weights: EFEWeights,
    model: CollisionModel,
) -> float:
    """argmin over the discrete action set. Ties broken by preferring the
    action closest to zero (do nothing rather than swerve)."""
    best_action = action_set[0]
    best_score = float("inf")
    for a in action_set:
        score = efe(self_state, other_state, a, belief_other_yields, weights, model)
        if score < best_score - 1e-12 or (
            abs(score - best_score) < 1e-12 and abs(a) < abs(best_action)
        ):
            best_score = score
            best_action = a
    return best_action