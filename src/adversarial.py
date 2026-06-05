"""Adversarial behavior policies.

HonestPolicy: emits signal=1 iff actually yielding. Two flavors are useful:
    - HonestYielder: brakes, emits signal=1.
    - HonestNonYielder: maintains velocity, emits signal=0.

DeceptivePolicy: emits signal=1 (yielding) while NOT actually yielding —
    i.e., maintains velocity or accelerates. This is the adversarial mode
    in the proposal. The deception can be permanent or stochastic.

All policies return (acceleration, signal).
"""

import random
from typing import Callable, Tuple

from src.agent import Agent

PolicyOutput = Tuple[float, int]
Policy = Callable[[Agent, Agent, int], PolicyOutput]


def honest_yielder(braking_accel: float = -3.0) -> Policy:
    """Brake to a near-stop, then proceed gently after the other agent has cleared.
    Always emits signal=1.
    """
    if braking_accel >= 0:
        raise ValueError("braking_accel must be negative.")
    target_resume_velocity = 8.0
    resume_accel = 1.5
    nearly_stopped = 0.5  # m/s
    intersection_far_edge = -4.0

    def policy(self_agent: Agent, other_agent: Agent, t: int) -> PolicyOutput:
        other_cleared = other_agent.d < intersection_far_edge
        # Only resume after fully stopping AND the other has cleared.
        if other_cleared and self_agent.v >= nearly_stopped:
            return 0.0, 1  # hold (don't accelerate while still moving)
        if other_cleared and self_agent.v < nearly_stopped:
            if self_agent.v < target_resume_velocity:
                return resume_accel, 1
            return 0.0, 1
        # Other not cleared: keep braking.
        if self_agent.v <= 0.0:
            return 0.0, 1
        return braking_accel, 1

    return policy


def honest_nonyielder() -> Policy:
    """Maintain current velocity, signal=0."""

    def policy(self_agent: Agent, other_agent: Agent, t: int) -> PolicyOutput:
        return 0.0, 0

    return policy


def deceptive(start_step: int = 0, accel_magnitude: float = 0.0) -> Policy:
    """Emit signal=1 (claiming yield) while NOT actually yielding.

    Args:
        start_step: timestep at which to begin emitting the deceptive signal.
            Before this, signal=0. Useful for studying detection latency
            relative to a known onset time.
        accel_magnitude: longitudinal acceleration (m/s^2) commanded while
            deceiving. Default 0.0 reproduces the original "maintains
            velocity" behavior used in all archived experiment results.
            Set positive (e.g. 1.5) to model the more extreme adversary
            described in the paper intro — "signaling yield while
            accelerating into the intersection."
    """

    def policy(self_agent: Agent, other_agent: Agent, t: int) -> PolicyOutput:
        if t >= start_step:
            return accel_magnitude, 1
        return 0.0, 0

    return policy


def stochastic_deceptive(adversarial_rate: float, seed: int = 0) -> Policy:
    """Per-EPISODE adversarial flag.

    With probability `adversarial_rate`, the policy is deceptive for the
    whole episode; otherwise it is an honest non-yielder.

    Note: this is a closure; the adversarial flag is sampled lazily on the
    first call. If you need per-episode reproducibility, instantiate a new
    policy per episode (which is what experiment scripts will do).
    """
    rng = random.Random(seed)
    state = {"is_adversarial": None}

    def policy(self_agent: Agent, other_agent: Agent, t: int) -> PolicyOutput:
        if state["is_adversarial"] is None:
            state["is_adversarial"] = rng.random() < adversarial_rate
        if state["is_adversarial"]:
            return 0.0, 1   # signal yield, but don't
        return 0.0, 0       # honest non-yielder

    return policy