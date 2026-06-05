"""Two-agent intersection simulator.

Geometry: agent A moves along the +x axis toward the origin from x > 0; agent
B moves along the +y axis toward the origin from y > 0. The intersection is a
square box of side `intersection_width` centered at the origin. A collision
occurs iff both agents are inside the intersection box at the same timestep.

Policies now return (acceleration, signal) tuples. The signal is an integer
0 or 1 representing whether the agent is currently emitting a "yielding"
intent signal. Signals are recorded in the trace for downstream consumers
(belief updates, detection).
"""

from dataclasses import dataclass
from typing import Callable, Tuple, Literal

import numpy as np

from src.agent import Agent


# A policy maps (own_agent, other_agent, timestep) -> (acceleration, signal).
PolicyOutput = Tuple[float, int]
Policy = Callable[[Agent, Agent, int], PolicyOutput]

Outcome = Literal["collision", "deadlock", "cleared", "timeout"]


@dataclass
class Trace:
    """Record of one episode.

    All arrays have the same length T = number of timesteps actually run.
    """

    d_a: np.ndarray
    v_a: np.ndarray
    a_a: np.ndarray   # action (acceleration) commanded to A
    s_a: np.ndarray   # signal emitted by A (0 or 1)
    d_b: np.ndarray
    v_b: np.ndarray
    a_b: np.ndarray
    s_b: np.ndarray
    outcome: Outcome
    final_step: int

    def __len__(self) -> int:
        return self.final_step + 1


# ---------------------------------------------------------------------------
# Minimal stand-alone policies for smoke-testing the simulator.
# Production policies are built inside src/evaluation.py and src/adversarial.py.
# ---------------------------------------------------------------------------

def constant_policy(self_agent: Agent, other_agent: Agent, t: int) -> PolicyOutput:
    """Maintain current velocity, no signal."""
    return 0.0, 0


def decelerate_policy(braking_accel: float = -3.0, signal_yield: bool = True) -> Policy:
    """Decelerate at a constant rate until stopped, then hold.

    Emits signal=1 (yielding) by default, since this policy is a yielder.
    """
    if braking_accel >= 0:
        raise ValueError("braking_accel must be negative.")
    sig = 1 if signal_yield else 0

    def policy(self_agent: Agent, other_agent: Agent, t: int) -> PolicyOutput:
        if self_agent.v <= 0.0:
            return 0.0, sig
        return braking_accel, sig

    return policy


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class Simulation:
    agent_a: Agent
    agent_b: Agent
    dt: float = 0.1
    max_steps: int = 200
    intersection_width: float = 4.0

    deadlock_velocity_threshold: float = 0.1
    deadlock_distance: float = 5.0
    deadlock_steps: int = 30

    def _collision(self) -> bool:
        return (
            self.agent_a.in_intersection(self.intersection_width)
            and self.agent_b.in_intersection(self.intersection_width)
        )

    def _both_cleared(self) -> bool:
        return (
            self.agent_a.has_passed(self.intersection_width)
            and self.agent_b.has_passed(self.intersection_width)
        )

    def _is_stuck(self, agent: Agent) -> bool:
        return (
            agent.v < self.deadlock_velocity_threshold
            and 0.0 < agent.d < self.deadlock_distance
        )

    def run_episode(self, policy_a: Policy, policy_b: Policy) -> Trace:
        d_a, v_a, a_a, s_a = [], [], [], []
        d_b, v_b, a_b, s_b = [], [], [], []

        outcome: Outcome = "timeout"
        deadlock_counter = 0
        final_step = self.max_steps - 1

        for t in range(self.max_steps):
            d_a.append(self.agent_a.d)
            v_a.append(self.agent_a.v)
            d_b.append(self.agent_b.d)
            v_b.append(self.agent_b.v)
            # Record the signals as they are at the start of this step.
            # Signals are stored on the agent itself so belief updates from
            # the OTHER agent's policy can read them via other_agent.signal.
            s_a.append(self.agent_a.signal)
            s_b.append(self.agent_b.signal)

            if self._collision():
                a_a.append(0.0)
                a_b.append(0.0)
                outcome = "collision"
                final_step = t
                break

            if self._both_cleared():
                a_a.append(0.0)
                a_b.append(0.0)
                outcome = "cleared"
                final_step = t
                break

            if self._is_stuck(self.agent_a) and self._is_stuck(self.agent_b):
                deadlock_counter += 1
            else:
                deadlock_counter = 0
            if deadlock_counter >= self.deadlock_steps:
                a_a.append(0.0)
                a_b.append(0.0)
                outcome = "deadlock"
                final_step = t
                break

            action_a, signal_a = policy_a(self.agent_a, self.agent_b, t)
            action_b, signal_b = policy_b(self.agent_b, self.agent_a, t)
            a_a.append(action_a)
            a_b.append(action_b)
            # Update each agent's signal AFTER both policies have run, so
            # neither agent sees the other's freshly-updated signal this step
            # (they see the previous step's signal). This avoids a chicken-and-
            # egg coupling and makes the dynamics well-defined.
            self.agent_a.signal = signal_a
            self.agent_b.signal = signal_b
            self.agent_a.step(action_a, self.dt)
            self.agent_b.step(action_b, self.dt)

        return Trace(
            d_a=np.array(d_a),
            v_a=np.array(v_a),
            a_a=np.array(a_a),
            s_a=np.array(s_a),
            d_b=np.array(d_b),
            v_b=np.array(v_b),
            a_b=np.array(a_b),
            s_b=np.array(s_b),
            outcome=outcome,
            final_step=final_step,
        )