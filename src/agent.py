"""Agent class.

Each agent moves along its own 1D axis toward the intersection. The simulator
owns the geometry (which axis each agent is on); the agent itself only tracks
its scalar distance-to-intersection and velocity.

Convention:
    d > 0  : agent has not yet reached the intersection (approaching)
    d = 0  : agent is at the near edge of the intersection
    d < 0  : agent has entered / passed the intersection
        in particular, d in [-W, 0] means the agent is inside the
        intersection box (W = intersection width along the agent's axis)

This sign convention means "approaching the intersection" = "d decreasing
toward zero," which matches the proposal's notation (Delta D < 0 in Fig. 1).
"""

from dataclasses import dataclass, field


@dataclass
class Agent:
    """Single agent state.

    Attributes:
        d: distance to the near edge of the intersection, in meters.
        v: velocity along the agent's axis, in m/s. Always nonnegative
            (agents do not reverse).
        v_min: lower bound on velocity (typically 0).
        v_max: upper bound on velocity.
        p: belief that the OTHER agent will yield, in [0, 1]. Updated by
            policies that maintain a belief over the other's intent.
        signal: yielding signal currently emitted by this agent, 0 or 1.
            Read by the other agent's belief update and by the detector.
    """

    d: float
    v: float
    v_min: float = 0.0
    v_max: float = 15.0
    p: float = 0.5
    signal: int = 0

    def step(self, action: float, dt: float) -> None:
        """Apply acceleration `action` (m/s^2) for one timestep of length dt.

        Velocity is clipped to [v_min, v_max]. Position is updated with the
        clipped velocity (semi-implicit Euler), which avoids the bug where an
        agent commanded to brake "overshoots" backward through the intersection.
        """
        new_v = self.v + action * dt
        if new_v < self.v_min:
            new_v = self.v_min
        elif new_v > self.v_max:
            new_v = self.v_max
        self.v = new_v
        self.d -= self.v * dt  # approaching: d decreases

    def in_intersection(self, intersection_width: float) -> bool:
        """True if the agent is currently inside the intersection box."""
        return -intersection_width <= self.d <= 0.0

    def has_passed(self, intersection_width: float) -> bool:
        """True if the agent has fully cleared the intersection."""
        return self.d < -intersection_width