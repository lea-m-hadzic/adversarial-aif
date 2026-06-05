"""Inconsistency score and threshold-based detector.

The inconsistency score s_t measures the contradiction between a yielding
signal and observed kinematics. We raise the score only when:
    1. The other agent is signaling yield (signal_t = 1), AND
    2. They are still moving fast enough to be a threat (v > velocity_threshold), AND
    3. Their acceleration is above a yielding threshold (not decelerating enough).

Concretely:
    raw_t = max(0, observed_accel_t - yield_accel_threshold)
            if signal_t == 1 and observed_velocity_t > velocity_threshold
            else 0

Smoothed with a leaky EMA:
    s_t = (1 - lambda) * s_{t-1} + lambda * raw_t

Detection fires when s_t > tau, sticky for the rest of the episode.

The velocity gate avoids false alarms on honest yielders that have braked
from high speed and are now coasting at low velocity — coasting at v < 5 m/s
is normal post-braking behavior, not adversarial.

Conflict-phase gating: observations are only counted while the OTHER agent is
still approaching or inside the intersection. Calling observe() with
in_conflict=False freezes the score.
"""

from dataclasses import dataclass


@dataclass
class DetectorParams:
    tau: float = 1.0
    yield_accel_threshold: float = -0.5
    velocity_threshold: float = 5.0    # only count inconsistency when the
                                        # other agent is moving fast enough
                                        # to actually be a threat
    smoothing: float = 0.3


@dataclass
class Detector:
    params: DetectorParams
    score: float = 0.0
    triggered: bool = False
    trigger_step: int = -1
    _step: int = 0

    def observe(
        self,
        signal: int,
        observed_accel: float,
        observed_velocity: float,
        in_conflict: bool = True,
    ) -> None:
        """Update the score with one observation.

        Args:
            signal: yielding signal emitted by the other agent (0 or 1).
            observed_accel: estimated acceleration of the other agent (m/s^2).
            observed_velocity: current velocity of the other agent (m/s).
            in_conflict: True iff still in the conflict phase. When False,
                the score is frozen — neither updated nor re-tested.
        """
        if not in_conflict:
            self._step += 1
            return

        raw = 0.0
        if signal == 1 and observed_velocity > self.params.velocity_threshold:
            raw = max(0.0, observed_accel - self.params.yield_accel_threshold)
        self.score = (1.0 - self.params.smoothing) * self.score + self.params.smoothing * raw

        if not self.triggered and self.score > self.params.tau:
            self.triggered = True
            self.trigger_step = self._step
        self._step += 1

    def reset(self) -> None:
        self.score = 0.0
        self.triggered = False
        self.trigger_step = -1
        self._step = 0


def latency_and_false_alarms(
    detected: bool,
    trigger_step: int,
    is_adversarial_episode: bool,
    adversarial_start_step: int = 0,
) -> dict:
    """Per-episode detection metrics.

    Args:
        detected: did the detector fire during the episode?
        trigger_step: step at which it fired (-1 if never).
        is_adversarial_episode: ground truth.
        adversarial_start_step: step at which adversarial signal began.

    Returns:
        Dict with true_positive, false_alarm, missed (bools) and latency
        (int or None — only defined for true positives).
    """
    true_positive = is_adversarial_episode and detected
    false_alarm = (not is_adversarial_episode) and detected
    missed = is_adversarial_episode and (not detected)
    latency = max(0, trigger_step - adversarial_start_step) if true_positive else None
    return {
        "true_positive": true_positive,
        "false_alarm": false_alarm,
        "missed": missed,
        "latency": latency,
    }