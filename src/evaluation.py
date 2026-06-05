"""Evaluation harness: run episodes and aggregate metrics.

This is f(tau): the function the three optimization formulations minimize.

A single call to evaluate_tau(...) runs n_episodes episodes — a mix of
adversarial and honest scenarios — and returns aggregate metrics:
collision_rate, mean_detection_latency, false_alarm_rate, mean_crossing_time,
deadlock_rate.

Design notes:
    - We sweep over a fixed seed schedule per evaluation so f(tau) is
      deterministic. Two calls with the same tau give identical results.
    - Adversarial vs honest is decided per-episode by an RNG seeded from the
      episode index. The mix ratio is `adversarial_rate`.
    - Observations are noisy: B's estimate of A's acceleration is corrupted by
      Gaussian noise. This makes the inconsistency score a noisy walk rather
      than a clean plateau, and creates a real latency-vs-false-alarms tradeoff.
    - Adversarial episodes start the deceptive signal at a randomized step,
      not always step 0. This makes detection latency a richer function of tau.
"""

from dataclasses import dataclass
from typing import Dict
import random

from src.agent import Agent
from src.simulator import Simulation
from src.beliefs import LikelihoodModel, update
from src.objective import EFEWeights, CollisionModel, select_action
from src.adversarial import honest_yielder, honest_nonyielder, deceptive
from src.detection import DetectorParams, Detector


@dataclass
class EvalConfig:
    n_episodes: int = 200
    adversarial_rate: float = 0.5

    # Initial condition randomization
    d_range: tuple = (15.0, 30.0)
    v_range: tuple = (8.0, 12.0)

    # Honest scenario: probability A is a yielder (vs non-yielder)
    honest_yield_rate: float = 0.5

    # Adversarial scenario: deceptive start step is uniform over this range.
    # This makes detection latency depend on tau in a non-trivial way.
    adversarial_start_range: tuple = (0, 30)

    # Observation noise on B's estimate of A's acceleration (m/s^2).
    # Higher noise => detection at low tau false-alarms more, detection at
    # high tau misses more. This is what creates a real tradeoff.
    accel_noise_std: float = 0.5

    # Simulator
    dt: float = 0.1
    max_steps: int = 200
    intersection_width: float = 4.0

    # Agent / objective / belief / detector params (fixed for the project)
    action_set: tuple = (-3.0, 0.0, 3.0)
    efe_alpha: float = 10.0
    efe_beta: float = 0.1
    efe_gamma: float = 0.0
    detector_yield_accel_threshold: float = -0.5
    detector_velocity_threshold: float = 5.0
    detector_smoothing: float = 0.3

    base_seed: int = 0


@dataclass
class EvalResult:
    tau: float
    collision_rate: float
    deadlock_rate: float
    cleared_rate: float
    mean_crossing_time: float
    mean_detection_latency: float
    false_alarm_rate: float
    detection_recall: float
    n_adversarial: int
    n_honest: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "tau": self.tau,
            "collision_rate": self.collision_rate,
            "deadlock_rate": self.deadlock_rate,
            "cleared_rate": self.cleared_rate,
            "mean_crossing_time": self.mean_crossing_time,
            "mean_detection_latency": self.mean_detection_latency,
            "false_alarm_rate": self.false_alarm_rate,
            "detection_recall": self.detection_recall,
            "n_adversarial": self.n_adversarial,
            "n_honest": self.n_honest,
        }


def _build_efe_detection_policy(tau: float, cfg: EvalConfig, noise_rng: random.Random):
    """Construct a fresh EFE+detection policy for one episode.

    The policy uses `noise_rng` for observation noise so that noise is
    deterministic per-episode (controlled by the per-episode seed).
    """
    weights = EFEWeights(alpha=cfg.efe_alpha, beta=cfg.efe_beta, gamma=cfg.efe_gamma)
    model = CollisionModel(
        intersection_width=cfg.intersection_width,
        horizon_seconds=4.0,
        dt=cfg.dt,
    )
    likelihood = LikelihoodModel()
    detector = Detector(
        params=DetectorParams(
            tau=tau,
            yield_accel_threshold=cfg.detector_yield_accel_threshold,
            velocity_threshold=cfg.detector_velocity_threshold,
            smoothing=cfg.detector_smoothing,
        )
    )
    state = {"belief": 0.5, "last_v": None, "detector": detector}

    def policy(self_agent, other_agent, t):
        if state["last_v"] is not None:
            true_accel = (other_agent.v - state["last_v"]) / cfg.dt
            noise = noise_rng.gauss(0.0, cfg.accel_noise_std)
            observed_accel = true_accel + noise

            in_conflict = other_agent.d > -cfg.intersection_width
            detector.observe(
                other_agent.signal, observed_accel, other_agent.v,
                in_conflict=in_conflict,
            )
            effective_signal = 0 if detector.triggered else other_agent.signal
            state["belief"] = update(
                state["belief"], observed_accel, effective_signal, likelihood
            )
        state["last_v"] = other_agent.v

        action = select_action(
            self_agent, other_agent, state["belief"],
            list(cfg.action_set), weights, model,
        )
        return action, 0

    policy.state = state
    return policy


def _run_one_episode(tau: float, cfg: EvalConfig, episode_seed: int) -> dict:
    """Run a single episode. Returns a dict of per-episode metrics."""
    rng = random.Random(episode_seed)
    # Separate noise RNG so changing the noise std doesn't shuffle which
    # episodes are adversarial.
    noise_rng = random.Random(episode_seed + 100_000)

    is_adversarial = rng.random() < cfg.adversarial_rate

    d_init_a = rng.uniform(*cfg.d_range)
    d_init_b = rng.uniform(*cfg.d_range)
    v_init_a = rng.uniform(*cfg.v_range)
    v_init_b = rng.uniform(*cfg.v_range)

    if is_adversarial:
        start_step = rng.randint(*cfg.adversarial_start_range)
        policy_a = deceptive(start_step=start_step)
    else:
        start_step = 0  # not used for honest episodes
        if rng.random() < cfg.honest_yield_rate:
            policy_a = honest_yielder(braking_accel=-3.0)
        else:
            policy_a = honest_nonyielder()

    policy_b = _build_efe_detection_policy(tau, cfg, noise_rng)

    sim = Simulation(
        agent_a=Agent(d=d_init_a, v=v_init_a),
        agent_b=Agent(d=d_init_b, v=v_init_b),
        dt=cfg.dt,
        max_steps=cfg.max_steps,
        intersection_width=cfg.intersection_width,
    )
    trace = sim.run_episode(policy_a, policy_b)

    detector = policy_b.state["detector"]
    return {
        "is_adversarial": is_adversarial,
        "outcome": trace.outcome,
        "final_step": trace.final_step,
        "detected": detector.triggered,
        "trigger_step": detector.trigger_step,
        "adversarial_start_step": start_step,
    }


def evaluate_tau(tau: float, cfg: EvalConfig) -> EvalResult:
    """Run a full evaluation at a given tau and return aggregate metrics."""
    per_episode = [
        _run_one_episode(tau, cfg, episode_seed=cfg.base_seed + i)
        for i in range(cfg.n_episodes)
    ]

    n_adv = sum(1 for e in per_episode if e["is_adversarial"])
    n_hon = cfg.n_episodes - n_adv

    n_collision = sum(1 for e in per_episode if e["outcome"] == "collision")
    n_deadlock = sum(1 for e in per_episode if e["outcome"] == "deadlock")
    n_cleared = sum(1 for e in per_episode if e["outcome"] == "cleared")

    cleared_steps = [e["final_step"] for e in per_episode if e["outcome"] == "cleared"]
    mean_crossing_time = (
        sum(cleared_steps) * cfg.dt / len(cleared_steps) if cleared_steps else float("nan")
    )

    # Latency now measured from the adversarial onset step, not step 0.
    # A detection that fires before the deceptive signal even starts is a
    # false alarm against the noise floor — count it as latency 0 if it's
    # post-onset, but if it's pre-onset, treat it as a false alarm in the
    # adversarial episode (subtle but matters).
    latencies = []
    pre_onset_detections = 0
    for e in per_episode:
        if e["is_adversarial"] and e["detected"]:
            delta = e["trigger_step"] - e["adversarial_start_step"]
            if delta >= 0:
                latencies.append(delta)
            else:
                pre_onset_detections += 1
    mean_detection_latency = (
        sum(latencies) * cfg.dt / len(latencies) if latencies else float("nan")
    )

    false_alarms = sum(
        1 for e in per_episode
        if (not e["is_adversarial"]) and e["detected"]
    )
    # Pre-onset detections in adversarial episodes also count as "detections
    # that fired without legitimate cause." Add them to the false alarm
    # numerator, normalized by total opportunities for a false alarm.
    false_alarm_rate = (
        (false_alarms + pre_onset_detections) / cfg.n_episodes
    )

    true_positives = len(latencies)  # post-onset detections only
    detection_recall = true_positives / n_adv if n_adv > 0 else float("nan")

    return EvalResult(
        tau=tau,
        collision_rate=n_collision / cfg.n_episodes,
        deadlock_rate=n_deadlock / cfg.n_episodes,
        cleared_rate=n_cleared / cfg.n_episodes,
        mean_crossing_time=mean_crossing_time,
        mean_detection_latency=mean_detection_latency,
        false_alarm_rate=false_alarm_rate,
        detection_recall=detection_recall,
        n_adversarial=n_adv,
        n_honest=n_hon,
    )