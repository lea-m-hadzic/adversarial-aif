"""Extended experiments:

    1. Seed variance per method: re-run each of the six methods (4 F1 weight
       settings + F2 penalty + F2 barrier) across multiple seeds, reporting
       mean ± std for tau, latency, FA, and recall.
    2. Conflict-phase ablation: compare detection performance with and
       without conflict-phase gating, holding everything else fixed.

Outputs:
    results/rigor/seed_variance_per_method/result.json
    results/rigor/conflict_gating_ablation/result.json

Usage:
    python -m experiments.extended_experiments
"""

from dataclasses import dataclass, replace
from pathlib import Path
import json
import statistics
import time
import random

from src.evaluation import EvalConfig, evaluate_tau, EvalResult
from src.agent import Agent
from src.simulator import Simulation
from src.beliefs import LikelihoodModel, update
from src.objective import EFEWeights, CollisionModel, select_action
from src.adversarial import honest_yielder, honest_nonyielder, deceptive
from src.detection import DetectorParams, Detector

from experiments.optimize_threshold import (
    LossWeights, loss_from_result, golden_section_search,
)
from experiments.constrained_optimization import (
    penalty_method, barrier_method, find_feasible_tau,
)


RESULTS_ROOT = Path("results/rigor")


# ---------------------------------------------------------------------------
# Part 1: Per-method seed variance
# ---------------------------------------------------------------------------

F1_WEIGHT_SETTINGS = {
    "balanced":      LossWeights(latency=1.0, false_alarm=5.0, miss=5.0),
    "latency-heavy": LossWeights(latency=5.0, false_alarm=1.0, miss=1.0),
    "FA-heavy":      LossWeights(latency=1.0, false_alarm=10.0, miss=1.0),
    "miss-heavy":    LossWeights(latency=1.0, false_alarm=1.0, miss=10.0),
}


def _f1_optimize(cfg: EvalConfig, weights: LossWeights):
    def f(tau):
        r = evaluate_tau(tau, cfg)
        return loss_from_result(r, weights), r
    best_tau, _ = golden_section_search(f, 0.01, 1.0, tol=1e-3)
    return best_tau, evaluate_tau(best_tau, cfg)


def _summary(runs, key):
    vals = [r[key] for r in runs]
    if not vals:
        return None
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def seed_variance_per_method(seeds, base_cfg: EvalConfig, out_dir: Path):
    """For each of the 6 methods, run across `seeds` and aggregate stats."""
    print("\n=== Part 1: per-method seed variance ===")
    fa_max = 0.05
    rho_schedule = [1.0, 10.0, 100.0, 1000.0]
    mu_schedule = [1.0, 0.1, 0.01, 0.001]

    methods = {}

    # --- F1: four weight settings ---
    for name, weights in F1_WEIGHT_SETTINGS.items():
        print(f"\n  F1 [{name}]")
        per_seed = []
        for seed in seeds:
            cfg = replace(base_cfg, base_seed=seed)
            best_tau, best_res = _f1_optimize(cfg, weights)
            entry = {
                "seed": seed,
                "tau": best_tau,
                "latency": best_res.mean_detection_latency
                if best_res.mean_detection_latency == best_res.mean_detection_latency  # nan check
                else 5.0,
                "false_alarm_rate": best_res.false_alarm_rate,
                "collision_rate": best_res.collision_rate,
                "recall": best_res.detection_recall,
            }
            # Replace NaN latency with the F1 default for honesty in reporting.
            import math
            if math.isnan(entry["latency"]):
                entry["latency"] = 5.0
            per_seed.append(entry)
            print(f"    seed={seed:3d}  tau={entry['tau']:.4f}  "
                  f"lat={entry['latency']:.3f}s  FA={entry['false_alarm_rate']:.2%}  "
                  f"recall={entry['recall']:.2%}")
        methods[f"F1:{name}"] = {
            "per_seed": per_seed,
            "summary": {k: _summary(per_seed, k)
                        for k in ["tau", "latency", "false_alarm_rate",
                                  "collision_rate", "recall"]},
        }

    # --- F2 penalty ---
    print(f"\n  F2 [penalty]")
    per_seed = []
    for seed in seeds:
        cfg = replace(base_cfg, base_seed=seed)
        pen = penalty_method(cfg, fa_max=fa_max, rho_schedule=rho_schedule)
        final = pen["history"][-1]
        entry = {
            "seed": seed,
            "tau": final["best_tau"],
            "latency": final["latency"],
            "false_alarm_rate": final["false_alarm_rate"],
            "collision_rate": final["result"]["collision_rate"],
            "recall": final["result"]["detection_recall"],
        }
        per_seed.append(entry)
        print(f"    seed={seed:3d}  tau={entry['tau']:.4f}  "
              f"lat={entry['latency']:.3f}s  FA={entry['false_alarm_rate']:.2%}  "
              f"recall={entry['recall']:.2%}")
    methods["F2:penalty"] = {
        "per_seed": per_seed,
        "summary": {k: _summary(per_seed, k)
                    for k in ["tau", "latency", "false_alarm_rate",
                              "collision_rate", "recall"]},
    }

    # --- F2 barrier ---
    print(f"\n  F2 [barrier]")
    per_seed = []
    coarse_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for seed in seeds:
        cfg = replace(base_cfg, base_seed=seed)
        try:
            tau_feasible = find_feasible_tau(cfg, fa_max=fa_max,
                                             tau_grid=coarse_grid)
            bar = barrier_method(cfg, fa_max=fa_max, mu_schedule=mu_schedule,
                                 tau_lo_feasible=tau_feasible)
            final = bar["history"][-1]
            entry = {
                "seed": seed,
                "tau": final["best_tau"],
                "latency": final["latency"],
                "false_alarm_rate": final["false_alarm_rate"],
                "collision_rate": final["result"]["collision_rate"],
                "recall": final["result"]["detection_recall"],
            }
        except RuntimeError as e:
            print(f"    seed={seed:3d}  SKIPPED ({e})")
            continue
        per_seed.append(entry)
        print(f"    seed={seed:3d}  tau={entry['tau']:.4f}  "
              f"lat={entry['latency']:.3f}s  FA={entry['false_alarm_rate']:.2%}  "
              f"recall={entry['recall']:.2%}")
    methods["F2:barrier"] = {
        "per_seed": per_seed,
        "summary": {k: _summary(per_seed, k)
                    for k in ["tau", "latency", "false_alarm_rate",
                              "collision_rate", "recall"]},
    }

    # Summary table
    print("\n  Summary (mean ± std across seeds):")
    print(f"  {'Method':<22} {'tau':>14} {'latency (s)':>16} {'FA':>14} {'recall':>14}")
    for name, m in methods.items():
        s = m["summary"]
        print(f"  {name:<22} "
              f"{s['tau']['mean']:.4f}±{s['tau']['std']:.4f}   "
              f"{s['latency']['mean']:.3f}±{s['latency']['std']:.3f}   "
              f"{s['false_alarm_rate']['mean']*100:.2f}±{s['false_alarm_rate']['std']*100:.2f}%   "
              f"{s['recall']['mean']*100:.2f}±{s['recall']['std']*100:.2f}%")

    out = {"experiment": "seed_variance_per_method",
           "seeds": list(seeds), "methods": methods}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


# ---------------------------------------------------------------------------
# Part 2: Conflict-phase gating ablation
# ---------------------------------------------------------------------------

def _build_policy_no_gating(tau: float, cfg: EvalConfig, noise_rng):
    """Same as the standard EFE+detection policy, but the detector's
    in_conflict argument is always True (no gating)."""
    weights = EFEWeights(alpha=cfg.efe_alpha, beta=cfg.efe_beta,
                         gamma=cfg.efe_gamma)
    model = CollisionModel(intersection_width=cfg.intersection_width,
                           horizon_seconds=4.0, dt=cfg.dt)
    likelihood = LikelihoodModel()
    detector = Detector(params=DetectorParams(
        tau=tau,
        yield_accel_threshold=cfg.detector_yield_accel_threshold,
        velocity_threshold=cfg.detector_velocity_threshold,
        smoothing=cfg.detector_smoothing,
    ))
    state = {"belief": 0.5, "last_v": None, "detector": detector}

    def policy(self_agent, other_agent, t):
        if state["last_v"] is not None:
            true_accel = (other_agent.v - state["last_v"]) / cfg.dt
            obs_accel = true_accel + noise_rng.gauss(0.0, cfg.accel_noise_std)
            # KEY DIFFERENCE: always pass in_conflict=True
            detector.observe(other_agent.signal, obs_accel, other_agent.v,
                             in_conflict=True)
            effective_signal = 0 if detector.triggered else other_agent.signal
            state["belief"] = update(state["belief"], obs_accel,
                                     effective_signal, likelihood)
        state["last_v"] = other_agent.v
        action = select_action(self_agent, other_agent, state["belief"],
                               list(cfg.action_set), weights, model)
        return action, 0
    policy.state = state
    return policy


def _run_episode_no_gating(tau, cfg, episode_seed):
    rng = random.Random(episode_seed)
    noise_rng = random.Random(episode_seed + 100_000)
    is_adv = rng.random() < cfg.adversarial_rate
    d_a, d_b = rng.uniform(*cfg.d_range), rng.uniform(*cfg.d_range)
    v_a, v_b = rng.uniform(*cfg.v_range), rng.uniform(*cfg.v_range)
    if is_adv:
        start = rng.randint(*cfg.adversarial_start_range)
        policy_a = deceptive(start_step=start)
    else:
        start = 0
        if rng.random() < cfg.honest_yield_rate:
            policy_a = honest_yielder(braking_accel=-3.0)
        else:
            policy_a = honest_nonyielder()
    policy_b = _build_policy_no_gating(tau, cfg, noise_rng)
    sim = Simulation(agent_a=Agent(d=d_a, v=v_a),
                     agent_b=Agent(d=d_b, v=v_b),
                     dt=cfg.dt, max_steps=cfg.max_steps,
                     intersection_width=cfg.intersection_width)
    trace = sim.run_episode(policy_a, policy_b)
    detector = policy_b.state["detector"]
    return {
        "is_adversarial": is_adv,
        "outcome": trace.outcome,
        "final_step": trace.final_step,
        "detected": detector.triggered,
        "trigger_step": detector.trigger_step,
        "adversarial_start_step": start,
    }


def evaluate_tau_no_gating(tau, cfg):
    """Aggregated metrics for the no-gating variant. Same logic as
    evaluate_tau() in src/evaluation.py.
    """
    import math
    per_episode = [
        _run_episode_no_gating(tau, cfg, episode_seed=cfg.base_seed + i)
        for i in range(cfg.n_episodes)
    ]
    n_adv = sum(1 for e in per_episode if e["is_adversarial"])
    n_hon = cfg.n_episodes - n_adv
    n_collision = sum(1 for e in per_episode if e["outcome"] == "collision")

    latencies, pre_onset = [], 0
    for e in per_episode:
        if e["is_adversarial"] and e["detected"]:
            delta = e["trigger_step"] - e["adversarial_start_step"]
            if delta >= 0:
                latencies.append(delta)
            else:
                pre_onset += 1
    mean_lat = (sum(latencies) * cfg.dt / len(latencies)
                if latencies else float("nan"))

    false_alarms = sum(1 for e in per_episode
                       if (not e["is_adversarial"]) and e["detected"])
    false_alarm_rate = (false_alarms + pre_onset) / cfg.n_episodes

    true_pos = len(latencies)
    recall = true_pos / n_adv if n_adv > 0 else float("nan")

    return {
        "tau": tau,
        "collision_rate": n_collision / cfg.n_episodes,
        "mean_detection_latency": mean_lat,
        "false_alarm_rate": false_alarm_rate,
        "detection_recall": recall,
        "n_adversarial": n_adv,
        "n_honest": n_hon,
    }


def conflict_gating_ablation(base_cfg: EvalConfig, taus, out_dir: Path):
    """Compare with and without conflict-phase gating across several tau."""
    print("\n=== Part 2: conflict-phase gating ablation ===")
    results = []
    print(f"  {'tau':>6}  {'gating':<7}  {'collision':>10}  "
          f"{'latency':>10}  {'FA':>8}  {'recall':>8}")
    for tau in taus:
        with_gating = evaluate_tau(tau, base_cfg).as_dict()
        without_gating = evaluate_tau_no_gating(tau, base_cfg)
        results.append({
            "tau": tau,
            "with_gating": with_gating,
            "without_gating": without_gating,
        })
        for label, r in [("ON", with_gating), ("OFF", without_gating)]:
            import math
            lat = r["mean_detection_latency"]
            lat_s = f"{lat:.3f}s" if not math.isnan(lat) else "  n/a "
            print(f"  {tau:6.3f}  {label:<7}  "
                  f"{r['collision_rate']*100:9.2f}%  "
                  f"{lat_s:>10}  "
                  f"{r['false_alarm_rate']*100:7.2f}%  "
                  f"{r['detection_recall']*100:7.2f}%")

    out = {"experiment": "conflict_gating_ablation", "results": results}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_all():
    base_cfg = EvalConfig(n_episodes=200, base_seed=42, accel_noise_std=2.0)
    t0 = time.time()

    seed_variance_per_method(
        seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        base_cfg=base_cfg,
        out_dir=RESULTS_ROOT / "seed_variance_per_method",
    )

    conflict_gating_ablation(
        base_cfg=base_cfg,
        taus=[0.1, 0.3, 0.5, 0.7, 1.0],
        out_dir=RESULTS_ROOT / "conflict_gating_ablation",
    )

    print(f"\nAll Phase 3 experiments complete in {time.time() - t0:.1f}s.")
    print(f"Results saved under {RESULTS_ROOT}/")


if __name__ == "__main__":
    run_all()
