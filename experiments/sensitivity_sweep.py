"""Unified sensitivity and robustness sweeps.

Runs four experiments, each writing its own JSON under
results/sensitivity/<sweep_name>/. Designed to be invoked together so a
single command produces the full quantitative basis for the writeup.

Sweeps:
    1. noise_sweep          — vary observation noise std
    2. seed_variance        — run F2 at many seeds, estimate variance
    3. oracle_comparison    — replace detector with a ground-truth oracle
    4. adversarial_rate     — vary the mix of adversarial episodes

Each sweep reuses the F1 and F2 machinery via the existing pipeline so the
loss / constraint / optimizer behavior matches what we ran before.

Usage:
    python -m experiments.sensitivity_sweep
"""

from dataclasses import dataclass, replace
from pathlib import Path
import json
import statistics
import time

from src.evaluation import EvalConfig, evaluate_tau, EvalResult
from src.agent import Agent
from src.simulator import Simulation
from src.beliefs import LikelihoodModel, update
from src.objective import EFEWeights, CollisionModel, select_action
from src.adversarial import honest_yielder, honest_nonyielder, deceptive
import random

from experiments.optimize_threshold import (
    LossWeights, loss_from_result, golden_section_search,
)
from experiments.constrained_optimization import (
    penalty_method, barrier_method, find_feasible_tau, _latency_value as _f2_latency,
)
from experiments.pareto_analysis import sweep as pareto_sweep, mark_pareto_front


RESULTS_ROOT = Path("results/sensitivity")


# ---------------------------------------------------------------------------
# Sweep 1: noise level
# ---------------------------------------------------------------------------

def noise_sweep(noise_levels: list[float], base_cfg: EvalConfig, out_dir: Path) -> dict:
    """For each noise level, run F1 (balanced weights) and a Pareto sweep.

    Tells us how the optimum shifts as sensor noise grows. Key claim
    candidate: at low noise there's no trade-off; at high noise the
    optimization matters.
    """
    print("\n=== Sweep 1: noise level ===")
    weights = LossWeights()
    results = []
    for sigma in noise_levels:
        cfg = replace(base_cfg, accel_noise_std=sigma)

        # F1: scalar optimum
        def f(tau):
            r = evaluate_tau(tau, cfg)
            return loss_from_result(r, weights), r
        best_tau, hist = golden_section_search(f, 0.01, 1.0, tol=1e-3)
        best_result = evaluate_tau(best_tau, cfg)

        # Pareto: dense sweep
        tau_grid = [round(0.02 * i, 4) for i in range(1, 51)]
        points = pareto_sweep(cfg, tau_grid)
        points = mark_pareto_front(points)
        n_pareto = sum(1 for p in points if p.is_pareto)

        entry = {
            "sigma": sigma,
            "f1_best_tau": best_tau,
            "f1_best_loss": loss_from_result(best_result, weights),
            "f1_best_result": best_result.as_dict(),
            "pareto_n_points": len(points),
            "pareto_n_frontier": n_pareto,
            "pareto_points": [p.as_dict() for p in points],
        }
        results.append(entry)
        print(f"  sigma={sigma:.2f}  tau*={best_tau:.4f}  "
              f"latency={best_result.mean_detection_latency:.3f}s  "
              f"FA={best_result.false_alarm_rate:.2%}  "
              f"recall={best_result.detection_recall:.2%}  "
              f"|frontier|={n_pareto}")

    out = {"sweep": "noise", "results": results}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


# ---------------------------------------------------------------------------
# Sweep 2: seed variance
# ---------------------------------------------------------------------------

def seed_variance(seeds: list[int], base_cfg: EvalConfig, out_dir: Path) -> dict:
    """Run F2 (penalty method) at multiple seeds. Reports mean and std of
    tau*, latency, FA, collision rate. This is what gives error bars to
    every other number in the writeup.
    """
    print("\n=== Sweep 2: seed variance (F2 penalty) ===")
    runs = []
    fa_max = 0.05
    rho_schedule = [1.0, 10.0, 100.0, 1000.0]
    for seed in seeds:
        cfg = replace(base_cfg, base_seed=seed)
        pen = penalty_method(cfg, fa_max=fa_max, rho_schedule=rho_schedule)
        final = pen["history"][-1]
        runs.append({
            "seed": seed,
            "tau": final["best_tau"],
            "latency": final["latency"],
            "false_alarm_rate": final["false_alarm_rate"],
            "collision_rate": final["result"]["collision_rate"],
            "recall": final["result"]["detection_recall"],
            "feasible": final["feasible"],
        })
        print(f"  seed={seed:3d}  tau={final['best_tau']:.4f}  "
              f"latency={final['latency']:.3f}s  "
              f"FA={final['false_alarm_rate']:.2%}  "
              f"recall={final['result']['detection_recall']:.2%}")

    # Summary stats
    def stats(key):
        vals = [r[key] for r in runs]
        return {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals), "max": max(vals)}
    summary = {k: stats(k) for k in ["tau", "latency", "false_alarm_rate", "collision_rate", "recall"]}
    print(f"\n  tau:      {summary['tau']['mean']:.4f} ± {summary['tau']['std']:.4f}")
    print(f"  latency:  {summary['latency']['mean']:.3f}s ± {summary['latency']['std']:.3f}s")
    print(f"  FA:       {summary['false_alarm_rate']['mean']:.2%} ± {summary['false_alarm_rate']['std']:.2%}")
    print(f"  recall:   {summary['recall']['mean']:.2%} ± {summary['recall']['std']:.2%}")

    out = {"sweep": "seed_variance", "fa_max": fa_max, "runs": runs, "summary": summary}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


# ---------------------------------------------------------------------------
# Sweep 3: oracle comparison
# ---------------------------------------------------------------------------

def _build_oracle_policy(cfg: EvalConfig, is_adversarial: bool, noise_rng: random.Random):
    """EFE policy that uses a ground-truth oracle instead of the detector.
    If is_adversarial, the oracle immediately discounts signals.
    """
    weights = EFEWeights(alpha=cfg.efe_alpha, beta=cfg.efe_beta, gamma=cfg.efe_gamma)
    model = CollisionModel(intersection_width=cfg.intersection_width,
                           horizon_seconds=4.0, dt=cfg.dt)
    likelihood = LikelihoodModel()
    state = {"belief": 0.5, "last_v": None}

    def policy(self_agent, other_agent, t):
        if state["last_v"] is not None:
            true_accel = (other_agent.v - state["last_v"]) / cfg.dt
            observed_accel = true_accel + noise_rng.gauss(0.0, cfg.accel_noise_std)
            # Oracle: ignore signal entirely in adversarial episodes
            effective_signal = 0 if is_adversarial else other_agent.signal
            state["belief"] = update(state["belief"], observed_accel,
                                     effective_signal, likelihood)
        state["last_v"] = other_agent.v
        action = select_action(self_agent, other_agent, state["belief"],
                               list(cfg.action_set), weights, model)
        return action, 0
    return policy


def _run_oracle_episode(cfg: EvalConfig, episode_seed: int) -> dict:
    rng = random.Random(episode_seed)
    noise_rng = random.Random(episode_seed + 100_000)
    is_adv = rng.random() < cfg.adversarial_rate
    d_a, d_b = rng.uniform(*cfg.d_range), rng.uniform(*cfg.d_range)
    v_a, v_b = rng.uniform(*cfg.v_range), rng.uniform(*cfg.v_range)

    if is_adv:
        start = rng.randint(*cfg.adversarial_start_range)
        policy_a = deceptive(start_step=start)
    else:
        if rng.random() < cfg.honest_yield_rate:
            policy_a = honest_yielder(braking_accel=-3.0)
        else:
            policy_a = honest_nonyielder()

    policy_b = _build_oracle_policy(cfg, is_adv, noise_rng)
    sim = Simulation(agent_a=Agent(d=d_a, v=v_a), agent_b=Agent(d=d_b, v=v_b),
                     dt=cfg.dt, max_steps=cfg.max_steps,
                     intersection_width=cfg.intersection_width)
    trace = sim.run_episode(policy_a, policy_b)
    return {"is_adversarial": is_adv, "outcome": trace.outcome}


def oracle_comparison(base_cfg: EvalConfig, out_dir: Path) -> dict:
    """Run the oracle agent and report collision/deadlock/clear rates.
    Compare to the F2 penalty-method result on the same config.
    """
    print("\n=== Sweep 3: oracle comparison ===")
    n = base_cfg.n_episodes
    episodes = [_run_oracle_episode(base_cfg, base_cfg.base_seed + i) for i in range(n)]
    n_coll = sum(1 for e in episodes if e["outcome"] == "collision")
    n_dead = sum(1 for e in episodes if e["outcome"] == "deadlock")
    n_clear = sum(1 for e in episodes if e["outcome"] == "cleared")
    oracle_result = {
        "collision_rate": n_coll / n,
        "deadlock_rate": n_dead / n,
        "cleared_rate": n_clear / n,
    }
    print(f"  Oracle:    collision={oracle_result['collision_rate']:.2%}  "
          f"deadlock={oracle_result['deadlock_rate']:.2%}  "
          f"cleared={oracle_result['cleared_rate']:.2%}")

    # Detector baseline (F2 penalty) on the same config
    pen = penalty_method(base_cfg, fa_max=0.05, rho_schedule=[1.0, 10.0, 100.0, 1000.0])
    detector_final = pen["history"][-1]
    detector_result = {
        "tau": detector_final["best_tau"],
        "collision_rate": detector_final["result"]["collision_rate"],
        "deadlock_rate": detector_final["result"]["deadlock_rate"],
        "cleared_rate": detector_final["result"]["cleared_rate"],
        "false_alarm_rate": detector_final["false_alarm_rate"],
    }
    print(f"  Detector:  collision={detector_result['collision_rate']:.2%}  "
          f"deadlock={detector_result['deadlock_rate']:.2%}  "
          f"cleared={detector_result['cleared_rate']:.2%}  "
          f"(tau*={detector_result['tau']:.3f})")

    # Naive baseline: no detection (tau = infinity simulated by large value)
    naive_cfg = base_cfg
    naive_result_raw = evaluate_tau(1e6, naive_cfg)
    naive_result = {
        "collision_rate": naive_result_raw.collision_rate,
        "deadlock_rate": naive_result_raw.deadlock_rate,
        "cleared_rate": naive_result_raw.cleared_rate,
    }
    print(f"  Naive:     collision={naive_result['collision_rate']:.2%}  "
          f"deadlock={naive_result['deadlock_rate']:.2%}  "
          f"cleared={naive_result['cleared_rate']:.2%}")

    # Gap analysis
    if naive_result['collision_rate'] - oracle_result['collision_rate'] > 1e-9:
        fraction_recovered = (
            (naive_result['collision_rate'] - detector_result['collision_rate']) /
            (naive_result['collision_rate'] - oracle_result['collision_rate'])
        )
        print(f"\n  Detector recovers {fraction_recovered:.1%} of the "
              f"naive→oracle collision gap.")
    else:
        fraction_recovered = None

    out = {
        "sweep": "oracle_comparison",
        "oracle": oracle_result,
        "detector": detector_result,
        "naive": naive_result,
        "fraction_of_gap_recovered": fraction_recovered,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


# ---------------------------------------------------------------------------
# Sweep 4: adversarial rate
# ---------------------------------------------------------------------------

def adversarial_rate_sweep(rates: list[float], base_cfg: EvalConfig, out_dir: Path) -> dict:
    """For each adversarial rate, run F2 penalty method. Tells us how the
    optimum depends on prior expectation of deception.
    """
    print("\n=== Sweep 4: adversarial rate ===")
    fa_max = 0.05
    results = []
    for r in rates:
        cfg = replace(base_cfg, adversarial_rate=r)
        pen = penalty_method(cfg, fa_max=fa_max, rho_schedule=[1.0, 10.0, 100.0, 1000.0])
        final = pen["history"][-1]
        entry = {
            "adversarial_rate": r,
            "tau": final["best_tau"],
            "latency": final["latency"],
            "false_alarm_rate": final["false_alarm_rate"],
            "collision_rate": final["result"]["collision_rate"],
            "recall": final["result"]["detection_recall"],
        }
        results.append(entry)
        print(f"  adv_rate={r:.2f}  tau*={entry['tau']:.4f}  "
              f"latency={entry['latency']:.3f}s  "
              f"FA={entry['false_alarm_rate']:.2%}  "
              f"collision={entry['collision_rate']:.2%}  "
              f"recall={entry['recall']:.2%}")

    out = {"sweep": "adversarial_rate", "fa_max": fa_max, "results": results}
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

    noise_sweep(
        noise_levels=[0.5, 1.0, 1.5, 2.0, 3.0],
        base_cfg=base_cfg,
        out_dir=RESULTS_ROOT / "noise",
    )

    seed_variance(
        seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        base_cfg=base_cfg,
        out_dir=RESULTS_ROOT / "seed_variance",
    )

    oracle_comparison(
        base_cfg=base_cfg,
        out_dir=RESULTS_ROOT / "oracle",
    )

    adversarial_rate_sweep(
        rates=[0.1, 0.3, 0.5, 0.7, 0.9],
        base_cfg=base_cfg,
        out_dir=RESULTS_ROOT / "adversarial_rate",
    )

    print(f"\nAll sweeps complete in {time.time() - t0:.1f}s.")
    print(f"Results saved under {RESULTS_ROOT}/")


if __name__ == "__main__":
    run_all()
