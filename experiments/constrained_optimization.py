"""Formulation 2: constrained optimization.

Problem:
    minimize    latency(tau)
    subject to  FA(tau) <= FA_max

The unconstrained latency minimum is at low tau (immediate detection); the
constraint is satisfied at high tau (rare detection -> few false alarms),
so the constraint genuinely binds. The optimum sits at the boundary
FA(tau*) = FA_max in the interior of the tau interval.

Two methods, comparing their behavior:
    - Quadratic penalty: solve a sequence of unconstrained problems with
      increasing penalty parameter rho. Allows constraint violation.
    - Log barrier: solve a sequence of unconstrained problems with decreasing
      barrier parameter mu. Requires a feasible starting point.

Each inner problem is 1D and solved by golden-section search.

Usage:
    python -m experiments.constrained_optimization
"""

from dataclasses import dataclass
from pathlib import Path
import json
import math

from src.evaluation import EvalConfig, evaluate_tau, EvalResult
from experiments.optimize_threshold import golden_section_search


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------

def _latency_value(result: EvalResult, miss_default: float = 5.0) -> float:
    """Return latency, substituting miss_default when no detections occur.
    Same convention as Formulation 1, so loss values are comparable.
    """
    if math.isnan(result.mean_detection_latency):
        return miss_default
    return result.mean_detection_latency


# ---------------------------------------------------------------------------
# Penalty method
# ---------------------------------------------------------------------------

def penalty_method(
    cfg: EvalConfig,
    fa_max: float,
    rho_schedule: list[float],
    tau_lo: float = 0.01,
    tau_hi: float = 1.0,
    inner_tol: float = 1e-3,
) -> dict:
    """Quadratic penalty method.

    For each rho in the schedule, solve:
        min_tau  latency(tau) + rho * max(0, FA(tau) - fa_max)^2

    using golden-section search.

    The penalty is one-sided (only when the constraint is violated), so
    increasing rho squeezes feasibility from above without affecting feasible
    points. Returns the trajectory of solutions as rho increases.
    """
    history = []
    for rho in rho_schedule:
        def f(tau):
            result = evaluate_tau(tau, cfg)
            latency = _latency_value(result)
            violation = max(0.0, result.false_alarm_rate - fa_max)
            penalty = rho * violation ** 2
            return latency + penalty, result

        best_tau, search = golden_section_search(f, tau_lo, tau_hi, tol=inner_tol)
        best_result = evaluate_tau(best_tau, cfg)
        history.append({
            "rho": rho,
            "best_tau": best_tau,
            "latency": _latency_value(best_result),
            "false_alarm_rate": best_result.false_alarm_rate,
            "feasible": best_result.false_alarm_rate <= fa_max + 1e-9,
            "result": best_result.as_dict(),
            "n_inner_evals": len(search.evaluations),
        })
    return {"method": "penalty", "fa_max": fa_max, "history": history}


# ---------------------------------------------------------------------------
# Barrier method
# ---------------------------------------------------------------------------

def barrier_method(
    cfg: EvalConfig,
    fa_max: float,
    mu_schedule: list[float],
    tau_lo_feasible: float,
    tau_hi: float = 1.0,
    inner_tol: float = 1e-3,
) -> dict:
    """Logarithmic barrier method.

    For each mu in the (decreasing) schedule, solve:
        min_tau  latency(tau) - mu * log(fa_max - FA(tau))

    using golden-section search. The barrier blows up as FA approaches the
    bound, so the inner optimum stays strictly feasible. As mu shrinks, the
    barrier weakens and the iterate approaches the constraint boundary.

    Args:
        tau_lo_feasible: lower bound of the search bracket. Must be feasible
            (FA at tau_lo_feasible <= fa_max), otherwise the barrier is
            undefined. The caller is responsible for picking this — typically
            from a coarse pre-sweep.
    """
    # Verify the bracket is feasible before starting.
    init_result = evaluate_tau(tau_lo_feasible, cfg)
    if init_result.false_alarm_rate > fa_max:
        raise ValueError(
            f"tau_lo_feasible={tau_lo_feasible} is infeasible: "
            f"FA={init_result.false_alarm_rate:.3f} > fa_max={fa_max}. "
            f"Pick a higher starting tau."
        )

    history = []
    for mu in mu_schedule:
        def f(tau):
            result = evaluate_tau(tau, cfg)
            latency = _latency_value(result)
            slack = fa_max - result.false_alarm_rate
            if slack <= 0:
                # Infeasible: return a large finite penalty so golden-section
                # can compare values, rather than +inf.
                return 1e6, result
            barrier = -mu * math.log(slack)
            return latency + barrier, result

        best_tau, search = golden_section_search(
            f, tau_lo_feasible, tau_hi, tol=inner_tol
        )
        best_result = evaluate_tau(best_tau, cfg)
        history.append({
            "mu": mu,
            "best_tau": best_tau,
            "latency": _latency_value(best_result),
            "false_alarm_rate": best_result.false_alarm_rate,
            "feasible": best_result.false_alarm_rate <= fa_max + 1e-9,
            "result": best_result.as_dict(),
            "n_inner_evals": len(search.evaluations),
        })
    return {"method": "barrier", "fa_max": fa_max, "history": history}


# ---------------------------------------------------------------------------
# Helper: find a feasible starting point for the barrier method.
# ---------------------------------------------------------------------------

def find_feasible_tau(cfg: EvalConfig, fa_max: float, tau_grid: list[float]) -> float:
    """Coarse linear scan to find the smallest tau in tau_grid with FA <= fa_max.
    Used to initialize the barrier method.
    """
    for tau in tau_grid:
        result = evaluate_tau(tau, cfg)
        if result.false_alarm_rate <= fa_max:
            return tau
    raise RuntimeError(
        f"No tau in {tau_grid} satisfies FA <= {fa_max}. Loosen the bound."
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    cfg: EvalConfig | None = None,
    fa_max: float = 0.05,
    out_dir: str | Path | None = None,
) -> dict:
    cfg = cfg or EvalConfig(n_episodes=200, base_seed=42, accel_noise_std=2.0)

    print(f"Constrained problem: minimize latency s.t. FA <= {fa_max:.2%}")
    print(f"Eval config: n_episodes={cfg.n_episodes}  noise_std={cfg.accel_noise_std}")
    print()

    # ----- Penalty method -----
    print("=== Penalty method ===")
    rho_schedule = [1.0, 10.0, 100.0, 1000.0]
    penalty_out = penalty_method(cfg, fa_max=fa_max, rho_schedule=rho_schedule)
    for entry in penalty_out["history"]:
        feas = "✓" if entry["feasible"] else "✗"
        print(f"  rho={entry['rho']:7.1f}  tau={entry['best_tau']:.4f}  "
              f"latency={entry['latency']:.3f}s  FA={entry['false_alarm_rate']:.2%}  "
              f"{feas}  ({entry['n_inner_evals']} evals)")
    print()

    # ----- Barrier method -----
    print("=== Barrier method ===")
    coarse_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tau_lo_feasible = find_feasible_tau(cfg, fa_max=fa_max, tau_grid=coarse_grid)
    print(f"  Feasible starting bracket lower bound: tau={tau_lo_feasible}")
    mu_schedule = [1.0, 0.1, 0.01, 0.001]
    barrier_out = barrier_method(
        cfg, fa_max=fa_max, mu_schedule=mu_schedule,
        tau_lo_feasible=tau_lo_feasible,
    )
    for entry in barrier_out["history"]:
        feas = "✓" if entry["feasible"] else "✗"
        print(f"  mu={entry['mu']:.4f}  tau={entry['best_tau']:.4f}  "
              f"latency={entry['latency']:.3f}s  FA={entry['false_alarm_rate']:.2%}  "
              f"{feas}  ({entry['n_inner_evals']} evals)")
    print()

    # ----- Comparison -----
    p_final = penalty_out["history"][-1]
    b_final = barrier_out["history"][-1]
    print("=== Final comparison ===")
    print(f"  Penalty: tau={p_final['best_tau']:.4f}  "
          f"latency={p_final['latency']:.3f}s  FA={p_final['false_alarm_rate']:.2%}")
    print(f"  Barrier: tau={b_final['best_tau']:.4f}  "
          f"latency={b_final['latency']:.3f}s  FA={b_final['false_alarm_rate']:.2%}")

    out = {
        "fa_max": fa_max,
        "penalty": penalty_out,
        "barrier": barrier_out,
        "config": {
            "n_episodes": cfg.n_episodes,
            "adversarial_rate": cfg.adversarial_rate,
            "accel_noise_std": cfg.accel_noise_std,
            "base_seed": cfg.base_seed,
        },
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "result.json"
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nSaved to {path}")

    return out


if __name__ == "__main__":
    run(out_dir="results/constrained_optimization")