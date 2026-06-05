"""Formulation 3: multi-objective Pareto analysis.

Sweep tau across a range, evaluate at each point, and extract the Pareto
frontier in objective space. Three objectives:
    - mean detection latency (seconds, lower is better)
    - false alarm rate (lower is better)
    - collision rate (lower is better)

For each (tau, result) point, we test whether it is dominated by any other
point in the sweep. Non-dominated points form the Pareto frontier.

We also overlay the optima found by Formulations 1 and 2 (if their result
JSONs exist) so the writeup can show how different problem formulations
project onto the same trade-off surface.

Usage:
    python -m experiments.pareto_analysis
"""

from dataclasses import dataclass
from pathlib import Path
import json
import math

from src.evaluation import EvalConfig, evaluate_tau


# ---------------------------------------------------------------------------
# Pareto helpers
# ---------------------------------------------------------------------------

@dataclass
class PareToPoint:
    tau: float
    latency: float
    false_alarm_rate: float
    collision_rate: float
    recall: float
    is_pareto: bool = False

    def as_dict(self):
        return {
            "tau": self.tau,
            "latency": self.latency,
            "false_alarm_rate": self.false_alarm_rate,
            "collision_rate": self.collision_rate,
            "recall": self.recall,
            "is_pareto": self.is_pareto,
        }


def dominates(a: PareToPoint, b: PareToPoint) -> bool:
    """True if `a` dominates `b` — i.e., a is at least as good on every
    objective and strictly better on at least one. All three objectives are
    minimized.
    """
    not_worse = (
        a.latency <= b.latency
        and a.false_alarm_rate <= b.false_alarm_rate
        and a.collision_rate <= b.collision_rate
    )
    strictly_better = (
        a.latency < b.latency
        or a.false_alarm_rate < b.false_alarm_rate
        or a.collision_rate < b.collision_rate
    )
    return not_worse and strictly_better


def mark_pareto_front(points: list[PareToPoint]) -> list[PareToPoint]:
    """Mark each point as Pareto-optimal or not. O(n^2)."""
    for i, p in enumerate(points):
        is_pareto = True
        for j, q in enumerate(points):
            if i == j:
                continue
            if dominates(q, p):
                is_pareto = False
                break
        p.is_pareto = is_pareto
    return points


# ---------------------------------------------------------------------------
# Sweep + analysis
# ---------------------------------------------------------------------------

def _latency_value(latency_seconds: float, miss_default: float = 5.0) -> float:
    """Substitute a default for NaN latencies (no detections), so high-tau
    points are still comparable on the frontier rather than being dropped.
    """
    if math.isnan(latency_seconds):
        return miss_default
    return latency_seconds


def sweep(cfg: EvalConfig, tau_values: list[float]) -> list[PareToPoint]:
    points = []
    for tau in tau_values:
        result = evaluate_tau(tau, cfg)
        points.append(PareToPoint(
            tau=tau,
            latency=_latency_value(result.mean_detection_latency),
            false_alarm_rate=result.false_alarm_rate,
            collision_rate=result.collision_rate,
            recall=result.detection_recall,
        ))
    return points


def load_formulation_optima(results_root: Path) -> dict:
    """Load the optima from Formulations 1 and 2, if they exist.

    Returns a dict mapping label -> {tau, latency, FA, collision} that we'll
    overlay on the Pareto plot.
    """
    overlays = {}

    # Formulation 1: sensitivity sweep wrote multiple subfolders.
    f1_dir = results_root / "optimize_threshold"
    if f1_dir.exists():
        for sub in sorted(f1_dir.iterdir()):
            if sub.is_dir():
                jsonp = sub / "result.json"
                if jsonp.exists():
                    with open(jsonp) as fh:
                        data = json.load(fh)
                    res = data["best_result"]
                    overlays[f"F1:{sub.name}"] = {
                        "tau": res["tau"],
                        "latency": _latency_value(res["mean_detection_latency"]),
                        "false_alarm_rate": res["false_alarm_rate"],
                        "collision_rate": res["collision_rate"],
                    }

    # Formulation 2: penalty and barrier each produced a final point.
    f2_path = results_root / "constrained_optimization" / "result.json"
    if f2_path.exists():
        with open(f2_path) as fh:
            data = json.load(fh)
        for method_key in ("penalty", "barrier"):
            history = data[method_key]["history"]
            if history:
                final = history[-1]
                overlays[f"F2:{method_key}"] = {
                    "tau": final["best_tau"],
                    "latency": final["latency"],
                    "false_alarm_rate": final["false_alarm_rate"],
                    "collision_rate": final["result"]["collision_rate"],
                }
    return overlays


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    cfg: EvalConfig | None = None,
    tau_grid: list[float] | None = None,
    results_root: str | Path = "results",
    out_dir: str | Path | None = None,
) -> dict:
    cfg = cfg or EvalConfig(n_episodes=200, base_seed=42, accel_noise_std=2.0)
    if tau_grid is None:
        # Dense sweep over the active region.
        tau_grid = [round(0.02 * i, 4) for i in range(1, 51)]  # 0.02, 0.04, ..., 1.00

    print(f"Pareto sweep over {len(tau_grid)} tau values in "
          f"[{min(tau_grid):.3f}, {max(tau_grid):.3f}]")
    print(f"Eval config: n_episodes={cfg.n_episodes}  noise_std={cfg.accel_noise_std}")
    print()

    points = sweep(cfg, tau_grid)
    points = mark_pareto_front(points)
    pareto = [p for p in points if p.is_pareto]

    print(f"  {len(points)} points evaluated")
    print(f"  {len(pareto)} on the Pareto frontier")
    print()
    print(f"{'tau':>6}  {'latency':>8}  {'FA':>6}  {'coll':>6}  {'recall':>6}  pareto")
    for p in points:
        marker = "*" if p.is_pareto else " "
        print(f"{p.tau:6.3f}  {p.latency:8.3f}  "
              f"{p.false_alarm_rate:6.2%}  {p.collision_rate:6.2%}  "
              f"{p.recall:6.2%}    {marker}")

    overlays = load_formulation_optima(Path(results_root))
    if overlays:
        print()
        print("Formulation 1 / 2 optima for overlay:")
        for label, pt in overlays.items():
            print(f"  {label:25s}  tau={pt['tau']:.4f}  "
                  f"latency={pt['latency']:.3f}s  FA={pt['false_alarm_rate']:.2%}  "
                  f"coll={pt['collision_rate']:.2%}")

    out = {
        "tau_grid": tau_grid,
        "points": [p.as_dict() for p in points],
        "pareto_points": [p.as_dict() for p in pareto],
        "overlays": overlays,
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
    run(out_dir="results/pareto_analysis")