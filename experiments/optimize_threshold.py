"""Formulation 1: 1D optimization of the detection threshold tau.

Minimize a scalar loss combining detection latency and false alarm rate:

    L(tau) = w_latency * latency(tau) + w_false_alarm * FA(tau) + w_miss * (1 - recall(tau))

The miss term is added because at high tau, latency() goes to NaN (no
detections at all), so we'd lose all signal. Miss rate fills that gap.

We use golden-section search on a bounded 1D interval. This is the
textbook 1D optimizer for unimodal smooth objectives — and our loss is
roughly unimodal in tau. We also save the full search trace in result.json
so convergence can be inspected.

Usage:
    python -m experiments.optimize_threshold
"""

from dataclasses import dataclass
from pathlib import Path
import json
import math
import time

from src.evaluation import EvalConfig, evaluate_tau, EvalResult


# ---------------------------------------------------------------------------
# Loss definition
# ---------------------------------------------------------------------------

@dataclass
class LossWeights:
    latency: float = 1.0          # per second of latency
    false_alarm: float = 5.0      # per unit FA rate (0..1)
    miss: float = 5.0             # per unit miss rate (1 - recall)
    miss_latency_default: float = 5.0  # latency credited to fully missed
                                        # episodes (seconds), so very high
                                        # tau is penalized by latency too


def loss_from_result(result: EvalResult, weights: LossWeights) -> float:
    """Scalar loss combining latency, false alarms, and miss rate.

    When recall = 0, latency is NaN. We substitute miss_latency_default to
    keep the loss well-defined at high tau (otherwise the optimizer sees a
    discontinuous "go to NaN" cliff and its behavior is undefined).
    """
    if math.isnan(result.mean_detection_latency):
        latency = weights.miss_latency_default
    else:
        latency = result.mean_detection_latency

    miss_rate = 1.0 - result.detection_recall

    return (
        weights.latency * latency
        + weights.false_alarm * result.false_alarm_rate
        + weights.miss * miss_rate
    )


# ---------------------------------------------------------------------------
# Golden-section search
# ---------------------------------------------------------------------------

GOLDEN_RATIO = (math.sqrt(5.0) - 1.0) / 2.0   # ~0.618


@dataclass
class SearchHistory:
    iterations: list   # list of dicts: {iter, a, b, x1, x2, f1, f2}
    evaluations: list  # list of (tau, loss, result.as_dict()) — every f-eval


def golden_section_search(
    f,                 # callable: tau -> (loss, EvalResult)
    a: float,
    b: float,
    tol: float = 1e-3,
    max_iter: int = 30,
) -> tuple[float, SearchHistory]:
    """Standard golden-section search for argmin f(x) on [a, b].

    Returns (best_tau, history). History contains both the bracket at
    each iteration and every individual evaluation, so the convergence
    curve can be reconstructed.
    """
    history = SearchHistory(iterations=[], evaluations=[])

    def evaluate(tau):
        loss, result = f(tau)
        history.evaluations.append((tau, loss, result.as_dict()))
        return loss, result

    x1 = b - GOLDEN_RATIO * (b - a)
    x2 = a + GOLDEN_RATIO * (b - a)
    f1, _ = evaluate(x1)
    f2, _ = evaluate(x2)

    for k in range(max_iter):
        history.iterations.append({
            "iter": k, "a": a, "b": b, "x1": x1, "x2": x2,
            "f1": f1, "f2": f2,
        })
        if abs(b - a) < tol:
            break

        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - GOLDEN_RATIO * (b - a)
            f1, _ = evaluate(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + GOLDEN_RATIO * (b - a)
            f2, _ = evaluate(x2)

    best_tau = 0.5 * (a + b)
    return best_tau, history


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    cfg: EvalConfig | None = None,
    weights: LossWeights | None = None,
    tau_lo: float = 0.01,
    tau_hi: float = 3.0,
    tol: float = 1e-3,
    out_dir: str | Path | None = None,
) -> dict:
    cfg = cfg or EvalConfig(n_episodes=200, base_seed=42)
    weights = weights or LossWeights()

    def f(tau: float):
        result = evaluate_tau(tau, cfg)
        return loss_from_result(result, weights), result

    print(f"Optimizing tau on [{tau_lo}, {tau_hi}] with tol={tol}")
    print(f"Loss weights: latency={weights.latency}  FA={weights.false_alarm}  "
          f"miss={weights.miss}  miss_latency_default={weights.miss_latency_default}")
    print(f"Eval config: n_episodes={cfg.n_episodes}  adversarial_rate={cfg.adversarial_rate}  "
          f"noise_std={cfg.accel_noise_std}")
    print()

    t0 = time.time()
    best_tau, history = golden_section_search(f, tau_lo, tau_hi, tol=tol)
    elapsed = time.time() - t0

    # Final evaluation at the best tau (also already in history but useful to print).
    best_result = evaluate_tau(best_tau, cfg)
    best_loss = loss_from_result(best_result, weights)

    print(f"Done in {elapsed:.1f}s, {len(history.evaluations)} evaluations.")
    print()
    print(f"  Best tau: {best_tau:.4f}")
    print(f"  Loss:     {best_loss:.4f}")
    print(f"  Recall:   {best_result.detection_recall:.2%}")
    print(f"  Latency:  {best_result.mean_detection_latency:.3f}s")
    print(f"  FA:       {best_result.false_alarm_rate:.2%}")
    print(f"  Coll:     {best_result.collision_rate:.2%}")
    print()

    print("Iteration log (bracket shrinking):")
    for it in history.iterations:
        print(f"  iter {it['iter']:2d}: [a={it['a']:.4f}, b={it['b']:.4f}]  "
              f"x1={it['x1']:.4f} f1={it['f1']:.4f}  "
              f"x2={it['x2']:.4f} f2={it['f2']:.4f}")

    out = {
        "best_tau": best_tau,
        "best_loss": best_loss,
        "best_result": best_result.as_dict(),
        "iterations": history.iterations,
        "evaluations": [
            {"tau": tau, "loss": loss, **res} for tau, loss, res in history.evaluations
        ],
        "config": {
            "tau_lo": tau_lo, "tau_hi": tau_hi, "tol": tol,
            "weights": weights.__dict__,
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

def sensitivity_sweep():
    """Sweep loss weights to see how the optimum moves."""
    cfg = EvalConfig(n_episodes=200, base_seed=42, accel_noise_std=2.0)
    weight_settings = [
        ("balanced",    LossWeights(latency=1.0, false_alarm=5.0, miss=5.0)),
        ("latency-heavy", LossWeights(latency=5.0, false_alarm=1.0, miss=1.0)),
        ("FA-heavy",    LossWeights(latency=1.0, false_alarm=10.0, miss=1.0)),
        ("miss-heavy",  LossWeights(latency=1.0, false_alarm=1.0, miss=10.0)),
    ]
    print("\n" + "=" * 60)
    print("Sensitivity sweep")
    print("=" * 60)
    for name, weights in weight_settings:
        print(f"\n--- {name} ---")
        out = run(cfg=cfg, weights=weights, tau_lo=0.01, tau_hi=1.0,
                  out_dir=f"results/optimize_threshold/{name}")

if __name__ == "__main__":
    sensitivity_sweep()
