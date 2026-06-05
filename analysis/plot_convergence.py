"""Plot convergence traces for Formulations 1 and 2 — clarity rewrite.

Three panels side by side. Same data as before, but:
    - font_scale=1.35 for readability
    - subplot titles removed; caption carries the role
    - bigger markers, more whitespace
    - cleaner annotations on F2 panels

Requires: pip install seaborn

Usage:
    python -m analysis.plot_convergence
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import seaborn as sns


F1_DIR = Path("results/optimize_threshold")
F2_PATH = Path("results/constrained_optimization/result.json")
OUT_PATH = Path("results/optimize_threshold/convergence.png")


def setup_style():
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette="muted",
        font="sans-serif",
        font_scale=1.35,
        rc={
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.4,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        },
    )


F1_STYLES = {
    "miss-heavy":    {"color": "#1f77b4", "label": "miss-heavy"},
    "latency-heavy": {"color": "#ff7f0e", "label": "latency-heavy"},
    "balanced":      {"color": "#2ca02c", "label": "balanced"},
    "FA-heavy":      {"color": "#d62728", "label": "FA-heavy"},
}


def load_f1_runs(f1_dir: Path) -> dict:
    runs = {}
    for sub in sorted(f1_dir.iterdir()):
        if not sub.is_dir():
            continue
        jp = sub / "result.json"
        if jp.exists():
            with open(jp) as fh:
                runs[sub.name] = json.load(fh)
    return runs


def plot_convergence(out_path: Path = OUT_PATH):
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax_f1, ax_f2_tau, ax_f2_fa = axes

    # --- Panel 1: F1 ---
    f1_runs = load_f1_runs(F1_DIR)
    for name, data in f1_runs.items():
        evals = data["evaluations"]
        losses = [e["loss"] for e in evals]
        running_best = []
        cur = float("inf")
        for L in losses:
            cur = min(cur, L)
            running_best.append(cur)
        style = F1_STYLES.get(name, {"color": "gray", "label": name})
        sns.lineplot(
            x=list(range(1, len(running_best) + 1)),
            y=running_best,
            ax=ax_f1,
            color=style["color"], label=style["label"],
            linewidth=1.8, marker="o", markersize=5,
        )
    ax_f1.set_xlabel("Evaluation index")
    ax_f1.set_ylabel("Running-best loss")
    ax_f1.legend(frameon=False, title="Loss weights", title_fontsize=10,
                 fontsize=10, loc="upper right")

    # --- F2 data ---
    with open(F2_PATH) as fh:
        f2_data = json.load(fh)
    fa_max = f2_data["fa_max"]

    pen_history = f2_data["penalty"]["history"]
    pen_rho = [h["rho"] for h in pen_history]
    pen_tau = [h["best_tau"] for h in pen_history]
    pen_fa = [h["false_alarm_rate"] for h in pen_history]

    bar_history = f2_data["barrier"]["history"]
    bar_mu = [h["mu"] for h in bar_history]
    bar_tau = [h["best_tau"] for h in bar_history]
    bar_fa = [h["false_alarm_rate"] for h in bar_history]

    iters_pen = list(range(1, len(pen_history) + 1))
    iters_bar = list(range(1, len(bar_history) + 1))

    # --- Panel 2: tau trajectory ---
    sns.lineplot(x=iters_pen, y=pen_tau, ax=ax_f2_tau, color="#9467bd",
                 marker="P", markersize=13, linewidth=1.8, label="penalty")
    sns.lineplot(x=iters_bar, y=bar_tau, ax=ax_f2_tau, color="#8c564b",
                 marker="X", markersize=13, linewidth=1.8, label="barrier")
    ax_f2_tau.set_xlabel("Outer iteration")
    ax_f2_tau.set_ylabel(r"$\tau$")
    ax_f2_tau.legend(frameon=False, loc="upper right", fontsize=10)
    for i, rho in enumerate(pen_rho):
        ax_f2_tau.annotate(rf"$\rho{{=}}{rho:g}$",
                           (iters_pen[i], pen_tau[i]),
                           textcoords="offset points", xytext=(10, 10),
                           fontsize=9, color="#9467bd")
    for i, mu in enumerate(bar_mu):
        ax_f2_tau.annotate(rf"$\mu{{=}}{mu:g}$",
                           (iters_bar[i], bar_tau[i]),
                           textcoords="offset points", xytext=(10, -16),
                           fontsize=9, color="#8c564b")

    # --- Panel 3: FA satisfaction ---
    sns.lineplot(x=iters_pen, y=[f * 100 for f in pen_fa], ax=ax_f2_fa,
                 color="#9467bd", marker="P", markersize=13,
                 linewidth=1.8, label="penalty")
    sns.lineplot(x=iters_bar, y=[f * 100 for f in bar_fa], ax=ax_f2_fa,
                 color="#8c564b", marker="X", markersize=13,
                 linewidth=1.8, label="barrier")
    ax_f2_fa.axhline(fa_max * 100, color="black", linestyle="--",
                     linewidth=1.2, alpha=0.7,
                     label=f"FA bound ({fa_max:.0%})")
    ax_f2_fa.set_xlabel("Outer iteration")
    ax_f2_fa.set_ylabel("False alarm rate (%)")
    ax_f2_fa.legend(frameon=False, loc="upper right", fontsize=10)

    sns.despine(fig=fig)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {out_path}")
    print(f"Saved {out_path.with_suffix('.pdf')}")
    plt.close(fig)


if __name__ == "__main__":
    plot_convergence()