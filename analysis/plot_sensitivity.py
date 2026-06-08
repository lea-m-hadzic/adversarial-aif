"""Plot the sensitivity sweep results. Three panels side by side.

Requires: pip install seaborn

Usage:
    python -m analysis.plot_sensitivity
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import seaborn as sns


SENS_ROOT = Path("results/sensitivity")
OUT_PATH = Path("results/sensitivity/sensitivity.png")


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


def plot_sensitivity(out_path: Path = OUT_PATH):
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax_noise, ax_adv, ax_oracle = axes

    # --- Panel 1: noise ---
    with open(SENS_ROOT / "noise" / "result.json") as fh:
        noise_data = json.load(fh)
    noise_results = noise_data["results"]
    sigmas = [r["sigma"] for r in noise_results]
    taus = [r["f1_best_tau"] for r in noise_results]
    latencies = [r["f1_best_result"]["mean_detection_latency"] for r in noise_results]

    color_tau = "#1f77b4"
    color_lat = "#d62728"

    sns.lineplot(x=sigmas, y=taus, ax=ax_noise, color=color_tau,
                 marker="o", markersize=10, linewidth=2.2, label=r"$\tau^*$")
    ax_noise.set_xlabel(r"Observation noise std $\sigma$ (m/s$^2$)")
    ax_noise.set_ylabel(r"Optimal threshold $\tau^*$", color=color_tau)
    ax_noise.tick_params(axis="y", labelcolor=color_tau)

    ax_noise2 = ax_noise.twinx()
    sns.lineplot(x=sigmas, y=latencies, ax=ax_noise2, color=color_lat,
                 marker="s", markersize=10, linewidth=2.2, linestyle="--",
                 label="latency")
    ax_noise2.set_ylabel("Latency at optimum (s)", color=color_lat)
    ax_noise2.tick_params(axis="y", labelcolor=color_lat)
    ax_noise2.spines["top"].set_visible(False)
    ax_noise2.grid(False)

    lines1, labels1 = ax_noise.get_legend_handles_labels()
    lines2, labels2 = ax_noise2.get_legend_handles_labels()
    if ax_noise.get_legend():
        ax_noise.get_legend().remove()
    if ax_noise2.get_legend():
        ax_noise2.get_legend().remove()
    ax_noise.legend(lines1 + lines2, labels1 + labels2,
                    loc="upper left", frameon=False, fontsize=11)

    # --- Panel 2: adversarial rate ---
    with open(SENS_ROOT / "adversarial_rate" / "result.json") as fh:
        adv_data = json.load(fh)
    adv_results = adv_data["results"]
    rates = [r["adversarial_rate"] for r in adv_results]
    adv_taus = [r["tau"] for r in adv_results]

    sns.lineplot(x=rates, y=adv_taus, ax=ax_adv, color="#2ca02c",
                 marker="o", markersize=11, linewidth=2.2)
    ax_adv.set_xlabel("Adversarial rate (prior P[deception])")
    ax_adv.set_ylabel(r"Optimal threshold $\tau^*$")
    ax_adv.set_ylim(-0.08, 1.15)

    for r, t in zip(rates, adv_taus):
        ax_adv.annotate(f"{t:.2f}", (r, t),
                        textcoords="offset points", xytext=(10, 8),
                        fontsize=10, color="#2ca02c")

    # --- Panel 3: oracle ---
    with open(SENS_ROOT / "oracle" / "result.json") as fh:
        oracle_data = json.load(fh)
    labels = ["Naive\n(no detection)", "Detector\n(F2 penalty)", "Oracle\n(ground truth)"]
    colors = ["#bcbd22", "#9467bd", "#17becf"]
    values = [
        oracle_data["naive"]["collision_rate"] * 100,
        oracle_data["detector"]["collision_rate"] * 100,
        oracle_data["oracle"]["collision_rate"] * 100,
    ]

    sns.barplot(x=labels, y=values, ax=ax_oracle, palette=colors,
                edgecolor="black", linewidth=0.8)
    ax_oracle.set_ylabel("Collision rate (%)")
    ax_oracle.set_xlabel("")
    ax_oracle.set_ylim(0, max(values) * 1.55)

    for i, val in enumerate(values):
        ax_oracle.text(i, val + 0.15, f"{val:.1f}%",
                       ha="center", va="bottom", fontsize=13,
                       fontweight="bold")

    frac = oracle_data.get("fraction_of_gap_recovered")
    if frac is not None:
        ax_oracle.text(0.5, 0.93,
                       f"Detector recovers {frac:.0%} of\n"
                       f"naive$\\rightarrow$oracle gap",
                       transform=ax_oracle.transAxes,
                       ha="center", va="top", fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.5",
                                 facecolor="white", edgecolor="gray",
                                 alpha=0.9))

    sns.despine(fig=fig)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {out_path}")
    print(f"Saved {out_path.with_suffix('.pdf')}")
    plt.close(fig)


if __name__ == "__main__":
    plot_sensitivity()
