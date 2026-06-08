"""Plot the Pareto frontier.

Requires: pip install seaborn

Usage:
    python -m analysis.plot_pareto_seaborn
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import seaborn as sns


RESULTS_PATH = Path("results/pareto_analysis/result.json")
OUT_PATH = Path("results/pareto_analysis/pareto_frontier.png")


# Seaborn theme + matplotlib font settings combined.
# "paper" context: smaller fonts suitable for figures inside a paper.
# "whitegrid" style: white background with subtle grid (academic look).
def setup_style():
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette="muted",
        font="sans-serif",
        font_scale=1.0,
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.4,
            "figure.dpi": 120,
        },
    )


OVERLAY_STYLES = {
    "F1:miss-heavy":    {"color": "#1f77b4", "marker": "s", "label": "F1: miss-heavy"},
    "F1:latency-heavy": {"color": "#ff7f0e", "marker": "^", "label": "F1: latency-heavy"},
    "F1:balanced":      {"color": "#2ca02c", "marker": "D", "label": "F1: balanced"},
    "F1:FA-heavy":      {"color": "#d62728", "marker": "v", "label": "F1: FA-heavy"},
    "F2:penalty":       {"color": "#9467bd", "marker": "P", "label": "F2: penalty"},
    "F2:barrier":       {"color": "#8c564b", "marker": "X", "label": "F2: barrier"},
}


def plot_pareto(results_path: Path = RESULTS_PATH, out_path: Path = OUT_PATH):
    with open(results_path) as fh:
        data = json.load(fh)

    setup_style()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    points = data["points"]
    pareto_points = [p for p in points if p["is_pareto"]]
    dominated_points = [p for p in points if not p["is_pareto"]]

    # Dominated points
    if dominated_points:
        sns.scatterplot(
            x=[p["false_alarm_rate"] * 100 for p in dominated_points],
            y=[p["latency"] for p in dominated_points],
            s=20, color="#bbbbbb", alpha=0.7,
            edgecolor=None, ax=ax, label="dominated", zorder=2,
        )

    # Pareto frontier — line plus markers
    pareto_sorted = sorted(pareto_points, key=lambda p: p["false_alarm_rate"])
    sns.lineplot(
        x=[p["false_alarm_rate"] * 100 for p in pareto_sorted],
        y=[p["latency"] for p in pareto_sorted],
        color="black", linewidth=1.4, ax=ax, zorder=3,
        label="Pareto frontier",
    )
    sns.scatterplot(
        x=[p["false_alarm_rate"] * 100 for p in pareto_sorted],
        y=[p["latency"] for p in pareto_sorted],
        s=32, color="black", marker="o", ax=ax, zorder=4, legend=False,
    )

    # F1/F2 overlays
    overlays = data.get("overlays", {})
    for key, pt in overlays.items():
        style = OVERLAY_STYLES.get(key, {"color": "black", "marker": "*", "label": key})
        ax.scatter(
            pt["false_alarm_rate"] * 100,
            pt["latency"],
            s=120, color=style["color"], marker=style["marker"],
            edgecolors="black", linewidths=0.7, zorder=5,
            label=style["label"],
        )

    ax.set_xlabel("False alarm rate (%)")
    ax.set_ylabel("Mean detection latency (s)")
    ax.set_title(
        "Pareto frontier: detection latency vs. false alarm rate\n"
        f"(50-point τ sweep; σ={data['config']['accel_noise_std']}, "
        f"adv rate={data['config']['adversarial_rate']}, "
        f"n={data['config']['n_episodes']} episodes per τ)",
        pad=12,
    )

    # Reorder legend
    handles, labels = ax.get_legend_handles_labels()
    primary = ["Pareto frontier", "dominated"]
    overlay_order = list(OVERLAY_STYLES.keys())
    overlay_labels = [OVERLAY_STYLES[k]["label"] for k in overlay_order if k in overlays]
    desired_order = primary + overlay_labels
    label_to_handle = dict(zip(labels, handles))
    ordered = [(lbl, label_to_handle[lbl]) for lbl in desired_order if lbl in label_to_handle]
    if ordered:
        ax.legend(
            [h for _, h in ordered], [l for l, _ in ordered],
            bbox_to_anchor=(1.02, 1.0), loc="upper left",
            frameon=False, borderaxespad=0,
        )

    sns.despine(ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {out_path}")
    print(f"Saved {out_path.with_suffix('.pdf')}")
    plt.close(fig)


if __name__ == "__main__":
    plot_pareto()
