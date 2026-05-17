# Optimizing Active Inference Agents for Robustness to Adversarial Behavior in Traffic Interactions

**Lea M. Hadzic**  
Stanford University  
Department of Computer Science

---

## Overview

This project studies how active inference agents behave when explicit communication in traffic interactions becomes unreliable. Specifically, we investigate adversarial agents that signal an intent to yield while behaving inconsistently with that signal.

We formulate the selection of an inconsistency-detection threshold as an optimization problem under three formulations:

- Scalarized weighted-sum optimization
- Constrained optimization
- Multi-objective Pareto optimization

The system is evaluated in a simulated two-agent uncontrolled intersection environment with randomized adversarial behavior and observation noise.

![Detection vs naive comparison](episode_comparison.gif)

---

## Key Contributions

- Developed a simulated adversarial traffic-interaction environment for active inference agents
- Designed an inconsistency detector based on signal–kinematic disagreement
- Formulated threshold selection as:
  - weighted-sum optimization
  - constrained optimization
  - Pareto optimization
- Compared:
  - golden-section search
  - quadratic-penalty methods
  - log-barrier methods
- Performed sensitivity analyses over:
  - observation noise
  - adversarial priors
  - oracle performance bounds

---

## Paper

See the full report here:

```text
CS361_Final_Project.pdf
```

---

## Repository Structure

```text
project/
├── figures/
├── src/
├── results/
├── notebooks/
├── README.md
├── requirements.txt
└── CS361_Final_Project.pdf
```

---

## Methodology

### Environment

We simulate a two-agent uncontrolled intersection where each agent:
- approaches perpendicularly
- emits explicit yielding signals
- reasons about the intent of the other agent

An adversarial agent falsely signals yielding while maintaining velocity.

---

### Active Inference Agent

The protagonist agent maintains a belief over whether the opposing agent will yield using Bayesian inference over:
- observed acceleration
- explicit signaling behavior

Action selection minimizes a simplified expected free energy objective balancing:
- collision probability
- crossing efficiency

---

### Inconsistency Detection

We define a running inconsistency score based on disagreement between:
- communicated intent
- observed kinematics

Detection triggers when the score exceeds threshold `τ`.

---

## Optimization Formulations

### F1 — Scalarized Optimization

Minimize:

```math
L(\tau) = w_l \ell(\tau) + w_f FA(\tau) + w_m (1 - \rho(\tau))
```

using golden-section search.

---

### F2 — Constrained Optimization

Minimize detection latency subject to a false alarm bound:

```math
\min_{\tau} \ell(\tau)
\quad \text{s.t.} \quad FA(\tau) \leq FA_{max}
```

Solved using:
- quadratic penalty methods
- logarithmic barrier methods

---

### F3 — Multi-Objective Optimization

Constructed an empirical Pareto frontier over:
- detection latency
- false alarm rate
- collision rate

---

## Main Results

| Method | Latency (s) | False Alarm (%) | Recall (%) |
|---|---|---|---|
| F1: miss-heavy | 0.086 | 16.5 | 85 |
| F1: balanced | 0.216 | 8.0 | 81 |
| F2: penalty | 0.245 | 5.0 | 79 |
| F2: barrier | 0.293 | 4.5 | 78 |

Key findings:
- Optimal thresholds varied dramatically across assumptions
- Explicit constrained optimization outperformed weighted-sum proxies under hard false-alarm requirements
- Detection-enhanced agents matched oracle collision-prevention performance at the studied noise level

---

## Sensitivity Analyses

We evaluated robustness under varying:
- observation noise
- adversarial priors
- random seeds

Findings showed:
- threshold optimization only becomes meaningful beyond a certain observation-noise regime
- optimal thresholds shift significantly depending on adversarial prevalence

---

## Running the Project

### Setup

```bash
conda create -n cs361 python=3.10
conda activate cs361

pip install -r requirements.txt
```

---

### Run Simulation

```bash
python run_simulation.py
```

---

### Run Optimization

```bash
python optimize_threshold.py
```

---

### Generate Figures

```bash
python make_plots.py
```

---

## Future Work

Potential extensions include:
- continuous action spaces
- richer adversarial policies
- learned inconsistency detectors
- multi-agent signaling dynamics
- full expected free energy formulations with information-gathering behavior

---

## References

- Friston, K. *The Free-Energy Principle*
- Kochenderfer & Wheeler. *Algorithms for Optimization*
- Boyd & Vandenberghe. *Convex Optimization*
- Schumann et al. *Active Inference for Road User Interactions*
