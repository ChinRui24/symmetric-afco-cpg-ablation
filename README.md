# Symmetric AFCO CPG Ablation Study
This repository accompanies an ongoing research study on CPG-based
quadruped locomotion control.

This repository provides a systematic ablation study of a symmetric
Adaptive Frequency-Controlled Oscillator (AFCO) based Central Pattern
Generator (CPG) for quadruped robot locomotion.

The objective of this project is to quantitatively evaluate the
contribution of individual control mechanisms—such as phase coupling,
frequency adaptation, and sensory feedback—under a unified experimental
pipeline.

---

## 1. Background and Motivation

Central Pattern Generators (CPGs) are widely used in legged robotics due
to their modular structure, robustness, and biological inspiration.
While numerous CPG-based controllers have been proposed, the specific
role and necessity of individual mechanisms are often insufficiently
analyzed.

In particular, adaptive frequency modulation, phase reset control, and
sensory feedback are frequently combined in practice, yet their isolated
effects on locomotion stability and robustness remain unclear.

This repository addresses this gap by providing a controlled ablation
study framework for a symmetric AFCO-based CPG architecture.

---

## 2. Contributions

The main contributions of this work are summarized as follows:

- A symmetric AFCO-based CPG control architecture for quadruped locomotion
- A systematic single-factor ablation design ensuring fair comparison
- Quantitative evaluation across synchronization, robustness, and gait
  stability metrics
- Aggregated experimental results and visual analyses for transparent
  comparison

---

## 3. Ablation Design

Each ablation variant removes or fixes exactly one functional component
while keeping the remaining control pipeline unchanged. This ensures
that observed performance differences can be directly attributed to the
removed mechanism.

The evaluated components include:

- Phase coupling topology
- Adaptive frequency modulation
- Sensory feedback integration (e.g., GRF-based feedback)
- Shock response mechanisms
- Minimal baseline oscillator configuration

A detailed explanation of the ablation logic and experimental rationale
is provided in `ABLATION_REPORT.md`.

---

## 4. Repository Structure

```plaintext
.
├── ablation_study.py        # Main ablation execution script
├── visualization.py        # Visualization utilities
├── quick_view.py           # Lightweight result inspection
├── ablation_results.csv    # Aggregated evaluation metrics
├── figures/                # Generated plots and figures
│   ├── fig1_radar_chart.png
│   ├── fig2_bar_comparison.png
│   ├── fig3_component_contribution.png
│   ├── fig4_performance_heatmap.png
│   └── fig5_robustness_analysis.png
├── ABLATION_REPORT.md      # Detailed experimental analysis
├── latex_tables.tex        # Tables prepared for manuscript use
└── README.md
````

---

## 5. Results Overview

Experimental results are summarized using aggregated metrics (mean
values across multiple trials) and visualized via radar charts, bar
plots, and heatmaps.

The evaluation focuses on:

* Phase synchronization quality
* Disturbance recovery performance
* Gait regularity and stability
* Overall robustness trends across ablation variants

Interpretation and discussion of the results are provided in
`ABLATION_REPORT.md`.

---

## 6. Usage

To clone this repository:

```bash
git clone https://github.com/ChinRui24/symmetric-afco-cpg-ablation.git
```

The provided codebase is intended for research analysis and methodological
reference. It is not designed as a plug-and-play controller for specific
robotic platforms.

---

## 7. Data Availability and Reproducibility

Due to ongoing research activities and platform-specific constraints,
raw experimental logs, simulator configurations, and hardware parameters
are not publicly released.

This repository provides the complete ablation methodology, control
architecture, evaluation metrics, and aggregated results to support
method-level reproducibility and comparative analysis.

---

## 8. License

This project is released under the MIT License.
See the `LICENSE` file for details.

---

## 9. Citation

If you find this repository useful in your research, please cite:

```bibtex
@misc{rui2026symmetricafco,
  title  = {Ablation Study on Symmetric AFCO for Quadruped Locomotion},
  author = {Rui, Chin},
  year   = {2026},
  note   = {GitHub repository}
}

