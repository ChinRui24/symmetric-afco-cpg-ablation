# Symmetric AFCO CPG Ablation Study

This repository presents a systematic ablation study of a symmetric
Adaptive Frequency-Controlled Oscillator (AFCO) based Central Pattern
Generator (CPG) for quadruped locomotion.

The project focuses on analyzing the contribution of individual control
mechanisms—such as phase coupling, frequency adaptation, and sensory
feedback—to gait stability, robustness, and synchronization performance.

---

## 1. Project Overview

Central Pattern Generators (CPGs) are widely adopted in legged robotics
for generating rhythmic locomotion patterns due to their modularity and
biologically inspired structure. However, the relative contribution of
individual CPG components is often underexplored.

This project provides a controlled ablation framework for a symmetric
AFCO-based CPG architecture, enabling a clear evaluation of how each
mechanism affects locomotion performance under identical experimental
conditions.

---

## 2. Ablation Design

Each ablation variant removes or fixes a single functional component
while keeping the remaining control pipeline unchanged, ensuring
fair and interpretable comparisons.

The evaluated components include:

- Phase coupling topology
- Adaptive frequency modulation
- Sensory feedback integration (e.g., GRF-based feedback)
- Shock response mechanisms
- Minimal baseline oscillator model

---

## 3. Repository Structure

```plaintext
.
├── ablation_study.py        # Main ablation execution script
├── visualization.py        # Visualization utilities
├── quick_view.py           # Lightweight result inspection
├── ablation_results.csv    # Aggregated evaluation metrics
├── figures/                # Generated plots and figures
│   ├── fig1_radar_chart.png
│   ├── fig2_bar_comparison.png
│   └── ...
├── ABLATION_REPORT.md      # Detailed analysis and discussion
├── latex_tables.tex        # Tables used for paper preparation
└── README.md



**最后更新**: 2026-02-05  
**版本**: 1.0  
**状态**: ✅ 已完成，可用于SCI论文投稿
## 数据与复现说明

由于该项目涉及正在进行的研究工作及平台相关限制，
原始实验日志与硬件/仿真平台参数未公开。
本仓库提供完整的实验框架、消融设计逻辑及聚合结果，
以支持方法层面的复现与分析。

This repository is provided for academic and research purposes only.

