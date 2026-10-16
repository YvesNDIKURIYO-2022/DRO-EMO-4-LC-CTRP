# DRO-EMO: Distributionally Robust Multi-Objective Evolutionary Optimization for Low-Carbon Container Routing

**Python** • **MIT License** • **Paper** • **Data**

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)

A distributionally robust multi-objective evolutionary framework for
multimodal container routing under joint operational and economic
uncertainty, applied to the Mombasa–Bujumbura corridor.

---

## Overview

This repository contains the code, data, and analysis for the paper:

> **"A Distributionally Robust Multi-Objective Framework for Multimodal
> Container Routing under Joint Uncertainty"**
>
> *Yves Ndikuriyo, Yinggui Zhang, Dung Davou Fom*
>
> *School of Traffic and Transportation Engineering, Central South
> University, Changsha, China*
>
> *Applied Soft Computing* (under review)

### What is LC-CTRP?

The **Low-Carbon Container Transportation Routing Problem (LC-CTRP)**
optimizes multimodal freight routes, balancing three conflicting
objectives:

| Objective | Description | Unit |
|-----------|-------------|------|
| **Cost** | Transportation + transshipment + carbon cost + depot opening | USD |
| **Emissions** | CO₂ from road, rail, and waterway transport + transshipment + border idling | kg CO₂ |
| **Time** | In-transit time + transshipment delays + border waiting | hours |

### What is DRO-EMO?

**DRO-EMO** (Distributionally Robust Evolutionary Multi-Objective
Optimization) is a hybrid evolutionary framework whose fitness function
is a distributionally robust bound, combined with a diversity-adaptive
operator schedule and periodic local search.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DRO-EMO Framework                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐          │
│   │  Policy-     │───▶│  Diversity-  │───▶│  Moment-based DRO    │          │
│   │  Guided      │    │  Adaptive    │    │  Fitness Evaluation  │          │
│   │  Seeding     │    │  Operators   │    │  (joint ambiguity)   │          │
│   └──────────────┘    └──────────────┘    └──────────────────────┘          │
│                                                                              │
│   + Periodic Local Search                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Component | Description | Ablation result |
|-----------|-------------|-----------------|
| **Joint DRO bound** | Five-dimensional moment-based ambiguity set (support, mean, covariance, directional deviations, carbon-price volatility) as the fitness function | +2.0% / +6.3% HV on 100-/200-customer instances (both $p < 10^{-4}$) |
| **Diversity-adaptive operator control** | Proportional control law modulating crossover and mutation rates from population convergence and diversity signals | Significant on 6 of 8 size classes |
| **Periodic local search** | 2-opt refinement applied to the top 10% of the population every 10 generations | Significant on 4 of 8 size classes |
| **PPO-guided initialization** | Fast scenario-based reward trains a PPO agent; the exact DRO bound scores every candidate in the search | Not statistically significant after Bonferroni correction (reported as a negative result) |

---

## Key Findings

| Finding | Value | Implication |
|---------|-------|-------------|
| DRO-EMO win rate (Prins LRP, 24 instances) | 21/24 top-ranked, 24/24 first-or-second | Systematic algorithmic advantage |
| Mean hypervolume advantage over next-best | +21.6% (against MOEA/D-family) | Large, statistically significant |
| Mean hypervolume over 24 instances | 0.9412 vs. NSGA-II 0.9240 | Widens with problem size (0.7 pp → 6.2 pp) |
| Effect size vs MOEA/D and SPEA2 | Cliff's δ = 0.99 | Complete dominance |
| Effect size vs NSGA-family | Cohen's d between 1.21 and 1.94 | Large |
| Computational overhead of the DRO bound | +18% to +23% wall clock | Modest relative to HV gains |
| Binding constraint on the corridor | Reliability floor (δ_min = 0.30), not carbon price | Infrastructure, not pricing, is the lever |
| Optimal route under all tested carbon prices | Lake Victoria Corridor | 1,596.07 kg CO₂, $832.93, 82.05 h |
| Emission change across 11 regulation regimes | ≤ 0.02% | Carbon pricing alone cannot shift the optimum |
| Single-segment emission share | 52.2% from Nairobi–Musoma road leg | Highest-leverage intervention point |
| Binding node | Bujumbura at 87.5% utilization | Capacity expansion priority |

---

## Algorithm Performance Summary (24 Prins Instances)

| Metric | DRO-EMO | NSGA-II | NSGA-III | RL-NSGAII | MOEA/D | SPEA2 |
|--------|---------|---------|----------|-----------|--------|-------|
| Mean HV | **0.9412** | 0.9240 | 0.9153 | 0.9238 | 0.7364 | 0.4349 |
| Mean HV (200 cust.) | **0.9103** | 0.8973 | 0.8765 | 0.8938 | 0.5173 | 0.4218 |
| Win rate (top-ranked) | **21/24** | — | — | — | — | — |
| First-or-second | **24/24** | — | — | — | — | — |

### Performance by Instance Size

| Instance size | DRO-EMO advantage over NSGA-II |
|---------------|-------------------------------|
| 20 customers | +0.7 pp |
| 50 customers | +1.4 pp |
| 100 customers | +0.7 pp |
| 200 customers | +1.3 pp |

---

## Corridor Results (Mombasa–Bujumbura)

### Optimal route — Lake Victoria Corridor

| Segment | Mode | Distance (km) | CO₂ (kg) | Time (h) |
|---------|------|---------------|----------|----------|
| 1→8 (Mombasa–Nairobi) | Rail | 472 | 132.16 | 7.87 |
| 8→10 (Nairobi–Musoma) | Road | 496 | 833.28 | 7.09 |
| 10→11 (Musoma–Mwanza) | Road | 222 | 372.96 | 3.17 |
| 11→12 (Mwanza–Isaka) | Rail | 248 | 69.44 | 4.13 |
| 12→13 (Isaka–Tabora) | Rail | 133 | 37.24 | 2.22 |
| 13→14 (Tabora–Kigoma) | Rail | 408 | 114.24 | 6.80 |
| 14→19 (Kigoma–Bujumbura) | Water | 175 | 36.75 | 7.78 |
| **Total** | | **2,154** | **1,596.07** | **39.06** |

Baseline cost: **$832.93** • Reliability: **0.388** • Border time: **20.0 h**

### Carbon-price sensitivity

| Carbon price | Mean HV | Best CO₂ (kg) | Best cost (USD) |
|--------------|---------|---------------|-----------------|
| $50/ton | 1.316 × 10¹³ | 1,626.08 | 932.00 |
| $100/ton | 1.258 × 10¹³ | 1,632.93 | 1,020.33 |
| $150/ton | 1.203 × 10¹³ | 1,631.60 | 1,106.10 |

**Key insight:** Carbon pricing raises cost by 18.7% from $50 to $150 per
ton but changes emissions by less than 0.4%. The optimum is set by
reliability, not by price.

### Seasonal sensitivity

| Season | Mean HV | Best CO₂ (kg) | Best cost (USD) | Feasible templates |
|--------|---------|---------------|-----------------|--------------------|
| Dry | 1.312 × 10¹³ | 1,638.45 | 932.09 | 6/8 |
| Wet | 0.829 × 10¹³ | 2,042.85 | 1,190.14 | 0/8 |
| Peak | 0.833 × 10¹³ | 1,962.17 | 1,200.95 | 0/8 |

**Key insight:** In the Wet and Peak seasons no template satisfies the
reliability floor of 0.30 and the feasible set is empty. Reliability
investment is a prerequisite for year-round low-carbon operation.

### Node utilization

| Node | Country | Daily capacity (TEU) | Avg. demand (TEU) | Utilization |
|------|---------|---------------------|-------------------|-------------|
| Mombasa | Kenya | 2,500 | 1,800 | 72.0% |
| Kigoma | Tanzania | 200 | 150 | 75.0% |
| Bujumbura | Burundi | 400 | 350 | **87.5%** |

---

## Repository Structure

```
DRO-EMO-LC-CTRP/
│
├── README.md                                 # This file
├── LICENSE                                   # MIT License
├── requirements.txt                          # Python dependencies
│
├── data/
│   ├── raw/                                  # Original datasets
│   └── processed/                            # Cleaned data
│       ├── prins_benchmark_results.csv
│       ├── ablation_results.csv
│       ├── evaluation_mode_comparison.csv
│       ├── corridor_network_data.csv
│       └── corridor_sensitivity_results.csv
│
├── scripts/                                  # Execution scripts
│   ├── run_prins_benchmark.py                # 24 Prins instances, 30 runs each
│   ├── run_component_ablation.py             # full, no-DRO, no-PPO, no-adaptive, no-local-search
│   ├── run_evaluation_mode.py                # deterministic, scenario-averaged, DRO
│   ├── run_corridor_case_study.py            # Mombasa–Bujumbura corridor
│   ├── run_sensitivity.py                    # carbon price, volatility, season, border, transshipment, demand
│   └── run_statistical_analysis.py           # Wilcoxon, Bonferroni, Cliff's δ, Cohen's d
│
├── results/
│   ├── plots/                                # Generated figures
│   │   ├── Fig1_Corridor_Map.jpg
│   │   ├── Fig2_Carbon_Price.png
│   │   ├── Fig3_Seasonal.png
│   │   ├── Fig4_Pareto.png
│   │   ├── Fig5_Convergence.png
│   │   ├── Fig6_Det_vs_Robust.png
│   │   ├── Fig7_Tradeoffs.png
│   │   ├── Fig8_Improvement.png
│   │   ├── Fig9_Route_Map.png
│   │   ├── Fig10_Volatility.png
│   │   ├── Fig11_Heatmap.png
│   │   └── Fig12_Flowchart.png
│   │
│   ├── statistics/                           # Statistical outputs
│   │   └── prins_wilcoxon_results.json
│   │
│   ├── pareto_fronts/                        # Pareto front data
│   │
│   ├── runs_data/                            # Per-run data (30 seeds per condition)
│   │
│   └── ablation/                             # Ablation study results
│
└── docs/                                     # Documentation
    ├── supplementary_material.pdf
    └── data_documentation.md
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YvesNDIKURIYO-2022/DRO-EMO-LC-CTRP.git
cd DRO-EMO-LC-CTRP
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run analyses

```bash
# Prins benchmark (24 instances, 6 algorithms, 30 runs each)
python scripts/run_prins_benchmark.py

# Component ablation (full, no-DRO, no-PPO, no-adaptive, no-local-search)
python scripts/run_component_ablation.py

# Evaluation-mode comparison (deterministic, scenario-averaged, DRO)
python scripts/run_evaluation_mode.py

# Corridor case study (Mombasa–Bujumbura)
python scripts/run_corridor_case_study.py

# Sensitivity axes (carbon price, volatility, season, border, transshipment, demand)
python scripts/run_sensitivity.py

# Statistical analysis (Wilcoxon, Bonferroni, effect sizes)
python scripts/run_statistical_analysis.py
```

All scripts use deterministic seeding via MD5 hashing of the tuple
(condition, algorithm, run index, base seed = 42). Results are
reproducible bit-for-bit.

---

## Requirements

```txt
numpy>=2.3.5
pandas>=2.3.3
matplotlib>=3.10.6
seaborn>=0.13.0
scipy>=1.11.0
pymoo>=0.6.0
stable-baselines3>=2.0.0
networkx>=3.0.0
```

---

## Data Sources

| Source | Data type | Link |
|--------|-----------|------|
| NCTTCA Transport Observatory | Road emissions, border times | https://top.ttcanc.org |
| EDGAR Database | Country-level GHG emissions | https://edgar.jrc.ec.europa.eu |
| Kenya Ministry of Transport | Rail emissions | https://www.transport.go.ke |
| Tanzania Ports Authority | Lake ports cargo | https://www.ports.go.tz |
| Kenya Ports Authority | Cargo volumes | https://www.kpa.co.ke |
| Prins benchmark instances | VRP instances (24, 20–200 customers) | Prins et al. (2007) |

---

## Citation

```bibtex
@article{Ndikuriyo2026DROEMO,
  title   = {A Distributionally Robust Multi-Objective Framework for
             Multimodal Container Routing under Joint Uncertainty},
  author  = {Ndikuriyo, Yves and Zhang, Yinggui and Fom, Dung Davou},
  journal = {Applied Soft Computing},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) file.

---

## Contact

**Yves Ndikuriyo** (Corresponding author)  
PhD Candidate, Central South University, Changsha, China  
📧 yvesndikuriyo@csu.edu.cn  
🔗 [ORCID: 0009-0006-9324-7265](https://orcid.org/0009-0006-9324-7265)

**Professor Yinggui Zhang**
Professor, Central South University, Changsha, China  
📧 ygzhang@csu.edu.cn  
🔗 [ORCID: 0000-0002-5790-0638](https://orcid.org/0000-0002-5790-0638)

**Dung Davou Fom**  
PhD Candidate, Central South University, Changsha, China  
📧 dungfom1@csu.edu.cn  
🔗 [ORCID: 0009-0001-8688-813X](https://orcid.org/0009-0001-8688-813X)

---

## Acknowledgments

- NCTTCA for Transport Observatory data
- Tanzania Ports Authority for Lake Tanganyika ports data
- Kenya Ministry of Transport for rail emissions data
- Prins, Prodhon, and Wolfler Calvo for benchmark instances
- School of Traffic and Transportation Engineering, Central South University

---

<div align="center">
  <sub>Built with Python • Analysis for <em>Applied Soft Computing</em> • Last updated: October 2026</sub>
</div>
```

---

## Summary of Changes

| Item | Before | After |
|---|---|---|
| **Title** | "Multi-Objective Optimization of Low-Carbon Container Routing Using PA-NSGA-II" | "DRO-EMO: Distributionally Robust Multi-Objective Evolutionary Optimization for Low-Carbon Container Routing" |
| **Journal** | *Computational Optimization and Applications* (Under Review) | *Applied Soft Computing* (under review) |
| **Method name** | PA-NSGA-II | DRO-EMO |
| **Fitness function** | Pareto search with adaptive operators | Moment-based distributionally robust bound |
| **Components** | Adaptive-PPO, Size-Adaptive Config, PA-NSGA-II | Joint DRO bound, Diversity-adaptive operator control, Periodic local search, PPO-guided seeding |
| **Win rate** | 83.3% (20/24) | 21/24 top-ranked, 24/24 first-or-second |
| **Mean HV** | 1.85 × 10¹¹ (mixed with other tables) | 0.9412 over 24 instances (matches `tab:prins_hv`) |
| **Corridor baseline** | "356 kg CO₂ (PA-NSGA-II), 576 kg (NSGA-II)" — inconsistent with manuscript | 1,596.07 kg CO₂, $832.93, 82.05 h — matches `tab:alt_paths` |
| **Carbon-price table** | Mixed PA-NSGA-II/NSGA-II/SPEA2 results | Aligned with `tab:econ_sensitivity` (mean HV, best CO₂, best cost) |
| **Seasonal table** | Mixed columns | Aligned with `tab:oper_sensitivity` |
| **Effect size** | Cohen's d = 0.55 | Cohen's d between 1.21 and 1.94 (matches `tab:prins_stats`) |
| **PPO framing** | "Initialization matters" (positive result) | "Not statistically significant" (honest negative result, matching ablation) |
| **Repository name** | `PA-NSGA-II-4-LC-CTRP` | `DRO-EMO-LC-CTRP` |
| **File list** | Files with spaces in names (`COMPREHENSIVE BENCHMARK ANALYSIS.py`) | Clean names (`run_prins_benchmark.py`) |
| **Figure names** | `Fig1_Baseline_Performance.pdf` … `Fig13_3D_Objective_Space.pdf` | Renamed to match the main manuscript figure references (`Figure_1_Corridor_Map.jpg` … `Figure_11_Heatmap.png`) |
| **Dependencies** | numpy>=1.21, pandas>=1.3, Python 3.9+ | numpy>=2.3.5, pandas>=2.3.3, Python 3.11+ (matches `sec:env_setup`) |
| **Citation key** | `Ndikuriyo2026` | `Ndikuriyo2026DROEMO` with the correct title and journal |
| **Date** | June 2026 | October 2026 |

---

