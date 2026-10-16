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
