# Overview

This repository contains code to generate synthetic longitudinal tumor-growth data, train bootstrap ensembles of TIMEVIEW models with fixed spline knots, and reproduce the figures and visualisations used in the accompanying paper.

The code is organized around four main scripts/notebooks, each corresponding to a specific step of the experimental pipeline.

---

## Repository Structure

```
.
├── LICENSE
├── requirements.txt
├── scripts/
│   ├── synthetic_datasets.py
│   ├── bootstrap_epistemic_uncertainty_fixed_knots.py
│   ├── figures_fixed_knots.ipynb
│   └── figure_composition_illustration.ipynb
└── vendor_TIMEVIEW/
```

---

## File Descriptions

### `scripts/synthetic_datasets.py`
**Purpose: Data generation**

This file defines the `SyntheticTumorDataset` class, which generates synthetic longitudinal tumor trajectories. 

---

### `scripts/bootstrap_epistemic_uncertainty_fixed_knots.py`
**Purpose: Train bootstrap ensemble**

This script trains an ensemble of TIMEVIEW models using bootstrap resampling to quantify epistemic uncertainty. 

---

### `scripts/figures_fixed_knots.ipynb`
**Purpose: Generate report figures and visualisations**

This notebook loads the saved bootstrap ensemble and reproduces the main figures used in the paper.


---

### `scripts/figure_composition_illustration.ipynb`
**Purpose: Graphical illustration of compositions**

This notebook produces a conceptual, illustrative figure explaining how trajectory compositions are represented within the model.

---

## `vendor_TIMEVIEW/`

This directory contains a lightly modified copy of the TIMEVIEW codebase used by the paper.

---

## Usage Notes

1. Install dependencies from `requirements.txt`
2. Run `bootstrap_epistemic_uncertainty_fixed_knots.py` to generate data and train models
3. Use the notebooks in `scripts/` to reproduce figures and visualisations

---
