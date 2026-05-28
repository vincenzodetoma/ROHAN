# ROHAN

Regional and Oceanic Heatwave Analysis Network

A Python suite for subsurface marine heatwave characterization in ocean reanalyses.

## Overview

This repository contains data processing, detection, diagnostics, and plotting workflows
for marine heatwave (MHW) analyses, with a focus on subsurface behavior and regional-to-
global perspectives across different datasets and experiments.

## Current workspace structure

- `cache/`: cached NetCDF outputs and intermediate regional products.
- `data/`: source and preprocessed data files (grids, model products, detrended fields,
  and preprocessing scripts).
- `detection_3d_tropical/`: scripts and artifacts for 3D/tropical detection workflows.
- `figures/`: generated figures for diagnostics, maps, profiles, and comparisons.
- `output/`: generated analysis outputs (NetCDF and derived products).
- `results/`: consolidated results from analysis pipelines.
- `scripts/`: main codebase with analysis, detection, verification, and plotting scripts,
  including:
  - standalone analysis scripts (e.g., depth comparisons, periods, transports),
  - `EFESTOMHW/` workflow,
  - `from_hobday/` tools,
  - `mhw_class/` and `mhw_detector/` modules,
  - vendored/related packages such as `ocetrac/` and `marineHeatWaves/`.
- `tmp/`: temporary files and scratch outputs.

## Suggested next steps

1. Add environment setup instructions (Conda/venv) and dependency pinning.
2. Add usage examples for key pipelines in `scripts/`.
3. Define a reproducible run order for data preprocessing and detection.

## Repository

GitHub remote target: `git@github.com:vincenzodetoma/ROHAN.git`
