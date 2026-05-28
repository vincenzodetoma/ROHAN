# ROHAN

Regional and Oceanic Heatwave Analysis Network

A Python suite for subsurface marine heatwave characterization in ocean reanalyses.

## Project goals

ROHAN is designed to:

1. preprocess multi-source ocean temperature fields,
2. detect marine heatwaves (MHW) at surface and subsurface levels,
3. compare products (e.g. CGLO/GLOR/ORAS/ESACCISST),
4. compute intensity/duration/category diagnostics,
5. aggregate and visualize regional and global patterns.

## Repository layout

- `cache/`: cached NetCDF outputs and temporary intermediate products.
- `data/`: input and preprocessed data grids and model fields.
- `detection_3d_tropical/`: dedicated 3D/tropical detection experiments.
- `figures/`: generated figures.
- `output/`: generated outputs.
- `results/`: consolidated analysis products.
- `scripts/`: main codebase (core scripts + integrated third-party tools).
- `tmp/`: scratch area.
- `environments/`: environment definitions for reproducibility.

## Complete code map

This section describes all code currently tracked in `scripts/` (109 source/config files,
excluding caches, generated outputs, and nested VCS metadata).

### 1) Core project scripts (`scripts/`)

- `scripts/anom_vs_depth.py`: compares anomalies along depth.
- `scripts/cat_vs_depth.py`: compares event categories versus depth.
- `scripts/climthresh_esacci.py`: climatology and threshold workflow for ESACCISST.
- `scripts/climthresh_model.py`: climatology and threshold workflow for model products.
- `scripts/days_cats.py`: counts days by MHW categories.
- `scripts/extremes.py`: extreme-event extraction/analysis utilities.
- `scripts/percent_days_surface_sub_mix.py`: percent-days diagnostics for surface/subsurface/mixed conditions.
- `scripts/period_comparison.py`: period-to-period comparison routines.
- `scripts/plot_cats_seas.py`: seasonal/category plotting routines.
- `scripts/plot_days.py`: plots day-based MHW diagnostics.
- `scripts/plot_mld.py`: mixed-layer-depth plotting support.
- `scripts/transports.py`: transport-related diagnostics.
- `scripts/verify_events.py`: event verification pipeline.
- `scripts/verify_events_ensmean.py`: event verification for ensemble means.
- `scripts/sub_clmthresh_model.sh`: batch/submit helper for model climatology-threshold runs.
- `scripts/sub_days_count.py`: day-count workflow launcher.
- `scripts/submit_catsdepth.sh`: batch/submit helper for category-vs-depth analyses.

### 2) Clustering workflows (`scripts/clustering/`)

- `scripts/clustering/cluster_events.py`: clustering of detected MHW objects/events.
- `scripts/clustering/do_clustering.sh`: batch launcher for clustering tasks.
- `scripts/clustering/sub_clust.sh`: additional scheduler helper for cluster jobs.

### 3) MHW class workflows (`scripts/mhw_class/`)

- `scripts/mhw_class/main.py`: main entry point for class-oriented MHW analysis.
- `scripts/mhw_class/make_mhw_anomaly_video.py`: creates anomaly videos.
- `scripts/mhw_class/make_mhw_anomaly_video_ensemble.py`: creates ensemble anomaly videos.
- `scripts/mhw_class/do_videos.sh`: batch helper for video generation.
- `scripts/mhw_class/test_mhw.sh`: workflow smoke test.
- `scripts/mhw_class/test_mhw_global.sh`: global workflow smoke test.

#### `scripts/mhw_class/mhw_detector/`

- `scripts/mhw_class/mhw_detector/__init__.py`: package init.
- `scripts/mhw_class/mhw_detector/accessors.py`: xarray/data accessors and helpers.
- `scripts/mhw_class/mhw_detector/cli.py`: command-line interface.
- `scripts/mhw_class/mhw_detector/utils.py`: shared detection utilities.

### 4) From Hobday workflows (`scripts/from_hobday/`)

- `scripts/from_hobday/main.py`: main entry point for Hobday-style pipeline.
- `scripts/from_hobday/test_mhw_global.sh`: global test launcher.
- `scripts/from_hobday/test_percentage.sh`: percentage diagnostics test launcher.
- `scripts/from_hobday/test_shallow_deep.sh`: shallow/deep diagnostics test launcher.

#### `scripts/from_hobday/mhw_detector/`

- `scripts/from_hobday/mhw_detector/cli.py`: command-line interface.
- `scripts/from_hobday/mhw_detector/clustering.py`: event clustering logic.
- `scripts/from_hobday/mhw_detector/do_violins.sh`: violin-plot batch launcher.
- `scripts/from_hobday/mhw_detector/gruber_metrics.py`: Gruber-style metric computations.
- `scripts/from_hobday/mhw_detector/make_anom_video.py`: anomaly video generation.
- `scripts/from_hobday/mhw_detector/make_video_single_point_profile.py`: single-point profile animation.
- `scripts/from_hobday/mhw_detector/percentage.py`: percentage diagnostics.
- `scripts/from_hobday/mhw_detector/percentage_shallow_deep.py`: shallow/deep percentage diagnostics.
- `scripts/from_hobday/mhw_detector/plotting.py`: plotting helpers.
- `scripts/from_hobday/mhw_detector/track.py`: event tracking routines.
- `scripts/from_hobday/mhw_detector/utils.py`: shared utilities.

### 5) EFESTOMHW integrated workflow (`scripts/EFESTOMHW/`)

- `scripts/EFESTOMHW/main.py`: primary run script.
- `scripts/EFESTOMHW/preprocess.py`: data preprocessing.
- `scripts/EFESTOMHW/detection.py`: MHW detection logic.
- `scripts/EFESTOMHW/analysis.py`: post-processing/analysis stage.
- `scripts/EFESTOMHW/diagnostics.py`: diagnostic metrics.
- `scripts/EFESTOMHW/plotting.py`: plotting tools.
- `scripts/EFESTOMHW/utils.py`: helper functions.
- `scripts/EFESTOMHW/rean.py`: reanalysis workflow.
- `scripts/EFESTOMHW/rean_v1.py`: legacy reanalysis variant.
- `scripts/EFESTOMHW/onepoint.py`: one-point analysis utilities.
- `scripts/EFESTOMHW/decompose_SSA.py`: SSA decomposition support.
- `scripts/EFESTOMHW/ds_to_csv.py`: export utilities.
- `scripts/EFESTOMHW/spectra_analysis.py`: spectral diagnostics.
- `scripts/EFESTOMHW/do_rean.sh`: scheduler helper for reanalysis runs.
- `scripts/EFESTOMHW/sub_rean.sh`: scheduler helper for reanalysis submission.
- `scripts/EFESTOMHW/sub_esacci_highres.sh`: scheduler helper for ESACCI high-res runs.
- `scripts/EFESTOMHW/environment.yml`: legacy environment spec used by the EFESTOMHW workflow.
- `scripts/EFESTOMHW/notebooks/marineHeatWaves.py`: notebook helper implementation.

### 6) Integrated `marineHeatWaves` package (`scripts/from_hobday/marineHeatWaves/`)

- `scripts/from_hobday/marineHeatWaves/marineHeatWaves.py`: core Hobday MHW algorithm implementation.
- `scripts/from_hobday/marineHeatWaves/setup.py`: package setup script.
- `scripts/from_hobday/marineHeatWaves/docs/mhw_stats.py`: example/statistical helper script.
- `scripts/from_hobday/marineHeatWaves/README.md`: package documentation.
- `scripts/from_hobday/marineHeatWaves/CHANGES.txt`: change log.
- `scripts/from_hobday/marineHeatWaves/LICENSE.txt`: license.
- `scripts/from_hobday/marineHeatWaves/MANIFEST` and `MANIFEST.in`: packaging manifests.

### 7) Integrated `ocetrac` package (`scripts/from_hobday/ocetrac/`)

#### Core package

- `scripts/from_hobday/ocetrac/ocetrac/tracker.py`: core object-tracking engine.
- `scripts/from_hobday/ocetrac/ocetrac/__init__.py`: package init.
- `scripts/from_hobday/ocetrac/ocetrac/_version.py`: package versioning.

#### Measures subpackage

- `scripts/from_hobday/ocetrac/ocetrac/measures/intensity_measures.py`: intensity-based object metrics.
- `scripts/from_hobday/ocetrac/ocetrac/measures/motion_measures.py`: motion/trajectory metrics.
- `scripts/from_hobday/ocetrac/ocetrac/measures/shape_measures.py`: geometry and shape metrics.
- `scripts/from_hobday/ocetrac/ocetrac/measures/temporal_measures.py`: duration and timing metrics.
- `scripts/from_hobday/ocetrac/ocetrac/measures/plotting.py`: measure plotting helpers.
- `scripts/from_hobday/ocetrac/ocetrac/measures/utils.py`: shared measure utilities.
- `scripts/from_hobday/ocetrac/ocetrac/measures/__init__.py`: measures package init.

#### Utilities

- `scripts/from_hobday/ocetrac/ocetrac/utils/cesm2_lens_utils.py`: CESM2-LENS helpers.
- `scripts/from_hobday/ocetrac/ocetrac/utils/cesm_anomalies.py`: anomaly preprocessing helpers.
- `scripts/from_hobday/ocetrac/ocetrac/utils/__init__.py`: utilities package init.

#### Tests, packaging and CI configuration

- `scripts/from_hobday/ocetrac/tests/test_model.py`: model/tracker tests.
- `scripts/from_hobday/ocetrac/tests/test_measures.py`: measures tests.
- `scripts/from_hobday/ocetrac/setup.py`, `setup.cfg`, `pyproject.toml`: packaging/build configs.
- `scripts/from_hobday/ocetrac/environment.yml`: package environment suggestion.
- `scripts/from_hobday/ocetrac/ci/environment-py3.7.yml`, `environment-py3.8.yml`: CI environments.
- `scripts/from_hobday/ocetrac/.github/workflows/*.yaml`: CI/lint/release workflows.
- `scripts/from_hobday/ocetrac/readthedocs.yml` and `docs/*`: docs build and content.

## Environment setup guide (based on `skywalker02`)

The commands below are based on the active environment used in this workspace:

- env name: `skywalker02`
- base path here: `/usr/local/mambaforge/envs/skywalker02`

### Option A (recommended): recreate from tracked environment file

1. Create the environment:

```bash
conda env create -f environments/skywalker02.from-history.yml
```

2. Activate it:

```bash
conda activate skywalker02
```

3. Install integrated local packages in editable mode:

```bash
pip install -e scripts/from_hobday/ocetrac
pip install -e scripts/from_hobday/marineHeatWaves
```

### Option B: use existing local environment

If `skywalker02` already exists on your machine:

```bash
conda activate skywalker02
```

### Sanity checks

Run these checks after activation:

```bash
python -c "import xarray, numpy, scipy, matplotlib, cartopy, netCDF4; print('core ok')"
python -c "import hdbscan, cfgrib, eccodes; print('detection stack ok')"
python -c "import tensorflow, torch; print('ml stack ok')"
python -c "import ocetrac; print('ocetrac ok')"
python -c "import marineHeatWaves; print('marineHeatWaves ok')"
```

### Optional: export current environment exactly as used

For byte-level reproducibility on a similar platform:

```bash
conda env export -n skywalker02 > environments/skywalker02.full.yml
```

## Suggested run order

1. Preprocess datasets from `data/`.
2. Run climatology/threshold scripts.
3. Run detection scripts.
4. Run verification and diagnostics.
5. Generate figures and videos.

## Repository

GitHub remote target: `git@github.com:vincenzodetoma/ROHAN.git`
