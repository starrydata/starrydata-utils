# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains consolidated Python libraries for analyzing materials data from the Starrydata database:
- **Thermoelectric materials**: `starrydata_utils.py`
- **Magnetic materials**: `starrydata_magnetic_utils.py`

Both were extracted from ~190 Jupyter notebook-derived `.py` files spanning 2017-2026, each representing a date-stamped analysis script.

## Key Files

- **`starrydata_utils.py`** (~2270 lines): Thermoelectric materials library. Organized into 11 sections covering composition parsing, data loading, interpolation, derived properties, material family classification, PCA/clustering, and plotting (matplotlib + plotly).
- **`starrydata_magnetic_utils.py`** (~760 lines): Magnetic materials library. Imports shared functions from `starrydata_utils.py` and adds magnetic-specific data loading, family classification, hysteresis plotting, clustering, and the Brillouin function.
- **`generate_data.py`**: Data conversion pipeline. Reads raw Starrydata CSV files (`data/raw/`), performs spline interpolation and derived property calculations, and outputs processed CSV files to `data/processed/`.
- **`build_notebook.py`**: Generates `260222_starrydata_utils.ipynb` from `starrydata_utils.py` by splitting it at section markers (`# ====...` banners with `# N. Section name` comments). After editing `starrydata_utils.py`, run this to regenerate the notebook.
- **`historical/`**: Date-stamped analysis scripts (`YYMMDD_*.py`) converted from `.ipynb` files. These are historical references, not actively maintained — gitignored so they exist only locally. The latest versions of shared functions live in the two utils files.

## Commands

```bash
# Regenerate the Colab notebook from starrydata_utils.py
python build_notebook.py

# Run the data conversion pipeline
python generate_data.py

# Import as a library
python -c "from starrydata_utils import *"
```

## Architecture

### starrydata_utils.py section structure

Sections are delimited by `# ====...` banners. `build_notebook.py` parses these to split the file into notebook cells.

1. **Constants** — `L_ELEMENT` (100 elements H-Fm), physical constants, `TE_PROPERTIES`, `DERIVED_PROPERTIES`, `L_PARENTS`
2. **Composition functions** — `comp2dict`, `comp2vec`, `vec2comp`, `contains` — convert between formula strings and 100-element vectors using pymatgen `Composition`
3. **Data processing** — `flatten_dict`, `r` (rounding), `weighted_mobility`, `parse_array_string`
4. **Data loading** — `download_dataset` (Google Drive), `load_curves`, `load_samples` — load Starrydata CSV/JSON into DataFrames
5. **Interpolation** — `spline_interpolate_curves` — cubic spline interpolation at 100K steps, producing `y_100K`..`y_1000K` columns
6. **Derived properties** — `calculate_derived_properties` — power factor, ZT, lattice thermal conductivity, weighted mobility from interpolated data
7. **Material family classification** — `classify_material_families` — rule-based classification into ~50 TE material families (PbTe, Bi2Te3, half-Heuslers, etc.) using composition thresholds
8. **Sample selection** — `selectsamples` — filter samples by parent compound proximity
9. **PCA & clustering** — `pca2`, `generate_rainbow_colors` — PCA on 100-element composition vectors + K-means clustering
10. **Plotting (matplotlib)** — Unified config system (`DEFAULT_FIGURE_CONFIG`, `DEFAULT_PLOT_CONFIG`, `_merge_config`), `PROPERTY_REGISTRY` for axis labels/units. Key functions: `single_plot`, `single_curves`, `single_pca_scatter`, `TEplot` (flexible N-panel), `TEplot4`, `TEplot6`, `TEstack` (vertical), `TErow` (horizontal)
11. **Plotting (plotly)** — `plotly_2d`, `plotly3`, `plotly_pca3`, `plotly_curves` — interactive HTML plots

### Data flow pattern

All analysis notebooks follow this pipeline:
1. `download_dataset()` → `load_curves()` + `load_samples()`
2. Filter to temperature-dependent TE curves (`prop_x == 'Temperature'`)
3. `spline_interpolate_curves()` at 100K intervals
4. `classify_material_families()` on samples
5. Build merged `df_int_all` DataFrame at each temperature with `calculate_derived_properties()`
6. Plotting via `TEplot`/`TEplot4`/`single_plot` etc.

Three core DataFrames are passed through most functions: `df_sample` (sample metadata + composition), `df_curve` (raw measurement curves), `df_int` (interpolated + derived properties at each temperature).

### Plotting config system (thermoelectric)

All matplotlib plot functions in `starrydata_utils.py` accept `figure_config` and `plot_config` dicts that override `DEFAULT_FIGURE_CONFIG` and `DEFAULT_PLOT_CONFIG`. Properties are resolved through `PROPERTY_REGISTRY` which maps short keys (e.g., `'S'`, `'sigma'`, `'ZT'`) to display names and default units. Plots auto-save as PNG and call `plt.close(fig)`.

### starrydata_magnetic_utils.py section structure

Imports shared functions (`L_ELEMENT`, `comp2dict`, `comp2vec`, `vec2comp`, `contains`, `pca2`, etc.) from `starrydata_utils.py` — do not duplicate these.

1. **Constants** — `MAGNETIC_PROPERTIES`, `COL_H`/`COL_M` column name shortcuts, `L_PARENTS_MAG` (Fe3O4, Nd2Fe14B, SmCo5, MnAl, etc.), `MAGNETIC_SAMPLE_INFO_KEYS`
2. **Data loading** — `load_magnetic_samples`, `load_magnetic_curves` (with outlier filtering), `extract_sample_info`
3. **Material family classification** — `classify_magnetic_families` — rule-based: Oxide, Sulfide/Selenide, Telluride, Antimonide, Silicide, Ferrite, Nd2Fe14B, SmCo10, Mn-Al
4. **Sample selection** — `selectsamples_mag` — composition vector Euclidean distance filtering
5. **Composition averaging** — `averagecomp` — average composition from formula string list
6. **Hysteresis plotting** — `alldataplot_mag` (family-colored), `dataplot_mag` (multi-sample), `sampleplot` (single sample with up/down loop separation)
7. **Clustering visualization** — `clusterplot`, `cluster_magnetic_compositions`, `reduce_dimensions` (PCA/t-SNE/NMF)
8. **Physics** — `Brillouin` function for magnetization curve fitting
9. **Utility** — `compgrid` (composition grid display), `prepare_magnetic_samples` (add d_comp + compvec columns)

### Magnetic data flow pattern

1. `load_magnetic_samples(datapath)` + `load_magnetic_curves(datapath)`
2. `prepare_magnetic_samples(df_mag)` — adds `d_comp` and `compvec` columns
3. `classify_magnetic_families(df_mag)` — adds `mf_if` column
4. Plotting via `alldataplot_mag`, `sampleplot`, or clustering via `cluster_magnetic_compositions` + `clusterplot`

Two core DataFrames: `df_mag` (sample metadata + composition + family), `df_data` (raw H-M curve data).

## Dependencies

- pandas, numpy, scipy (core data)
- pymatgen (`Composition` for formula parsing)
- matplotlib, plotly (visualization)
- scikit-learn (`PCA`, `KMeans`, `TSNE`, `NMF`)
- gdown (Google Drive dataset download)
- tqdm (progress bars in `generate_data.py`)

## Conventions

- Composition vectors are always 100-element arrays indexed by `L_ELEMENT` (H through Fm)
- Interpolated columns use the naming pattern `y_{T}K` (e.g., `y_300K`, `y_400K`)
- Section banners in `starrydata_utils.py` must be preserved exactly — `build_notebook.py` depends on them for notebook generation
- `starrydata_magnetic_utils.py` imports from `starrydata_utils.py` — both files must be in the same directory (or on `PYTHONPATH`)
