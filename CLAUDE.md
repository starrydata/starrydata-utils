# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Consolidated Python libraries for analyzing materials data from the [Starrydata](https://www.starrydata.org/) database:
- **Thermoelectric materials**: `starrydata_utils.py` (~2270 lines)
- **Magnetic materials**: `starrydata_magnetic_utils.py` (~760 lines)

Both were extracted from ~190 Jupyter notebook-derived `.py` files spanning 2017-2026.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download raw data (or manually download and extract to data/starrydata_dataset/)
python -c "from starrydata_utils import download_dataset; download_dataset('1py40fDLkTW2kcGx-ie7xHxG2Iqisfcuk', 'data/starrydata_dataset')"

# Run the data conversion pipeline (thermoelectric + magnetic)
python generate_data.py

# Regenerate the Colab notebook to colab/
python build_notebook.py

# Verify imports work
python -c "from starrydata_utils import *"
python -c "from starrydata_magnetic_utils import *"
```

There are no tests, linting, or type checking configured in this project.

## Key Files

- **`starrydata_utils.py`**: Thermoelectric materials library (11 sections). The canonical source — `build_notebook.py` reads this file to generate the Colab notebook.
- **`starrydata_magnetic_utils.py`**: Magnetic materials library (9 sections). Imports shared functions from `starrydata_utils.py` — do not duplicate functions that already exist there.
- **`generate_data.py`**: Data conversion pipeline for both thermoelectric and magnetic materials. Reads raw CSVs from `data/starrydata_dataset/` and writes processed files to `data/processed/`. Outputs: `df_curves.csv`, `df_samples.csv`, `df_int_{T}K.csv` (thermoelectric), `df_mag_samples.csv`, `df_mag_curves.csv` (magnetic).
- **`build_notebook.py`**: Generates `colab/260222_starrydata_utils.ipynb` from `starrydata_utils.py` by splitting at section markers. Output filename is hardcoded.
- **`data/`**: Data directory (contents gitignored via `data/.gitignore`). Raw data goes in `data/starrydata_dataset/`, processed output in `data/processed/`.
- **`colab/`**: Generated Colab notebooks (`.ipynb` files gitignored via `colab/.gitignore`).
- **`historical/`**: Date-stamped analysis scripts — gitignored, exist only locally. Not actively maintained.

## Architecture

### Section banner format (critical)

Both utils files are organized with `# ====...` banners:
```python
# =============================================================================
# N. Section name
# =============================================================================
```
**These must be preserved exactly** — `build_notebook.py` parses them to split the file into notebook cells. When adding code, place it within the correct section.

### starrydata_utils.py sections

1. **Constants** — `L_ELEMENT` (100 elements H-Fm), physical constants, `TE_PROPERTIES`, `DERIVED_PROPERTIES`, `L_PARENTS`
2. **Composition functions** — `comp2dict`, `comp2vec`, `vec2comp`, `contains`
3. **Data processing** — `flatten_dict`, `r`, `weighted_mobility`, `parse_array_string`
4. **Data loading** — `download_dataset`, `load_curves`, `load_samples`
5. **Interpolation** — `spline_interpolate_curves` (cubic spline at 100K steps → `y_100K`..`y_1000K` columns)
6. **Derived properties** — `calculate_derived_properties` (power factor, ZT, lattice κ, weighted mobility)
7. **Material family classification** — `classify_material_families` (~50 TE families via composition thresholds)
8. **Sample selection** — `selectsamples`
9. **PCA & clustering** — `pca2`, `generate_rainbow_colors`
10. **Plotting (matplotlib)** — Config system (`DEFAULT_FIGURE_CONFIG`, `DEFAULT_PLOT_CONFIG`, `PROPERTY_REGISTRY`), single-panel (`single_plot`, `single_curves`, `single_pca_scatter`), multi-panel (`TEplot`, `TEplot4`, `TEplot6`, `TEstack`, `TErow`)
11. **Plotting (plotly)** — `plotly_2d`, `plotly3`, `plotly_pca3`, `plotly_curves`

### starrydata_magnetic_utils.py sections

Imports `L_ELEMENT`, `comp2dict`, `comp2vec`, `vec2comp`, `contains`, `pca2`, `generate_rainbow_colors` etc. from `starrydata_utils.py`.

1. Constants — 2. Data loading — 3. Family classification — 4. Sample selection — 5. Composition averaging — 6. Hysteresis plotting — 7. Clustering visualization — 8. Brillouin function — 9. Utility

### Data flow (thermoelectric)

```
download_dataset() → load_curves() + load_samples()
  → filter prop_x == 'Temperature'
  → spline_interpolate_curves() at 100K intervals
  → classify_material_families()
  → calculate_derived_properties() at each T
  → TEplot / single_plot / plotly_2d / ...
```

Three core DataFrames: `df_sample` (metadata + composition), `df_curve` (raw curves), `df_int` (interpolated + derived properties).

### Data flow (magnetic)

```
load_magnetic_samples() + load_magnetic_curves()
  → prepare_magnetic_samples() (adds d_comp, compvec)
  → classify_magnetic_families() (adds mf_if)
  → alldataplot_mag / sampleplot / cluster_magnetic_compositions
```

Two core DataFrames: `df_mag` (sample metadata + composition + family), `df_data` (raw H-M curve data).

### Plotting config system

Matplotlib plot functions accept `figure_config` and `plot_config` dicts that override defaults. Properties are resolved through `PROPERTY_REGISTRY` (maps short keys like `'S'`, `'sigma'`, `'ZT'` to display names and units). All plots auto-save as PNG and call `plt.close(fig)`.

## Conventions

- Composition vectors are always 100-element arrays indexed by `L_ELEMENT` (H through Fm)
- Interpolated columns use the naming pattern `y_{T}K` (e.g., `y_300K`, `y_400K`)
- `data/` contents are gitignored via `data/.gitignore` — raw data must be downloaded to `data/starrydata_dataset/`; processed data is generated by `generate_data.py` into `data/processed/`
- Generated notebooks live in `colab/` (gitignored via `colab/.gitignore`) — regenerate with `python build_notebook.py`
- `starrydata_magnetic_utils.py` imports from `starrydata_utils.py` — both must be in the same directory (or on `PYTHONPATH`)

## Dependencies

pandas, numpy, scipy, pymatgen (composition parsing), matplotlib, plotly, scikit-learn (PCA, KMeans, t-SNE, NMF), gdown (Google Drive download), tqdm
