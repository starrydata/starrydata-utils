# starrydata-utils

Python utility libraries for analyzing thermoelectric and magnetic materials data from the [Starrydata](https://www.starrydata.org/) database.

## Overview

- **`starrydata_utils.py`** — Thermoelectric materials: composition parsing, data loading, spline interpolation, derived property calculation (power factor, ZT, weighted mobility), material family classification (~50 families), PCA/clustering, and plotting (matplotlib + plotly).
- **`starrydata_magnetic_utils.py`** — Magnetic materials: data loading, family classification, hysteresis plotting, clustering (PCA/t-SNE/NMF), and the Brillouin function. Imports shared functions from `starrydata_utils.py`.
- **`generate_data.py`** — Data conversion pipeline: reads raw Starrydata CSV files, performs spline interpolation and derived property calculations, and outputs processed CSV files.
- **`build_notebook.py`** — Generates a Colab notebook from `starrydata_utils.py` by splitting at section markers.

## Quick start

```bash
pip install -r requirements.txt
```

### Download and process the dataset

1. Download the latest Starrydata dataset from Google Drive and place it in `data/raw/`:

```
data/raw/starrydata_dataset_YYMMDD/
├── starrydata_curves.csv
├── starrydata_samples.csv
└── starrydata_papers.csv
```

2. Update the `RAW_DIR` path in `generate_data.py` if needed, then run:

```bash
python generate_data.py
```

This produces processed files in `data/processed/`.

### Use as a library

```python
from starrydata_utils import (
    download_dataset, load_curves, load_samples,
    spline_interpolate_curves, classify_material_families,
    calculate_derived_properties, TEplot4, single_plot,
)

# Download dataset
datapath = download_dataset('1py40fDLkTW2kcGx-ie7xHxG2Iqisfcuk')

# Load data
df_curves = load_curves(datapath)
df_samples = load_samples(datapath)

# Classify material families
df_samples = classify_material_families(df_samples)
```

### Generate Colab notebook

```bash
python build_notebook.py
```

## Data flow

```
download_dataset() → load_curves() + load_samples()
    → spline_interpolate_curves() at 100K intervals
    → classify_material_families()
    → calculate_derived_properties()
    → TEplot / TEplot4 / single_plot / plotly_2d / ...
```

Three core DataFrames: `df_sample` (metadata + composition), `df_curve` (raw curves), `df_int` (interpolated + derived properties).

## Dependencies

- pandas, numpy, scipy
- pymatgen (composition parsing)
- matplotlib, plotly (visualization)
- scikit-learn (PCA, KMeans, t-SNE, NMF)
- gdown (Google Drive download)
- tqdm (progress bars)

## Links

- [Starrydata](https://www.starrydata.org/)
