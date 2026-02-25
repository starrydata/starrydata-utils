#!/usr/bin/env python3
"""Build 260222_starrydata_utils.ipynb from starrydata_utils.py."""

import json
import uuid

def md(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.strip().split('\n'),
            "id": str(uuid.uuid4())[:8]}

def code(source):
    return {"cell_type": "code", "metadata": {}, "outputs": [],
            "execution_count": None,
            "source": source.strip().split('\n'),
            "id": str(uuid.uuid4())[:8]}

# Fix: join source lines with newlines for proper rendering
def fix_source(lines):
    """Add newlines to all but the last line."""
    if not lines:
        return lines
    return [l + '\n' for l in lines[:-1]] + [lines[-1]]

# Read the source file
with open('starrydata_utils.py', 'r') as f:
    src = f.read()
    lines = src.split('\n')

# Find section boundaries by line number
section_starts = {}
for i, line in enumerate(lines):
    if line.startswith('# ===='):
        # Next line has the section title
        if i + 1 < len(lines):
            title_line = lines[i + 1].strip('# \n')
            section_starts[title_line] = i

# Find key markers
def find_line(marker):
    for i, line in enumerate(lines):
        if marker in line:
            return i
    return None

# Section line numbers
sec_constants = find_line('# 1. Constants')
sec_composition = find_line('# 2. Composition functions')
sec_dataproc = find_line('# 3. Data processing')
sec_loading = find_line('# 4. Data loading')
sec_interp = find_line('# 5. Interpolation')
sec_derived = find_line('# 6. Derived properties')
sec_family = find_line('# 7. Material family')
sec_selection = find_line('# 8. Sample selection')
sec_pca = find_line('# 9. PCA')
sec_plotting = find_line('# 10. Plotting - matplotlib')
sec_plotly = find_line('# 11. Plotting - plotly')

# Key sub-markers within section 10
line_unified = find_line('# Unified configuration defaults')
line_resolve = find_line('def _resolve_prop(')
line_list_props = find_line('def list_properties(')
line_plot_scatter = find_line('def plot_scatter(')
line_single_plot = find_line('def single_plot(')
line_pca_scatter = find_line('def plot_pca_scatter(')
line_plot_curves = find_line('def plot_curves(')
line_single_curves = find_line('def single_curves(')
line_prepare = find_line('def _prepare_teplot_data(')
line_setup_font = find_line('def _setup_matplotlib_font(')
line_single_pca = find_line('def single_pca_scatter(')
line_panels = find_line('PANELS_TEPLOT4')
line_teplot = find_line('def TEplot(')
line_teplot4 = find_line('def TEplot4(')
line_teplot6 = find_line('def TEplot6(')
line_prepare_line = find_line('def _prepare_line_data(')
line_normalize = find_line('def _normalize_panels(')
line_render = find_line('def _render_panel(')
line_testack = find_line('def TEstack(')
line_terow = find_line('def TErow(')
line_plotly_curves = find_line('def plotly_curves(')

def extract(start, end):
    """Extract lines from start to end (exclusive), stripping section banners."""
    block = lines[start:end]
    # Remove leading/trailing blank lines
    while block and not block[0].strip():
        block = block[1:]
    while block and not block[-1].strip():
        block = block[:-1]
    return '\n'.join(block)

# Build cells
cells = []

# Title
cells.append(md("""# starrydata_utils.py - Colab Notebook

Consolidated utility functions for Starrydata thermoelectric materials analysis.

**Usage**: Run all cells to define functions, then use them in your analysis.

Generated: 2026-02-22"""))

# Install dependencies
cells.append(md("## Setup"))
cells.append(code("!pip install pymatgen plotly scikit-learn -q"))

# Imports
cells.append(md("## Imports"))
cells.append(code(extract(0, sec_constants - 2)))

# 1. Constants
cells.append(md("## 1. Constants"))
cells.append(code(extract(sec_constants + 2, sec_composition - 2)))

# 2. Composition functions
cells.append(md("## 2. Composition functions\n\n`comp2dict`, `comp2vec`, `vec2comp`, `contains` - Convert between composition representations."))
cells.append(code(extract(sec_composition + 2, sec_dataproc - 2)))

# 3. Data processing
cells.append(md("## 3. Data processing utilities\n\n`flatten_dict`, `r`, `weighted_mobility`, `parse_array_string`"))
cells.append(code(extract(sec_dataproc + 2, sec_loading - 2)))

# 4. Data loading
cells.append(md("## 4. Data loading\n\n`load_curves`, `load_samples` - Load data from Starrydata CSV/JSON files."))
cells.append(code(extract(sec_loading + 2, sec_interp - 2)))

# 5. Interpolation
cells.append(md("## 5. Interpolation\n\n`spline_interpolate_curves` - Spline interpolation of temperature-property curves."))
cells.append(code(extract(sec_interp + 2, sec_derived - 2)))

# 6. Derived properties
cells.append(md("## 6. Derived properties\n\n`calculate_derived_properties` - Compute power factor, ZT, weighted mobility, etc."))
cells.append(code(extract(sec_derived + 2, sec_family - 2)))

# 7. Material family classification
cells.append(md("## 7. Material family classification\n\n`classify_material_families` - Classify samples into thermoelectric material families."))
cells.append(code(extract(sec_family + 2, sec_selection - 2)))

# 8. Sample selection
cells.append(md("## 8. Sample selection\n\n`selectsamples` - Select and filter sample data."))
cells.append(code(extract(sec_selection + 2, sec_pca - 2)))

# 9. PCA & Clustering
cells.append(md("## 9. PCA & Clustering\n\n`pca2`, `generate_rainbow_colors` - PCA analysis and K-means clustering on elemental compositions."))
cells.append(code(extract(sec_pca + 2, sec_plotting - 2)))

# 10. Plotting - matplotlib
cells.append(md("""## 10. Plotting - matplotlib

### Configuration system

All plotting functions use a unified configuration system:

- **`DEFAULT_FIGURE_CONFIG`**: Font, size, DPI, show/hide settings
- **`DEFAULT_PLOT_CONFIG`**: Marker size, alpha, label settings
- **`_merge_config(defaults, overrides, extra_defaults)`**: Merge config dicts
- **`_auto_filename(func_name, ...)`**: Auto-generate descriptive filenames

All plots are auto-saved as PNG and `plt.close(fig)` is called to prevent double display."""))

# 10a. Property registry + unified config + _resolve_prop + list_properties
cells.append(md("### 10a. Property registry, configuration defaults, and helpers"))
line_prop_reg = find_line('PROPERTY_REGISTRY = {')
cells.append(code(extract(line_prop_reg, line_unified - 1)))
cells.append(code(extract(line_unified, line_list_props)))
cells.append(code(extract(line_list_props, line_plot_scatter)))

# 10b. Scatter and curve plot functions
cells.append(md("### 10b. Low-level plot functions\n\n`plot_scatter`, `plot_pca_scatter`, `plot_curves` - Axes-level plotting functions."))
cells.append(code(extract(line_plot_scatter, line_single_plot)))
cells.append(code(extract(line_pca_scatter, line_plot_curves)))
cells.append(code(extract(line_plot_curves, line_single_curves)))

# 10c. Single-panel plot functions
cells.append(md("### 10c. Single-panel plot functions\n\n`single_plot`, `single_curves`, `single_pca_scatter` - Standalone plot functions with auto-save."))
cells.append(code(extract(line_single_plot, line_pca_scatter)))
cells.append(code(extract(line_single_curves, line_prepare)))
cells.append(code(extract(line_prepare, line_setup_font)))
cells.append(code(extract(line_setup_font, line_single_pca)))
cells.append(code(extract(line_single_pca, line_panels)))

# 10d. Multi-panel functions
cells.append(md("### 10d. Multi-panel plot functions\n\n`TEplot`, `TEplot4`, `TEplot6`, `TEstack`, `TErow` - Multi-panel thermoelectric visualizations with unified config."))
cells.append(code(extract(line_panels, line_teplot)))
cells.append(code(extract(line_teplot, line_teplot4)))
cells.append(code(extract(line_teplot4, line_teplot6)))
cells.append(code(extract(line_teplot6, line_prepare_line)))
cells.append(code(extract(line_prepare_line, line_normalize)))
cells.append(code(extract(line_normalize, line_render)))
cells.append(code(extract(line_render, line_testack)))
cells.append(code(extract(line_testack, line_terow)))
cells.append(code(extract(line_terow, sec_plotly - 2)))

# 11. Plotly
cells.append(md("## 11. Plotting - plotly (interactive)\n\n`plotly_2d`, `plotly3`, `plotly_pca3`, `plotly_curves` - Interactive HTML plots."))
cells.append(code(extract(sec_plotly + 2, len(lines))))

# Examples — preserve user-written example cells
cells.append(md("## Example usage"))

cells.append(code("""# Download the latest Starrydata dataset from Google Drive
file_id = '1py40fDLkTW2kcGx-ie7xHxG2Iqisfcuk'
datapath = download_dataset(file_id)"""))

cells.append(code("""# Load data
df_curves = load_curves(datapath)
df_samples = load_samples(datapath)

print(f'{len(df_curves)} curves, {len(df_samples)} samples')
df_curves.head()"""))

cells.append(code("""# Filter temperature-dependent TE property curves
df_curves_t = df_curves[df_curves['prop_x'] == 'Temperature']
df_curves_t = df_curves_t[df_curves_t['prop_y'].isin(TE_PROPERTIES)]
print(f'{len(df_curves_t)} TE curves')"""))

cells.append(code("""# Spline interpolation at 100K steps
df_curves_t = spline_interpolate_curves(df_curves_t, x_btm=100, x_top=1000, dx=100)"""))

cells.append(code("""# Classify material families
df_samples = classify_material_families(df_samples)
df_samples['mf_if'].value_counts().head(15)"""))

cells.append(code("""# Prepare sample_information and filter valid samples
df_samples_t = df_samples.copy()
df_samples_t['sample_information'] = df_samples_t['sample_info'].apply(lambda s: flatten_dict(s))
l_samples_t = list(df_curves_t['sample_id'].drop_duplicates())
df_samples_t = df_samples_t[df_samples_t['sample_id'].isin(l_samples_t)]
df_samples_t = df_samples_t[df_samples_t['sum_elements'] > 0.99]
print(f'{len(df_samples_t)} valid samples with TE curves')"""))

cells.append(code("""# Build interpolated data at each temperature and calculate derived properties
l_col_info = ['SID','DOI','sample_id','sample_name','composition','sample_information','Temperature']
l_col_prop = TE_PROPERTIES
l_col_calc = DERIVED_PROPERTIES

df_int_all = None
for T in range(100, 1100, 100):
    print(f"--- {T}K ---")
    df_int_T = pd.DataFrame(df_samples_t, columns=l_col_info)
    df_int_T['Temperature'] = T
    for prop_y in l_col_prop:
        df_y = df_curves_t[df_curves_t['prop_y'] == prop_y]
        df_y = pd.DataFrame(df_y, columns=['sample_id', f'y_{T}K']).rename(columns={f'y_{T}K': prop_y})
        df_y[prop_y] = df_y[prop_y].astype('float')
        df_int_T = pd.merge(df_int_T, df_y, on='sample_id', how='left')
    df_int_T = pd.DataFrame(df_int_T, columns=l_col_info + l_col_prop + l_col_calc)
    df_int_T = calculate_derived_properties(df_int_T, T)
    for col in l_col_prop + l_col_calc:
        df_int_T[col] = df_int_T[col].apply(lambda x: r(x))
    df_int_T = df_int_T.drop_duplicates()
    df_int_all = df_int_T if df_int_all is None else pd.concat([df_int_all, df_int_T])

print(f'\\nTotal interpolated data: {len(df_int_all)} rows')
df_int_all.head()"""))

cells.append(code("""# Common parameters for examples
composition_filter = 'Pb+Sn+Ge > 0.45 and S+Se+Te > 0.45'
n_cluster = 6
l_label = ['PbTe', 'PbSe', 'PbS', 'SnTe', 'SnSe', 'GeTe']
T = 400"""))

cells.append(code("""# Standalone scatter plot: Seebeck vs ZT at 400K
single_plot(df_samples_t, df_curves_t, df_int_all,
            composition_filter=composition_filter,
            n_cluster=n_cluster,
            prop_x='S', prop_y='ZT',
            unit_x='uV/K',
            T=T, l_label=l_label,
            figure_config={'dpi': 150, 'font_size': 12,
                           'width_cm': 15, 'height_cm': 10},
            plot_config={'marker_size': 2, 'alpha': 0.5})"""))

cells.append(code("""# Standalone T-kappa curves
single_curves(df_samples_t, df_curves_t, df_int_all,
              composition_filter=composition_filter,
              n_cluster=n_cluster,
              prop_y='kappa', unit_y='W/(m K)',
              T=T, l_label=l_label,
              figure_config={'dpi': 150, 'font_size': 12,
                             'width_cm': 15, 'height_cm': 10},
              plot_config={'marker_size': 2, 'alpha': 0.5})"""))

cells.append(code("""custom_panels = [
    ('pca',),
    ('curves', 'S', 'uV/K'),
    ('curves', 'sigma', 'S/cm'),
    ('curves', 'ZT', None),
    ('scatter', 'S', 'PF', 'uV/K', 'mW/(m K^2)'),
    ('scatter', 'S', 'ZT', 'uV/K', None),
]

TEplot(df_samples_t, df_curves_t, df_int_all,
       composition_filter=composition_filter,
       n_cluster=n_cluster, T=T,
       panels=custom_panels,
       l_label=l_label,
       filename='PbTe_custom')"""))

# Fix all source lines
for cell in cells:
    cell['source'] = fix_source(cell['source'])

# Build notebook
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "colab": {
            "provenance": []
        }
    },
    "cells": cells
}

with open('260222_starrydata_utils.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f'Generated notebook with {len(cells)} cells '
      f'({sum(1 for c in cells if c["cell_type"] == "markdown")} markdown, '
      f'{sum(1 for c in cells if c["cell_type"] == "code")} code)')
