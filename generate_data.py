#!/usr/bin/env python
"""Generate processed data files for Starrydata thermoelectric and magnetic materials.

Before running, download the latest dataset from Google Drive:
    1. Download: https://drive.google.com/uc?id=1py40fDLkTW2kcGx-ie7xHxG2Iqisfcuk
    2. Extract the ZIP to data/starrydata_dataset/
Or use download_dataset() from starrydata_utils:
    from starrydata_utils import download_dataset
    download_dataset('1py40fDLkTW2kcGx-ie7xHxG2Iqisfcuk', 'data/starrydata_dataset')

Input:
    data/starrydata_dataset/starrydata_curves.csv
    data/starrydata_dataset/starrydata_samples.csv
    data/starrydata_dataset/starrydata_papers.csv

Output (thermoelectric):
    data/processed/df_curves.csv
    data/processed/df_samples.csv
    data/processed/df_int_{T}K.csv  (T = 100, 200, …, 1000)

Output (magnetic):
    data/processed/df_mag_samples.csv
    data/processed/df_mag_curves.csv

Dependencies:
    pip install pymatgen scipy tqdm
"""

import json
import math
import os

import numpy as np
import pandas as pd
import tqdm
from pymatgen.core.composition import Composition
from scipy.interpolate import interp1d

from starrydata_magnetic_utils import (
    classify_magnetic_families, MAGNETIC_PROPERTIES, MAGNETIC_SAMPLE_INFO_KEYS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_DIR = 'data/starrydata_dataset/'
OUT_DIR = 'data/processed/'

L_ELEMENT = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
]

L_PROP_Y = [
    'Electrical resistivity', 'Electrical conductivity',
    'Seebeck coefficient', 'Thermal conductivity',
    'Power factor', 'ZT', 'Carrier mobility', 'Hall coefficient',
]

TEMPS = list(range(100, 1100, 100))  # 100K to 1000K

# Paper metadata columns (used for tooltip assembly at runtime)
L_PAPER_META = ['first_author', 'year', 'journal_short', 'title', 'citation']

# Physical constants
_pi = 3.141592653
_e = 1.602176634e-19
_kB = 1.380649e-23
L_LORENZ = _pi**2 / 3 * (_kB / _e)**2


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def comp2dict(str_comp):
    """Convert a composition string to an element-fraction dictionary."""
    d = {}
    try:
        comp = Composition(str_comp)
        for el in L_ELEMENT:
            d[el] = np.round(comp.get_atomic_fraction(el), 5)
    except Exception:
        pass
    return d


def flatten_dict(str_d_si):
    """Extract non-empty category/comment pairs from sample_info JSON.

    Returns a concise string truncated to 80 characters, e.g.
    ``DataType:Experiment | MaterialFamily:PbTe``
    """
    result = ''
    try:
        d_si = eval(str_d_si)
    except Exception:
        return result
    try:
        for key, d_cc in d_si.items():
            if d_cc != '':
                sep = '' if result == '' else ' | '
                if d_cc['category'] != '':
                    if d_cc['comment'] != '':
                        result += f"{sep}{key}:{d_cc['category']} ({d_cc['comment']})"
                    else:
                        result += f"{sep}{key}:{d_cc['category']}"
    except Exception:
        pass
    return result[:80]


def r(value, precision=5):
    """Round *value* to *precision* significant figures."""
    return float(np.format_float_scientific(value, precision))


def load_papers(raw_dir):
    """Load papers CSV and extract first author surname, year, journal short."""
    df = pd.read_csv(raw_dir + 'starrydata_papers.csv')

    def extract_first_author(s):
        try:
            return json.loads(s)[0].get('family', '')
        except Exception:
            return ''

    def extract_year(s):
        try:
            return str(json.loads(s)['date_parts'][0][0])
        except Exception:
            return ''

    def extract_journal(s):
        try:
            return s.strip('"')
        except Exception:
            return ''

    def strip_quotes(s):
        try:
            return str(s).strip('"')
        except Exception:
            return ''

    df['first_author'] = df['author'].apply(extract_first_author)
    df['year'] = df['issued'].apply(extract_year)
    df['journal_short'] = df['container_title_short'].apply(extract_journal)
    df['title'] = df['title'].apply(strip_quotes)

    # Build citation: "Author et al., Journal Vol, Page (Year)"
    def build_citation(row):
        parts = [row['first_author'] + ' et al.']
        journal = row['journal_short']
        vol = strip_quotes(str(row.get('volume', '')))
        page = strip_quotes(str(row.get('page', '')))
        if journal:
            jstr = journal
            if vol and vol != 'nan':
                jstr += ' ' + vol
            if page and page != 'nan':
                jstr += ', ' + page
            parts.append(jstr)
        if row['year']:
            parts.append('(' + row['year'] + ')')
        return ' '.join(parts)

    df['citation'] = df.apply(build_citation, axis=1)
    return df[['SID', 'first_author', 'year', 'journal_short', 'title', 'citation']]


def weighted_mobility(S_SI, rho_SI, T):
    """Weighted mobility (Snyder et al., Adv. Mater. 2020)."""
    S = S_SI * 1e6        # V/K -> µV/K
    rho = rho_SI * 1e5    # Ω·m -> mΩ·cm
    A = np.abs(S) / (_kB / _e * 1e6)
    try:
        muw = (331e-4 / rho * (T / 300)**(-1.5)
               * (np.exp(A - 2) / (1 + np.exp(-5 * (A - 1)))
                  + (3 / _pi**2) * A / (1 + np.exp(5 * (A - 1)))))
    except Exception:
        muw = 0
    return r(muw)


# ===========================================================================
# Step 1 — Process curves
# ===========================================================================
print('Step 1: Processing curves …')

df_curves_raw = pd.read_csv(RAW_DIR + 'starrydata_curves.csv')
df_curves = df_curves_raw[
    (df_curves_raw['prop_x'] == 'Temperature')
    & (df_curves_raw['prop_y'].isin(L_PROP_Y))
].copy()

print('  Spline interpolation …')
a_xint = np.array(TEMPS)

for idx in tqdm.tqdm(df_curves.index):
    try:
        a_x = np.array(eval(df_curves.at[idx, 'x']))
        a_y = np.array(eval(df_curves.at[idx, 'y']))
        x_min, x_max = a_x.min(), a_x.max()

        # De-duplicate x values (average y for repeated x)
        unique_x = np.unique(a_x)
        unique_y = np.array([a_y[a_x == ux].mean() for ux in unique_x])

        spline = interp1d(unique_x, unique_y, kind='cubic')

        for T in a_xint:
            if x_min <= T <= x_max:
                df_curves.at[idx, f'y_{T}K'] = np.format_float_scientific(
                    spline(T), precision=5)
    except Exception:
        pass

cols_curves = (
    ['SID', 'DOI', 'composition', 'sample_id', 'figure_id',
     'prop_x', 'prop_y', 'unit_x', 'unit_y', 'x', 'y', 'project_names']
    + [f'y_{T}K' for T in TEMPS]
)
df_curves_out = df_curves.reindex(columns=cols_curves)

os.makedirs(OUT_DIR, exist_ok=True)
df_curves_out.to_csv(OUT_DIR + 'df_curves.csv', index=False)
print(f'  -> {OUT_DIR}df_curves.csv  ({len(df_curves_out)} rows)')


# ===========================================================================
# Step 2 — Process samples
# ===========================================================================
print('\nStep 2: Processing samples …')

df_samples = pd.read_csv(RAW_DIR + 'starrydata_samples.csv', engine='c')

print('  Parsing compositions …')
d_comp = {}
for i in tqdm.tqdm(df_samples.index):
    str_comp = df_samples.at[i, 'composition']
    try:
        if '_' in str_comp:
            str_comp = str_comp[:str_comp.index('_')]
        d_comp[i] = comp2dict(str_comp)
    except Exception:
        pass

df_comp = pd.DataFrame(d_comp).T
df_samples = pd.merge(df_samples, df_comp,
                       left_index=True, right_index=True, how='left')

idx_H = df_samples.columns.get_loc('H')
df_samples['sum_elements'] = df_samples.iloc[:, idx_H:idx_H + 100].sum(axis=1)

print('  Generating sample_information …')
df_samples['sample_information'] = df_samples['sample_info'].apply(flatten_dict)

# Merge paper metadata (first_author, year, journal_short)
print('  Merging paper metadata …')
df_papers = load_papers(RAW_DIR)
df_samples = pd.merge(df_samples, df_papers, on='SID', how='left')

# Save full samples (with compositions) before TE-specific filtering — reused in Step 4
df_samples_all = df_samples.copy()

# Keep only samples that have TE curves and valid compositions
sample_ids_with_curves = df_curves['sample_id'].drop_duplicates()
df_samples = df_samples[df_samples['sample_id'].isin(sample_ids_with_curves)]
df_samples = df_samples[df_samples['sum_elements'] > 0.99]

cols_samples = (
    ['sample_name', 'sample_id', 'composition', 'SID', 'DOI', 'sample_info']
    + L_ELEMENT
    + ['sum_elements', 'sample_information']
    + L_PAPER_META
)
df_samples_out = df_samples.reindex(columns=cols_samples)
df_samples_out.to_csv(OUT_DIR + 'df_samples.csv', index=False)
print(f'  -> {OUT_DIR}df_samples.csv  ({len(df_samples_out)} rows)')


# ===========================================================================
# Step 3 — Per-temperature property tables
# ===========================================================================
print('\nStep 3: Generating per-temperature property tables …')

l_col_info = ['SID', 'DOI', 'sample_id', 'sample_name',
              'composition', 'sample_information'] + L_PAPER_META
l_col_prop = ['Seebeck coefficient', 'Electrical resistivity',
              'Electrical conductivity', 'Thermal conductivity',
              'Power factor', 'ZT', 'Hall coefficient', 'Carrier mobility']
l_col_calc = ['log10(Electrical conductivity)',
              'Lattice thermal conductivity', 'Z', 'Weighted mobility']

for T in TEMPS:
    print(f'  {T} K …')
    df_int = df_samples[l_col_info].copy()

    # Merge each property's interpolated value at this temperature
    for prop in l_col_prop:
        df_y = df_curves.loc[
            df_curves['prop_y'] == prop, ['sample_id', f'y_{T}K']
        ].copy()
        df_y.rename(columns={f'y_{T}K': prop}, inplace=True)
        df_y[prop] = df_y[prop].astype(float)
        df_int = pd.merge(df_int, df_y, on='sample_id', how='left')

    # Initialise derived columns
    for col in l_col_calc:
        df_int[col] = np.nan

    # Compute derived quantities row by row
    for i in tqdm.tqdm(df_int.index):
        S     = df_int.at[i, 'Seebeck coefficient']
        rho   = df_int.at[i, 'Electrical resistivity']
        sigma = df_int.at[i, 'Electrical conductivity']
        kappa = df_int.at[i, 'Thermal conductivity']
        ZT    = df_int.at[i, 'ZT']

        # Reciprocal conversion between resistivity and conductivity
        if rho > 0:
            sigma = 1 / rho
            df_int.at[i, 'Electrical conductivity'] = sigma
            df_int.at[i, 'log10(Electrical conductivity)'] = np.log10(sigma)
        elif sigma > 0:
            rho = 1 / sigma
            df_int.at[i, 'Electrical resistivity'] = rho
            df_int.at[i, 'log10(Electrical conductivity)'] = np.log10(sigma)

        # Lattice thermal conductivity (Wiedemann–Franz)
        if kappa > 0 and sigma > 0:
            df_int.at[i, 'Lattice thermal conductivity'] = (
                kappa - L_LORENZ * sigma * T)

        # Figure of merit parameter Z
        if ZT > 0:
            df_int.at[i, 'Z'] = ZT / T

        # Weighted mobility
        if (not math.isnan(S)) and rho > 0:
            df_int.at[i, 'Weighted mobility'] = weighted_mobility(S, rho, T)

    # Round all numeric columns
    for col in l_col_prop + l_col_calc:
        df_int[col] = df_int[col].apply(lambda x: r(x))

    # Write output
    df_int = df_int[l_col_info + l_col_prop + l_col_calc].drop_duplicates()
    df_int.to_csv(OUT_DIR + f'df_int_{T}K.csv')

# ===========================================================================
# Step 4 — Process magnetic samples
# ===========================================================================
print('\nStep 4: Processing magnetic samples …')

# Filter curves for magnetic materials
df_mag_curves = df_curves_raw[
    df_curves_raw['project_names'].str.contains('MagneticMaterials', na=False)
]
mag_sample_ids = df_mag_curves['sample_id'].drop_duplicates()

# Filter samples to magnetic ones (reuse df_samples_all from Step 2 with compositions)
df_mag_samples = df_samples_all[df_samples_all['sample_id'].isin(mag_sample_ids)].copy()
df_mag_samples = df_mag_samples[df_mag_samples['sum_elements'] > 0.99]

# Build d_comp dicts from element columns (needed by classify_magnetic_families)
print('  Building composition dicts …')
df_mag_samples['d_comp'] = df_mag_samples[L_ELEMENT].apply(
    lambda row: row.to_dict(), axis=1)

# Classify magnetic material families
print('  Classifying magnetic families …')
df_mag_samples = classify_magnetic_families(df_mag_samples)

# Extract sample info keys
print('  Extracting sample info fields …')
for key in MAGNETIC_SAMPLE_INFO_KEYS:
    df_mag_samples[key] = ''
    df_mag_samples[key + '_details'] = ''

for i in tqdm.tqdm(df_mag_samples.index):
    try:
        d_si = eval(df_mag_samples.at[i, 'sample_info'])
    except Exception:
        continue
    for key in MAGNETIC_SAMPLE_INFO_KEYS:
        try:
            df_mag_samples.at[i, key] = d_si[key]['category']
            df_mag_samples.at[i, key + '_details'] = d_si[key]['comment']
        except Exception:
            pass

# Output columns
cols_mag_samples = (
    ['sample_name', 'sample_id', 'composition', 'SID', 'DOI', 'sample_info']
    + L_ELEMENT
    + ['sum_elements', 'sample_information']
    + L_PAPER_META
    + ['mf_if']
    + MAGNETIC_SAMPLE_INFO_KEYS
    + [k + '_details' for k in MAGNETIC_SAMPLE_INFO_KEYS]
)
df_mag_samples_out = df_mag_samples.reindex(columns=cols_mag_samples)
df_mag_samples_out.to_csv(OUT_DIR + 'df_mag_samples.csv', index=False)
print(f'  -> {OUT_DIR}df_mag_samples.csv  ({len(df_mag_samples_out)} rows)')


# ===========================================================================
# Step 5 — Process magnetic curves
# ===========================================================================
print('\nStep 5: Processing magnetic curves …')

df_mag_curves_out = df_curves_raw[
    df_curves_raw['prop_y'].isin(MAGNETIC_PROPERTIES)
].copy()

cols_mag_curves = [
    'SID', 'DOI', 'composition', 'sample_id', 'figure_id',
    'prop_x', 'prop_y', 'unit_x', 'unit_y', 'x', 'y', 'project_names',
]
df_mag_curves_out = df_mag_curves_out.reindex(columns=cols_mag_curves)
df_mag_curves_out.to_csv(OUT_DIR + 'df_mag_curves.csv', index=False)
print(f'  -> {OUT_DIR}df_mag_curves.csv  ({len(df_mag_curves_out)} rows)')


print('\nDone! All files written to', OUT_DIR)
