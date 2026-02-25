"""
starrydata_utils.py - Consolidated utility functions for Starrydata analysis

Extracted from notebook-derived Python files (2017-2026).
Each function is the latest working version from the most recent notebook.

Sections:
    1. Constants (element list, physical constants)
    2. Composition functions (comp2dict, comp2vec, vec2comp, contains)
    3. Data processing (flatten_dict, r, weighted_mobility, parse_array_string)
    4. Data loading (load_curves, load_samples)
    5. Interpolation (spline_interpolate_curves)
    6. Derived properties (calculate_derived_properties)
    7. Material family classification (classify_material_families)
    8. Sample selection (selectsamples)
    9. PCA & clustering (pca2, generate_rainbow_colors)
    10. Plotting - matplotlib (TEplot4, TEplot6)
    11. Plotting - plotly (plotly_2d, plotly3, plotly_pca3)
"""

import numpy as np
import pandas as pd
import math
from pymatgen.core.composition import Composition


# =============================================================================
# 1. Constants
# =============================================================================

# 100 elements from H to Fm (used as composition vector indices)
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

# Physical constants
PI = 3.141592653
E_CHARGE = 1.602176634e-19       # elementary charge [C]
M_ELECTRON = 9.1093837015e-31    # electron mass [kg]
K_BOLTZMANN = 1.380649e-23       # Boltzmann constant [J/K]
LORENZ = PI**2 / 3 * (K_BOLTZMANN / E_CHARGE)**2  # Lorenz number [W Omega / K^2]

# Standard thermoelectric property names in Starrydata
TE_PROPERTIES = [
    'Electrical resistivity', 'Electrical conductivity',
    'Seebeck coefficient', 'Thermal conductivity',
    'Power factor', 'ZT', 'Carrier mobility', 'Hall coefficient',
]

# Derived property column names
DERIVED_PROPERTIES = [
    'log10(Electrical conductivity)', 'Lattice thermal conductivity',
    'Z', 'Weighted mobility',
]

# Standard parent compound list for TE materials
L_PARENTS = [
    'PbTe', 'PbSe', 'PbS', 'SnTe', 'SnSe', 'SnS', 'GeTe', 'GeSe', 'GeS',
    'Bi2Te3', 'Sb2Te3', 'Bi2Se3', 'Sb2Se3', 'Bi2S3', 'Sb2S3',
    'CoSb3', 'CeFe4Sb12',
    'TiNiSn', 'ZrNiSn', 'HfNiSn', 'TiCoSb', 'ZrCoSb', 'HfCoSb',
    'VFeSb', 'NbFeSb', 'TaFeSb',
    'Mg2Si', 'Mg2Ge', 'Mg2Sn', 'Mg3Sb2',
    'Cu2Se', 'Cu2S', 'Cu2Te', 'Ag2Se', 'Ag2S', 'Ag2Te',
    'Ba8Ga16Ge30', 'Ba8Al16Ge30', 'Sr8Ga16Ge30',
    'SrTiO3', 'CaMnO3', 'ZnO', 'Ca3Co4O9', 'NaCoO2',
    'Zn4Sb3', 'YbZn2Sb2', 'Yb14MnSb11', 'MgAgSb',
    'Si', 'Ge', 'MnSi1.7',
    'YbAl3', 'CeCoIn5',
]


# =============================================================================
# 2. Composition functions
# =============================================================================

def comp2dict(str_comp):
    """Convert composition string to element-fraction dictionary (100 elements).

    Source: 250603_TEfamilies.py (latest)

    Parameters:
        str_comp (str): Chemical formula, e.g. 'PbTe', 'Bi0.5Sb1.5Te3'

    Returns:
        dict: {element: atomic_fraction} for all 100 elements
    """
    d_comp = {}
    try:
        comp = Composition(str_comp)
        for element in L_ELEMENT:
            d_comp[element] = np.round(comp.get_atomic_fraction(element), 5)
    except Exception:
        pass
    return d_comp


def comp2vec(str_comp):
    """Convert composition string to 100-element numpy vector.

    Source: 210712_TEcompositions.py

    Parameters:
        str_comp (str): Chemical formula

    Returns:
        np.ndarray: Length-100 vector of atomic fractions (H to Fm)
    """
    vec = np.zeros(100)
    try:
        comp = Composition(str_comp)
        for i, element in enumerate(L_ELEMENT):
            vec[i] = comp.get_atomic_fraction(element)
    except Exception:
        pass
    return vec


def vec2comp(vec, dp=4):
    """Convert composition vector back to pymatgen Composition.

    Source: 210712_TEcompositions.py

    Parameters:
        vec (np.ndarray): Length-100 atomic fraction vector
        dp (int): Decimal precision for rounding

    Returns:
        Composition or None: pymatgen Composition object
    """
    str_comp = ''
    for i in range(len(vec)):
        if vec[i] > 0:
            str_comp += L_ELEMENT[i] + str(round(vec[i], dp))
    try:
        return Composition(str_comp)
    except Exception:
        return None


def contains(d_comp, l_elements, min_amount, d=0.0001):
    """Check if composition contains specified elements above threshold.

    Source: 260128_alldataplots.py

    Parameters:
        d_comp (dict): Composition dictionary from comp2dict()
        l_elements (list): Element symbols to check, e.g. ['Pb', 'Sn']
        min_amount (float): Minimum total atomic fraction
        d (float): Small offset to avoid zero-division

    Returns:
        bool: True if total fraction of specified elements exceeds min_amount
    """
    amount = d
    for element in l_elements:
        amount += d_comp.get(element, 0)
    return amount > min_amount


# =============================================================================
# 3. Data processing utilities
# =============================================================================

def flatten_dict(str_d_si, parent_key='', sep='.', result=''):
    """Convert sample_info JSON string to readable short string.

    Source: 250603_TEfamilies.py (latest)

    Parameters:
        str_d_si (str): JSON string from sample_info column

    Returns:
        str: Formatted string like 'key1:category1 (comment1) | key2:category2'
    """
    str_si = ''
    try:
        d_si = eval(str_d_si)
    except Exception:
        return str_si
    try:
        for key, d_cc in d_si.items():
            if d_cc != '':
                delimiter = '' if str_si == '' else ' | '
                if d_cc['category'] != '':
                    if d_cc['comment'] != '':
                        str_si += f"{delimiter}{key}:{d_cc['category']} ({d_cc['comment']})"
                    else:
                        str_si += f"{delimiter}{key}:{d_cc['category']}"
    except Exception:
        pass
    return str_si


def r(value, precision=5):
    """Round float to scientific notation with given precision.

    Source: 250603_TEfamilies.py

    Parameters:
        value (float): Number to round
        precision (int): Significant digits

    Returns:
        float: Rounded value
    """
    try:
        return float(np.format_float_scientific(value, precision))
    except Exception:
        return float('nan')


def weighted_mobility(S_SI, rho_SI, T):
    """Calculate weighted mobility from Seebeck coefficient and resistivity.

    Based on Snyder's quality factor formalism.
    Source: 250603_TEfamilies.py (latest)

    Parameters:
        S_SI (float): Seebeck coefficient [V/K]
        rho_SI (float): Electrical resistivity [Ohm*m]
        T (float): Temperature [K]

    Returns:
        float: Weighted mobility [cm^2/Vs]
    """
    S = S_SI * 1e6          # V/K -> uV/K
    rho = rho_SI * 1e5      # Ohm*m -> mOhm*cm
    A = np.abs(S) / (K_BOLTZMANN / E_CHARGE * 1e6)
    # Clip exponent arguments to prevent overflow (exp(709) ~ 1e308 = float64 max)
    A_clip = np.clip(A, -500, 500)
    exp_A2 = np.exp(np.clip(A_clip - 2, -500, 500))
    exp_neg5 = np.exp(np.clip(-5 * (A_clip - 1), -500, 500))
    exp_pos5 = np.exp(np.clip(5 * (A_clip - 1), -500, 500))
    try:
        muw = (331e-4 / rho * (T / 300)**(-1.5)
               * (exp_A2 / (1 + exp_neg5)
                  + (3 / PI**2) * A_clip / (1 + exp_pos5)))
    except Exception:
        muw = 0
    return r(muw)


def parse_array_string(array_string):
    """Convert string representation of array to numpy array.

    Source: 260128_alldataplots.py

    Parameters:
        array_string (str): String like '[1.0, 2.0, 3.0]'

    Returns:
        np.ndarray: Numeric array
    """
    import ast
    array_list = ast.literal_eval(array_string)
    return np.array(array_list, dtype=float)


# =============================================================================
# 4. Data loading
# =============================================================================

def download_dataset(file_id, output_dir='./starrydata_dataset', quiet=False):
    """Download and extract Starrydata dataset from Google Drive.

    Source: 241216_starrydata_analysis_basic.py

    Requires: pip install gdown

    Parameters:
        file_id (str): Google Drive file ID for the ZIP archive.
                       Find the latest ID from the Starrydata website or
                       use the known ID: '1py40fDLkTW2kcGx-ie7xHxG2Iqisfcuk'
        output_dir (str): Directory to extract dataset into
        quiet (bool): Suppress download progress output

    Returns:
        str: Path to the extracted dataset directory

    Example:
        >>> datapath = download_dataset('1py40fDLkTW2kcGx-ie7xHxG2Iqisfcuk')
        >>> df_curves = load_curves(datapath)
        >>> df_samples = load_samples(datapath)
    """
    import gdown
    import zipfile
    import os
    import tempfile

    # Download ZIP from Google Drive
    zip_path = os.path.join(tempfile.gettempdir(), 'starrydata_dataset.zip')
    url = f'https://drive.google.com/uc?id={file_id}'
    print(f'Downloading from Google Drive (file_id={file_id})...')
    gdown.download(url, zip_path, quiet=quiet)

    # Extract
    os.makedirs(output_dir, exist_ok=True)
    print(f'Extracting to {output_dir}...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Show contents and timestamp
    files = os.listdir(output_dir)
    print(f'Extracted files: {files}')

    snapshot_path = os.path.join(output_dir, 'db_snapshot.txt')
    if os.path.exists(snapshot_path):
        with open(snapshot_path, 'r') as f:
            print(f'Dataset timestamp: {f.read().strip()}')

    # Clean up ZIP
    os.remove(zip_path)

    return output_dir


def load_curves(datapath):
    """Load Starrydata curves CSV file.

    Supports both naming conventions:
        - starrydata_curves.csv (2025+ format)
        - all_curves.csv (2024 format)

    Parameters:
        datapath (str): Path to dataset directory (with trailing slash)

    Returns:
        pd.DataFrame: Curves dataframe
    """
    import os
    for filename in ['starrydata_curves.csv', 'all_curves.csv']:
        filepath = os.path.join(datapath, filename)
        if os.path.exists(filepath):
            print(f'Loading curves from {filepath}')
            return pd.read_csv(filepath)
    raise FileNotFoundError(f'No curves CSV found in {datapath}')


def load_samples(datapath):
    """Load Starrydata samples CSV file and convert compositions.

    Supports both naming conventions:
        - starrydata_samples.csv (2025+ format)
        - all_samples.csv (2024 format)

    Parameters:
        datapath (str): Path to dataset directory (with trailing slash)

    Returns:
        pd.DataFrame: Samples dataframe with element columns (H to Fm)
    """
    import os
    import tqdm

    for filename in ['starrydata_samples.csv', 'all_samples.csv']:
        filepath = os.path.join(datapath, filename)
        if os.path.exists(filepath):
            print(f'Loading samples from {filepath}')
            df_samples = pd.read_csv(filepath, engine='c')
            break
    else:
        raise FileNotFoundError(f'No samples CSV found in {datapath}')

    # Convert compositions to element dictionaries
    d_comp = {}
    for i in tqdm.tqdm(df_samples.index, desc='Parsing compositions'):
        str_comp = df_samples.at[i, 'composition']
        try:
            if '_' in str(str_comp):
                str_comp = str_comp[:str_comp.index('_')]
            d_comp_i = comp2dict(str_comp)
            d_comp[i] = d_comp_i
            df_samples.at[i, 'd_comp'] = d_comp_i
        except Exception:
            pass

    # Merge element columns
    df_comp = pd.DataFrame(d_comp).T
    df_samples = pd.merge(df_samples, df_comp, left_index=True, right_index=True, how='left')

    # Calculate sum_elements for validation
    index_H = df_samples.columns.get_loc('H')
    df_samples['sum_elements'] = df_samples.iloc[:, index_H:index_H + 100].sum(axis=1)

    return df_samples


# =============================================================================
# 5. Interpolation
# =============================================================================

def spline_interpolate_curves(df_curves_t, x_btm=100, x_top=1000, dx=100):
    """Perform cubic spline interpolation on temperature-dependent curves.

    Source: 251112_starrydata_exotic.py / 250603_TEfamilies.py (latest)

    Parameters:
        df_curves_t (pd.DataFrame): Curves filtered for prop_x='Temperature'
        x_btm (int): Minimum temperature for interpolation [K]
        x_top (int): Maximum temperature for interpolation [K]
        dx (int): Temperature step [K]

    Returns:
        pd.DataFrame: df_curves_t with added columns y_{T}K for each temperature
    """
    from scipy.interpolate import interp1d
    import tqdm

    a_xint = np.arange(x_btm, x_top + dx, dx)

    for i in tqdm.tqdm(df_curves_t.index, desc='Spline interpolation'):
        try:
            a_x = np.array(eval(df_curves_t.at[i, 'x']))
            a_y = np.array(eval(df_curves_t.at[i, 'y']))

            x_min, x_max = min(a_x), max(a_x)

            # Remove duplicate x values (average y for duplicates)
            unique_x = np.unique(a_x)
            unique_y = []
            for uv in unique_x:
                mean_y = np.mean(a_y[a_x == uv])
                unique_y.append(mean_y)

            if len(unique_x) < 4:
                continue

            spline = interp1d(unique_x, unique_y, kind='cubic')

            for x in a_xint:
                if x_min <= x <= x_max:
                    df_curves_t.at[i, f'y_{x}K'] = np.format_float_scientific(
                        spline(x), precision=5)
        except Exception:
            pass

    return df_curves_t


# =============================================================================
# 6. Derived properties calculation
# =============================================================================

def calculate_derived_properties(df_int_T, T):
    """Calculate derived thermoelectric properties at a given temperature.

    Computes: log10(sigma), lattice thermal conductivity, Z, weighted mobility,
    and fills in missing sigma/rho from each other.

    Source: 251112_starrydata_exotic.py (latest)

    Parameters:
        df_int_T (pd.DataFrame): Interpolated data at temperature T
        T (float): Temperature [K]

    Returns:
        pd.DataFrame: Updated dataframe with derived property columns
    """
    import tqdm

    for i in tqdm.tqdm(df_int_T.index, desc=f'Calculating properties at {T}K'):
        S = df_int_T.at[i, 'Seebeck coefficient']
        rho = df_int_T.at[i, 'Electrical resistivity']
        sigma = df_int_T.at[i, 'Electrical conductivity']
        kappa = df_int_T.at[i, 'Thermal conductivity']
        ZT = df_int_T.at[i, 'ZT']

        # sigma <-> rho conversion
        if rho > 0:
            sigma = 1 / rho
            df_int_T.at[i, 'Electrical conductivity'] = sigma
            df_int_T.at[i, 'log10(Electrical conductivity)'] = np.log10(sigma)
        elif sigma > 0:
            rho = 1 / sigma
            df_int_T.at[i, 'Electrical resistivity'] = rho
            df_int_T.at[i, 'log10(Electrical conductivity)'] = np.log10(sigma)

        # Lattice thermal conductivity
        if kappa > 0 and sigma > 0:
            kappaL = kappa - LORENZ * sigma * T
            df_int_T.at[i, 'Lattice thermal conductivity'] = kappaL

        # Z from ZT
        if ZT > 0:
            df_int_T.at[i, 'Z'] = ZT / T

        # Weighted mobility
        if (not math.isnan(S)) and rho > 0:
            muw = weighted_mobility(S, rho, T)
            df_int_T.at[i, 'Weighted mobility'] = muw

    return df_int_T


# =============================================================================
# 7. Material family classification
# =============================================================================

# Default classification rules.  Copy and modify before calling
# classify_material_families() to customise the classification.
#
# Rule types:
#   'general'  – element-sum threshold (applied first, can be overwritten)
#       elements:  list of element symbols
#       threshold: minimum sum to trigger classification
#
#   'specific' – stoichiometric ratio check (overwrites general)
#       groups:        list of element-group lists (ordered left→right in periodic table)
#       stoichiometry: tuple of expected atom counts matching each group
#       label:         (optional) mf_if label if different from dict key
#       extra_total:   (optional) extra elements included in the total-sum check
#       sum_base:      (optional) total threshold = sum_base - cr  (default 1.0)
#
# For 'specific' rules the function checks, for each group i > 0:
#   |g[0]/g[i] - s[0]/s[i]| < cr * s[0]/s[i]
# and that the total sum of all groups (+ extra_total) > sum_base - cr.

DEFAULT_FAMILY_RULES = {
    # --- General classifications (applied first, overwritten by specific) ---
    'Oxide': {
        'type': 'general',
        'elements': ['O'],
        'threshold': 0.2,
    },
    'Silicide': {
        'type': 'general',
        'elements': ['Si', 'Ge'],
        'threshold': 0.2,
    },
    'Sulfide-Selenide': {
        'type': 'general',
        'elements': ['S', 'Se'],
        'threshold': 0.2,
    },
    'Telluride': {
        'type': 'general',
        'elements': ['Te'],
        'threshold': 0.2,
    },
    'Antimonide': {
        'type': 'general',
        'elements': ['Sb'],
        'threshold': 0.2,
    },
    # --- Specific classifications (overwrite general) ---
    'PbTe': {
        'type': 'specific',
        'groups': [
            ['Ca', 'Sr', 'Ba', 'Ge', 'Sn', 'Pb'],  # IV-B / II-A site
            ['S', 'Se', 'Te'],                        # VI-A site
        ],
        'stoichiometry': (1, 1),
    },
    'Bi2Te3': {
        'type': 'specific',
        'groups': [
            ['As', 'Sb', 'Bi'],   # V-A site
            ['S', 'Se', 'Te'],    # VI-A site
        ],
        'stoichiometry': (2, 3),
    },
    'Mg2Si': {
        'type': 'specific',
        'groups': [
            ['Mg', 'Al'],          # II-A / III-A site
            ['Si', 'Ge', 'Sn'],   # IV-A site
        ],
        'stoichiometry': (2, 1),
    },
    'Cu2Se': {
        'type': 'specific',
        'groups': [
            ['Cu', 'Ag'],         # I-B site
            ['S', 'Se', 'Te'],    # VI-A site
        ],
        'stoichiometry': (2, 1),
    },
    'Skutterudite': {
        'type': 'specific',
        'groups': [
            ['Mn', 'Re', 'Fe', 'Co', 'Ni',
             'Ru', 'Rh', 'Pd', 'Os', 'Ir', 'Pt'],  # TM site
            ['P', 'As', 'Sb', 'Bi'],                 # Pnictogen site
        ],
        'stoichiometry': (1, 3),
        'extra_total': [
            'Y', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd',
            'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
        ],
    },
    'Half-Heusler': {
        'type': 'specific',
        'label': 'Heusler',
        'groups': [
            ['Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta'],                  # IV-B / V-B site
            ['Fe', 'Co', 'Ni', 'Ru', 'Rh', 'Pd', 'Os', 'Ir', 'Pt'],  # VIII-B site
            ['Sn', 'P', 'As', 'Sb', 'Bi', 'Te'],                  # main-group site
        ],
        'stoichiometry': (1, 1, 1),
    },
    'Full-Heusler': {
        'type': 'specific',
        'label': 'Heusler',
        'groups': [
            ['Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta'],                        # IV-B / V-B site
            ['Mn', 'Fe', 'Co', 'Ni', 'Ru', 'Rh', 'Pd', 'Os', 'Ir', 'Pt'],  # VIII-B site
            ['Al', 'Ga', 'In', 'Ge', 'Sn', 'Sb'],                       # III-A / IV-A site
        ],
        'stoichiometry': (1, 2, 1),
    },
    'Clathrate': {
        'type': 'specific',
        'groups': [
            ['Na', 'K', 'Rb', 'Cs', 'Ca', 'Sr', 'Ba'],   # guest site
            ['Al', 'Ga', 'In', 'Si', 'Ge', 'Sn'],        # framework site
        ],
        'stoichiometry': (8, 46),
    },
    'Perovskite': {
        'type': 'specific',
        'groups': [
            ['Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Cr', 'Mo', 'W',
             'Mn', 'Re', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir',
             'Ni', 'Pd', 'Pt'],   # B site
            ['O'],                  # anion site
        ],
        'stoichiometry': (1, 3),
        'sum_base': 0.8,
    },
}


def classify_material_families(df_samples, cr=0.2, d=0.0001,
                               family_rules=None):
    """Heuristic material family classification based on composition.

    Assigns 'mf_if' column using rules from ``family_rules`` dictionary.
    General rules are applied first; specific rules overwrite them.

    Source: 250603_TEfamilies.py (latest), refactored to dictionary-driven.

    Parameters:
        df_samples (pd.DataFrame): Samples with 'd_comp' column
        cr (float): Composition tolerance range
        d (float): Small offset to avoid zero-division
        family_rules (dict or None): Classification rules. If None, uses
            DEFAULT_FAMILY_RULES. See the module-level dictionary for
            the expected format.

    Returns:
        pd.DataFrame: Updated with 'mf_if' column
    """
    import tqdm

    if family_rules is None:
        family_rules = DEFAULT_FAMILY_RULES

    # Separate general and specific rules (preserve dict order)
    general_rules = {k: v for k, v in family_rules.items()
                     if v.get('type') == 'general'}
    specific_rules = {k: v for k, v in family_rules.items()
                      if v.get('type') != 'general'}

    df_samples['mf_if'] = ''

    for i in tqdm.tqdm(df_samples.index,
                       desc='Classifying material families'):
        dc = df_samples.at[i, 'd_comp']
        if not isinstance(dc, dict):
            continue

        # --- General classifications ---
        for name, rule in general_rules.items():
            try:
                el_sum = sum(dc.get(el, 0) for el in rule['elements']) + d
                if el_sum > rule.get('threshold', 0.2):
                    df_samples.at[i, 'mf_if'] = name
            except Exception:
                pass

        # --- Specific classifications (overwrite general) ---
        for name, rule in specific_rules.items():
            label = rule.get('label', name)
            groups = rule['groups']
            stoich = rule['stoichiometry']
            sum_base = rule.get('sum_base', 1.0)

            try:
                # Compute group sums
                g = [sum(dc.get(el, 0) for el in grp) + d
                     for grp in groups]

                # Check stoichiometric ratios: g[0]/g[i] ≈ s[0]/s[i]
                all_match = True
                for j in range(1, len(groups)):
                    target = stoich[0] / stoich[j]
                    tol = cr * target
                    if np.abs(g[0] / g[j] - target) >= tol:
                        all_match = False
                        break

                if not all_match:
                    continue

                # Total-sum check
                total = sum(g)
                if 'extra_total' in rule:
                    total += (sum(dc.get(el, 0)
                                  for el in rule['extra_total']) + d)
                if total > sum_base - cr:
                    df_samples.at[i, 'mf_if'] = label

            except Exception:
                pass

    return df_samples


# =============================================================================
# 8. Sample selection
# =============================================================================

def selectsamples(df_s, parent, threshold=0.1):
    """Filter samples by composition distance to a parent compound.

    Source: 200818_Jonkerplot.py / 210212_ijp_ztcalc.py

    Parameters:
        df_s (pd.DataFrame): Sample dataframe (must have composition column
                             or 'parent' column)
        parent (str): Parent compound formula (e.g. 'PbTe')
        threshold (float): Maximum L2 distance in composition space

    Returns:
        pd.DataFrame: Filtered samples
    """
    if 'parent' in df_s.columns:
        # Simple parent-label based selection
        df_ss = df_s[df_s['parent'] == parent]
    else:
        # Distance-based selection using composition vectors
        vec_parent = comp2vec(parent)
        nsamples = len(df_s)
        a_dist = np.empty(nsamples)
        for idx, (i, row) in enumerate(df_s.iterrows()):
            try:
                vec_sample = comp2vec(row['composition'])
                a_dist[idx] = np.linalg.norm(vec_sample - vec_parent)
            except Exception:
                a_dist[idx] = 999
        df_ss = df_s[a_dist < threshold]

    print(f'{len(df_ss)} samples found for {parent}.')
    return df_ss


# =============================================================================
# 9. PCA & Clustering
# =============================================================================

def generate_rainbow_colors(num_colors):
    """Generate a list of rainbow colors in HTML hex format.

    Source: 251114_TEfamilyplot4.py

    Parameters:
        num_colors (int): Number of colors to generate

    Returns:
        list[str]: Color codes like '#ff0000'
    """
    colors = []
    increment = 360 / num_colors
    for i in range(num_colors):
        hue = int(i * increment)
        rgb = _hsl_to_rgb(hue, 100, 50)
        colors.append("#{:02x}{:02x}{:02x}".format(*rgb))
    return colors


def _hsl_to_rgb(h, s, l):
    """Convert HSL to RGB tuple."""
    h /= 360
    s /= 100
    l /= 100
    if s == 0:
        r = g = b = l
    else:
        def hue_to_rgb(p, q, t):
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)
    return int(r * 255), int(g * 255), int(b * 255)


def pca2(df_sample, n_clusters):
    """PCA analysis + K-means clustering on elemental composition data.

    Source: 251114_TEfamilyplot4.py (latest, documented in CLAUDE.md)

    Parameters:
        df_sample (pd.DataFrame): Samples with element columns ('H' to 'Fm')
        n_clusters (int): Number of K-means clusters

    Returns:
        pd.DataFrame: With PCA1, PCA2, cluster, color, n_elements, len_composition
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    pca_features = df_sample.loc[:, 'H':'Fm']
    n_elements = (pca_features != 0).sum(axis=1)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(pca_features)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(pca_result)

    l_color = generate_rainbow_colors(n_clusters)
    d_color = {i: l_color[i] for i in range(n_clusters)}

    df_sample = df_sample.copy()
    df_sample['PCA1'] = pca_result[:, 0]
    df_sample['PCA2'] = pca_result[:, 1]
    df_sample['cluster'] = clusters
    df_sample['color'] = df_sample['cluster'].map(d_color)
    df_sample['n_elements'] = n_elements
    df_sample['len_composition'] = df_sample['composition'].apply(lambda x: len(str(x)))

    # Print candidate parent compounds (no decimal points in formula)
    df_parent = (
        df_sample[~df_sample['composition'].str.contains(r'\.', regex=True, na=True)]
        .sort_values('len_composition')
        .drop_duplicates(subset=['composition'])
    )
    print("Candidate parent compounds:")
    print(list(df_parent['composition']))

    return df_sample[['sample_id', 'composition', 'PCA1', 'PCA2',
                       'cluster', 'color', 'n_elements', 'len_composition']]


# =============================================================================
# 10. Plotting - matplotlib (multi-panel TE plots)
# =============================================================================

# ---------------------------------------------------------------------------
# Property registry
# ---------------------------------------------------------------------------
# Each entry maps a short key to:
#   column   : column name in df_int
#   si_range : default plot range in SI units [min, max]
#   units    : dict of {unit_label: (factor, offset)}
#              converted_value = original_SI_value * factor + offset
#   default_unit : which unit label to use when none is specified
#   symbol   : LaTeX symbol for axis labels
#
# Source: 01_Sample_Explorer.py (d_plotrange, d_unit_options)

PROPERTY_REGISTRY = {
    'S': {
        'column': 'Seebeck coefficient',
        'si_range': [-1e-3, 1e-3],
        'units': {
            'V/K': (1.0, 0),
            'uV/K': (1e6, 0),
            'mV/K': (1e3, 0),
        },
        'default_unit': 'uV/K',
        'symbol': r'Seebeck coefficient, $S$',
    },
    'sigma': {
        'column': 'Electrical conductivity',
        'si_range': [0, 1e6],
        'units': {
            'S/m': (1.0, 0),
            'S/cm': (1e-2, 0),
            '10^5 S/m': (1e-5, 0),
        },
        'default_unit': 'S/m',
        'symbol': r'Electrical conductivity, $\sigma$',
    },
    'log_sigma': {
        'column': 'log10(Electrical conductivity)',
        'si_range': [-2, 8],
        'units': {
            'log10(S/m)': (1.0, 0),
            'log10(S/cm)': (1.0, -2),
        },
        'default_unit': 'log10(S/m)',
        'symbol': r'log(Electrical conductivity, $\sigma$)',
    },
    'rho': {
        'column': 'Electrical resistivity',
        'si_range': [0, 1e-3],
        'units': {
            'Ohm m': (1.0, 0),
            'mOhm cm': (1e5, 0),
            'uOhm cm': (1e8, 0),
            'Ohm cm': (1e2, 0),
        },
        'default_unit': 'Ohm m',
        'symbol': r'Electrical resistivity, $\rho$',
    },
    'kappa': {
        'column': 'Thermal conductivity',
        'si_range': [0, 30],
        'units': {
            'W/(m K)': (1.0, 0),
            'W/(cm K)': (1e-2, 0),
        },
        'default_unit': 'W/(m K)',
        'symbol': r'Thermal conductivity, $\kappa$',
    },
    'kappaL': {
        'column': 'Lattice thermal conductivity',
        'si_range': [0, 30],
        'units': {
            'W/(m K)': (1.0, 0),
            'W/(cm K)': (1e-2, 0),
        },
        'default_unit': 'W/(m K)',
        'symbol': r'Lattice thermal conductivity, $\kappa_L$',
    },
    'ZT': {
        'column': 'ZT',
        'si_range': [0, 1.5],
        'units': {'': (1.0, 0)},
        'default_unit': '',
        'symbol': r'$ZT$',
    },
    'Z': {
        'column': 'Z',
        'si_range': [0, 0.01],
        'units': {
            '1/K': (1.0, 0),
            '10^-3 /K': (1e3, 0),
        },
        'default_unit': '1/K',
        'symbol': r'$Z$',
    },
    'PF': {
        'column': 'Power factor',
        'si_range': [0, 1e-2],
        'units': {
            'W/(m K^2)': (1.0, 0),
            'uW/(cm K^2)': (1e4, 0),
            'mW/(m K^2)': (1e3, 0),
        },
        'default_unit': 'W/(m K^2)',
        'symbol': r'Power factor, $S^2 \sigma$',
    },
    'muw': {
        'column': 'Weighted mobility',
        'si_range': [0, 0.1],
        'units': {
            'm^2/(V s)': (1.0, 0),
            'cm^2/(V s)': (1e4, 0),
        },
        'default_unit': 'cm^2/(V s)',
        'symbol': r'Weighted mobility, $\mu_w$',
    },
    'mu': {
        'column': 'Carrier mobility',
        'si_range': [0, 0.1],
        'units': {
            'm^2/(V s)': (1.0, 0),
            'cm^2/(V s)': (1e4, 0),
        },
        'default_unit': 'cm^2/(V s)',
        'symbol': r'Carrier mobility, $\mu$',
    },
    'RH': {
        'column': 'Hall coefficient',
        'si_range': [-1e-6, 1e-6],
        'units': {
            'm^3/C': (1.0, 0),
            'cm^3/C': (1e6, 0),
        },
        'default_unit': 'm^3/C',
        'symbol': r'Hall coefficient, $R_H$',
    },
    'PCA1': {
        'column': 'PCA1',
        'si_range': [None, None],
        'units': {'': (1.0, 0)},
        'default_unit': '',
        'symbol': 'PCA component 1',
    },
    'PCA2': {
        'column': 'PCA2',
        'si_range': [None, None],
        'units': {'': (1.0, 0)},
        'default_unit': '',
        'symbol': 'PCA component 2',
    },
}


# ---------------------------------------------------------------------------
# Unified configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_FIGURE_CONFIG = {
    'font_family': 'DejaVu Sans',
    'font_size': 9,
    'width_cm': 10,
    'height_cm': 7,
    'dpi': 300,
    'show': True,
    'bbox_inches': None,
    # Multi-panel specific (TEplot, TEstack, TErow):
    'ncols': 2,
    'hspace': 0.0,
    'wspace': 0.3,
    'box_aspect': 0.6,
}

DEFAULT_PLOT_CONFIG = {
    'marker_size': 1,
    'alpha': 0.2,
    'curve_alpha': 0.01,
    'label_marker_size': 1,
    'pca_margin': 0.1,
    'label_offset': 0.04,
    'label_fontsize': 8,
}


def _merge_config(defaults, overrides, extra_defaults=None):
    """Merge configuration dicts: defaults <- extra_defaults <- overrides.

    Parameters:
        defaults (dict): Base defaults (DEFAULT_FIGURE_CONFIG or DEFAULT_PLOT_CONFIG)
        overrides (dict or None): User-supplied overrides
        extra_defaults (dict or None): Function-specific defaults that override
                                       base defaults but are overridden by user values
    Returns:
        dict: Merged configuration
    """
    result = dict(defaults)
    if extra_defaults:
        result.update(extra_defaults)
    if overrides:
        result.update(overrides)
    return result


def _auto_filename(func_name, prop_x=None, prop_y=None, T=None,
                   composition_filter=None):
    """Generate a descriptive filename for auto-save."""
    import re
    parts = [func_name]
    if prop_x and prop_y:
        parts.append(f'{prop_x}_vs_{prop_y}')
    elif prop_y:
        parts.append(prop_y)
    if T:
        parts.append(f'{T}K')
    if composition_filter:
        safe = re.sub(r'[^\w]', '_', composition_filter)
        safe = re.sub(r'_+', '_', safe).strip('_')
        if len(safe) > 40:
            safe = safe[:40]
        parts.append(safe)
    return '_'.join(parts)


def _resolve_prop(key, unit=None):
    """Resolve property key and unit into plotting parameters.

    Parameters:
        key (str): Short key (e.g. 'S') or raw column name
        unit (str): Unit label (e.g. 'uV/K'). None = use default unit.

    Returns:
        tuple: (column_name, factor, offset, label, range_min, range_max)
    """
    if key in PROPERTY_REGISTRY:
        p = PROPERTY_REGISTRY[key]
        if unit is None:
            unit = p['default_unit']
        if unit in p['units']:
            factor, offset = p['units'][unit]
        else:
            # Unknown unit — fall back to default
            factor, offset = p['units'][p['default_unit']]
            unit = p['default_unit']
        # Convert SI range to display range
        si_lo, si_hi = p['si_range']
        if si_lo is not None:
            disp_lo = si_lo * factor + offset
            disp_hi = si_hi * factor + offset
        else:
            disp_lo, disp_hi = None, None
        # Build label
        if unit:
            label = f"{p['symbol']} [{unit}]"
        else:
            label = p['symbol']
        return p['column'], factor, offset, label, disp_lo, disp_hi
    # Fallback: treat key as raw column name
    return key, 1.0, 0, key, None, None


def list_properties():
    """Print available property keys and their units."""
    for key, p in PROPERTY_REGISTRY.items():
        units = ', '.join(p['units'].keys()) or '(dimensionless)'
        default = p['default_unit'] or '(dimensionless)'
        print(f"  {key:12s}  {p['column']:45s}  units: {units}  (default: {default})")


def plot_scatter(ax, df, prop_x, prop_y, unit_x=None, unit_y=None,
                 plot_config=None):
    """Generalized scatter plot of any two properties.

    Parameters:
        ax: matplotlib Axes
        df (pd.DataFrame): Data with property columns and 'color' column
        prop_x (str): Property key for x-axis (see PROPERTY_REGISTRY)
        prop_y (str): Property key for y-axis (see PROPERTY_REGISTRY)
        unit_x (str): Unit for x-axis (None = default unit)
        unit_y (str): Unit for y-axis (None = default unit)
        plot_config (dict): Override marker_size, alpha, x_min, x_max, y_min, y_max
    """
    pc = plot_config or {}
    col_x, fx, ox, label_x, def_xmin, def_xmax = _resolve_prop(prop_x, unit_x)
    col_y, fy, oy, label_y, def_ymin, def_ymax = _resolve_prop(prop_y, unit_y)

    x_vals = df[col_x] * fx + ox
    y_vals = df[col_y] * fy + oy

    if def_xmin is not None:
        ax.set_xlim(pc.get('x_min', def_xmin), pc.get('x_max', def_xmax))
    if def_ymin is not None:
        ax.set_ylim(pc.get('y_min', def_ymin), pc.get('y_max', def_ymax))

    ax.scatter(x_vals, y_vals,
               marker='o', s=pc.get('marker_size', 1),
               alpha=pc.get('alpha', 0.2), color=df['color'])

    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.grid()


def single_plot(df_sample, df_curve, df_int,
                composition_filter, n_cluster,
                prop_x, prop_y,
                unit_x=None, unit_y=None,
                l_label=None, T=400,
                plot_config=None, figure_config=None,
                filename=None, save_path=None):
    """Standalone scatter plot of any two properties.

    Parameters:
        df_sample: Sample dataframe with element columns
        df_curve: Curve dataframe
        df_int: Interpolated data
        composition_filter (str): Pandas query string
        n_cluster (int): Number of PCA clusters
        prop_x (str): Property key for x-axis (see PROPERTY_REGISTRY)
        prop_y (str): Property key for y-axis (see PROPERTY_REGISTRY)
        unit_x (str): Unit for x-axis (None = default). See list_properties().
        unit_y (str): Unit for y-axis (None = default). See list_properties().
        l_label (list): Compositions to highlight (used for data prep)
        T (int): Temperature [K]
        plot_config (dict): Override plot appearance (marker_size, alpha,
                            x_min, x_max, y_min, y_max)
        figure_config (dict): Override figure layout (font_family, font_size,
                              width_cm, height_cm)
        filename (str): If set, save figure as <filename>.png
        save_path (str): Directory to save figure

    Returns:
        tuple: (fig, ax)
    """
    import matplotlib.pyplot as plt
    fc = _merge_config(DEFAULT_FIGURE_CONFIG, figure_config,
                       {'width_cm': 10, 'height_cm': 7})
    _setup_matplotlib_font(fc['font_family'], fc['font_size'])
    df_s, df_label, df_int_s, _ = _prepare_teplot_data(
        df_sample, df_curve, df_int, composition_filter, n_cluster, l_label, T)
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(fc['width_cm'] * cm, fc['height_cm'] * cm))
    plot_scatter(ax, df_int_s, prop_x, prop_y, unit_x, unit_y, plot_config)
    ax.set_title(f'{T} K')
    fig.tight_layout()
    if filename is None:
        filename = _auto_filename('scatter', prop_x, prop_y, T, composition_filter)
    filepath = f'{save_path or "."}/{filename}.png'
    fig.savefig(filepath, dpi=fc['dpi'], bbox_inches=fc.get('bbox_inches'))
    print(f'Saved: {filepath}')
    if fc['show']:
        plt.show()
    plt.close(fig)
    return fig, ax


def plot_pca_scatter(ax, df_s, df_label, plot_config):
    """PCA scatter plot with cluster coloring and labeled points.

    Source: 251114_TEfamilyplot4.py
    """
    margin = plot_config.get('pca_margin', 0.1)
    ax.set_xlim(df_s['PCA1'].min() - margin, df_s['PCA1'].max() + margin * 3)
    ax.set_ylim(df_s['PCA2'].min() - margin * 2, df_s['PCA2'].max() + margin * 2)

    ax.scatter(df_s['PCA1'], df_s['PCA2'],
               marker='o', s=plot_config.get('marker_size', 0.5),
               alpha=plot_config.get('alpha', 0.15), color=df_s['color'])

    ax.scatter(df_label['PCA1'], df_label['PCA2'],
               marker='o', s=plot_config.get('label_marker_size', 1),
               alpha=1, color='black')

    offset = plot_config.get('label_offset', 0.04)
    fontsize = plot_config.get('label_fontsize', 8)
    for _, row in df_label.iterrows():
        ax.text(row['PCA1'] + offset, row['PCA2'], row['composition'],
                fontsize=fontsize, ha='left', va='center')

    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")


def plot_curves(ax, df_alldata, prop_y='ZT', unit_y=None, plot_config=None):
    """Temperature vs property line plot for all samples.

    Parameters:
        ax: matplotlib Axes
        df_alldata (pd.DataFrame): Data with 'x', 'y', 'color' columns
            (x = temperature arrays, y = property arrays)
        prop_y (str): Property key for y-axis (see PROPERTY_REGISTRY).
            Used for axis label and default range. Default 'ZT'.
        unit_y (str): Unit for y-axis (None = default unit)
        plot_config (dict): Override x_min, x_max, y_min, y_max, curve_alpha
    """
    pc = plot_config or {}
    _, fy, oy, label_y, def_ymin, def_ymax = _resolve_prop(prop_y, unit_y)

    ax.set_xlim(pc.get('x_min', pc.get('temp_min', 100)),
                pc.get('x_max', pc.get('temp_max', 1000)))
    if def_ymin is not None:
        ax.set_ylim(pc.get('y_min', def_ymin), pc.get('y_max', def_ymax))

    for i in df_alldata.index:
        try:
            l_x = eval(df_alldata.at[i, 'x'])
            l_y = np.array(eval(df_alldata.at[i, 'y'])) * fy + oy
            color = df_alldata.at[i, 'color']
            ax.plot(l_x, l_y, alpha=pc.get('curve_alpha', 0.01), color=color)
        except Exception:
            pass

    ax.set_xlabel(r"Temperature, $T$ [K]")
    ax.set_ylabel(label_y)


def plot_temperature_zt_curves(ax, df_alldata, plot_config):
    """Temperature-ZT curves. Legacy wrapper around plot_curves."""
    pc = {'y_min': plot_config.get('zt_min', 0),
          'y_max': plot_config.get('zt_max', 3),
          'x_min': plot_config.get('temp_min', 100),
          'x_max': plot_config.get('temp_max', 1000),
          'curve_alpha': plot_config.get('curve_alpha', 0.01)}
    plot_curves(ax, df_alldata, prop_y='ZT', unit_y=None, plot_config=pc)


def single_curves(df_sample, df_curve, df_int,
                  composition_filter, n_cluster,
                  prop_y='ZT', unit_y=None,
                  l_label=None, T=400,
                  plot_config=None, figure_config=None,
                  filename=None, save_path=None):
    """Standalone Temperature vs property line plot.

    Parameters:
        df_sample: Sample dataframe with element columns
        df_curve: Curve dataframe
        df_int: Interpolated data
        composition_filter (str): Pandas query string
        n_cluster (int): Number of PCA clusters
        prop_y (str): Property key for y-axis (see PROPERTY_REGISTRY), or
            raw column name matching prop_y in df_curve (e.g.
            'Seebeck coefficient', 'Thermal conductivity', 'ZT').
        unit_y (str): Unit for y-axis (None = default)
        l_label (list): Compositions to highlight
        T (int): Temperature [K] (used for data prep, not the plot x-axis)
        plot_config (dict): Override x_min, x_max, y_min, y_max, curve_alpha
        figure_config (dict): Override font_family, font_size, width_cm, height_cm
        filename (str): If set, save figure as <filename>.png
        save_path (str): Directory to save figure

    Returns:
        tuple: (fig, ax)
    """
    import matplotlib.pyplot as plt
    fc = _merge_config(DEFAULT_FIGURE_CONFIG, figure_config,
                       {'width_cm': 10, 'height_cm': 8})
    _setup_matplotlib_font(fc['font_family'], fc['font_size'])
    # Resolve the curve property name for filtering df_curve
    if prop_y in PROPERTY_REGISTRY:
        curve_prop = PROPERTY_REGISTRY[prop_y]['column']
    else:
        curve_prop = prop_y
    _, _, _, df_alldata = _prepare_teplot_data(
        df_sample, df_curve, df_int, composition_filter, n_cluster, l_label, T,
        curve_prop=curve_prop)
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(fc['width_cm'] * cm, fc['height_cm'] * cm))
    plot_curves(ax, df_alldata, prop_y, unit_y, plot_config)
    ax.set_title(f'{len(df_alldata)} curves')
    fig.tight_layout()
    if filename is None:
        filename = _auto_filename('curves', prop_y=prop_y, T=T,
                                  composition_filter=composition_filter)
    filepath = f'{save_path or "."}/{filename}.png'
    fig.savefig(filepath, dpi=fc['dpi'], bbox_inches=fc.get('bbox_inches'))
    print(f'Saved: {filepath}')
    if fc['show']:
        plt.show()
    plt.close(fig)
    return fig, ax


def plot_inverse_jonker(ax, df_int_s, plot_config):
    """Inverse Jonker plot (Seebeck vs log conductivity). Wrapper around plot_scatter."""
    pc = {'marker_size': plot_config.get('marker_size', 0.5),
          'alpha': plot_config.get('alpha', 0.15),
          'x_min': plot_config.get('seebeck_min', -500),
          'x_max': plot_config.get('seebeck_max', 500),
          'y_min': plot_config.get('logsigma_min', 0),
          'y_max': plot_config.get('logsigma_max', 7)}
    plot_scatter(ax, df_int_s, 'S', 'log_sigma', 'uV/K', 'log10(S/m)', pc)


def plot_seebeck_zt(ax, df_int_s, plot_config):
    """Seebeck coefficient vs ZT scatter plot. Wrapper around plot_scatter."""
    pc = {'marker_size': plot_config.get('marker_size_zt', 1),
          'alpha': plot_config.get('alpha_zt', 0.2),
          'x_min': plot_config.get('seebeck_min', -500),
          'x_max': plot_config.get('seebeck_max', 500),
          'y_min': plot_config.get('zt_min', 0),
          'y_max': plot_config.get('zt_max', 3)}
    plot_scatter(ax, df_int_s, 'S', 'ZT', 'uV/K', None, pc)


def plot_sigma_kappa(ax, df_int_s, plot_config):
    """Electrical conductivity vs thermal conductivity scatter. Wrapper around plot_scatter."""
    pc = {'marker_size': plot_config.get('marker_size', 0.5),
          'alpha': plot_config.get('alpha', 0.15),
          'x_min': plot_config.get('sigma_min', 0),
          'x_max': plot_config.get('sigma_max', 10),
          'y_min': plot_config.get('kappa_min', 0),
          'y_max': plot_config.get('kappa_max', 10)}
    plot_scatter(ax, df_int_s, 'sigma', 'kappa', '10^5 S/m', 'W/(m K)', pc)


def plot_logsigma_zt(ax, df_int_s, plot_config):
    """Log electrical conductivity vs ZT scatter. Wrapper around plot_scatter."""
    pc = {'marker_size': plot_config.get('marker_size', 0.5),
          'alpha': plot_config.get('alpha', 0.15),
          'x_min': plot_config.get('logsigma_min', 0),
          'x_max': plot_config.get('logsigma_max', 7),
          'y_min': plot_config.get('zt_min', 0),
          'y_max': plot_config.get('zt_max', 3)}
    plot_scatter(ax, df_int_s, 'log_sigma', 'ZT', 'log10(S/m)', None, pc)


def _prepare_teplot_data(df_sample, df_curve, df_int,
                         composition_filter, n_cluster, l_label, T,
                         curve_prop='ZT'):
    """Shared data preparation for TEplot functions.

    Parameters:
        curve_prop (str): Property name for Temperature-Y curve filtering.
            Must match a prop_y value in df_curve (e.g. 'ZT',
            'Seebeck coefficient', 'Thermal conductivity').
            Default 'ZT' for backward compatibility.

    Returns:
        tuple: (df_s, df_label, df_int_s, df_alldata)
    """
    if l_label is None:
        l_label = []
    df_s = pca2(df_sample.query(composition_filter), n_cluster)
    df_label = df_s.query(f'composition in {l_label}').drop_duplicates('composition')
    df_int_s = pd.merge(df_s, df_int[df_int['Temperature'] == T],
                         on='sample_id', how='left')
    df_curve_s = pd.merge(df_s, df_curve, on='sample_id', how='left')
    df_prop = df_curve_s.query(
        'prop_x=="Temperature" and prop_y==@curve_prop')
    df_alldata = pd.merge(df_s[['sample_id', 'cluster', 'color']],
                           df_prop[['sample_id', 'x', 'y']],
                           on='sample_id', how='inner')
    return df_s, df_label, df_int_s, df_alldata


def _setup_matplotlib_font(font_family='DejaVu Sans', font_size=9):
    """Configure matplotlib font with fallback."""
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager
    font = font_family
    available = {f.name for f in fontManager.ttflist}
    if font not in available:
        font = 'DejaVu Sans'
    plt.rcParams.update({
        'font.family': font, 'font.size': font_size,
        'mathtext.fontset': 'custom', 'mathtext.rm': font,
        'mathtext.it': f'{font}:italic', 'mathtext.bf': f'{font}:bold',
    })
    return font


def single_pca_scatter(df_sample, df_curve, df_int,
                       composition_filter, n_cluster,
                       l_label=None, T=400,
                       plot_config=None, figure_config=None,
                       filename=None, save_path=None):
    """Standalone PCA scatter plot with cluster coloring.

    Returns:
        tuple: (fig, ax)
    """
    import matplotlib.pyplot as plt
    pc = _merge_config(DEFAULT_PLOT_CONFIG, plot_config,
                       {'marker_size': 0.5, 'alpha': 0.15})
    fc = _merge_config(DEFAULT_FIGURE_CONFIG, figure_config,
                       {'width_cm': 10, 'height_cm': 8})
    _setup_matplotlib_font(fc['font_family'], fc['font_size'])
    df_s, df_label, _, _ = _prepare_teplot_data(
        df_sample, df_curve, df_int, composition_filter, n_cluster, l_label, T)
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(fc['width_cm'] * cm, fc['height_cm'] * cm))
    plot_pca_scatter(ax, df_s, df_label, pc)
    ax.set_title(f'{len(df_s)} samples')
    fig.tight_layout()
    if filename is None:
        filename = _auto_filename('pca', T=T,
                                  composition_filter=composition_filter)
    filepath = f'{save_path or "."}/{filename}.png'
    fig.savefig(filepath, dpi=fc['dpi'], bbox_inches=fc.get('bbox_inches'))
    print(f'Saved: {filepath}')
    if fc['show']:
        plt.show()
    plt.close(fig)
    return fig, ax


def single_temperature_zt(df_sample, df_curve, df_int,
                          composition_filter, n_cluster, **kwargs):
    """Standalone T-ZT curve plot. Shortcut for single_curves(..., 'ZT')."""
    return single_curves(df_sample, df_curve, df_int,
                         composition_filter, n_cluster,
                         prop_y='ZT', **kwargs)


# Legacy aliases — call single_plot(... prop_x, prop_y) for the general case
def single_inverse_jonker(df_sample, df_curve, df_int,
                          composition_filter, n_cluster, **kwargs):
    """Standalone inverse Jonker plot. Shortcut for single_plot(..., 'S', 'log_sigma')."""
    return single_plot(df_sample, df_curve, df_int,
                       composition_filter, n_cluster,
                       prop_x='S', prop_y='log_sigma', **kwargs)


def single_seebeck_zt(df_sample, df_curve, df_int,
                      composition_filter, n_cluster, **kwargs):
    """Standalone S vs ZT plot. Shortcut for single_plot(..., 'S', 'ZT')."""
    return single_plot(df_sample, df_curve, df_int,
                       composition_filter, n_cluster,
                       prop_x='S', prop_y='ZT', **kwargs)


def single_sigma_kappa(df_sample, df_curve, df_int,
                       composition_filter, n_cluster, **kwargs):
    """Standalone sigma vs kappa plot. Shortcut for single_plot(..., 'sigma', 'kappa')."""
    return single_plot(df_sample, df_curve, df_int,
                       composition_filter, n_cluster,
                       prop_x='sigma', prop_y='kappa', **kwargs)


def single_logsigma_zt(df_sample, df_curve, df_int,
                       composition_filter, n_cluster, **kwargs):
    """Standalone log(sigma) vs ZT plot. Shortcut for single_plot(..., 'log_sigma', 'ZT')."""
    return single_plot(df_sample, df_curve, df_int,
                       composition_filter, n_cluster,
                       prop_x='log_sigma', prop_y='ZT', **kwargs)


# Default panel layouts for TEplot4 and TEplot6
PANELS_TEPLOT4 = [
    ('pca',),
    ('curves', 'ZT'),
    ('scatter', 'S', 'log_sigma', 'uV/K', 'log10(S/m)'),
    ('scatter', 'S', 'ZT', 'uV/K', None),
]

PANELS_TEPLOT6 = [
    ('pca',),
    ('curves', 'ZT'),
    ('scatter', 'S', 'log_sigma', 'uV/K', 'log10(S/m)'),
    ('scatter', 'S', 'ZT', 'uV/K', None),
    ('scatter', 'sigma', 'kappa', '10^5 S/m', 'W/(m K)'),
    ('scatter', 'log_sigma', 'ZT', 'log10(S/m)', None),
]


def TEplot(df_sample, df_curve, df_int,
           composition_filter, n_cluster,
           panels=None,
           l_label=None, filename='TEplot', T=400,
           plot_config=None, figure_config=None, save_path=None):
    """Configurable multi-panel thermoelectric materials visualization.

    Parameters:
        df_sample: Sample dataframe with element columns
        df_curve: Curve dataframe
        df_int: Interpolated data (all temperatures concatenated)
        composition_filter (str): Pandas query string
        n_cluster (int): Number of PCA clusters
        panels (list): Panel specifications. Each element is a tuple:
            - ('pca',)                                  PCA scatter
            - ('curves', prop_y)                        T-Y line plot
            - ('curves', prop_y, unit_y)                T-Y with unit
            - ('scatter', prop_x, prop_y)               X-Y scatter
            - ('scatter', prop_x, prop_y, ux, uy)       X-Y with units
            Default: PANELS_TEPLOT4 (4 panels).
        l_label (list): Compositions to highlight with labels
        filename (str): Output filename base
        T (int): Temperature [K] for scatter plots
        plot_config (dict): Override plot appearance (marker_size, alpha,
                            curve_alpha, x_min/x_max/y_min/y_max per panel)
        figure_config (dict): Override figure layout (font_family, font_size,
                              width_cm, height_cm, ncols, height_ratios,
                              hspace, wspace, box_aspect)
        save_path (str): Directory to save figures

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if panels is None:
        panels = PANELS_TEPLOT4
    if l_label is None:
        l_label = []
    if save_path is None:
        save_path = '.'

    n_panels = len(panels)
    fc = _merge_config(DEFAULT_FIGURE_CONFIG, figure_config,
                       {'width_cm': 15})

    ncols = fc['ncols']
    nrows = -(-n_panels // ncols)  # ceiling division

    # Auto-compute height if not specified
    if 'height_cm' not in (figure_config or {}):
        fc['height_cm'] = 7 * nrows
    if 'height_ratios' not in (figure_config or {}):
        fc['height_ratios'] = [1] * nrows

    pc = _merge_config(DEFAULT_PLOT_CONFIG, plot_config,
                       {'marker_size': 0.5, 'alpha': 0.15, 'alpha_zt': 0.2,
                        'marker_size_zt': 1})

    # Determine which curve properties are needed
    curve_props = set()
    for panel in panels:
        if panel[0] == 'curves':
            prop_y = panel[1] if len(panel) > 1 else 'ZT'
            col = PROPERTY_REGISTRY[prop_y]['column'] if prop_y in PROPERTY_REGISTRY else prop_y
            curve_props.add(col)
    if not curve_props:
        curve_props.add('ZT')  # always prepare ZT for backward compat

    # Prepare data — use the first curve_prop for df_alldata
    primary_curve = list(curve_props)[0]
    df_s, df_label, df_int_s, df_alldata = _prepare_teplot_data(
        df_sample, df_curve, df_int, composition_filter, n_cluster, l_label, T,
        curve_prop=primary_curve)

    # Prepare additional curve data if multiple curve properties are used
    all_curve_data = {primary_curve: df_alldata}
    if len(curve_props) > 1:
        df_curve_s = pd.merge(df_s, df_curve, on='sample_id', how='left')
        for cp in curve_props:
            if cp not in all_curve_data:
                df_cp = df_curve_s.query(
                    'prop_x=="Temperature" and prop_y==@cp')
                all_curve_data[cp] = pd.merge(
                    df_s[['sample_id', 'cluster', 'color']],
                    df_cp[['sample_id', 'x', 'y']],
                    on='sample_id', how='inner')

    print(f'{len(df_s)} samples')

    # Configure matplotlib
    _setup_matplotlib_font(fc['font_family'], fc['font_size'])

    cm = 1 / 2.54
    fig = plt.figure(figsize=(fc['width_cm'] * cm, fc['height_cm'] * cm))
    gs = gridspec.GridSpec(nrows, ncols, height_ratios=fc['height_ratios'],
                           hspace=fc['hspace'], wspace=fc['wspace'])

    strT = f'{T} K'
    panel_labels = [chr(ord('a') + i) for i in range(n_panels)]

    for idx, panel in enumerate(panels):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])

        ptype = panel[0]

        if ptype == 'pca':
            ax.text(0.03, 0.97, f'({panel_labels[idx]}) {len(df_s)} samples',
                    transform=ax.transAxes, va='top', ha='left')
            plot_pca_scatter(ax, df_s, df_label, pc)

        elif ptype == 'curves':
            prop_y = panel[1] if len(panel) > 1 else 'ZT'
            unit_y = panel[2] if len(panel) > 2 else None
            col_name = PROPERTY_REGISTRY[prop_y]['column'] if prop_y in PROPERTY_REGISTRY else prop_y
            curve_df = all_curve_data[col_name]
            ax.text(0.03, 0.97, f'({panel_labels[idx]}) {len(curve_df)} curves',
                    transform=ax.transAxes, va='top', ha='left')
            plot_curves(ax, curve_df, prop_y, unit_y, pc)

        elif ptype == 'scatter':
            prop_x = panel[1]
            prop_y = panel[2]
            unit_x = panel[3] if len(panel) > 3 else None
            unit_y = panel[4] if len(panel) > 4 else None
            ax.text(0.03, 0.97, f'({panel_labels[idx]}) {strT}',
                    transform=ax.transAxes, va='top', ha='left')
            ax.set_box_aspect(fc.get('box_aspect', 0.6))
            plot_scatter(ax, df_int_s, prop_x, prop_y, unit_x, unit_y, pc)

    fig.tight_layout()
    fig.suptitle(f'{filename}: {composition_filter}; {strT}', y=1.02)
    filepath = f'{save_path}/{filename}.png'
    fig.savefig(filepath, dpi=fc['dpi'], bbox_inches=fc.get('bbox_inches', 'tight'))
    print(f'Saved: {filepath}')
    if fc['show']:
        plt.show()
    plt.close(fig)
    return fig


def TEplot4(df_sample, df_curve, df_int,
            composition_filter, n_cluster,
            panels=None,
            l_label=None, filename='TEplot4', T=400,
            alpha1=0.2, alpha2=0.05,
            plot_config=None, figure_config=None, save_path=None):
    """4-panel thermoelectric materials visualization.

    Default panels: (a) PCA, (b) T-ZT curves, (c) Inverse Jonker, (d) S-ZT.
    Override with the `panels` parameter.

    Parameters:
        df_sample: Sample dataframe with element columns
        df_curve: Curve dataframe
        df_int: Interpolated data (all temperatures concatenated)
        composition_filter (str): Pandas query string
        n_cluster (int): Number of PCA clusters
        panels (list): Custom panel specs (see TEplot). Default: PANELS_TEPLOT4.
        l_label (list): Compositions to highlight with labels
        filename (str): Output filename base
        T (int): Temperature [K] for scatter plots
        alpha1 (float): Scatter plot transparency
        alpha2 (float): Curve plot transparency
        plot_config (dict): Override plot appearance
        figure_config (dict): Override figure layout
        save_path (str): Directory to save figures
    """
    pc = _merge_config(DEFAULT_PLOT_CONFIG,
                       plot_config,
                       {'alpha': alpha1, 'alpha_zt': alpha1, 'curve_alpha': alpha2})
    fc = _merge_config(DEFAULT_FIGURE_CONFIG,
                       figure_config,
                       {'width_cm': 15, 'height_cm': 14})
    return TEplot(df_sample, df_curve, df_int,
                  composition_filter, n_cluster,
                  panels=panels or PANELS_TEPLOT4,
                  l_label=l_label, filename=filename, T=T,
                  plot_config=pc, figure_config=fc, save_path=save_path)


def TEplot6(df_sample, df_curve, df_int,
            composition_filter, n_cluster,
            panels=None,
            l_label=None, filename='TEplot6', T=400,
            plot_config=None, figure_config=None, save_path=None):
    """6-panel thermoelectric materials visualization.

    Default panels: (a) PCA, (b) T-ZT, (c) Inv Jonker, (d) S-ZT,
                    (e) sigma-kappa, (f) log(sigma)-ZT.
    Override with the `panels` parameter.

    Parameters:
        Same as TEplot4, but defaults to 6 panels (3x2 grid).
    """
    fc = _merge_config(DEFAULT_FIGURE_CONFIG,
                       figure_config,
                       {'font_size': 8, 'width_cm': 15, 'height_cm': 18})
    return TEplot(df_sample, df_curve, df_int,
                  composition_filter, n_cluster,
                  panels=panels or PANELS_TEPLOT6,
                  l_label=l_label, filename=filename, T=T,
                  plot_config=plot_config, figure_config=fc, save_path=save_path)


def _prepare_line_data(panels, df_s, df_curve, all_curve_data):
    """Ensure all_curve_data has entries for every 'curves' panel.

    Mutates all_curve_data in place and returns it.
    """
    curve_props = set()
    for panel in panels:
        if panel[0] == 'curves':
            prop_y = panel[1] if len(panel) > 1 else 'ZT'
            col = PROPERTY_REGISTRY[prop_y]['column'] if prop_y in PROPERTY_REGISTRY else prop_y
            curve_props.add(col)

    missing = curve_props - set(all_curve_data.keys())
    if missing:
        df_curve_s = pd.merge(df_s, df_curve, on='sample_id', how='left')
        for cp in missing:
            df_cp = df_curve_s.query(
                'prop_x=="Temperature" and prop_y==@cp')
            all_curve_data[cp] = pd.merge(
                df_s[['sample_id', 'cluster', 'color']],
                df_cp[['sample_id', 'x', 'y']],
                on='sample_id', how='inner')
    return all_curve_data


def _normalize_panels(panels):
    """Normalize a mixed panels list.

    Accepts:
        - str shorthand:  'S'            -> ('curves', 'S')
        - tuple shorthand: ('S', 'uV/K') -> ('curves', 'S', 'uV/K')
        - full tuple:      ('scatter', 'S', 'ZT', 'uV/K', None) -> as-is
        - full tuple:      ('curves', 'kappa')                   -> as-is
        - full tuple:      ('pca',)                              -> as-is

    Returns:
        list of normalized panel tuples
    """
    out = []
    for p in panels:
        if isinstance(p, str):
            out.append(('curves', p))
        elif isinstance(p, tuple) and p[0] not in ('pca', 'curves', 'scatter'):
            # Tuple shorthand like ('S',) or ('S', 'uV/K')
            out.append(('curves',) + p)
        else:
            out.append(p)
    return out


def _render_panel(ax, panel, df_s, df_label, df_int_s, all_curve_data,
                  pc, label_text, T):
    """Render a single panel onto an axes.

    Parameters:
        ax: matplotlib Axes
        panel: normalized panel tuple
        df_s, df_label, df_int_s: PCA/interpolated data
        all_curve_data: dict of {column_name: df_alldata}
        pc: plot_config dict
        label_text: e.g. '(a)'
        T: temperature [K]
    """
    ptype = panel[0]

    ax.text(0.02, 0.95, label_text,
            transform=ax.transAxes, va='top', ha='left')

    if ptype == 'pca':
        plot_pca_scatter(ax, df_s, df_label, pc)

    elif ptype == 'curves':
        prop_y = panel[1] if len(panel) > 1 else 'ZT'
        unit_y = panel[2] if len(panel) > 2 else None
        col_name = PROPERTY_REGISTRY[prop_y]['column'] if prop_y in PROPERTY_REGISTRY else prop_y
        curve_df = all_curve_data[col_name]
        plot_curves(ax, curve_df, prop_y, unit_y, pc)

    elif ptype == 'scatter':
        prop_x = panel[1]
        prop_y = panel[2]
        unit_x = panel[3] if len(panel) > 3 else None
        unit_y = panel[4] if len(panel) > 4 else None
        plot_scatter(ax, df_int_s, prop_x, prop_y, unit_x, unit_y, pc)


def TEstack(df_sample, df_curve, df_int,
            composition_filter, n_cluster,
            panels=None,
            l_label=None, filename='TEstack', T=400,
            sharex=False,
            plot_config=None, figure_config=None, save_path=None):
    """Vertically stacked plots in a single image.

    Supports mixed panel types: curves, scatter, and PCA.

    Parameters:
        df_sample: Sample dataframe with element columns
        df_curve: Curve dataframe
        df_int: Interpolated data (all temperatures concatenated)
        composition_filter (str): Pandas query string
        n_cluster (int): Number of PCA clusters
        panels (list): Panel specifications (top to bottom). Each element is:
            - str shorthand:   'S'              -> T-S curve plot
            - tuple shorthand: ('S', 'uV/K')    -> T-S curve with unit
            - ('curves', prop_y)                 -> T-Y curve plot
            - ('curves', prop_y, unit_y)         -> T-Y curve with unit
            - ('scatter', prop_x, prop_y)        -> X-Y scatter
            - ('scatter', px, py, ux, uy)        -> X-Y scatter with units
            - ('pca',)                           -> PCA scatter
            Default: ['S', 'log_sigma', 'kappa', 'ZT'] (all curves)
        l_label (list): Compositions to highlight with labels
        filename (str): Output filename base
        T (int): Temperature [K] for scatter plots and data prep
        sharex (bool): If True, all panels share the x-axis and only the
                       bottom panel shows x-axis labels. Default False.
        plot_config (dict): Override plot appearance. Supports per-panel
                            overrides via 'panel_configs' key (list of dicts).
        figure_config (dict): Override font_family, font_size,
                              width_cm, height_cm, height_ratios, hspace
        save_path (str): Directory to save figures

    Returns:
        tuple: (fig, list of axes)
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if panels is None:
        panels = ['S', 'log_sigma', 'kappa', 'ZT']
    panels = _normalize_panels(panels)
    if l_label is None:
        l_label = []
    if save_path is None:
        save_path = '.'

    n_panels = len(panels)

    fc = _merge_config(DEFAULT_FIGURE_CONFIG, figure_config,
                       {'width_cm': 10,
                        'hspace': 0.05 if not sharex else 0.0,
                        'bbox_inches': 'tight'})

    if 'height_cm' not in (figure_config or {}):
        fc['height_cm'] = 5 * n_panels
    if 'height_ratios' not in (figure_config or {}):
        fc['height_ratios'] = [1] * n_panels

    pc = _merge_config(DEFAULT_PLOT_CONFIG, plot_config,
                       {'marker_size': 0.5, 'alpha': 0.15})
    panel_configs = pc.pop('panel_configs', [{}] * n_panels)
    panel_configs = list(panel_configs) + [{}] * (n_panels - len(panel_configs))

    # Prepare data
    df_s, df_label, df_int_s, df_alldata = _prepare_teplot_data(
        df_sample, df_curve, df_int, composition_filter, n_cluster, l_label, T)
    all_curve_data = {'ZT': df_alldata}
    _prepare_line_data(panels, df_s, df_curve, all_curve_data)

    print(f'{len(df_s)} samples')

    _setup_matplotlib_font(fc['font_family'], fc['font_size'])

    cm = 1 / 2.54
    fig = plt.figure(figsize=(fc['width_cm'] * cm, fc['height_cm'] * cm))
    gs = gridspec.GridSpec(n_panels, 1, height_ratios=fc['height_ratios'],
                           hspace=fc['hspace'])

    axes = []
    panel_labels = [chr(ord('a') + i) for i in range(n_panels)]

    for idx, panel in enumerate(panels):
        if sharex and idx > 0:
            ax = fig.add_subplot(gs[idx, 0], sharex=axes[0])
        else:
            ax = fig.add_subplot(gs[idx, 0])

        panel_pc = dict(pc)
        panel_pc.update(panel_configs[idx])

        _render_panel(ax, panel, df_s, df_label, df_int_s, all_curve_data,
                      panel_pc, f'({panel_labels[idx]})', T)

        if sharex and idx < n_panels - 1:
            ax.set_xlabel('')
            plt.setp(ax.get_xticklabels(), visible=False)

        axes.append(ax)

    fig.tight_layout()
    fig.suptitle(f'{filename}: {composition_filter}', y=1.02)
    filepath = f'{save_path}/{filename}.png'
    fig.savefig(filepath, dpi=fc['dpi'], bbox_inches=fc.get('bbox_inches'))
    print(f'Saved: {filepath}')
    if fc['show']:
        plt.show()
    plt.close(fig)
    return fig, axes


def TErow(df_sample, df_curve, df_int,
          composition_filter, n_cluster,
          panels=None,
          l_label=None, filename='TErow', T=400,
          plot_config=None, figure_config=None, save_path=None):
    """Horizontally arranged plots in a single image.

    Supports mixed panel types: curves, scatter, and PCA.

    Parameters:
        df_sample: Sample dataframe with element columns
        df_curve: Curve dataframe
        df_int: Interpolated data (all temperatures concatenated)
        composition_filter (str): Pandas query string
        n_cluster (int): Number of PCA clusters
        panels (list): Panel specifications (left to right). Same format
            as TEstack: str shorthand, tuple shorthand, or full tuples.
            Default: ['S', 'log_sigma', 'kappa', 'ZT'] (all curves)
        l_label (list): Compositions to highlight with labels
        filename (str): Output filename base
        T (int): Temperature [K] for scatter plots and data prep
        plot_config (dict): Override plot appearance. Supports per-panel
                            overrides via 'panel_configs' key (list of dicts).
        figure_config (dict): Override font_family, font_size,
                              width_cm, height_cm, wspace
        save_path (str): Directory to save figures

    Returns:
        tuple: (fig, list of axes)
    """
    import matplotlib.pyplot as plt

    if panels is None:
        panels = ['S', 'log_sigma', 'kappa', 'ZT']
    panels = _normalize_panels(panels)
    if l_label is None:
        l_label = []
    if save_path is None:
        save_path = '.'

    n_panels = len(panels)

    fc = _merge_config(DEFAULT_FIGURE_CONFIG, figure_config,
                       {'height_cm': 6, 'wspace': 0.35,
                        'bbox_inches': 'tight'})

    if 'width_cm' not in (figure_config or {}):
        fc['width_cm'] = 6 * n_panels

    pc = _merge_config(DEFAULT_PLOT_CONFIG, plot_config,
                       {'marker_size': 0.5, 'alpha': 0.15})
    panel_configs = pc.pop('panel_configs', [{}] * n_panels)
    panel_configs = list(panel_configs) + [{}] * (n_panels - len(panel_configs))

    # Prepare data
    df_s, df_label, df_int_s, df_alldata = _prepare_teplot_data(
        df_sample, df_curve, df_int, composition_filter, n_cluster, l_label, T)
    all_curve_data = {'ZT': df_alldata}
    _prepare_line_data(panels, df_s, df_curve, all_curve_data)

    print(f'{len(df_s)} samples')

    _setup_matplotlib_font(fc['font_family'], fc['font_size'])

    cm = 1 / 2.54
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(fc['width_cm'] * cm, fc['height_cm'] * cm))
    fig.subplots_adjust(wspace=fc['wspace'])
    if n_panels == 1:
        axes = [axes]

    panel_labels = [chr(ord('a') + i) for i in range(n_panels)]

    for idx, panel in enumerate(panels):
        ax = axes[idx]
        panel_pc = dict(pc)
        panel_pc.update(panel_configs[idx])

        _render_panel(ax, panel, df_s, df_label, df_int_s, all_curve_data,
                      panel_pc, f'({panel_labels[idx]})', T)

    fig.tight_layout()
    fig.suptitle(f'{filename}: {composition_filter}', y=1.02)
    filepath = f'{save_path}/{filename}.png'
    fig.savefig(filepath, dpi=fc['dpi'], bbox_inches=fc.get('bbox_inches'))
    print(f'Saved: {filepath}')
    if fc['show']:
        plt.show()
    plt.close(fig)
    return fig, axes


# =============================================================================
# 11. Plotting - plotly (interactive)
# =============================================================================

def plotly_2d(df, prop_1, prop_2, T, datapath='./'):
    """2D interactive scatter plot of two properties at given temperature.

    Source: 251112_starrydata_exotic.py

    Parameters:
        df (pd.DataFrame): Merged curves data with interpolated columns
        prop_1 (str): Property name for x-axis
        prop_2 (str): Property name for y-axis
        T (int): Temperature [K]
        datapath (str): Output directory for HTML file
    """
    import plotly.express as px

    col_y = f'y_{T}K'
    df_prop1 = df[df['prop_y'] == prop_1]
    df_prop2 = df[df['prop_y'] == prop_2]

    df_plot = pd.merge(
        df_prop1.rename(columns={col_y: 'x_plot'}),
        df_prop2[['sample_id', col_y]].rename(columns={col_y: 'y_plot'}),
        on='sample_id', how='inner')

    print(f'{len(df_plot)} samples')

    fig = px.scatter(df_plot, x='x_plot', y='y_plot',
                     color='mf_if' if 'mf_if' in df_plot.columns else None)

    filename = f"{datapath}{prop_2.replace(' ', '')}_{prop_1.replace(' ', '')}{T}K.html"
    fig.write_html(filename)
    print(f'Saved: {filename}')
    return fig


def plotly3(df, col_info, prop1, prop2, prop3, prop4,
            range1=None, range2=None, range3=None, range4=(0, 2),
            T=400, datapath='./', autoscale=False):
    """3D interactive scatter plot of three properties with color scale.

    Source: 251112_starrydata_exotic.py / 250603_TEfamilies.py

    Parameters:
        df (pd.DataFrame): Interpolated data at temperature T
        col_info (list): Info columns to include, e.g. ['SID','sample_id','composition']
        prop1, prop2, prop3 (str): Property names for x, y, z axes
        prop4 (str): Property name for color scale
        range1..range4 (tuple): Axis ranges (None for autoscale)
        T (int): Temperature [K]
        datapath (str): Output directory
        autoscale (bool): If True, ignore axis ranges
    """
    import plotly.express as px

    df_plot = pd.DataFrame(df, columns=col_info + [prop1, prop2, prop3, prop4])
    df_plot.dropna(subset=[prop1, prop2, prop3, prop4], inplace=True)

    n_samples = len(df_plot)
    print(f'{n_samples} samples')

    fig = px.scatter_3d(
        df_plot, x=prop1, y=prop2, z=prop3, color=prop4,
        hover_data=['SID', 'sample_id', 'composition'] if 'SID' in df_plot.columns else None,
        title=f'{prop1} vs {prop2} vs {prop3} at {T}K ({n_samples} samples)',
        color_continuous_scale=px.colors.sequential.Turbo,
        range_color=range4)

    scene_kwargs = {}
    for axis, rng in [('xaxis', range1), ('yaxis', range2), ('zaxis', range3)]:
        d = dict(gridcolor='gray', backgroundcolor='black', color='white')
        if rng and not autoscale:
            d['range'] = rng
        scene_kwargs[axis] = d

    fig.update_layout(
        scene=scene_kwargs,
        plot_bgcolor='rgba(0,0,0,1)',
        paper_bgcolor='rgba(0,0,0,1)',
        font=dict(color='white'),
        coloraxis_colorbar=dict(title=prop4))

    fig.update_traces(marker=dict(size=1, symbol='circle', opacity=0.5))

    suffix = '_autoscale' if autoscale else ''
    p1 = prop1.replace(' ', '_')
    p2 = prop2.replace(' ', '_')
    p3 = prop3.replace(' ', '_')
    p4 = prop4.replace(' ', '_')
    filename = f"{datapath}{p1}_{p2}_{p3}_{p4}_{T}K{suffix}.html"
    fig.write_html(filename)
    print(f'Saved: {filename}')
    return fig


def plotly_pca3(df, df_labels, col_info,
                prop1='pca_x', prop2='pca_y', prop3='pca_z', prop4='ZT',
                range1=(-1, 1), range2=(-1, 1), range3=(-1, 1), range4=(0, 1.5),
                T=400, datapath='./', suffix='all'):
    """3D PCA scatter plot with composition labels.

    Source: 250603_TEfamilies.py / 251112_starrydata_exotic.py

    Parameters:
        df (pd.DataFrame): Data with PCA columns and properties
        df_labels (pd.DataFrame): Subset of df for text labels
        col_info (list): Info columns
        prop1..prop3 (str): PCA axis column names
        prop4 (str): Color scale property
        range1..range4 (tuple): Axis/color ranges
        T (int): Temperature [K]
        datapath (str): Output directory
        suffix (str): Material family name for filename
    """
    import plotly.express as px
    import plotly.graph_objects as go

    df_plot = pd.DataFrame(df, columns=col_info + [prop1, prop2, prop3, prop4])
    df_plot.dropna(subset=[prop1, prop2, prop3, prop4], inplace=True)

    n_samples = len(df_plot)
    print(f'{n_samples} samples')

    fig = px.scatter_3d(
        df_plot, x=prop1, y=prop2, z=prop3, color=prop4,
        hover_data=['SID', 'sample_id', 'composition'] if 'SID' in df_plot.columns else None,
        title=f'PCA {suffix}, {prop4} at {T}K ({n_samples} samples)',
        color_continuous_scale=px.colors.sequential.Turbo,
        range_color=range4)

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=range1, gridcolor='gray', backgroundcolor='black', color='white'),
            yaxis=dict(range=range2, gridcolor='gray', backgroundcolor='black', color='white'),
            zaxis=dict(range=range3, gridcolor='gray', backgroundcolor='black', color='white')),
        plot_bgcolor='rgba(0,0,0,1)',
        paper_bgcolor='rgba(0,0,0,1)',
        font=dict(color='white'),
        coloraxis_colorbar=dict(title=prop4))

    fig.update_traces(marker=dict(size=1, symbol='circle', opacity=0.5))

    # Add composition text labels
    pca_x_col = 'pca_x' if suffix == 'all' else 'pca_fx'
    pca_y_col = 'pca_y' if suffix == 'all' else 'pca_fy'
    pca_z_col = 'pca_z' if suffix == 'all' else 'pca_fz'
    for _, row in df_labels.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[row.get(pca_x_col, 0)], y=[row.get(pca_y_col, 0)], z=[row.get(pca_z_col, 0)],
            text=row['composition'], mode='text',
            textfont=dict(color='white', size=12), showlegend=False))

    if suffix == '':
        suffix = 'other'
    filename = f"{datapath}PCA_{suffix}_{prop4}_{T}K.html"
    fig.write_html(filename)
    print(f'Saved: {filename}')
    return fig


def plotly_curves(df_sample, df_curve, df_int,
                  composition_filter, n_cluster,
                  prop_y='ZT', unit_y=None,
                  l_label=None, T=400,
                  plot_config=None,
                  filename=None, save_path=None):
    """Interactive Temperature vs property line plot using plotly.

    Parameters:
        df_sample: Sample dataframe with element columns
        df_curve: Curve dataframe
        df_int: Interpolated data
        composition_filter (str): Pandas query string
        n_cluster (int): Number of PCA clusters
        prop_y (str): Property key for y-axis (see PROPERTY_REGISTRY), or
            raw column name matching prop_y in df_curve.
        unit_y (str): Unit for y-axis (None = default)
        l_label (list): Compositions to highlight
        T (int): Temperature [K] (used for data prep)
        plot_config (dict): Override curve_alpha, x_min, x_max, y_min, y_max
        filename (str): Output filename (without .html). Auto-generated if None.
        save_path (str): Directory to save HTML file

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    pc = _merge_config(DEFAULT_PLOT_CONFIG, plot_config)

    # Resolve property for filtering and axis label
    if prop_y in PROPERTY_REGISTRY:
        curve_prop = PROPERTY_REGISTRY[prop_y]['column']
    else:
        curve_prop = prop_y
    _, fy, oy, label_y, def_ymin, def_ymax = _resolve_prop(prop_y, unit_y)

    _, _, _, df_alldata = _prepare_teplot_data(
        df_sample, df_curve, df_int, composition_filter, n_cluster,
        l_label, T, curve_prop=curve_prop)

    print(f'{len(df_alldata)} curves')

    fig = go.Figure()

    for i in df_alldata.index:
        try:
            l_x = eval(df_alldata.at[i, 'x'])
            l_y = np.array(eval(df_alldata.at[i, 'y'])) * fy + oy
            color = df_alldata.at[i, 'color']
            sid = df_alldata.at[i, 'sample_id']
            fig.add_trace(go.Scatter(
                x=l_x, y=l_y, mode='lines',
                line=dict(color=color, width=1),
                opacity=pc.get('curve_alpha', 0.1),
                name=str(sid),
                hovertemplate=f'sample_id={sid}<br>T=%{{x}} K<br>{prop_y}=%{{y:.3g}}<extra></extra>',
                showlegend=False))
        except Exception:
            pass

    x_min = pc.get('x_min', pc.get('temp_min', 100))
    x_max = pc.get('x_max', pc.get('temp_max', 1000))
    fig.update_xaxes(title='Temperature, T [K]', range=[x_min, x_max])
    fig.update_yaxes(title=label_y)
    if def_ymin is not None:
        fig.update_yaxes(range=[pc.get('y_min', def_ymin),
                                pc.get('y_max', def_ymax)])

    fig.update_layout(
        title=f'{composition_filter}: {len(df_alldata)} curves',
        template='plotly_white')

    if filename is None:
        filename = _auto_filename('plotly_curves', prop_y=prop_y, T=T,
                                  composition_filter=composition_filter)
    filepath = f'{save_path or "."}/{filename}.html'
    fig.write_html(filepath)
    print(f'Saved: {filepath}')
    return fig
