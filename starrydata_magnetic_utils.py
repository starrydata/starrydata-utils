"""
starrydata_magnetic_utils.py - Utility functions for Starrydata magnetic materials analysis

Extracted from magnetic materials notebook-derived Python files (2020-2022).
Each function is the latest working version from the most recent notebook.

Sections:
    1. Constants (magnetic properties, parent compounds, column names)
    2. Data loading (load_magnetic_samples, load_magnetic_curves)
    3. Material family classification (classify_magnetic_families)
    4. Sample selection (selectsamples_mag)
    5. Composition averaging (averagecomp)
    6. Plotting - hysteresis (alldataplot_mag, dataplot_mag, sampleplot)
    7. Plotting - clustering (clusterplot)
    8. Physics (Brillouin)
    9. Utility (compgrid)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core.composition import Composition

# Import shared functions from starrydata_utils
from starrydata_utils import (
    L_ELEMENT, comp2dict, comp2vec, vec2comp, contains,
    flatten_dict, download_dataset, load_curves, load_samples,
    pca2, generate_rainbow_colors,
)


# =============================================================================
# 1. Constants
# =============================================================================

# Magnetic property column names in Starrydata
MAGNETIC_PROPERTIES = [
    'Magnetic field strength (H)',
    'Magnetization',
    'magnetization_per_weight',
    'magnetization_per_volume',
    'magnetization_Bohr',
    'Magnetic field',
    'magnetic　susceptibility',
    'Magnetic susceptibility_per_weight',
    'Magnetic susceptibility_volume',
    'inverse susceptibility',
    'inverse magnetic susceptibility_mol',
    'Coercivity Field',
    'Remanent Magnetization_Absolute',
]

# Standard column name shortcuts
COL_H = 'Magnetic field strength (H)'
COL_M = 'magnetization_per_weight'

# Standard magnetic material parent compounds
L_PARENTS_MAG = [
    'Fe3O4', 'CoFe2O4', 'CuFe2O4', 'NiFe2O4', 'MnFe2O4', 'ZnFe2O4',
    'BaFe12O19', 'SrFe12O19',
    'Nd2Fe14B', 'NdFeB',
    'SmCo5', 'SmCo10', 'Sm2Co17',
    'Mn54Al46', 'MnAl',
    'Fe', 'Co', 'Ni', 'FeCo', 'FeNi',
]

# Sample info keys relevant to magnetic materials
MAGNETIC_SAMPLE_INFO_KEYS = [
    'DataType', 'Form', 'FabricationProcess', 'Purity', 'RelativeDensity',
    'GrainSize', 'MagneticMeasurement', 'Measurement temperature',
    'magnetic field', 'Saturation magnetization', 'coercivity',
    'remanence magnetion',
]

MAGNETIC_FIELD_PROP_X = [
    'Magnetic field strength (H)',
    'Magnetic field',
    'Magnetic Field',
    'Applied Field',
    'magnetic field',
]

MAGNETIZATION_PROP_Y = [
    'magnetization_per_weight',
    'magnetization_per_volume',
    'Magnetization',
]


# =============================================================================
# 2. Data loading
# =============================================================================

def load_magnetic_samples(datapath):
    """Load and filter magnetic material samples from Starrydata dataset.

    Source: 220710_magnetic_alldata.py (latest)

    Parameters:
        datapath (str): Path to Starrydata dataset directory

    Returns:
        pd.DataFrame: Magnetic samples with columns
            [sid, DOI, sampleid, samplename, composition, sampleinfo]
    """
    df_curves, df_samples = _load_raw_data(datapath)

    df_mag = df_curves[df_curves['projectname'].str.contains('MagneticMaterials')]
    df_mag = pd.DataFrame(
        df_mag,
        columns=['sid', 'DOI', 'sampleid', 'samplename', 'composition', 'sampleinfo']
    ).drop_duplicates()

    return df_mag


def load_magnetic_curves(datapath, col_H=COL_H, col_M=COL_M,
                         H_range=(-2e6, 2e6), M_range=(-2e2, 2e2)):
    """Load magnetic hysteresis curve data with outlier filtering.

    Source: 220710_magnetic_alldata.py (latest)

    Parameters:
        datapath (str): Path to Starrydata dataset directory
        col_H (str): Column name for magnetic field
        col_M (str): Column name for magnetization
        H_range (tuple): (min, max) range for H filtering
        M_range (tuple): (min, max) range for M filtering

    Returns:
        pd.DataFrame: Filtered curve data with [sid, sampleid, col_H, col_M]
    """
    df_curves, _ = _load_raw_data(datapath)

    df = pd.DataFrame(
        df_curves.dropna(subset=[col_H, col_M]),
        columns=['sid', 'sampleid', col_H, col_M]
    )

    # Filter outliers
    df = df[(df[col_H] > H_range[0]) & (df[col_H] < H_range[1])]
    df = df[(df[col_M] > M_range[0]) & (df[col_M] < M_range[1])]

    return df


def _load_raw_data(datapath):
    """Load raw Starrydata CSV. Returns (df_raw, df_samples) or (df_raw, None).

    Tries both the load_curves/load_samples API and direct CSV loading.
    """
    import glob
    csv_files = glob.glob(datapath + '*_rawdata*.csv')
    if csv_files:
        df_raw = pd.read_csv(csv_files[0])
        return df_raw, None
    # Fall back to starrydata_utils loaders
    df_curves = load_curves(datapath)
    df_samples = load_samples(datapath)
    return df_curves, df_samples


def extract_sample_info(df_mag, keys=None):
    """Extract sample information fields from sampleinfo column.

    Source: 220710_magnetic_alldata.py

    Parameters:
        df_mag (pd.DataFrame): Magnetic samples DataFrame with 'sampleinfo' column
        keys (list): List of info keys to extract.
            Defaults to MAGNETIC_SAMPLE_INFO_KEYS.

    Returns:
        pd.DataFrame: Input DataFrame with added columns for each key
    """
    if keys is None:
        keys = MAGNETIC_SAMPLE_INFO_KEYS

    for key in keys:
        df_mag[key] = ''
        df_mag[key + '_details'] = ''

    for i in df_mag.index:
        try:
            d_si = eval(df_mag.at[i, 'sampleinfo'])
        except Exception:
            continue
        for key in keys:
            try:
                df_mag.at[i, key] = d_si[key]['category']
                df_mag.at[i, key + '_details'] = d_si[key]['comment']
            except Exception:
                pass

    return df_mag


# =============================================================================
# 3. Material family classification
# =============================================================================

def classify_magnetic_families(df_mag, cr=0.1, d=0.0001):
    """Classify magnetic material samples into material families.

    Source: 220710_magnetic_alldata.py (latest, identical in 211026)

    Rule-based classification using elemental composition thresholds.
    Classification is applied in order; later rules can override earlier ones.

    Families:
        - Oxide (O > 0.3)
        - Sulfide/Selenide (S + Se > 0.3)
        - Telluride (Te > 0.3)
        - Antimonide (Sb > 0.3)
        - Silicide (Si + Ge > 0.3)
        - Ferrite ((Fe+Co)/O ≈ 3/4 and Fe+Co+O > 1-cr)
        - Nd2Fe14B (RE/B ≈ 2, Fe/B ≈ 14, RE+Fe+B > 1-cr)
        - SmCo10 (Co+Fe+Ni)/Sm ≈ 10 and sum > 1-cr)
        - Mn-Al (Mn/(Al+Ga+In) ≈ 54/46 and sum > 1-cr)

    Parameters:
        df_mag (pd.DataFrame): Magnetic samples with 'd_comp' column
            (dict of element fractions from comp2dict)
        cr (float): Composition error tolerance (default 0.1)
        d (float): Small constant to prevent zero division (default 0.0001)

    Returns:
        pd.DataFrame: Input DataFrame with 'mf_if' column added
    """
    df_mag['mf_if'] = ''

    for i in df_mag.index:
        d_comp = df_mag.at[i, 'd_comp']
        if not isinstance(d_comp, dict):
            continue

        # Oxide
        try:
            n_O = d_comp['O'] + d
            if n_O > 0.3:
                df_mag.at[i, 'mf_if'] = 'Oxide'
        except Exception:
            pass

        # Sulfide/Selenide
        try:
            n_S = d_comp['S'] + d_comp['Se'] + d
            if n_S > 0.3:
                df_mag.at[i, 'mf_if'] = 'Sulfide/Selenide'
        except Exception:
            pass

        # Telluride
        try:
            n_Te = d_comp['Te'] + d
            if n_Te > 0.3:
                df_mag.at[i, 'mf_if'] = 'Telluride'
        except Exception:
            pass

        # Antimonide
        try:
            n_Sb = d_comp['Sb'] + d
            if n_Sb > 0.3:
                df_mag.at[i, 'mf_if'] = 'Antimonide'
        except Exception:
            pass

        # Silicide
        try:
            n_Si = d_comp['Si'] + d_comp['Ge'] + d
            if n_Si > 0.3:
                df_mag.at[i, 'mf_if'] = 'Silicide'
        except Exception:
            pass

        # Ferrite
        try:
            n_Fe = d_comp['Fe'] + d_comp['Co'] + d
            n_O = d_comp['O'] + d
            if (abs(n_Fe / n_O - 3 / 4) < 4 / 3 * cr
                    and (n_Fe + n_O > 1 - cr)):
                df_mag.at[i, 'mf_if'] = 'Ferrite'
        except Exception:
            pass

        # Nd2Fe14B (RE = rare earth elements)
        try:
            n_RE = (d_comp['Sc'] + d_comp['Y']
                    + d_comp['La'] + d_comp['Ce'] + d_comp['Pr'] + d_comp['Nd']
                    + d_comp['Sm'] + d_comp['Eu'] + d_comp['Gd'] + d_comp['Tb']
                    + d_comp['Dy'] + d_comp['Ho'] + d_comp['Er'] + d_comp['Tm']
                    + d_comp['Yb'] + d_comp['Lu'] + d)
            n_Fe = d_comp['Fe'] + d
            n_B = d_comp['B'] + d
            if (abs(n_RE / n_B - 2) < 2 * cr
                    and abs(n_Fe / n_B - 14) < 14 * cr
                    and (n_RE + n_Fe + n_B > 1 - cr)):
                df_mag.at[i, 'mf_if'] = 'Nd2Fe14B'
        except Exception:
            pass

        # SmCo10
        try:
            n_Co = d_comp['Co'] + d_comp['Fe'] + d_comp['Ni'] + d
            n_Sm = d_comp['Sm'] + d
            if (abs(n_Co / n_Sm - 10) < 10 * cr
                    and (n_Co + n_Sm > 1 - cr)):
                df_mag.at[i, 'mf_if'] = 'SmCo10'
        except Exception:
            pass

        # Mn-Al
        try:
            n_Mn = d_comp['Mn'] + d
            n_Al = d_comp['Al'] + d_comp['Ga'] + d_comp['In'] + d
            if (abs(n_Mn / n_Al - 54 / 46) < 46 / 54 * cr
                    and (n_Mn + n_Al > 1 - cr)):
                df_mag.at[i, 'mf_if'] = 'Mn-Al'
        except Exception:
            pass

    return df_mag


# =============================================================================
# 4. Sample selection
# =============================================================================

def selectsamples_mag(df, parent, threshold=0.2):
    """Select samples with composition close to a parent compound.

    Source: 210316_magneticdata.py

    Uses Euclidean distance between composition vectors.

    Parameters:
        df (pd.DataFrame): DataFrame with 'compvec' column
            (100-element composition vectors from comp2vec)
        parent (str): Parent compound formula, e.g. 'CoFe2O4'
        threshold (float): Maximum composition vector distance (default 0.2)

    Returns:
        pd.DataFrame: Filtered samples within threshold distance
    """
    parentvec = comp2vec(parent)
    distances = df['compvec'].apply(lambda v: np.linalg.norm(v - parentvec))
    return df[distances < threshold]


# =============================================================================
# 5. Composition averaging
# =============================================================================

def averagecomp(l_strcomps, threshold=0.01, decimal=3):
    """Calculate the average composition from a list of composition strings.

    Source: 200602_magnetic_alldata.py

    Parameters:
        l_strcomps (list): List of composition formula strings
        threshold (float): Minimum average fraction to keep an element
            (default 0.01)
        decimal (int): Decimal places for rounding (default 3)

    Returns:
        str or None: IUPAC formula string of the averaged composition,
            or None if conversion fails
    """
    vec = np.zeros(100)
    N = len(l_strcomps)
    if N == 0:
        return None

    for strcomp in l_strcomps:
        try:
            fcomp = Composition(strcomp).fractional_composition
            vec = vec + comp2vec(str(fcomp))
        except Exception:
            pass

    # Zero out elements below threshold
    vecsum = 0
    for i in range(100):
        if vec[i] < threshold * N:
            vec[i] = 0
        else:
            vecsum += vec[i]

    if vecsum == 0:
        return None

    fvec = vec / vecsum
    try:
        comp = vec2comp(fvec.round(decimal))
        return comp
    except Exception:
        return None


# =============================================================================
# 6. Plotting - hysteresis
# =============================================================================

def alldataplot_mag(df_data, df_mag, mf_col, l_families, l_colors,
                    col_H=COL_H, col_M=COL_M,
                    H_range=(-1.7e6, 1.7e6), M_range=(-2e2, 2e2),
                    alpha=0.2, figsize=(5, 5), savepath=None):
    """Plot magnetization curves for all samples, colored by material family.

    Source: 220710_magnetic_alldata.py (latest)

    Parameters:
        df_data (pd.DataFrame): Raw curve data with col_H, col_M, sampleid
        df_mag (pd.DataFrame): Samples DataFrame with sampleid and mf_col
        mf_col (str): Material family column name (e.g., 'mf_if')
        l_families (list): List of family names to plot
        l_colors (list): Corresponding list of colors
        col_H (str): Column name for magnetic field
        col_M (str): Column name for magnetization
        H_range (tuple): (xmin, xmax) for x-axis
        M_range (tuple): (ymin, ymax) for y-axis
        alpha (float): Line transparency (default 0.2)
        figsize (tuple): Figure size (default (5, 5))
        savepath (str): If provided, save PNG to this path

    Returns:
        tuple: (fig, ax) matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    for family, color in zip(l_families, l_colors):
        sr_sampleid = df_mag[df_mag[mf_col] == family]['sampleid']
        for sampleid in sr_sampleid:
            df_plot = df_data[df_data['sampleid'] == sampleid]
            ax.plot(df_plot[col_H], df_plot[col_M], alpha=alpha, color=color)

    ax.set_xlim(H_range)
    ax.set_ylim(M_range)
    ax.set_xlabel('Magnetic Field (A/m)')
    ax.set_ylabel('Magnetization per weight (T/kg)')
    ax.grid()

    if savepath:
        fig.savefig(savepath, dpi=200)
    plt.close(fig)
    return fig, ax


def dataplot_mag(df, col_x=COL_H, col_y=COL_M, plot_type='line',
                 alpha=0.2, x_range=(-5e5, 5e5), y_range=(-100, 100),
                 title=None, figsize=(5, 3), savepath=None):
    """Plot multiple sample curves on a single figure.

    Source: 210316_magneticdata.py

    Parameters:
        df (pd.DataFrame): Data with sampleid, col_x, col_y columns
        col_x (str): X-axis column name
        col_y (str): Y-axis column name
        plot_type (str): 'line' or 'scatter' (default 'line')
        alpha (float): Plot transparency (default 0.2)
        x_range (tuple): (xmin, xmax) axis range
        y_range (tuple): (ymin, ymax) axis range
        title (str): Optional title text at top of plot
        figsize (tuple): Figure size (default (5, 3))
        savepath (str): If provided, save PNG to this path

    Returns:
        tuple: (fig, ax) matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    sr_samples = df['sampleid'].drop_duplicates()
    for sampleid in sr_samples:
        df_plot = df[df['sampleid'] == sampleid]
        x = df_plot[col_x]
        y = df_plot[col_y]
        if plot_type == 'line':
            ax.plot(x, y, alpha=alpha)
        elif plot_type == 'scatter':
            ax.scatter(x, y, alpha=alpha)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)

    if title:
        sr_doi = df['DOI'].drop_duplicates() if 'DOI' in df.columns else []
        ax.text(x_range[0] + 10, y_range[1] * 1.02,
                f'{title}  {len(sr_doi)} papers / {len(sr_samples)} samples')

    ax.grid()

    if savepath:
        fig.savefig(savepath, dpi=200)
    plt.close(fig)
    return fig, ax


def sampleplot(df, sampleid, col_H=COL_H, col_M=COL_M,
               figsize=(10, 7), alpha=0.2, savepath=None):
    """Plot a single sample's hysteresis curve with up/down loop separation.

    Source: 210316_magneticdata.py

    Separates the hysteresis loop into field-increasing (up) and
    field-decreasing (down) branches by detecting the field sweep direction.

    Parameters:
        df (pd.DataFrame): Raw data with sampleid, col_H, col_M
        sampleid (int): Sample ID to plot
        col_H (str): Column name for magnetic field
        col_M (str): Column name for magnetization
        figsize (tuple): Figure size (default (10, 7))
        alpha (float): Point transparency (default 0.2)
        savepath (str): If provided, save PNG to this path

    Returns:
        tuple: (df_down, df_up) DataFrames for each branch of the loop
    """
    df_plot = df[df['sampleid'] == sampleid].reset_index(drop=True)

    xmin = df_plot[col_H].min()
    xmax = df_plot[col_H].max()
    ymin = df_plot[col_M].min()
    ymax = df_plot[col_M].max()
    mx = 0.1 * (xmax - xmin)
    my = 0.1 * (ymax - ymin)

    # Separate into up/down branches using reorder_hysteresis
    result = reorder_hysteresis(df_plot[col_H].values, df_plot[col_M].values)
    if result is not None:
        df_up = pd.DataFrame({col_H: result['H_up'], col_M: result['M_up']})
        df_down = pd.DataFrame({col_H: result['H_down'], col_M: result['M_down']})
    else:
        df_up = pd.DataFrame(columns=[col_H, col_M])
        df_down = pd.DataFrame(columns=[col_H, col_M])

    # Plot
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    ax.scatter(df_up[col_H], df_up[col_M], c='red', alpha=alpha, label='up')
    ax.scatter(df_down[col_H], df_down[col_M], c='blue', alpha=alpha, label='down')

    ax.set_xlim([xmin - mx, xmax + mx])
    ax.set_ylim([ymin - my, ymax + my])
    ax.set_xlabel(col_H)
    ax.set_ylabel(col_M)
    ax.text(xmin + 10, ymax + 0.3,
            f'sid={df_plot.at[0, "sid"] if "sid" in df_plot.columns else "?"}'
            f' sampleid={sampleid}'
            f' composition={df_plot.at[0, "composition"] if "composition" in df_plot.columns else "?"}')
    ax.grid()
    ax.legend()

    if savepath:
        fig.savefig(savepath, dpi=200)
    plt.close(fig)
    return df_down, df_up


# =============================================================================
# 7. Plotting - clustering
# =============================================================================

def clusterplot(df_samples, df_km, key_x, key_y, figsize=(6, 6),
                alpha=0.2, savepath=None):
    """Plot composition clustering results (PCA, t-SNE, or NMF coordinates).

    Source: 200602_magnetic_alldata.py

    Parameters:
        df_samples (pd.DataFrame): Samples with kmeans labels and
            coordinate columns (key_x, key_y)
        df_km (pd.DataFrame): Cluster summary with 'mode' column
        key_x (str): Column name for x coordinates (e.g., 'pca_x', 'tsne_x')
        key_y (str): Column name for y coordinates
        figsize (tuple): Figure size (default (6, 6))
        alpha (float): Point transparency (default 0.2)
        savepath (str): If provided, save PNG to this path

    Returns:
        tuple: (fig, ax) matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i in df_km.index:
        df = df_samples[df_samples['kmeans'] == i]
        ax.scatter(df[key_x], df[key_y],
                   label=f'{i}: {df_km.loc[i, "mode"]}', alpha=alpha)

    ax.legend(bbox_to_anchor=(1.8, 1), loc='upper right')

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return fig, ax


def cluster_magnetic_compositions(df_samples, a_comp, n_clusters=12):
    """K-means clustering on magnetic material compositions.

    Source: 200602_magnetic_alldata.py

    Parameters:
        df_samples (pd.DataFrame): Samples DataFrame
        a_comp (np.ndarray): Composition vectors array (N x 100)
        n_clusters (int): Number of clusters (default 12)

    Returns:
        tuple: (df_samples, df_km) where df_samples has 'kmeans' column added
            and df_km is the cluster summary
    """
    from sklearn.cluster import KMeans

    model_km = KMeans(n_clusters=n_clusters, n_init=10)
    model_km.fit(a_comp)
    df_samples['kmeans'] = model_km.labels_

    df_km = pd.DataFrame(range(n_clusters), columns=['cluster_id'])
    df_km['mode'] = ''
    df_km['mode_count'] = 0
    df_km['count'] = 0
    df_km['average_comp'] = ''

    for i in range(n_clusters):
        df = df_samples[df_samples['kmeans'] == i]
        df_km.loc[i, 'mode'] = df['mp_composition'].mode()[0] if len(df) > 0 else ''
        df_km.loc[i, 'mode_count'] = len(df[df['mp_composition'] == df_km.loc[i, 'mode']])
        df_km.loc[i, 'count'] = len(df)
        df_km.loc[i, 'average_comp'] = averagecomp(
            list(df['mp_composition']), threshold=0.05)

    df_km = df_km.sort_values('count', ascending=False)
    return df_samples, df_km


def reduce_dimensions(a_comp, method='pca', n_components=2, random_state=1):
    """Dimensionality reduction on composition vectors.

    Source: 200602_magnetic_alldata.py

    Parameters:
        a_comp (np.ndarray): Composition vectors array (N x 100)
        method (str): 'pca', 'tsne', or 'nmf' (default 'pca')
        n_components (int): Number of output dimensions (default 2)
        random_state (int): Random seed for reproducibility

    Returns:
        np.ndarray: Reduced coordinates array (N x n_components)
    """
    if method == 'pca':
        from sklearn.decomposition import PCA
        model = PCA(n_components=n_components)
        return model.fit_transform(a_comp)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        return TSNE(n_components=n_components,
                    random_state=random_state).fit_transform(a_comp)
    elif method == 'nmf':
        from sklearn.decomposition import NMF
        model = NMF(n_components=n_components)
        return model.fit_transform(a_comp)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'pca', 'tsne', or 'nmf'.")


# =============================================================================
# 8. Physics
# =============================================================================

def Brillouin(C, J, x):
    """Brillouin function for theoretical magnetization curve fitting.

    Source: 210316_magneticdata.py

    B_J(x) = C * ((2J+1)/(2J) * coth((2J+1)x/(2J)) - 1/(2J) * coth(x/(2J)))

    Parameters:
        C (float): Weiss constant (amplitude scaling)
        J (float): Total angular momentum quantum number
        x (float or np.ndarray): Reduced magnetic field (μ_B * B / (k_B * T))

    Returns:
        float or np.ndarray: Brillouin function value(s)
    """
    tanh_1 = np.tanh((2 * J + 1) * x / (2 * J))
    tanh_2 = np.tanh(x / (2 * J))
    return C * (((2 * J + 1) / (2 * J)) / tanh_1 - (1 / (2 * J)) / tanh_2)


# =============================================================================
# 9. Utility
# =============================================================================

def compgrid(df_mag, mf, mf_col='mf_if', n_col=9):
    """Display composition list in a grid format.

    Source: 220710_magnetic_alldata.py (latest)

    Parameters:
        df_mag (pd.DataFrame): Magnetic samples DataFrame
        mf (str): Material family name to filter
        mf_col (str): Material family column name (default 'mf_if')
        n_col (int): Number of columns in grid (default 9)

    Returns:
        pd.DataFrame: Grid of composition strings
    """
    df = df_mag[df_mag[mf_col] == mf].reset_index(drop=True)
    sr = df['composition']
    n_row = int(np.round(len(sr) / n_col, 0)) + 1

    df_grid = pd.DataFrame(np.zeros([n_row, n_col])).astype('object')
    for row in range(n_row):
        for col in range(n_col):
            idx = row * n_col + col
            if idx < len(sr):
                df_grid.at[row, col] = sr.iloc[idx]
            else:
                df_grid.at[row, col] = ''
    return df_grid


def prepare_magnetic_samples(df_mag):
    """Add composition dicts and vectors to magnetic samples DataFrame.

    Convenience function that adds 'd_comp' and 'compvec' columns.

    Parameters:
        df_mag (pd.DataFrame): Magnetic samples with 'composition' column

    Returns:
        pd.DataFrame: Input DataFrame with 'd_comp' and 'compvec' columns added
    """
    df_mag['d_comp'] = ''
    df_mag['d_comp'] = df_mag['d_comp'].astype('object')
    df_mag['compvec'] = ''
    df_mag['compvec'] = df_mag['compvec'].astype('object')

    for i in df_mag.index:
        try:
            str_comp = df_mag.at[i, 'composition']
            df_mag.at[i, 'd_comp'] = comp2dict(str_comp)
            df_mag.at[i, 'compvec'] = comp2vec(str_comp)
        except Exception:
            pass

    return df_mag


def reorder_hysteresis(a_H, a_M):
    """Separate a hysteresis loop into sorted up and down branches.

    Traces the hysteresis loop as a continuous path using nearest-neighbor
    in normalized (H, M) space. Starting from the top-right (max H, max M),
    the loop follows the upper curve with H decreasing to the bottom-left,
    then the lower curve with H increasing back to the top-right. The path
    is split at the minimum-H turning point into down and up branches.

    Parameters:
        a_H (array-like): Magnetic field values
        a_M (array-like): Magnetization values

    Returns:
        dict: Keys 'H_up', 'M_up' (ascending H), 'H_down', 'M_down'
            (descending H), 'is_hysteresis' (bool).
            Returns None for degenerate input (<2 points).
    """
    a_H = np.asarray(a_H, dtype=float)
    a_M = np.asarray(a_M, dtype=float)
    n = len(a_H)

    if n < 2:
        return None

    H_range = a_H.max() - a_H.min()
    M_range = a_M.max() - a_M.min()

    # Degenerate: no spread in H or M
    if H_range == 0 or M_range == 0:
        order = np.argsort(a_H)
        return {
            'H_up': a_H[order],
            'M_up': a_M[order],
            'H_down': np.array([], dtype=float),
            'M_down': np.array([], dtype=float),
            'is_hysteresis': False,
        }

    # Normalize to [0, 1]
    H_norm = (a_H - a_H.min()) / H_range
    M_norm = (a_M - a_M.min()) / M_range

    # Start from the point closest to top-right (max H, max M)
    start_dist = (H_norm - 1.0) ** 2 + (M_norm - 1.0) ** 2
    start_idx = np.argmin(start_dist)

    # Nearest-neighbor traversal in normalized space
    order = np.empty(n, dtype=int)
    order[0] = start_idx
    visited = np.zeros(n, dtype=bool)
    visited[start_idx] = True

    for step in range(1, n):
        cur = order[step - 1]
        dists = (H_norm - H_norm[cur]) ** 2 + (M_norm - M_norm[cur]) ** 2
        dists[visited] = np.inf
        nearest = np.argmin(dists)
        order[step] = nearest
        visited[nearest] = True

    H_ordered = a_H[order]
    M_ordered = a_M[order]

    # Split at the minimum-H turning point
    min_H_pos = np.argmin(H_ordered)

    # Down branch: top-right → bottom-left (H descending)
    H_down = H_ordered[:min_H_pos + 1]
    M_down = M_ordered[:min_H_pos + 1]

    # Up branch: bottom-left → top-right (H ascending)
    H_up = H_ordered[min_H_pos:]
    M_up = M_ordered[min_H_pos:]

    is_hysteresis = len(H_down) > 1 and len(H_up) > 1

    return {
        'H_up': H_up,
        'M_up': M_up,
        'H_down': H_down,
        'M_down': M_down,
        'is_hysteresis': is_hysteresis,
    }


def evaluate_hysteresis_properties(H_down, M_down, H_up, M_up):
    """Evaluate coercivity and saturation magnetization from hysteresis branches.

    Parameters:
        H_down (array-like): Magnetic field values on the down branch
        M_down (array-like): Magnetization values on the down branch
        H_up (array-like): Magnetic field values on the up branch
        M_up (array-like): Magnetization values on the up branch

    Returns:
        dict: Keys 'Hc_down', 'Hc_up', 'Hc' (coercivity), 'Ms' (saturation
            magnetization). Values are np.nan when not computable.
    """
    H_down = np.asarray(H_down, dtype=float)
    M_down = np.asarray(M_down, dtype=float)
    H_up = np.asarray(H_up, dtype=float)
    M_up = np.asarray(M_up, dtype=float)

    def _find_zero_crossing(H, M):
        """Find H where M crosses zero via linear interpolation."""
        if len(H) < 2:
            return np.nan
        for i in range(len(M) - 1):
            if M[i] * M[i + 1] < 0:
                # Linear interpolation: H at M=0
                frac = M[i] / (M[i] - M[i + 1])
                return H[i] + frac * (H[i + 1] - H[i])
            if M[i] == 0:
                return H[i]
        if M[-1] == 0:
            return H[-1]
        return np.nan

    Hc_down = _find_zero_crossing(H_down, M_down)
    Hc_up = _find_zero_crossing(H_up, M_up)

    # Average coercivity
    if not np.isnan(Hc_down) and not np.isnan(Hc_up):
        Hc = (abs(Hc_down) + abs(Hc_up)) / 2
    elif not np.isnan(Hc_down):
        Hc = abs(Hc_down)
    elif not np.isnan(Hc_up):
        Hc = abs(Hc_up)
    else:
        Hc = np.nan

    # Saturation magnetization: max |M| across both branches
    all_M = np.concatenate([
        M_down[np.isfinite(M_down)] if len(M_down) > 0 else np.array([]),
        M_up[np.isfinite(M_up)] if len(M_up) > 0 else np.array([]),
    ])
    Ms = float(np.max(np.abs(all_M))) if len(all_M) > 0 else np.nan

    return {'Hc_down': Hc_down, 'Hc_up': Hc_up, 'Hc': Hc, 'Ms': Ms}
