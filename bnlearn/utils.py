"""Shared utilities for bnlearn structure and parameter learning."""
# ------------------------------------
# Name        : utils.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------

import numpy as np
import pandas as pd


def infer_data_type(df):
    """Classify columns as discrete or continuous for score / CI / parameter selection.

    Numeric columns with many distinct values are treated as continuous;
    low-cardinality integer (and non-numeric) columns as discrete.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.

    Returns
    -------
    dict
        Keys: 'dtype' ('discrete' | 'continuous' | 'mixed'),
        'discrete' (list of column names), 'continuous' (list of column names).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('[bnlearn] >Expected a pandas DataFrame.')

    n_samples = max(len(df), 1)
    cardinality_threshold = max(10, int(0.05 * n_samples))

    continuous = []
    discrete = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            discrete.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            nuniq = int(series.nunique(dropna=True))
            if pd.api.types.is_integer_dtype(series) and nuniq <= cardinality_threshold:
                discrete.append(col)
            else:
                continuous.append(col)
        else:
            discrete.append(col)

    if len(continuous) == 0:
        dtype = 'discrete'
    elif len(discrete) == 0:
        dtype = 'continuous'
    else:
        dtype = 'mixed'

    return {
        'dtype': dtype,
        'discrete': discrete,
        'continuous': continuous,
    }


def default_scoretype(data_type):
    """Map detected data type to a default structure score."""
    if data_type == 'continuous':
        return 'bic-g'
    if data_type == 'mixed':
        return 'bic-cg'
    return 'bic'


def default_ci_test(data_type, ci_test):
    """Pick a CI test that matches the data type when the user left the default."""
    if ci_test != 'chi_square':
        return ci_test
    if data_type == 'continuous':
        return 'pearsonr'
    return ci_test


def edges_and_nodes_from_adjmat(adjmat):
    """Return edge list and node list from an adjacency matrix DataFrame.

    Parameters
    ----------
    adjmat : pandas.DataFrame
        Square adjacency matrix (rows/columns = node names).

    Returns
    -------
    edges : list of tuple
        (source, target) pairs where the matrix entry is non-zero.
    nodes : list
        Node names as strings.
    """
    nodes = list(adjmat.columns.astype(str))
    edges = []
    for source in adjmat.index.astype(str):
        for target in adjmat.columns.astype(str):
            val = adjmat.loc[source, target] if source in adjmat.index and target in adjmat.columns else 0
            try:
                if float(val) != 0:
                    edges.append((str(source), str(target)))
            except (TypeError, ValueError):
                pass
    return edges, nodes


# %% Model-type helpers
def model_kind(model):
    """Return 'discrete' | 'linear-gaussian' | 'cg' from a bnlearn result dict."""
    if not isinstance(model, dict):
        return 'discrete'
    if model.get('continuous_cpds'):
        return 'cg'
    m = model.get('model')
    if m is not None and 'LinearGaussian' in type(m).__name__:
        return 'linear-gaussian'
    cfg = model.get('config') or {}
    method = str(cfg.get('method', '')).lower()
    if method in ('linear-gaussian', 'lg'):
        return 'linear-gaussian'
    if method in ('cg', 'conditional-gaussian'):
        return 'cg'
    return 'discrete'