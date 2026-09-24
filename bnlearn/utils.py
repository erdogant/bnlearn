"""Shared utilities for bnlearn structure and parameter learning."""
# ------------------------------------
# Name        : utils.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------

import logging
import numpy as np
import pandas as pd
from ismember import ismember

logger = logging.getLogger('bnlearn')

# Optional TRACE level (also registered in bnlearn.__init__)
if not hasattr(logging, 'TRACE'):
    logging.TRACE = 5
    logging.addLevelName(logging.TRACE, 'TRACE')


def set_logger(verbose='info'):
    """Set the bnlearn logger level.

    Parameters
    ----------
    verbose : str or int or None
        * None, 0, 'silent', 'off' -> nothing
        * 'error' / 40
        * 'warning' / 30
        * 'info' / 20  (default)
        * 'debug' / 10
        * 'trace' / 5
        Legacy ints 1..5 (error..trace) are still accepted.
    """
    if verbose is None or verbose in (0, 'silent', 'off', 'no', 'None'):
        level = logging.CRITICAL + 1
    elif isinstance(verbose, str):
        name = verbose.strip().lower()
        mapping = {
            'critical': logging.CRITICAL,
            'error': logging.ERROR,
            'warning': logging.WARNING,
            'warn': logging.WARNING,
            'info': logging.INFO,
            'debug': logging.DEBUG,
            'trace': getattr(logging, 'TRACE', 5),
        }
        if name not in mapping:
            raise ValueError(f'[bnlearn] Unknown log level: {verbose}')
        level = mapping[name]
    elif isinstance(verbose, int):
        if verbose >= 10:
            level = verbose
        else:
            legacy = {
                1: logging.ERROR,
                2: logging.WARNING,
                3: logging.INFO,
                4: logging.DEBUG,
                5: getattr(logging, 'TRACE', 5),
                6: logging.CRITICAL + 1,
            }
            level = legacy.get(verbose, logging.INFO)
    else:
        raise TypeError(f'[bnlearn] verbose must be str, int, or None; got {type(verbose)}')
    logger.setLevel(level)
    return level


def get_logger():
    """Return the bnlearn package logger."""
    return logger

# %%  Convert vector into sparse dataframe
def vec2df(source, target, weights=None):
    """Convert source-target edges into sparse dataframe.

    vec2df overview
    ----------------
    Convert edges between source and target into a dataframe based on the weight.
    A weight of 2 will result that a row with the edge is created 2x.

    Parameters
    ----------
    source : array-like
        The source node.
    target : array-like
        The target node.
    weights : array-like of int
        The weights between the source-target values.

    Returns
    -------
    pd.DataFrame

    Examples
    --------
    >>> import bnlearn as bn
    >>> source = ['Cloudy', 'Cloudy', 'Sprinkler', 'Rain']
    >>> target = ['Sprinkler', 'Rain', 'Wet_Grass', 'Wet_Grass']
    >>> weights = [1, 2, 1, 3]
    >>> df = bn.vec2df(source, target, weights=weights)

    >>> import bnlearn as bn
    >>> vec = bn.import_example("stormofswords")
    >>> df = bn.vec2df(vec['source'], vec['target'], weights=vec['weight'])

    """
    if isinstance(source, (pd.DataFrame, pd.Series)):
        source = source.values
    if isinstance(target, (pd.DataFrame, pd.Series)):
        target = target.values
    if isinstance(weights, (pd.DataFrame, pd.Series)):
        weights = weights.values

    rows = []
    edges = list(zip(source, target))
    if weights is None:
        weights = np.ones_like(source).astype(int)

    columns = np.unique(np.c_[source, target].ravel())
    for i, edge in enumerate(edges):
        row = [np.logical_or(columns == edge[0], columns == edge[1])] * int(weights[i])
        rows = rows + row

    return pd.DataFrame(np.array(rows), columns=columns)


# %%  Convert source/target vectors into adjacency matrix
def vec2adjmat(source, target, weights=None, symmetric: bool = True, aggfunc='sum') -> pd.DataFrame:
    """Convert source and target into adjacency matrix.

    Parameters
    ----------
    source : list
        The source node.
    target : list
        The target node.
    weights : list of int
        The weights between the source-target values.
    symmetric : bool, optional
        Make the adjacency matrix symmetric with the same number of rows as
        columns. The default is True.
    aggfunc : str, optional
        Aggregate function in case multiple values exist for the same
        relationship. 'sum' (default).

    Returns
    -------
    pd.DataFrame
        Adjacency matrix.

    Examples
    --------
    >>> source = ['Cloudy', 'Cloudy', 'Sprinkler', 'Rain']
    >>> target = ['Sprinkler', 'Rain', 'Wet_Grass', 'Wet_Grass']
    >>> vec2adjmat(source, target)

    >>> weights = [1, 2, 1, 3]
    >>> vec2adjmat(source, target, weights=weights)

    """
    if len(source) != len(target):
        raise ValueError('[bnlearn] >Source and Target should have equal elements.')
    if weights is None:
        weights = [1] * len(source)
    logger.info('Converting source-target into adjacency matrix..')
    df = pd.DataFrame(np.c_[source, target], columns=['source', 'target'])
    adjmat = pd.crosstab(df['source'], df['target'], values=weights, aggfunc=aggfunc).fillna(0)
    nodes = np.unique(list(adjmat.columns.values) + list(adjmat.index.values))

    if symmetric:
        logger.info('Making the matrix symmetric..')
        IA, _ = ismember(nodes, adjmat.columns.values)
        node_columns = nodes[~IA]
        if len(node_columns) > 0:
            df_new_columns = pd.DataFrame(0, index=adjmat.index, columns=node_columns)
            adjmat = pd.concat([adjmat, df_new_columns], axis=1)

        IA, _ = ismember(nodes, adjmat.index.values)
        node_rows = nodes[~IA]
        if len(node_rows) > 0:
            df_new_rows = pd.DataFrame(0, index=node_rows, columns=adjmat.columns)
            adjmat = pd.concat([adjmat, df_new_rows], axis=0)

        logger.debug('Order columns and rows.')
        _, IB = ismember(adjmat.columns.values, adjmat.index.values)
        adjmat = adjmat.iloc[IB, :]
        adjmat.index.name = 'source'
        adjmat.columns.name = 'target'

    adjmat.columns = adjmat.columns.astype(str)
    return adjmat


# %%  Convert adjacency matrix into source/target vector
def adjmat2vec(adjmat, min_weight=1):
    """Convert adjacency matrix into vector with source and target.

    Parameters
    ----------
    adjmat : pd.DataFrame
        Adjacency matrix.
    min_weight : float
        Edges are returned with a minimum weight.

    Returns
    -------
    pd.DataFrame
        Nodes that are connected based on source and target.

    Examples
    --------
    >>> import bnlearn as bn
    >>> source = ['Cloudy', 'Cloudy', 'Sprinkler', 'Rain']
    >>> target = ['Sprinkler', 'Rain', 'Wet_Grass', 'Wet_Grass']
    >>> adjmat = vec2adjmat(source, target)
    >>> vector = bn.adjmat2vec(adjmat)

    """
    adjmat = adjmat.stack().reset_index()
    adjmat.columns = ['source', 'target', 'weight']
    Iloc1 = adjmat['source'] != adjmat['target']
    Iloc2 = adjmat['weight'] >= min_weight
    Iloc = Iloc1 & Iloc2
    adjmat = adjmat.loc[Iloc, :]
    adjmat.reset_index(drop=True, inplace=True)
    return adjmat


# %%  Convert adjacency matrix to dict
def adjmat2dict(adjmat):
    """Convert adjacency matrix to dict.

    Parameters
    ----------
    adjmat : pd.DataFrame
        Adjacency matrix.

    Returns
    -------
    dict
        Graph as adjacency dict.

    """
    adjmat = adjmat.astype(bool)
    graph = {}
    rows = adjmat.index.values
    for r in rows:
        graph.update({r: list(rows[adjmat.loc[r, :]])})
    return graph

# %%  Normalise weights into [minscale, maxscale]
def _normalize_weights(weights, minscale=1, maxscale=5):
    """Scale *weights* linearly into [minscale, maxscale].

    Parameters
    ----------
    weights : np.ndarray
        1-D array of raw weights.
    minscale : float
        Lower bound of the output range. Default 1.
    maxscale : float
        Upper bound of the output range. Default 5.

    Returns
    -------
    np.ndarray
        Scaled weights.

    """
    from sklearn.preprocessing import MinMaxScaler

    if len(weights.shape) == 1:
        weights = weights.reshape(-1, 1)
    weights = MinMaxScaler(feature_range=(minscale, maxscale)).fit_transform(weights).flatten()
    return weights


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
        (source, target) pairs where the matrix entry is non-zero / True.
    nodes : list
        Node names as strings.
    """
    nodes = list(adjmat.columns.astype(str))
    edges = []
    for src, row in adjmat.iterrows():
        for tgt, val in row.items():
            try:
                if float(val) != 0:
                    edges.append((str(src), str(tgt)))
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