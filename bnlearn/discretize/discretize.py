from typing import Tuple, List

import pandas as pd

from .learn_discrete_bayes_net import discretize_all


def discretize(
    data: pd.DataFrame,
    edges: List[Tuple[str, str]],
    continuous_columns: List[str],
    max_iterations=8,
    verbose=3,
    ) -> pd.DataFrame:
    """
    Discretize the continuous columns in a pandas DataFrame based on a given graph.

    Parameters
    ----------
    data : pandas DataFrame
        The data to be discretized.
    edges : list of tuple of str
        A list of edges representing the graph.
    continuous_columns : list of str
        The names of the columns in the DataFrame that should be discretized.
    max_iterations : int, optional
        The maximum number of iterations to use when optimizing (default is 8).

    Returns
    -------
    pandas DataFrame
        The discretized DataFrame where every continuous column is converted
        into categories.
    """
    nodes = list(data.columns)
    graph = _bayes_net_graph(nodes, edges)
    continuous_index = [nodes.index(c) for c in continuous_columns]

    data_disc, continuous_edges = discretize_all(
        data,
        graph,
        continuous_index,
        max_iterations,
        verbose=verbose,
    )

    # Extend the breaks to the left with 1% to deal with (open,closed] intervals
    for i, col in enumerate(continuous_columns):
        breaks = continuous_edges[i]
        breaks[0] -= 0.01 * (breaks[-1] - breaks[0])
        interval_index = pd.IntervalIndex.from_breaks(breaks)
        dtype = pd.CategoricalDtype(interval_index, ordered=True)

        # TODO Let discretize_all return values starting from zero instead of from one
        data_disc[col] = pd.Categorical.from_codes(data_disc[col] - 1, dtype=dtype)

    return data_disc


def _bayes_net_graph(nodes: List[str], edges: List[Tuple[str, str]]):
    """
    >>> nodes = ['A', 'B', 'C', 'D']
    >>> edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')]
    >>> _bayes_net_graph(nodes, edges)
    [0, (0, 1), (0, 2), (2, 1, 3)]
    """
    sources_and_target = [[target] for target in range(len(nodes))]

    for source_node, target_node in edges:
        source = nodes.index(source_node)
        target = nodes.index(target_node)
        sources_and_target[target].insert(0, source)

    return [
        tuple(st for st in sts) if len(sts) > 1 else sts[0]
        for sts in sources_and_target
    ]


def discretize_value(dtype, value):
    """
    Discretize a numeric value by looking it up in the pandas IntervalIndex.

    Parameters
    ----------
    dtype : pandas.Series or pandas.CategoricalDtype
        The dtype of the pandas Series to use for discretization.
        If a Series is passed, its dtype will be used.
    value : numeric
        The value to discretize.

    Returns
    -------
    pandas.Interval
        The interval in the IntervalIndex that contains the value.

    Raises
    ------
    AttributeError
        If the dtype is not a pandas IntervalIndex CategoricalDtype.
    """
    if isinstance(dtype, pd.Series):
        dtype = dtype.dtype

    if dtype != "category" or not isinstance(dtype.categories, pd.IntervalIndex):
        raise AttributeError("Only IntervalIndex categories supported")

    return dtype.categories[dtype.categories.get_loc(value)]
