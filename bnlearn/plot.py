"""Plot module."""
# ------------------------------------
# Name        : plot.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------

# %% Libraries
import logging
logger = logging.getLogger("bnlearn")
import os
import copy
import re
import json
import html as _html
import itertools
import webbrowser
from packaging import version

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from ismember import ismember
from setgraphviz import setgraphviz

import bnlearn as bn
try:
    from bnlearn.utils import vec2adjmat, adjmat2vec, _normalize_weights
except:
    from utils import vec2adjmat, adjmat2vec, _normalize_weights

# %% Get node properties
def get_node_properties(model, node_color='#ADD8E6', node_size=None):
    """Collect node properties.

    Parameters
    ----------
    model : dict
        dict containing (initialized) model.
    node_color : str, (Default: '#000000')
        The default color of the edges.
    node_size : float, (Default: 1)
        The default weight of the edges.
    Print progress to screen. The default is 3.
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

    Returns
    -------
    dict.
        Node properties.

    Examples
    --------
    >>> import bnlearn as bn
    >>> edges = [('A', 'B'), ('A', 'C'), ('A', 'D')]
    >>> # Create DAG and store in model
    >>> model = bn.make_DAG(edges)
    >>> node_properties = bn.get_node_properties(model)
    >>> # Adjust the properties
    >>> node_properties['A']['node_size']=100
    >>> node_properties['A']['node_color']='#000000'
    >>> # Make plot
    >>> fig = bn.plot(model, interactive=False, node_properties=node_properties)
    >>>
    >>> # Example: Specify all nodes
    >>> node_properties = bn.get_node_properties(model, node_size=10, node_color='#000000')
    >>> bn.plot(model, interactive=True, node_properties=node_properties)

    """
    # https://networkx.org/documentation/networkx-1.7/reference/generated/networkx.drawing.nx_pylab.draw_networkx_nodes.html
    nodes = {}
    defaults = {'node_color': node_color, 'node_size': node_size}
    adjmat = model.get('adjmat', None)

    if adjmat is not None:
        logger.info('Set node properties.')
        # For each node, use the default node properties.
        for node in adjmat.columns:
            node_property = defaults.copy()
            nodes.update({node: node_property})

    # Return dict with node properties
    return nodes


# %% Get edge properties
def get_edge_properties(model, color='#000000', weight=1, minscale=1, maxscale=5):
    """Collect edge properties.

    Parameters
    ----------
    model : dict
        dict containing (initialized) model.
    color : str, (Default: '#000000')
        The default color of the edges.
    weight : float, (Default: 1)
        The default weight of the edges.
    minscale : float, (Default: 1)
        The minimum weight of the edge in case of test statistics are used.
    maxscale : float, (Default: 10)
        The maximum weight of the edge in case of test statistics are used.

    Returns
    -------
    dict
        Mapping ``(source, target) -> properties`` with keys:

        * ``color`` – edge color
        * ``weight`` – display thickness (scaled ``-log10(p_value)`` when
          ``independence_test`` is present, else the default weight)
        * ``p_value`` – raw p-value from ``independence_test`` (1.0 if absent)
        * ``logp`` – ``-log10(p_value)`` (0.0 if absent)
        * ``value`` – structure / coefficient entry from ``model['adjmat']``

    Examples
    --------
    >>> # Example 1:
    >>> import bnlearn as bn
    >>> edges = [('A', 'B'), ('A', 'C'), ('A', 'D')]
    >>>
    >>> # Create DAG and store in model
    >>> model = bn.make_DAG(edges)
    >>>
    >>> # Adjust the properties
    >>> edge_properties = bn.get_edge_properties(model)
    >>> edge_properties[('A', 'B')]['weight']=10
    >>> edge_properties[('A', 'B')]['color']='#8A0707'
    >>>
    >>> # Make plot
    >>> fig = bn.plot(model, interactive=False, edge_properties=edge_properties)

    >>> # Example 2:
    >>> # Load asia DAG
    >>> df = bn.import_example(data='asia')
    >>>
    >>> # Structure learning of sampled dataset
    >>> model = bn.structure_learning.fit(df)
    >>>
    >>> # Compute edge weights based on chi_square test statistic
    >>> model = bn.independence_test(model, df, test='chi_square')
    >>>
    >>> # Get the edge properties
    >>> edge_properties = bn.get_edge_properties(model)
    >>> # Make adjustments
    >>> edge_properties[('tub', 'either')]['color']='#8A0707'
    >>>
    >>> # Make plot
    >>> fig = bn.plot(model, interactive=True, edge_properties=edge_properties)

    """

    edges = {}
    defaults = {'color': color, 'weight': weight, 'p_value': 1.0, 'logp': 0.0, 'value': 1}
    indep = model.get('independence_test', None)
    adjmat_weight = None
    adjmat_p = None
    adjmat_logp = None

    if indep is not None:
        if 'p_value' not in indep.columns:
            raise KeyError("[bnlearn] >independence_test must contain column 'p_value'.")
        stat_cols = [c for c in indep.columns if c not in ('source', 'target', 'stat_test', 'p_value', 'dof')]
        stat_name = stat_cols[0] if stat_cols else 'independence_test'
        logger.info('Set edge weights based on the [%s] test statistic.' % (stat_name))
        raw_p = indep['p_value']
        logp = compute_logp(raw_p)
        weights = _normalize_weights(logp.values, minscale=minscale, maxscale=maxscale)
        if model.get('config', {}).get('method') and 'lingam' in str(model['config'].get('method', '')):
            adjmat_weight = (model['adjmat'].abs() > 0).astype(float)
        else:
            adjmat_weight = vec2adjmat(indep['source'], indep['target'], weights=weights)
        adjmat_p = vec2adjmat(indep['source'], indep['target'], weights=raw_p)
        adjmat_logp = vec2adjmat(indep['source'], indep['target'], weights=logp)
    else:
        adjmat_weight = model.get('adjmat', None)
        if adjmat_weight is not None:
            adjmat_weight = (adjmat_weight.abs() > 0).astype(float)

    if adjmat_weight is None:
        return edges
    model_edges = adjmat2vec(adjmat_weight)[['source', 'target']].values

    logger.info('Set edge properties.')
    for u, v in model_edges:
        edge_property = defaults.copy()
        if not isinstance(adjmat_weight.loc[u, v], np.bool_):
            edge_property['weight'] = float(adjmat_weight.loc[u, v])
            if model.get('adjmat', None) is not None and u in model['adjmat'].index and v in model['adjmat'].columns:
                edge_property['value'] = model['adjmat'].loc[u, v]
            if adjmat_p is not None and u in adjmat_p.index and v in adjmat_p.columns:
                edge_property['p_value'] = float(adjmat_p.loc[u, v])
            if adjmat_logp is not None and u in adjmat_logp.index and v in adjmat_logp.columns:
                edge_property['logp'] = float(adjmat_logp.loc[u, v])
        edges.update({(u, v): edge_property})

    return edges


# %% Internal helpers
# def compute_logp(p_value):
#     logp = -np.log10(p_value)
#     if logp.empty:
#         return logp

#     Iloc = np.isinf(logp)
#     max_logp = np.max(logp[~Iloc]) * 1.5
#     if np.isnan(max_logp): max_logp = 1
#     logp.loc[Iloc] = max_logp
#     return logp

def compute_logp(p_value):
    logp = -np.log10(p_value)

    if logp.empty:
        return logp

    Iloc = np.isinf(logp)
    valid = logp[~Iloc & np.isfinite(logp)]
    max_logp = valid.max() * 1.5 if not valid.empty else 1
    if not np.isfinite(max_logp): max_logp = 1
    logp.loc[Iloc] = max_logp
    return logp


def normalize_independence_frame(df_indep, test_name='stat'):
    """Ensure independence_test has a consistent column set.

    Columns: source, target, stat_test, p_value, <test_name>, dof

    Empty frames (no edges in the DAG) still get the full schema so downstream
    code can rely on column names without KeyError.
    """
    if df_indep is None or len(df_indep) == 0:
        return pd.DataFrame(columns=['source', 'target', 'stat_test', 'p_value', test_name, 'dof'])

    required = ['source', 'target', 'p_value']
    for col in required:
        if col not in df_indep.columns:
            raise KeyError("[bnlearn] >independence_test missing required column '%s'." % col)
    out = df_indep.copy()
    if 'stat_test' not in out.columns:
        out['stat_test'] = out['p_value'] <= 0.05
    if 'dof' not in out.columns:
        out['dof'] = 1
    if test_name not in out.columns:
        out[test_name] = np.nan
    front = ['source', 'target', 'stat_test', 'p_value']
    rest = [c for c in out.columns if c not in front + ['dof']]
    ordered = front + rest + (['dof'] if 'dof' in out.columns else [])
    return out.loc[:, ordered]


# %% Hierarchical (top-down) layout with crossing minimisation
def hierarchical_layout(G, scale=1, n_passes=4):
    """Compute a top-down hierarchical layout with crossing minimisation.

    Nodes are placed in topological generations (layer 0 at the top, the
    deepest layer at the bottom), replicating the look of the graphviz ``dot``
    layout without requiring graphviz to be installed.

    Within-layer ordering is optimised by an iterative barycenter sweep
    (Sugiyama et al., 1981): each layer is repeatedly reordered so that every
    node sits at the mean x-position of its already-placed neighbours.
    Forward sweeps (top-down, driven by predecessors) and backward sweeps
    (bottom-up, driven by successors) are alternated, and the best ordering
    found across all sweeps is kept.  This reduces — and often eliminates —
    edge crossings without any external dependencies.

    Parameters
    ----------
    G : nx.DiGraph
        The graph to lay out.  Every node in G receives a position, including
        isolated nodes (no in- or out-edges).
    scale : float
        Linear scale factor applied to all coordinates.
    n_passes : int
        Number of forward+backward sweep pairs.  4 is enough for all typical
        Bayesian networks; increase for very large graphs.

    Returns
    -------
    dict
        Mapping ``node -> np.ndarray([x, y])`` ready to pass as ``pos`` to
        NetworkX drawing functions.
    """
    def _count_crossings(l1, l2):
        idx1 = {n: i for i, n in enumerate(l1)}
        idx2 = {n: i for i, n in enumerate(l2)}
        pairs = [(idx1[u], idx2[v])
                 for u in l1
                 for v in G.successors(u)
                 if v in idx2]
        return sum(1
                   for k in range(len(pairs))
                   for j in range(k)
                   if (pairs[j][0] - pairs[k][0]) * (pairs[j][1] - pairs[k][1]) < 0)

    def _total_crossings(ordered):
        return sum(_count_crossings(ordered[i], ordered[i + 1])
                   for i in range(len(ordered) - 1))

    def _barycenter_sort(layer, pos_lookup, use_preds):
        scored = []
        for i, node in enumerate(layer):
            nbrs = (list(G.predecessors(node)) if use_preds
                    else list(G.successors(node)))
            connected = [n for n in nbrs if n in pos_lookup]
            score = (np.mean([pos_lookup[n] for n in connected])
                     if connected else None)
            scored.append((score, i, node))
        real = [s for s, _, _ in scored if s is not None]
        fallback = np.mean(real) if real else 0.0
        scored = [(fallback if s is None else s, i, n) for s, i, n in scored]
        return [n for _, _, n in sorted(scored, key=lambda t: (t[0], t[1]))]

    generations = list(nx.topological_generations(G))
    n_layers = len(generations)

    ordered = [sorted(generations[0])]
    all_pos = {n: i for i, n in enumerate(ordered[0])}
    for layer in generations[1:]:
        layer_sorted = _barycenter_sort(list(layer), all_pos, use_preds=True)
        ordered.append(layer_sorted)
        all_pos.update({n: i for i, n in enumerate(layer_sorted)})

    best = [l[:] for l in ordered]
    best_c = _total_crossings(ordered)

    for _ in range(n_passes):
        # Forward pass
        all_pos = {n: i for i, n in enumerate(ordered[0])}
        for li in range(1, len(ordered)):
            ordered[li] = _barycenter_sort(ordered[li], all_pos, use_preds=True)
            all_pos.update({n: i for i, n in enumerate(ordered[li])})
        c = _total_crossings(ordered)
        if c < best_c:
            best_c = c
            best = [l[:] for l in ordered]

        # Backward pass
        all_pos = {n: i for i, n in enumerate(ordered[-1])}
        for li in range(len(ordered) - 2, -1, -1):
            ordered[li] = _barycenter_sort(ordered[li], all_pos, use_preds=False)
            all_pos.update({n: i for i, n in enumerate(ordered[li])})
        c = _total_crossings(ordered)
        if c < best_c:
            best_c = c
            best = [l[:] for l in ordered]

    pos = {}
    for layer_idx, layer in enumerate(best):
        n = len(layer)
        y = ((1.0 - 2.0 * layer_idx / (n_layers - 1)) * scale
             if n_layers > 1 else 0.0)
        for i, node in enumerate(layer):
            x = ((-1.0 + 2.0 * i / (n - 1)) * scale if n > 1 else 0.0)
            pos[node] = np.array([x, y])

    return pos


# %% plot_graphviz
def plot_graphviz(model,
                  edge_filter='weight',
                  params={'prediction_feature_indices': None,
                          'prediction_target_label': "Y(pred)",
                          'prediction_line_color': "red",
                          'prediction_coefs': None,
                          'prediction_feature_importance': None,
                          'path': None,
                          'path_color': None,
                          'detect_cycle': False,
                          'ignore_shape': False},
                  verify_certificate=True):
    """Plot a causal or Bayesian network using Graphviz based on an adjacency matrix.

    This function visualizes the causal or Bayesian model structure in
    model['adjmat'] using the Graphviz library.

    Parameters
    ----------
    model : dict
        A dictionary containing the network model.
    edge_filter : str or None
        None: Do not show numeric edge labels.
        'weight': Show the adjacency / coefficient values.
        'p_value': Show raw edge p-values (requires independence_test).
        'logp': Show -log10(p_value) strength (requires independence_test).
    params : dict, optional
        Visualization parameters (see docstring of bnlearn.plot_graphviz).
    verify_certificate : bool (default: True)

    Returns
    -------
    dot_graph : graphviz.Source or None
    """
    from lingam.utils import make_dot

    dot_graph = None
    if model['adjmat'].sum().sum() == 0:
        logger.info('Nothing to plot because no edges are present between nodes. ')
        return None
    if model.get('config', {}).get('method') == 'DBN':
        logger.info('DynamicBayesianNetwork (DBN) can not be plot with Graphviz.')
        return None

    GraphvizPath = setgraphviz(verify_certificate=verify_certificate)
    if GraphvizPath is None:
        logger.error('Graphviz is not found in path and can therefore cause an error in producing the dot image.')
    defaults = {'prediction_feature_indices': None, 'prediction_target_label': "Y(pred)",
                'prediction_line_color': "red", 'prediction_coefs': None,
                'prediction_feature_importance': None, 'path': None, 'path_color': None,
                'detect_cycle': False, 'ignore_shape': False}
    params = {**defaults, **params}

    model = copy.deepcopy(model)

    logger.info(f'Setting edge mode to {edge_filter}.')
    indep = model.get('independence_test')
    if indep is not None and edge_filter in ('logp', 'p_value') and 'stat_test' in indep.columns:
        Iloc = indep['stat_test'].astype(bool)
        source = indep['source'].loc[Iloc]
        target = indep['target'].loc[Iloc]
        logger.info(f'Number of significant edges detected: {int(Iloc.sum())}')
        if edge_filter == 'logp':
            logp = compute_logp(indep['p_value'])
            adjmat = vec2adjmat(source, target, weights=logp.loc[Iloc], symmetric=True, aggfunc='sum')
        else:
            adjmat = vec2adjmat(source, target, weights=indep['p_value'].loc[Iloc], symmetric=True, aggfunc='sum')
    else:
        edges = sum(1 for item in (model.get('model_edges') or []) if isinstance(item, tuple))
        logger.info(f'Number of edges detected: {edges}')
        adjmat = model['adjmat'].copy()

    node_labels = list(adjmat.T.columns) if edge_filter is not None else None

    if node_labels is not None and len(node_labels) > 0:
        dot_graph = make_dot(adjmat.T.values.astype(float), labels=node_labels, lower_limit=0, **params)

    return dot_graph


# %% plot
def plot(model,
         edge_filter='weight',
         pos=None,
         scale=1,
         interactive=False,
         title='bnlearn Directed Acyclic Graph (DAG)',
         node_color=None,
         node_size=None,
         node_properties=None,
         edge_properties=None,
         params_interactive={
             'type': 'flowgraph',       # choose between 'd3graph' or 'flowgraph'
             'figsize': [None, None],
             'notebook': False,
             'filepath': None,
             'direction': 'LR',          # 'LR' | 'TB' | 'RL' | 'BT'
             'layout': 'dagre',          # 'dagre' | 'elk'
             'animate': False,
             'show_side_panel': True,
             'show_top_panel': True,
             'dark_mode': True,
             'bgcolor': None,
             # d3graph
             'minmax_distance': [50, 100],
             'font_color': None,
             'show_slider': True},
         params_static={'minscale': 1, 'maxscale': 5, 'figsize': (10, 10), 'width': None, 'height': None, 'font_size': 10, 'font_family': 'sans-serif', 'alpha': 0.8, 'node_shape': 'o', 'layout': 'graphviz_layout_custom', 'font_color': '#000000', 'facecolor': 'white', 'edge_alpha': 0.8, 'arrowstyle': '-|>', 'arrowsize': 20, 'visible': True, 'showplot': True, 'dpi': 200},
         ):
    """
    Plot the learned structure.

    Parameters
    ----------
    model : dict
        Learned model from the .fit() function.
    pos : graph, optional
        Coordinates of the network. If provided, the same structure will be used to plot the network. The default is None.
    scale : int, optional
        Scaling parameter for the network. A larger number will linearly increase the network. The default is 1.
    interactive : bool, optional
        True: Interactive web-based graph.
        False: Static plot.
    title : str, optional
        Title for the plots.
    node_color : str, optional
        Color each node in the network using a hex-color, such as '#8A0707'.
    node_size : int, optional
        Set the node size for each node in the network.
    node_properties : dict, optional
        Dictionary containing custom node_color and node_size parameters for the network.
    edge_properties : dict, optional
        Dictionary containing custom edge_color and edge_size parameters for the network.
    edge_filter : str or None, optional
        None: Do not annotate edges with numbers.
        'weight': Show the structure / coefficient values.
        'p_value': Show raw edge p-values (requires independence_test).
        'logp': Show -log10(p_value) strength (requires independence_test).
    params_interactive : dict, optional
        Dictionary containing various settings for interactive plots.
    params_static : dict, optional
        Dictionary containing various settings for static plots.
        layout: 'graphviz_layout_custom', 'graphviz_layout', 'spring_layout',
        'planar_layout', 'shell_layout', 'spectral_layout', 'pydot_layout',
        'circular_layout', 'random_layout', 'bipartite_layout', 'multipartite_layout'.

    Returns
    -------
    dict
        pos, G, node_properties, edge_properties, fig/ax.
    """
    fig = None
    if model is None or (model.get('adjmat', None) is None) or model['adjmat'].sum().sum() == 0:
        logger.info('Nothing to plot because no edges are present between nodes. ')
        return None

    if model.get('config', {}).get('method') == 'DBN' and interactive:
        logger.info('DynamicBayesianNetwork (DBN) can not be plot with Graphviz.')
        return None

    if model.get('independence_test', None) is None and edge_filter in ('p_value', 'logp'):
        logger.warning('edge_filter p_value/logp require: model=bn.independence_test(model, df)')
        edge_filter = None

    model = copy.deepcopy(model)
    model['adjmat'] = model['adjmat'].astype(float)

    defaults = {
        'type': 'flowgraph',        # 'flowgraph' or 'd3graph'
        'direction': 'TB',          # 'LR' | 'TB' | 'RL' | 'BT'
        'layout': 'dagre',          # 'dagre' | 'elk'
        'animate': False,
        # generic
        'figsize': [None, None],
        'show_side_panel': True,
        'show_top_panel': True,
        'dark_mode': True,
        'filepath': None,
        'bgcolor': None,            # maps to color_background if you want
        'notebook': False,
        # d3graph
        'minmax_distance': [50, 100],
        'font_color': None,
        'show_slider': True,
    }
    params_interactive = {**defaults, **params_interactive}

    defaults = {'minscale': 1, 'maxscale': 5, 'figsize': (15, 10), 'height': None, 'width': None,
                'font_size': 14, 'font_family': 'sans-serif', 'alpha': 0.8, 'layout': 'graphviz_layout_custom',
                'font_color': 'k', 'facecolor': '#ffffff', 'node_shape': 'o', 'edge_alpha': 0.8,
                'arrowstyle': '-|>', 'arrowsize': 20, 'visible': True, 'showplot': True, 'dpi': 200}
    params_static = {**defaults, **params_static}

    if (params_static.get('width') is not None) or (params_static.get('height') is not None):
        params_static['figsize'] = (
            15 if params_static['width'] is None else params_static['width'],
            10 if params_static['height'] is None else params_static['height'],
        )

    out = {}
    G = nx.DiGraph()
    if model.get('adjmat', None) is not None:
        G.add_nodes_from(model['adjmat'].columns.values)

    node_size_default = 10 if interactive else 800
    if (node_properties is not None) and (node_size is not None):
        logger.warning('Warning: if both "node_size" and "node_properties" are used, "node_size" will be used.')
    if node_properties is None:
        node_properties = get_node_properties(model, node_size=node_size_default)
    if edge_properties is None:
        edge_properties = get_edge_properties(model, minscale=params_static['minscale'], maxscale=params_static['maxscale'])

    if edge_filter in ('p_value', 'logp') and model.get('independence_test') is not None:
        indep = model['independence_test']
        if 'stat_test' in indep.columns and len(indep) > 0:
            sig = indep.loc[indep['stat_test'].astype(bool), ['source', 'target']]
            sig_set = set(zip(sig['source'], sig['target']))
            n_before = len(edge_properties)
            edge_properties = {e: p for e, p in edge_properties.items() if e in sig_set}
            logger.info(f'Number of significant edges detected: {len(edge_properties)} (of {n_before})')
    for key in node_properties.keys():
        if node_properties[key]['node_size'] is None:
            node_properties[key]['node_size'] = node_size_default

    for edge, properties in edge_properties.items():
        G.add_edge(
            *edge,
            weight=properties.get('weight', 0),
            p_value=properties.get('p_value', 1.0),
            logp=properties.get('logp', 0.0),
            value=properties.get('value', 0),
        )

    if 'dict' in str(type(model)):
        bnmodel = model.get('model', None)
    else:
        bnmodel = copy.deepcopy(model)

    nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights, edge_p_value, edge_value =  _plot_properties(G, node_properties, edge_properties, node_color, node_size)
    tooltip = nodelist
    
    # print(model["model"], 'get_cpds')
    
    if interactive:
        if hasattr(model["model"], 'get_cpds'):
            tooltip = []
            # Collect model nodes if available to avoid ValueError when node is missing
            try:
                model_nodes = set(model["model"].nodes())
            except Exception:
                model_nodes = None
            for node in nodelist:
                # Only call get_cpds if the node is present in the underlying model
                if model_nodes is not None and node not in model_nodes:
                    tip = node
                else:
                    try:
                        tip = model["model"].get_cpds(node)
                        if tip is None:
                            tip = node
                    except Exception:
                        # Fall back to the node label if anything goes wrong
                        tip = node
                tooltip.append(tip)

        # Make d3block
        if params_interactive['type']=='d3graph':
            fig = _plot_interactive_d3graph(params_interactive, nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights, title, tooltip)
        elif params_interactive['type']=='flowgraph':
            # Neat CPD (of the target node) shown when hovering over an edge
            edge_tooltip = _edge_cpd_tooltips(model, edgelist)
            # Same neat CPD table for the nodes (instead of the ASCII table of str(cpd))
            node_tooltip = _node_cpd_tooltips(model, nodelist)
            if node_tooltip:
                tooltip = [node_tooltip.get(n, None) for n in nodelist]
            fig = _plot_interactive_flowgraph(params_interactive, nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights, title, tooltip, edge_tooltip=edge_tooltip)
        else:
            logger.warning(f"D3Block for {params_interactive['type']} not implemented in bnlearn.")

    else:
        if ('bayes' in str(type(bnmodel)).lower()) or ('pgmpy' in str(type(bnmodel)).lower()) or ('lingam' in model['config']['method']):
            logger.info('Plot based on Bayesian model')
            if params_static['layout'] == 'graphviz_layout_custom' and pos is None:
                logger.info('Using hierarchical top-down layout (graphviz_layout_custom).')
                pos = hierarchical_layout(G, scale=scale)
            pos = bn.network.graphlayout(G, pos=pos, scale=scale, layout=params_static['layout'])
        elif 'networkx' in str(type(bnmodel)):
            logger.info('Plot based on networkx model')
            G = bnmodel
            if params_static['layout'] == 'graphviz_layout_custom' and pos is None:
                logger.info('Using hierarchical top-down layout (graphviz_layout_custom).')
                pos = hierarchical_layout(G, scale=scale)
            pos = bn.network.graphlayout(G, pos=pos, scale=scale, layout=params_static['layout'])
        else:
            logger.info('Plot based on adjacency matrix')
            G = bn.network.adjmat2graph(model['adjmat'].abs() > 0)
            if params_static['layout'] == 'graphviz_layout_custom' and pos is None:
                logger.info('Using hierarchical top-down layout (graphviz_layout_custom).')
                pos = hierarchical_layout(G, scale=scale)
            pos = bn.network.graphlayout(G, pos=pos, scale=scale, layout=params_static['layout'])

        fig = _plot_static(model, params_static, nodelist, node_colors, node_sizes, G, pos,
                           edge_colors, edge_weights, showplot=params_static['showplot'],
                           visible=params_static['visible'], title=title, dpi=params_static['dpi'],
                           edge_filter=edge_filter)

    out['fig'] = fig
    out['ax'] = fig
    out['pos'] = pos
    out['G'] = G
    out['node_properties'] = node_properties
    out['edge_properties'] = edge_properties
    return out


# %% Static plot
def _plot_static(model, params_static, nodelist, node_colors, node_sizes, G, pos, edge_colors, edge_weights, title, visible=True, showplot=True, dpi=100, edge_filter='weight'):

    # Create figure
    fig = plt.figure(figsize=params_static['figsize'], facecolor=params_static['facecolor'], dpi=dpi)
    fig.set_visible(visible)

    # Make networkx
    nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_size=node_sizes, alpha=params_static['alpha'], node_color=node_colors, node_shape=params_static['node_shape'])
    nx.draw_networkx_edges(G, pos, arrowstyle=params_static['arrowstyle'], arrowsize=params_static['arrowsize'], edge_color=edge_colors, width=edge_weights, alpha=params_static['edge_alpha'])

    if edge_filter == 'weight':
        edge_label = nx.get_edge_attributes(G, 'value')
        edge_label = {key: float(f'{value:.2f}'[:4]) for key, value in edge_label.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label)
    elif edge_filter == 'p_value':
        edge_label = nx.get_edge_attributes(G, 'p_value')
        edge_label = {key: float(f'{value:.2g}') for key, value in edge_label.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label)
    elif edge_filter == 'logp':
        edge_label = nx.get_edge_attributes(G, 'logp')
        edge_label = {key: float(f'{value:.2f}'[:4]) for key, value in edge_label.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label)

    # Draw final figure
    nx.draw_networkx_labels(G, pos, font_size=params_static['font_size'], font_family=params_static['font_family'], font_color=params_static['font_color'])

    # Making figure nice
    # fig = plt.gca()
    # fig.set_axis_off()
    plt.title(title)
    if showplot:
        plt.show()
    return fig

# %% Interactive flowgraph
def _plot_interactive_flowgraph(params_interactive, nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights, title, tooltip, edge_tooltip=None):
    """Create an interactive flowgraph (layered process diagram) with d3blocks.

    ``edge_tooltip`` is an optional dict ``{(source, target): html}``. d3blocks only
    supports plain-text edge hovers, so the HTML is injected into the generated file.
    """
    try:
        from d3blocks import D3Blocks
        import d3blocks as d3
    except ModuleNotFoundError:
        raise Exception(
            '[bnlearn] >"d3blocks" library is not installed. '
            'Pip install first: "pip install d3blocks"'
        )

    if version.parse(d3.__version__) < version.parse("2.0.0"):
        raise ImportError(
            '[bnlearn] >Error: d3blocks version should be >= 2.0.0. '
            'Pip install: "pip install -U d3blocks"'
        )

    defaults = {
        'figsize': [None, None],
        'notebook': False,
        'filepath': None,
        'direction': 'TB',
        'layout': 'dagre',
        'animate': False,
        'show_side_panel': True,
        'show_top_panel': True,
        'dark_mode': True,
        'bgcolor': None,
    }
    params_interactive = {**defaults, **params_interactive}

    if params_interactive['filepath'] is None:
        params_interactive['filepath'] = title.strip().replace(' ', '_') + '.html'

    nodelist = [
        x.encode('ascii', 'ignore').decode('utf-8').replace(' ', '_')
        for x in list(nodelist)
    ]

    X = pd.DataFrame(data=edgelist, columns=['source', 'target'])
    X['source'] = X['source'].astype(str).str.replace(' ', '_')
    X['target'] = X['target'].astype(str).str.replace(' ', '_')
    X['weight'] = edge_weights

    nodes = {}
    for i, name in enumerate(nodelist):
        nodes[name] = {
            'color': node_colors[i] if node_colors is not None else None,
            'tooltip': tooltip[i] if tooltip is not None else None,
        }

    # Edge tooltips need a post-processed HTML file, so the browser/notebook is
    # opened by us after patching instead of by d3blocks.
    edge_html = {}
    if edge_tooltip:
        for (u, v), h in edge_tooltip.items():
            edge_html[str(u).replace(' ', '_') + '->' + str(v).replace(' ', '_')] = h
    node_has_html = any(isinstance(t, str) and 'bn-cpd' in t for t in (tooltip or []))
    patch = len(edge_html) > 0 or node_has_html
    show_notebook = params_interactive['notebook']

    d3b = D3Blocks()
    d3b.flowgraph(
        X,
        nodes=nodes,
        direction=params_interactive['direction'],
        layout=params_interactive['layout'],
        # edge_color=edge_color,
        animate=params_interactive['animate'],
        title=title,
        filepath=params_interactive['filepath'],
        figsize=params_interactive['figsize'],
        notebook=False if patch else show_notebook,
        showfig=False if patch else True,
        show_side_panel=params_interactive['show_side_panel'],
        show_top_panel=params_interactive['show_top_panel'],
        dark_mode=params_interactive['dark_mode'],
        color_background=params_interactive['bgcolor'],
    )

    filepath = os.path.abspath(d3b.config['filepath'])
    if patch:
        # If patching fails (d3blocks changed?), the plain graph is still shown.
        _inject_edge_tooltips(filepath, edge_html)
        _open_html(filepath, notebook=show_notebook, figsize=params_interactive['figsize'])
    return filepath


# %% Edge CPD tooltips
_EDGE_TOOLTIP_CSS = """
<style>
#fg-tooltip { max-width: 520px !important; }
#fg-tooltip table.bn-cpd { border-collapse: collapse; margin-top: 5px; font-size: 11px; }
#fg-tooltip table.bn-cpd th, #fg-tooltip table.bn-cpd td { padding: 2px 7px; text-align: right; border: 1px solid rgba(128,128,128,.35); white-space: nowrap; }
#fg-tooltip table.bn-cpd th { font-weight: 600; background: rgba(128,128,128,.18); }
#fg-tooltip table.bn-cpd th.s, #fg-tooltip table.bn-cpd th.e { text-align: left; }
#fg-tooltip table.bn-cpd th.hl, #fg-tooltip table.bn-cpd td.hl { outline: 1.5px solid #f0b429; outline-offset: -1.5px; }
#fg-tooltip .bn-cpd-title { font-family: Georgia, serif; font-style: italic; margin-top: 4px; }
#fg-tooltip .bn-cpd-eq { font-family: Consolas, monospace; font-size: 11px; margin-top: 5px; }
#fg-tooltip .bn-cpd-note { color: var(--text-muted); font-size: 10.5px; margin-top: 3px; }
</style>
"""

_EDGE_TOOLTIP_JS_OLD = re.compile(r"if \(d\.label\) html \+= '<div>' \+ escapeHtml\(d\.label\) \+ '</div>';")
_EDGE_TOOLTIP_JS_NEW = ("if (d.label) html += '<div>' + escapeHtml(d.label) + '</div>';"
                        " var __cpd = (window.__BN_EDGE_CPD || {})[d.source + '->' + d.target];"
                        " if (__cpd) html += __cpd;")


def _inject_edge_tooltips(filepath, edge_html):
    """Patch a d3blocks flowgraph HTML file so edge hovers also show ``edge_html``."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            page = f.read()
        block = _EDGE_TOOLTIP_CSS
        if edge_html:
            if _EDGE_TOOLTIP_JS_OLD.search(page):
                page = _EDGE_TOOLTIP_JS_OLD.sub(lambda m: _EDGE_TOOLTIP_JS_NEW, page, count=1)
                data = json.dumps(edge_html, ensure_ascii=False).replace('</', '<\\/')
                block += '<script>window.__BN_EDGE_CPD = ' + data + ';</script>\n'
            else:
                logger.warning('Edge CPD tooltips are not supported by this d3blocks version; skipped.')
        if '</head>' in page:
            page = page.replace('</head>', block + '</head>', 1)
        else:
            page = block + page
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(page)
        return True
    except Exception as err:
        logger.warning('Could not add edge CPD tooltips: %s' % err)
        return False


def _open_html(filepath, notebook=False, figsize=(None, None)):
    """Show an HTML file inline (notebook) or in the browser."""
    if notebook:
        try:
            from IPython.display import IFrame, display
            width = figsize[0] if figsize and figsize[0] else '100%'
            height = figsize[1] if figsize and figsize[1] else 650
            display(IFrame(src=filepath, width=width, height=height))
            return
        except Exception:
            pass
    webbrowser.open('file://' + filepath)


def _fmt_num(x, nd=3):
    """Compact number formatting for tooltips."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return _html.escape(str(x))
    if x != 0 and abs(x) < 10 ** (-nd):
        return '%.1e' % x
    if abs(x) >= 1e5:
        return '%.3g' % x
    return '%.*f' % (nd, x)


def _cpd_title(variable, evidence, parent):
    """P(variable | e1, e2, ...) with the hovered parent in bold."""
    if evidence:
        ev = ', '.join('<b>%s</b>' % _html.escape(str(e)) if e == parent else _html.escape(str(e)) for e in evidence)
        return '<div class="bn-cpd-title">P(%s | %s)</div>' % (_html.escape(str(variable)), ev)
    return '<div class="bn-cpd-title">P(%s)</div>' % _html.escape(str(variable))


def _tabular_cpd_html(cpd, parent=None, max_rows=8, max_cols=6):
    """HTML table for a (pgmpy) TabularCPD: states as rows, parent configurations as columns."""
    var = cpd.variable
    evidence = list(cpd.variables[1:])
    values = np.asarray(cpd.get_values(), dtype=float)
    names = cpd.state_names
    var_states = list(names[var])
    combos = list(itertools.product(*[list(names[e]) for e in evidence])) if evidence else [()]
    n_rows, n_cols = values.shape
    show_r, show_c = min(n_rows, max_rows), min(n_cols, max_cols)

    h = [_cpd_title(var, evidence, parent), '<table class="bn-cpd">']
    if evidence:
        for i, e in enumerate(evidence):
            cls = 'e hl' if e == parent else 'e'
            h.append('<tr><th class="%s">%s</th>' % (cls, _html.escape(str(e))))
            h.extend('<th>%s</th>' % _html.escape(str(combos[j][i])) for j in range(show_c))
            h.append('<th>…</th></tr>' if show_c < n_cols else '</tr>')
    else:
        h.append('<tr><th class="e"></th><th>P</th></tr>')
    for r in range(show_r):
        h.append('<tr><th class="s">%s=%s</th>' % (_html.escape(str(var)), _html.escape(str(var_states[r]))))
        for c in range(show_c):
            p = float(values[r, c])
            h.append('<td style="background:rgba(66,133,244,%.2f)">%s</td>' % (0.05 + 0.55 * min(max(p, 0.0), 1.0), _fmt_num(p)))
        h.append('<td>…</td></tr>' if show_c < n_cols else '</tr>')
    if show_r < n_rows:
        h.append('<tr><th class="s">…</th>' + '<td>…</td>' * (show_c + (1 if show_c < n_cols else 0)) + '</tr>')
    h.append('</table>')
    notes = []
    if show_c < n_cols:
        notes.append('showing %d of %d parent configurations' % (show_c, n_cols))
    if show_r < n_rows:
        notes.append('showing %d of %d states' % (show_r, n_rows))
    if parent is not None and parent not in evidence:
        notes.append('edge from %s is not used in this discrete CPT' % _html.escape(str(parent)))
    if notes:
        h.append('<div class="bn-cpd-note">%s</div>' % '; '.join(notes))
    return ''.join(h)


def _gaussian_cpd_html(variable, disc_parents, cont_parents, configs, parent=None, max_rows=8):
    """HTML for a (conditional) linear Gaussian: one regression row per discrete-parent configuration.

    configs : dict {tuple(discrete states): {'beta': [intercept, b1, ...], 'std': float, 'n': int | None}}
    """
    all_parents = list(disc_parents) + list(cont_parents)
    items = list(configs.items())
    show = items[:max_rows]
    has_n = any(c.get('n') is not None for _, c in items)

    h = [_cpd_title(variable, all_parents, parent)]
    # Equation for the common single-regression case
    if len(items) == 1 and len(disc_parents) == 0:
        c = items[0][1]
        terms = [_fmt_num(c['beta'][0])] + ['%s·%s' % (_fmt_num(b), _html.escape(str(p))) for b, p in zip(c['beta'][1:], cont_parents)]
        h.append('<div class="bn-cpd-eq">%s = %s + ε,&nbsp; ε ~ N(0, %s²)</div>' % (
            _html.escape(str(variable)), ' + '.join(terms), _fmt_num(c['std'])))

    h.append('<table class="bn-cpd"><tr>')
    for p in disc_parents:
        h.append('<th class="e%s">%s</th>' % (' hl' if p == parent else '', _html.escape(str(p))))
    h.append('<th>intercept</th>')
    for p in cont_parents:
        h.append('<th%s>β(%s)</th>' % (' class="hl"' if p == parent else '', _html.escape(str(p))))
    h.append('<th>σ</th>' + ('<th>n</th>' if has_n else '') + '</tr>')
    for key, c in show:
        h.append('<tr>')
        for k in key:
            h.append('<th class="s">%s</th>' % _html.escape(str(k)))
        beta = list(c['beta'])
        h.append('<td>%s</td>' % _fmt_num(beta[0]))
        for i, p in enumerate(cont_parents):
            h.append('<td%s>%s</td>' % (' class="hl"' if p == parent else '', _fmt_num(beta[i + 1]) if i + 1 < len(beta) else ''))
        h.append('<td>%s</td>' % _fmt_num(c['std']))
        if has_n:
            h.append('<td>%s</td>' % (int(c['n']) if c.get('n') is not None else ''))
        h.append('</tr>')
    h.append('</table>')
    if len(items) > len(show):
        h.append('<div class="bn-cpd-note">showing %d of %d configurations</div>' % (len(show), len(items)))
    return ''.join(h)


def _linear_gaussian_cpd_html(cpd, parent=None):
    """HTML for a pgmpy LinearGaussianCPD (version tolerant)."""
    beta = getattr(cpd, 'beta', None)
    if beta is None:
        beta = getattr(cpd, 'mean', None)
    std = getattr(cpd, 'std', None)
    if std is None and getattr(cpd, 'variance', None) is not None:
        std = float(np.sqrt(cpd.variance))
    if beta is None or std is None:
        return None
    evidence = list(getattr(cpd, 'evidence', []) or [])
    configs = {(): {'beta': [float(b) for b in np.ravel(beta)], 'std': float(std), 'n': None}}
    return _gaussian_cpd_html(cpd.variable, [], evidence, configs, parent=parent)


def _node_cpd_tooltips(model, nodelist, max_rows=8, max_cols=6):
    """Create a neat HTML CPD for every node that has learned parameters.

    Returns
    -------
    dict
        ``{node: html}``. Nodes without a CPD are left out.
    """
    out = {}
    if not isinstance(model, dict):
        return out
    cpds = {}
    bnmodel = model.get('model', None)
    if hasattr(bnmodel, 'get_cpds'):
        try:
            cpds = {c.variable: c for c in bnmodel.get_cpds()}
        except Exception as err:
            logger.debug('Could not collect CPDs: %s' % err)
    cg = {c['variable']: c for c in (model.get('continuous_cpds') or []) if isinstance(c, dict) and 'variable' in c}
    for node in nodelist:
        h = None
        try:
            if node in cg:
                c = cg[node]
                configs = {k: {'beta': d['beta'], 'std': d['std'], 'n': d.get('n')} for k, d in c['configs'].items()}
                h = _gaussian_cpd_html(node, c.get('disc_parents', []), c.get('cont_parents', []), configs, max_rows=max_rows)
            elif node in cpds:
                cpd = cpds[node]
                if hasattr(cpd, 'state_names') and hasattr(cpd, 'get_values'):
                    h = _tabular_cpd_html(cpd, max_rows=max_rows, max_cols=max_cols)
                else:
                    h = _linear_gaussian_cpd_html(cpd)
        except Exception as err:
            logger.debug('No CPD tooltip for node %s: %s' % (node, err))
        if h:
            out[node] = h
    return out


def _edge_cpd_tooltips(model, edgelist, max_rows=8, max_cols=6):
    """Create a neat HTML CPD for every edge.

    For an edge ``A -> B`` the CPD of the target ``B`` is shown, because ``B`` is the node whose
    distribution is conditioned on ``A``. Supports discrete (tabular), linear-gaussian and
    conditional-gaussian (``model['continuous_cpds']``) models. Edges without learned parameters
    get no entry, so the default hover is kept.

    Returns
    -------
    dict
        ``{(source, target): html}``
    """
    out = {}
    if not isinstance(model, dict):
        return out

    cpds = {}
    bnmodel = model.get('model', None)
    if hasattr(bnmodel, 'get_cpds'):
        try:
            cpds = {c.variable: c for c in bnmodel.get_cpds()}
        except Exception as err:
            logger.debug('Could not collect CPDs: %s' % err)
    cg = {c['variable']: c for c in (model.get('continuous_cpds') or []) if isinstance(c, dict) and 'variable' in c}

    if not cpds and not cg:
        logger.info('No learned parameters found: use bn.parameter_learning.fit() first to show CPDs on edge hover.')
        return out

    for u, v in edgelist:
        h = None
        try:
            if v in cg:
                c = cg[v]
                configs = {k: {'beta': d['beta'], 'std': d['std'], 'n': d.get('n')} for k, d in c['configs'].items()}
                h = _gaussian_cpd_html(v, c.get('disc_parents', []), c.get('cont_parents', []), configs, parent=u, max_rows=max_rows)
            elif v in cpds:
                cpd = cpds[v]
                if hasattr(cpd, 'state_names') and hasattr(cpd, 'get_values'):
                    h = _tabular_cpd_html(cpd, parent=u, max_rows=max_rows, max_cols=max_cols)
                else:
                    h = _linear_gaussian_cpd_html(cpd, parent=u)
        except Exception as err:
            logger.debug('No CPD tooltip for edge %s -> %s: %s' % (u, v, err))
        if h:
            out[(str(u), str(v))] = h
    return out

# %% Interactive d3graph
def _plot_interactive_d3graph(params_interactive, nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights, title, tooltip):
    try:
        from d3blocks import D3Blocks
        import d3blocks as d3
    except ModuleNotFoundError:
        raise Exception('[bnlearn] >"d3blocks" library is not installed. Pip install first: "pip install d3blocks"')

    # Check version
    if version.parse(d3.__version__) < version.parse("2.0.0"):
        raise ImportError('[bnlearn] >Error: d3blocks version should be >= 2.0.0. Pip install: "pip install -U d3blocks"')

    if params_interactive['filepath'] is None:
        params_interactive['filepath'] = title.strip().replace(' ', '_') + '.html'

    nodelist = list(map(lambda x: x.encode('ascii', 'ignore').decode("utf-8").replace(' ', '_'), list(nodelist)))

    # Initialize
    d3 = D3Blocks()

    X = pd.DataFrame(data=edgelist, columns=['source', 'target'])
    X['weight'] = edge_weights

    # If there are no edges or all weights are non-positive, d3graph/d3blocks
    # will raise an exception. Handle this case gracefully by logging and
    # returning None so callers can proceed (or fall back to a static plot).
    if X.shape[0] == 0 or (X['weight'] <= 0).all():
        logger.info('No edges with positive weight for interactive d3graph; skipping interactive plot.')
        return None

    d3.d3graph(X, showfig=False, title=title, notebook=params_interactive['notebook'])

    d3.D3graph.set_edge_properties(directed=True,
                                   minmax_distance=params_interactive['minmax_distance'],
                                   marker_color=edge_colors)

    _, IB = ismember([*d3.D3graph.node_properties.keys()], nodelist)

    d3.D3graph.set_node_properties(tooltip=np.array(tooltip)[IB],
                                   size=np.array(node_sizes)[IB],
                                   color=np.array(node_colors)[IB],
                                   fontcolor=params_interactive['font_color'])

    d3.D3graph.show(show_slider=params_interactive['show_slider'],
                    filepath=params_interactive['filepath'],
                    figsize=params_interactive['figsize'])

    return os.path.abspath(d3.D3graph.config['filepath'])


# %% Plot properties
def _plot_properties(G, node_properties, edge_properties, node_color, node_size):
    edges = list(edge_properties.keys())
    for edge in edges:
        props = edge_properties.get((edge[0], edge[1]), {})
        G.add_edge(
            edge[0], edge[1],
            weight=props.get('weight', 1),
            color=props.get('color', '#000000'),
            p_value=props.get('p_value', 1.0),
            logp=props.get('logp', 0.0),
            value=props.get('value', 0),
        )

    edgelist = list(G.edges())
    edge_colors = [G[u][v].get('color') for u, v in G.edges()]
    edge_weights = [G[u][v].get('weight') for u, v in G.edges()]
    edge_p_value = [G[u][v].get('p_value') for u, v in G.edges()]
    edge_value = [G[u][v].get('value') for u, v in G.edges()]

    # Use all nodes in G so that isolated nodes (no edges) are included.
    nodelist = sorted(G.nodes())
    node_colors = []
    node_sizes = []
    for node in nodelist:
        props = node_properties.get(node, {})
        if node_color is not None:
            node_colors.append(node_color)
        else:
            node_colors.append(props.get('node_color', '#ADD8E6'))
        if node_size is not None:
            node_sizes.append(node_size)
        else:
            node_sizes.append(props.get('node_size', 800))

    return nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights, edge_p_value, edge_value
