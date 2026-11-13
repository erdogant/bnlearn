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
import copy

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from setgraphviz import setgraphviz

import bnlearn as bn
try:
    from bnlearn.utils import vec2adjmat, adjmat2vec, _normalize_weights
    from bnlearn.plot_interactive import plot_interactive
    from bnlearn.mermaid import to_mermaid, write_mermaid
except:
    from utils import vec2adjmat, adjmat2vec, _normalize_weights
    from plot_interactive import plot_interactive
    from mermaid import to_mermaid, write_mermaid

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
    >>> df = bn.fetch(name='asia', as_dataframe=True)
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
                  edge_label='auto',
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
    edge_label : str or None
        None: Do not show numeric edge labels.
        'auto': Automatically choose for logp and otherwise weight when available
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

    logger.info(f'Setting edge mode to {edge_label}.')
    # Set edge label
    edge_label = _set_edge_label(model, edge_label)
    indep = model.get('independence_test')
    if indep is not None and edge_label in ('logp', 'p_value') and 'stat_test' in indep.columns:
        Iloc = indep['stat_test'].astype(bool)
        source = indep['source'].loc[Iloc]
        target = indep['target'].loc[Iloc]
        logger.info(f'Number of significant edges detected: {int(Iloc.sum())}')
        if edge_label == 'logp':
            logp = compute_logp(indep['p_value'])
            adjmat = vec2adjmat(source, target, weights=logp.loc[Iloc], symmetric=True, aggfunc='sum')
        else:
            adjmat = vec2adjmat(source, target, weights=indep['p_value'].loc[Iloc], symmetric=True, aggfunc='sum')
    else:
        edges = sum(1 for item in (model.get('model_edges') or []) if isinstance(item, tuple))
        logger.info(f'Number of edges detected: {edges}')
        adjmat = model['adjmat'].copy()

    node_labels = list(adjmat.T.columns) if edge_label is not None else None

    if node_labels is not None and len(node_labels) > 0:
        dot_graph = make_dot(adjmat.T.values.astype(float), labels=node_labels, lower_limit=0, **params)

    return dot_graph


# %% plot
def plot(model,
         edge_label='auto',
         pos=None,
         scale=1,
         title='Directed Acyclic Graph',
         node_color=None,
         node_size=None,
         node_properties=None,
         edge_properties=None,
         params_interactive={
             'type': 'flowgraph',       # choose between 'd3graph' or 'flowgraph'
             'figsize': [None, None],
             'notebook': False,
             'filepath': None,
             'direction': 'TB',          # 'LR' | 'TB' | 'RL' | 'BT'
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
         params_static=None,
         output='static',
         params_mermaid=None,
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
    output : str, optional
        'static': Static matplotlib plot (default).
        'interactive': Interactive web-based graph (d3blocks).
        'mermaid': Mermaid code (flowchart). The code is returned in out['mermaid'].
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
    edge_label : str or None, optional
        None: Do not annotate edges with numbers.
        'auto': Automatically choose for logp and otherwise weight when available
        'weight': Show the structure / coefficient values.
        'p_value': Show raw edge p-values (requires independence_test).
        'logp': Show -log10(p_value) strength (requires independence_test).
    params_interactive : dict, optional
        Dictionary containing various settings for interactive plots.
    params_mermaid : dict, optional
        Settings for output='mermaid': {'direction': 'TB' ('TB' | 'BT' | 'LR' | 'RL'), 'filepath': None
        (save the code; '.md' adds a ```mermaid fence, else raw code), 'show': True (print the code)}.
    params_static : dict, optional
        Dictionary containing various settings for static plots. Missing keys get the default (see _static_defaults).
        layout: 'graphviz_layout_custom', 'graphviz_layout', 'spring_layout',
        'planar_layout', 'shell_layout', 'spectral_layout', 'pydot_layout',
        'circular_layout', 'random_layout', 'bipartite_layout', 'multipartite_layout'.

    Returns
    -------
    dict
        pos, G, node_properties, edge_properties, fig/ax. With output='mermaid' also the code in 'mermaid'.
    """
    fig = None
    output = _resolve_output(output)
    if model is None or (model.get('adjmat', None) is None) or model['adjmat'].sum().sum() == 0:
        logger.info('Nothing to plot because no edges are present between nodes. ')
        return None

    if model.get('config', {}).get('method') == 'DBN' and output == 'interactive':
        logger.info('DynamicBayesianNetwork (DBN) can not be plot with Graphviz.')
        return None
    
    # Set edge_label
    edge_label = _set_edge_label(model, edge_label)

    model = copy.deepcopy(model)
    model['adjmat'] = model['adjmat'].astype(float)
    params_interactive = resolve_interactive_params(params_interactive)
    params_static = resolve_static_params(params_static)

    out = {}
    G = nx.DiGraph()
    if model.get('adjmat', None) is not None:
        G.add_nodes_from(model['adjmat'].columns.values)

    node_size_default = 10 if output == 'interactive' else 800
    if (node_properties is not None) and (node_size is not None):
        logger.warning('Warning: if both "node_size" and "node_properties" are used, "node_size" will be used.')
    if node_properties is None:
        node_properties = get_node_properties(model, node_size=node_size_default)
    if edge_properties is None:
        edge_properties = get_edge_properties(model, minscale=params_static['minscale'], maxscale=params_static['maxscale'])

    edge_properties = _filter_significant_edges(model, edge_properties, edge_label)
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
    if output == 'interactive':
        fig = plot_interactive(model, params_interactive, nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights, title, edge_properties=edge_properties, edge_label=edge_label, return_mermaid=False, showfig=True)
    elif output == 'mermaid':
        out['mermaid'] = plot_interactive(model, params_interactive, nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights, title, edge_properties=edge_properties, edge_label=edge_label, return_mermaid=True, showfig=False)
    else:
        if ('bayes' in str(type(bnmodel)).lower()) or ('pgmpy' in str(type(bnmodel)).lower()) or ('lingam' in model['config']['method']):
            logger.info('Plot based on Bayesian model')
            pos = compute_static_layout(G, pos=pos, scale=scale, layout=params_static['layout'])
        elif 'networkx' in str(type(bnmodel)):
            logger.info('Plot based on networkx model')
            G = bnmodel
            pos = compute_static_layout(G, pos=pos, scale=scale, layout=params_static['layout'])
        else:
            logger.info('Plot based on adjacency matrix')
            G = bn.network.adjmat2graph(model['adjmat'].abs() > 0)
            pos = compute_static_layout(G, pos=pos, scale=scale, layout=params_static['layout'])

        fig = _plot_static(model, params_static, nodelist, node_colors, node_sizes, G, pos,
                           edge_colors, edge_weights, showplot=params_static['showplot'],
                           visible=params_static['visible'], title=title, dpi=params_static['dpi'],
                           edge_label=edge_label)

    out['fig'] = fig
    out['ax'] = fig
    out['pos'] = pos
    out['G'] = G
    out['node_properties'] = node_properties
    out['edge_properties'] = edge_properties
    return out


# %% Shared helpers (static plot and mermaid)
def _set_edge_label(model, edge_label):
    if model.get('independence_test', None) is not None and edge_label=='auto':
        logger.info('edge_label is automatically set to logp')
        edge_label = 'logp'
    elif model.get('independence_test', None) is None and edge_label=='auto':
        edge_label = 'weight'

    # Remove pvalue label is no independence test is performed but pvalue is requested
    if model.get('independence_test', None) is None and edge_label in ('p_value', 'logp'):
        logger.warning('edge_label p_value/logp require: model=bn.independence_test(model, df)')
        edge_label = None

    # Return
    return edge_label


def _format_edge_label(value, edge_label):
    """Format the value of an edge label according to edge_label."""
    if edge_label == 'p_value':
        return float(f'{value:.2g}')
    return float(f'{value:.2f}'[:4])


def _filter_significant_edges(model, edge_properties, edge_label):
    """Keep only the significant edges when edge_label is 'p_value' or 'logp'."""
    if edge_label in ('p_value', 'logp') and model.get('independence_test') is not None:
        indep = model['independence_test']
        if 'stat_test' in indep.columns and len(indep) > 0:
            sig = indep.loc[indep['stat_test'].astype(bool), ['source', 'target']]
            sig_set = set(zip(sig['source'], sig['target']))
            n_before = len(edge_properties)
            edge_properties = {e: p for e, p in edge_properties.items() if e in sig_set}
            logger.info(f'Number of significant edges detected: {len(edge_properties)} (of {n_before})')
    return edge_properties


# %% Shared static-plot helpers
def _static_defaults():
    """Default parameters of the static plot. Single source for plot() and compare(), so both figures look the same."""
    return {'minscale': 1, 'maxscale': 5, 'figsize': (10, 10), 'width': None, 'height': None, 'font_size': 10,
            'font_family': 'sans-serif', 'alpha': 0.8, 'node_shape': 'o', 'layout': 'graphviz_layout_custom',
            'font_color': '#000000', 'facecolor': 'white', 'edge_alpha': 0.8, 'arrowstyle': '-|>', 'arrowsize': 20,
            'visible': True, 'showplot': True, 'dpi': 200}


def _interactive_defaults():
    """Default parameters of the interactive plot. Single source for plot() and compare()."""
    return {
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


def resolve_interactive_params(params_interactive=None):
    """Merge user-defined interactive parameters with the defaults."""
    return {**_interactive_defaults(), **(params_interactive or {})}


def resolve_static_params(params_static=None, width=None, height=None):
    """Merge user-defined static parameters with the defaults.

    An explicit width and/or height overrides figsize (same behaviour as plot()).
    """
    params_static = {**_static_defaults(), **(params_static or {})}
    width = params_static.get('width') if width is None else width
    height = params_static.get('height') if height is None else height
    if (width is not None) or (height is not None):
        params_static['width'], params_static['height'] = width, height
        base = params_static['figsize']
        params_static['figsize'] = (base[0] if width is None else width, base[1] if height is None else height)
    return params_static


def compute_static_layout(G, pos=None, scale=1, layout='graphviz_layout_custom'):
    """Compute node coordinates using the same logic as plot().

    The default layout 'graphviz_layout_custom' is the hierarchical top-down layout. It requires a
    DAG; when the graph contains cycles (e.g., a comparison of two networks with reversed edges)
    it falls back to 'graphviz_layout' (which itself falls back to spring_layout).
    """
    if pos is not None:
        return bn.network.graphlayout(G, pos=pos, scale=scale, layout=layout)
    if layout == 'graphviz_layout_custom':
        try:
            logger.info('Using hierarchical top-down layout (graphviz_layout_custom).')
            return hierarchical_layout(G, scale=scale)
        except nx.NetworkXUnfeasible:
            logger.warning('Graph contains cycles; hierarchical layout is not possible. Using graphviz_layout instead.')
            layout = 'graphviz_layout'
    return bn.network.graphlayout(G, pos=pos, scale=scale, layout=layout)


# %% Static plot
def _plot_static(model, params_static, nodelist, node_colors, node_sizes, G, pos, edge_colors, edge_weights, title, visible=True, showplot=True, dpi=100, edge_label='weight', edge_labels=None, edge_styles=None, edge_arrowstyles=None):
    # Create figure
    fig = plt.figure(figsize=params_static['figsize'], facecolor=params_static['facecolor'], dpi=dpi)
    fig.set_visible(visible)

    # Make networkx
    nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_size=node_sizes, alpha=params_static['alpha'], node_color=node_colors, node_shape=params_static['node_shape'])
    if edge_styles is None and edge_arrowstyles is None:
        nx.draw_networkx_edges(G, pos, arrowstyle=params_static['arrowstyle'], arrowsize=params_static['arrowsize'], edge_color=edge_colors, width=edge_weights, alpha=params_static['edge_alpha'])
    else:
        # Optional per-edge line style (e.g., 'dashed') and arrow style (e.g., '<|-|>'), aligned with G.edges().
        # Edges are drawn per (style, arrowstyle) group, which works for all networkx versions.
        edgelist = list(G.edges())
        styles = edge_styles if edge_styles is not None else ['solid'] * len(edgelist)
        arrows = edge_arrowstyles if edge_arrowstyles is not None else [params_static['arrowstyle']] * len(edgelist)
        for key in dict.fromkeys(zip(styles, arrows)):
            idx = [i for i in range(len(edgelist)) if (styles[i], arrows[i]) == key]
            nx.draw_networkx_edges(G, pos, edgelist=[edgelist[i] for i in idx], style=key[0], arrowstyle=key[1], arrowsize=params_static['arrowsize'],
                                   edge_color=[edge_colors[i] for i in idx], width=[edge_weights[i] for i in idx], alpha=params_static['edge_alpha'])

    if edge_labels is not None:
        # Custom labels, e.g., provided by network.compare()
        if len(edge_labels) > 0:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    elif edge_label in _EDGE_LABEL_ATTR:
        edge_label = nx.get_edge_attributes(G, _EDGE_LABEL_ATTR[edge_label])
        edge_label = {key: _format_edge_label(value, edge_label) for key, value in edge_label.items()}
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
