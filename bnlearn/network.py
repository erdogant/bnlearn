"""Functionalities for network creation, clustering and plotting."""
# ------------------------------------
# Name        : network.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------


# %% Libraries
import logging
logger = logging.getLogger("bnlearn")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import community
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from ismember import ismember
import bnlearn

# %% Make graph from adjacency matrix
def to_graph(adjmat):
    assert float(nx.__version__)>2, 'This function requires networkx to be v2 or higher. Try to: pip install --upgrade networkx'
    adjmat = is_DataFrame(adjmat)
    logger.info('Making graph')
    G=nx.from_pandas_adjacency(adjmat)
    return(G)


# %% Convert Adjmat to graph (G) (also works with lower versions of networkx)
def adjmat2graph(adjmat):
    G = nx.DiGraph()  # Directed graph
    # Add all nodes first, so that nodes without any incoming or outgoing
    # edges (isolated nodes) are still part of the graph.
    G.add_nodes_from(adjmat.columns.values)
    # Convert adjmat to source target
    df_edges=adjmat.stack().reset_index()
    df_edges.columns=['source', 'target', 'weight']
    df_edges['weight']=df_edges['weight'].astype(float)

    # Add directed edge with weigth
    for i in range(df_edges.shape[0]):
        if df_edges['weight'].iloc[i]!=0:
            # Setup color
            if df_edges['weight'].iloc[i]==1:
                color='k'
            elif df_edges['weight'].iloc[i]>1:
                color='r'
            elif df_edges['weight'].iloc[i]<0:
                color='b'
            else:
                color='p'

            # Create edge in graph
            G.add_edge(df_edges['source'].iloc[i], df_edges['target'].iloc[i], weight=np.abs(df_edges['weight'].iloc[i]), color=color)
    # Return
    return(G)

# %% Compute similarity matrix
def compute_centrality(G, centrality='betweenness'):
    logger.info('Computing centrality %s' %(centrality))
    if centrality=='betweenness':
        bb=nx.centrality.betweenness_centrality(G)
    elif centrality=='closeness':
        bb=nx.centrality.closeness_centrality(G)
    elif centrality=='eigenvector':
        bb=nx.centrality.eigenvector_centrality(G)
    elif centrality=='degree':
        bb=nx.centrality.degree_centrality(G)
    elif centrality=='edge':
        bb=nx.centrality.edge_betweenness(G)
    elif centrality=='harmonic':
        bb=nx.centrality.harmonic_centrality(G)
    elif centrality=='katz':
        bb=nx.centrality.katz_centrality(G)
    elif centrality=='local':
        bb=nx.centrality.local_reaching_centrality(G)
    elif centrality=='out_degree':
        bb=nx.centrality.out_degree_centrality(G)
    elif centrality=='percolation':
        bb=nx.centrality.percolation_centrality(G)
    elif centrality=='second_order':
        bb=nx.centrality.second_order_centrality(G)
    elif centrality=='subgraph':
        bb=nx.centrality.subgraph_centrality(G)
    elif centrality=='subgraph_exp':
        bb=nx.centrality.subgraph_centrality_exp(G)
    elif centrality=='information':
        bb=nx.centrality.information_centrality(G)
    else:
        logger.error('Error: Centrality <%s> does not exist!' %(centrality))
    # Set the attributes
    score=np.array([*bb.values()])
    nx.set_node_attributes(G, bb, centrality)

    return(G, score)

# %% compute clusters
def cluster(G):
    logger.info('Clustering using best partition')
    # Partition
    partition=community.best_partition(G)
    # Set property to node
    nx.set_node_attributes(G, partition, 'clusterlabel')
    # Extract labels
    labx=[partition.get(node) for node in G.nodes()]
    labx=np.array(labx)

    return(G, labx)

# %% Normalize in good d3 range
def normalize_size(getsizes, minscale=0.1, maxscale=4):
    getsizes = MinMaxScaler(feature_range=(minscale, maxscale)).fit_transform(getsizes).flatten()
    return(getsizes)

# %% Convert dataframe to Graph
def df2G(df_nodes, df_edges):
    # Put edge information in G
    #    G = nx.from_pandas_edgelist(df_edges, 'source', 'target', ['weight', 'edge_weight','edge_width','source_label','target_label'])

    colnames=list(df_edges.columns.values[~np.isin(df_edges.columns.values, ['source', 'target'])])
    G = nx.from_pandas_edgelist(df_edges, 'source', 'target', colnames)

    # Put node info in G
    getnodes=[*G.nodes]
    for col in df_nodes.columns:
        for i in range(0, df_nodes.shape[0]):
            if np.any(np.isin(getnodes, df_nodes.index.values[i])):
                G.nodes[df_nodes.index.values[i]][col] = str(df_nodes[col].iloc[i])

    return(G)

# %% Convert Graph to dataframe
def G2df(G, node_color=None, node_label=None, node_size=100, edge_distance_minmax=[1, 100]):
    # Nodes
    df_node_names=pd.DataFrame([*G.nodes], columns=['node_name'])
    df_node_props=pd.DataFrame([*G.nodes.values()])
    df_nodes=pd.concat([df_node_names, df_node_props], axis=1)

    if not np.any(df_nodes.columns=='node_color'):
        df_nodes['node_color']='#000080'
    if not np.any(df_nodes.columns=='node_color_edge'):
        df_nodes['node_color_edge']='#000000'
    if not np.any(df_nodes.columns=='node_size_edge'):
        df_nodes['node_size_edge']=1
    if not isinstance(node_label, type(None)):
        df_nodes['node_label']=node_label

    if np.any(df_nodes.columns=='node_size'):
        df_nodes['node_size']=normalize_size(df_nodes['node_size'].values.reshape(-1, 1), 1, 10)
    else:
        df_nodes['node_size']=10

    # Edges
    df_edge_links=pd.DataFrame([*G.edges], columns=['source_label', 'target_label'])
    df_edge_props=pd.DataFrame([*G.edges.values()])
    df_edges=pd.concat([df_edge_links, df_edge_props], axis=1)

    # Source and target values
    df_nodes['index_value']=None
    df_edges['source']=None
    df_edges['target']=None
#    uinodes=np.unique(np.append(df_edges['source_label'], df_edges['target_label']))
    uinodes=np.unique(df_nodes['node_name'])
    for i in range(0, len(uinodes)):
        I=(uinodes[i]==df_edges['source_label'])
        df_edges['source'].loc[I]=i
        I=(uinodes[i]==df_edges['target_label'])
        df_edges['target'].loc[I]=i

        I=df_nodes['node_name']==uinodes[i]
        df_nodes['index_value'].loc[I]=i

    df_nodes.set_index(df_nodes['index_value'], inplace=True)
    del df_nodes['index_value']

    # Include width and weights
    if not np.any(df_edges.columns=='edge_weight') and np.any(df_edges.columns=='weight'):
        df_edges['edge_weight']=normalize_size(df_edges['weight'].values.reshape(-1, 1), edge_distance_minmax[0], edge_distance_minmax[1])
    else:
        df_edges['edge_weight']=2
    if not np.any(df_edges.columns=='edge_width'):
        df_edges['edge_width']=2
    if not np.any(df_edges.columns=='weight'):
        df_edges['weight']=2

    # Remove self-loops
    I=df_edges['source']!=df_edges['target']
    df_edges=df_edges.loc[I, :].reset_index(drop=True)

    return(df_nodes, df_edges)


# %% Comparison of two networks
# Missing edges: thicker line with a long dash pattern (offset, (dash, gap)); the pattern scales with the line width.
# The default 'dashed' pattern at the default edge width looks almost solid on diagonal lines.
_DASHED = (0, (3, 2.5))
_DASHED_WIDTH = 2

def compare(adjmat_true, adjmat_pred, pos=None, showfig=True, width=None, height=None, layout='graphviz_layout_custom',
            scale=1, title='Network comparison', node_color=None, node_size=None, params_static=None):
    """Compare two networks and plot the differences.

    The figure uses the same static plot logic as bnlearn.plot (shared layout, node/edge styling,
    figure parameters). Edges: black = correct edge, red = extra edge (false positive),
    black dashed = missing edge (false negative), green = reversed edge (the edge exists in both
    networks but in opposite directions, i.e., a false positive and a false negative between the same
    two nodes). The arrow of a reversed edge points in the predicted (wrong) direction.

    Parameters
    ----------
    adjmat_true : pd.DataFrame
        Adjacency matrix of the true network.
    adjmat_pred : pd.DataFrame
        Adjacency matrix of the predicted network.
    pos : dict, optional
        Existing node coordinates, e.g., ``bn.plot(truth)['pos']``. The default is None, in which case
        the layout is computed from the first input network (adjmat_true) with the same logic as plot(),
        so the nodes are positioned as in ``bn.plot(truth)`` and the second network is mapped on top of it.
    showfig : bool
        Show the confusion matrix and the network plot.
    width, height : float, optional
        Figure size. Overrules params_static['figsize'].
    layout : str
        Layout of the network. Default 'graphviz_layout_custom' (hierarchical top-down), as in plot().
    scale : float
        Scale of the layout.
    title : str
        Title of the network plot.
    node_color, node_size : optional
        Overrule the default node color and size.
    params_static : dict, optional
        Static plot parameters, identical to those of plot() (font_size, alpha, arrowstyle, dpi, ...).

    Returns
    -------
    scores : dict
        Confusion matrix scores.
    adjmat_diff : pd.DataFrame
        1: correct edge, 2: extra edge (false positive), -1: missing edge (false negative).
    """
    # Make sure columns and indices to match
    [IArow, IBrow]=ismember(adjmat_true.index.values, adjmat_pred.index.values)
    [IAcol, IBcol]=ismember(adjmat_true.columns.values, adjmat_pred.columns.values)
    adjmat_true = adjmat_true.loc[IArow, IAcol]
    adjmat_pred = adjmat_pred.iloc[IBrow, IBcol]

    # Make sure it is boolean adjmat
    adjmat_true = adjmat_true>0
    adjmat_pred = adjmat_pred>0

    # Check whether order is correct
    assert np.all(adjmat_true.columns.values==adjmat_pred.columns.values), 'Column order of both input values could not be matched'
    assert np.all(adjmat_true.index.values==adjmat_pred.index.values), 'Row order of both input values could not be matched'

    # Get edges
    y_true = adjmat_true.stack().reset_index()[0].values
    y_pred = adjmat_pred.stack().reset_index()[0].values

    # Confusion matrix
    scores = bnlearn.confmatrix.twoclass(
        y_true,
        y_pred,
        threshold=0.5,
        classnames=['Disconnected', 'Connected'],
        title='',
        cmap=plt.cm.Blues,
        showfig=showfig,
        xlabel='Predicted network',
        ylabel='True network'
    )

    # Setup graph
    adjmat_diff = adjmat_true.astype(int)
    adjmat_diff[(adjmat_true.astype(int) - adjmat_pred.astype(int))<0]=2
    adjmat_diff[(adjmat_true.astype(int) - adjmat_pred.astype(int))>0]=-1

    if showfig:
        _plot_compare(adjmat_true, adjmat_diff, pos=pos, scale=scale, layout=layout, title=title, width=width, height=height,
                      node_color=node_color, node_size=node_size, params_static=params_static)

    # Return
    return(scores, adjmat_diff)


# %% Plot the difference graph using the static plot logic of bnlearn.plot
def _plot_compare(adjmat_true, adjmat_diff, pos=None, scale=1, layout='graphviz_layout_custom', title='Network comparison',
                  width=None, height=None, node_color=None, node_size=None, params_static=None):
    # Lazy import to prevent circular imports (plot imports network).
    from matplotlib.lines import Line2D
    from bnlearn.plot import resolve_static_params, compute_static_layout, get_node_properties, get_edge_properties, _plot_properties, _plot_static

    # Same parameter handling as plot()
    params_static = resolve_static_params(params_static, width=width, height=height)
    if layout is not None:
        params_static['layout'] = layout

    # Graph with all nodes (also isolated). Values in adjmat_diff: 1=correct, 2=extra (FP), -1=missing (FN)
    G = adjmat2graph(adjmat_diff)

    # Node and edge properties, same defaults as plot()
    # Created with the same functions as plot(), so node size/color and edge thickness are identical.
    model_diff = {'adjmat': adjmat_diff.astype(float)}
    node_properties = get_node_properties(model_diff, node_size=800)
    edge_properties = get_edge_properties(model_diff, minscale=params_static['minscale'], maxscale=params_static['maxscale'])
    # Only color and line/arrow style differ: they indicate the status of the edge.
    # correct: black solid | extra (FP): red solid | missing (FN): black dashed
    # reversed (FP and FN between the same two nodes): ONE green edge with only the arrow of the wrong
    # (predicted) direction. The FN edge in the true direction is not drawn.
    arrow = params_static['arrowstyle']
    for (u, v), props in edge_properties.items():
        value = adjmat_diff.loc[u, v]
        props['color'], props['style'], props['arrowstyle'] = {1: ('k', 'solid', arrow), 2: ('r', 'solid', arrow), -1: ('k', _DASHED, arrow)}[value]
        if value == -1:
            props['weight'] = _DASHED_WIDTH
    for (u, v) in list(edge_properties.keys()):
        if (u, v) in edge_properties and (v, u) in edge_properties and u != v:
            if adjmat_diff.loc[u, v] == 2 and adjmat_diff.loc[v, u] == -1:
                edge_properties[(u, v)].update({'color': 'g', 'style': 'solid', 'arrowstyle': arrow})
                del edge_properties[(v, u)]
                G.remove_edge(v, u)
    nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights, _, _ = _plot_properties(G, node_properties, edge_properties, node_color, node_size)
    edge_styles = [edge_properties[e]['style'] for e in edgelist]
    edge_arrowstyles = [edge_properties[e]['arrowstyle'] for e in edgelist]

    # Layout: the node positions are determined by the FIRST input network (the truth), using the same
    # logic as plot(). The differences with the second network are drawn on top of these positions.
    # (The diff graph itself is not used for the layout, as it can contain cycles and extra edges.)
    if pos is None:
        G_truth = adjmat2graph(adjmat_true.astype(int))
        pos = compute_static_layout(G_truth, pos=None, scale=scale, layout=params_static['layout'])
    else:
        pos = dict(pos)
    # Safety net: nodes without coordinates (e.g., a user-provided pos that is incomplete)
    missing = [node for node in G.nodes() if node not in pos]
    if len(missing) > 0:
        logger.warning('No coordinates for nodes %s in <pos>. These are positioned using the layout of the diff graph.' %(missing))
        pos_fill = compute_static_layout(G, pos=None, scale=scale, layout=params_static['layout'])
        pos.update({node: pos_fill[node] for node in missing})

    # Draw using the shared static plotter. Edge colors indicate the status, therefore no numeric labels.
    fig = _plot_static(adjmat_diff, params_static, nodelist, node_colors, node_sizes, G, pos,
                       edge_colors, edge_weights, title=title, visible=params_static['visible'],
                       showplot=False, dpi=params_static['dpi'], edge_filter=None, edge_labels={},
                       edge_styles=edge_styles, edge_arrowstyles=edge_arrowstyles)

    # Legend
    handles = [Line2D([0], [0], color='k', lw=2, label='Correct edge'),
               Line2D([0], [0], color='r', lw=2, label='Extra edge (false positive)'),
               Line2D([0], [0], color='k', lw=2, linestyle=_DASHED, label='Missing edge (false negative)'),
               Line2D([0], [0], color='g', lw=2, label='Reversed edge')]
    plt.legend(handles=handles, loc='best', frameon=False)
    if params_static['showplot']:
        plt.show()
    return fig


# %% Make graph layout
def graphlayout(G, pos, scale=1, layout='graphviz_layout'):
    if pos is None:
        if layout is not None:
            try:
                if layout=='graphviz_layout':
                    from networkx.drawing.nx_pydot import graphviz_layout
                    import copy
                    # Add edges with weights based on independence test results
                    # Set the edges weight to 1. Otherwise the graphviz_layout does not work good.
                    G = copy.deepcopy(G)
                    for edge in G.edges():
                        G.add_edge(*edge, weight=1)
                    # Compute positions
                    pos = graphviz_layout(G, prog="dot")
                else:
                    # Get the layout
                    layout_func = getattr(nx, layout)
                    pos = layout_func(G, scale=scale)
            except:
                logger.warning('Warning: [%s] layout not found. The layout [spring_layout] is used instead.' %(layout))
                pos = nx.spring_layout(G, scale=scale)
        else:
            pos = nx.spring_layout(G, scale=scale)
    else:
        logger.info('Existing coordinates from <pos> are used.')
    return pos


# %% Convert to pandas dataframe
def is_DataFrame(data):
    if isinstance(data, list):
        data=pd.DataFrame(data)
    elif isinstance(data, np.ndarray):
        data=pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        pass
    else:
        print('[bnlearn] >Typing should be pd.DataFrame()!')
        data=None

    return(data)