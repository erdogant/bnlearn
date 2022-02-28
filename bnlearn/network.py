"""Functionalities for network creation, clustering and plotting."""
# ------------------------------------
# Name        : network.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------


# %% Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import community
import networkx as nx
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler
from ismember import ismember
import bnlearn

# %% Make graph from adjacency matrix
def to_graph(adjmat, verbose=3):
    assert float(nx.__version__)>2, 'This function requires networkx to be v2 or higher. Try to: pip install --upgrade networkx'
    config = dict()
    config['verbose'] = verbose

    adjmat = is_DataFrame(adjmat)
    if config['verbose']>=3: print('[bnlearn] >Making graph')
    G=nx.from_pandas_adjacency(adjmat)

    return(G)


# %% Convert Adjmat to graph (G) (also works with lower versions of networkx)
def adjmat2graph(adjmat):
    G = nx.DiGraph()  # Directed graph
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
def compute_centrality(G, centrality='betweenness', verbose=3):
    if verbose>=3: print('[bnlearn] >Computing centrality %s' %(centrality))

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
        print('[bnlearn] >Error: Centrality <%s> does not exist!' %(centrality))

    # Set the attributes
    score=np.array([*bb.values()])
    nx.set_node_attributes(G, bb, centrality)

    return(G, score)

# %% compute clusters
def cluster(G, verbose=3):
    if verbose>=3: print('[bnlearn] >Clustering using best partition')
    # Partition
    partition=community.best_partition(G)
    # Set property to node
    nx.set_node_attributes(G, partition, 'clusterlabel')
    # Extract labels
    labx=[partition.get(node) for node in G.nodes()]
    labx=np.array(labx)

    return(G, labx)

# %% Compute cluster comparison
def cluster_comparison_centralities(G, width=5, height=4, showfig=False, methodtype='default', verbose=3):
    config=dict()
    config['showfig']=showfig
    config['width']=width
    config['height']=height
    config['verbose']=verbose

    if verbose>=3: print('[bnlearn] >Compute a dozen of centralities and clusterlabels')

    # compute labx for each of the centralities
    centralities=['betweenness', 'closeness', 'eigenvector', 'degree', 'edge', 'harmonic', 'katz', 'local', 'out_degree', 'percolation', 'second_order', 'subgraph', 'subgraph_exp', 'information']

    # Compute best positions for the network
    pos=nx.spring_layout(G)

    # Cluster data nd store label in G
    [G, score] = cluster(G)

    # Compute centrality score for each of the centralities and store in G
    for centrality in centralities:
        [G, score]=compute_centrality(G, centrality=centrality)

    # Store
    df=pd.DataFrame([*G.nodes.values()])
    df.set_index(np.array([*G.nodes.keys()]), inplace=True)

    # Make plots
    for centrality in centralities:
        if config['showfig']:
            plot(G, node_color=df['clusterlabel'], node_size=df[centrality], pos=pos, cmap='Set1', title=centrality, width=config['width'], height=config['height'], methodtype='default')

    return(G, df)

# %% Make plot
def plot(G, node_color=None, node_label=None, node_size=100, node_size_scale=[25, 200], alpha=0.8, font_size=18, cmap='Set1', width=40, height=30, pos=None, filename=None, title=None, methodtype='default', verbose=3):
    # https://networkx.github.io/documentation/networkx-1.7/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html
    config=dict()
    config['filename']=filename
    config['width']=width
    config['height']=height
    config['verbose']=verbose
    config['node_size_scale']=node_size_scale

    if verbose>=3: print('[bnlearn] >Creating network plot')

    if 'pandas' in str(type(node_size)):
        node_size=node_size.values

    # scaling node sizes
    if config['node_size_scale']!=None and 'numpy' in str(type(node_size)):
        if verbose>=3: print('[bnlearn] >Scaling node sizes')
        node_size=minmax_scale(node_size, feature_range=(node_size_scale[0], node_size_scale[1]))

    # Node positions
#    if isinstance(pos, type(None)):
#        pos=nx.spring_layout(G)

#    if isinstance(node_label, type(None)):
#        node_label=[*G.nodes])

    fig=plt.figure(figsize=(config['width'], config['height']))

    # Make the graph
    if methodtype=='circular':
        nx.draw_circular(G, labels=node_label, node_size=node_size, alhpa=alpha, node_color=node_color, cmap=cmap, font_size=font_size, with_labels=True)
    elif methodtype=='kawai':
        nx.draw_kamada_kawai(G, labels=node_label, node_size=node_size, alhpa=alpha, node_color=node_color, cmap=cmap, font_size=font_size, with_labels=True)
    else:
        nx.draw_networkx(G, labels=node_label, pos=pos, node_size=node_size, alhpa=alpha, node_color=node_color, cmap=cmap, font_size=font_size, with_labels=True)
#        nx.draw_networkx(G, pos=pos, node_size=node_size, alhpa=alpha, node_color=node_color, cmap=cmap, font_size=font_size)

    plt.title(title)
    plt.grid(True)
    plt.show()

    # Savefig
    if not isinstance(config['filename'], type(None)):
        if verbose>=3: print('[bnlearn] >Saving figure')
        plt.savefig(config['filename'])

    return(fig)

# %% Normalize in good d3 range
def normalize_size(getsizes, minscale=0.1, maxscale=4):
    getsizes = MinMaxScaler(feature_range=(minscale, maxscale)).fit_transform(getsizes).flatten()
    return(getsizes)

# %% Convert dataframe to Graph
def df2G(df_nodes, df_edges, verbose=3):
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
def G2df(G, node_color=None, node_label=None, node_size=100, edge_distance_minmax=[1, 100], verbose=3):
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

# %% Make plot
def bokeh(G, node_color=None, node_label=None, node_size=100, node_size_scale=[25, 200], alpha=0.8, font_size=18, cmap='Set1', width=40, height=30, pos=None, filename=None, title=None, methodtype='default', verbose=3):
    import networkx as nx
    from bokeh.io import show, output_file
    from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, BoxZoomTool, ResetTool
    from bokeh.models.graphs import from_networkx
    from bokeh.palettes import Spectral4

    SAME_CLUB_COLOR, DIFFERENT_CLUB_COLOR = "black", "red"
    edge_attrs = {}

    for start_node, end_node, _ in G.edges(data=True):
        edge_color = SAME_CLUB_COLOR if G.nodes[start_node]["club"] == G.nodes[end_node]["club"] else DIFFERENT_CLUB_COLOR
        edge_attrs[(start_node, end_node)] = edge_color

    nx.set_edge_attributes(G, edge_attrs, "edge_color")

    # Show with Bokeh
    plot = Plot(plot_width=400, plot_height=400,
                x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
    plot.title.text = "Graph Interaction Demonstration"

    node_hover_tool = HoverTool(tooltips=[("index", "@index"), ("club", "@club")])
    plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())

    graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0, 0))

    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
    graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha=0.8, line_width=1)
    plot.renderers.append(graph_renderer)

    output_file("interactive_graphs.html")
    show(plot)

# %% Comparison of two networks
def compare_networks(adjmat_true, adjmat_pred, pos=None, showfig=True, width=15, height=8, verbose=3):
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
    scores=bnlearn.confmatrix.twoclass(y_true, y_pred, threshold=0.5, classnames=['Disconnected', 'Connected'], title='', cmap=plt.cm.Blues, showfig=1, verbose=0)
    #bayes.plot(out_bayes['adjmat'], pos=G['pos'])

    # Setup graph
    adjmat_diff = adjmat_true.astype(int)
    adjmat_diff[(adjmat_true.astype(int) - adjmat_pred.astype(int))<0]=2
    adjmat_diff[(adjmat_true.astype(int) - adjmat_pred.astype(int))>0]=-1

    if showfig:
        # Setup graph
        #    G_true = adjmat2graph(adjmat_true)
        G_diff = adjmat2graph(adjmat_diff)
        # Graph layout
        pos = graphlayout(G_diff, pos=pos, scale=1, layout='fruchterman_reingold', verbose=verbose)
        # Bootup figure
        plt.figure(figsize=(width, height))
        # nodes
        nx.draw_networkx_nodes(G_diff, pos, node_size=700)
        # edges
        colors = [G_diff[u][v]['color'] for u, v in G_diff.edges()]
        #weights = [G_diff[u][v]['weight'] for u,v in G_diff.edges()]
        nx.draw_networkx_edges(G_diff, pos, arrowstyle='->', edge_color=colors, width=1)
        # Labels
        nx.draw_networkx_labels(G_diff, pos, font_size=20, font_family='sans-serif')
        # Get labels of weights
        #labels = nx.get_edge_attributes(G,'weight')
        # Plot weights
        nx.draw_networkx_edge_labels(G_diff, pos, edge_labels=nx.get_edge_attributes(G_diff, 'weight'))
        # Making figure nice
        # plt.legend(['Nodes','TN','FP','test'])
        ax = plt.gca()
        ax.set_axis_off()
        plt.show()

    # Return
    return(scores, adjmat_diff)

# %% Make graph layout
def graphlayout(model, pos, scale=1, layout='fruchterman_reingold', verbose=3):
    if isinstance(pos, type(None)):
        if layout=='fruchterman_reingold':
            pos = nx.fruchterman_reingold_layout(model, scale=scale, iterations=50)
        else:
            pos = nx.spring_layout(model, scale=scale, iterations=50)
    else:
        if verbose>=3: print('[bnlearn] >Existing coordinates from <pos> are used.')

    return(pos)

# %% Convert to pandas dataframe
def is_DataFrame(data, verbose=0):
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
