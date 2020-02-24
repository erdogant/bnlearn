"""Bayesian techniques for structure learning, parameter learning, inference and sampling."""
# ------------------------------------
# Name        : bnlearn.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------


# %% Libraries
import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from ismember import ismember

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling  # GibbsSampling

from pgmpy import readwrite
import bnlearn.helpers.network as network
curpath = os.path.dirname(os.path.abspath(__file__))
PATH_TO_DATA = os.path.join(curpath,'DATA')


# %% Make DAG
def make_DAG(DAG, CPD=None, checkmodel=True, verbose=3):
    """Create Directed Acyclic Graph based on list.

    Parameters
    ----------
    DAG : list
        list containing source and target in the form of [('A','B'), ('B','C')].
    CPD : list, array-like
        Containing TabularCPD for each node.
    checkmodel : bool
        Check the validity of the model. The default is True
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None
        1: ERROR
        2: WARNING
        3: INFO
        4: DEBUG
        5: TRACE

    Raises
    ------
    Exception
        Should be list.

    Returns
    -------
    pgmpy.models.BayesianModel.BayesianModel
        model of the DAG.

    """
    if (CPD is not None) and (not isinstance(CPD, list)):
        CPD=[CPD]

    if (not isinstance(DAG, list)) and ('pgmpy' not in str(type(DAG))):
        raise Exception("[BNLEARN] ERROR: Input DAG should be a list. in the form [('A','B'), ('B','C')] or a <pgmpy.models.BayesianModel.BayesianModel>")
    elif ('pgmpy' in str(type(DAG))):
        if verbose>=3: print('[BNLEARN] No changes made to existing Bayesian DAG.')
    elif isinstance(DAG, list):
        if verbose>=3: print('[BNLEARN] Bayesian DAG created.')
        DAG = BayesianModel(DAG)

    if CPD is not None:
        for cpd in CPD:
            DAG.add_cpds(cpd)
            if verbose>=3: print('[BNLEARN] Add CPD: %s' %(cpd.variable))

        if checkmodel:
            print('[BNLEARN.print_CPD] Model correct: %s' %(DAG.check_model()))

    return DAG


# %% Print DAG
def print_CPD(DAG, checkmodel=False):
    """Print DAG-model to screen.

    Parameters
    ----------
    DAG : pgmpy.models.BayesianModel.BayesianModel
        model of the DAG.
    checkmodel : bool
        Check the validity of the model. The default is True

    Returns
    -------
    None.

    """
    if isinstance(DAG, dict):
        DAG = DAG['model']

    if len(DAG.get_cpds())==0:
        print('[BNLEARN.print_CPD] No CPDs to print. Use bnlearn.plot(DAG) to make a plot.')
        return

    print('[BNLEARN.print_CPD] Independencies:\n%s' %(DAG.get_independencies()))
    print('[BNLEARN.print_CPD] Nodes: %s' %(DAG.nodes()))
    print('[BNLEARN.print_CPD] Edges: %s' %(DAG.edges()))

    for cpd in DAG.get_cpds():
        print("CPD of {variable}:".format(variable=cpd.variable))
        print(cpd)

    if checkmodel:
        print('[BNLEARN.print_CPD] Model correct: %s' %(DAG.check_model()))


# %% Make DAG
def import_DAG(filepath='sprinkler', CPD=True, verbose=3):
    """Import Directed Acyclic Graph.

    Parameters
    ----------
    filepath : str, optional
        Pre-defined examples are depicted below, or provide the absolute file path to the .bif model file.. The default is 'sprinkler'.
        'sprinkler'(default)
        'alarm'
        'andes'
        'asia'
        'pathfinder'
        'sachs'
        'miserables'
        'filepath/to/model.bif'
    CPD : bool, optional
        Directed Acyclic Graph (DAG). The default is True.
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None
        1: ERROR
        2: WARNING
        3: INFO
        4: DEBUG
        5: TRACE

    Returns
    -------
    dict.


    Examples
    --------
    >>> model = bnlearn.import_DAG('sprinkler')
    >>> bnlearn.plot(model)

    """
    out=dict()
    model=None
    filepath=filepath.lower()

    # Load data
    if filepath=='sprinkler':
        model = _DAG_sprinkler(CPD=CPD)
    elif filepath=='asia':
        model = _bif2bayesian(os.path.join(PATH_TO_DATA,'ASIA/asia.bif'), verbose=verbose)
    elif filepath=='alarm':
        model = _bif2bayesian(os.path.join(PATH_TO_DATA,'ALARM/alarm.bif'), verbose=verbose)
    elif filepath=='andes':
        model = _bif2bayesian(os.path.join(PATH_TO_DATA,'ANDES/andes.bif'), verbose=verbose)
    elif filepath=='pathfinder':
        model = _bif2bayesian(os.path.join(PATH_TO_DATA,'PATHFINDER/pathfinder.bif'), verbose=verbose)
    elif filepath=='sachs':
        model = _bif2bayesian(os.path.join(PATH_TO_DATA,'SACHS/sachs.bif'), verbose=verbose)
    elif filepath=='miserables':
        f = open(os.path.join(PATH_TO_DATA,'miserables.json'))
        data = json.loads(f.read())
        L=len(data['links'])
        edges=[(data['links'][k]['source'], data['links'][k]['target']) for k in range(L)]
        model=nx.Graph(edges, directed=False)
    else:
        if os.path.isfile(filepath):
            model = _bif2bayesian(filepath, verbose=verbose)
        else:
            if verbose>=3: print('[BNLEARN][import_DAG] Filepath does not exist! <%s>' %(filepath))
            return(out)

    # check_model check for the model structure and the associated CPD and returns True if everything is correct otherwise throws an exception
    if not isinstance(model, type(None)) and verbose>=3 and CPD:
        print_CPD(model)

    # Setup simmilarity matrix
    adjmat = pd.DataFrame(data=False, index=model.nodes(), columns=model.nodes()).astype('Bool')
    # Fill adjmat with edges
    edges=model.edges()
    for edge in edges:
        adjmat.loc[edge[0],edge[1]]=True
    adjmat.index.name='source'
    adjmat.columns.name='target'

    out['model']=model
    out['adjmat']=adjmat
    return(out)


# %%  Convert adjacency matrix to vector
def vec2adjmat(source, target, symmetric=True):
    """Convert source and target into adjacency matrix.

    Parameters
    ----------
    source : list
        The source node.
    target : list
        The target node.
    symmetric : bool, optional
        Make the adjacency matrix symmetric with the same number of rows as columns. The default is True.

    Returns
    -------
    adjacency matrix : pd.DataFrame().

    """
    # Make adjacency matrix
    adjmat = pd.crosstab(source, target)
    # Get all unique nodes
    nodes = np.unique(np.c_[adjmat.columns.values, adjmat.index.values].flatten())

    # Make the adjacency matrix symmetric
    if symmetric:
        # Add missing columns
        node_columns = np.setdiff1d(nodes, adjmat.columns.values)
        for node in node_columns:
            adjmat[node]=0

        # Add missing rows
        node_rows = np.setdiff1d(nodes, adjmat.index.values)
        adjmat=adjmat.T
        for node in node_rows:
            adjmat[node]=0
        adjmat=adjmat.T

        # Sort to make ordering of columns and rows similar
        [IA, IB] = ismember(adjmat.columns.values, adjmat.index.values)
        adjmat = adjmat.iloc[IB,:]
        adjmat.index.name='source'
        adjmat.columns.name='target'

    return(adjmat)


# %%  Convert adjacency matrix to vector
def adjmat2vec(adjmat):
    """Convert adjacency matrix into vector with source and target.

    Parameters
    ----------
    adjmat : pd.DataFrame()
        Adjacency matrix.

    Returns
    -------
    nodes that are connected based on source and target : pd.DataFrame()

    """
    # Convert adjacency matrix into vector
    adjmat = adjmat.stack().reset_index()
    # Set columns
    adjmat.columns = ['source', 'target', 'weight']
    # Remove self loops and no-connected edges
    Iloc1 = adjmat['source']!=adjmat['target']
    Iloc2 = adjmat['weight']>0
    Iloc = Iloc1 & Iloc2
    # Take only connected nodes
    adjmat = adjmat.loc[Iloc,:]
    adjmat.reset_index(drop=True, inplace=True)
    return(adjmat)


# %% Sampling from model
def sampling(model, n=1000, verbose=3):
    """Sample based on DAG.

    Parameters
    ----------
    model : dict
        Contains model and adjmat.
    n : int, optional
        Number of samples to generate. The default is 1000.
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: NONE
        1: ERROR
        2: WARNING
        3: INFO (default)
        4: DEBUG
        5: TRACE

    Returns
    -------
    df : pd.DataFrame().


    Example
    -------
    >>> import bnlearn
    >>> model = bnlearn.import_DAG('sprinkler')
    >>> df = bnlearn.sampling(model, n=1000)

    """
    assert n>0, 'n must be 1 or larger'
    assert 'BayesianModel' in str(type(model['model'])), 'Model must contain DAG from BayesianModel. Note that <misarables> example does not include DAG.'
    if verbose>=3: print('[BNLEARN][sampling] Forward sampling for %.0d samples..' %(n))

    # http://pgmpy.org/sampling.html
    inference = BayesianModelSampling(model['model'])
    # inference = GibbsSampling(model)
    # Forward sampling and make dataframe
    df=inference.forward_sample(size=n, return_type='dataframe')
    return(df)


# %% Model Sprinkler
def _DAG_sprinkler(CPD=True):
    """Create DAG-model for the sprinkler example.

    Parameters
    ----------
    CPD : bool, optional
        Directed Acyclic Graph (DAG).. The default is True.

    Returns
    -------
    model.

    """
    # Define the network structure
    model = BayesianModel([('Cloudy', 'Sprinkler'),
                           ('Cloudy', 'Rain'),
                           ('Sprinkler', 'Wet_Grass'),
                           ('Rain', 'Wet_Grass')])

    if CPD:
        # Cloudy
        cpt_cloudy = TabularCPD(variable='Cloudy', variable_card=2, values=[[0.5], [0.5]])
        # Sprinkler
        cpt_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2,
                                   values=[[0.5, 0.9], [0.5, 0.1]],
                                   evidence=['Cloudy'], evidence_card=[2])
        # Rain
        cpt_rain = TabularCPD(variable='Rain', variable_card=2,
                              values=[[0.8, 0.2], [0.2, 0.8]],
                              evidence=['Cloudy'], evidence_card=[2])
        # Wet Grass
        cpt_wet_grass = TabularCPD(variable='Wet_Grass', variable_card=2,
                                   values=[[1, 0.1, 0.1, 0.01],
                                           [0, 0.9, 0.9, 0.99]],
                                   evidence=['Sprinkler', 'Rain'],
                                   evidence_card=[2, 2])
        # Connect DAG with CPTs
        # Associating the parameters with the model structure.
        model.add_cpds(cpt_cloudy, cpt_sprinkler, cpt_rain, cpt_wet_grass)

    return(model)


# %% Convert BIF model to bayesian model
def _bif2bayesian(pathname, verbose=3):
    """Return the fitted bayesian model.

    Example
    -------
    >>> from pgmpy.readwrite import BIFReader
    >>> reader = BIFReader("bif_test.bif")
    >>> reader.get_model()
    <pgmpy.models.BayesianModel.BayesianModel object at 0x7f20af154320>
    """
    if verbose>=3: print('[BNLEARN] Loading bif file <%s>' %(pathname))

    bifmodel=readwrite.BIF.BIFReader(path=pathname)

    try:
        model = BayesianModel(bifmodel.variable_edges)
        model.name = bifmodel.network_name
        model.add_nodes_from(bifmodel.variable_names)

        tabular_cpds = []
        for var in sorted(bifmodel.variable_cpds.keys()):
            values = bifmodel.variable_cpds[var]
            cpd = TabularCPD(var, len(bifmodel.variable_states[var]), values,
                             evidence=bifmodel.variable_parents[var],
                             evidence_card=[len(bifmodel.variable_states[evidence_var])
                                            for evidence_var in bifmodel.variable_parents[var]])
            tabular_cpds.append(cpd)

        model.add_cpds(*tabular_cpds)
#        for node, properties in bifmodel.variable_properties.items():
#            for prop in properties:
#                prop_name, prop_value = map(lambda t: t.strip(), prop.split('='))
#                model.node[node][prop_name] = prop_value

        return model

    except AttributeError:
        raise AttributeError('[BNLEARN] First get states of variables, edges, parents and network names')


# %% Make directed graph from adjmatrix
def to_undirected(adjmat):
    """Transform directed adjacency matrix to undirected.

    Parameters
    ----------
    adjmat : np.array()
        Adjacency matrix.

    Returns
    -------
    Directed adjacency matrix.

    """
    num_rows=adjmat.shape[0]
    num_cols=adjmat.shape[1]
    adjmat_directed=np.zeros((num_rows, num_cols), dtype=int)
    tmpadjmat=adjmat.astype(int)

    for i in range(num_rows):
        for j in range(num_cols):
            adjmat_directed[i,j] = tmpadjmat.iloc[i,j] + tmpadjmat.iloc[j,i]

    adjmat_directed=pd.DataFrame(index=adjmat.index, data=adjmat_directed, columns=adjmat.columns, dtype=bool)
    return(adjmat_directed)


# %% Comparison of two networks
def compare_networks(model_1, model_2, pos=None, showfig=True, figsize=(15,8), verbose=3):
    """Compare networks of two models.

    Parameters
    ----------
    model_1 : dict
        Results of model 1..
    model_2 : dict
        Results of model 2..
    pos : graph, optional
        Coordinates of the network. If there are provided, the same structure will be used to plot the network.. The default is None.
    showfig : bool, optional
        plot figure. The default is True.
    figsize : tuple, optional
        Figure size.. The default is (15,8).
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: NONE
        1: ERROR
        2: WARNING
        3: INFO (default)
        4: DEBUG
        5: TRACE

    Returns
    -------
    dict.

    """
    [scores, adjmat_diff] = network.compare_networks(model_1['adjmat'], model_2['adjmat'], pos=pos, showfig=showfig, width=figsize[0], height=figsize[1], verbose=verbose)
    return(scores, adjmat_diff)


# %% PLOT
def plot(model, pos=None, scale=1, figsize=(15,8), verbose=3):
    """Plot the learned stucture.

    Parameters
    ----------
    model : dict
        Learned model from the .fit() function..
    pos : graph, optional
        Coordinates of the network. If there are provided, the same structure will be used to plot the network.. The default is None.
    scale : int, optional
        Scaling parameter for the network. A larger number will linearily increase the network.. The default is 1.
    figsize : tuple, optional
        Figure size. The default is (15,8).
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: NONE
        1: ERROR
        2: WARNING
        3: INFO (default)
        4: DEBUG
        5: TRACE

    Returns
    -------
    dict.

    """
    out=dict()
    G = nx.DiGraph()  # Directed graph
    layout='fruchterman_reingold'

    # Extract model if in dict
    if 'dict' in str(type(model)):
        model = model.get('model', None)

    # Bayesian model
    if 'BayesianModel' in str(type(model)) or 'pgmpy' in str(type(model)):
        if verbose>=3: print('[BNLEARN][plot] Making plot based on BayesianModel')
        # positions for all nodes
        pos = network.graphlayout(model, pos=pos, scale=scale, layout=layout)
        # Add directed edge with weigth
        # edges=model.edges()
        edges=[*model.edges()]
        for i in range(len(edges)):
            G.add_edge(edges[i][0], edges[i][1], weight=1, color='k')
    elif 'networkx' in str(type(model)):
        if verbose>=3: print('[BNLEARN][plot] Making plot based on networkx model')
        G=model
        pos = network.graphlayout(G, pos=pos, scale=scale, layout=layout)
    else:
        if verbose>=3: print('[BNLEARN][plot] Making plot based on adjacency matrix')
        G = network.adjmat2graph(model)
        # Convert adjmat to source target
#        df_edges=model.stack().reset_index()
#        df_edges.columns=['source', 'target', 'weight']
#        df_edges['weight']=df_edges['weight'].astype(float)
#
#        # Add directed edge with weigth
#        for i in range(df_edges.shape[0]):
#            if df_edges['weight'].iloc[i]!=0:
#                color='k' if df_edges['weight'].iloc[i]>0 else 'r'
#                G.add_edge(df_edges['source'].iloc[i], df_edges['target'].iloc[i], weight=np.abs(df_edges['weight'].iloc[i]), color=color)
        # Get positions
        pos = network.graphlayout(G, pos=pos, scale=scale, layout=layout)

    # Bootup figure
    plt.figure(figsize=figsize)
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, with_labels=True, alpha=0.85)
    # edges
    colors = [G[u][v].get('color','k') for u,v in G.edges()]
    weights = [G[u][v].get('weight',1) for u,v in G.edges()]
    nx.draw_networkx_edges(G, pos, arrowstyle='->', edge_color=colors, width=weights)
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
    # Get labels of weights
    # labels = nx.get_edge_attributes(G,'weight')
    # Plot weights
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G,'weight'))
    # Making figure nice
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()

    # Store
    out['pos']=pos
    out['G']=G
    return(out)


# %% Example data
def import_example():
    """Load sprinkler example.

    Returns
    -------
    df : pd.DataFrame()

    """
    curpath = os.path.dirname(os.path.abspath(__file__))
    PATH_TO_DATA_1 = os.path.join(curpath,'data','sprinkler_data.zip')
    if os.path.isfile(PATH_TO_DATA_1):
        df=pd.read_csv(PATH_TO_DATA_1, sep=',')
        return df
    else:
        print('[KM] Oops! Example data not found! Try to get it at: www.github.com/erdogant/bnlearn/')
        return None

# %% Convert Adjmat to graph (G)
# def adjmat2graph(adjmat):
#     G = nx.DiGraph() # Directed graph
#     # Convert adjmat to source target
#     df_edges=adjmat.stack().reset_index()
#     df_edges.columns=['source', 'target', 'weight']
#     df_edges['weight']=df_edges['weight'].astype(float)

#     # Add directed edge with weigth
#     for i in range(df_edges.shape[0]):
#         if df_edges['weight'].iloc[i]!=0:
#             # Setup color
#             if df_edges['weight'].iloc[i]==1:
#                 color='k'
#             elif df_edges['weight'].iloc[i]>1:
#                 color='r'
#             elif df_edges['weight'].iloc[i]<0:
#                 color='b'
#             else:
#                 color='p'

#             # Create edge in graph
#             G.add_edge(df_edges['source'].iloc[i], df_edges['target'].iloc[i], weight=np.abs(df_edges['weight'].iloc[i]), color=color)
#     # Return
#     return(G)
