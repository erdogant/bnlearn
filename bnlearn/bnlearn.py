"""Bayesian techniques for structure learning, parameter learning, inference and sampling."""
# ------------------------------------
# Name        : bnlearn.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------


# %% Libraries
import os
import itertools
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from ismember import ismember
from tqdm import tqdm

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling  # GibbsSampling

from pgmpy import readwrite
# import bnlearn.network as network
# from bnlearn.network import compare_networks, graphlayout, adjmat2graph
# import bnlearn.inference as inference
import bnlearn

curpath = os.path.dirname(os.path.abspath(__file__))
PATH_TO_DATA = os.path.join(curpath, 'data')


#%%  Convert adjmat to bayesian model
def to_bayesianmodel(model, verbose=3):
    """Convert adjacency matrix to BayesianModel.

    Description
    -----------
    Convert a adjacency to a BayesianModel. This is required as some of the
    functionalities, such as ``structure_learning`` output a DAGmodel.
    If the output of ``structure_learning`` is provided, the adjmat is extracted and processed.

    Parameters
    ----------
    model : pd.DataFrame()
        Adjacency matrix.

    Raises
    ------
    Exception
        The input should not be None and if a model (as dict) is provided, the key 'adjmat' should be included.

    Returns
    -------
    bayesianmodel : Object
        BayesianModel that can be used in ``parameter_learning.fit``.

    """
    if isinstance(model, dict):
        adjmat = model.get('adjmat', None)
    else:
        adjmat = model
    if adjmat is None: raise Exception('[bnlearn] >Error: input for "to_bayesianmodel" should be adjmat or a dict containing a key "adjmat".')

    if verbose>=3: print('[bnlearn] >Conversion of adjmat to BayesianModel.')

    # Convert to vector
    vec = adjmat2vec(adjmat)[['source', 'target']].values.tolist()
    # Make BayesianModel
    bayesianmodel = BayesianModel(vec)
    # Return
    return bayesianmodel


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
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

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
    if isinstance(DAG, dict):
        DAG = DAG.get('model', None)
    if (not isinstance(DAG, list)) and ('pgmpy' not in str(type(DAG))):
        raise Exception("[bnlearn] >Error: Input DAG should be a list. in the form [('A','B'), ('B','C')] or a <pgmpy.models.BayesianModel.BayesianModel>")
    elif ('pgmpy' in str(type(DAG))):
        if verbose>=3: print('[bnlearn] >No changes made to existing Bayesian DAG.')
    elif isinstance(DAG, list):
        if verbose>=3: print('[bnlearn] >Bayesian DAG created.')
        DAG = BayesianModel(DAG)

    if CPD is not None:
        for cpd in CPD:
            DAG.add_cpds(cpd)
            if verbose>=3: print('[bnlearn] >Add CPD: %s' %(cpd.variable))
        # Check model
        if checkmodel:
            _check_model(DAG, verbose=verbose)

    # Create adjacency matrix from DAG
    out = {}
    out['adjmat'] = _dag2adjmat(DAG)
    out['model'] = DAG
    return out
    # return DAG


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
    # config = None
    if isinstance(DAG, dict):
        DAG = DAG.get('model', None)

    # Print CPDs
    # if config['method']=='ml' or config['method']=='maximumlikelihood':
    try:
        if 'MaximumLikelihood' in str(type(DAG)):
            # print CPDs using Maximum Likelihood Estimators
            for node in DAG.state_names:
                print(DAG.estimate_cpd(node))
        elif 'BayesianModel' in str(type(DAG)):
            # print CPDs using Bayesian Parameter Estimation
            if len(DAG.get_cpds())==0:
                raise Exception('[bnlearn] >Error! This is a Bayesian DAG containing only edges, and no CPDs. Tip: you need to specify or learn the CPDs. Try: DAG=bn.parameter_learning.fit(DAG, df). At this point you can make a plot with: bn.plot(DAG).')
                return
            for cpd in DAG.get_cpds():
                print("CPD of {variable}:".format(variable=cpd.variable))
                print(cpd)

            print('[bnlearn] >Independencies:\n%s' %(DAG.get_independencies()))
            print('[bnlearn] >Nodes: %s' %(DAG.nodes()))
            print('[bnlearn] >Edges: %s' %(DAG.edges()))

        if checkmodel:
            _check_model(DAG, verbose=3)
    except:
        print('[bnlearn] >No CPDs to print. Tip: use bnlearn.plot(DAG) to make a plot.')


# %%
def _check_model(DAG, verbose=3):
    if verbose>=3:
        print('[bnlearn] >Checking CPDs..')
    for cpd in DAG.get_cpds():
        # print(cpd)
        if not np.all(cpd.values.sum(axis=0)==1):
            print('[bnlearn] >Warning: CPD [%s] does not add up to 1 but is: %s' %(cpd.variable, cpd.values.sum(axis=0)))
    if verbose>=3:
        print('[bnlearn] >Check for DAG structure. Correct: %s' %(DAG.check_model()))


# %% Make DAG
def import_DAG(filepath='sprinkler', CPD=True, checkmodel=True, verbose=3):
    """Import Directed Acyclic Graph.

    Parameters
    ----------
    filepath : str, (default: sprinkler)
        Pre-defined examples are depicted below, or provide the absolute file path to the .bif model file.. The default is 'sprinkler'.
        'sprinkler', 'alarm', 'andes', 'asia', 'pathfinder', 'sachs', 'miserables', 'filepath/to/model.bif',
    CPD : bool, optional
        Directed Acyclic Graph (DAG). The default is True.
    checkmodel : bool
        Check the validity of the model. The default is True
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

    Returns
    -------
    dict containing model and adjmat.
        model : BayesianModel
        adjmat : Adjacency matrix

    Examples
    --------
    >>> import bnlearn as bn
    >>> model = bn.import_DAG('sprinkler')
    >>> bn.plot(model)

    """
    out={}
    model=None
    filepath=filepath.lower()

    # Load data
    if filepath=='sprinkler':
        model = _DAG_sprinkler(CPD=CPD)
    elif filepath=='asia':
        model = _bif2bayesian(os.path.join(PATH_TO_DATA, 'ASIA/asia.bif'), verbose=verbose)
    elif filepath=='alarm':
        model = _bif2bayesian(os.path.join(PATH_TO_DATA, 'ALARM/alarm.bif'), verbose=verbose)
    elif filepath=='andes':
        model = _bif2bayesian(os.path.join(PATH_TO_DATA, 'ANDES/andes.bif'), verbose=verbose)
    elif filepath=='pathfinder':
        model = _bif2bayesian(os.path.join(PATH_TO_DATA, 'PATHFINDER/pathfinder.bif'), verbose=verbose)
    elif filepath=='sachs':
        model = _bif2bayesian(os.path.join(PATH_TO_DATA, 'SACHS/sachs.bif'), verbose=verbose)
    elif filepath=='water':
        model = _bif2bayesian(os.path.join(PATH_TO_DATA, 'WATER/water.bif'), verbose=verbose)
    elif filepath=='miserables':
        f = open(os.path.join(PATH_TO_DATA, 'miserables.json'))
        data = json.loads(f.read())
        L=len(data['links'])
        edges=[(data['links'][k]['source'], data['links'][k]['target']) for k in range(L)]
        model=nx.Graph(edges, directed=False)
    else:
        if os.path.isfile(filepath):
            model = _bif2bayesian(filepath, verbose=verbose)
        else:
            if verbose>=3: print('[bnlearn] >filepath does not exist! <%s>' %(filepath))
            return(out)

    # Setup adjacency matrix
    adjmat = _dag2adjmat(model)

    # Store
    out['model']=model
    out['adjmat']=adjmat

    # check_model check for the model structure and the associated CPD and returns True if everything is correct otherwise throws an exception
    if (model is not None) and CPD and checkmodel:
        _check_model(out['model'], verbose=verbose)
        if verbose>=4:
            print_CPD(out)

    return(out)


# %% Convert DAG into adjacency matrix
def _dag2adjmat(model, verbose=3):
    adjmat = None
    if hasattr(model, 'nodes') and hasattr(model, 'edges'):
        adjmat = pd.DataFrame(data=False, index=model.nodes(), columns=model.nodes()).astype('bool')
        # Fill adjmat with edges
        edges = model.edges()
        # Run over the edges
        for edge in edges:
            adjmat.loc[edge[0], edge[1]]=True
        adjmat.index.name='source'
        adjmat.columns.name='target'
    else:
        if verbose>=1: print('[bnlearn] >Could not convert to adjmat because nodes and/or edges were missing.')
    return(adjmat)


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
    pd.DataFrame
        adjacency matrix.

    Examples
    --------
    >>> source=['Cloudy','Cloudy','Sprinkler','Rain']
    >>> target=['Sprinkler','Rain','Wet_Grass','Wet_Grass']
    >>> vec2adjmat(source, target)

    """
    df = pd.DataFrame(np.c_[source, target], columns=['source', 'target'])
    # Make adjacency matrix
    adjmat = pd.crosstab(df['source'], df['target'])
    # Get all unique nodes
    # nodes = np.unique(np.c_[adjmat.columns.values, adjmat.index.values].flatten())
    nodes = np.unique(list(adjmat.columns.values) + list(adjmat.index.values))

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
        adjmat = adjmat.iloc[IB, :]
        adjmat.index.name='source'
        adjmat.columns.name='target'

    return(adjmat)


# %%  Convert adjacency matrix to vector
def adjmat2vec(adjmat, min_weight=1):
    """Convert adjacency matrix into vector with source and target.

    Parameters
    ----------
    adjmat : pd.DataFrame()
        Adjacency matrix.

    min_weight : float
        edges are returned with a minimum weight.

    Returns
    -------
    pd.DataFrame()
        nodes that are connected based on source and target

    Examples
    --------
    >>> source=['Cloudy','Cloudy','Sprinkler','Rain']
    >>> target=['Sprinkler','Rain','Wet_Grass','Wet_Grass']
    >>> adjmat = vec2adjmat(source, target)
    >>> vector = adjmat2vec(adjmat)

    """
    # Convert adjacency matrix into vector
    adjmat = adjmat.stack().reset_index()
    # Set columns
    adjmat.columns = ['source', 'target', 'weight']
    # Remove self loops and no-connected edges
    Iloc1 = adjmat['source']!=adjmat['target']
    Iloc2 = adjmat['weight']>=min_weight
    Iloc = Iloc1 & Iloc2
    # Take only connected nodes
    adjmat = adjmat.loc[Iloc, :]
    adjmat.reset_index(drop=True, inplace=True)
    return(adjmat)


# %% Sampling from model
def sampling(DAG, n=1000, verbose=3):
    """Generate sample(s) using forward sampling from joint distribution of the bayesian network.

    Parameters
    ----------
    DAG : dict
        Contains model and adjmat of the DAG.
    n : int, optional
        Number of samples to generate. The default is 1000.
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

    Returns
    -------
    df : pd.DataFrame().
        Dataframe containing sampled data from the input DAG model.


    Example
    -------
    >>> import bnlearn
    >>> DAG = bnlearn.import_DAG('sprinkler')
    >>> df = bnlearn.sampling(DAG, n=1000)

    """
    if n<=0: raise ValueError('n must be 1 or larger')
    if 'BayesianModel' not in str(type(DAG['model'])): raise ValueError('DAG must contain BayesianModel.')
    if verbose>=3: print('[bnlearn] >Forward sampling for %.0d samples..' %(n))

    if len(DAG['model'].get_cpds())==0:
        raise Exception('[bnlearn] >Error! This is a Bayesian DAG containing only edges, and no CPDs. Tip: you need to specify or learn the CPDs. Try: DAG=bn.parameter_learning.fit(DAG, df). At this point you can make a plot with: bn.plot(DAG).')
        return

    # http://pgmpy.org/sampling.html
    infer_model = BayesianModelSampling(DAG['model'])
    # inference1 = GibbsSampling(model['model'])
    # Forward sampling and make dataframe
    df=infer_model.forward_sample(size=n, return_type='dataframe')
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
    if verbose>=3: print('[bnlearn] >Loading bif file <%s>' %(pathname))

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
        raise AttributeError('[bnlearn] >First get states of variables, edges, parents and network names')


# %% Make directed graph from adjmatrix
def to_undirected(adjmat):
    """Transform directed adjacency matrix to undirected.

    Parameters
    ----------
    adjmat : np.array()
        Adjacency matrix.

    Returns
    -------
    Directed adjacency matrix : pd.DataFrame()
        Converted adjmat with undirected edges.

    """
    num_rows=adjmat.shape[0]
    num_cols=adjmat.shape[1]
    adjmat_directed=np.zeros((num_rows, num_cols), dtype=int)
    tmpadjmat=adjmat.astype(int)

    for i in range(num_rows):
        for j in range(num_cols):
            adjmat_directed[i, j] = tmpadjmat.iloc[i, j] + tmpadjmat.iloc[j, i]

    adjmat_directed=pd.DataFrame(index=adjmat.index, data=adjmat_directed, columns=adjmat.columns, dtype=bool)
    return(adjmat_directed)


# %% Comparison of two networks
def compare_networks(model_1, model_2, pos=None, showfig=True, figsize=(15, 8), verbose=3):
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
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

    Returns
    -------
    tuple containing (scores, adjmat_diff)
        scores : Score of differences between the two input models.
        adjmat_diff : Adjacency matrix depicting the differences between the two input models.

    """
    scores, adjmat_diff = bnlearn.network.compare_networks(model_1['adjmat'], model_2['adjmat'], pos=pos, showfig=showfig, width=figsize[0], height=figsize[1], verbose=verbose)
    return(scores, adjmat_diff)


# %% PLOT
def plot(model, pos=None, scale=1, figsize=(15, 8), verbose=3):
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
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

    Returns
    -------
    dict containing pos and G
        pos : list
            Positions of the nodes.
        G : Graph
            Graph model

    """
    out = {}
    G = nx.DiGraph()  # Directed graph
    layout='fruchterman_reingold'

    # Extract model if in dict
    if 'dict' in str(type(model)):
        model = model.get('model', None)

    # Bayesian model
    if 'BayesianModel' in str(type(model)) or 'pgmpy' in str(type(model)):
        if verbose>=3: print('[bnlearn] >Plot based on BayesianModel')
        # positions for all nodes
        pos = bnlearn.network.graphlayout(model, pos=pos, scale=scale, layout=layout, verbose=verbose)
        # Add directed edge with weigth
        # edges=model.edges()
        edges=[*model.edges()]
        for i in range(len(edges)):
            G.add_edge(edges[i][0], edges[i][1], weight=1, color='k')
    elif 'networkx' in str(type(model)):
        if verbose>=3: print('[bnlearn] >Plot based on networkx model')
        G = model
        pos = graphlayout(G, pos=pos, scale=scale, layout=layout, verbose=verbose)
    else:
        if verbose>=3: print('[bnlearn] >Plot based on adjacency matrix')
        G = adjmat2graph(model)
        # Get positions
        pos = graphlayout(G, pos=pos, scale=scale, layout=layout, verbose=verbose)

    # Bootup figure
    plt.figure(figsize=figsize)
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.85)
    # edges
    colors = [G[u][v].get('color', 'k') for u, v in G.edges()]
    weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, arrowstyle='->', edge_color=colors, width=weights)
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
    # Get labels of weights
    # labels = nx.get_edge_attributes(G,'weight')
    # Plot weights
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    # Making figure nice
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()

    # Store
    out['pos']=pos
    out['G']=G
    return(out)

# %%
def topological_sort(adjmat, start=None):
    """Topological sort.
    
    Description
    -----------
    Get nodes list in the topological sort order.

    Parameters
    ----------
    adjmat : pd.DataFrame or bnlearn object.
        Adjacency matrix.
    start : str, optional
        Start position. The default is None and the whole network is examined.

    Returns
    -------
    list
        Topological sort order.

    Example
    -----------
    import bnlearn as bn
    DAG = bn.import_DAG('sprinkler', verbose=0)
    bn.topological_sort(DAG, 'Rain')
    bn.topological_sort(DAG)


    References
    ----------
    https://stackoverflow.com/questions/47192626/deceptively-simple-implementation-of-topological-sorting-in-python

    """
    # Convert to adjmat
    if isinstance(adjmat, dict):
        adjmat = adjmat.get('adjmat', None)
    elif np.all(np.isin(adjmat.columns, ['source','target','weight'])):
        adjmat = vec2adjmat(adjmat['source'], adjmat['target'])

    # Convert to graph
    graph = adjmat2dict(adjmat)
    # Do the topological sort
    seen = set()
    stack = []    # path variable is gone, stack and order are new
    order = []    # order will be in reverse order at first
    if start is None:
        q = list(graph)
    else:
        q = [start]
    while q:
        v = q.pop()
        if v not in seen:
            seen.add(v) # no need to append to path any more
            q.extend(graph[v])

            while stack and v not in graph[stack[-1]]: # new stuff here!
                order.append(stack.pop())
            stack.append(v)

    return stack + order[::-1]


# %%
def adjmat2dict(adjmat):
    """Convert adjacency matrix to dict.

    Parameters
    ----------
    adjmat : pd.DataFrame
        Adjacency matrix.

    Returns
    -------
    graph : dict
        Graph.

    """
    graph={}
    rows=adjmat.index.values
    for r in rows:
        graph.update({r: list(rows[adjmat.loc[r,:]])})
    return graph


# %% Example data
def import_example(data='sprinkler', n=10000, verbose=3):
    """Load example dataset.

    Parameters
    ----------
    data : str, (default: sprinkler)
        Pre-defined examples.
        'titanic', 'sprinkler', 'alarm', 'andes', 'asia', 'pathfinder', 'sachs', 'water'
    n : int, optional
        Number of samples to generate. The default is 1000.
    verbose : int, (default: 3)
        Print progress to screen.
        0: None, 1: ERROR, 2: WARN, 3: INFO, 4: DEBUG, 5: TRACE

    Returns
    -------
    df : pd.DataFrame()

    """
    import wget

    url = 'https://erdogant.github.io/datasets/'
    if data=='sprinkler':
        url=url + 'sprinkler.zip'
    elif data=='titanic':
        url=url + 'titanic_train.zip'
    else:
        try:
            DAG = import_DAG(data, verbose=2)
            df = sampling(DAG, n=n, verbose=2)
        except:
            print('[bnlearn] >Error: Loading data not possible!')
            df = None
        return df

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    PATH_TO_DATA = os.path.join(curpath, wget.filename_from_url(url))
    if not os.path.isdir(curpath):
        os.mkdir(curpath)

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        if verbose>=3: print('[bnlearn] >Downloading example dataset..')
        wget.download(url, curpath)

    # Import local dataset
    if verbose>=3: print('[bnlearn] >Import dataset..')
    df = pd.read_csv(PATH_TO_DATA)
    return df


# %% Pre-processing of input raw dataset
def df2onehot(df, y_min=10, perc_min_num=0.8, dtypes='pandas', excl_background=None, verbose=3):
    """Convert dataframe to one-hot matrix.

    Parameters
    ----------
    df : pd.DataFrame()
        Input dataframe for which the rows are the features, and colums are the samples.
    dtypes : list of str or 'pandas', optional
        Representation of the columns in the form of ['cat','num']. By default the dtype is determiend based on the pandas dataframe.
    y_min : int [0..len(y)], optional
        Minimal number of sampels that must be present in a group. All groups with less then y_min samples are labeled as _other_ and are not used in the enriching model. The default is None.
    perc_min_num : float [None, 0..1], optional
        Force column (int or float) to be numerical if unique non-zero values are above percentage. The default is None. Alternative can be 0.8
    verbose : int, optional
        Print message to screen. The default is 3.
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

    Returns
    -------
    pd.DataFrame()
        One-hot dataframe.

    """
    from df2onehot import df2onehot as df2hot

    # Convert dataframe to onehot by keeping only categorical variables.
    out = df2hot(df, y_min=y_min, perc_min_num=perc_min_num, dtypes=dtypes, excl_background=excl_background, hot_only=True, verbose=verbose)
    # Numerical
    df_num = out['numeric'].iloc[:, out['dtypes']=='cat']
    df_num = df_num.astype(int)
    # One-hot
    df_hot = out['onehot']
    df_hot.columns = df_hot.columns.str.replace('_4.0', '_4')
    df_hot.columns = df_hot.columns.str.replace('_3.0', '_3')
    df_hot.columns = df_hot.columns.str.replace('_2.0', '_2')
    df_hot.columns = df_hot.columns.str.replace('_1.0', '_1')
    df_hot.columns = df_hot.columns.str.replace('_0.0', '_0')

    return df_hot, df_num


def _filter_df(adjmat, df, verbose=3):
    """Adjacency matrix and dataframe columns are checked for consistency."""
    remcols = df.columns[~np.isin(df.columns.values, adjmat.columns.values)].values
    if len(remcols)>0:
        if verbose>=3: print('[bnlearn] >Removing columns from dataframe to make consistent with DAG [%s]' %(remcols))
        df.drop(labels=remcols, axis=1, inplace=True)
    return df


# %% Make prediction in inference model
def predict(model, df, variables, to_df=True, verbose=3):
    """Predict on data from a Bayesian network.
    
    Description
    -----------
    The inference on the dataset is performed sample-wise by using all the available nodes as evidence (obviously, with the exception of the node whose values we are predicting).
    The states with highest probability are returned.

    Parameters
    ----------
    model : Object
        An object of class from bn.fit.
    df : pd.DataFrame
        Each row in the DataFrame will be predicted
    variables : str or list of str
        The label(s) of node(s) to be predicted.
    to_df : Bool, (default is True)
        The output is converted to dataframe output. Note that this heavily impacts the speed. 
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

    Returns
    -------
    P : dict or DataFrame
        Predict() returns a dict with the evidence and states that resulted in the highest probability for the input variable.
    
    Examples
    --------
    >>> import bnlearn as bn
    >>> model = bn.import_DAG('sprinkler')
    >>>
    >>> # Make single inference
    >>> query = bn.inference.fit(model, variables=['Rain', 'Cloudy'], evidence={'Wet_Grass':1})
    >>> print(query)
    >>> print(query2df(query))
    >>> 
    >>> # Lets create an example dataset with 100 samples and make inferences on the entire dataset.
    >>> df = bn.sampling(model, n=1000)
    >>>
    >>> # Each sample will be assesed and the states with highest probability are returned.
    >>> Pout = bn.predict(model, df, variables=['Rain', 'Cloudy'])
    >>>
    >>> print(Pout)
    >>> #     Cloudy  Rain         p
    >>> # 0        0     0  0.647249
    >>> # 1        0     0  0.604230
    >>> # ..     ...   ...       ...
    >>> # 998      0     0  0.604230
    >>> # 999      1     1  0.878049

    """
    if not isinstance(df, pd.DataFrame): raise Exception('[bnlearn] >Error: Input requires a pd.DataFrame.')
    if not isinstance(model, dict): raise Exception('[bnlearn] >Error: Input requires a dict that contains the key: model.')
    if isinstance(variables, str): variables=[variables]
    # Remove columns that are used as priors
    dfX = df.loc[:, ~np.isin(df.columns.values, variables)]
    if verbose>=3: print('[bnlearn]> Remaining columns for inference: %d' %(dfX.shape[1]))

    # Get only the unique records in the DataFrame to reduce computation time.
    dfU = dfX.drop_duplicates()
    # Make empty array
    P = np.array([None]*dfX.shape[0])
    for i in tqdm(range(dfU.shape[0])):
        # Get input data and create a dict.
        evidence = dfU.iloc[i,:].to_dict()
        # Do the inference.
        query = bnlearn.inference.fit(model, variables=variables, evidence=evidence, to_df=False, verbose=0)
        # Find original location of the input data.
        loc = np.sum((dfX==dfU.iloc[i,:]).values, axis=1)==dfU.shape[1]
        # Store inference
        P[loc] = _get_max_prob(query)
    P = list(P)

    # Loop the dataframe
    # P1 = []
    # for i in tqdm(range(dfX.shape[0])):
    #     # Setup input data
    #     evidence = dfX.iloc[i,:].to_dict()
    #     # Do the inferemce
    #     query = inference.fit(model, variables=variables, evidence=evidence, to_df=False, verbose=0)
    #     # Store in list
    #     P1.append(_get_max_prob(query))

    if to_df: P = pd.DataFrame(P)
    return P


# %% 
def _get_max_prob(query):
    # Setup all combinations
    allcomb = list(itertools.product([0, 1], repeat=len(query.variables)))
    # Get highest P-value and gather data
    idx = np.argmax(query.values.flatten())
    comb = allcomb[idx]
    p = query.values.flatten()[idx]
    # Store in dict
    out = dict(zip(query.variables, comb))
    out['p']=p
    return out


# %%
def query2df(query):
    """Convert query from inference model to a dataframe.

    Parameters
    ----------
    query : Object from the inference model.
        Convert query object to a dataframe.

    Returns
    -------
    df : pd.DataFrame()
        Dataframe with inferences.

    """
    df = pd.DataFrame(data = list(itertools.product([0, 1], repeat=len(query.variables))), columns=query.variables)
    df['p'] = query.values.flatten()
    return df

# %% Comparison of two networks
# def _compare_networks(adjmat_true, adjmat_pred, pos=None, showfig=True, width=15, height=8, verbose=3):
#     # Make sure columns and indices to match
#     [IArow,IBrow]=ismember(adjmat_true.index.values, adjmat_pred.index.values)
#     [IAcol,IBcol]=ismember(adjmat_true.columns.values, adjmat_pred.columns.values)
#     adjmat_true = adjmat_true.loc[IArow,IAcol]
#     adjmat_pred = adjmat_pred.iloc[IBrow,IBcol]
    
#     # Make sure it is boolean adjmat
#     adjmat_true = adjmat_true>0
#     adjmat_pred = adjmat_pred>0

#     # Check whether order is correct
#     assert np.all(adjmat_true.columns.values==adjmat_pred.columns.values), 'Column order of both input values could not be matched'
#     assert np.all(adjmat_true.index.values==adjmat_pred.index.values), 'Row order of both input values could not be matched'
    
#     # Get edges
#     y_true = adjmat_true.stack().reset_index()[0].values
#     y_pred = adjmat_pred.stack().reset_index()[0].values
    
#     # Confusion matrix
#     scores=confmatrix.twoclass(y_true, y_pred, threshold=0.5, classnames=['Disconnected','Connected'], title='', cmap=plt.cm.Blues, showfig=1, verbose=0)
#     #bayes.plot(out_bayes['adjmat'], pos=G['pos'])
    
#     # Setup graph
#     adjmat_diff = adjmat_true.astype(int)
#     adjmat_diff[(adjmat_true.astype(int) - adjmat_pred.astype(int))<0]=2
#     adjmat_diff[(adjmat_true.astype(int) - adjmat_pred.astype(int))>0]=-1
    
#     if showfig:
#         # Setup graph
#     #    G_true = adjmat2graph(adjmat_true)
#         G_diff = adjmat2graph(adjmat_diff)
#         # Graph layout
#         pos = graphlayout(G_diff, pos=pos, scale=1, layout='fruchterman_reingold', verbose=verbose)
#         # Bootup figure
#         plt.figure(figsize=(width,height))
#         # nodes
#         nx.draw_networkx_nodes(G_diff, pos, node_size=700)
#         # edges
#         colors  = [G_diff[u][v]['color'] for u,v in G_diff.edges()]
#         #weights = [G_diff[u][v]['weight'] for u,v in G_diff.edges()]
#         nx.draw_networkx_edges(G_diff, pos, arrowstyle='->', edge_color=colors, width=1)
#         # Labels
#         nx.draw_networkx_labels(G_diff, pos, font_size=20, font_family='sans-serif')
#         # Get labels of weights
#         #labels = nx.get_edge_attributes(G,'weight')
#         # Plot weights
#         nx.draw_networkx_edge_labels(G_diff, pos, edge_labels=nx.get_edge_attributes(G_diff,'weight'))
#         # Making figure nice
#         #plt.legend(['Nodes','TN','FP','test'])
#         ax = plt.gca()
#         ax.set_axis_off()
#         plt.show()
    
#     # Return
#     return(scores, adjmat_diff)

# # Make graph layout
# def graphlayout(model, pos, scale=1, layout='fruchterman_reingold', verbose=3):
#     if isinstance(pos, type(None)):
#         if layout=='fruchterman_reingold':
#             pos = nx.fruchterman_reingold_layout(model, scale=scale, iterations=50)
#         else:
#             pos = nx.spring_layout(model, scale=scale, iterations=50)
#     else:
#         if verbose>=3: print('[bnlearn] >Existing coordinates from <pos> are used.')

#     return(pos)

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
