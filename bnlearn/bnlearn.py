"""Bayesian techniques for structure learning, parameter learning, inference and sampling."""
# ------------------------------------
# Name        : bnlearn.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------


# %% Libraries
import os
import wget
import zipfile
import itertools
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from pgmpy.models import BayesianModel, NaiveBayes
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling  # GibbsSampling
from pgmpy import readwrite

from ismember import ismember
import pypickle
import bnlearn


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
def make_DAG(DAG, CPD=None, methodtype='bayes', checkmodel=True, verbose=3):
    """Create Directed Acyclic Graph based on list.

    Parameters
    ----------
    DAG : list
        list containing source and target in the form of [('A','B'), ('B','C')].
    CPD : list, array-like
        Containing TabularCPD for each node.
    methodtype : str (default: 'bayes')
        * 'bayes': Bayesian model
        * 'naivebayes': Special case of Bayesian Model where the only edges in the model are from the feature variables to the dependent variable. Or in other words, each tuple should start with the same variable name such as: edges = [('A', 'B'), ('A', 'C'), ('A', 'D')]
    checkmodel : bool
        Check the validity of the model. The default is True
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

    Returns
    -------
    dict keys:
        * 'adjmat': Adjacency matrix
        * 'model': pgmpy.models
        * 'methodtype': methodtype
        * 'model_edges': Edges

    Examples
    --------
    >>> import bnlearn as bn
    >>> edges = [('A', 'B'), ('A', 'C'), ('A', 'D')]
    >>> DAG = bn.make_DAG(edges, methodtype='naivebayes')
    >>> bn.plot(DAG)

    """
    if (CPD is not None) and (not isinstance(CPD, list)):
        CPD=[CPD]

    if isinstance(DAG, dict):
        DAG = DAG.get('model', None)

    if (not isinstance(DAG, list)) and ('pgmpy' not in str(type(DAG))):
        raise Exception("[bnlearn] >Error: Input DAG should be a list. in the form [('A','B'), ('B','C')] or a <pgmpy.models.BayesianModel.BayesianModel>")
    elif ('pgmpy' in str(type(DAG))):
        # Extract methodtype from existing model.
        if ('bayesianmodel' in str(type(DAG)).lower()):
            methodtype='bayes'
        elif('naivebayes' in str(type(DAG)).lower()):
            methodtype='naivebayes'
        if verbose>=3: print('[bnlearn] >No changes made to existing %s DAG.' %(methodtype))
    elif isinstance(DAG, list) and methodtype=='naivebayes':
        if verbose>=3: print('[bnlearn] >%s DAG created.' %(methodtype))
        edges=DAG
        DAG = NaiveBayes()
        DAG.add_edges_from(edges)
        # modeel.add_nodes_from(DAG)
    elif isinstance(DAG, list) and methodtype=='bayes':
        if verbose>=3: print('[bnlearn] >%s DAG created.' %(methodtype))
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
    out['methodtype'] = methodtype
    out['model_edges'] = DAG.edges()
    return out


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
        elif ('bayesianmodel' in str(type(DAG)).lower()) or ('naivebayes' in str(type(DAG)).lower()):
            # print CPDs using Bayesian Parameter Estimation
            if len(DAG.get_cpds())==0:
                raise Exception('[bnlearn] >Error! This is a Bayesian DAG containing only edges, and no CPDs. Tip: you need to specify or learn the CPDs. Try: DAG=bn.parameter_learning.fit(DAG, df). At this point you can make a plot with: bn.plot(DAG).')
                return
            for cpd in DAG.get_cpds():
                print("CPD of {variable}:".format(variable=cpd.variable))
                print(cpd)
            if ('bayesianmodel' in str(type(DAG)).lower()):
                print('[bnlearn] >Independencies:\n%s' %(DAG.get_independencies()))
            print('[bnlearn] >Nodes: %s' %(DAG.nodes()))
            print('[bnlearn] >Edges: %s' %(DAG.edges()))

        if checkmodel:
            _check_model(DAG, verbose=3)
    except:
        print('[bnlearn] >No CPDs to print. Hint: Add CPDs as following: <bn.make_DAG(DAG, CPD=[cpd_A, cpd_B, etc])> and use bnlearn.plot(DAG) to make a plot.')


# %%
def _check_model(DAG, verbose=3):
    if verbose>=3: print('[bnlearn] >Checking CPDs..')
    for cpd in DAG.get_cpds():
        # print(cpd)
        if not np.all(cpd.values.sum(axis=0)==1):
            print('[bnlearn] >Warning: CPD [%s] does not add up to 1 but is: %s' %(cpd.variable, cpd.values.sum(axis=0)))
    if verbose>=3:
        print('[bnlearn] >Check for DAG structure. Correct: %s' %(DAG.check_model()))


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
    df=infer_model.forward_sample(size=n)
    return(df)


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


# %%
def query2df(query, variables=None):
    """Convert query from inference model to a dataframe.

    Parameters
    ----------
    query : Object from the inference model.
        Convert query object to a dataframe.
    variables : list
        Order or select variables.

    Returns
    -------
    df : pd.DataFrame()
        Dataframe with inferences.

    """
    df = pd.DataFrame(data = list(itertools.product(np.arange(0, len(query.values)), repeat=len(query.variables))), columns=query.variables)
    df['p'] = query.values.flatten()
    # Order or filter on input variables
    if variables is not None:
        # Add Pvalue column
        variables=variables+['p']
        df = df[variables]
    return df


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
        Results of model 1.
    model_2 : dict
        Results of model 2.
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


# %% Get node properties
def get_node_properties(model, node_color='#1f456e', node_size=None, verbose=3):
    # https://networkx.org/documentation/networkx-1.7/reference/generated/networkx.drawing.nx_pylab.draw_networkx_nodes.html
    nodes = {}
    defaults={'node_color':node_color, 'node_size': node_size}
    adjmat = model.get('adjmat', None)

    if adjmat is not None:
        if verbose>=3: print('[bnlearn]> Set node properties.')
        # For each node, use the default node properties.
        for node in adjmat.columns:
            node_property = defaults.copy()
            nodes.update({node: node_property})

    # Return dict with node properties
    return nodes


# %% Get node properties
def get_edge_properties(model, color='#000000', weight=1, verbose=3):
    """Collect edge properties.

    Parameters
    ----------
    model : dict
        dict containing (initialized) model.
    color : str, (Default: '#000000')
        The default color of the edges.
    weight : float, (Default: 1)
        The default weight of the edges. 
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

    Returns
    -------
    edges : dict.
        Edge properties.
    
    Examples
    --------
    >>> # Example 1
    >>> import bnlearn as bn
    >>> edges = [('A', 'B'), ('A', 'C'), ('A', 'D')]
    >>> # Create DAG and store in model
    >>> model = bn.make_DAG(edges)
    >>> edge_properties = bn.get_edge_properties(model)
    >>> # Adjust the properties
    >>> edge_properties[('A', 'B')]['weight']=10
    >>> edge_properties[('A', 'B')]['color']='#8A0707'
    >>> # Make plot
    >>> bn.plot(model, interactive=False, edge_properties=edge_properties)

    """
    # https://networkx.org/documentation/networkx-1.7/reference/generated/networkx.drawing.nx_pylab.draw_networkx_nodes.html
    edges = {}
    defaults = {'color':color, 'weight':weight}
    adjmat = model.get('adjmat', None)
    # Get model edges
    model_edges = model['model'].edges() if (model.get('model_edges', None) is None) else model['model_edges']

    # Store edge properties
    if adjmat is not None:
        if verbose>=3: print('[bnlearn]> Set edge properties.')
        # For each edge, use the default properties.
        for u, v in model_edges:
            edge_property = defaults.copy()
            # Use the edge weight from the adjmat
            if not isinstance(adjmat.loc[u,v], np.bool_):
                edge_property['weight']=adjmat.loc[u,v]
            # Update edges dict
            edges.update({(u,v): edge_property})

    # Return dict with node properties
    return edges


# %% PLOT
def plot(model,
         pos=None,
         scale=1,
         interactive=False,
         title='bnlearn causal network',
         node_color=None,
         node_size=None,
         node_properties=None,
         edge_properties=None,
         params_interactive={'height':'800px', 'width':'70%', 'notebook':False, 'layout':None, 'font_color': False, 'bgcolor':'#ffffff'},
         params_static={'width':15, 'height':8, 'font_size':14, 'font_family':'sans-serif', 'alpha':0.8, 'node_shape':'o', 'layout':'fruchterman_reingold', 'font_color': '#000000', 'facecolor':'white', 'edge_alpha':0.8, 'arrowstyle':'-|>', 'arrowsize':30},
         verbose=3):
    """Plot the learned stucture.

    Parameters
    ----------
    model : dict
        Learned model from the .fit() function.
    pos : graph, optional
        Coordinates of the network. If there are provided, the same structure will be used to plot the network.. The default is None.
    scale : int, optional
        Scaling parameter for the network. A larger number will linearily increase the network.. The default is 1.
    interactive : Bool, (default: True)
        True: Interactive web-based graph.
        False: Static plot
    title : str, optional
        Title for the plots.
    node_color : str, optional
        Color each node in the network using a hex-color, such as '#8A0707' 
    node_size : int, optional
        Set the node size for each node in the network. The default size when using static plolts is 800, and for interactive plots it is 10. 
    node_properties : dict (default: None).
        Dictionary containing custom node_color and node_size parameters for the network.
        The node properties can easily be retrieved using the function: node_properties = bn.get_node_properties(model)
        node_properties = {'node1':{'node_color':'#8A0707','node_size':10},
                           'node2':{'node_color':'#000000','node_size':30}}
    edge_properties : dict (default: None).
        Dictionary containing custom node_color and node_size parameters for the network.
        The edge properties can easily be retrieved using the function: edge_properties = bn.get_edge_properties(model)
    params_interactive : dict.
        Dictionary containing various settings in case of creating interactive plots.
    params_static : dict.
        Dictionary containing various settings in case of creating static plots.
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None, 1: Error, 2: Warning, 3: Info (default), 4: Debug, 5: Trace

    Returns
    -------
    dict containing pos and G
        pos : list.
            Positions of the nodes.
        G : Graph.
            Graph model
        node_properties: dict.
            Node properties.

    Examples
    --------
    >>> import bnlearn as bn
    >>>
    >>> # Load asia DAG
    >>> df = bn.import_example(data='asia')
    >>>
    >>> # Structure learning of sampled dataset
    >>> model = bn.structure_learning.fit(df)
    >>>
    >>> # plot static
    >>> G = bn.plot(model)
    >>>
    >>> # plot interactive
    >>> G = bn.plot(model, interactive=True)
    >>>
    >>> # plot interactive with various settings
    >>> bn.plot(model, node_color='#8A0707', node_size=35, interactive=True, params_interactive = {'height':'800px', 'width':'70%', 'layout':None, 'bgcolor':'#0f0f0f0f'})
    >>>
    >>> # plot with node properties
    >>> node_properties = bn.get_node_properties(model)
    >>> # Make some changes
    >>> node_properties['xray']['node_color']='#8A0707'
    >>> node_properties['xray']['node_size']=50
    >>> # Plot
    >>> bn.plot(model, interactive=True, node_properties=node_properties)
    >>>

    """
    # Plot properties
    defaults = {'height':'800px', 'width':'70%', 'notebook':False, 'layout':None, 'font_color': False, 'bgcolor':'#ffffff', 'directed':True}
    params_interactive = {**defaults, **params_interactive}
    defaults = {'height':8, 'width':15, 'font_size':14, 'font_family':'sans-serif', 'alpha':0.8, 'layout':'fruchterman_reingold', 'font_color': 'k', 'facecolor':'#ffffff', 'node_shape':'o', 'edge_alpha':0.8, 'arrowstyle':'-|>', 'arrowsize':30}
    params_static = {**defaults, **params_static}
    out = {}
    G = nx.DiGraph()  # Directed graph
    node_size_default = 10 if interactive else 800
    if (node_properties is not None) and (node_size is not None):
        if verbose>=2: print('[bnlearn]> Warning: if both "node_size" and "node_properties" are used, "node_size" will be used.')

    # Get node properties
    if node_properties is None:
        node_properties = bnlearn.get_node_properties(model, node_size=node_size_default)
    if edge_properties is None:
        edge_properties = bnlearn.get_edge_properties(model)

    # Set default node size based on interactive True/False
    for key in node_properties.keys():
        if node_properties[key]['node_size'] is None:
            node_properties[key]['node_size']=node_size_default

    # Extract model if in dict
    if 'dict' in str(type(model)):
        bnmodel = model.get('model', None)
    else:
        bnmodel = model.copy()

    # Bayesian model
    if ('bayes' in str(type(bnmodel)).lower()) or ('pgmpy' in str(type(bnmodel)).lower()):
        if verbose>=3: print('[bnlearn] >Plot based on Bayesian model')
        # positions for all nodes
        pos = bnlearn.network.graphlayout(bnmodel, pos=pos, scale=scale, layout=params_static['layout'], verbose=verbose)
    elif 'networkx' in str(type(bnmodel)):
        if verbose>=3: print('[bnlearn] >Plot based on networkx model')
        G = bnmodel
        pos = bnlearn.network.graphlayout(G, pos=pos, scale=scale, layout=params_static['layout'], verbose=verbose)
    else:
        if verbose>=3: print('[bnlearn] >Plot based on adjacency matrix')
        G = bnlearn.network.adjmat2graph(model['adjmat'])
        # Get positions
        pos = bnlearn.network.graphlayout(G, pos=pos, scale=scale, layout=params_static['layout'], verbose=verbose)

    # get node properties
    nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights = _plot_properties(G, bnmodel, node_properties, edge_properties, node_color, node_size)

    # Plot
    if interactive:
        # Make interactive plot
        _plot_interactive(model, params_interactive, nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights, title, verbose=verbose)
    else:
        # Make static plot
        _plot_static(model, params_static, nodelist, node_colors, node_sizes, G, pos, edge_colors, edge_weights)

    # Store
    out['pos']=pos
    out['G']=G
    out['node_properties']=node_properties
    out['edge_properties']=edge_properties
    return(out)


# %% Plot interactive
# def _plot_static(model, params_static, nodelist, node_colors, node_sizes, title, verbose=3):
def _plot_static(model, params_static, nodelist, node_colors, node_sizes, G, pos, edge_color, edge_weights):
    # Bootup figure
    plt.figure(figsize=(params_static['width'], params_static['height']), facecolor=params_static['facecolor'])
    # nodes
    nx.draw_networkx_nodes(G, pos, nodelist=nodelist , node_size=node_sizes, alpha=params_static['alpha'], node_color=node_colors, node_shape=params_static['node_shape'])
    # edges
    # nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=30, edge_color=edge_color, width=edge_weights)
    nx.draw_networkx_edges(G, pos, arrowstyle=params_static['arrowstyle'], arrowsize=params_static['arrowsize'], edge_color=edge_color, width=edge_weights, alpha=params_static['edge_alpha'])
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=params_static['font_size'], font_family=params_static['font_family'], font_color=params_static['font_color'])
    # Plot text of the weights
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), font_color=params_static['font_color'])
    # Making figure nice
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()


# %% Plot interactive
def _plot_interactive(model, params_interactive, nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights, title, verbose=3):
    try:
        from pyvis import network as net
        from IPython.core.display import display, HTML
    except ModuleNotFoundError:
        if verbose>=1: raise Exception('[bnlearn] >"pyvis" module is not installed. Please pip install first: "pip install pyvis"')
    # Convert adjacency matrix into Networkx Graph
    G = bnlearn.network.adjmat2graph(model['adjmat'])
    # Setup of the interactive network figure
    g = net.Network(**params_interactive)
    # Convert from graph G
    g.from_nx(G)
    # Nodes
    for i,_ in enumerate(g.nodes):
        g.nodes[i]['color']=node_colors[np.where(nodelist==g.nodes[i].get('label'))[0][0]]
        g.nodes[i]['size']=node_sizes[np.where(nodelist==g.nodes[i].get('label'))[0][0]]

    # Edges
    g_edges = list(map(lambda x: (x.get('from'), x.get('to')), g.edges))
    for i,_ in enumerate(g.edges):
        idx = np.where(list(map(lambda x: g_edges[i]==x, edgelist)))[0][0]
        g.edges[i]['color']=edge_colors[idx]
        g.edges[i]['weight']=edge_weights[idx]

    # Create advanced buttons
    g.show_buttons(filter_=['physics'])
    # Display
    filename = title.strip().replace(' ','_') + '.html'
    g.show(filename)
    display(HTML(filename))
    # webbrowser.open('bnlearn.html')


# %% Plot properties
def _plot_properties(G, bnmodel, node_properties, edge_properties, node_color, node_size):
    # Set edge properties in Graph G
    edges=[*bnmodel.edges()]
    for edge in edges:
        color = edge_properties.get((edge[0], edge[1])).get('color', '#000000')
        weight = edge_properties.get((edge[0], edge[1])).get('weight', 1)
        G.add_edge(edge[0], edge[1], weight=weight, color=color)
        # arrowstyle = edge_properties.get((edge[0], edge[1])).get('arrowstyle', '-|>')
        # arrowsize = edge_properties.get((edge[0], edge[1])).get('arrowsize', 30)
        # G.add_edge(edge[0], edge[1], weight=weight, color=color, arrowstyle=arrowstyle, arrowsize=arrowsize)
    
    edgelist = list(G.edges())
    edge_colors = [G[u][v].get('color') for u, v in G.edges()]
    edge_weights = [G[u][v].get('weight') for u, v in G.edges()]
    # edge_arrowstyles = [G[u][v].get('arrowstyle') for u, v in G.edges()]
    # edge_arrowsizes = [G[u][v].get('arrowsize') for u, v in G.edges()]

    # Node properties
    nodelist = np.unique(edgelist)
    node_colors = []
    node_sizes = []
    for node in nodelist:
        if node_color is not None:
            node_colors.append(node_color)
        else:
            node_colors.append(node_properties[node].get('node_color'))
        if node_size is not None:
            node_sizes.append(node_size)
        else:
            node_sizes.append(node_properties[node].get('node_size'))
    # Return
    return nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights


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


# %% Example data
def import_example(data='sprinkler', n=10000, verbose=3):
    """Load example dataset.

    Parameters
    ----------
    data : str, (default: sprinkler)
        Pre-defined examples.
        'titanic', 'sprinkler', 'alarm', 'andes', 'asia', 'sachs', 'water', 'random'
    n : int, optional
        Number of samples to generate. The default is 1000.
    verbose : int, (default: 3)
        Print progress to screen.
        0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace

    Returns
    -------
    df : pd.DataFrame()

    """

    if data=='random':
        return pd.DataFrame(np.random.randint(low=0, high=2, size=(n, 5)), columns=['A', 'B', 'C', 'D', 'E'])

    # Change name for downloading
    if data=='titanic': data = 'titanic_train'

    # Download example dataset from github source
    PATH_TO_DATA = _download_example(data, verbose=verbose)

    # Import dataset
    if (data=='sprinkler') or (data=='titanic_train'):
        if verbose>=3: print('[bnlearn] >Import dataset..')
        df = pd.read_csv(PATH_TO_DATA)
    else:
        try:
            getPath = _unzip(PATH_TO_DATA, verbose=verbose)
            DAG = import_DAG(data, verbose=2)
            df = sampling(DAG, n=n, verbose=2)
        except:
            print('[bnlearn] >Error: Loading data not possible!')
            df = None

    return df


#%% Download data from github source
def _download_example(data, verbose=3):
    # Set url location
    url = 'https://erdogant.github.io/datasets/'
    url=url + data+'.zip'

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    PATH_TO_DATA = os.path.join(curpath, wget.filename_from_url(url))
    if not os.path.isdir(curpath):
        os.mkdir(curpath)

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        if verbose>=3: print('[bnlearn] >Downloading example [%s] dataset..' %(data))
        wget.download(url, curpath)
    
    return PATH_TO_DATA


# %% Make DAG
def import_DAG(filepath='sprinkler', CPD=True, checkmodel=True, verbose=3):
    """Import Directed Acyclic Graph.

    Parameters
    ----------
    filepath : str, (default: sprinkler)
        Pre-defined examples are depicted below, or provide the absolute file path to the .bif model file.. The default is 'sprinkler'.
        'sprinkler', 'alarm', 'andes', 'asia', 'sachs', 'filepath/to/model.bif',
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
    PATH_TO_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    out={}
    model=None
    filepath=filepath.lower()
    if verbose>=3: print('[bnlearn] >Import <%s>' %(filepath))

    # Load data
    if filepath=='sprinkler':
        model = _DAG_sprinkler(CPD=CPD)
    elif (filepath=='asia') or (filepath=='alarm') or (filepath=='andes') or (filepath=='sachs') or (filepath=='water'):
        getfile = os.path.join(PATH_TO_DATA, filepath+'.bif')
        if not os.path.isfile(getfile):
            PATH_TO_DATA = _download_example(filepath, verbose=3)
            getPath = _unzip(PATH_TO_DATA, verbose=verbose)
        model = _bif2bayesian(getfile, verbose=verbose)
    # elif filepath=='miserables':
    #     getfile = os.path.join(PATH_TO_DATA, filepath+'.json')
    #     if not os.path.isfile(getfile):
    #         PATH_TO_DATA = _download_example(filepath, verbose=3)
    #         getPath = _unzip(PATH_TO_DATA, ext='.json', verbose=verbose)

    #     f = open(os.path.join(PATH_TO_DATA, 'miserables.json'))
    #     data = json.loads(f.read())
    #     L=len(data['links'])
    #     edges=[(data['links'][k]['source'], data['links'][k]['target']) for k in range(L)]
    #     model=nx.Graph(edges, directed=False)
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


# %% unzip
def _unzip(getZip, ext='.bif', verbose=3):
    if not os.path.isdir(getZip):
        if verbose>=3: print('[bnlearn] >Extracting files..')
        [pathname, _] = os.path.split(getZip)
        # Unzip
        zip_ref = zipfile.ZipFile(getZip, 'r')
        zip_ref.extractall(pathname)
        zip_ref.close()
        getPath = getZip.replace('.zip', ext)
        if not os.path.isfile(getPath):
            getPath = None

    return getPath


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
def predict(model, df, variables, to_df=True, method='max', verbose=3):
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
    method : str
        The method that is used to select the for the inferences.
        'max' : Return the variable values based on the maximum probability.
        None : Returns all Probabilities
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
    >>> print(bn.query2df(query))
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
        P[loc] = _get_prob(query, method=method)
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
def _get_prob(query, method='max'):
    # Setup all combinations
    allcomb = np.array(list(itertools.product([0, 1], repeat=len(query.variables))))
    # Get highest P-value and gather data
    Pq = query.values.flatten()
    if method=='max':
        idx = np.argmax(Pq)
        comb = allcomb[idx]
        p = Pq[idx]
        # Store in dict
        out = dict(zip(query.variables, comb))
        out['p']=p
    else:
        out = bnlearn.query2df(query).to_dict()
    return out


# %% Save model
def save(model, filepath='bnlearn_model.pkl', overwrite=False, verbose=3):
    """Save learned model in pickle file.

    Parameters
    ----------
    filepath : str, (default: 'bnlearn_model.pkl')
        Pathname to store pickle files.
    overwrite : bool, (default=False)
        Overwite file if exists.
    verbose : int, optional
        Show message. A higher number gives more informatie. The default is 3.

    Returns
    -------
    bool : [True, False]
        Status whether the file is saved.

    """
    if (filepath is None) or (filepath==''):
        filepath = 'bnlearn_model.pkl'
    if filepath[-4:] != '.pkl':
        filepath = filepath + '.pkl'
    filepath = str(Path(filepath).absolute())

    # Store data
    # storedata = {}
    # storedata['model'] = model
    # Save
    status = pypickle.save(filepath, model, overwrite=overwrite, verbose=verbose)
    # return
    return status


# %% Load model.
def load(filepath='bnlearn_model.pkl', verbose=3):
    """Load learned model.

    Parameters
    ----------
    filepath : str
        Pathname to stored pickle files.
    verbose : int, optional
        Show message. A higher number gives more information. The default is 3.

    Returns
    -------
    Object.

    """
    if (filepath is None) or (filepath==''):
        filepath = 'bnlearn_model.pkl'
    if filepath[-4:]!='.pkl':
        filepath = filepath + '.pkl'
    filepath = str(Path(filepath).absolute())
    # Load
    model = pypickle.load(filepath, verbose=verbose)
    # Store in self.
    if model is not None:
        return model
    else:
        if verbose>=2: print('[bnlearn] >WARNING: Could not load data from [%s]' %(filepath))


# %% Make graph layout
# def graphlayout(model, pos, scale=1, layout='fruchterman_reingold', verbose=3):
#     if isinstance(pos, type(None)):
#         if layout=='fruchterman_reingold':
#             pos = nx.fruchterman_reingold_layout(model, scale=scale, iterations=50)
#         else:
#             pos = nx.spring_layout(model, scale=scale, iterations=50)
#     else:
#         if verbose>=3: print('[bnlearn] >Existing coordinates from <pos> are used.')

#     return(pos)
