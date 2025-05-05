"""Bayesian techniques for structure learning, parameter learning, inference and sampling."""
# ------------------------------------
# Name        : bnlearn.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See LICENSE
# ------------------------------------


# %% Libraries
import os
import copy
import itertools
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# import logging

from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from decimal import Decimal
from itertools import product
from collections import defaultdict

from pgmpy.models import BayesianNetwork, NaiveBayes, MarkovNetwork
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.metrics import structure_score

from setgraphviz import setgraphviz
from ismember import ismember
import datazets as dz
import pypickle
import bnlearn

# %%  Convert adjmat to bayesian model
def to_bayesiannetwork(model, verbose=3):
    """Convert adjacency matrix to BayesianNetwork.

    Convert a adjacency to a Bayesian model. This is required as some of the
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
    BayesianNetwork : Object
        BayesianNetwork that can be used in ``parameter_learning.fit``.

    """
    if isinstance(model, dict):
        adjmat = model.get('adjmat', None)
    else:
        adjmat = model
    if adjmat is None: raise Exception('[bnlearn] >Error: input for "bayesiannetwork" should be adjmat or a dict containing a key "adjmat".')

    if verbose>=3: print('[bnlearn] >Converting adjmat to BayesianNetwork.')

    # Convert to vector
    vec = adjmat2vec(adjmat)[['source', 'target']].values.tolist()
    # Make BayesianNetwork
    bayesianmodel = BayesianNetwork(vec)
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
        * 'nb' or 'naivebayes': Special case of Bayesian Model where the only edges in the model are from the feature variables to the dependent variable. Or in other words, each tuple should start with the same variable name such as: edges = [('A', 'B'), ('A', 'C'), ('A', 'D')]
        * 'markov': Markov model
        * 'DBN': DynamicBayesianNetwork
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
    >>> fig = bn.plot(DAG)

    Examples
    --------
    >>> import bnlearn as bn
    >>> edges = [('A', 'B'), ('A', 'C'), ('A', 'D')]
    >>> CPD = build_cpts_from_structure(DAG, variable_card=3)
    >>> DAG = bn.make_DAG(edges, CPD=CPD, methodtype='naivebayes')
    >>> fig = bn.plot(DAG)

    Examples
    --------
    >>> import bnlearn as bn
    >>> edges = [('A', 'B'), ('A', 'C'), ('A', 'D')]
    >>> DAG = bn.make_DAG(edges, methodtype='markov')
    >>> fig = bn.plot(DAG)

    Examples
    --------
    >>> import bnlearn as bn
    >>> edges = [('A', 'B'), ('A', 'C'), ('A', 'D')]
    >>> DAG = bn.make_DAG(edges, methodtype='DBN')
    >>> fig = bn.plot(DAG)
    >>>
    >>> # Set custom timeslice
    >>> edges = [(('A', 0), ('B', 1)), (('A', 0), ('C', 0)), (('A', 0), ('D', 0))]
    >>> DAG = bn.make_DAG(edges, methodtype='DBN')
    >>> fig = bn.plot(DAG)

    """
    # Set names to lower
    if methodtype == 'nb': methodtype = 'naivebayes'
    if methodtype == 'dbn': methodtype = 'DBN'
    # Automatically generate placeholder values for the CPTs
    if CPD is None and isinstance(DAG, list):
        CPD = build_cpts_from_structure(DAG, variable_card=2, methodtype=methodtype)
    # Make list if required
    if (CPD is not None) and (not isinstance(CPD, list)):
        CPD = [CPD]

    if isinstance(DAG, dict):
        DAG = DAG.get('model', None)

    if (not isinstance(DAG, list)) and ('pgmpy' not in str(type(DAG))):
        raise Exception("[bnlearn] >Error: Input DAG should be a list. in the form [('A','B'), ('B','C')] or a <pgmpy.models.BayesianNetwork>")
    elif ('pgmpy' in str(type(DAG))):
        # Extract methodtype from existing model.
        if ('bayesianmodel' in str(type(DAG)).lower()):
            methodtype = 'bayes'
        elif ('naivebayes' in str(type(DAG)).lower()):
            methodtype = 'naivebayes'
        if verbose>=3: print('[bnlearn] >No changes made to existing %s DAG.' %(methodtype))
    elif isinstance(DAG, list) and methodtype == 'naivebayes':
        if verbose>=3: print('[bnlearn] >%s DAG created.' %(methodtype))
        edges = DAG
        DAG = NaiveBayes()
        DAG.add_edges_from(edges)
        DAG.add_nodes_from(CPD)
        for cpd in CPD: DAG.add_cpds(cpd)
    elif isinstance(DAG, list) and methodtype == 'bayes':
        if verbose>=3: print('[bnlearn] >%s DAG created.' %(methodtype))
        edges = DAG
        DAG = BayesianNetwork()
        DAG.add_edges_from(edges)
        DAG.add_nodes_from(CPD)
        for cpd in CPD: DAG.add_cpds(cpd)
    elif isinstance(DAG, list) and methodtype == 'markov':
        if verbose>=3: print('[bnlearn] >%s DAG created.' %(methodtype))
        edges = DAG
        DAG = MarkovNetwork()
        DAG.add_edges_from(edges)
        DAG.add_nodes_from(CPD)
        # for cpd in CPD: DAG.add_cpds(cpd)
    elif isinstance(DAG, list) and methodtype == 'DBN':
        if verbose>=3: print('[bnlearn] >DynamicBayesianNetwork (DBN) DAG created.')
        # Make edges with time slice
        if not has_valid_time_slice(DAG):
            edges = convert_edges_with_time_slice(DAG, verbose=verbose)
        else:
            edges = DAG

        DAG = DBN()
        DAG.add_edges_from(edges)
        # DAG.add_nodes_from(CPD)
        # DAG.add_cpds(CPD)
        for cpd in CPD: DAG.add_cpds(cpd)

        # Print edges
        # print("Edges in the DBN:", DAG.edges())

    if CPD is not None:
        for cpd in CPD:
            # DAG.add_cpds(cpd)
            if verbose>=3: print(f'[bnlearn] >[Conditional Probability Table (CPT)] >[Update Probabilities] >[Node {cpd.variable}]')
        # Check model
        if checkmodel:
            check_model(DAG, verbose=verbose)

    # Create adjacency matrix from DAG
    out = {}
    out['adjmat'] = dag2adjmat(DAG)
    out['model'] = DAG
    out['methodtype'] = methodtype
    out['model_edges'] = DAG.edges()
    return out


# %%
def convert_edges_with_time_slice(edges, time_slice=0, verbose=3):
    if verbose>=3: print(f'[bnlearn]> Converting edges to time slice.')
    return [( (u, time_slice), (v, time_slice) ) for u, v in edges]

def has_valid_time_slice(edges):
    """ Example usage.
    edges = [('A', 'B'), ('A', 'C'), ('A', 'D')]
    edgest = [(('A', 0), ('B', 0)), (('A', 0), ('C', 0)), (('A', 0), ('D', 0))]

    # Check edges
    has_valid_time_slice(edges)

    # Check edges
    has_valid_time_slice(edgest)

    """
    for edge in edges:
        u, v = edge
        # Check if both ends of the edge are tuples of the form (node, time_slice)
        if not (isinstance(u, tuple) and isinstance(v, tuple)):
            return False
        if not (isinstance(u[1], int) and isinstance(v[1], int)):
            return False
    return True

# %% Print DAG
def print_CPD(DAG, checkmodel=False, verbose=3):
    """Print DAG-model to screen.

    Parameters
    ----------
    DAG : pgmpy.models.BayesianNetwork
        model of the DAG.
    checkmodel : bool
        Check the validity of the model. The default is True

    Returns
    -------
    dict
        Dictionary containing the CPDs.

    Examples
    --------
    >>> # Import library
    >>> import bnlearn as bn
    >>>
    >>> # Load example dataset
    >>> df = bn.import_example('sprinkler')
    >>>
    >>> # Set edges
    >>> edges = [('Cloudy', 'Sprinkler'),
    >>>          ('Cloudy', 'Rain'),
    >>>          ('Sprinkler', 'Wet_Grass'),
    >>>          ('Rain', 'Wet_Grass')]
    >>>
    >>> # Make the actual Bayesian DAG
    >>> DAG = bn.make_DAG(edges)
    >>> model = bn.parameter_learning.fit(DAG, df)
    >>>
    >>> # Gather and store the CPDs in dictionary contaning dataframes for each node.
    >>> CPD = bn.print_CPD(model, verbose=0)
    >>>
    >>> CPD['Cloudy']
    >>> CPD['Rain']
    >>> CPD['Wet_Grass']
    >>> CPD['Sprinkler']
    >>>
    >>> # Print nicely
    >>> from tabulate import tabulate
    >>> print(tabulate(CPD['Cloudy'], tablefmt="grid", headers="keys"))

    """
    # config = None
    CPDs = {}
    if isinstance(DAG, dict):
        DAG = DAG.get('model', None)

    try:
        if ('markovnetwork' in str(type(DAG)).lower()):
            if verbose>=3: print('[bnlearn] >Converting markovnetwork to Bayesian model')
            DAG = DAG.to_bayesian_model()

        if 'maximumlikelihood' in str(type(DAG)).lower():
            # print CPDs using Maximum Likelihood Estimators
            for node in DAG.state_names:
                if verbose>=3: print(DAG.estimate_cpd(node))
        elif ('bayesiannetwork' in str(type(DAG)).lower()) or ('naivebayes' in str(type(DAG)).lower()):
            # print CPDs using Bayesian Parameter Estimation
            if len(DAG.get_cpds())==0:
                raise Exception('[bnlearn] >Error! This is a Bayesian DAG containing only edges, and no CPDs. Tip: you need to specify or learn the CPDs. Try: DAG=bn.parameter_learning.fit(DAG, df). At this point you can make a plot with: bn.plot(DAG).')
                return
            for cpd in DAG.get_cpds():
                CPDs[cpd.variable] = query2df(cpd, verbose=verbose)
                if verbose>=3:
                    print("CPD of {variable}:".format(variable=cpd.variable))
                    print(cpd)
            if ('bayesiannetwork' in str(type(DAG)).lower()):
                if verbose>=3: print('[bnlearn] >Independencies:\n%s' %(DAG.get_independencies()))

            if verbose>=3:
                print('[bnlearn] >Nodes: %s' %(DAG.nodes()))
                print('[bnlearn] >Edges: %s' %(DAG.edges()))

        if checkmodel:
            check_model(DAG, verbose=3)
    except:
        if verbose>=2: print('[bnlearn] >No CPDs to print. Hint: Add CPDs as following: <bn.make_DAG(DAG, CPD=[cpd_A, cpd_B, etc])> and use bnlearn.plot(DAG) to make a plot.')

    # Returning dict with CPDs
    return CPDs


# %% Check model CPDs
def check_model(DAG, verbose=3):
    """Check if the CPDs associated with the nodes are consistent.

    Parameters
    ----------
    DAG : Object.
        Object containing CPDs.
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

    Returns
    -------
    None.

    """
    # Get the model
    if isinstance(DAG, dict):
        DAG = DAG.get('model', None)

    # Not all models do have get_cpds function
    if DAG is not None and hasattr(DAG, 'get_cpds'):
        for cpd in DAG.get_cpds():
            if not np.all(cpd.values.astype(Decimal).sum(axis=0)==1):
                if verbose>=3: print(f'[bnlearn] >[Conditional Probability Table (CPT)] >[Check Probabilities] >[Node {cpd.variable}] >Table Error: Does not sum to 1 but is [{cpd.values.sum(axis=0)}]')
            else:
                if verbose>=3: print(f'[bnlearn] >[Conditional Probability Table (CPT)] >[Check Probabilities] >[Node {cpd.variable}] >OK')
        # if verbose>=3: print('[bnlearn] >Check whether CPDs associated with the nodes are consistent: %s' %(DAG.check_model()))
    elif 'markovnetwork' in str(type(DAG)).lower():
        pass
        # if verbose>=3: print(f'[bnlearn] >[Conditional Probability Table (CPT)] >[Check Probabilities] >Unknown')
    else:
        if verbose>=2: print('[bnlearn] >No model found containing CPDs.')


# %% Convert DAG into adjacency matrix
def dag2adjmat(model, verbose=3):
    """Convert model into adjacency matrix.

    Parameters
    ----------
    model : object
        Model object.
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

    Returns
    -------
    pd.DataFrame
        adjacency matrix.

    Examples
    --------
    >>> import bnlearn as bn
    >>> # Load DAG
    >>> DAG = bn.import_DAG('Sprinkler')
    >>> # Extract edges from model and store in adjacency matrix
    >>> adjmat=bn.dag2adjmat(DAG['model'])

    """
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
    return adjmat


# %%  Convert vector into sparse dataframe
def vec2df(source, target, weights=None):
    """Convert source-target edges into sparse dataframe.

    Description
    -----------
    Convert edges between source and taget into a dataframe based on the weight.
    A weight of 2 will result that a row with the edge is created 2x.

    Parameters
    ----------
    source : array-like
        The source node.
    target : array-like
        The target node.
    weights : array-like of int
        The Weights between the source-target values

    Returns
    -------
    pd.DataFrame

    Examples
    --------
    >>> # Example 1
    >>> import bnlearn as bn
    >>> source=['Cloudy','Cloudy','Sprinkler','Rain']
    >>> target=['Sprinkler','Rain','Wet_Grass','Wet_Grass']
    >>> weights=[1,2,1,3]
    >>> df = bn.vec2df(source, target, weights=weights)

    >>> # Example 2
    >>> import bnlearn as bn
    >>> vec = bn.import_example("stormofswords")
    >>> df = bn.vec2df(vec['source'], vec['target'], weights=vec['weight'])

    """
    if (isinstance(source, pd.DataFrame)) or (isinstance(source, pd.Series)):
        source=source.values
    if (isinstance(target, pd.DataFrame)) or (isinstance(target, pd.Series)):
        target=target.values
    if (isinstance(weights, pd.DataFrame)) or (isinstance(weights, pd.Series)):
        weights=weights.values

    rows = []
    edges = list(zip(source, target))
    if weights is None:
        weights = np.ones_like(source).astype(int)

    columns=np.unique(np.c_[source, target].ravel())
    for i, edge in enumerate(edges):
        row = [np.logical_or(columns==edge[0], columns==edge[1])] * int(weights[i])
        rows = rows + row

    return pd.DataFrame(np.array(rows), columns=columns)


# %%  Convert adjacency matrix to vector
def vec2adjmat(source, target, weights=None, symmetric: bool = True, aggfunc='sum', verbose=3) -> pd.DataFrame:
    """Convert source and target into adjacency matrix.

    Parameters
    ----------
    source : list
        The source node.
    target : list
        The target node.
    weights : list of int
        The Weights between the source-target values
    symmetric : bool, optional
        Make the adjacency matrix symmetric with the same number of rows as columns. The default is True.
    aggfunc : str, optional
        Aggregate function in case multiple values exists for the same relationship.
        'sum' (default)

    Returns
    -------
    pd.DataFrame
        adjacency matrix.

    Examples
    --------
    >>> source = ['Cloudy', 'Cloudy', 'Sprinkler', 'Rain']
    >>> target = ['Sprinkler', 'Rain', 'Wet_Grass', 'Wet_Grass']
    >>> vec2adjmat(source, target)
    >>>
    >>> weights = [1, 2, 1, 3]
    >>> vec2adjmat(source, target, weights=weight)

    """
    if len(source) != len(target): raise ValueError('[d3graph] >Source and Target should have equal elements.')
    if weights is None: weights = [1] * len(source)
    if verbose>=3: print('[bnlearn] >Converting source-target into adjacency matrix..')

    df = pd.DataFrame(np.c_[source, target], columns=['source', 'target'])
    # Make adjacency matrix
    adjmat = pd.crosstab(df['source'], df['target'], values=weights, aggfunc=aggfunc).fillna(0)
    # Get all unique nodes
    nodes = np.unique(list(adjmat.columns.values) + list(adjmat.index.values))
    # nodes = np.unique(np.c_[adjmat.columns.values, adjmat.index.values].flatten())

    # Make the adjacency matrix symmetric
    if symmetric:
        if verbose>=3: print('[bnlearn] >Making the matrix symmetric..')
        # # Add missing columns
        # node_columns = np.setdiff1d(nodes, adjmat.columns.values)
        # for node in node_columns:
        #     adjmat[node] = 0

        # # Add missing rows
        # node_rows = np.setdiff1d(nodes, adjmat.index.values)
        # adjmat = adjmat.T
        # for node in node_rows:
        #     adjmat[node] = 0

        # Add missing columns
        IA, _ = ismember(nodes, adjmat.columns.values)
        node_columns = nodes[~IA]
        if len(node_columns) > 0:
            df_new_columns = pd.DataFrame(0, index=adjmat.index, columns=node_columns)
            adjmat = pd.concat([adjmat, df_new_columns], axis=1)

        # # Add missing rows
        IA, _ = ismember(nodes, adjmat.index.values)
        node_rows = nodes[~IA]
        # node_rows = np.setdiff1d(nodes, adjmat.index.values)
        if len(node_rows) > 0:
            df_new_rows = pd.DataFrame(0, index=node_rows, columns=adjmat.columns)
            adjmat = pd.concat([adjmat, df_new_rows], axis=0)

        # adjmat = adjmat.T

        # Sort to make ordering of columns and rows similar
        if verbose>=4: print('[bnlearn] >Order columns and rows.')
        _, IB = ismember(adjmat.columns.values, adjmat.index.values)
        adjmat = adjmat.iloc[IB, :]
        adjmat.index.name = 'source'
        adjmat.columns.name = 'target'

    # Force columns to be string type
    adjmat.columns = adjmat.columns.astype(str)
    return adjmat


# %%  Convert adjacency matrix to vector
def adjmat2vec(adjmat, min_weight=1):
    """Convert adjacency matrix into vector with source and target.

    Parameters
    ----------
    adjmat : pd.DataFrame()
        Adjacency matrix.

    min_weight : float
        edges are returned with a minimum weight.
    absolute: bool
        True: Make all values absolute
        False: Keep as is

    Returns
    -------
    pd.DataFrame()
        nodes that are connected based on source and target

    Examples
    --------
    >>> import bnlearn as bn
    >>> source=['Cloudy','Cloudy','Sprinkler','Rain']
    >>> target=['Sprinkler','Rain','Wet_Grass','Wet_Grass']
    >>> adjmat = vec2adjmat(source, target)
    >>> vector = bn.adjmat2vec(adjmat)

    """
    # Convert adjacency matrix into vector
    adjmat = adjmat.stack().reset_index()
    # Set columns
    adjmat.columns = ['source', 'target', 'weight']
    # Remove self loops and no-connected edges
    Iloc1 = adjmat['source'] != adjmat['target']

    # Make all values absolute or not
    Iloc2 = adjmat['weight'] >= min_weight

    # Select
    Iloc = Iloc1 & Iloc2
    # Remove edges that have exactly value: rem_weight
    # if rem_weight is not None:
        # Iloc[adjmat['weight']==rem_weight]=False

    # Take only connected nodes
    adjmat = adjmat.loc[Iloc, :]
    adjmat.reset_index(drop=True, inplace=True)
    return adjmat


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
    adjmat=adjmat.astype(bool)
    graph={}
    rows=adjmat.index.values
    for r in rows:
        graph.update({r: list(rows[adjmat.loc[r, :]])})
    return graph


# %% Sampling from model
# def sampling(DAG, n=1000, methodtype='bayes', verbose=0):
#     """Generate sample(s) using the joint distribution of the network.

#     Parameters
#     ----------
#     DAG : dict
#         Contains model and the adjmat of the DAG.
#     methodtype : str (default: 'bayes')
#         * 'bayes': Forward sampling using Bayesian.
#         * 'gibbs' : Gibbs sampling.
#     n : int, optional
#         Number of samples to generate. The default is 1000.
#     verbose : int, optional
#         Print progress to screen. The default is 3.
#         0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

#     Returns
#     -------
#     df : pd.DataFrame().
#         Dataframe containing sampled data from the input DAG model.

#     Example
#     -------
#     >>> # Example 1
#     >>>
#     >>> # Import library
#     >>> import bnlearn as bn
#     >>> # Load DAG with model
#     >>> DAG = bn.import_DAG('sprinkler')
#     >>> # Sampling
#     >>> df = bn.sampling(DAG, n=1000, methodtype='bayes')
#     >>>
#     >>> # Example 2:
#     >>>
#     >>> # Load example dataset
#     >>> df = bn.import_example('sprinkler')
#     >>> edges = [('Cloudy', 'Sprinkler'),
#     >>>         ('Cloudy', 'Rain'),
#     >>>         ('Sprinkler', 'Wet_Grass'),
#     >>>         ('Rain', 'Wet_Grass')]
#     >>>
#     >>> # Make the actual Bayesian DAG
#     >>> DAG = bn.make_DAG(edges, verbose=3, methodtype='bayes')
#     >>> # Fit model
#     >>> model = bn.parameter_learning.fit(DAG, df, verbose=3, methodtype='bayes')
#     >>> # Sampling using gibbs
#     >>> df = bn.sampling(model, n=100, methodtype='gibbs', verbose=0)

#     """
#     if n<=0: raise ValueError('Number of samples (n) must be 1 or larger!')
#     if (DAG is None) or ('BayesianNetwork' not in str(type(DAG['model']))):
#         raise ValueError('The input model (DAG) must contain BayesianNetwork.')

#     if len(DAG['model'].get_cpds())==0:
#         raise Exception('[bnlearn] >Error! This is a Bayesian DAG containing only edges, and no CPDs. Tip: you need to specify or learn the CPDs. Try: DAG=bn.parameter_learning.fit(DAG, df). At this point you can make a plot with: bn.plot(DAG).')
#         return

#     if methodtype=='bayes':
#         if verbose>=3: print('[bnlearn] >Bayesian forward sampling for %.0d samples..' %(n))
#         # Bayesian Forward sampling and make dataframe
#         infer_model = BayesianModelSampling(DAG['model'])
#         df = infer_model.forward_sample(size=n, seed=None, show_progress=(True if verbose>=3 else False))
#     elif methodtype=='gibbs':
#         if verbose>=3: print('[bnlearn] >Gibbs sampling for %.0d samples..' %(n))
#         # Gibbs sampling
#         gibbs = GibbsSampling(DAG['model'])
#         df = gibbs.sample(size=n, seed=None)
#     else:
#         if verbose>=3: print('[bnlearn] >Methodtype [%s] unknown' %(methodtype))
#     return df


# %% Convert BIF model to bayesian model
def _bif2bayesian(pathname, verbose=3):
    """Return the fitted bayesian model.

    Example
    -------
    >>> from pgmpy.readwrite import BIFReader
    >>> reader = BIFReader("bif_test.bif")
    >>> reader.get_model()
    <pgmpy.models.BayesianNetwork object at 0x7f20af154320>
    """
    from pgmpy.readwrite import BIFReader
    if verbose>=3: print('[bnlearn] >Loading bif file <%s>' %(pathname))

    bifmodel = BIFReader(path=pathname)

    try:
        model = BayesianNetwork(bifmodel.variable_edges)
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


def query2df(query, variables=None, groupby=None, verbose=3):
    """Convert query from inference model to a dataframe.

    Parameters
    ----------
    query : Object from the inference model.
        Convert query object to a dataframe.
    variables : list
        Order or select variables.
    groupby: list of strings (default: None)
        The query is grouped on the variable name by taking the maximum P value for each catagory.

    Returns
    -------
    df : pd.DataFrame()
        Dataframe with inferences.

    """
    if ((groupby is not None) and np.any(np.isin(groupby, variables))):
        # Needs to be set to true.
        groupby = list(np.array(groupby)[np.isin(groupby, variables)])
    else:
        if verbose>=2: print('[bnlearn] >Warning: variable(s) [%s] does not exists in DAG.' %(groupby))
        groupby=None

    states = []
    getP = []
    for value_index, prob in enumerate(itertools.product(*[range(card) for card in query.cardinality])):
        states.append(prob)
        getP.append(query.values.ravel()[value_index])

    df = pd.DataFrame(data=states, columns=query.scope())
    df['p'] = getP

    # Convert the numbers into variable names
    for col in query.scope():
        df[col] = np.array(query.state_names[col])[df[col].values.astype(int)]

    # Order or filter on input variables
    if variables is not None:
        # Add Pvalue column
        variables = variables + ['p']
        df = df[variables]

    # groupby
    if groupby is not None:
        df = df.groupby(groupby).apply(lambda x: x.loc[x['p'].idxmax()])
        df.reset_index(drop=True, inplace=True)

    # Print table to screen
    if verbose>=3:
        print('[bnlearn] >Output is stored in Dataframe [query.df]')
        print(tabulate(df, tablefmt="grid", headers="keys"))

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
    model = BayesianNetwork([('Cloudy', 'Sprinkler'),
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
def get_node_properties(model, node_color='#ADD8E6', node_size=None, verbose=3):
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
    defaults={'node_color': node_color, 'node_size': node_size}
    adjmat = model.get('adjmat', None)

    if adjmat is not None:
        if verbose>=3: print('[bnlearn] >Set node properties.')
        # For each node, use the default node properties.
        for node in adjmat.columns:
            node_property = defaults.copy()
            nodes.update({node: node_property})

    # Return dict with node properties
    return nodes


# %% Get node properties
def get_edge_properties(model, color='#000000', weight=1, minscale=1, maxscale=5, verbose=3):
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
        The minimum weight of the edge in case of test statisics are used.
    maxscale : float, (Default: 10)
        The maximum weight of the edge in case of test statisics are used.
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

    Returns
    -------
    dict.
        Edge properties.

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
    # https://networkx.org/documentation/networkx-1.7/reference/generated/networkx.drawing.nx_pylab.draw_networkx_nodes.html
    edges = {}
    defaults = {'color': color, 'weight': weight, 'pvalue': 1, 'value': 1}
    adjmat = model.get('independence_test', None)
    # Use edge weights from test statistic
    # if adjmat is not None and ('lingam' in model['config']['method']):
    #     # Add to adjmat
    #     adjmat = (model['adjmat'].abs() > 0).astype(float)
    #     adjmatP = vec2adjmat(model['independence_test']['source'], model['independence_test']['target'], weights=model['independence_test']['p_value'])
    # if adjmat is not None and ('lingam' not in model['config']['method']):
    if adjmat is not None:
        if verbose>=3: print('[bnlearn]> Set edge weights based on the [%s] test statistic.' %(model['independence_test'].columns[-2]))
        # Compute logp
        logp = compute_logp(model['independence_test']['p_value'])
        # logp = -np.log10(model['independence_test']['p_value'])
        # Iloc = np.isinf(logp)
        # max_logp = np.max(logp[~Iloc]) * 1.5  # For visualization purposes, set the max higher then what is present to mark the difference.
        # if np.isnan(max_logp): max_logp = 1
        # logp.loc[Iloc] = max_logp
        # Rescale the weights
        weights = _normalize_weights(logp.values, minscale=minscale, maxscale=maxscale)
        # Add to adjmat
        if 'lingam' in model['config']['method']:
            adjmat = (model['adjmat'].abs() > 0).astype(float)
        else:
            adjmat = vec2adjmat(model['independence_test']['source'], model['independence_test']['target'], weights=weights)
        adjmatP = vec2adjmat(model['independence_test']['source'], model['independence_test']['target'], weights=logp)
    else:
        adjmat = model.get('adjmat', None)
        adjmat = (adjmat.abs() > 0).astype(float)

    # Get model edges
    model_edges = adjmat2vec(adjmat)[['source', 'target']].values
    # model_edges = model['model'].edges() if (model.get('model_edges', None) is None) else model['model_edges']

    # Store edge properties
    if adjmat is not None:
        if verbose>=3: print('[bnlearn] >Set edge properties.')
        # For each edge, use the default properties.
        for u, v in model_edges:
            edge_property = defaults.copy()
            # Use the edge weight from the adjmat
            if not isinstance(adjmat.loc[u, v], np.bool_):
                edge_property['weight'] = adjmat.loc[u, v]
                edge_property['value'] = model['adjmat'].loc[u, v]
                if model.get('independence_test', None) is not None:
                    edge_property['pvalue'] = adjmatP.loc[u, v]
                else:
                    edge_property['pvalue'] = 1
            # Update edges dict
            edges.update({(u, v): edge_property})

    # Return dict with node properties
    return edges


def compute_logp(p_value):
    logp = -np.log10(p_value)
    Iloc = np.isinf(logp)
    # For visualization purposes, set the max higher then what is present to mark the difference.
    max_logp = np.max(logp[~Iloc]) * 1.5
    if np.isnan(max_logp): max_logp = 1
    logp.loc[Iloc] = max_logp
    return logp

# %% PLOT
def plot_graphviz(model,
                  edge_labels='weight',
                  params = {'prediction_feature_indices': None,
                            'prediction_target_label': "Y(pred)",
                            'prediction_line_color': "red",
                            'prediction_coefs': None,
                            'prediction_feature_importance': None,
                            'path': None,
                            'path_color': None,
                            'detect_cycle': False,
                            'ignore_shape': False},
                  verify_certificate=True,
                  verbose=3):
    """Plot a causal or Bayesian network using Graphviz based on an adjacency matrix.

    This function visualizes the causal or Bayesian model structure in model['adjmat'] using the Graphviz library. The user can customize various aspects of the plot such as colors, path highlights, and edge labels.

    Parameters
    ----------
    model : dict
        A dictionary containing the network model.
        Must include an adjacency matrix under the key "adjmat" and when the independence_test is performed, the -log10(pvalues) will be shown on the edges to highlight the strength of the significant relationships between variables.
    edge_labels : Bool (default: True)
         None: Do not show edge labels
        'weight': Show the input values in the array
        'pvalue': Show the edge pvalues (this requires doing the independence_test: model = bn.independence_test(model, df)
    params : dict, optional
        A dictionary of parameters to control the visualization. Default values are provided,
        but users can modify specific parameters. The available options are:

        - `prediction_feature_indices` (list or None): The feature indices to be used in the 
          prediction path, or `None` for no specific feature highlights.
        - `prediction_target_label` (str): Label for the prediction target node.
        - `prediction_line_color` (str): Color for the prediction path in the graph.
        - `prediction_coefs` (list or None): Coefficients for the prediction line (optional).
        - `prediction_feature_importance` (list or None): Feature importance values for features 
          in the prediction path.
        - `path` (list or None): Specific path between nodes to highlight, or `None` to ignore.
        - `path_color` (str or None): Color of the path to highlight, or `None`.
        - `detect_cycle` (bool): If `True`, attempt to detect cycles in the graph.
        - `ignore_shape` (bool): If `True`, ignore the node shapes when plotting.

    verify_certificate : bool (default: True)
        True: Verify the certificates
        False: Do not verify
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

    Returns
    -------
    dot_graph : graphviz.Source or None
        A Graphviz `Source` object containing the DOT source code for the graph. This can be 
        rendered or exported. Returns `None` if no edges are found in the adjacency matrix.

    Notes
    -----
    - This function depends on the `graphviz` and `lingam.utils` libraries.
    - If the `independence_test` key is present in `model`, the graph's edges are updated based on the p-values from the test, and edges are weighted accordingly.
    - If the Graphviz executable is not set in the system environment, it will attempt to set it using the `setgraphviz` function.

    Examples
    --------
    >>> # Import library
    >>> import bnlearn as bn
    >>>
    >>> # Load example mixed dataset
    >>> df = bn.import_example(data='sprinkler')
    >>>
    >>> # Structure learning
    >>> model = bn.structure_learning.fit(df)
    >>>
    >>> # Independence test
    >>> model = bn.independence_test(model, df, test='chi_square', prune=False)
    >>>
    >>> # Dot graph
    >>> dotgraph = bn.plot_graphviz(model)
    >>> dotgraph
    >>> # Create pdf
    >>> dotgraph.view(filename=r'c:/temp/dotgraph')

    """
    # Check whether edges are available
    if model['adjmat'].sum().sum()==0:
        if verbose>=3: print('[bnlearn]> Nothing to plot because no edges are present between nodes. ')
        return None
    if model.get('config', {}).get('method') == 'DBN':
        if verbose>=3: print('[bnlearn]> DynamicBayesianNetwork (DBN) can not be plot with Graphviz.')
        return None

    # Import libraries
    from lingam.utils import make_dot
    from graphviz import Source

    # Set Graphviz path to envoirement if required.
    GraphvizPath = setgraphviz(verify_certificate=verify_certificate, verbose=verbose)
    if GraphvizPath is None:
        if verbose>=1:print('Graphviz is not found in path and can therefore cause an error in producint the dot image.')

    # Set default params in case the user adjusted only one parameter
    defaults = {'prediction_feature_indices': None, 'prediction_target_label': "Y(pred)", 'prediction_line_color': "red", 'prediction_coefs': None, 'prediction_feature_importance': None, 'path': None, 'path_color': None, 'detect_cycle': False, 'ignore_shape': False}
    params = {**defaults, **params}

    # Set a copy
    model = copy.deepcopy(model)

    # In case independence test is performed, take that Pvalues
    if model.get('independence_test') is not None and edge_labels=='pvalue':
        if verbose>=3: print('[bnlearn] >Setting edge labels to pvalue.')
        source = model.get('independence_test')['source']
        target = model.get('independence_test')['target']
        # Compute logp
        logp = compute_logp(model['independence_test']['p_value'])
        # logp = -np.log10(model.get('independence_test')['p_value'])
        # Iloc = np.isinf(logp)
        # For visualization purposes, set the max higher then what is present to mark the difference.
        # max_logp = np.max(logp[~Iloc]) * 1.5
        # if np.isnan(max_logp): max_logp = 1
        # logp.loc[Iloc] = max_logp
        # Create new adjmat based on indepdence test
        adjmat = vec2adjmat(source, target, weights=logp, symmetric=True, aggfunc='sum', verbose=verbose)
    else:
        adjmat = model['adjmat'].copy()

    if edge_labels is not None:
        edge_labels = list(adjmat.T.columns)

    # Make the dot and output Directed graph source code in the DOT language.
    dot_graph = make_dot(adjmat.T.values.astype(float), labels=edge_labels, lower_limit=0, **params)

    # Return
    return dot_graph

# %% PLOT
def plot(model,
         pos=None,
         scale=1,
         interactive=False,
         title='bnlearn Directed Acyclic Graph (DAG)',
         node_color=None,
         node_size=None,
         node_properties=None,
         edge_properties=None,
         edge_labels='weight',
         params_interactive={'minmax_distance': [100, 250], 'figsize': (1500, 800), 'notebook': False, 'font_color': '#000000', 'bgcolor': '#ffffff', 'show_slider': True, 'filepath': None},
         params_static={'minscale': 1, 'maxscale': 5, 'figsize': (10, 10), 'width': None, 'height': None, 'font_size': 10, 'font_family': 'sans-serif', 'alpha': 0.8, 'node_shape': 'o', 'layout': 'graphviz_layout', 'font_color': '#000000', 'facecolor': 'white', 'edge_alpha': 0.8, 'arrowstyle': '-|>', 'arrowsize': 20, 'visible': True, 'showplot': True, 'dpi': 200},
         verbose=3,
         ):
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
    node_properties : dict (default: None)
        Dictionary containing custom node_color and node_size parameters for the network.
        The node properties can easily be retrieved using the function: node_properties = bn.get_node_properties(model)
        node_properties = {'node1':{'node_color':'#8A0707','node_size':10},
                           'node2':{'node_color':'#000000','node_size':30}}
    edge_properties : dict (default: None)
        Dictionary containing custom node_color and node_size parameters for the network. The edge properties can be retrieved with:
        edge_properties = bn.get_edge_properties(model)
    edge_labels : Bool (default: True)
        None: Do not show edge labels
        'weight': Show the input values in the array
        'pvalue': Show the edge pvalues (this requires doing the independence_test: model = bn.independence_test(model, df)
    params_interactive : dict.
        Dictionary containing various settings in case of creating interactive plots.
    params_static : dict.
        Dictionary containing various settings in case of creating static plots.
        layout: 'graphviz_layout', 'spring_layout', 'planar_layout', 'shell_layout', 'spectral_layout', 'pydot_layout', 'circular_layout', 'spring_layout', 'random_layout', 'bipartite_layout', 'multipartite_layout',
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
    >>> fig = bn.plot(model)
    >>>
    >>> # plot interactive
    >>> fig = bn.plot(model, interactive=True)
    >>>
    >>> # plot interactive with various settings
    >>> fig = bn.plot(model, interactive=True, node_color='#8A0707', node_size=35, params_interactive = {'figsize':(800, 600), 'font_color': 'node_color', 'bgcolor':'#0f0f0f0f'})
    >>>
    >>> # plot with node properties
    >>> node_properties = bn.get_node_properties(model)
    >>> # Make some changes
    >>> node_properties['xray']['node_color']='#8A0707'
    >>> node_properties['xray']['node_size']=50
    >>> # Plot
    >>> fig = bn.plot(model, interactive=True, node_properties=node_properties)
    >>>

    """
    fig = None
    # Check whether edges are available
    if model['adjmat'].sum().sum()==0:
        if verbose>=3: print('[bnlearn]> Nothing to plot because no edges are present between nodes. ')
        return None

    if model.get('config', {}).get('method') == 'DBN' and interactive:
        if verbose>=3: print('[bnlearn]> DynamicBayesianNetwork (DBN) can not be plot with Graphviz.')
        return None

    if model.get('independence_test', None) is None and edge_labels=='pvalue':
        if verbose>=2: print('[bnlearn] >Edge labels with Pvalues can only be shown after executing: model=bn.independence_test(model, df)')
        edge_labels = None

    model = copy.deepcopy(model)
    model['adjmat'] = model['adjmat'].astype(float)

    # Plot properties
    defaults = {'minmax_distance': [100, 250], 'figsize': (1500, 800), 'notebook': False, 'font_color': '#000000', 'bgcolor': '#ffffff', 'directed': True, 'show_slider': True, 'filepath': None}
    params_interactive = {**defaults, **params_interactive}
    defaults = {'minscale': 1, 'maxscale': 5, 'figsize': (15, 10), 'height': None, 'width': None, 'font_size': 14, 'font_family': 'sans-serif', 'alpha': 0.8, 'layout': 'graphviz_layout', 'font_color': 'k', 'facecolor': '#ffffff', 'node_shape': 'o', 'edge_alpha': 0.8, 'arrowstyle': '-|>', 'arrowsize': 20, 'visible': True, 'showplot': True, 'dpi': 200}
    params_static = {**defaults, **params_static}

    # DEPRECATED IN LATER VERSION
    if (params_static.get('width') is not None) or (params_static.get('height') is not None):
        # if verbose>=2: print('[bnlearn]> Warning: [height] and [width] will be removed in further version. Please use: params_static={"figsize": (15, 10)}.')
        params_static['figsize'] = (15 if params_static['width'] is None else params_static['width'], 10 if params_static['height'] is None else params_static['height'])

    out = {}
    G = nx.DiGraph()  # Directed graph
    node_size_default = 10 if interactive else 800
    if (node_properties is not None) and (node_size is not None):
        if verbose>=2: print('[bnlearn]> Warning: if both "node_size" and "node_properties" are used, "node_size" will be used.')

    # Get node and edge properties if not user-defined
    if node_properties is None:
        node_properties = bnlearn.get_node_properties(model, node_size=node_size_default)
    if edge_properties is None:
        edge_properties = bnlearn.get_edge_properties(model, minscale=params_static['minscale'], maxscale=params_static['maxscale'])

    # Set default node size based on interactive True/False
    for key in node_properties.keys():
        if node_properties[key]['node_size'] is None:
            node_properties[key]['node_size']=node_size_default

    # Add edges with weights based on independence test results
    for edge, properties in edge_properties.items():
        # strength = properties.get("weight", 0)
        G.add_edge(*edge, weight=properties.get("weight", 0), pvalue=properties.get("pvalue", 1), value=properties.get("value", 0))

    # Update the dataframe with the normalized weights or P-values
    # if edge_labels=='pvalue':
    #     for (source, target), value in edge_properties.items():
    #         model['adjmat'].loc[source, target] = value['pvalue']

    # Extract model if in dict
    if 'dict' in str(type(model)):
        bnmodel = model.get('model', None)
    else:
        bnmodel = copy.deepcopy(model)

    # get node properties
    nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights, edge_pvalue, edge_value = _plot_properties(G, node_properties, edge_properties, node_color, node_size)
    tooltip = nodelist

    # Plot
    if interactive:
        if hasattr(model["model"], 'get_cpds'):
            tooltip = []
            for node in nodelist:
                tip = model["model"].get_cpds(node)
                if tip is None: tip = node
                tooltip.append(tip)

        # Make interactive plot
        fig = _plot_interactive(params_interactive,
                                nodelist,
                                node_colors,
                                node_sizes,
                                edgelist,
                                edge_colors,
                                edge_weights,
                                title,
                                tooltip,
                                verbose=verbose)
    else:
        # Bayesian model
        if ('bayes' in str(type(bnmodel)).lower()) or ('pgmpy' in str(type(bnmodel)).lower()) or ('lingam' in model['config']['method']):
            if verbose>=3: print('[bnlearn] >Plot based on Bayesian model')
            # positions for all nodes
            # G = nx.DiGraph(model['adjmat'])
            pos = bnlearn.network.graphlayout(G, pos=pos, scale=scale, layout=params_static['layout'], verbose=verbose)
        elif 'networkx' in str(type(bnmodel)):
            if verbose>=3: print('[bnlearn] >Plot based on networkx model')
            G = bnmodel
            pos = bnlearn.network.graphlayout(G, pos=pos, scale=scale, layout=params_static['layout'], verbose=verbose)
        else:
            if verbose>=3: print('[bnlearn] >Plot based on adjacency matrix')
            G = bnlearn.network.adjmat2graph(model['adjmat'].abs()>0)
            # Get positions
            pos = bnlearn.network.graphlayout(G, pos=pos, scale=scale, layout=params_static['layout'], verbose=verbose)

        # Make static plot
        fig = _plot_static(model,
                           params_static,
                           nodelist,
                           node_colors,
                           node_sizes,
                           G,
                           pos,
                           edge_colors,
                           edge_weights,
                           showplot=params_static['showplot'],
                           visible=params_static['visible'],
                           title=title,
                           dpi=params_static['dpi'],
                           edge_labels=edge_labels,
                           )

    # Store
    out['fig'] = fig
    out['ax'] = fig  # Should be removed in later releases
    out['pos'] = pos
    out['G'] = G
    out['node_properties'] = node_properties
    out['edge_properties'] = edge_properties
    return out


# %% Plot interactive
def _plot_static(model, params_static, nodelist, node_colors, node_sizes, G, pos, edge_colors, edge_weights, title, visible=True, showplot=True, dpi=100, edge_labels='weight'):

    # Initialize
    fig = plt.figure(figsize=params_static['figsize'], facecolor=params_static['facecolor'], dpi=dpi)
    # Set visible
    fig.set_visible(visible)
    # nodes
    nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_size=node_sizes, alpha=params_static['alpha'], node_color=node_colors, node_shape=params_static['node_shape'])
    # edges
    nx.draw_networkx_edges(G, pos, arrowstyle=params_static['arrowstyle'], arrowsize=params_static['arrowsize'], edge_color=edge_colors, width=edge_weights, alpha=params_static['edge_alpha'])

    # Plot text of the weights
    if edge_labels=='weight':
        edge_label = nx.get_edge_attributes(G, 'value')
        edge_label = {key: float(f'{value:.2f}'[:4]) for key, value in edge_label.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label)
        # edge_label = nx.get_edge_attributes(G, 'weight')
        # edge_label = {key: float(f'{value:.2f}'[:4]) for key, value in edge_label.items()}
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label)
    elif edge_labels=='pvalue':
        edge_label = nx.get_edge_attributes(G, 'pvalue')
        # Truncate or format values in the edge_weight dictionary to be no more than 3 characters
        edge_label = {key: float(f'{value:.2f}'[:4]) for key, value in edge_label.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label)

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=params_static['font_size'], font_family=params_static['font_family'], font_color=params_static['font_color'])

    # Making figure nice
    # fig = plt.gca()
    # fig.set_axis_off()
    plt.title(title)
    if showplot:
        plt.show()
    # Return
    return fig


# %% Plot interactive
def _plot_interactive(params_interactive, nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights, title, tooltip, verbose=3):
    # Try to import d3blocks
    from packaging import version
    try:
        # Load library
        from d3blocks import D3Blocks
    except ModuleNotFoundError:
        if verbose>=1: raise Exception('[bnlearn] >"d3blocks" library is not installed. Pip install first: "pip install d3blocks"')

    import d3blocks as d3
    if version.parse(d3.__version__) < version.parse("1.4.9"):
        raise ImportError('[bnlearn] >Error: d3blocks version should be >= 1.4.9. Pip install to the latest version first: "pip install -U d3blocks"')

    if params_interactive['filepath'] is None: params_interactive['filepath'] = title.strip().replace(' ', '_') + '.html'

    # Need to run this as this is also done in d3
    nodelist = list(map(lambda x: x.encode('ascii', 'ignore').decode("utf-8").replace(' ', '_'), list(nodelist)))

    # Initialize
    d3 = D3Blocks()

    # Set the min_weight
    X = pd.DataFrame(data=edgelist, columns=['source', 'target'])
    X['weight'] = edge_weights
    # X = vec2adjmat(target=X['target'], source=X['source'], weight=X['weight'])

    # Create network using default
    d3.d3graph(X,
               showfig=False,
               title=title,
               notebook=params_interactive['notebook'])

    # Change edge properties
    d3.D3graph.set_edge_properties(directed=params_interactive['directed'],
                                   minmax_distance=params_interactive['minmax_distance'],
                                   marker_color=edge_colors)

    # Change node properties
    _, IB = ismember([*d3.D3graph.node_properties.keys()], nodelist)

    d3.D3graph.set_node_properties(tooltip=np.array(tooltip)[IB],
                                   size=np.array(node_sizes)[IB],
                                   color=np.array(node_colors)[IB],
                                   fontcolor=params_interactive['font_color'],
                                   )

    # Show the interactive plot
    d3.D3graph.show(show_slider=params_interactive['show_slider'],
                    filepath=params_interactive['filepath'],
                    figsize=params_interactive['figsize'])

    # Return
    return os.path.abspath(d3.D3graph.config['filepath'])



# %% Plot properties
def _plot_properties(G, node_properties, edge_properties, node_color, node_size):
    # Set edge properties in Graph G
    # edges=[*bnmodel.edges()]
    edges = list(edge_properties.keys())
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
    edge_pvalue = [G[u][v].get('pvalue') for u, v in G.edges()]
    edge_value = [G[u][v].get('value') for u, v in G.edges()]
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
    return nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights, edge_pvalue, edge_value


# %%
def topological_sort(adjmat, start=None):
    """Topological sort.

    Description
    -----------
    Get nodes list in the topological sort order.

    Parameters
    ----------
    adjmat: pd.DataFrame or bnlearn object.
        Adjacency matrix.
    start: str, optional
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
    elif np.all(np.isin(adjmat.columns, ['source', 'target', 'weight'])):
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
            seen.add(v)  # no need to append to path any more
            q.extend(graph[v])

            while stack and v not in graph[stack[-1]]:  # new stuff here!
                order.append(stack.pop())
            stack.append(v)

    return stack + order[::-1]


# %% Example data
def import_example(data='sprinkler', url=None, sep=',', n=10000, verbose=3):
    """Load example dataset.

    Parameters
    ----------
    data: str, (default: sprinkler)
        Pre-defined examples.
            * 'sprinkler'
            * 'alarm'
            * 'andes'
            * 'asia'
            * 'sachs'
            * 'water'
        Continous data sets:
            * 'auto_mpg'
    n: int, optional
        Number of samples to generate. The default is 10000.
    verbose: int, (default: 3)
        Print progress to screen.
        0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace

    Returns
    -------
    df: pd.DataFrame()

    """

    if (data=='alarm') or (data=='andes') or (data=='asia') or (data=='sachs') or (data=='water'):
        try:
            DAG = import_DAG(data, verbose=2)
            df = bnlearn.sampling(DAG, n=n, verbose=2)
        except:
            print('[bnlearn] >Error: Loading data not possible!')
            df = None

    else:
        df = dz.get(data, url=url, sep=sep, n=n, verbose=0)

    return df


# %% Make DAG
def import_DAG(filepath='sprinkler', CPD=True, checkmodel=True, verbose=3):
    """Import Directed Acyclic Graph.

    Parameters
    ----------
    filepath: str, (default: sprinkler)
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
        model : BayesianNetwork
        adjmat : Adjacency matrix

    Examples
    --------
    >>> import bnlearn as bn
    >>> model = bn.import_DAG('sprinkler')
    >>> fig = bn.plot(model)

    """
    out = {}
    model = None
    # filepath= filepath.lower()
    if verbose>=3: print('[bnlearn] >Import <%s>' %(filepath))
    # Get the data properties
    dataproperties = dz.get_dataproperties(filepath)
    # Get path to data
    # PATH_TO_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    PATH_TO_DATA = dataproperties['curpath']

    # Load data
    if filepath=='sprinkler':
        model = _DAG_sprinkler(CPD=CPD)
    elif (filepath=='asia') or (filepath=='alarm') or (filepath=='andes') or (filepath=='sachs') or (filepath=='water'):
        getfile = os.path.join(PATH_TO_DATA, filepath +'.bif')
        if not os.path.isfile(getfile):
            PATH_TO_DATA = dz.download_from_url(dataproperties['filename'], url=dataproperties['url'])
            _ = dz.unzip(PATH_TO_DATA)

        model = _bif2bayesian(getfile, verbose=verbose)
    else:
        if os.path.isfile(filepath):
            model = _bif2bayesian(filepath, verbose=verbose)
        else:
            if verbose>=3: print('[bnlearn] >filepath does not exist! <%s>' %(filepath))
            return out

    # Setup adjacency matrix
    adjmat = dag2adjmat(model)

    # Store
    out['model']=model
    out['adjmat']=adjmat

    # check_model check for the model structure and the associated CPD and returns True if everything is correct otherwise throws an exception
    if (model is not None) and CPD and checkmodel:
        check_model(out['model'], verbose=verbose)
        if verbose>=4:
            print_CPD(out)

    return(out)


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
    df_hot.columns = df_hot.columns.str.replace('_4.0', '_4', regex=True)
    df_hot.columns = df_hot.columns.str.replace('_3.0', '_3', regex=True)
    df_hot.columns = df_hot.columns.str.replace('_2.0', '_2', regex=True)
    df_hot.columns = df_hot.columns.str.replace('_1.0', '_1', regex=True)
    df_hot.columns = df_hot.columns.str.replace('_0.0', '_0', regex=True)

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
    dfU.reset_index(drop=True, inplace=True)
    # Make empty array
    P = np.array([None] *dfX.shape[0])

    evidences = list(map(lambda x: dfU.iloc[x, :].to_dict(), range(dfU.shape[0])))
    dfU_shape = dfU.shape[1]
    # for i in tqdm(range(dfU.shape[0])):
    for evidence in tqdm(evidences):
        # Get input data and create a dict.
        # evidence = dfU.iloc[i, :].to_dict()
        # Do the inference.
        query = bnlearn.inference.fit(model, variables=variables, evidence=evidence, to_df=False, verbose=0)
        # Find original location of the input data.
        # loc = np.sum((dfX==dfU.iloc[i, :]).values, axis=1)==dfU_shape
        loc = np.sum(dfX.values==[*evidence.values()], axis=1)==dfU_shape
        # Store inference
        P[loc] = _get_prob(query, method=method)

    # Make list
    P = list(P)
    # Make dataframe
    if to_df: P = pd.DataFrame(P)
    return P


# %%
def _get_prob(query, method='max'):
    if method=='max':
        # Setup all combinations
        possible_values = query.state_names.values()
        allcomb = np.array(list(itertools.product(*possible_values)))
        # Get highest P-value and gather data
        Pq = query.values.flatten()
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


# %% Compute Pvalues using independence test.
def independence_test(model, df, test="chi_square", alpha=0.05, prune=False, verbose=3):
    """Compute edge strength using test statistic.

    Description
    -----------
    Compute the edge strength using a statistical test of independence based using the model structure (DAG) and the data.
    For the pairs in the DAG (either by structure learning or user-defined), an statistical test is performed.
    Any two variables are associated if the test's p-value < significance_level.

    Parameters
    ----------
    model: Instance of bnlearn.structure_learning.
        The (learned) model which needs to be tested.
    df: pandas.DataFrame instance
        The dataset against which to test the model structure.
    test: str or function
        The statistical test to compute associations.
            * chi_square
            * g_sq
            * log_likelihood
            * freeman_tuckey
            * modified_log_likelihood
            * neyman
            * cressie_read
    alpha: float
        A value between 0 and 1. If p_value < significance_level, the variables are
        considered uncorrelated.
    prune: bool (default: False)
        True: Keep only edges that are significant (<=alpha) based on the independence test.

    Returns
    -------
    df: pandas.DataFrame instance
        The dataset against which to test the model structure.

    Examples
    --------
    >>> import bnlearn as bn
    >>> df = bn.import_example(data='asia')
    >>> # Structure learning of sampled dataset
    >>> model = bn.structure_learning.fit(df)
    >>> # Compute arc strength
    >>> model = bn.independence_test(model, df, test='chi_square')
    >>> print(model['independence_test'])
    >>>
    >>>       source     target  stat_test       p_value  chi_square  dof
    >>> 0     Cloudy  Sprinkler       True  8.383708e-53  233.906474    1
    >>> 1  Sprinkler  Wet_Grass       True  1.196919e-23  100.478455    1
    >>> 2       Rain     Cloudy       True  1.080606e-87  394.061629    1
    >>> 3       Rain  Wet_Grass       True  3.886511e-64  285.901702    1

    """
    from pgmpy.estimators.CITests import chi_square, g_sq, log_likelihood, freeman_tuckey, modified_log_likelihood, neyman, cressie_read  # noqa
    from pgmpy.models import BayesianNetwork
    from pgmpy.base import DAG
    from lingam import DirectLiNGAM, ICALiNGAM
    if model.get('model', None) is None: raise ValueError('[bnlearn]> No model detected.')
    if not isinstance(model['model'], (DAG, BayesianNetwork, DirectLiNGAM, ICALiNGAM)): raise ValueError("[bnlearn]> model must be an instance of pgmpy.base.DAG or pgmpy.models.BayesianNetwork. Got {type(model)}")
    if not isinstance(df, pd.DataFrame): raise ValueError("[bnlearn]> data must be a pandas.DataFrame instance. Got {type(data)}")
    if isinstance(model['model'], (DAG, BayesianNetwork)):
        if not np.all(np.isin(model['model'].nodes(), df.columns)): raise ValueError("[bnlearn]> Missing columns in data. Can't find values for the following variables: { set(model.nodes()) - set(data.columns) }")

    # Get a copy of the model
    model_update = copy.deepcopy(model)

    if isinstance(model['model'], (DirectLiNGAM, ICALiNGAM)):
        # Get a copy of the model
        if verbose >= 3: print(f'[bnlearn] >Compute edge strength with {model["config"]["method"]}')
        test = 'direct-lingam'

        # Extract info from the independence_p_values
        p_value = pd.DataFrame(model_update['model'].get_error_independence_p_values(df), columns=df.columns.values, index=df.columns.values)
        # Sort the correction_matrix index and columns to match adjmat
        p_value = p_value.reindex(index=model['adjmat'].index, columns=model['adjmat'].columns)
        # Make sure the Pvalue matrix is consistent with the adjmat
        p_value = p_value[p_value * model['adjmat'].abs() > 0].fillna(value=1)

        # out['independence_test'] = p_value
        independence_test = bnlearn.adjmat2vec(p_value, min_weight=0)
        independence_test.rename(columns={'weight': 'p_value'}, inplace=True)
        independence_test['dof'] = 1
        model_update['independence_test'] = independence_test
        # Set the significant edges to True
        model_update['independence_test']['stat_test'] = model_update['independence_test']['p_value'] <= alpha
    else:
        if verbose>=3: print('[bnlearn] >Compute edge strength with [%s]' %(test))
        # Get the statistical test
        statistical_test = eval(test)
        # Compute significance
        results = []
        for i, j in model_update['model_edges']:
            # test_result = power_divergence(i, j, [], df, boolean=False, lambda_="cressie-read", significance_level=0.05)
            # chi, p_value, dof, expected = stats.chi2_contingency( df.groupby([i, j]).size().unstack(j, fill_value=0), lambda_="cressie-read" )
            test_result = statistical_test(X=i, Y=j, Z=[], data=df, boolean=False, significance_level=alpha)
            results.append({"source": i, "target": j, "stat_test": test_result[1]<=alpha, 'p_value': test_result[1], test: test_result[0], 'dof': test_result[2]})

        # Update model
        model_update['independence_test'] = pd.DataFrame(results)

    # Remove not significant edges
    if prune and len(model_update['model_edges']) > 0:
        model_update = _prune(model_update, test, alpha, verbose=verbose)

    # Return
    return model_update


# %% Remove not significant edges.
def _prune(model, test, alpha, verbose=3):

    # Prune based on significance alpha
    if model.get('independence_test', None) is not None:
        # Find the none significant associations.
        Irem = ~model['independence_test']['stat_test']
        idxrem = np.where(Irem)[0]

        # Set not-significant edges to False
        # for idx in idxrem:
        #     edge = list(model['independence_test'].iloc[idx][['source', 'target']])

        #     if test=='direct-lingam':
        #         pass
        #     else:
        #         model['adjmat'].loc[edge[0], edge[1]] = False
        #         model['adjmat'].loc[edge[1], edge[0]] = False
        #         # Remove edges
        #         if np.any(np.isin(model['model_edges'], edge).sum(axis=1)==2) and (edge[0], edge[1]) in model['model_edges']:
        #             model['model_edges'].remove((edge[0], edge[1]))
        #             # Remove from list
        #             if verbose >= 3: print('[bnlearn] >Edge [%s <-> %s] [P=%g] is excluded because it was not significant (P<%.2f) with [%s]' %(edge[0], edge[1], model['independence_test'].iloc[idx]['p_value'], alpha, test))
        if len(idxrem)>0:
            if verbose >= 3: print(f'[bnlearn] >{sum(Irem)} edges are removed with P-value > {alpha} based on {test}')
            # Keek only the significant edges from the test statistics
            model['independence_test'] = model['independence_test'].loc[~Irem, :]
            model['independence_test'].reset_index(inplace=True, drop=True)
            model['model_edges'] = list(zip(model['independence_test']['source'], model['independence_test']['target']))
            # Update adjmat
            correction_matrix = bnlearn.vec2adjmat(model['independence_test']['source'], model['independence_test']['target'])
            # Sort the correction_matrix index and columns to match adjmat
            correction_matrix = correction_matrix.reindex(index=model['adjmat'].index, columns=model['adjmat'].columns)
            # Update the adjmat
            model['adjmat'] = model['adjmat'] * correction_matrix
            # Fill all NaN with zeros
            model['adjmat'].fillna(value=0, inplace=True)

    # Return
    return model


# %% Normalize weights in range
def _normalize_weights(weights, minscale=1, maxscale=5):
    if len(weights.shape)==1:
        weights=weights.reshape(-1, 1)

    from sklearn.preprocessing import MinMaxScaler
    weights = MinMaxScaler(feature_range=(minscale, maxscale)).fit_transform(weights).flatten()
    return(weights)


# %% Compute structure scores.
def structure_scores(model, df, scoring_method=['k2', 'bic', 'bdeu', 'bds'], verbose=3, **kwargs):
    """Compute structure scores.

    Each model can be scored based on its structure. However, the score doesn't have very straight forward
    interpretebility but can be used to compare different models. A higher score represents a better fit.
    This method only needs the model structure to compute the score. The structure score functionality
    can be found here: :func:`bnlearn.bnlearn.structure_scores`.

    Parameters
    ----------
    model: The bnlearn instance such as pgmpy.base.DAG or pgmpy.models.BayesianNetwork
        The model whose score needs to be computed.

    df: pd.DataFrame instance
        The dataset against which to score the model.

    scoring_method: str ( k2 | bdeu | bds | bic )
        The following four scoring methods are supported currently: 1) K2Score
        2) BDeuScore 3) BDsScore 4) BicScore

    kwargs: kwargs
        Any additional parameters parameters that needs to be passed to the
        scoring method.

    Returns
    -------
    Model score: float
        A score value for the model.

    Examples
    --------
    >>> import bnlearn as bn
    >>> # Load example dataset
    >>>
    >>> df = bn.import_example('sprinkler')
    >>> edges = [('Cloudy', 'Sprinkler'), ('Cloudy', 'Rain'), ('Sprinkler', 'Wet_Grass'), ('Rain', 'Wet_Grass')]
    >>>
    >>> # Make the Bayesian DAG
    >>> DAG = bn.make_DAG(edges)
    >>> model = bn.parameter_learning.fit(DAG, df)
    >>>
    >>> # Structure scores are stored in the model dictionary.
    >>> model['structure_scores']
    >>>
    >>> # Compute the structure score for as specific scoring-method.
    >>> bn.structure_scores(model, df, scoring_method="bic")
    """
    method = None
    show_message = True
    scores = {}
    # Get models and method
    if isinstance(model, dict):
        method = model.get('config')['method']
        model = model.get('model', None)
    if isinstance(scoring_method, str): scoring_method = [scoring_method]
    if verbose>=3: print('[bnlearn] >Compute structure scores for model comparison (higher is better).' %(scoring_method))

    # Return if method not supported
    if np.any(np.isin(method, ['cs', 'constraintsearch'])):
        if verbose>=2: print('[bnlearn] >Warning: Structure scoring could not be computed. Method [%s] not supported.' %(method))
        return scores

    # Compute structure scores
    if model is not None:
        for s in scoring_method:
            try:
                scores[s] = structure_score(model, df, scoring_method=s)
            except:
                if verbose>=2 and show_message:
                    print('[bnlearn] >WARNING> Skipping computing structure score for [%s].' %(s))
                    show_message=False
    # Return
    return scores


#%%
def get_parents(edges):
    """
    Build a dictionary mapping each node to its list of parents based on edges.

    Parameters
    ----------
    edges : list of tuple
        A list of (parent, child) tuples defining the DAG structure.

    Returns
    -------
    dict
        A dictionary mapping each node to its list of parents. Nodes with no parents are included with an empty list.

    Examples
    --------
    >>> # Import library
    >>> import bnlearn as bn
    >>> #
    >>> edges = [('Cloudy', 'Rain'), ('Cloudy', 'Sprinkler')]
    >>> bn.get_parents(edges)
    >>> # {'Rain': ['Cloudy'], 'Sprinkler': ['Cloudy'], 'Cloudy': []}

    """
    parent_dict = defaultdict(list)
    all_nodes = set()
    for parent, child in edges:
        parent_dict[child].append(parent)
        all_nodes.update([parent, child])
    for node in all_nodes:
        parent_dict.setdefault(node, [])
    return dict(parent_dict)


def probs_rulebook(node, rulebook, variable_card, all_combos):
    """
    Generate probability values for a node using a rulebook.

    Parameters
    ----------
    node : str
        The node name.
    rulebook : dict
        A dictionary mapping node names to callable functions that return probabilities.
    variable_card : int
        Number of possible values the node can take.
    all_combos : list of tuple
        All possible combinations of parent values.

    Returns
    -------
    list of list
        A 2D list where each sublist contains probabilities for a node state across all parent configurations.

    Raises
    ------
    ValueError
        If the rule function does not return a float or a list of correct length.

    Examples
    --------
    >>> # Import library
    >>> import bnlearn as bn
    >>> #
    >>> def rule(c): return [0.7, 0.2, 0.1] if c == 0 else [0.2, 0.5, 0.3]
    >>> bn.probs_rulebook('Sprinkler', {'Sprinkler': rule}, 3, [(0,), (1,)])
    >>> # [[0.7, 0.2], [0.2, 0.5], [0.1, 0.3]]

    """
    rule_func = rulebook[node]
    raw_probs = []
    for combo in all_combos:
        out = rule_func(*combo)
        if isinstance(out, (float, int)):
            row = [1 - out, out]
        elif isinstance(out, (list, tuple)) and len(out) == variable_card:
            row = list(out)
        else:
            raise ValueError(f"[bnlearn] >Rule for '{node}' must return float or list of len {variable_card}")
        raw_probs.append(row)
    probs = list(map(list, zip(*raw_probs)))
    return probs


def generate_cpt(node, parents, variable_card=2, rulebook=None, verbose=3):
    """
    Generate a TabularCPD object for a given node.

    Parameters
    ----------
    node : str
        The name of the node.
    parents : list of str
        List of parent nodes.
    variable_card : int, optional
        Number of possible values the node can take (default is 2).
    rulebook : dict, optional
        Dictionary of {node: callable} to generate conditional probabilities.
    verbose : int, optional
        Verbosity level (default is 3). If >= 3, prints the CPT.

    Returns
    -------
    TabularCPD
        A pgmpy TabularCPD object representing the CPT for the node.

    Examples
    --------
    >>> # Import library
    >>> import bnlearn as bn
    >>> #
    >>> edges = [('Cloudy', 'Rain'), ('Cloudy', 'Sprinkler')]
    >>> parents = bn.get_parents(edges)
    >>> # {'Rain': ['Cloudy'], 'Sprinkler': ['Cloudy'], 'Cloudy': []}
    >>> #
    >>> # Generate the CPTs
    >>> cpt_Rain = bn.generate_cpt('Rain', parents.get('Rain'), variable_card=2)
    >>> cpt_Sprinkler = bn.generate_cpt('Sprinkler', parents.get('Sprinkler'), variable_card=4)
    >>> #
    >>> # Create DAG with default CPD values
    >>> DAG = bn.make_DAG(edges, CPD=[cpt_Rain, cpt_Sprinkler])
    >>> bn.plot(DAG)

    Examples
    --------
    >>> # Example with DBN
    >>> #
    >>> # Import library
    >>> import bnlearn as bn
    >>> #
    >>> edges = [('Cloudy', 'Rain'), ('Cloudy', 'Sprinkler')]
    >>> edges = convert_edges_with_time_slice(edges)
    >>> #
    >>> # Get parents
    >>> parents = bn.get_parents(edges)
    >>> # {('Rain', 0): [('Cloudy', 0)], ('Sprinkler', 0): [('Cloudy', 0)], ('Cloudy', 0): []}
    >>> #
    >>> # Generate the CPTs
    >>> cpt_Rain = bn.generate_cpt(('Rain', 0), parents.get(('Rain', 0)), variable_card=2)
    >>> cpt_Rain = bn.generate_cpt(('Sprinkler', 0), parents.get(('Sprinkler', 0)), variable_card=2)
    >>> #
    >>> # Create DAG with default CPD values
    >>> DAG = bn.make_DAG(edges, CPD=[cpt_Rain, cpt_Sprinkler])
    >>> bn.plot(DAG)

    """
    parents = parents or []
    n_parents = len(parents)
    parent_card = [variable_card] * n_parents
    all_combos = list(product(range(variable_card), repeat=n_parents))
    n_combos = len(all_combos)

    if rulebook and node in rulebook:
        probs = probs_rulebook(node, rulebook, variable_card, all_combos)
    else:
        probs = [[1 / variable_card] * n_combos for _ in range(variable_card)]

    cpt = TabularCPD(variable=node,
                     variable_card=variable_card,
                     values=probs,
                     evidence=parents if parents else None,
                     evidence_card=parent_card if parents else None)

    if verbose >= 3:
        print(f'[bnlearn] >CPT for {node}:')
        print(cpt)

    return cpt
    # n_parents = len(parents)
    # parent_card = [variable_card] * n_parents
    # all_combos = list(product(range(variable_card), repeat=n_parents))
    # n_combos = len(all_combos)

    # if rulebook and node in rulebook:
    #     probs = probs_rulebook(node, rulebook, variable_card, all_combos)
    # else:
    #     probs = [[1 / variable_card] * n_combos for _ in range(variable_card)]

    # cpt = TabularCPD(variable=node,
    #                  variable_card=variable_card,
    #                  values=probs,
    #                  evidence=parents if parents else None,
    #                  evidence_card=parent_card if parents else None)

    # if verbose >= 3:
    #     print(f'[bnlearn] >CPT for {node}:')
    #     print(f'{cpt}')
    # return cpt


def build_cpts_from_structure(edges, variable_card=2, rulebook=None, methodtype=None, verbose=3):
    """
    Automatically generates placeholder CPTs for all nodes in a network structure.

    Parameters
    ----------
    edges : list of tuple
        A list of (parent, child) edges defining the structure of the network.
    variable_card : int, optional
        The number of values each variable can take (default is 2).
    rulebook : dict, optional
        A dictionary of {node: callable} functions returning probabilities.
        Each function must return either a float (binary) or a list of probabilities summing to 1.
    verbose : int, optional
        Verbosity level (default is 3).

    Returns
    -------
    list of TabularCPD
        A list of TabularCPD objects representing the conditional probability tables for each node.

    Examples
    --------
    >>> import bnlearn as bn
    >>> edges = [('Cloudy', 'Sprinkler'), ('Cloudy', 'Rain')]
    >>> #
    >>> CPD = bn.build_cpts_from_structure(edges)
    >>> # [<TabularCPD for Cloudy>, <TabularCPD for Sprinkler>, <TabularCPD for Rain>]
    >>> #
    >>> # Create DAG with default CPD values
    >>> DAG = bn.make_DAG(edges, CPD=CPD)
    >>> bn.plot(DAG)

    """
    if verbose>=3: print('[bnlearn]> Auto generate placeholders for the CPTs.')
    # Convert edges with time for DBN
    if methodtype=='DBN' and not has_valid_time_slice(edges):
        edges = convert_edges_with_time_slice(edges, verbose=verbose)

    cpts = []
    parents_map = get_parents(edges)

    for node, parents in parents_map.items():
        cpt = generate_cpt(node, parents, variable_card=variable_card, rulebook=rulebook, verbose=verbose)
        cpts.append(cpt)

    return cpts


# %%
# def set_logger(verbose: [str, int] = 'info'):
#     """Set the logger for verbosity messages.

#     Parameters
#     ----------
#     verbose : [str, int], default is 'info' or 20
#         Set the verbose messages using string or integer values.
#         * [0, 60, None, 'silent', 'off', 'no']: No message.
#         * [10, 'debug']: Messages from debug level and higher.
#         * [20, 'info']: Messages from info level and higher.
#         * [30, 'warning']: Messages from warning level and higher.
#         * [50, 'critical']: Messages from critical level and higher.

#     Returns
#     -------
#     None.

#     > # Set the logger to warning
#     > set_logger(verbose='warning')
#     > # Test with different messages
#     > logger.debug("Hello debug")
#     > logger.info("Hello info")
#     > logger.warning("Hello warning")
#     > logger.critical("Hello critical")

#     """
#     # Set 0 and None as no messages.
#     if (verbose==0) or (verbose is None):
#         verbose=60
#     # Convert str to levels
#     if isinstance(verbose, str):
#         levels = {'silent': 60,
#                   'off': 60,
#                   'no': 60,
#                   'debug': 10,
#                   'info': 20,
#                   'warning': 30,
#                   'critical': 50}
#         verbose = levels[verbose]

#     # Show examples
#     logger.setLevel(verbose)

# # %%
# def disable_tqdm():
#     """Set the logger for verbosity messages."""
#     return (True if (logger.getEffectiveLevel()>=30) else False)
