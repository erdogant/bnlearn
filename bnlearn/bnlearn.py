"""This package provides several bayesian techniques for structure learning, sampling and parameter learning.
  
    import bnlearn as bnlearn

    model            = bnlearn.import_DAG('sprinkler')
    df               = bnlearn.import_example()
    df               = bnlearn.sampling(model)
    q                = bnlearn.inference(model)
    model_sl         = bnlearn.structure_learning.fit(df)
    model_pl         = bnlearn.parameter_learning(model_sl['model'], df)
    [scores, adjmat] = bnlearn.compare_networks(model_sl, model_pl)


    Description
    -----------
    Learning a Bayesian network can be split into two problems:
        * Parameter learning: Given a set of data samples and a DAG that captures the dependencies between the variables,
          estimate the (conditional) probability distributions of the individual variables.
        * Structure learning: Given a set of data samples, estimate a DAG that captures the dependencies between the variables.
    Currently, the library supports:
        * Parameter learning for *discrete* nodes:
        * Maximum Likelihood Estimation
        * Bayesian Estimation
    Structure learning for *discrete*, *fully observed* networks:
        * Score-based structure estimation (BIC/BDeu/K2 score; exhaustive search, hill climb/tabu search)
        * Constraint-based structure estimation (PC)
        * Hybrid structure estimation (MMHC)


    Requirements
    ------------
    conda create -n env_BNLEARN python=3.6
    conda activate env_BNLEARN
    conda install pytorch -c pytorch
    pip install sklearn pandas tqdm funcsigs numpy
    pip install pgmpy==v0.1.9
    pip install networkx==v1.11
    pip install matplotlib==2.2.3


    Example
    -------
    import bnlearn as bnlearn

    # =========================================================================
    # CREATE SPRINKLER DAG
    model = bnlearn.import_DAG('sprinkler')
    bnlearn.plot(model)

    # =========================================================================
    # CREATE DATAFRAME FROM MODEL
    df = bnlearn.sampling(model, n=1000)

    # =========================================================================
    # PARAMETER LEARNING
    model = bnlearn.import_DAG('sprinkler', CPD=False)
    model_update = bnlearn.parameter_learning(model, df)
    bnlearn.plot(model_update)
    
    # =========================================================================
    # EXACT INFERENCE
    out = bnlearn.inference(model, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})
    out = bnlearn.inference(model, variables=['Wet_Grass','Rain'], evidence={'Sprinkler':1})
    
    # =========================================================================
    # LOAD BIF FILE
    model = bnlearn.import_DAG('alarm', verbose=0)
    bnlearn.plot(model, figsize=(20,12))
    
    df = bnlearn.sampling(model, n=1000)
    model_alarm = bnlearn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
    G = bnlearn.plot(model_alarm)
    bnlearn.plot(model, pos=G['pos'])

"""

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
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
# DAG
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
# SAMPLING
from pgmpy.sampling import BayesianModelSampling  # GibbsSampling
# PARAMETER LEARNING
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator  # ParameterEstimator
# INFERENCE
from pgmpy.inference import VariableElimination
from pgmpy import readwrite
# MICROSERVICES
import bnlearn.helpers.network as network
# ASSERTS
assert (nx.__version__)=='1.11', 'This function requires networkx to be v1.11. Try to: pip install networkx==v1.11'
assert (mpl.__version__)=='2.2.3', 'This function requires matplotlib to be v2.2.3. Try to: pip install matplotlib==v2.2.3'
curpath = os.path.dirname(os.path.abspath(__file__))
PATH_TO_DATA=os.path.join(curpath,'DATA')


# %% Exact inference using Variable Elimination
def inference(model, variables=None, evidence=None, verbose=3):
    """Inference using using Variable Elimination.

    Description
    -----------
    Inference is same as asking conditional probability questions to the models. 
    i.e., What is the probability of a sprinkler is on given that it is raining which is basically equivalent of asking $ P(g^1 | i^1) $. 
    Inference algorithms deals with efficiently finding these conditional probability queries.


    Parameters
    ----------
    model : dict
        Contains model

    variables : list of strings, optional (default: None)
        For exact inference, P(variables | evidence).
        ['Name_of_node_1']
        ['Name_of_node_1', 'Name_of_node_2']

    evidence : dict, optional  (default: None)
        For exact inference, P(variables | evidence).
        {'Rain':1}
        {'Rain':1, 'Sprinkler':0, 'Cloudy':1}

    verbose : int [0-5] (default: 3)
        Print messages to screen.
        0: NONE
        1: ERROR
        2: WARNING
        3: INFO (default)
        4: DEBUG
        5: TRACE

    Returns
    -------
    None.


    Description Extensive
    ---------------------
    There are two main categories for inference algorithms:
        1. Exact Inference: These algorithms find the exact probability values for our queries.
        2. Approximate Inference: These algorithms try to find approximate values by saving on computation.

    Exact Inference
        There are multiple algorithms for doing exact inference.

    Two common Inference algorithms with variable Elimination
        1. Clique Tree Belief Propagation
        2. Variable Elimination

    The basic concept of variable elimination is same as doing marginalization over Joint Distribution.
    But variable elimination avoids computing the Joint Distribution by doing marginalization over much smaller factors.
    So basically if we want to eliminate $ X $ from our distribution, then we compute
    the product of all the factors involving $ X $ and marginalize over them,
    thus allowing us to work on much smaller factors.

    In the above equation we can see that we pushed the summation inside and operated
    the summation only factors that involved that variable and hence avoiding computing the
    complete joint distribution.
    """
    
    model_infer = VariableElimination(model['model'])
    # Computing the probability of Wet Grass given Rain.
    q = model_infer.query(variables=variables, evidence=evidence)
    print(q)
    # for varname in variables: 
        # print(q[varname])
    return(q)


# %% Sampling from model
def parameter_learning(model, df, methodtype='bayes', verbose=3):
    """Parameter Learning.
    
    Description
    ----------
    Parameter learning is the task to estimate the values of the conditional probability distributions (CPDs), 
    for the variables cloudy, sprinkler, rain and wet grass.
    State counts
        To make sense of the given data, we can start by counting how often each state of the variable occurs.
        If the variable is dependent on parents, the counts are done conditionally on the parents states,
        i.e. for seperately for each parent configuration


    Parameters
    ----------
    model       : [DICT] Contains model and adjmat.

    df          : [pd.DataFrame] Pandas DataFrame containing the data
                   f1  ,f2  ,f3
                s1 0   ,0   ,1
                s2 0   ,1   ,0
                s3 1   ,1   ,0

    methodtype  : [STRING] strategy for parameter learning.
                'nl' or 'maximumlikelihood' (default) :Learning CPDs using Maximum Likelihood Estimators
                'bayes' :Bayesian Parameter Estimation

    verbose : int [0-5] (default: 3)
        Print messages to screen.
        0: NONE
        1: ERROR
        2: WARNING
        3: INFO (default)
        4: DEBUG
        5: TRACE

    Returns
    -------
    model

    """

    config = dict()
    config['verbose'] = verbose
    config['method'] = methodtype
    model = model['model']
    if verbose>=3: print('[BNLEARN][PARAMETER LEARNING] Computing parameters using [%s]' %(config['method']))

#    pe = ParameterEstimator(model, df)
#    print("\n", pe.state_counts('Cloudy'))
#    print("\n", pe.state_counts('Sprinkler'))

    """
    Maximum Likelihood Estimation
        A natural estimate for the CPDs is to simply use the *relative frequencies*,
        with which the variable states have occured. We observed x cloudy` among a total of `all clouds`,
        so we might guess that about `50%` of `cloudy` are `sprinkler or so.
        According to MLE, we should fill the CPDs in such a way, that $P(\text{data}|\text{model})$ is maximal.
        This is achieved when using the *relative frequencies*.

    While very straightforward, the ML estimator has the problem of *overfitting* to the data.
    If the observed data is not representative for the underlying distribution, ML estimations will be extremly far off.
    When estimating parameters for Bayesian networks, lack of data is a frequent problem.
    Even if the total sample size is very large, the fact that state counts are done conditionally
    for each parents configuration causes immense fragmentation.
    If a variable has 3 parents that can each take 10 states, then state counts will
    be done seperately for `10^3 = 1000` parents configurations.
    This makes MLE very fragile and unstable for learning Bayesian Network parameters.
    A way to mitigate MLE's overfitting is *Bayesian Parameter Estimation*.
    """

    # Learning CPDs using Maximum Likelihood Estimators
    if config['method']=='ml' or config['method']=='maximumlikelihood':
        mle = MaximumLikelihoodEstimator(model, df)
        for node in mle.state_names:
            print(mle.estimate_cpd(node))


    """
    Bayesian Parameter Estimation
        The Bayesian Parameter Estimator starts with already existing prior CPDs,
        that express our beliefs about the variables *before* the data was observed.
        Those "priors" are then updated, using the state counts from the observed data.
    
    One can think of the priors as consisting in *pseudo state counts*, that are added
    to the actual counts before normalization. Unless one wants to encode specific beliefs
    about the distributions of the variables, one commonly chooses uniform priors,
    i.e. ones that deem all states equiprobable.
    
    A very simple prior is the so-called *K2* prior, which simply adds `1` to the count of every single state.
    A somewhat more sensible choice of prior is *BDeu* (Bayesian Dirichlet equivalent uniform prior).
    For BDeu we need to specify an *equivalent sample size* `N` and then the pseudo-counts are
    the equivalent of having observed `N` uniform samples of each variable (and each parent configuration).
    """
    if config['method']=='bayes':
        model.fit(df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=1000) # default equivalent_sample_size=5

        for cpd in model.get_cpds():
            if verbose>=3: print("CPD of {variable}:".format(variable=cpd.variable))
            if verbose>=3: print(cpd)

    return(model)


# %% Sampling from model
def sampling(model, n=1000, verbose=3):
    """Sampling based on DAG.

    Parameters
    ----------
    model : dict
        Contains model and adjmat

    n : int
        Number of samples to generate
        n=1000 (default)

    verbose : int [0-5] (default: 3)
        Print messages to screen.
        0: NONE
        1: ERROR
        2: WARNING
        3: INFO (default)
        4: DEBUG
        5: TRACE

    Returns
    -------
    Pandas DataFrame

    """
    assert n>0, 'n must be 1 or larger'
    assert 'BayesianModel' in str(type(model['model'])), 'Model must contain DAG from BayesianModel. Note that <misarables> example does not include DAG.'

    # http://pgmpy.org/sampling.html
    inference = BayesianModelSampling(model['model'])
    # inference = GibbsSampling(model)
    # Forward sampling and make dataframe
    df=inference.forward_sample(size=n, return_type='dataframe')
    return(df)


# %% Make DAG
def import_DAG(filepath='sprinkler', CPD=True, verbose=3):
    """

    Parameters
    ----------
    filepath : String, optional (default: 'sprinkler')
        Pre-defined examples are depicted below, or provide the absolute file path to the .bif model file.
        'sprinkler'(default)
        'alarm'
        'andes'
        'asia'
        'pathfinder'
        'sachs'
        'miserables'
        'filepath/to/model.bif'

    CPD : Bool, optional (default: True)
        Directed Acyclic Graph (DAG).
        True (default)
        False

    verbose : int [0-5] (default: 3)
        Print messages to screen.
        0: NONE
        1: ERROR
        2: WARNING
        3: INFO (default)
        4: DEBUG
        5: TRACE
                   

    Returns
    -------
    None.

    """
    out=dict()
    model=None
    filepath=filepath.lower()

    # Load data
    if filepath=='sprinkler':
        model = _DAG_sprinkler(CPD=CPD, verbose=verbose)
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
            if verbose>=3: print('[BNLEARN] Filepath does not exist! <%s>' %(filepath))
            return(out)

    # check_model check for the model structure and the associated CPD and returns True if everything is correct otherwise throws an exception
    if not isinstance(model, type(None)) and verbose>=3:
        if CPD:
            print('[BNLEARN] Model correct: %s' %(model.check_model()))
            for cpd in model.get_cpds():
                print("CPD of {variable}:".format(variable=cpd.variable))
                print(cpd)

            print('[BNLEARN] Nodes: %s' %(model.nodes()))
            print('[BNLEARN] Edges: %s' %(model.edges()))
            print('[BNLEARN] Independencies:\n%s' %(model.get_independencies()))

    # Setup simmilarity matrix
    adjmat = pd.DataFrame(data=False, index=model.nodes(), columns=model.nodes()).astype('Bool')
    # Fill adjmat with edges
    edges=model.edges()
    for edge in edges:
        adjmat.loc[edge[0],edge[1]]=True

    out['model']=model
    out['adjmat']=adjmat
    return(out)


# %% Model Sprinkler
def _DAG_sprinkler(CPD=True, verbose=3):
    """Create DAG-model for the sprinkler example.

    Parameters
    ----------
    CPD : Bool, optional (default: True)
        Directed Acyclic Graph (DAG).
        True (default)
        False

    verbose : int [0-5], optional (default: 3)
        Print messages.
        0: (default)
        1: ERROR
        2: WARN
        3: INFO
        4: DEBUG

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
    """
    Returns the fitted bayesian model

    Example
    ----------
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
    """


    Parameters
    ----------
    adjmat : numpy aray
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
        Results of model 1.
    model_2 : dict
        Results of model 2.
    pos : graph, optional (default: None)
        Coordinates of the network. If there are provided, the same structure will be used to plot the network.
    showfig : Bool, optional (default: True)
        Show figure.
    figsize : tuple, optional (default: (15,8))
        Figure size.
    verbose : int [0-5], optional (default: 3)
        Print messages.
        0: (default)
        1: ERROR
        2: WARN
        3: INFO
        4: DEBUG

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
        Learned model from the .fit() function.
    pos : graph, optional (default: None)
        Coordinates of the network. If there are provided, the same structure will be used to plot the network.
    scale : int, optional (default: 1)
        Scaling parameter for the network. A larger number will linearily increase the network.
    figsize : tuple, optional (default: (15,8))
        Figure size.
    verbose : int [0-5], optional (default: 3)
        Print messages.
        0: (default)
        1: ERROR
        2: WARN
        3: INFO
        4: DEBUG


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
        if verbose>=3: print('[BNLEARN.plot] Making plot based on BayesianModel')
        # positions for all nodes
        pos = network.graphlayout(model, pos=pos, scale=scale, layout=layout)
        # Add directed edge with weigth
        # edges=model.edges()
        edges=[*model.edges()]
        for i in range(len(edges)):
            G.add_edge(edges[i][0], edges[i][1], weight=1, color='k')
    elif 'networkx' in str(type(model)):
        if verbose>=3: print('[BNLEARN.plot] Making plot based on networkx model')
        G=model
        pos = network.graphlayout(G, pos=pos, scale=scale, layout=layout)
    else:
        if verbose>=3: print('[BNLEARN.plot] Making plot based on adjacency matrix')
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
    df : pd.DataFrame
        Dataframe.

    """
    curpath = os.path.dirname(os.path.abspath(__file__))
    PATH_TO_DATA=os.path.join(curpath,'data','sprinkler_data.zip')
    if os.path.isfile(PATH_TO_DATA):
        df=pd.read_csv(PATH_TO_DATA, sep=',')
        return df
    else:
        print('[KM] Oops! Example data not found! Try to get it at: www.github.com/erdogant/bnlearn/')
        return None

#%% Convert Adjmat to graph (G)
#def adjmat2graph(adjmat):
#    G = nx.DiGraph() # Directed graph
#    # Convert adjmat to source target
#    df_edges=adjmat.stack().reset_index()
#    df_edges.columns=['source', 'target', 'weight']
#    df_edges['weight']=df_edges['weight'].astype(float)
#    
#    # Add directed edge with weigth
#    for i in range(df_edges.shape[0]):
#        if df_edges['weight'].iloc[i]!=0:
#            # Setup color
#            if df_edges['weight'].iloc[i]==1:
#                color='k'
#            elif df_edges['weight'].iloc[i]>1:
#                color='r'
#            elif df_edges['weight'].iloc[i]<0:
#                color='b'
#            else:
#                color='p'
#            
#            # Create edge in graph
#            G.add_edge(df_edges['source'].iloc[i], df_edges['target'].iloc[i], weight=np.abs(df_edges['weight'].iloc[i]), color=color)    
#    # Return
#    return(G)
    