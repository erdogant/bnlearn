"""Structure learning. Given a set of data samples, estimate a DAG that captures the dependencies between the variables. """
# ------------------------------------
# Name        : structure_learning.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------


# %% Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pgmpy
from pgmpy.estimators import BDeuScore, K2Score, BicScore
from pgmpy.estimators import ExhaustiveSearch, HillClimbSearch, ConstraintBasedEstimator
# from packaging import version
from bnlearn.bnlearn import _dag2adjmat


# %% Structure Learning
def fit(df, methodtype='hc', scoretype='bic', black_list=None, white_list=None, bw_list_method='enforce', max_indegree=None, epsilon=1e-4, max_iter=1e6, verbose=3):
    """Structure learning fit model.

    Description
    -----------
    Search strategies for structure learning
    The search space of DAGs is super-exponential in the number of variables and the above scoring functions allow for local maxima.

    To learn model structure (a DAG) from a data set, there are three broad techniques:
        1. Score-based structure learning (BIC/BDeu/K2 score; exhaustive search, hill climb/tabu search)
            a. exhaustivesearch
            b. hillclimbsearch
        2. Constraint-based structure learning (PC)
            a. chi-square test
        3. Hybrid structure learning (The combination of both techniques) (MMHC)

    Score-based Structure Learning
    This approach construes model selection as an optimization task. It has two building blocks:
    A scoring function sD:->R that maps models to a numerical score, based on how well they fit to a given data set D.
    A search strategy to traverse the search space of possible models M and select a model with optimal score.
    Commonly used scoring functions to measure the fit between model and data are Bayesian Dirichlet scores such as BDeu or K2
    and the Bayesian Information Criterion (BIC, also called MDL).
    As before, BDeu is dependent on an equivalent sample size.

    Parameters
    ----------
    df : pd.DataFrame()
        Input dataframe.
    methodtype : str, (default : 'hc')
        String Search strategy for structure_learning.
        'hc' or 'hillclimbsearch' (default)
        'ex' or 'exhaustivesearch'
        'cs' or 'constraintsearch'
    scoretype : str, (default : 'bic')
        Scoring function for the search spaces.
        'bic', 'k2', 'bdeu'
    black_list : List or None, (default : None)
        If a list of edges is provided as black_list, they are excluded from the search and the resulting model will not contain any of those edges. The default is None.
        Works only in case of methodtype='hc'. See also paramter: `bw_list_method`
    white_list : List or None, (default : None)
        If a list of edges is provided as white_list, the search is limited to those edges. The resulting model will then only contain edges that are in white_list. The default is None.
        Works only in case of methodtype='hc'/ See also paramter: `bw_list_method`
    bw_list_method : str, (default : 'enforce')
        'enforce' : A list of edges can optionally be passed as `black_list` or `white_list` to exclude those edges or to limit the search. This option is limited to only methodtype='hc'
        'filter' : Filter the dataframe based on `black_list` or `white_list`. Filtering can be done for every methodtype/scoretype.
    max_indegree : int, (default : None)
        If provided and unequal None, the procedure only searches among models where all nodes have at most max_indegree parents. (only in case of methodtype='hc')
    epsilon: float (default: 1e-4)
        Defines the exit condition. If the improvement in score is less than `epsilon`, the learned model is returned. (only in case of methodtype='hc')
    max_iter: int (default: 1e6)
        The maximum number of iterations allowed. Returns the learned model when the number of iterations is greater than `max_iter`. (only in case of methodtype='hc')
    verbose : int, (default : 3)
        Print progress to screen.
        0: NONE
        1: ERROR
        2: WARNING
        3: INFO (default)
        4: DEBUG
        5: TRACE

    Returns
    -------
    dict with model.

    Examples
    --------
    >>> # Load asia DAG
    >>> model = bnlearn.import_DAG('asia')
    >>>
    >>> # plot ground truth
    >>> G = bnlearn.plot(model)
    >>>
    >>> # Sampling
    >>> df = bnlearn.sampling(model, n=10000)
    >>>
    >>> # Structure learning of sampled dataset
    >>> model_sl = bnlearn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
    >>>
    >>> # Plot based on structure learning of sampled data
    >>> bnlearn.plot(model_sl, pos=G['pos'])
    >>>
    >>> # Compare networks and make plot
    >>> bnlearn.compare_networks(model, model_sl, pos=G['pos'])

    """
    assert isinstance(pd.DataFrame(), type(df)), 'df must be of type pd.DataFrame()'
    assert (scoretype=='bic') | (scoretype=='k2') | (scoretype=='bdeu'), 'scoretype must be string: "bic", "k2" or "bdeu"'
    assert (methodtype=='hc') | (methodtype=='ex')| (methodtype=='cs') | (methodtype=='exhaustivesearch')| (methodtype=='hillclimbsearch')| (methodtype=='constraintsearch'), 'Methodtype string is invalid'  # noqa
    if isinstance(white_list, str): white_list=[white_list]
    if isinstance(black_list, str): black_list=[black_list]
    if (white_list is not None) and len(white_list)==0: white_list=None
    if (black_list is not None) and len(black_list)==0: black_list=None
    if (bw_list_method is None) : bw_list_method='enforce'

    config = {}
    config['verbose'] = verbose
    config['method'] = methodtype
    config['scoring'] = scoretype
    config['black_list'] = black_list
    config['white_list'] = white_list
    config['bw_list_method'] = bw_list_method
    config['max_indegree'] = max_indegree
    config['epsilon'] = epsilon
    config['max_iter'] = max_iter

    # Show warnings
    # PGMPY_VER = version.parse(pgmpy.__version__)>version.parse("0.1.9")  # Can be be removed if pgmpy >v0.1.9
    # if (not PGMPY_VER) and ((black_list is not None) or (white_list is not None)):
        # if config['verbose']>=2: print('[bnlearn] >Warning: black_list and white_list only works for pgmpy > v0.1.9')  # Can be be removed if pgmpy >v0.1.9
    if df.shape[1]>10 and df.shape[1]<15:
        if config['verbose']>=2: print('[bnlearn] >Warning: Computing DAG with %d nodes can take a very long time!' %(df.shape[1]))
    if (black_list is not None) and methodtype!='hc':
        if config['verbose']>=2: print('[bnlearn] >Warning: blacklist only works in case of methodtype="hc"')
    if (white_list is not None) and methodtype!='hc':
        if config['verbose']>=2: print('[bnlearn] >Warning: white_list only works in case of methodtype="hc"')
    if (max_indegree is not None) and methodtype!='hc':
        if config['verbose']>=2: print('[bnlearn] >Warning: max_indegree only works in case of methodtype="hc"')

    if config['verbose']>=3: print('[bnlearn] >Computing best DAG using [%s]' %(config['method']))

    # Make sure columns are of type string
    df.columns = df.columns.astype(str)
    # Filter on white_list and black_list
    df = _white_black_list(df, white_list, black_list, bw_list_method=config['bw_list_method'], verbose=verbose)

    # ExhaustiveSearch can be used to compute the score for every DAG and returns the best-scoring one:
    if config['method']=='ex' or config['method']=='exhaustivesearch':
        """The first property makes exhaustive search intractable for all but very small networks,
        the second prohibits efficient local optimization algorithms to always find the optimal structure.
        Thus, identifiying the ideal structure is often not tractable.
        Despite these bad news, heuristic search strategies often yields good results
        If only few nodes are involved (read: less than 5)."""
        if (df.shape[1]>15) and (config['verbose']>=3):
            print('[bnlearn] >Warning: Structure learning with more then 15 nodes is computationally not feasable with exhaustivesearch. Use hillclimbsearch or constraintsearch instead!!')  # noqa
        out = _exhaustivesearch(df, scoretype=config['scoring'], verbose=config['verbose'])

    # HillClimbSearch
    if config['method']=='hc' or config['method']=='hillclimbsearch':
        out = _hillclimbsearch(df, 
                               scoretype=config['scoring'],
                               black_list=config['black_list'],
                               white_list=config['white_list'],
                               max_indegree=config['max_indegree'],
                               bw_list_method=config['bw_list_method'],
                               epsilon=config['epsilon'],
                               max_iter=config['max_iter'],
                               verbose=config['verbose'],
                               )

    # Constraint-based Structure Learning
    if config['method']=='cs' or config['method']=='constraintsearch':
        """Constraint-based Structure Learning
        A different, but quite straightforward approach to build a DAG from data is this:
        Identify independencies in the data set using hypothesis tests
        Construct DAG (pattern) according to identified independencies (Conditional) Independence Tests
        Independencies in the data can be identified using chi2 conditional independence tests."""
        out = _constraintsearch(df, verbose=config['verbose'])

    # Setup simmilarity matrix
    adjmat = _dag2adjmat(out['model'])

    # adjmat = pd.DataFrame(data=False, index=out['model'].nodes(), columns=out['model'].nodes()).astype('Bool')
    # # Fill adjmat with edges
    # edges = out['model'].edges()
    # for edge in edges:
    #     adjmat.loc[edge[0],edge[1]]=True
    # adjmat.index.name = 'source'
    # adjmat.columns.name = 'target'

    # Store
    out['adjmat'] = adjmat
    out['config'] = config
    
    # return
    return(out)


# %% white_list and black_list
def _white_black_list(df, white_list, black_list, bw_list_method='enforce', verbose=3):
    if bw_list_method=='filter':
        # Keep only variables that are in white_list.
        if white_list is not None:
            if verbose>=3: print('[bnlearn] >Filter variables on white_list..')
            white_list = [x.lower() for x in white_list]
            Iloc = np.isin(df.columns.str.lower(), white_list)
            df = df.loc[:,Iloc]

        # Exclude variables that are in black_list.
        if black_list is not None:
            if verbose>=3: print('[bnlearn] >Filter variables on black_list..')
            black_list = [x.lower() for x in black_list]
            Iloc = ~np.isin(df.columns.str.lower(), black_list)
            df = df.loc[:,Iloc]

        if (white_list is not None) or (black_list is not None):
            if verbose>=3: print('[bnlearn]  >Number of features after white/black listing: %d' %(df.shape[1]))
        if df.shape[1]<=1: raise Exception('[bnlearn] >Error: [%d] variables are remaining. A minimum of 2 would be nice.' %(df.shape[1]))
    return df


# %% Constraint-based Structure Learning
def _constraintsearch(df, significance_level=0.05, verbose=3):
    """Contrain search.

    test_conditional_independence() returns a tripel (chi2, p_value, sufficient_data),
    consisting in the computed chi2 test statistic, the p_value of the test, and a heuristig
    flag that indicates if the sample size was sufficient.
    The p_value is the probability of observing the computed chi2 statistic (or an even higher chi2 value),
    given the null hypothesis that X and Y are independent given Zs.
    This can be used to make independence judgements, at a given level of significance.
    """
    out=dict()
    # Set search algorithm
    model = ConstraintBasedEstimator(df)

    # Some checks for dependency
    #    print(_is_independent(est, 'Sprinkler', 'Rain', significance_level=significance_level))
    #    print(_is_independent(est, 'Cloudy', 'Rain', significance_level=significance_level))
    #    print(_is_independent(est, 'Sprinkler', 'Rain',  ['Wet_Grass'], significance_level=significance_level))

    """
    DAG (pattern) construction
    With a method for independence testing at hand, we can construct a DAG from the data set in three steps:
        1. Construct an undirected skeleton - `estimate_skeleton()`
        2. Orient compelled edges to obtain partially directed acyclid graph (PDAG; I-equivalence class of DAGs) - `skeleton_to_pdag()`
        3. Extend DAG pattern to a DAG by conservatively orienting the remaining edges in some way - `pdag_to_dag()`

        Step 1.&2. form the so-called PC algorithm, see [2], page 550. PDAGs are `DirectedGraph`s, that may contain both-way edges, to indicate that the orientation for the edge is not determined.
    """
    # Estimate using chi2
    [skel, seperating_sets] = model.estimate_skeleton(significance_level=significance_level)

    print("Undirected edges: ", skel.edges())
    pdag = model.skeleton_to_pdag(skel, seperating_sets)
    print("PDAG edges: ", pdag.edges())
    dag = model.pdag_to_dag(pdag)
    print("DAG edges: ", dag.edges())

    out['undirected'] = skel
    out['undirected_edges'] = skel.edges()
    out['pdag'] = pdag
    out['pdag_edges'] = pdag.edges()
    out['dag'] = dag
    out['dag_edges'] = dag.edges()

    # Search using "estimate()" method provides a shorthand for the three steps above and directly returns a "BayesianModel"
    best_model = model.estimate(significance_level=significance_level)
    out['model'] = best_model
    out['model_edges'] = best_model.edges()

    print(best_model.edges())

    """
    PC PDAG construction is only guaranteed to work under the assumption that the
    identified set of independencies is *faithful*, i.e. there exists a DAG that
    exactly corresponds to it. Spurious dependencies in the data set can cause
    the reported independencies to violate faithfulness. It can happen that the
    estimated PDAG does not have any faithful completions (i.e. edge orientations
    that do not introduce new v-structures). In that case a warning is issued.
    """
    return(out)


# %% hillclimbsearch
def _hillclimbsearch(df, scoretype='bic', black_list=None, white_list=None, max_indegree=None, epsilon=1e-4, max_iter=1e6, bw_list_method='enforce', verbose=3):
    """ heuristic hill climb searches for DAGs, to learn network structure from data. `estimate` attempts to find a model with optimal score.
    
    Description
    -----------
    Performs local hill climb search to estimates the `DAG` structure
    that has optimal score, according to the scoring method supplied in the constructor.
    Starts at model `start` and proceeds by step-by-step network modifications
    until a local maximum is reached. Only estimates network structure, no parametrization.
        
    Once more nodes are involved, one needs to switch to heuristic search.
    HillClimbSearch implements a greedy local search that starts from the DAG
    "start" (default: disconnected DAG) and proceeds by iteratively performing
    single-edge manipulations that maximally increase the score.
    The search terminates once a local maximum is found.
    
    For details on scoring see Koller & Friedman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
    If a number `max_indegree` is provided, only modifications that keep the number
    of parents for each node below `max_indegree` are considered. A list of
    edges can optionally be passed as `black_list` or `white_list` to exclude those
    edges or to limit the search.
    
    """

    out=dict()
    # Set scoring type
    scoring_method = _SetScoringType(df, scoretype)
    # Set search algorithm
    model = HillClimbSearch(df, scoring_method=scoring_method)

    # Compute best DAG
    # PGMPY_VER = version.parse(pgmpy.__version__)>version.parse("0.1.9")
    # if PGMPY_VER:
    #     best_model = model.estimate(max_indegree=max_indegree, black_list=black_list, white_list=white_list)
    # else:
    #     best_model = model.estimate(max_indegree=max_indegree)  # Can be be removed if pgmpy >v0.1.9

    # Compute best DAG
    if bw_list_method=='enforce':
        if (black_list is not None) or (white_list is not None):
            if verbose>=3: print('[bnlearn]  >Enforcing nodes based on black_list and/or white_list.')
        best_model = model.estimate(max_indegree=max_indegree, epsilon=epsilon, max_iter=max_iter, black_list=black_list, white_list=white_list)
    else:
        # At this point, variables are readily filtered based on bw_list_method or not (if nothing defined).
        best_model = model.estimate(max_indegree=max_indegree, epsilon=epsilon, max_iter=max_iter)

    # Store
    out['model']=best_model
    out['model_edges']=best_model.edges()
    # Return
    return(out)


# %% ExhaustiveSearch
def _exhaustivesearch(df, scoretype='bic', return_all_dags=False, verbose=3):
    out=dict()

    # Set scoring type
    scoring_method = _SetScoringType(df, scoretype)
    # Exhaustive search across all dags
    model = ExhaustiveSearch(df, scoring_method=scoring_method)
    # Compute best DAG
    best_model = model.estimate()
    # Store
    out['model']=best_model
    out['model_edges']=best_model.edges()

    # Compute all possible DAGs
    if return_all_dags:
        out['scores']=[]
        out['dag']=[]
        # print("\nAll DAGs by score:")
        for [score, dag] in reversed(model.all_scores()):
            out['scores'].append(score)
            out['dag'].append(dag)
            # print(score, dag.edges())

        plt.plot(out['scores'])
        plt.show()

    return(out)


# %% Set scoring type
def _SetScoringType(df, scoretype, verbose=3):
    if verbose>=3: print('[bnlearn]  >Set scoring type at [%s]' %(scoretype))

    if scoretype=='bic':
        scoring_method = BicScore(df)
    elif scoretype=='k2':
        scoring_method = K2Score(df)
    elif scoretype=='bdeu':
        scoring_method = BDeuScore(df, equivalent_sample_size=5)

    return(scoring_method)


# %%
def _is_independent(model, X, Y, Zs=[], significance_level=0.05):
    return model.test_conditional_independence(X, Y, Zs)[1] >= significance_level
