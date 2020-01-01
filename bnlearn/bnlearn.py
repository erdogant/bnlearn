""" This function computes several bayes techniques for structure learning, sampling and parameter learning.
  
  import bnlearn as bnlearn

  model            = bnlearn.load_examples('sprinkler')
  df               = bnlearn.sampling(model)
  q                = bnlearn.inference(model)
  [out, model_1]   = bnlearn.structure_learning(df)
  model_1          = bnlearn.parameter_learning(model, df)
  [scores, adjmat] = bnlearn.compare_networks(model['adjmat'], out['adjmat'])

  
 INPUT:
   df:             [pd.DataFrame] Pandas DataFrame

                      f1  ,f2  ,f3
                   s1 0   ,0   ,1
                   s2 0   ,1   ,0
                   s3 1   ,1   ,0

 OPTIONAL

   methodtype:     [string]: Search strategy for structure_learning
                   'ex' or 'exhaustivesearch'          : Exhaustive search for very small networks
                   'hc' or 'hillclimbsearch' (default) : HillClimbSearch implements a greedy local search if many more nodes are involved
                   'cs' or 'constraintsearch'          : Constraint-based Structure Learning by first identifing independencies in the data set using hypothesis test (chi2)

   scoretype:      [string]: Scoring function for the search spaces
                   'bic'          : 
                   'k2' (default) : 
                   'bdeu'         : 

   n:              [int]: Number of samples to generate from the model for the dataframe
                   n=1000 (default)

   variables:      [list of strings]: For exact inference, P(variables | evidence)
                   ['Name_of_node_1']
                   ['Name_of_node_1', 'Name_of_node_2']

   evidence:       [dict]: For exact inference, P(variables | evidence)
                   {'Rain':1}
                   {'Rain':1, 'Sprinkler':0, 'Cloudy':1}

   CPD:            [Bool]: DAG is created with (True) or without (False) CPDs.
                   True (default)
                   False

   pos:            [Graph]: Coordinates of the network. If there are provided, the same structure will be used to plot the network.
                   None (default)
                   

   verbose:        Integer [0..5] if verbose >= DEBUG: print('debug message')
                   0: (default)
                   1: ERROR
                   2: WARN
                   3: INFO
                   4: DEBUG

   #### DAG_example ####

   models:          [String]: Models that can be used for testing
                   'sprinkler'
                   'alarm'
                   'andes'
                   'asia'
                   'pathfinder'
                   'sachs'
                   'miserables'

                       
                   

 OUTPUT
	output

 DESCRIPTION
   This function provides several bayesian techniques for structure learning, sampling and parameter learning
   # http://pgmpy.org/estimators.html#structure-score
   # https://programtalk.com/python-examples/pgmpy.factors.discrete.TabularCPD/
   # http://www.bnlearn.com/bnrepository/
   # http://www.bnlearn.com/
    
   Learning a Bayesian network can be split into two problems:
      **Parameter learning:** Given a set of data samples and a DAG that captures the dependencies between the variables, estimate the (conditional) probability distributions of the individual variables.
      **Structure learning:** Given a set of data samples, estimate a DAG that captures the dependencies between the variables.
 
   This notebook aims to illustrate how parameter learning and structure learning can be done with pgmpy.
   Currently, the library supports:
     - Parameter learning for *discrete* nodes:
       - Maximum Likelihood Estimation
       - Bayesian Estimation
     - Structure learning for *discrete*, *fully observed* networks:
       - Score-based structure estimation (BIC/BDeu/K2 score; exhaustive search, hill climb/tabu search)
       - Constraint-based structure estimation (PC)
       - Hybrid structure estimation (MMHC)
           

 REQUIREMENTS
   conda create -n env_BNLEARN python=3.7
   conda activate env_BNLEARN
   conda install pytorch torchvision -c pytorch
   conda install spyder
   pip install sklearn pandas tqdm funcsigs 
   pip install pgmpy
   pip install networkx==v1.11
   pip install matplotlib==2.2.3

 EXAMPLE
   import bnlearn as bnlearn

   # ==========================================================================
   # CREATE SPRINKLER DAG
   model = bnlearn.load_examples('sprinkler')
   bnlearn.plot(model)

   # ==========================================================================
   # CREATE DATAFRAME FROM MODEL
   df=bnlearn.sampling(model, n=1000)

   # ==========================================================================
   # PARAMETER LEARNING
   model   = bnlearn.load_examples('sprinkler', CPD=False)
   model_update = bnlearn.parameter_learning(model, df)
   bnlearn.plot(model_update)

   # ==========================================================================
   # EXACT INFERENCE
   out   = bnlearn.inference(model, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})
   out   = bnlearn.inference(model, variables=['Wet_Grass','Rain'], evidence={'Sprinkler':1})

   # ==========================================================================
   # SPRINKLER EXAMPLE
   df = pd.read_csv('../DATA/NETWORKS/bayesian/SPRINKLER/sprinkler_data.csv')

   out  = bnlearn.structure_learning(df, methodtype='hc', scoretype='bic')
   out  = bnlearn.structure_learning(df, methodtype='hc', scoretype='k2')
   out  = bnlearn.structure_learning(df, methodtype='hc', scoretype='bdeu')
   out  = bnlearn.structure_learning(df, methodtype='ex', scoretype='bic')
   out  = bnlearn.structure_learning(df, methodtype='ex', scoretype='k2')
   out  = bnlearn.structure_learning(df, methodtype='ex', scoretype='bdeu')
   out  = bnlearn.structure_learning(df, methodtype='cs')

   G=bnlearn.plot(model)
   bnlearn.plot(out['model'], pos=G['pos'])
   bnlearn.plot(out['adjmat'], pos=G['pos'])


   # ==========================================================================
   # LOAD BIF FILE
   model=bnlearn.load_examples('alarm', verbose=0)
   bnlearn.plot(model, width=20, height=12)
   
   df=bnlearn.sampling(model, n=1000)
   out = bnlearn.structure_learning(df, methodtype='hc', scoretype='bic')
   G=bnlearn.plot(out['model'])
   bnlearn.plot(model, pos=G['pos'])


   # ==========================================================================
   # EXAMPLE LARGE AMOUNT OF NODES
   df=pd.read_csv('../DATA/OTHER/titanic/titanic_train.csv')
   df = df[['Survived','Sex','Pclass']]

   A = bnlearn.structure_learning(df, methodtype='hc', scoretype='bic')
   A = bnlearn.structure_learning(df, methodtype='cs')

   from TRANSFORMERS.df2onehot import df2onehot
   [_, Xhot, Xlabx, _]=df2onehot(df, min_y=10, hot_only=True)
#   Xhot = Xhot[['Survived_1.0','Sex_female','Sex_male','Pclass_1.0']].astype(int)
   out = bnlearn.structure_learning(Xhot, methodtype='hc', scoretype='bic')
   bnlearn.plot(out)

 SEE ALSO
   hnet
"""

#--------------------------------------------------------------------------
# Name        : bnlearn.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : Jan. 2020
#--------------------------------------------------------------------------

#%% Libraries
import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
# STRUCTURE LEARNING
from pgmpy.estimators import BdeuScore, K2Score, BicScore
from pgmpy.estimators import ExhaustiveSearch, HillClimbSearch, ConstraintBasedEstimator
# DAG
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
# SAMPLING
from pgmpy.sampling import BayesianModelSampling#, GibbsSampling
# PARAMETER LEARNING
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator#, ParameterEstimator
# INFERENCE
from pgmpy.inference import VariableElimination
from pgmpy import readwrite
# MICROSERVICES
from bnlearn.helpers.df2onehot import df2onehot
import bnlearn.helpers.network as network
# ASSERTS
assert (nx.__version__)=='1.11', 'This function requires networkx to be v1.11. Try to: pip install networkx==v1.11'
assert (mpl.__version__)=='2.2.3', 'This function requires matplotlib to be v2.2.3. Try to: pip install matplotlib==v2.2.3'
curpath = os.path.dirname(os.path.abspath( __file__ ))
PATH_TO_DATA=os.path.join(curpath,'DATA')


#%% Exact inference using Variable Elimination
def inference(model, variables=None, evidence=None, verbose=3):
    '''
    Inference is same as asking conditional probability questions to the models. 
    i.e., What is the probability of a sprinkler is on given that it is raining which is basically equivalent of asking $ P(g^1 | i^1) $. 
    Inference algorithms deals with efficiently finding these conditional probability queries.
    
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
    '''
    
    model_infer = VariableElimination(model['model'])
    # Computing the probability of Wet Grass given Rain.
    q = model_infer.query(variables=variables, evidence=evidence)
    print(q)
    # for varname in variables: 
        # print(q[varname])
    return(q)
    
#%% Sampling from model
def parameter_learning(model, df, methodtype='bayes', verbose=3):
    '''
    Parameter learning is the task to estimate the values of the conditional 
    probability distributions (CPDs), for the variables cloudy, sprinkler, rain and wet grass. 
    State counts
        To make sense of the given data, we can start by counting how often each state of the variable occurs. 
        If the variable is dependent on parents, the counts are done conditionally on the parents states, 
        i.e. for seperately for each parent configuration:
    '''

#    model = BayesianModel([('Cloudy', 'Sprinkler'), 
#                           ('Cloudy', 'Rain'),
#                           ('Sprinkler', 'Wet_Grass'),
#                           ('Rain', 'Wet_Grass')])

    config            = dict()
    config['verbose'] = verbose
    config['method']  = methodtype
    model = model['model']
    if verbose>=3: print('[BNLEARN][PARAMETER LEARNING] Computing parameters using [%s]' %(config['method']))

#    pe = ParameterEstimator(model, df)
#    print("\n", pe.state_counts('Cloudy'))
#    print("\n", pe.state_counts('Sprinkler'))

    '''
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
    '''

    # Learning CPDs using Maximum Likelihood Estimators
    if config['method']=='ml' or config['method']=='maximumlikelihood':
        mle = MaximumLikelihoodEstimator(model, df)
        for node in mle.state_names:
            print(mle.estimate_cpd(node))


    '''
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
    '''
    if config['method']=='bayes':
        model.fit(df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=1000) # default equivalent_sample_size=5

        for cpd in model.get_cpds():
            if verbose>=3: print("CPD of {variable}:".format(variable=cpd.variable))
            if verbose>=3: print(cpd)
    
    return(model)
    
#%% Sampling from model
def sampling(model, n=1000, verbose=3):
    assert n>0, 'n must be 1 or larger'
    assert 'BayesianModel' in str(type(model['model'])), 'Model must contain DAG from BayesianModel. Note that <misarables> example does not include DAG.'

    # http://pgmpy.org/sampling.html
    inference = BayesianModelSampling(model['model'])
    # inference = GibbsSampling(model)
    # Forward sampling and make dataframe
    df=inference.forward_sample(size=n, return_type='dataframe')
    return(df)
    
#%% Structure Learning
def structure_learning(df, methodtype='hc', scoretype='bic', min_y=None, verbose=3):
    assert isinstance(pd.DataFrame(), type(df)), 'df must be of type pd.DataFrame()'
    assert (scoretype=='bic') | (scoretype=='k2') | (scoretype=='bdeu'), 'scoretype must be string: "bic", "k2" or "bdeu"'
    assert (methodtype=='hc') | (methodtype=='ex')|  (methodtype=='cs') | (methodtype=='exhaustivesearch')| (methodtype=='hillclimbsearch')| (methodtype=='constraintsearch'), 'Methodtype string is invalid'
    assert float(nx.__version__)==1.11, 'This function requires networkx to be v1.11 or so. Try to: pip install networkx==v1.11'

    config            = dict()
    config['verbose'] = verbose
    config['method']  = methodtype
    config['scoring'] = scoretype
    
    # Show warning
    if df.shape[1]>10 and df.shape[1]<15:
        if verbose>=3: print('[BNLEARN][STRUCTURE LEARNING] Note that computing DAG with %d nodes can take a very long time!' %(df.shape[1]))
    
    # Make sure columns are of type string
    df.columns = df.columns.astype(str)
    # Make onehot
#    df,labx = makehot(df, min_y=min_y)
    
        
    '''
    Search strategies for structure learning
    The search space of DAGs is super-exponential in the number of variables and the above scoring functions allow for local maxima. 
    http://pgmpy.chrisittner.de/pages/gsoc-proposal.html
    
    To learn model structure (a DAG) from a data set, there are three broad techniques:
        1. Score-based structure learning
            a. exhaustivesearch
            b. hillclimbsearch
        2. Constraint-based structure learning
            a. chi-square test
        3. Hybrid structure learning (The combination of both techniques)
    
        Score-based Structure Learning
        This approach construes model selection as an optimization task. It has two building blocks:
        A scoring function sD:->R that maps models to a numerical score, based on how well they fit to a given data set D.
        A search strategy to traverse the search space of possible models M and select a model with optimal score.
        Commonly used scoring functions to measure the fit between model and data are Bayesian Dirichlet scores such as BDeu or K2 and the Bayesian Information Criterion (BIC, also called MDL). See [1], Section 18.3 for a detailed introduction on scores. As before, BDeu is dependent on an equivalent sample size.
    '''
    
    if verbose>=3: print('[BNLEARN][STRUCTURE LEARNING] Computing best DAG using [%s]' %(config['method']))

    #ExhaustiveSearch can be used to compute the score for every DAG and returns the best-scoring one:
    if config['method']=='ex' or config['method']=='exhaustivesearch':
        '''
        The first property makes exhaustive search intractable for all but very small networks, 
        the second prohibits efficient local optimization algorithms to always find the optimal structure. 
        Thus, identifiying the ideal structure is often not tractable. 
        Despite these bad news, heuristic search strategies often yields good results.
        If only few nodes are involved (read: less than 5), 
        '''
        assert df.shape[1]<15, 'Structure learning with more then 15 nodes is computationally not feasable with exhaustivesearch. Use hillclimbsearch or constraintsearch'
        out  = exhaustivesearch(df, scoretype=config['scoring'], verbose=config['verbose'])

    # HillClimbSearch
    if config['method']=='hc' or config['method']=='hillclimbsearch':
        '''
        Once more nodes are involved, one needs to switch to heuristic search. 
        HillClimbSearch implements a greedy local search that starts from the DAG 
        "start" (default: disconnected DAG) and proceeds by iteratively performing 
        single-edge manipulations that maximally increase the score. 
        The search terminates once a local maximum is found.
        '''
        out = hillclimbsearch(df, scoretype=config['scoring'], verbose=config['verbose'])
    
    # Constraint-based Structure Learning
    if config['method']=='cs' or config['method']=='constraintsearch':
        '''
        Constraint-based Structure Learning
        A different, but quite straightforward approach to build a DAG from data is this:
        Identify independencies in the data set using hypothesis tests
        Construct DAG (pattern) according to identified independencies (Conditional) Independence Tests
        Independencies in the data can be identified using chi2 conditional independence tests. 
        To this end, constraint-based estimators in pgmpy have a test_conditional_independence(X, Y, Zs)-method, 
        that performs a hypothesis test on the data sample. It allows to check if X is independent from Y given a set of variables Zs:
        '''
        out = constraintsearch(df, verbose=config['verbose'])

    # Setup simmilarity matrix
    adjmat = pd.DataFrame(data=False, index=out['model'].nodes(), columns=out['model'].nodes()).astype('Bool')
    # Fill adjmat with edges
    edges=out['model'].edges()
    for edge in edges:
        adjmat.loc[edge[0],edge[1]]=True
    
    # Store
    out['adjmat']=adjmat
    # return
    return(out)

#%% Constraint-based Structure Learning
def constraintsearch(df, significance_level=0.05, verbose=3):
    '''
    test_conditional_independence() returns a tripel (chi2, p_value, sufficient_data), 
    consisting in the computed chi2 test statistic, the p_value of the test, and a heuristig 
    flag that indicates if the sample size was sufficient. 
    The p_value is the probability of observing the computed chi2 statistic (or an even higher chi2 value), 
    given the null hypothesis that X and Y are independent given Zs.
    This can be used to make independence judgements, at a given level of significance:
    '''

    out=dict()
    # Set search algorithm
    model = ConstraintBasedEstimator(df)

    # Some checks for dependency
    #    print(is_independent(est, 'Sprinkler', 'Rain', significance_level=significance_level))
    #    print(is_independent(est, 'Cloudy', 'Rain', significance_level=significance_level))
    #    print(is_independent(est, 'Sprinkler', 'Rain',  ['Wet_Grass'], significance_level=significance_level))
    
    '''
    DAG (pattern) construction
    With a method for independence testing at hand, we can construct a DAG from the data set in three steps:
        1. Construct an undirected skeleton - `estimate_skeleton()`
        2. Orient compelled edges to obtain partially directed acyclid graph (PDAG; I-equivalence class of DAGs) - `skeleton_to_pdag()`
        3. Extend DAG pattern to a DAG by conservatively orienting the remaining edges in some way - `pdag_to_dag()`
        
        Step 1.&2. form the so-called PC algorithm, see [2], page 550. PDAGs are `DirectedGraph`s, that may contain both-way edges, to indicate that the orientation for the edge is not determined.
    '''
    # Estimate using chi2
    [skel, seperating_sets] = model.estimate_skeleton(significance_level=significance_level)

    print("Undirected edges: ", skel.edges())
    pdag = model.skeleton_to_pdag(skel, seperating_sets)
    print("PDAG edges: ", pdag.edges())
    dag = model.pdag_to_dag(pdag)
    print("DAG edges: ", dag.edges())

    out['undirected']=skel
    out['undirected_edges']=skel.edges()
    out['pdag']=pdag
    out['pdag_edges']=pdag.edges()
    out['dag']=dag
    out['dag_edges']=dag.edges()

    # Search using "estimate()" method provides a shorthand for the three steps above and directly returns a "BayesianModel"
    best_model = model.estimate(significance_level=significance_level)
    out['model']=best_model
    out['model_edges']=best_model.edges()

    print(best_model.edges())

    '''
    PC PDAG construction is only guaranteed to work under the assumption that the 
    identified set of independencies is *faithful*, i.e. there exists a DAG that 
    exactly corresponds to it. Spurious dependencies in the data set can cause 
    the reported independencies to violate faithfulness. It can happen that the 
    estimated PDAG does not have any faithful completions (i.e. edge orientations 
    that do not introduce new v-structures). In that case a warning is issued.
    '''
    return(out)

#%% hillclimbsearch
def hillclimbsearch(df, scoretype='bic', verbose=3):
    out=dict()
    # Set scoring type
    scoring_method = SetScoringType(df, scoretype)
    # Set search algorithm
    model = HillClimbSearch(df, scoring_method=scoring_method)
    # Compute best DAG
    best_model = model.estimate()
    # Store
    out['model']=best_model
    out['model_edges']=best_model.edges()
    # Return
    return(out)

#%% ExhaustiveSearch
def exhaustivesearch(df, scoretype='bic', return_all_dags=False, verbose=3):
    out=dict()

    # Set scoring type
    scoring_method = SetScoringType(df, scoretype)
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
        #print("\nAll DAGs by score:")
        for [score, dag] in reversed(model.all_scores()):
            out['scores'].append(score)
            out['dag'].append(dag)
            #print(score, dag.edges())
        
        plt.plot(out['scores'])
        plt.show()

    return(out)

#%% Set scoring type
def SetScoringType(df, scoretype, verbose=3):
    if verbose>=3: print('[BNLEARN][STRUCTURE LEARNING] Set scoring type at [%s]' %(scoretype))
    
    if scoretype=='bic':
        scoring_method = BicScore(df)
    elif scoretype=='k2':
        scoring_method = K2Score(df)
    elif scoretype=='bdeu':
        scoring_method = BdeuScore(df, equivalent_sample_size=5)

    return(scoring_method)
#%%
def is_independent(model, X, Y, Zs=[], significance_level=0.05):
    return model.test_conditional_independence(X, Y, Zs)[1] >= significance_level

#%% Make one-hot matrix
def makehot(df, min_y=None):
    labx=[]
    colExpand=[]
#    colOK=[]
    Xhot=pd.DataFrame()
    dfOK=pd.DataFrame()
    for i in range(0,df.shape[1]):
        if len(df.iloc[:,i].unique())>2:
            colExpand.append(df.columns[i])
        else:
            if df[df.columns[i]].dtype=='O':
                uicol=df[df.columns[i]].unique()
                dfOK[uicol[0]]=df[df.columns[i]]==uicol[0]
            else:
                dfOK = pd.concat([dfOK, Xhot], axis=1)
                labx.append(df.columns[i])
                #colOK.append(df.columns[i])
    
    if len(colExpand)>0:
        [_, Xhot, Xlabx, _]=df2onehot(df[colExpand], min_y=min_y, hot_only=True)
        labx.append(Xlabx)
        Xhot=Xhot.astype(int)
    
    out = pd.concat([Xhot, dfOK], axis=1)
    out = out.astype(int)

    return(out, labx[0])

#%% Make DAG
def load_examples(datatype='sprinkler', CPD=True, verbose=3):
    out=dict()
    model=None
    datatype=datatype.lower()
    
    # Load data
    if datatype=='sprinkler':
        model = DAG_sprinkler(CPD=CPD)
    if datatype=='asia':
        model = bif2bayesian(os.path.join(PATH_TO_DATA,'ASIA/asia.bif'))
    if datatype=='alarm':
        model = bif2bayesian(os.path.join(PATH_TO_DATA,'ALARM/alarm.bif'))
    if datatype=='andes':
        model = bif2bayesian(os.path.join(PATH_TO_DATA,'ANDES/andes.bif'))
    if datatype=='pathfinder':
        model = bif2bayesian(os.path.join(PATH_TO_DATA,'PATHFINDER/pathfinder.bif'))
    if datatype=='sachs':
        model = bif2bayesian(os.path.join(PATH_TO_DATA,'SACHS/sachs.bif'))
    if datatype=='miserables':
        f = open(os.path.join(PATH_TO_DATA,'miserables.json'))
        data = json.loads(f.read())
        L=len(data['links'])
        edges=[(data['links'][k]['source'], data['links'][k]['target']) for k in range(L)]
        model=nx.Graph(edges, directed=False)


    # check_model check for the model structure and the associated CPD and returns True if everything is correct otherwise throws an exception
    if not isinstance(model, type(None)) and verbose>=3:
        if CPD:
            print('Model correct: %s' %(model.check_model()))
            for cpd in model.get_cpds():
                print("CPD of {variable}:".format(variable=cpd.variable))
                print(cpd)

            print('Nodes: %s' %(model.nodes()))
            print('Edges: %s' %(model.edges()))
            print('Independencies:\n%s' %(model.get_independencies()))

    # Setup simmilarity matrix
    adjmat = pd.DataFrame(data=False, index=model.nodes(), columns=model.nodes()).astype('Bool')
    # Fill adjmat with edges
    edges=model.edges()
    for edge in edges:
        adjmat.loc[edge[0],edge[1]]=True

    out['model']=model
    out['adjmat']=adjmat
    return(out)

#%% Model Sprinkler
def DAG_sprinkler(verbose=3, CPD=True):
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

#%% Convert BIF model to bayesian model
def bif2bayesian(pathname):
    """
    Returns the fitted bayesian model
 
    Example
    ----------
    >>> from pgmpy.readwrite import BIFReader
    >>> reader = BIFReader("bif_test.bif")
    >>> reader.get_model()
    <pgmpy.models.BayesianModel.BayesianModel object at 0x7f20af154320>
    """

    bifmodel=readwrite.BIF.BIFReader(path=pathname)
    #bifmodel.get_edges()

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
        raise AttributeError('First get states of variables, edges, parents and network name')

#%% Make directed graph from adjmatrix
def to_undirected(adjmat):
    num_rows=adjmat.shape[0]
    num_cols=adjmat.shape[1]
    adjmat_directed=np.zeros((num_rows, num_cols), dtype=int)
    tmpadjmat=adjmat.astype(int)
    
    for i in range(num_rows):
        for j in range(num_cols):
            adjmat_directed[i,j] = tmpadjmat.iloc[i,j]+tmpadjmat.iloc[j,i]
    
    adjmat_directed=pd.DataFrame(index=adjmat.index, data=adjmat_directed, columns=adjmat.columns, dtype=bool)
    return(adjmat_directed)

#%% Comparison of two networks
def compare_networks(adjmat_true, adjmat_pred, pos=None, showfig=True, width=15, height=8, verbose=3):
    [scores, adjmat_diff] = network.compare_networks(adjmat_true, adjmat_pred, pos=pos, showfig=showfig, width=width, height=height, verbose=verbose)
    return(scores, adjmat_diff)

#%% PLOT
def plot(model, pos=None, scale=1, width=15, height=8, verbose=3):
    out=dict()
    G = nx.DiGraph() # Directed graph
    layout='fruchterman_reingold'
    
    # Extract model if in dict
    if 'dict' in str(type(model)):
        model = model.get('model', None)
    
    # Bayesian model
    if 'BayesianModel' in str(type(model)):
        if verbose>=3: print('[BNLEARN.plot] Making plot based on BayesianModel')
        # positions for all nodes
        pos = network.graphlayout(model, pos=pos, scale=scale, layout=layout)
        # Add directed edge with weigth
        edges=model.edges()
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
    plt.figure(figsize=(width,height))
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, with_labels=True)
    # edges
    colors  = [G[u][v].get('color','k') for u,v in G.edges()]
    weights = [G[u][v].get('weight',1) for u,v in G.edges()]
    nx.draw_networkx_edges(G, pos, arrowstyle='->', edge_color=colors, width=weights)
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
    # Get labels of weights
    #labels = nx.get_edge_attributes(G,'weight')
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
    