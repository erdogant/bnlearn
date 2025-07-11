from bnlearn.bnlearn import (
    to_bayesiannetwork,
    make_DAG,
    print_CPD,
    import_DAG,
    import_example,
    to_undirected,
    compare_networks,
    plot,
    plot_graphviz,
    adjmat2vec,
    adjmat2dict,
    vec2adjmat,
    dag2adjmat,
    df2onehot,
    topological_sort,
    predict,
    query2df,
    vec2df,
    get_node_properties,
    get_edge_properties,
    _filter_df,
    independence_test,
    save,
    load,
    check_model,
    structure_scores,
    compute_logp,
    get_parents,
    generate_cpt,
    build_cpts_from_structure,
    convert_edges_with_time_slice,
    # cpd_to_dataframe,
    # dataframe_to_cpd,
)

# Import function in new level
import bnlearn.structure_learning as structure_learning
import bnlearn.parameter_learning as parameter_learning
import bnlearn.inference as inference
import bnlearn.network as network
import bnlearn.confmatrix as confmatrix
from bnlearn.impute import knn_imputer, mice_imputer
from bnlearn.discretize import discretize, discretize_value
from bnlearn.learn_discrete_bayes_net import discretize_all
from bnlearn.sampling import sampling

from packaging import version

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.12.0'

import pgmpy
# Check version pgmpy
if version.parse(pgmpy.__version__) < version.parse("0.1.18"):
    raise ImportError('[bnlearn] >Error: This release requires pgmpy to be version == 0.1.26. Try to: <pip install -U pgmpy==0.1.26>')

# Version check
import matplotlib
if not version.parse(matplotlib.__version__) >= version.parse("3.3.4"):
    raise ImportError('[bnlearn] >Error: Matplotlib version should be >= v3.3.4\nTry to: pip install -U matplotlib')

import networkx as nx
if version.parse(nx.__version__) < version.parse("2.7.1"):
    raise ImportError('[bnlearn] >Error: networkx version should be > 2.7.1\nTry to: pip install -U networkx')

import numpy as np
if version.parse(np.__version__) < version.parse("1.24.1"):
    raise ImportError('[bnlearn] >Error: numpy version should be > 1.24.1\nTry to: pip install -U numpy')

# import pandas as pd
# if version.parse(pd.__version__) > version.parse("1.5.3"):
#     raise ImportError('[bnlearn] >Error: pands version should be <= 1.5.3')

# This one is moved towards the interactive plot function because it is not required in the setup.
# import d3blocks as d3
# if version.parse(d3.__version__) < version.parse("1.4.9"):
#     raise ImportError('[bnlearn] >Error: d3blocks version should be >= 1.4.9')

# module level doc-string
__doc__ = """
bnlearn - bnlearn is an Python package for learning the graphical structure of Bayesian networks, estimate their parameters, perform inference, sampling and comparing networks.
================================================================================================================================================================================

Description
-----------
* Learning a Bayesian network can be split into:
    * Structure learning: Given a set of data samples, estimate a DAG that captures the dependencies between the variables.
    * Parameter learning: Given a set of data samples and a DAG that captures the dependencies between the variables.
    * Making inferences.
    * Parameter and structure learning is for *discrete* nodes
        * Score-based structure estimation (BIC/BDeu/K2 score; exhaustive search, hill climb/tabu search)
        * Constraint-based structure estimation (PC)
        * Hybrid structure estimation (MMHC)

Example
-------
>>> # Import library
>>> import bnlearn as bn
>>> model = bn.import_DAG('sprinkler')
>>> # Print CPDs
>>> bn.print_CPD(model)
>>> # Plot DAG
>>> bn.plot(model)
>>>
>>> # Sampling using DAG and CPDs
>>> df = bn.sampling(model)
>>>
>>> # Do the inference
>>> q1 = bn.inference.fit(model, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})
>>> q2 = bn.inference.fit(model, variables=['Wet_Grass','Rain'], evidence={'Sprinkler':1})
>>>
>>> # Structure learning
>>> model_sl = bn.structure_learning.fit(df)
>>> # Compute edge strength using chi-square independence test
>>> model_sl = bn.independence_test(model_sl, df)
>>> # Plot DAG
>>> bn.plot(model_sl)
>>>
>>> # Parameter learning
>>> model_pl = bn.parameter_learning.fit(model_sl, df)
>>> # Compute edge strength using chi-square independence test
>>> model_pl = bn.independence_test(model_pl, df)
>>> # Plot DAG
>>> bn.plot(model_pl)
>>>
>>> # Compare networks
>>> scores, adjmat = bn.compare_networks(model_sl, model)

References
----------
* Blog: https://towardsdatascience.com/a-step-by-step-guide-in-detecting-causal-relationships-using-bayesian-structure-learning-in-python-c20c6b31cee5
* Blog: https://towardsdatascience.com/a-step-by-step-guide-in-designing-knowledge-driven-models-using-bayesian-theorem-7433f6fd64be
* Github: https://github.com/erdogant/bnlearn
* Documentation: https://erdogant.github.io/bnlearn/

"""
