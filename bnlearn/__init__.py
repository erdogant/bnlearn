from bnlearn.bnlearn import (
    to_bayesiannetwork,
    make_DAG,
    print_CPD,
    import_DAG,
    import_example,
    sampling,
    to_undirected,
    compare_networks,
    plot,
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
)

# Import function in new level
import bnlearn.structure_learning as structure_learning
import bnlearn.parameter_learning as parameter_learning
import bnlearn.inference as inference
import bnlearn.network as network
import bnlearn.confmatrix as confmatrix
from packaging import version

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.7.10'

try:
    import pgmpy
except:
    raise ImportError('[bnlearn] >Error: pgmpy version "0.1.13" or higher must be installed manually. Try to: <conda install -c ankurankan pgmpy> or <pip install -U pgmpy>=0.1.13>')

# Check version pgmpy
if version.parse(pgmpy.__version__) < version.parse("0.1.13"):
    raise ImportError('[bnlearn] >Error: This release requires pgmpy to be version >= 0.1.13. Try to: <conda install -c ankurankan pgmpy> or <pip install -U pgmpy>=0.1.13>')

# Version check
import matplotlib
if not version.parse(matplotlib.__version__) >= version.parse("3.3.4"):
    raise ImportError('[bnlearn] >Error: Matplotlib version should be >= v3.3.4.\nTry to: pip install -U matplotlib')

import networkx as nx
if not version.parse(nx.__version__) >= version.parse("2.7.1"):
    raise ImportError('[bnlearn] >Error: networkx version should be > 2.7.1.\nTry to: pip install -U networkx')


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
* Documentation: https://bnlearn.readthedocs.io

"""
