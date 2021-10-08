from bnlearn.bnlearn import (
    to_bayesianmodel,
    make_DAG,
    print_CPD,
    import_DAG,
    import_example,
    sampling,
    to_undirected,
    compare_networks,
    plot,
    adjmat2vec,
    vec2adjmat,
    df2onehot,
    topological_sort,
    predict,
    query2df,
    _dag2adjmat,
    _filter_df,
    save,
    load,
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
__version__ = '0.4.3'

try:
    import pgmpy
except:
    raise ImportError('[bnlearn] >Error: pgmpy version "0.1.13" or higher must be installed manually. Try to: <conda install -c ankurankan pgmpy> or <pip install -U pgmpy>=0.1.13>')

# Check version pgmpy
if version.parse(pgmpy.__version__) < version.parse("0.1.13"):
    raise Exception('[bnlearn] >Error: This release requires pgmpy to be version >= 0.1.13. Try to: <conda install -c ankurankan pgmpy> or <pip install -U pgmpy>=0.1.13>')

# Version check
import matplotlib
if not version.parse(matplotlib.__version__) >= version.parse("3.3.4"):
    raise Exception('[bnlearn] >Error: Matplotlib version should be >= v3.3.4.\nTry to: pip install -U matplotlib')

import networkx as nx
if not version.parse(nx.__version__) > version.parse("2.5"):
    raise Exception('[bnlearn] >Error: networkx version should be > 2.5.\nTry to: pip install -U networkx')


# module level doc-string
__doc__ = """
BNLEARN - bnlearn is an Python package for learning the graphical structure of Bayesian networks, estimate their parameters, perform inference, sampling and comparing networks.
================================================================================================================================================================================

Description
-----------
* Learning a Bayesian network can be split into two problems:
    * Parameter learning: Given a set of data samples and a DAG that captures the dependencies between the variables,
      estimate the (conditional) probability distributions of the individual variables.
    * Structure learning: Given a set of data samples, estimate a DAG that captures the dependencies between the variables.
* Currently, the library supports:
    * Parameter learning for *discrete* nodes:
    * Maximum Likelihood Estimation
    * Bayesian Estimation
* Structure learning for *discrete*, *fully observed* networks:
    * Score-based structure estimation (BIC/BDeu/K2 score; exhaustive search, hill climb/tabu search)
    * Constraint-based structure estimation (PC)
    * Hybrid structure estimation (MMHC)


Example
-------
>>> # Import library
>>> import bnlearn as bn
>>> model = bn.import_DAG('sprinkler')
>>> bn.plot(model)
>>>
>>> # Import example
>>> df = bn.import_example('sprinkler')
>>> df = bn.sampling(model)
>>>
>>> # Do the inference
>>> q1 = bn.inference.fit(model, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})
>>> q2 = bn.inference.fit(model, variables=['Wet_Grass','Rain'], evidence={'Sprinkler':1})
>>>
>>> # Structure learning
>>> model_sl = bn.structure_learning.fit(df)
>>> # Parameter learning
>>> model_pl = bn.parameter_learning.fit(model_sl, df)
>>> # Compare networks
>>> scores, adjmat = bn.compare_networks(model_sl, model)

References
----------
* https://bnlearn.readthedocs.io
* https://github.com/erdogant/bnlearn

"""
