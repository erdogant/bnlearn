from bnlearn.bnlearn import (
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
)

# Import function in new level
import bnlearn.structure_learning
import bnlearn.parameter_learning
import bnlearn.inference

import pgmpy
from packaging import version
assert version.parse(pgmpy.__version__)>=version.parse("0.1.10"), 'This release requires pgmpy to be v0.1.10. or higher. Try to: conda install -c ankurankan pgmpy'

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.2.3'

# module level doc-string
__doc__ = """
BNLEARN - bnlearn is an Python package for learning the graphical structure of Bayesian networks, estimate their parameters, perform inference, sampling and comparing networks.
================================================================================================================================================================================

Description
-----------
bnlearn

Example
-------
import bnlearn

model            = bnlearn.import_DAG('sprinkler')
df               = bnlearn.import_example()
df               = bnlearn.sampling(model)
q                = bnlearn.inference.fit(model)
model_sl         = bnlearn.structure_learning.fit(df)
model_pl         = bnlearn.parameter_learning.fit(model_sl, df)
[scores, adjmat] = bnlearn.compare_networks(model_sl, model)


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

References
----------
https://bnlearn.readthedocs.io
https://github.com/erdogant/bnlearn

"""
