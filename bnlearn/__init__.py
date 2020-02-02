from bnlearn.bnlearn import (
    sampling,
    import_DAG,
    import_example,
	to_undirected,
	compare_networks,
    plot,
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
__version__ = '0.2.1'

# module level doc-string
__doc__ = """
BNLEARN - bnlearn is an Python package for learning the graphical structure of Bayesian networks, estimate their parameters, perform some inference, sampling and comparing networks.
=====================================================================

**bnlearn** 
See README.md file for more information.

"""
