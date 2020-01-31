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

import networkx as nx
import matplotlib as mpl
#assert (nx.__version__)=='1.11', 'This function requires networkx to be v1.11. Try to: pip install networkx==v1.11'
#assert (mpl.__version__)=='2.2.3', 'This function requires matplotlib to be v2.2.3. Try to: pip install matplotlib==v2.2.3'

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
