from bnlearn.bnlearn import (
    sampling,
	inference,
	structure_learning,
	parameter_learning,
    load_example,
	compare_networks,
	to_undirected,
    plot,
)

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
#__version__ = '0.1.0'

# Automatic version control
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


# module level doc-string
__doc__ = """
BNLEARN - bnlearn is an Python package for learning the graphical structure of Bayesian networks, estimate their parameters, perform some inference, sampling and comparing networks.
=====================================================================

**bnlearn** 
See README.md file for more information.

"""
