Generating Synthetic Data is based on sampling from multinomial distributions, which naturally extend to Bayesian Forward Sampling.
At its core, Bayesian Sampling refers to generating data points from a probabilistic model defined by a Directed Acyclic Graph (DAG)
and its associated Conditional Probability Distributions (CPDs).
The structure of the DAG encodes the dependencies between variables, while the CPDs define the exact probability of each variable conditioned on its parents.
When combined, they form a joint probability distribution over all variables in the network.


Forward Sampling
================
Bayesian Forward Sampling is one of the most intuitive sampling techniques.
It proceeds by traversing the graph in topological order, starting with root nodes (with no parents),
and sampling values for each variable based on its CPD and the already-sampled values of its parent nodes.
This method is ideal when you want to simulate new data that follows the generative assumptions of your Bayesian Network.
In bnlearn this is the default method. It is particularly powerful for creating synthetic datasets from expert-defined DAGs,
where we explicitly encode our domain knowledge without requiring observational data.


.. code-block:: python
 
	# Import library
	import bnlearn as bn
	
	# Load example DAG with CPD
	model = bn.import_DAG('sprinkler', CPD=True)
	
	# Take 1000 samples from the CPD distribution
	df = bn.sampling(model, n=1000, methodtype='bayes')

	df.head()


.. table::

  +--------+-----------+------+-------------+
  |Cloudy  | Sprinkler | Rain |  Wet_Grass  |
  +========+===========+======+=============+
  |    0   |      1    |  0   |      1      |
  +--------+-----------+------+-------------+
  |    1   |      1    |  1   |      1      |
  +--------+-----------+------+-------------+
  |    1   |      0    |  1   |      1      |
  +--------+-----------+------+-------------+
  |    ... |      ...  | ...  |     ...     |
  +--------+-----------+------+-------------+
  |    0   |      0    |  0   |      0      |
  +--------+-----------+------+-------------+
  |    1   |      0    |  0   |      0      |
  +--------+-----------+------+-------------+
  |    1   |      0    |  1   |      1      |
  +--------+-----------+------+-------------+


Gibbs Sampling
==============
When some values are missing or when exact inference is computationally expensive, Gibbs Sampling can be used.
This is a Markov Chain Monte Carlo (MCMC) method that iteratively samples from the conditional distribution of each variable given the current values of all others.
Over time, this produces samples from the joint distribution, even without needing to compute it explicitly.
While Forward Sampling is better suited for full synthetic data generation, Gibbs Sampling excels in scenarios involving partial observations, imputation, or approximate inference.

.. code-block:: python
 
	# Import library
	import bnlearn as bn
	
	# Load example DAG with CPD
	model = bn.import_DAG('sprinkler', CPD=True)
	
	# Take 1000 samples from the CPD distribution
	df = bn.sampling(model, n=1000, methodtype='gibbs')

	df.head()


Another example with Gibbs sampling but now by creating some user-defined edges 

.. code-block:: python

	# Load example dataset
	df = bn.import_example('sprinkler')

	# Create some edges
	edges = [('Cloudy', 'Sprinkler'),
	        ('Cloudy', 'Rain'),
	        ('Sprinkler', 'Wet_Grass'),
	        ('Rain', 'Wet_Grass')]

	# Make the actual Bayesian DAG
	DAG = bn.make_DAG(edges, methodtype='bayes', verbose=3)

	# Fit model
	model = bn.parameter_learning.fit(DAG, df, verbose=3, methodtype='bayes')

	# Sampling using gibbs
	df = bn.sampling(model, n=100, methodtype='gibbs', verbose=3)



.. table::

  +--------+-----------+------+-------------+
  |Cloudy  | Sprinkler | Rain |  Wet_Grass  |
  +========+===========+======+=============+
  |    0   |      1    |  0   |      1      |
  +--------+-----------+------+-------------+
  |    1   |      0    |  0   |      1      |
  +--------+-----------+------+-------------+
  |    1   |      0    |  1   |      1      |
  +--------+-----------+------+-------------+
  |    ... |      ...  | ...  |     ...     |
  +--------+-----------+------+-------------+
  |    0   |      0    |  0   |      0      |
  +--------+-----------+------+-------------+
  |    1   |      0    |  0   |      0      |
  +--------+-----------+------+-------------+
  |    1   |      0    |  1   |      1      |
  +--------+-----------+------+-------------+



.. include:: add_bottom.add