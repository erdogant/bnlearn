Sampling of data is based on the joint distribution of the network.
In order to do that, it requires a DAG connected with CPDs.
It is also possible to create a DAG manually and learn it's model parameters.


Forward Sampling
================

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