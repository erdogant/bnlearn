Sampling and datasets
=====================

Sampling of data is based on the joint distribution of the network.
In order to do that, it requires as input a DAG connected with CPDs.
It is also possible to create a DAG manually (see create DAG section) or load an existing one as depicted below.

Datasets
''''''''

Various DAGs available that can be easily loaded:

.. code-block:: python

   import bnlearn as bn

   # The following models can be loaded:
   loadfile = 'sprinkler'
   loadfile = 'alarm'
   loadfile = 'andes'
   loadfile = 'asia'
   loadfile = 'pathfinder'
   loadfile = 'sachs'
   loadfile = 'miserables'

   DAG = bn.import_DAG(loadfile)


Models are usually stored in bif files which can also be imported:

.. code-block:: python

   filepath = 'directory/to/model.bif'

   DAG = bn.import_DAG(filepath)



Forward Sampling
''''''''''''''''

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
''''''''''''''

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

.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>

