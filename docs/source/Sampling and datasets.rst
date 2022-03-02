Sampling and datasets
=====================

Sampling of data is based on forward sampling from joint distribution of the bayesian network.
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



Example Sampling
''''''''''''''''


.. code-block:: python
 
   import bnlearn as bn

   DAG = bn.import_DAG('sprinkler', CPD=True)

   df = bn.sampling(DAG, n=1000)

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


.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>

