Example Datasets
================

`bnlearn` contins various example datasets that can be used to better understand how the datasets should be structured as an input to the `bnlearn` functions 
It is also possible to create a DAG manually (see create DAG section) but here we are going to load some of the well-known bayesian models.

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


.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>

