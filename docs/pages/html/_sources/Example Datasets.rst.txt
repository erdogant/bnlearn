Example Datasets
================

``bnlearn`` contains various example datasets that can be used to better understand how the datasets should be structured as an input to the ``bnlearn`` functions.
It is also possible to import other DAG files manually that are stored in bif files. DAGs can be loaded and/or imported as following:

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

