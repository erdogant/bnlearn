.. include:: add_top.add

``bnlearn`` contains several examples within the library that can be used to practice with the functionalities of :func:`bnlearn.structure_learning`, :func:`bnlearn.parameter_learning` and :func:`bnlearn.inference`.


DataFrames
^^^^^^^^^^

The **sprinkler** dataset is one of the few internal datasets to import a pandas dataframe. This dataset is readily one-hot coded and without missing values. Therefore it does not require any further pre-processing steps. Note that 

.. code-block:: python

   import bnlearn as bn
   # Import dataset
   df = bn.import_example('sprinkler')

   print(df)
   #     Cloudy  Sprinkler  Rain  Wet_Grass
   # 0         0          0     0          0
   # 1         1          0     1          1
   # 2         0          1     0          1
   # ..      ...        ...   ...        ...
   # 998       0          0     0          0
   # 999       0          1     1          1

   # Structure learning
   model = bn.structure_learning.fit(df)

   # Plot
   G = bn.plot(model)



Import DAG/BIF
^^^^^^^^^^^^^^^

Each Bayesian DAG model that is loaded with :func:`bnlearn.bnlearn.import_DAG` is derived from a *bif* file. The *bif* file is a common format for Bayesian networks that can be used for the exchange of knowledge and experimental results in the community. More information can be found (here)[http://www.cs.washington.edu/dm/vfml/appendixes/bif.htm].


.. code-block:: python

	import bnlearn as bn

	bif_file= 'sprinkler'
	bif_file= 'alarm'
	bif_file= 'andes'
	bif_file= 'asia'
	bif_file= 'pathfinder'
	bif_file= 'sachs'
	bif_file= 'miserables'
	bif_file= 'filepath/to/model.bif'

	# Loading DAG with model parameters from bif file.
	model = bn.import_DAG(bif_file)



With the :func:`bnlearn.bnlearn.sampling` function a ``DataFrame`` can be created for *n* samples.


.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>



.. include:: add_bottom.add