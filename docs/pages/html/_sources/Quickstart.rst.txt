Quickstart
==========


Installation (short)
^^^^^^^^^^^^^^^^^^^^

Installation of bnlearn is straightforward. 
It is advisable to create a new environment. 

.. code-block:: console

   conda create -n env_bnlearn python=3.8
   conda activate env_bnlearn
   pip install bnlearn


Quick Examples
^^^^^^^^^^^^^^^^

Let's start by importing some data. We need a DAG and CPD.


.. code:: python

    import bnlearn as bn

    df = bn.import_example()

    model = bn.structure_learning.fit(df)

    model = bn.independence_test(model, df)

    G = bn.plot(model)



.. code:: python

    import bnlearn as bn

    model = bn.import_DAG('sprinkler')

    df = bn.import_example()

    df = bn.sampling(model)

    query = bn.inference.fit(model, variables=['Rain'], evidence={'Cloudy':1, 'Wet_Grass':1})
    print(query.df)

    model_sl = bn.structure_learning.fit(df)

    model_pl = bn.parameter_learning.fit(model_sl, df)

    scores, adjmat = bn.compare_networks(model_sl, model)


.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>

