Quickstart
==========


Installation (short)
--------------------

Installation of bnlearn is straightforward. 
It is advisable to create a new environment. 

.. code-block:: console

   conda create -n BNLEARN python=3.6
   conda activate BNLEARN
   conda install -c ankurankan pgmpy

   conda deactivate
   conda activate BNLEARN

   pip install bnlearn


Examples
--------

Let's start by importing some data. We need a DAG and CPD.


.. code:: python

    import bnlearn

    df = bnlearn.import_example()

    model = bnlearn.structure_learning.fit(df)

    G = bnlearn.plot(model)



.. code:: python

    import bnlearn

    model = bnlearn.import_DAG('sprinkler')

    df = bnlearn.import_example()

    df = bnlearn.sampling(model)

    q = bnlearn.inference.fit(model)

    model_sl = bnlearn.structure_learning.fit(df)

    model_pl = bnlearn.parameter_learning.fit(model_sl, df)

    [scores, adjmat] = bnlearn.compare_networks(model_sl, model)
