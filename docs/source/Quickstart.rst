.. _code_directive:

-------------------------------------

Quickstart
''''''''''


Installation
------------

It is advisable to create a new environment. 
You may need to deactivate and then activate your environment otherwise the packages may not been recognized.


.. code-block:: console

   conda create -n env_BNLEARN python=3.6
   conda activate env_BNLEARN
   conda install -c ankurankan pgmpy

   conda deactivate
   conda activate env_BNLEARN


.. code-block:: console

   pip install bnlearn
    

Quickstart
-----------

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
