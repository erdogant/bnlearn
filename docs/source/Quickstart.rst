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

    # Example dataframe sprinkler_data.csv can be loaded with: 
    import bnlearn

    df = bnlearn.import_example()

    model = bnlearn.structure_learning.fit(df)

    G = bnlearn.plot(model)

