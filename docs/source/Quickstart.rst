.. _code_directive:

-------------------------------------

Quickstart
''''''''''


Installation
------------

Install via ``pip``:
* It is advisable to create a new environment.

.. code-block:: python

    conda create -n env_BNLEARN python=3.6
    conda activate env_BNLEARN
    conda install -c ankurankan pgmpy
    # You may need to deactivate and then activate your environment otherwise the packages may not been recognized.
    conda deactivate
    conda activate env_BNLEARN

    # The packages below are handled by the requirements in the bnlearn pip installer. So you dont need to do them manually.
    pip install sklearn pandas tqdm funcsigs statsmodels community packaging

    pip install bnlearn
    

bnlearn
---------------------------------------------------

Let's start by importing some data. We need a DAG and CPD.


.. code:: python

    # Example dataframe sprinkler_data.csv can be loaded with: 
    import bnlearn

    df = bnlearn.import_example()

    model = bnlearn.structure_learning.fit(df)

    G = bnlearn.plot(model)

