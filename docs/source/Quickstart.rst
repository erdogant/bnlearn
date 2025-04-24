Quickstart
==========

Installation
------------

.. _quickstart-installation:

Installing bnlearn is straightforward. 
It is recommended to create a new environment for the installation.

.. code-block:: console

   conda create -n env_bnlearn python=3.8
   conda activate env_bnlearn
   pip install bnlearn

Quick Examples
^^^^^^^^^^^^^^^^

Let's start by importing some data. We need a DAG and CPD (Conditional Probability Distribution).

.. code:: python

    import bnlearn as bn

    # Import example dataset
    df = bn.import_example()

    # Learn the structure from data
    model = bn.structure_learning.fit(df)

    # Perform independence tests
    model = bn.independence_test(model, df)

    # Visualize the network
    G = bn.plot(model)

Here's another example demonstrating a complete workflow:

.. code:: python

    import bnlearn as bn

    # Import a predefined DAG (Sprinkler network)
    model = bn.import_DAG('sprinkler')

    # Import example dataset
    df = bn.import_example()

    # Generate samples from the model
    df = bn.sampling(model)

    # Perform inference
    query = bn.inference.fit(model, variables=['Rain'], evidence={'Cloudy':1, 'Wet_Grass':1})
    print(query.df)

    # Learn structure from data
    model_sl = bn.structure_learning.fit(df)

    # Learn parameters
    model_pl = bn.parameter_learning.fit(model_sl, df)

    # Compare networks
    scores, adjmat = bn.compare_networks(model_sl, model)

.. include:: add_bottom.add