Structure learning
======================================

Learning Bayesian Networks from continuous data is an challanging task.
In ``bnlearn`` this task is now accomplished by learning discrete bayesian networks from continuous data.

In order to do this, I am using a Bayesian discretization method for continuous
variables in Bayesian networks with quadratic complexity instead of the cubic
complexity of other standard techniques. Empirical demonstrations show that the
proposed method is superior to the established minimum description length algorithm.
The method is described in the paper of Yi-Chun Chen et al.

The underlying idea i that the implemented method now can perform structure learning by first discretizng the continuous
variables and simultaneously learn Bayesian network structures.



To demonstrate the usage of structure learning on continuous data, I will use the well known **auto mpg** data set.


.. code-block:: python

    # Import
    import bnlearn as bn
    # Load data set
    df = bn.import_example(data='auto_mpg')

    # Define the edges
    edges = [
        ("cylinders", "displacement"),
        ("displacement", "model_year"),
        ("displacement", "weight"),
        ("displacement", "horsepower"),
        ("weight", "model_year"),
        ("weight", "mpg"),
        ("horsepower", "acceleration"),
        ("mpg", "model_year"),
    ]

    # Create DAG based on edges
    DAG = bn.make_DAG(edges)

    # A good habbit is to set the columns with continuous data as float
    continuous_columns = ["mpg", "displacement", "horsepower", "weight", "acceleration"]

    # Discretize the continous columns by specifying
    df_discrete = bn.discretize(df, edges, continuous_columns, max_iterations=1)

    #                 mpg  cylinders  ... model_year origin
    # 0     (17.65, 21.3]          8  ...         70      1
    # 1    (8.624, 15.25]          8  ...         70      1
    # 2     (17.65, 21.3]          8  ...         70      1
    # 3    (15.25, 17.65]          8  ...         70      1
    # 4    (15.25, 17.65]          8  ...         70      1
    # ..              ...        ...  ...        ...    ...
    # 387   (25.65, 28.9]          4  ...         82      1
    # 388    (28.9, 46.6]          4  ...         82      2
    # 389    (28.9, 46.6]          4  ...         82      1
    # 390   (25.65, 28.9]          4  ...         82      1
    # 391    (28.9, 46.6]          4  ...         82      1
    # 
    # [392 rows x 8 columns]

    # Fit model based on DAG and discretized continous columns
    model = bn.parameter_learning.fit(DAG, df_discrete)
    
    # Use MLE method
    # model_mle = bn.parameter_learning.fit(DAG, df_discrete, methodtype="maximumlikelihood")

    # Independence test
    model = bn.independence_test(model, df, prune=True)

    # Make plot
    bn.plot(model)
    bn.plot(model, interactive=True)


.. _fig_cont_1:

.. figure:: ../figs/fig_cont_1.png


Let's print some categories and evidence.

.. code-block:: python

    # Print CPDs
    print(model["model"].get_cpds("mpg"))

.. table::

    +---------------------+-----+--------------------------+
    | weight              | ... | weight((3657.5, 5140.0]) |
    +---------------------+-----+--------------------------+
    | mpg((8.624, 15.25]) | ... | 0.29931972789115646      |
    +---------------------+-----+--------------------------+
    | mpg((15.25, 17.65]) | ... | 0.19727891156462582      |
    +---------------------+-----+--------------------------+
    | mpg((17.65, 21.3])  | ... | 0.13313896987366375      |
    +---------------------+-----+--------------------------+
    | mpg((21.3, 25.65])  | ... | 0.12439261418853255      |
    +---------------------+-----+--------------------------+
    | mpg((25.65, 28.9])  | ... | 0.12439261418853255      |
    +---------------------+-----+--------------------------+
    | mpg((28.9, 46.6])   | ... | 0.12147716229348882      |
    +---------------------+-----+--------------------------+

.. code-block:: python

    print("Weight categories: ", df_disc["weight"].dtype.categories)
    # Weight categories:  IntervalIndex([(1577.73, 2217.0], (2217.0, 2959.5], (2959.5, 3657.5], (3657.5, 5140.0]], dtype='interval[float64, right]')
    
    evidence = {"weight": bn.discretize_value(df_discrete["weight"], 3000.0)}
    print(evidence)
    # {'weight': Interval(2959.5, 3657.5, closed='right')}


Inference
======================================

Making inferences is performed in the usual manner.

.. code-block:: python

    print(bn.inference.fit(model, variables=["mpg"], evidence=evidence, verbose=0))

.. table::

    +---------------------+------------+
    | mpg                 |   phi(mpg) |
    +=====================+============+
    | mpg((8.624, 15.25]) |     0.1510 |
    +---------------------+------------+
    | mpg((15.25, 17.65]) |     0.1601 |
    +---------------------+------------+
    | mpg((17.65, 21.3])  |     0.2665 |
    +---------------------+------------+
    | mpg((21.3, 25.65])  |     0.1540 |
    +---------------------+------------+
    | mpg((25.65, 28.9])  |     0.1327 |
    +---------------------+------------+
    | mpg((28.9, 46.6])   |     0.1358 |
    +---------------------+------------+    


References
----------
.. [1] Yi-Chun Chen, Tim Allan Wheeler, Mykel John Kochenderfer (2015),
       Learning Discrete Bayesian Networks from Continuous Data :arxiv:`1512.02406`

.. [2] Julia 0.4 implementation:
       https://github.com/sisl/LearnDiscreteBayesNets.jl
       

.. include:: add_bottom.add