Learning Bayesian Networks from continuous data is an challanging task.
In ``bnlearn`` this task is now accomplished by learning discrete bayesian networks from continuous data.

In order to do this, I am using a Bayesian discretization method for continuous
variables in Bayesian networks with quadratic complexity instead of the cubic
complexity of other standard techniques. Empirical demonstrations show that the
proposed method is superior to the established minimum description length algorithm.
The method is described in the paper of Yi-Chun Chen et al.

The underlying idea i that the implemented method now can perform structure learning by first discretizng the continuous
variables and simultaneously learn Bayesian network structures.



Advanced discretizing continous data
=========================================

To demonstrate the usage of parameter learning on continuous data, I will use the well known **auto mpg** data set.


.. code-block:: python

    # Import
    import bnlearn as bn
    
    # Load data set
    df = bn.import_example(data='auto_mpg')
    # Print
    print(df)

    #       mpg  cylinders  displacement  ...  acceleration  model_year  origin
    # 0    18.0          8         307.0  ...          12.0          70       1
    # 1    15.0          8         350.0  ...          11.5          70       1
    # 2    18.0          8         318.0  ...          11.0          70       1
    # 3    16.0          8         304.0  ...          12.0          70       1
    # 4    17.0          8         302.0  ...          10.5          70       1
    # ..    ...        ...           ...  ...           ...         ...     ...
    # 387  27.0          4         140.0  ...          15.6          82       1
    # 388  44.0          4          97.0  ...          24.6          82       2
    # 389  32.0          4         135.0  ...          11.6          82       1
    # 390  28.0          4         120.0  ...          18.6          82       1
    # 391  31.0          4         119.0  ...          19.4          82       1
    # 
    # [392 rows x 8 columns]

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


We can now discretize the continuous columns as following:

.. code-block:: python

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

At this point it is not different than any other discrete data set. We can specify the DAG together with the
discrete data frame and fit a model using ``bnlearn``.


Structure learning
======================================

We will learn the structure on the continuous data. Note that the data is also discretezed on a set of edges which will
likely introduce a bias in the learned structure.

.. code-block:: python

    # Learn the structure
    model = bn.structure_learning.fit(df_discrete, methodtype='hc', scoretype='bic')

    # Independence test
    model = bn.independence_test(model, df, prune=True)
    # [bnlearn] >Compute edge strength with [chi_square]
    # [bnlearn] >Edge [weight <-> mpg] [P=0.999112] is excluded because it was not significant (P<0.05) with [chi_square]

    # Make plot
    bn.plot(model)
    # Create interactive plot
    bn.plot(model, interactive=True)

.. _fig_cont_2:

.. figure:: ../figs/fig_cont_2.png


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3blocks/bnlearn_continous_example_1.html" height="400px" width="750px", frameBorder="0"></iframe>


Parameter learning
======================================

To demonstrate the usage of parameter learning on continuous data, I will use the well known **auto mpg** data set.


.. code-block:: python

    # Fit model based on DAG and discretized continous columns
    model = bn.parameter_learning.fit(DAG, df_discrete)
    
    # Use MLE method
    # model_mle = bn.parameter_learning.fit(DAG, df_discrete, methodtype="maximumlikelihood")


After fitting the model on the DAG and data frame, we can perform the independence test to remove any spurious edges and
create a plot. In this case, the tooltips will contain the CPDs as these are computed with parameter learning.

.. code-block:: python

    # Independence test
    model = bn.independence_test(model, df, prune=True)

    # Make plot
    bn.plot(model)
    # Create interactive plot.
    bn.plot(model, interactive=True)


.. _fig_cont_1:

.. figure:: ../figs/fig_cont_1.png

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3blocks/bnlearn_continous_example_2.html" height="400px" width="750px", frameBorder="0"></iframe>


There are various manners to deeper investigate the results such as looking at the CPDs.

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

Making inferences can be perfomred using the fitted model.

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

    1. Yi-Chun Chen, Tim Allan Wheeler, Mykel John Kochenderfer (2015),
       Learning Discrete Bayesian Networks from Continuous Data arxiv 1512.02406

    2. Julia 0.4 implementation:
       https://github.com/sisl/LearnDiscreteBayesNets.jl
       

.. include:: add_bottom.add