Working with Continous data
=========================================
Learning Bayesian Networks from continuous data is a challanging task.
There are different manner on how to work with continuous and/or hybrid datasets. Each manner has their own advantages and disadvantages. 

In ``bnlearn`` the following options are available to work with continuous datasets:

* 1. Discretize continuous datasets manually using domain knowledge.
* 2. Discretize continuous datasets using a principled Bayesian discretization method.
* 3. Model continuous and hybrid datasets in a semi-parametric approach that assumes a linear relationships.


Discretize continuous datasets manually
=========================================

Discretizing continuous datasets manually using domain knowledge involves dividing a continuous variable into a set of discrete intervals based on an understanding of the data's context and the relationships between variables. This method allows for meaningful groupings of data points, which can simplify analysis and improve interpretability in models. 

By leveraging expertise in the subject matter, the intervals or thresholds can be chosen to reflect real-world significance, such as categorizing weather conditions into meaningful ranges (e.g., "freezing," "warm," "hot"). This approach contrasts with automatic binning methods (as depicted in approach 2), such as equal-width or equal-frequency binning, where intervals may not correspond to meaningful domain-specific boundaries.

For instance, lets load the auto mpg data set and based on automotive standards, we can define horsepower categories:

* Low: Cars with horsepower less than 100 (typically small, fuel-efficient cars)
* Medium: Cars with horsepower between 100 and 150 (moderate performance cars)
* High: Cars with horsepower above 150 (high-performance vehicles)

After all continuous variables are catagorized, the normal structure learning procedure can be applied.


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

    # Define horsepower bins based on domain knowledge
    bins = [0, 100, 150, df['horsepower'].max()]
    labels = ['low', 'medium', 'high']

    # Discretize horsepower using the defined bins
    df['horsepower_category'] = pd.cut(df['horsepower'], bins=bins, labels=labels, include_lowest=True)

    print(df[['horsepower', 'horsepower_category']].head())
    #    horsepower horsepower_category
    # 0       130.0              medium
    # 1       165.0                high
    # 2       150.0              medium
    # 3       150.0              medium
    # 4       140.0              medium



Discretize continuous datasets automatically
============================================

Automatic discritizing datasets is accomplished by using a principled Bayesian discretization method.
The method is created by Yi-Chun Chen et al in Julia. The code is ported to Python and is now part of ``bnlearn``.
Yi-Chun Chen demonstrates that his proposed method is superior to the established minimum description length algorithm.
A disadvantage of this approach is that you need to pre-define the edges before you can apply the discritization method.
The underlying idea is that after applying this discritization method, structure learning approaches can then be applied.
To demonstrate the usage of automatically discritizing continuous data, lets use the **auto mpg** dataset again.




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

    # Plot the DAG
    bn.plot(DAG)

    # Plot the DAG using graphviz
    bn.plot_graphviz(DAG)

.. _fig_auto_mpg_DAG_edges:

.. figure:: ../figs/auto_mpg_DAG_edges.png



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
******************

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
    bn.plot(model, edge_labels='pvalue')

    # Make plot with graphviz
    bn.plot_graphviz(model, edge_labels='pvalue')

    # Create interactive plot
    bn.plot(model, interactive=True)


.. |fig2a| image:: ../figs/fig2a.png
.. |fig2b| image:: ../figs/fig2b.png

.. table::
   :align: center

   +----------+
   | |fig2a|  |
   +----------+
   | |fig2b|  |
   +----------+


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3blocks/bnlearn_continous_example_1.html" height="700px" width="750px", frameBorder="0"></iframe>


Parameter learning
******************

Let's continue with parameter learning on the continuous data set and see whether we can estimate the CPDs.


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
    bn.plot(model, edge_labels='pvalue')

    # Make plot graphviz
    bn.plot_graphviz(model, edge_labels='pvalue')

    # Create interactive plot.
    bn.plot(model, interactive=True)



.. |fig3a| image:: ../figs/fig_cont_1.png
.. |fig3b| image:: ../figs/fig_cont_1b.png

.. table::
   :align: center

   +----------+
   | |fig3a|  |
   +----------+
   | |fig3b|  |
   +----------+




.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3blocks/bnlearn_continous_example_2.html" height="700px" width="750px", frameBorder="0"></iframe>


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
    

Inferences
**********

Making inferences can be perfomred using the fitted model. Note that the evidence should be discretized for which we can 
use the ``discretize_value`` function.

.. code-block:: python

    evidence = {"weight": bn.discretize_value(df_discrete["weight"], 3000.0)}
    print(evidence)
    # {'weight': Interval(2959.5, 3657.5, closed='right')}

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


Modelling Continuous Datasets
=========================================

LiNGAM Example
*********************

To demonstrate how the LiNGAM works, it is best to do it with a small toy example.

Here's the improved version of the text:

Let's create test data containing six variables.
The goal of this dataset is to demonstrate the contribution of different variables and their causal impact on other variables.
All variables must be consistent, as in any other dataset. The sample size is set to n=1000 with a uniform distribution.
If the number of samples is much smaller, say in the tens, the method becomes less reliable due to insufficient information to determine causality.

We will establish dependencies between variables and then allow the model to infer the original values.

	1. Step 1: ``x3`` is the root node and is initialized with a uniform distribution.
	2. Step 2: ``x0`` and ``x2`` are created by multiplying with the values of ``x3``, making them dependent on ``x3``.
	3. Step 3: ``x5`` is created by multiplying with the values of ``x0``, making it dependent on ``x0``.
	4. Step 4: ``x1`` and ``x4`` are created by multiplying with the values of ``x0``, making them dependent on ``x0``.


.. |fig8a| image:: ../figs/fig_lingam_example_input.png

.. table::
   :align: center

   +----------+
   | |fig8a|  |
   +----------+


.. code-block:: python

	import numpy as np
	import pandas as pd
	from lingam.utils import make_dot

	# Number of samples
	n=1000

	# step 1
	x3 = np.random.uniform(size=n)
	# step 2
	x0 = 3.0*x3 + np.random.uniform(size=n)
	x2 = 6.0*x3 + np.random.uniform(size=n)
	# step 3
	x5 = 4.0*x0 + np.random.uniform(size=n)
	# step 4
	x1 = 3.0*x0 + 2.0*x2 + np.random.uniform(size=n)
	x4 = 8.0*x0 - 1.0*x2 + np.random.uniform(size=n)
	df = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T ,columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
	df.head()

	m = np.array([[0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
		      [3.0, 0.0, 2.0, 0.0, 0.0, 0.0],
		      [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
		      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		      [8.0, 0.0,-1.0, 0.0, 0.0, 0.0],
		      [4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
	dot = make_dot(m)
	dot


Structure learning can be applied with the direct-lingam method for fitting.


.. code-block:: python

	model = bn.structure_learning.fit(df, methodtype='direct-lingam')

	# When we no look at the output, we can see that the dependency values are very well recovered for the various variables.

	print(model['adjmat'])
	# target        x0        x1       x2   x3        x4       x5
	# source                                                     
	# x0      0.000000  2.987320  0.00000  0.0  8.057757  3.99624
	# x1      0.000000  0.000000  0.00000  0.0  0.000000  0.00000
	# x2      0.000000  2.010043  0.00000  0.0 -0.915306  0.00000
	# x3      2.971198  0.000000  5.98564  0.0 -0.704964  0.00000
	# x4      0.000000  0.000000  0.00000  0.0  0.000000  0.00000
	# x5      0.000000  0.000000  0.00000  0.0  0.000000  0.00000
	bn.plot(model)

	# Compute edge strength with the chi_square test statistic
	model = bn.independence_test(model, df, prune=False)
	
	print(model['adjmat'])
	# target        x0        x1       x2   x3        x4       x5
	# source                                                     
	# x0      0.000000  2.987320  0.00000  0.0  8.057757  3.99624
	# x1      0.000000  0.000000  0.00000  0.0  0.000000  0.00000
	# x2      0.000000  2.010043  0.00000  0.0 -0.915306  0.00000
	# x3      2.971198  0.000000  5.98564  0.0 -0.704964  0.00000
	# x4      0.000000  0.000000  0.00000  0.0  0.000000  0.00000
	# x5      0.000000  0.000000  0.00000  0.0  0.000000  0.00000

	# Using the causal_order_ properties, we can see the causal ordering as a result of the causal discovery.

	print(model['causal_order'])
	# ['x3', 'x0', 'x5', 'x2', 'x1', 'x4']

	# We can draw a causal graph by utility funciton.
	bn.plot(model)

.. |fig7a| image:: ../figs/fig_lingam_example_1.png

.. table::
   :align: center

   +----------+
   | |fig7a|  |
   +----------+

This example nicely demonstrates that we can capture the dependencies with the causal factors acurately.


Direct-LiNGAM method
*********************

The Direct-LiNGAM method 'direct-lingam' is a semi-parametric approach that assumes a linear relationship among observed variables while ensuring that the error terms follow a non-Gaussian distribution, with the constraint that the graph remains acyclic. This method involves repeated regression analysis and independence assessments using linear regression with least squares. In each regression, one variable serves as the dependent variable (outcome), while the other acts as the independent variable (predictor). This process is applied to each type of variable. When regression analysis is conducted in the correct causal order, the independent variables and error terms will exhibit independence. Conversely, if the regression is performed under an incorrect causal order, the independence of the explanatory variables and error terms is disrupted. By leveraging the dependency properties (where both residuals and explanatory variables share common error terms), it becomes possible to infer the causal order among the variables. Furthermore, for a given observed variable, any explanatory variable that remains independent of the residuals, regardless of the other variables considered, can be inferred as the first in the causal hierarchy.

Or in other words, the lingam-direct method allows you to model continuous and mixed datasets.
A disadvantage is that causal discovery of structure learning is the end-point when uing this method. It is not possible to perform parameter learning and inferences.

.. code-block:: python

    # Import
    import bnlearn as bn
    
    # Load data set
    df = bn.import_example(data='auto_mpg')

    # Structure learning
    model = bn.structure_learning.fit(df, methodtype='direct-lingam', params_lingam = {'random_state': 2})

    # Compute edge strength
    model = bn.independence_test(model, df)

    # Plot
    bn.plot(model)

    # Plot with graphviz
    dotgraph = bn.plot_graphviz(model)
    dotgraph
    dotgraph.view(filename=r'dotgraph_auto_mpg_lingam_direct')


Using the LINGAM method, the values on the edges describe the dependency using a multiplication factor of one variable to another. As an example, Origin -> -10 -> Displacement tells us Displacement has values that are factor -10 lower than origin.

.. |fig4a| image:: ../figs/fig_auto_mpg_lingam_a.png
.. |fig4b| image:: ../figs/fig_auto_mpg_lingam_b.png

.. table::
   :align: center

   +----------+
   | |fig4a|  |
   +----------+
   | |fig4b|  |
   +----------+


ICA-LiNGAM method
*********************

The ICA-LiNGAM method 'ica-lingam' is also from lingam and follows the same procedure.

.. code-block:: python

    # Import
    import bnlearn as bn
    
    # Load data set
    df = bn.import_example(data='auto_mpg')

    # Structure learning
    model = bn.structure_learning.fit(df, methodtype='ica-lingam')

    # Compute edge strength
    model = bn.independence_test(model, df)

    # Plot
    bn.plot(model)

    # Plot with graphviz
    dotgraph = bn.plot_graphviz(model)
    dotgraph
    dotgraph.view(filename=r'dotgraph_auto_mpg_lingam_ica')


.. |fig6a| image:: ../figs/fig_auto_mpg_lingam_ica_a.png
.. |fig6b| image:: ../figs/fig_auto_mpg_lingam_ica_b.png

.. table::
   :align: center

   +----------+
   | |fig6a|  |
   +----------+
   | |fig6b|  |
   +----------+


PC method
*********************

A different, but quite straightforward approach to build a DAG from data is identifing independencies in the data set using hypothesis tests and then construct DAG (pattern) according to identified independencies (Conditional) Independence Tests. Independencies in the data can be identified using chi2 conditional independence tests.

The Constraint-Based PC Algorithm (named after Peter and Clark, its inventors) is a popular method in causal inference and Bayesian network learning. It is a type of constraint-based algorithm, which uses conditional independence tests to build a causal graph from data. This algorithm is widely used to learn the structure of Bayesian networks and causal graphs by identifying relationships between variables.

DAG (pattern) construction
With a method for independence testing at hand, we can construct a DAG from the data set in three steps:
	* 1. Construct an undirected skeleton.
        * 2. Orient compelled edges to obtain partially directed acyclid graph.
        * 3. Extend DAG pattern to a DAG by conservatively orienting the remaining edges in some way.


.. code-block:: python

    # Import
    import bnlearn as bn
    
    # Load data set
    df = bn.import_example(data='auto_mpg')

    # Structure learning
    model = bn.structure_learning.fit(df, methodtype='pc')

    # Compute edge strength
    model = bn.independence_test(model, df)

    # Plot
    bn.plot(model, edge_labels='pvalue')

    # Plot with graphviz
    dotgraph = bn.plot_graphviz(model, edge_labels='pvalue')
    dotgraph
    dotgraph.view(filename=r'dotgraph_auto_mpg_PC')


PC PDAG construction is only guaranteed to work under the assumption that the identified set of independencies is *faithful*, i.e. there exists a DAG that exactly corresponds to it. Spurious dependencies in the data set can cause the reported independencies to violate faithfulness. It can happen that the estimated PDAG does not have any faithful completions (i.e. edge orientations that do not introduce new v-structures). In that case a warning is issued.

.. |fig5a| image:: ../figs/fig_auto_mpg_PC_a.png
.. |fig5b| image:: ../figs/fig_auto_mpg_PC_b.png

.. table::
   :align: center

   +----------+
   | |fig5a|  |
   +----------+
   | |fig5b|  |
   +----------+



References
----------

    1. Yi-Chun Chen, Tim Allan Wheeler, Mykel John Kochenderfer (2015),
    `Learning Discrete Bayesian Networks from Continuous Data <https://arxiv.org/abs/1512.02406>`_

    2. Julia 0.4 implementation:
       https://github.com/sisl/LearnDiscreteBayesNets.jl
       

.. include:: add_bottom.add