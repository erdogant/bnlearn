.. _code_directive:

-------------------------------------


Structure learning
==================

With structure learning we can estimate a DAG that captures the dependencies between the variables in the data set.
However, realize that the search space of DAGs is super-exponential in the number of variables and you may end up in finding a local maxima. To learn model structure (a DAG) from a data set, there are three broad techniques:

  1. Score-based structure learning (BIC/BDeu/K2 score; exhaustive search, hill climb/tabu search)
      a. exhaustivesearch
      b. hillclimbsearch
  2. Constraint-based structure learning (PC)
      a. chi-square test
  3. Hybrid structure learning (The combination of both techniques) (MMHC)


Learning the graph of the data using structure learning and a score-based approach.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

This approach contains model selection as an optimization task with two building blocks:
  1. A scoring function sD:->R that maps models to a numerical score, based on how well they fit to a given data set D.
  2. A search strategy to traverse the search space of possible models M and select a model with optimal score.
  
Commonly used scoring functions to measure the fit between model and data are Bayesian Dirichlet scores such as BDeu or K2 and the Bayesian Information Criterion (BIC, also called MDL). BDeu is dependent on an equivalent sample size.

For this example, we will be investigating the sprinkler data set. This is a very simple data set with 4 variables and each variable can contain value [1] or [0]. The question we can ask: What are the relationships and dependencies across the variables? Note that his data set is already pre-processed and no missing values are present.


Let's bring in our dataset.

.. code-block:: python

  import bnlearn
  df = bnlearn.import_example()
  df.head()


.. table::

  +--------+-----------+------+-------------+
  |Cloudy  | Sprinkler | Rain |  Wet_Grass  |
  +========+===========+======+=============+
  |    0   |      1    |  0   |      1      |
  +--------+-----------+------+-------------+
  |    1   |      1    |  1   |      1      |
  +--------+-----------+------+-------------+
  |    1   |      0    |  1   |      1      |
  +--------+-----------+------+-------------+
  |    ... |      ...  | ...  |     ...     |
  +--------+-----------+------+-------------+
  |    0   |      0    |  0   |      0      |
  +--------+-----------+------+-------------+
  |    1   |      0    |  0   |      0      |
  +--------+-----------+------+-------------+
  |    1   |      0    |  1   |      1      |
  +--------+-----------+------+-------------+

From the *bnlearn* library, we'll need the
:class:`~bnlearn.structure_learning.fit` for this exercise:


.. code-block:: python

  model = bnlearn.structure_learning.fit(df)
  G = bnlearn.plot(model)


