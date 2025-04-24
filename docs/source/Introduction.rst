Introduction
''''''''''''''

Learning a Bayesian network can be split into two main components: **structure learning** and **parameter learning**, both of which are implemented in ``bnlearn``.

* **Structure learning**: Given a set of data samples, estimate a Directed Acyclic Graph (DAG) that captures the dependencies between variables.
* **Parameter learning**: Given a set of data samples and a DAG that captures the dependencies between variables, estimate the (conditional) probability distributions of the individual variables.

The library supports parameter learning for *discrete* nodes using:
  * Maximum Likelihood Estimation
  * Bayesian Estimation

**Available Functions**

* Structure learning
* Parameter learning
* Inference
* Sampling
* Plotting
* Network comparison
* Loading BIF files
* Conversion of directed to undirected graphs

Structure Learning Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``bnlearn`` contains **score-based**, **local discovery**, **Bayesian network**, and **constraint-based** structure learning algorithms for *discrete*, *fully observed* networks.

**Score-based approaches consist of two main components**:
* A search algorithm to optimize throughout the search space of all possible DAGs
* A scoring function that indicates how well the Bayesian network fits the data

Score-based algorithms can be used with the following score functions for categorical data (multinomial distribution):
   * Bayesian Information Criterion (BIC)
   * K2 score
   * Score equivalent Dirichlet posterior density (BDeu)

Constraint-based Structure Learning Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Constraint-based structure learning constructs a DAG by identifying independencies in the dataset using hypothesis tests, such as the chi-square test statistic.
This approach relies on statistical tests and conditional hypotheses to learn independence among the variables in the model.
The p-value of the chi-square test represents the probability of observing the computed chi-square statistic, given the null hypothesis that X and Y are independent given Z.
This can be used to make independence judgments at a given significance level.
An example of a constraint-based approach is the PC algorithm, which starts with a complete, fully connected graph and removes edges based on the results of independence tests until a stopping criterion is achieved.

* constraintsearch (cs)

.. include:: add_bottom.add