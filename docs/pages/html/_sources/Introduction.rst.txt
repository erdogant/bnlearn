Introduction
''''''''''''''

Learning a Bayesian network can be split into **structure learning** and **parameter learning** which are both implemented in ``bnlearn``.

* **Structure learning**: Given a set of data samples, estimate a DAG that captures the dependencies between the variables.
* **Parameter learning**: Given a set of data samples and a DAG that captures the dependencies between the variables, estimate the (conditional) probability distributions of the individual variables.

The library supports Parameter learning for *discrete* nodes:
  * Maximum Likelihood Estimation
  * Bayesian Estimation

**The following functions are available**

* Structure learning
* Parameter learning
* Inference
* Sampling
* Plot
* comparing two networks
* loading bif files
* conversion of directed to undirected graphs



Structure Learning algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``bnlearn`` contains **score-based**, **local discovery**, **Bayesian network**, and **constraint-based** structure learning algorithms for *discrete*, *fully observed* networks.

**Score-based approaches have two main components**:
* The search algorithm to optimize throughout the search space of all possible DAGs.
* The scoring function indicates how well the Bayesian network fits the data.

Score-based algorithm can be used with the following score functions:

* categorical data (multinomial distribution):
   * the Bayesian Information Criterion (bic)
   * the K2 score (k2)
   * a score equivalent Dirichlet posterior density (bdeu);


Constraint-based structure learning algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With constraint-based structure learning, a DAG can be constructed by identifying independencies in the data set using hypothesis tests, such as chi2 test statistic.
This approach does rely on statistical tests and conditional hypotheses to learn independence among the variables in the model.
The P-value of the chi2 test is the probability of observing the computed chi2 statistic, given the null hypothesis that X and Y are independent given Z.
This can be used to make independent judgments, at a given level of significance.
An example of a constraint-based approach is the PC algorithm which starts with a complete fully connected graph and removes edges based on the results of the tests if the nodes are independent until a stopping criterion is achieved.

* constraintsearch (cs)



.. include:: add_bottom.add