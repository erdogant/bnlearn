.. _code_directive:

-------------------------------------

Introduction
''''''''''''

Learning a Bayesian network can be split into two problems which are both implemented in this package:

* Structure learning: Given a set of data samples, estimate a DAG that captures the dependencies between the variables.
* Parameter learning: Given a set of data samples and a DAG that captures the dependencies between the variables, estimate the (conditional) probability distributions of the individual variables.


The library supports Parameter learning for *discrete* nodes:
  * Maximum Likelihood Estimation
  * Bayesian Estimation

Structure learning for *discrete*, *fully observed* networks:
  * Score-based structure estimation (BIC/BDeu/K2 score; exhaustive search, hill climb/tabu search)
  * Constraint-based structure estimation (PC)
  * Hybrid structure estimation (MMHC)


**The following functions are available after importing bnlearn:**

* Structure learning
* Parameter learning
* Inference
* Sampling
* Plot
* comparing two networks
* loading bif files
* conversion of directed to undirected graphs
