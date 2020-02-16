.. _code_directive:

-------------------------------------

Introduction
''''''''''''

Learning a Bayesian network can be split into two problems which are both implemented in this package:
* Structure learning: Given a set of data samples, estimate a DAG that captures the dependencies between the variables.
* Parameter learning: Given a set of data samples and a DAG that captures the dependencies between the variables, estimate the (conditional) probability distributions of the individual variables.

The following functions are available after importing bnlearn.
--------------------------------------------------------------

* Structure learning
* Parameter learning
* Inference
* Sampling
* Compare 2 graphs
* Plot
* Make graph undirected
