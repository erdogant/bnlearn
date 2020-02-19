bnlearn's documentation!
========================

*bnlearn* is for learning the graphical structure of Bayesian networks in Python! What benefits does bnlearn offer over other bayesian analysis implementations?

* build on top of pgmpy library
* contains the most-wanted pipelines
* simple and intuitive
* focus on structure learning, parameter learning and inference.


Content
========

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   Introduction

.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   Quickstart

.. toctree::
  :maxdepth: 2
  :caption: Methods

  Structure learning
  Parameter learning
  Inference
  Sampling
  Datasets
  Creating DAG

.. toctree::
  :maxdepth: 2
  :caption: Additional Documentation

  API Reference
  
  
Installation
------------

It is advisable to create a new environment. 
You may need to deactivate and then activate your environment otherwise the packages may not been recognized.


.. code-block:: console

   conda create -n env_BNLEARN python=3.6
   conda activate env_BNLEARN
   conda install -c ankurankan pgmpy

   conda deactivate
   conda activate env_BNLEARN


.. code-block:: console

   pip install bnlearn


Source code and issue tracker
------------------------------

Available on Github, `erdogant/bnlearn <https://github.com/erdogant/bnlearn/>`_.
Please report bugs, issues and feature extensions there.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
