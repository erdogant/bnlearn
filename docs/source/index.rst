bnlearn's documentation!
========================

WORK IN PROGRES!

*bnlearn* is Python package for learning the graphical structure of Bayesian networks. What benefits does bnlearn offer over other bayesian analysis implementations?

* build on top of pgmpy library
* contains the most-wanted pipelines
* simple and intuitive
* focus on structure learning, parameter learning and inference.


Content
=======
.. toctree::
   :maxdepth: 1
   :caption: Introduction
   Introduction

.. toctree::
   :maxdepth: 1
   :caption: Quickstart
   Quickstart

.. toctree::
  :maxdepth: 1
  :caption: Structure learning
  Structure learning

.. toctree::
  :maxdepth: 1
  :caption: Parameter learning
  Parameter learning

.. toctree::
  :maxdepth: 1
  :caption: Inference
  Inference


Installation
------------

.. code-block:: console

   # It is advisable to create a new environment. 
   conda create -n env_BNLEARN python=3.6
   conda activate env_BNLEARN
   conda install -c ankurankan pgmpy

   # You may need to deactivate and then activate your environment otherwise the packages may not been recognized.
   conda deactivate
   conda activate env_BNLEARN

.. code-block:: console

   pip install bnlearn


Source code and issue tracker
------------------------------

Available on Github, `erdogant/bnlearn <https://github.com/erdogant/bnlearn/>`_.
Please report bugs, issues and feature extensions there.

Citing *bnlearn*
----------------
Here is an example BibTeX entry:

@misc{erdogant2019bnlearn,
  title={bnlearn},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdogant/bnlearn}}}



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
