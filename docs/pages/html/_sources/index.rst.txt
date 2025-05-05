BNLearn's Documentation
=======================

|python| |pypi| |docs| |LOC| |downloads_month| |downloads_total| |license| |forks| |open issues| |project status| |medium| |colab| |DOI| |donate|


-----------------------------------

.. |landscape| image:: ../figs/Landscape_Causal_Discovery_Bnlearn_for_python.png

.. table::
   :align: center

   +---------------+
   | |landscape|   |
   +---------------+

-----------------------------------

*Bnlearn* is for causal discovery using in Python!

* Contains the most-wanted Bayesian pipelines for Causal Discovery
* Simple and intuitive
* Focus on structure learning, parameter learning and inference.


-----------------------------------

Support
-----------
Yes! This library is entirely free but it runs on `coffee <https://buymeacoffee.com/erdogant>`_! :) 

.. raw:: html

    <iframe 
        srcdoc='<a href="https://www.buymeacoffee.com/erdogant" target="_blank"><img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&amp;emoji=&amp;slug=erdogant&amp;button_colour=FFDD00&amp;font_colour=000000&amp;font_family=Cookie&amp;outline_colour=000000&amp;coffee_colour=ffffff" /></a>' 
        style="border:none; width:250px; height:80px;">
    </iframe>


-----------------------------------

.. tip::
	* `Detecting causal relationships using Bayesian Structure Learning in Python. <https://towardsdatascience.com/a-step-by-step-guide-in-detecting-causal-relationships-using-bayesian-structure-learning-in-python-c20c6b31cee5>`_
	* `Designing knowledge-driven models using Bayesian theorem. <https://towardsdatascience.com/a-step-by-step-guide-in-designing-knowledge-driven-models-using-bayesian-theorem-7433f6fd64be>`_
	* `Chat with Your Dataset using Bayesian Inferences. <https://towardsdatascience.com/chat-with-your-dataset-using-bayesian-inferences-bfd4dc7f8dcd>`_
	* `The Power of Bayesian Causal Inference: A Comparative Analysis of Libraries to Reveal Hidden Causality in Your Dataset. <https://towardsdatascience.com/the-power-of-bayesian-causal-inference-a-comparative-analysis-of-libraries-to-reveal-hidden-d91e8306e25e>`_
	* `Find the Best Boosting Model using Bayesian Hyperparameter Tuning but without Overfitting. <https://towardsdatascience.com/a-guide-to-find-the-best-boosting-model-using-bayesian-hyperparameter-tuning-but-without-c98b6a1ecac8>`_


-----------------------------------

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3blocks/bnlearn_continous_example_2.html" height="700px" width="750px", frameBorder="0"></iframe>



-----------------------------------

.. note::
	**Your ❤️ is important to keep maintaining this package.** You can `support <https://erdogant.github.io/bnlearn/pages/html/Documentation.html>`_ in various ways, have a look at the `sponser page <https://erdogant.github.io/bnlearn/pages/html/Documentation.html>`_.
	Report bugs, issues and feature extensions at `github <https://github.com/erdogant/bnlearn/>`_ page.

	.. code-block:: console

	   pip install bnlearn

-----------------------------------


Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Quickstart

   Introduction
   Quickstart


.. toctree::
   :maxdepth: 2
   :caption: Installation

   Installation

.. toctree::
  :maxdepth: 2
  :caption: Discretizing

  Discretizing

.. toctree::
  :maxdepth: 2
  :caption: Structure learning

  Structure learning


.. toctree::
  :maxdepth: 2
  :caption: Parameter learning

  Parameter learning


.. toctree::
  :maxdepth: 2
  :caption: Inference

  Inference


.. toctree::
  :maxdepth: 2
  :caption: Continuous Data

  Continuous Data


.. toctree::
  :maxdepth: 2
  :caption:   Predict

  Predict

.. toctree::
  :maxdepth: 2
  :caption: Synthetic Data

  Sampling


.. toctree::
  :maxdepth: 2
  :caption: Plot

  Plot


.. toctree::
  :maxdepth: 2
  :caption: Other functionalities

  independence_test
  Create DAG
  impute
  Example Datasets
  whitelist_blacklist
  topological_sort
  dataframe conversions
  Structure_scores
  saving and loading


.. toctree::
  :maxdepth: 2
  :caption: Examples

  Examples

.. toctree::
  :maxdepth: 2
  :caption: Use Cases

  UseCases


.. toctree::
  :maxdepth: 2
  :caption: Parameters and attributes

  bnlearn.structure_learning
  bnlearn.parameter_learning
  bnlearn.inference
  bnlearn.bnlearn


.. toctree::
  :maxdepth: 1
  :caption: Documentation

  Documentation




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




.. |python| image:: https://img.shields.io/pypi/pyversions/bnlearn.svg
    :alt: |Python
    :target: https://erdogant.github.io/bnlearn/

.. |pypi| image:: https://img.shields.io/pypi/v/bnlearn.svg
    :alt: |Python Version
    :target: https://pypi.org/project/bnlearn/

.. |docs| image:: https://img.shields.io/badge/Sphinx-Docs-blue.svg
    :alt: Sphinx documentation
    :target: https://erdogant.github.io/bnlearn/

.. |LOC| image:: https://sloc.xyz/github/erdogant/bnlearn/?category=code
    :alt: lines of code
    :target: https://github.com/erdogant/bnlearn

.. |downloads_month| image:: https://static.pepy.tech/personalized-badge/bnlearn?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month
    :alt: Downloads per month
    :target: https://pepy.tech/project/bnlearn

.. |downloads_total| image:: https://static.pepy.tech/personalized-badge/bnlearn?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
    :alt: Downloads in total
    :target: https://pepy.tech/project/bnlearn

.. |license| image:: https://img.shields.io/badge/license-MIT-green.svg
    :alt: License
    :target: https://github.com/erdogant/bnlearn/blob/master/LICENSE

.. |forks| image:: https://img.shields.io/github/forks/erdogant/bnlearn.svg
    :alt: Github Forks
    :target: https://github.com/erdogant/bnlearn/network

.. |open issues| image:: https://img.shields.io/github/issues/erdogant/bnlearn.svg
    :alt: Open Issues
    :target: https://github.com/erdogant/bnlearn/issues

.. |project status| image:: http://www.repostatus.org/badges/latest/active.svg
    :alt: Project Status
    :target: http://www.repostatus.org/#active

.. |medium| image:: https://img.shields.io/badge/Medium-Blog-green.svg
    :alt: Medium Blog
    :target: https://erdogant.github.io/bnlearn/pages/html/Documentation.html#medium-blog

.. |donate| image:: https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors
    :alt: donate
    :target: https://erdogant.github.io/bnlearn/pages/html/Documentation.html#

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Colab example
    :target: https://erdogant.github.io/bnlearn/pages/html/Documentation.html#colab-notebook

.. |DOI| image:: https://zenodo.org/badge/231263493.svg
    :alt: Cite
    :target: https://zenodo.org/badge/latestdoi/231263493


.. include:: add_bottom.add
