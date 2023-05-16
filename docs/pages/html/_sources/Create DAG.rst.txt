Directed Acyclic Graphs
========================

This example is to better understand the importance and working of a Directed Acyclic Graph. The underneath topics are going to be explained:

Overview
''''''''

* Building a DAG
* plotting a DAG
* Specifying your own probability distributions
* Estimating parameters of CPDs
* Inference on the causal generative model


Building a causal DAG
'''''''''''''''''''''
If you readily know (or you have domain knowledge) of the relationships between variables, we can setup the (causal) relationships between the variables with a directed graph (DAG). 
Each node corresponds to a variable and each edge represents conditional dependencies between pairs of variables.
In bnlearn, we can graphically represent the relationships between variables. To demonstrate this I will create the simple Sprinkler example by hand.

First we need to define the one-to-one relationships (edges) between the variables. Here we make the edges:

* Cloudy    -> Sprinkler
* Cloudy    -> Rain
* Sprinkler -> Wet_Grass
* Rain      -> Wet_Grass


.. code-block:: python

   # Import the library
   import bnlearn

   # Define the network structure
   edges = [('Cloudy', 'Sprinkler'),
            ('Cloudy', 'Rain'),
            ('Sprinkler', 'Wet_Grass'),
            ('Rain', 'Wet_Grass')]

   # Make the actual Bayesian DAG
   DAG = bnlearn.make_DAG(edges)


Lets make the plot. Note that the plot can be differently orientiated if you re-make the plot.

.. code-block:: python
   
   bnlearn.plot(DAG)


.. _fig-sprinkler:

.. figure:: ../figs/fig_sprinkler_sl.png

  Causal DAG.

We call this a causal DAG because we have assumed that the edges we encoded represent our causal assumptions about the system.


The causal DAG as a generative representation of joint probability
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Any DAG (causal or otherwise) that we might specify for this data represents a factorization of the joint probability distribution of the variables.

.. code-block:: python
   
   bnlearn.print_CPD(DAG)

   # [BNLEARN.print_CPD] No CPDs to print. Use bnlearn.plot(DAG) to make a plot.


There are no CPDs attached to the DAG yet. Therefore there is nothing to show.


Specifying the probability distributions on your own
''''''''''''''''''''''''''''''''''''''''''''''''''''

Each factor is a conditional probability distribution (CPD). In the discrete case the CPD is sometimes called a conditional probability table (CPT).
Though we can factorize over any DAG and get a set of CPDs, when we factorize along a DAG we consider to be a representation of causality, we call each CPD a causal Markov kernel (CMK).
The factorization that provides a set of CMKs is the most useful factorization because CMKs correspond to independent causal mechanisms we assume to be invariant across data sets. 
Here again, the term CPD is more often used than CMK.

For each node we can specify the probability distributions as following:

.. code-block:: python

   # Import the library
   from pgmpy.factors.discrete import TabularCPD

   # Cloudy
   cpt_cloudy = TabularCPD(variable='Cloudy', variable_card=2, values=[[0.3], [0.7]])
   print(cpt_cloudy)


.. table::

   +-----------+-----+
   | Cloudy(0) | 0.3 |
   +-----------+-----+
   | Cloudy(1) | 0.7 |
   +-----------+-----+


.. code-block:: python

   # Sprinkler
   cpt_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2,
                              values=[[0.5, 0.9], 
			              [0.5, 0.1]],
                              evidence=['Cloudy'], evidence_card=[2])
   print(cpt_sprinkler)

   # Rain
   cpt_rain = TabularCPD(variable='Rain', variable_card=2,
                         values=[[0.8, 0.2],
			         [0.2, 0.8]],
                         evidence=['Cloudy'], evidence_card=[2])
   print(cpt_rain)

   # Wet Grass
   cpt_wet_grass = TabularCPD(variable='Wet_Grass', variable_card=2,
                              values=[[1, 0.1, 0.1, 0.01],
                                      [0, 0.9, 0.9, 0.99]],
                              evidence=['Sprinkler', 'Rain'],
                              evidence_card=[2, 2])
   print(cpt_wet_grass)

Now need to connect the DAG with CPDs.

.. code-block:: python

   DAG = bnlearn.make_DAG(DAG, CPD=[cpt_cloudy, cpt_sprinkler, cpt_rain, cpt_wet_grass])



Nice work! You created a directed acyclic graph with probability tables connected to it.
To further examine the CPDs, print the DAG as following:

.. code-block:: python

   bnlearn.print_CPD(DAG)


Inference on the causal generative model
''''''''''''''''''''''''''''''''''''''''

This is an great basis to make inferences or update your this model with new data (parameter learning).

.. code-block:: python
   
   q1 = bnlearn.inference.fit(DAG, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})



.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>



.. include:: add_bottom.add