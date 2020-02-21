Directed Acyclic Graphs
=======================

It is also possible to create a DAG manually. In this example I will create the simple Sprinkler example by hand.

First we need to define the one-to-one relationships (edges) between the variables. Here we make the edges:

* Cloudy->Sprinkler
* Cloudy->Rain
* Sprinkler->Wet_Grass
* Rain->Wet_Grass


.. code-block:: python

   # Import the library
   from pgmpy.models import BayesianModel
   from pgmpy.factors.discrete import TabularCPD

   # Define the network structure
   DAG = BayesianModel([('Cloudy', 'Sprinkler'),
                          ('Cloudy', 'Rain'),
                          ('Sprinkler', 'Wet_Grass'),
                          ('Rain', 'Wet_Grass')])


Lets make the plot. Note that the plot can be differently orientiated if you re-make the plot.

.. code-block:: python

   bnlearn.plot(model)


.. _fig-sprinkler:

.. figure:: ../figs/fig_sprinkler_sl.png

  Created DAG.


For each node we can specifying the probability distributions:

.. code-block:: python

   # Cloudy
   cpt_cloudy = TabularCPD(variable='Cloudy', variable_card=2, values=[[0.5], [0.5]])

   # Sprinkler
   cpt_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2,
                              values=[[0.5, 0.9], 
			              [0.5, 0.1]],
                              evidence=['Cloudy'], evidence_card=[2])
   # Rain
   cpt_rain = TabularCPD(variable='Rain', variable_card=2,
                         values=[[0.8, 0.2],
			         [0.2, 0.8]],
                         evidence=['Cloudy'], evidence_card=[2])

   # Wet Grass
   cpt_wet_grass = TabularCPD(variable='Wet_Grass', variable_card=2,
                              values=[[1, 0.1, 0.1, 0.01],
                                      [0, 0.9, 0.9, 0.99]],
                              evidence=['Sprinkler', 'Rain'],
                              evidence_card=[2, 2])

Now need to connect the DAG with CPDs.

.. code-block:: python

   DAG.add_cpds(cpt_cloudy, cpt_sprinkler, cpt_rain, cpt_wet_grass)


Nice work! You created a directed acyclic graph with probability tables connected to it.
This is an great basis to make inferences or update your this model with new data (parameter learning).

.. code-block:: python
   
   q1 = bnlearn.inference.fit(DAG, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})

