Directed Acyclic Graphs
=======================

It is also possible to create a DAG manually. In this example I will create the simple Sprinkler example by hand.

First we need to define the one-to-one relationships (edges) between the variables. Here we make the edges:

* Cloudy->Sprinkler
* Cloudy->Rain
* Sprinkler->Wet_Grass
* Rain->Wet_Grass

>>> Import the library
>>> import from pgmpy.models import BayesianModel
>>>
>>> # Define the network structure
>>> model = BayesianModel([('Cloudy', 'Sprinkler'),
>>>                        ('Cloudy', 'Rain'),
>>>                        ('Sprinkler', 'Wet_Grass'),
>>>                        ('Rain', 'Wet_Grass')])

Here we can define the CPD:

>>> # Cloudy
>>> cpt_cloudy = TabularCPD(variable='Cloudy', variable_card=2, values=[[0.5], [0.5]])
>>>
>>> # Sprinkler
>>> cpt_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2,
>>>                            values=[[0.5, 0.9], [0.5, 0.1]],
>>>                            evidence=['Cloudy'], evidence_card=[2])
>>> # Rain
>>> cpt_rain = TabularCPD(variable='Rain', variable_card=2,
>>>                       values=[[0.8, 0.2], [0.2, 0.8]],
>>>                       evidence=['Cloudy'], evidence_card=[2])
>>>
>>> # Wet Grass
>>> cpt_wet_grass = TabularCPD(variable='Wet_Grass', variable_card=2,
>>>                            values=[[1, 0.1, 0.1, 0.01],
>>>                                   [0, 0.9, 0.9, 0.99]],
>>>                            evidence=['Sprinkler', 'Rain'],
>>>                            evidence_card=[2, 2])

Now need to connect the DAG with CPDs.

>>> model.add_cpds(cpt_cloudy, cpt_sprinkler, cpt_rain, cpt_wet_grass)
