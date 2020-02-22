Sampling and datasets
=====================

Generating samples can be very usefull. In order to do that, it requires a DAG connected with CPDs.
It is also possible to create a DAG manually (see the create DAG) or load an existing one as shown below.


Various DAGs available that can be easily loaded:

.. code-block:: python

   import bnlearn

   # The following models can be loaded:
   loadfile = 'sprinkler'
   loadfile = 'alarm'
   loadfile = 'andes'
   loadfile = 'asia'
   loadfile = 'pathfinder'
   loadfile = 'sachs'
   loadfile = 'miserables'

   DAG = bnlearn.import_DAG(loadfile)


Models are usually stored in bif files which can also be imported:

.. code-block:: python

   filepath = 'directory/to/model.bif'

   DAG = bnlearn.import_DAG(filepath)


Example Sampling 1
''''''''''''''''''


.. code-block:: python
 
   import bnlearn

   DAG = bnlearn.import_DAG('sprinkler', CPD=True)

   df = bnlearn.sampling(DAG, n=1000)

   df.head()


.. table::

  +--------+-----------+------+-------------+
  |Cloudy  | Sprinkler | Rain |  Wet_Grass  |
  +========+===========+======+=============+
  |    0   |      1    |  0   |      1      |
  +--------+-----------+------+-------------+
  |    1   |      1    |  1   |      1      |
  +--------+-----------+------+-------------+
  |    1   |      0    |  1   |      1      |
  +--------+-----------+------+-------------+
  |    ... |      ...  | ...  |     ...     |
  +--------+-----------+------+-------------+
  |    0   |      0    |  0   |      0      |
  +--------+-----------+------+-------------+
  |    1   |      0    |  0   |      0      |
  +--------+-----------+------+-------------+
  |    1   |      0    |  1   |      1      |
  +--------+-----------+------+-------------+


