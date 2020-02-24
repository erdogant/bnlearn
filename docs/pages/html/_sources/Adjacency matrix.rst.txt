Adjacency matrix
=================

The adjacency matrix is a important way to store relationships across variables or nodes.
In graph theory, it is a square matrix used to represent a finite graph. The elements of the matrix indicate whether pairs of vertices are adjacent or not in the graph. 


Algorithms
''''''''''

Bnlearn outputs an adjacency matrix in some functionalities. Values 0 or False indicate that nodes are not connected whereas pairs of vertices with value >0 or True are connected.


**Importing a DAG**

Extracting adjacency matrix from imported DAG:

.. code-block:: python
   
   import bnlearn
   # Import DAG
   model = bnlearn.import_DAG('sachs', verbose=0)
   # adjacency matrix:
   model['adjmat']

   # print
   print(model['adjmat'])

Reading the table from left to right, we see that gene Erk is connected to Akt in a directed manner. 
This indicates that Erk influences gene Ark but not the otherway arround because gene Akt does not show a edge with Erk. In this example form, there may be a connection at the "...".

.. table::

  +------+-----+------+------+------+------+-----+-----+------+------+------+------+
  |      |  Erk|   Akt|   PKA|   Mek|   Jnk| ... |  Raf|   P38|  PIP3|  PIP2|  Plcg|
  +------+-----+------+------+------+------+-----+-----+------+------+------+------+
  |Erk   |False| True | False| False| False| ... |False| False| False| False| False|
  +------+-----+------+------+------+------+-----+-----+------+------+------+------+
  |Akt   |False| False| False| False| False| ... |False| False| False| False| False|
  +------+-----+------+------+------+------+-----+-----+------+------+------+------+
  |PKA   |True | True | False| True | True | ... |True | True | False| False| False|
  +------+-----+------+------+------+------+-----+-----+------+------+------+------+
  |Mek   |True | False| False| False| False| ... |False| False| False| False| False|
  +------+-----+------+------+------+------+-----+-----+------+------+------+------+
  |Jnk   |False| False| False| False| False| ... |False| False| False| False| False|
  +------+-----+------+------+------+------+-----+-----+------+------+------+------+
  |PKC   |False| False| True | True | True | ... |True | True | False| False| False|
  +------+-----+------+------+------+------+-----+-----+------+------+------+------+
  |Raf   |False| False| False| True | False| ... |False| False| False| False| False|
  +------+-----+------+------+------+------+-----+-----+------+------+------+------+
  |P38   |False| False| False| False| False| ... |False| False| False| False| False|
  +------+-----+------+------+------+------+-----+-----+------+------+------+------+
  |PIP3  |False| False| False| False| False| ... |False| False| False| True | False|
  +------+-----+------+------+------+------+-----+-----+------+------+------+------+
  |PIP2  |False| False| False| False| False| ... |False| False| False| False| False|
  +------+-----+------+------+------+------+-----+-----+------+------+------+------+
  |Plcg  |False| False| False| False| False| ... |False| False| True | True | False|
  +------+-----+------+------+------+------+-----+-----+------+------+------+------+


**Structure learning**

Extracting adjacency matrix after structure learning:

.. code-block:: python
   
   # Load dataframe
   df = bnlearn.import_example()
   # Learn structure
   model = bnlearn.structure_learning.fit(df)
   # adjacency matrix:
   model['adjmat']

   # print
   print(model['adjmat'])


Reading the table from left to right we see that Cloudy is connected to Sprinkler and also to Rain in a directed manner.
Sprinkler is connect to Wet_grass.
Rain is connected to Wet_grass.
Wet_grass is connected to nothing.


.. table::
  
  +-----------+--------+-----------+-------+-----------+
  |           | Cloudy | Sprinkler | Rain  | Wet_Grass |
  +===========+========+===========+=======+===========+
  | Cloudy    | False  | True      | True  | False     |
  +-----------+--------+-----------+-------+-----------+
  | Sprinkler | False  | False     | False | True      |
  +-----------+--------+-----------+-------+-----------+
  | Rain      | False  | False     | False | True      |
  +-----------+--------+-----------+-------+-----------+
  | Wet_Grass | False  | False     | False | False     |
  +-----------+--------+-----------+-------+-----------+



**Parameter learning**

Extracting adjacency matrix after Parameter learning:

.. code-block:: python
   
   # Load dataframe
   df = bnlearn.import_example()
   # Import DAG
   DAG = bnlearn.import_DAG('sprinkler', CPD=False)
   # Learn parameters
   model = bnlearn.parameter_learning.fit(DAG, df)
   # adjacency matrix:
   model['adjmat']

   # print
   print(model['adjmat'])

.. table::
  
  +-----------+--------+-----------+-------+-----------+
  |           | Cloudy | Sprinkler | Rain  | Wet_Grass |
  +===========+========+===========+=======+===========+
  | Cloudy    | False  | True      | True  | False     |
  +-----------+--------+-----------+-------+-----------+
  | Sprinkler | False  | False     | False | True      |
  +-----------+--------+-----------+-------+-----------+
  | Rain      | False  | False     | False | True      |
  +-----------+--------+-----------+-------+-----------+
  | Wet_Grass | False  | False     | False | False     |
  +-----------+--------+-----------+-------+-----------+
