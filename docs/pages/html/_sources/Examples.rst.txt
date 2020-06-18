Examples
=================

``bnlearn`` contains several examples within the library that can be used to practice with the functionalities of :func:`bnlearn.structure_learning`, :func:`bnlearn.parameter_learning` and :func:`bnlearn.inference`.


Example with DataFrames
'''''''''''''''''''''''

In ``bnlearn``, there is one example dataset that can be imported; the **sprinkler** dataset. Note this dataset is readily one-hot, without missing values, and as such does not require any further pre-processing steps. The DAG example models (see Example DAG section) can however be converted from the model to a dataframe.


.. code-block:: python

   # Import dataset
   df = bnlearn.import_example()
   # Structure learning
   model = bnlearn.structure_learning.fit(df)
   # Plot
   G = bnlearn.plot(model)


Example with DAG
'''''''''''''''''

``bnlearn`` contains several example Directed Acyclic Graphs:
	* 'sprinkler' (default)
	* 'alarm'
	* 'andes'
	* 'asia'
	* 'pathfinder'
	* 'sachs'
	* 'miserables'

Each DAG can be loaded using the :func:`bnlearn.bnlearn.import_DAG` function. With the :func:`bnlearn.bnlearn.sampling` function a ``DataFrame`` can be created for *n* samples.
The sprinkler DAG is a special case because it is not loaded from a *bif* file but created manually. Therefore, the **sprinkler** model can be generated with(out) a CPD by: ``CPD=False``.


.. code-block:: python
   
   # Import dataset
   DAG = bnlearn.import_DAG('sachs', CPD=True)
   # plot the keys of the DAG
   DAG.keys()
   # dict_keys(['model', 'adjmat'])

   # The model contains the BayesianModel with the CPDs.
   # The adjmat contains the adjacency matrix with the relationships between the nodes.

   # plot ground truth
   G = bnlearn.plot(DAG)

   # Sampling
   df = bnlearn.sampling(DAG, n=1000)




Import from BIF
'''''''''''''''''''

Each Bayesian DAG model that is loaded with :func:`bnlearn.bnlearn.import_DAG` is derived from a *bif* file. The *bif* file is a common format for Bayesian networks that can be used for the exchange of knowledge and experimental results in the community. More information can be found (here)[http://www.cs.washington.edu/dm/vfml/appendixes/bif.htm].

.. code-block:: python
   
   # Import dataset
   DAG = bnlearn.import_DAG('filepath/to/model.bif')



Start with RAW data
'''''''''''''''''''

Lets demonstrate by example how to process your own dataset containing mixed variables. I will demonstrate this by the titanic case. This dataset contains both continues as well as categorical variables and can easily imported using :func:`bnlearn.bnlearn.import_example`.
With the function :func:`bnlearn.bnlearn.df2onehot` it can help to convert the mixed dataset towards a one-hot matrix. The settings are adjustable, but by default the unique non-zero values must be above 80% per variable, and the minimal number of samples must be at least 10 per variable.


.. code-block:: python

   # Load titanic dataset containing mixed variables
   df_raw = bnlearn.import_example(data='titanic')
   # Pre-processing of the input dataset
   dfhot, dfnum = bnlearn.df2onehot(df_raw)
   # Structure learning
   DAG = bnlearn.structure_learning.fit(dfnum)
   # Plot
   G = bnlearn.plot(DAG)

.. _fig-titanic:

.. figure:: ../figs/fig_titanic.png


From this point we can learn the parameters using the DAG and input dataframe.

.. code-block:: python

   # Parameter learning
   model = bnlearn.parameter_learning.fit(DAG, df)

Finally, we can start making inferences. Note that the variable and evidence names should exactly match the input data (case sensitive).

.. code-block:: python

   # Print CPDs
   bnlearn.print_CPD(model)
   # Make inference
   q = bnlearn.inference.fit(model, variables=['Survived'], evidence={'Sex':0, 'Pclass':1})
   
   print(q.values)
   print(q.variables)
   print(q._str())
   

.. table::

     +-------------+-----------------+
     | Survived    |   phi(Survived) |
     +=============+=================+
     | Survived(0) |          0.3312 |
     +-------------+-----------------+
     | Survived(1) |          0.6688 |
     +-------------+-----------------+



Create a Bayesian Network, learn its parameters from data and perform the inference
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Lets make an example were we have data with many measurements, and we have expert information of the relations between nodes. 
Our goal is to create DAG on the expert knowledge and learn the CPDs. To showcase this, I will use the sprinkler example.

Import example dataset of the sprinkler dataset.

.. code-block:: python

    df = bnlearn.import_example('sprinkler')
    print(tabulate(df.head(), tablefmt="grid", headers="keys"))

.. table::

     +----+----------+-------------+--------+-------------+
     |    |   Cloudy |   Sprinkler |   Rain |   Wet_Grass |
     +====+==========+=============+========+=============+
     |  0 |        0 |           0 |      0 |           0 |
     +----+----------+-------------+--------+-------------+
     |  1 |        1 |           0 |      1 |           1 |
     +----+----------+-------------+--------+-------------+
     |  2 |        0 |           1 |      0 |           1 |
     +----+----------+-------------+--------+-------------+
     |  3 |        1 |           1 |      1 |           1 |
     +----+----------+-------------+--------+-------------+
     |  4 |        1 |           1 |      1 |           1 |
     +----+----------+-------------+--------+-------------+
     | .. |      ... |         ... |    ... |         ... |
     +----+----------+-------------+--------+-------------+


Define the network structure. This can be based on expert knowledge.

.. code-block:: python

    edges = [('Cloudy', 'Sprinkler'),
             ('Cloudy', 'Rain'),
             ('Sprinkler', 'Wet_Grass'),
             ('Rain', 'Wet_Grass')]

Make the actual Bayesian DAG

.. code-block:: python
    
    DAG = bnlearn.make_DAG(edges)
    # [BNLEARN] Bayesian DAG created.

    # Print the CPDs
    bnlearn.print_CPD(DAG)
    # [BNLEARN.print_CPD] No CPDs to print. Use bnlearn.plot(DAG) to make a plot.

Plot the DAG

.. code-block:: python

    bnlearn.plot(DAG)


.. _fig-DAG-sprinkler:

.. figure:: ../figs/DAG_sprinkler.png

Parameter learning on the user-defined DAG and input data using maximumlikelihood.

.. code-block:: python
    
    DAG = bnlearn.parameter_learning.fit(DAG, df, methodtype='maximumlikelihood')


Lets print the learned CPDs:

.. code-block:: python

    bnlearn.print_CPD(DAG)
    
    # [BNLEARN.print_CPD] Independencies:
    # (Cloudy _|_ Wet_Grass | Rain, Sprinkler)
    # (Sprinkler _|_ Rain | Cloudy)
    # (Rain _|_ Sprinkler | Cloudy)
    # (Wet_Grass _|_ Cloudy | Rain, Sprinkler)
    # [BNLEARN.print_CPD] Nodes: ['Cloudy', 'Sprinkler', 'Rain', 'Wet_Grass']
    # [BNLEARN.print_CPD] Edges: [('Cloudy', 'Sprinkler'), ('Cloudy', 'Rain'), ('Sprinkler', 'Wet_Grass'), ('Rain', 'Wet_Grass')]

CPD of Cloudy:
    +-----------+-------+
    | Cloudy(0) | 0.494 |
    +-----------+-------+
    | Cloudy(1) | 0.506 |
    +-----------+-------+

CPD of Sprinkler:
    +--------------+--------------------+--------------------+
    | Cloudy       | Cloudy(0)          | Cloudy(1)          |
    +--------------+--------------------+--------------------+
    | Sprinkler(0) | 0.4807692307692308 | 0.7075098814229249 |
    +--------------+--------------------+--------------------+
    | Sprinkler(1) | 0.5192307692307693 | 0.2924901185770751 |
    +--------------+--------------------+--------------------+

CPD of Rain:
    +---------+--------------------+---------------------+
    | Cloudy  | Cloudy(0)          | Cloudy(1)           |
    +---------+--------------------+---------------------+
    | Rain(0) | 0.6518218623481782 | 0.33695652173913043 |
    +---------+--------------------+---------------------+
    | Rain(1) | 0.3481781376518219 | 0.6630434782608695  |
    +---------+--------------------+---------------------+

CPD of Wet_Grass:
    +--------------+--------------------+---------------------+---------------------+---------------------+
    | Rain         | Rain(0)            | Rain(0)             | Rain(1)             | Rain(1)             |
    +--------------+--------------------+---------------------+---------------------+---------------------+
    | Sprinkler    | Sprinkler(0)       | Sprinkler(1)        | Sprinkler(0)        | Sprinkler(1)        |
    +--------------+--------------------+---------------------+---------------------+---------------------+
    | Wet_Grass(0) | 0.7553816046966731 | 0.33755274261603374 | 0.25588235294117645 | 0.37910447761194027 |
    +--------------+--------------------+---------------------+---------------------+---------------------+
    | Wet_Grass(1) | 0.2446183953033268 | 0.6624472573839663  | 0.7441176470588236  | 0.6208955223880597  |
    +--------------+--------------------+---------------------+---------------------+---------------------+

Lets make an inference:

.. code-block:: python
    
    q1 = bnlearn.inference.fit(DAG, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})

    +--------------+------------------+
    | Wet_Grass    |   phi(Wet_Grass) |
    +==============+==================+
    | Wet_Grass(0) |           0.2559 |
    +--------------+------------------+
    | Wet_Grass(1) |           0.7441 |
    +--------------+------------------+

Print the values:

.. code-block:: python
    
    print(q1.values)
    # array([0.25588235, 0.74411765])