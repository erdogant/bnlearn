Use Cases
=================

In this section I will describe some use cases using ``bnlearn``. The differences with the examples section is that we will start with a possible goal. Perhaps you can use it, or will inspire you for something new.


Make inferences when you have data and know-how
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Suppose you work in the medical field and you have records of hundreds or even thousands patients treatment regarding shortness-of-breath (dyspnoea). In this context you may readily expect/know few associatons from literature and/or experience, like smoking is related to lung cancer. The data I will use is a small synthetic data set from Lauritzen and Spiegelhalter (1988) about lung diseases (tuberculosis, lung cancer or bronchitis) and visits to Asia. The exact motivation is described below.

Description
^^^^^^^^^^^^
Motivation of Lauritzen and Spiegelhalter (1988)
	"*Shortness-of-breath (dyspnoea) may be due to tuberculosis, lung cancer or bronchitis, or none of them, or more than one of them. A recent visit to Asia increases the chances of tuberculosis, while smoking is known to be a risk factor for both lung cancer and bronchitis. The results of a single chest X-ray do not discriminate between lung cancer and tuberculosis, as neither does the presence or absence of dyspnoea.*"

Source
	*Lauritzen S, Spiegelhalter D (1988). Local Computation with Probabilities on Graphical Structures and their Application to Expert Systems (with discussion). Journal of the Royal Statistical Society*

Aim: Make inferences about shortness-of-breath (dyspnoea) when:
	1. You have measured data
	2. You have know-how/expert knowledge

Import data
^^^^^^^^^^^^
The first step is to import the data. In my case I will load the data, which is readily a **structured** dataset. If you have **unstructured** data, you can use the ``df2onehot`` functionality :func:`bnlearn.bnlearn.df2onehot`. The *examples section* also contains examples how to import raw data followed by (basic) structering approaches (see the *Titanic* example).

.. code-block:: python

    # Load dataset with 10000 samples
    df = bnlearn.import_example('asia', n=10000)
    # Print to screen
    print(df)

+----+---------+---------+--------+--------+-------+----------+--------+--------+
|    |   smoke |   bronc |   lung |   asia |   tub |   either |   dysp |   xray |
+====+=========+=========+========+========+=======+==========+========+========+
|  0 |       0 |       1 |      1 |      1 |     1 |        1 |      0 |      1 |
+----+---------+---------+--------+--------+-------+----------+--------+--------+
|  1 |       1 |       1 |      1 |      1 |     1 |        1 |      1 |      0 |
+----+---------+---------+--------+--------+-------+----------+--------+--------+
|  2 |       1 |       0 |      1 |      0 |     1 |        0 |      1 |      1 |
+----+---------+---------+--------+--------+-------+----------+--------+--------+
|... |     ... |     ... |    ... |    ... |   ... |      ... |    ... |    ... |
+----+---------+---------+--------+--------+-------+----------+--------+--------+
|9999|       0 |       1 |      1 |      1 |     1 |        1 |      0 |      1 |
+----+---------+---------+--------+--------+-------+----------+--------+--------+

The asia data set contains only yes/no, true/false or 1/0 values. ``bnlearn`` can also handle multiple catagories if thats your case. The data generated contains 10.000 samples (aka  the patients). Is this a appropriate number? Well, it also depends on the number of variables you are including in the model. The ground-truth of the example data sets are available too, so you can play arround with various number of samples and determine when you can re-construct the entire DAG.
In case of the **sprinkler** data set, 1000 samples is sufficient because there are only 4 variables, each with discrete states (yes/no). Some other data sets (such as **alarm**) are way more complicated and 1000 samples would not be sufficient.


Define Directed Acyclic Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At this point, the data set is pre-processed and ready. The second step is to include expert knowledge in the form of a Directed Acyclic Graph (DAG). The DAG will describe the relationships between variables and with this we will learn the CPDs. 

Define the network structure. This can be based on expert knowledge or other hypothesis. I will keep this simple but bayesian modeling is especially interesting because you can make very complex DAGs. My knowledge about *dyspnoea* is limited to: smoking is related to lung cancer, smoking is related to bronchitis, and if you have lung or bronchitus you may need an xray examination. Note that the direction is very important. The first column is "from" or "source" and the second column "to" or "destination". Furthermore, this is a **very simple model** and **not** representative in real-life applications. But it is a easy example for demonstration purposes.

.. code-block:: python

    edges = [('smoke', 'lung'),
             ('smoke', 'bronc'),
             ('lung', 'xray'),
             ('bronc', 'xray')]


Lets plot the Bayesian DAG.

.. code-block:: python
    
    # Create the DAG from the edges
    DAG = bnlearn.make_DAG(edges)

    # Plot and make sure the arrows are correct.
    bnlearn.plot(DAG)

.. _fig_lung_simple_dag:

.. figure:: ../figs/lung_simple_dag.png


Compute Conditional Probability Distributions (CPDs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At this point we have the data set in our dataframe (df), and we have the DAG based on expert knowledge. The next step is to connect your brains (DAG) to the data set. We can do this with the function :func:`bnlearn.bnlearn.parameter_learning.fit`. See section **Parameter learning** to learn more about conditional probability distributions (CPDs) and how parameters can be learned. In general, it is the task to estimate the values of the CPDs in the DAG based on the input data set. How cool is that!


Parameter learning on the user-defined DAG and input data set.

.. code-block:: python

    # Check the current CPDs in the DAG. None are specified at this point so this will print empty.
    bnlearn.print_CPD(DAG)
    # [bnlearn] >No CPDs to print. Tip: use bnlearn.plot(DAG) to make a plot.

    # Learn its parameters from data and perform the inference. As input we have the DAG without CPDs, the output will be a DAG with learned CPDs.
    DAG = bnlearn.parameter_learning.fit(DAG, df, methodtype='bayes')

    # Print the CPDs
    bnlearn.print_CPD(DAG)


The learned Conditional Probability Distributions are depicted in the tables below. As an example, it can be seen that the probability of smoking, or not is the more-or-less the same. Thus the probability that a patient does not smoke: P(smoke=0)=0.49 whereas the probability of a patient smoking, P(smoke=1)=0.5. Slightly more complicated are the patients that smoke and have lung-cancer which is basically a intersection. Logically, the more edges towards a node in combination with multiple catagories, the more complicated it becomes. Luckily we have ``bnlearn`` to do the heavy lifting!

+----------+----------+
| smoke(0) | 0.495273 |
+----------+----------+
| smoke(1) | 0.504727 |
+----------+----------+

CPD of lung:

+---------+---------------------+---------------------+
| smoke   | smoke(0)            | smoke(1)            |
+---------+---------------------+---------------------+
| lung(0) | 0.13913362701908957 | 0.05457492795389049 |
+---------+---------------------+---------------------+
| lung(1) | 0.8608663729809104  | 0.9454250720461095  |
+---------+---------------------+---------------------+

CPD of bronc:

+----------+--------------------+--------------------+
| smoke    | smoke(0)           | smoke(1)           |
+----------+--------------------+--------------------+
| bronc(0) | 0.5936123348017621 | 0.3114193083573487 |
+----------+--------------------+--------------------+
| bronc(1) | 0.4063876651982379 | 0.6885806916426513 |
+----------+--------------------+--------------------+

CPD of xray:

+---------+---------------------+---------------------+--------------------+---------------------+
| bronc   | bronc(0)            | bronc(0)            | bronc(1)           | bronc(1)            |
+---------+---------------------+---------------------+--------------------+---------------------+
| lung    | lung(0)             | lung(1)             | lung(0)            | lung(1)             |
+---------+---------------------+---------------------+--------------------+---------------------+
| xray(0) | 0.7651245551601423  | 0.08089070665757782 | 0.7334669338677354 | 0.08396533044420368 |
+---------+---------------------+---------------------+--------------------+---------------------+
| xray(1) | 0.23487544483985764 | 0.9191092933424222  | 0.2665330661322645 | 0.9160346695557963  |
+---------+---------------------+---------------------+--------------------+---------------------+


Make inferences
^^^^^^^^^^^^^^^^^^^

When you are at this part, you combined your expert knowledge with a data set, and now we can make inferences. Thus basically ask questions to the model.


**Question 1**

What is the probability of lung-cancer, given that we know that patient does smoke?
The model returns that the probability of lung-cancer or lung(1) is 0.94 when the patient does smoke. Or P(lung=1 | smoke=1)=0.94.

.. code-block:: python
    
    q1 = bnlearn.inference.fit(DAG, variables=['lung'], evidence={'smoke':1})

    # Finding Elimination Order: : 100% 2/2 [00:00<00:00, 401.14it/s]
    # Eliminating: bronc: 100%| 2/2 [00:00<00:00, 200.50it/s][bnlearn] >Variable Elimination..

+---------+-------------+
| lung    |   phi(lung) |
+=========+=============+
| lung(0) |      0.0546 |
+---------+-------------+
| lung(1) |      0.9454 |
+---------+-------------+


**Question 2**

What is the probability of bronchitis, given that we know that patient does smoke?
The model returns that the probability of bronchitis or bronc(1) is 0.68 when the patient does smoke. Or P(bronc=1 | smoke=1)=0.68.


.. code-block:: python
    
    q2 = bnlearn.inference.fit(DAG, variables=['bronc'], evidence={'smoke':1})

    # Finding Elimination Order: : 100% 2/2 [00:00<00:00, 286.31it/s]
    # Eliminating: lung: 100% 2/2 [00:00<00:00, 143.26it/s][bnlearn] >Variable Elimination..

+----------+--------------+
| bronc    |   phi(bronc) |
+==========+==============+
| bronc(0) |       0.3114 |
+----------+--------------+
| bronc(1) |       0.6886 |
+----------+--------------+


**Question 3**

Lets add more information to our inference. What is the probability of lung-cancer, given that we know that patient does smoke and also has bronchitis? 

.. code-block:: python
    
    q3 = bnlearn.inference.fit(DAG, variables=['lung'], evidence={'smoke':1, 'bronc':1})

    # Finding Elimination Order: : 100%  1/1 [00:00<00:00, 334.31it/s]
    # Eliminating: xray: 100%  1/1 [00:00<00:00, 338.47it/s][bnlearn] >Variable Elimination..

+---------+-------------+
| lung    |   phi(lung) |
+=========+=============+
| lung(0) |      0.0546 |
+---------+-------------+
| lung(1) |      0.9454 |
+---------+-------------+



**Question 4**

Lets specify the question even more. What is the probability of lung-cancer or bronchitis, given that we know that patient does smoke but did not had xray? 

.. code-block:: python
    
    q4 = bnlearn.inference.fit(DAG, variables=['bronc','lung'], evidence={'smoke':1, 'xray':0})

+---------+----------+-------------------+
| lung    | bronc    |   phi(lung,bronc) |
+=========+==========+===================+
| lung(0) | bronc(0) |            0.1092 |
+---------+----------+-------------------+
| lung(0) | bronc(1) |            0.2315 |
+---------+----------+-------------------+
| lung(1) | bronc(0) |            0.2001 |
+---------+----------+-------------------+
| lung(1) | bronc(1) |            0.4592 |
+---------+----------+-------------------+

The highest probability for the patient under these condition is that lung-cancer is true and bronchitus is true too (P=0.45). Note that, if you put xray=1, then the probability becomes even higher (P=0.67).


Determine causalities when you have data
'''''''''''''''''''''''''''''''''''''''''
Comming soon.


Make inference when you have data
'''''''''''''''''''''''''''''''''''''''''
Comming soon.
