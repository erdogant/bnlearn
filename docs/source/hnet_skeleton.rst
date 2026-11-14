Structure learning with an HNET skeleton
=========================================

Hillclimbsearch can add an edge between any two variables. With ``skeleton='hnet'`` the search is restricted to the pairs of variables for which an association is detected using `HNET <https://github.com/erdogant/hnet>`_.
The skeleton is undirected: it decides *which* variables can be connected, and the hill climbing search decides *which of those edges are added* and in *which direction*, based on the structure score (for example BIC).
This is a hybrid approach: a constraint-based skeleton combined with a score-based search.

HNET tests for associations between all combinations of categorical and continuous variables, and therefore the skeleton can be created for discrete, continuous and mixed data sets.
Note that HNET tests every pair of variables marginally, i.e., without conditioning on other variables. This is different from the constraint-based approach (``methodtype='pc'``) that uses conditional independence tests (``ci_test``).

.. code-block:: python

    import bnlearn as bn

    # Hillclimbsearch without a skeleton
    DAG = bn.structure_learning.fit(df, methodtype='hc')

    # Hillclimbsearch within a skeleton that is created with HNET
    DAG = bn.structure_learning.fit(df, methodtype='hc', skeleton='hnet')

.. note::
    The HNET approach requires the hnet library: ``pip install -U hnet``. It is an optional dependency of bnlearn.
    bnlearn uses the -log10(P) matrix ``simmatLogP_cat`` of hnet, which requires a version of hnet in which very small P-values are not capped or removed from this matrix.

How the skeleton is created
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The skeleton is created in the following steps, which are performed on a copy of the input data (``df`` is not modified):

1. **Typing.** hnet determines for each variable whether it is categorical or numeric (see the hnet parameters ``dtypes`` and ``perc_min_num``).
2. **Categorical-categorical and categorical-numeric pairs.** hnet performs the association learning with ``k=1``. Categorical variables are tested with the hypergeometric test, numeric variables with the Wilcoxon rank-sum test, and the P-values are corrected for multiple testing (``multtest``). The P-values of all categories of a variable pair are then combined with Fishers method, which results in one -log10(P) value per pair of variables (``simmatLogP_cat``).
3. **Numeric-numeric pairs.** HNET has no test for two numeric variables. These pairs are tested with the Pearson correlation test (``pearsonr`` in ``bnlearn.CITests``), without conditioning, and the P-values are corrected with the same ``multtest``. Pairs of numeric variables are therefore tested for linear associations only.
4. **Symmetrize.** The tests of hnet are directed (row is the response, column is the feature) and numeric-categorical pairs are tested in one direction only. A pair of variables is therefore associated if it is associated in *either* direction (the maximum -log10(P) of both directions).
5. **Threshold.** A pair of variables is part of the skeleton if its P-value is at most ``alpha``, i.e., -log10(P) >= -log10(alpha).
6. **Search space.** Each skeleton edge is added to the search space of the hill climbing in both directions. Variables without any association are isolated nodes in the DAG.

Because the pairs are tested marginally, the skeleton also contains pairs that are only indirectly associated (for example ``A - X2`` when ``A -> X1 -> X2``). These edges are in the search space but are, in general, not added by the hill climbing because they do not improve the score.

Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Discrete data**

.. code-block:: python

    import numpy as np
    import bnlearn as bn

    # Create a data set from the sprinkler example
    model = bn.fetch('sprinkler')
    np.random.seed(1)
    df = bn.sampling(model, n=1000)

    # Structure learning with a skeleton
    DAG = bn.structure_learning.fit(df, methodtype='hc', skeleton='hnet')

    # The undirected skeleton
    print(DAG['skeleton']['edges'])
    # [('Cloudy', 'Sprinkler'), ('Cloudy', 'Rain'), ('Cloudy', 'Wet_Grass'),
    #  ('Sprinkler', 'Rain'), ('Sprinkler', 'Wet_Grass'), ('Rain', 'Wet_Grass')]

    # The learned (directed) edges
    print(DAG['model_edges'])
    # [('Cloudy', 'Sprinkler'), ('Sprinkler', 'Wet_Grass'), ('Rain', 'Cloudy'), ('Rain', 'Wet_Grass')]

In this example all variables are associated, because the variables are connected through the network. The skeleton then contains all pairs, including ``Cloudy - Wet_Grass`` that is only indirectly associated, and the hill climbing decides which edges are kept.

**Mixed data**

.. code-block:: python

    import numpy as np
    import pandas as pd
    import bnlearn as bn

    rng = np.random.default_rng(1)
    n = 2000
    A = rng.integers(0, 2, n)                          # categorical
    B = (A ^ (rng.random(n) < 0.2)).astype(int)        # categorical, depends on A
    C = rng.integers(0, 3, n)                          # categorical, independent
    X1 = A * 2.0 + rng.normal(size=n)                  # numeric, depends on A
    X2 = 0.8 * X1 + rng.normal(size=n)                 # numeric, depends on X1
    X3 = rng.normal(size=n)                            # numeric, independent
    df = pd.DataFrame(dict(A=A, B=B, C=C, X1=X1, X2=X2, X3=X3))

    DAG = bn.structure_learning.fit(df, methodtype='hc', skeleton='hnet')

    print(DAG['skeleton']['edges'])
    # [('A', 'B'), ('A', 'X1'), ('A', 'X2'), ('B', 'X1'), ('B', 'X2'), ('X1', 'X2')]

    print(DAG['model_edges'])
    # [('A', 'B'), ('X1', 'X2'), ('X1', 'A')]

    # -log10(P) for each pair of variables
    print(DAG['skeleton']['logp'].round(1))
    # target      A      B    C     X1     X2   X3
    # source
    # A         0.0  146.6  0.0  452.9  255.8  0.0
    # B       146.6    0.0  0.0  121.0   68.0  0.0
    # C         0.0    0.0  0.0    0.0    0.0  0.0
    # X1      452.9  121.0  0.0    0.0  323.0  0.4
    # X2      255.8   68.0  0.0  323.0    0.0  0.1
    # X3        0.0    0.0  0.0    0.4    0.1  0.0

The variables ``C`` and ``X3`` are independent of the other variables and are isolated nodes. The pairs ``A - X2``, ``B - X1`` and ``B - X2`` are in the skeleton because of indirect associations, but are not part of the learned DAG.
For mixed data the score ``bic-cg`` is selected automatically.

.. note::
    The -log10(P) values are meant for thresholding. P-values below 1e-323 cannot be represented, and the values of numeric-numeric pairs are therefore capped at 323. The values of the other pairs are combined across categories and can be larger. Compare the values within the same type of pair only.

Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``skeleton``
     - ``None`` (default): the search is not restricted. ``'hnet'``: the search is restricted to the skeleton that is created with HNET. Only available for ``methodtype='hc'``, any other methodtype raises an error.
   * - ``params_hnet``
     - Dictionary with the settings of hnet. Default: ``{'alpha': 0.05, 'y_min': 10, 'multtest': 'holm'}``. Missing keys get the default value.
   * - ``params_hnet['alpha']``
     - Significance level. A pair of variables is part of the skeleton if its (corrected) P-value is at most alpha.
   * - ``params_hnet['y_min']``
     - Minimum number of samples in a group (category). Smaller groups are not tested by hnet.
   * - ``params_hnet['multtest']``
     - Multiple test correction, for example ``'holm'``, ``'bonferroni'``, ``'fdr_bh'`` or ``None``. It is applied to the tests of hnet and to the numeric-numeric tests.

Other keys in ``params_hnet`` are passed to hnet. ``k=1`` is always used, because combinations of features (``k>1``) are not variables of the DAG.

.. code-block:: python

    DAG = bn.structure_learning.fit(df, methodtype='hc', skeleton='hnet',
                                    params_hnet={'alpha': 0.01, 'multtest': 'fdr_bh', 'y_min': 10})

Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The output of ``bn.structure_learning.fit`` contains the key ``skeleton`` if a skeleton is used:

* ``DAG['skeleton']['edges']``: List with the undirected edges of the skeleton. The variables are in the order of the columns of the dataframe.
* ``DAG['skeleton']['adjmat']``: Symmetric boolean adjacency matrix of the skeleton. All variables are included.
* ``DAG['skeleton']['logp']``: Symmetric matrix with the -log10(P) of each pair of variables.
* ``DAG['config']['skeleton']`` and ``DAG['config']['params_hnet']``: The settings that are used.

Combining the skeleton with other options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``bw_list_method='nodes'``: Nodes in ``black_list`` or not in ``white_list`` are removed from the dataframe first. The skeleton is computed on the remaining variables.
* ``bw_list_method='edges'`` and ``black_list``: These edges are never added, also if they are part of the skeleton.
* ``bw_list_method='edges'`` and ``white_list``: The search space is the *intersection* of the skeleton and the white list. An edge in the white list that is not in the skeleton is never added.
* ``fixed_edges``: These edges are always part of the DAG, also if they are not part of the skeleton.
* If the skeleton does not contain any edge, no edge can be added and the DAG contains all variables without edges.

.. code-block:: python

    # Edge from X1 to A is not allowed
    DAG = bn.structure_learning.fit(df, methodtype='hc', skeleton='hnet', black_list=[('X1', 'A')], bw_list_method='edges')

    # Remove variable X3 before the skeleton is computed
    DAG = bn.structure_learning.fit(df, methodtype='hc', skeleton='hnet', black_list=['X3'], bw_list_method='nodes')

    # The edge from C to X3 is always included
    DAG = bn.structure_learning.fit(df, methodtype='hc', skeleton='hnet', fixed_edges=[('C', 'X3')])

Benchmark on random networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The script ``bnlearn/examples/benchmark_hnet_skeleton.py`` compares hillclimbsearch with and without the HNET skeleton on random discrete Bayesian networks with a known structure:

1. A random network is created with 8 to 12 variables. Each variable can have an edge from every earlier variable with probability 0.25 (at most 3 parents), and has 2 to 4 states. The columns of the conditional probability tables are drawn from a Dirichlet distribution with concentration 0.4, which gives dependencies that are clear but not deterministic.
2. Data is sampled from the network.
3. The structure is learned with ``methodtype='hc'`` and with ``methodtype='hc', skeleton='hnet'``.
4. The learned edges are compared with the true edges. *Precision* is the fraction of the learned edges that are in the true network, and *recall* is the fraction of the true edges that are found. These are computed for the undirected edges (the skeleton of the DAG) and for the directed edges (an edge with the wrong direction is an error).

The results are averaged over 20 random networks per sample size. The networks and data are determined by the seed (the first network uses seed 0). The hill climbing of pgmpy breaks ties between equally scoring edges in an order that depends on the hash seed of Python (with and without a skeleton), which can change the directed metrics by about 0.02 between runs. For exactly reproducible numbers, set ``PYTHONHASHSEED=0``. The runtime depends on the machine.

.. code-block:: bash

    PYTHONHASHSEED=0 python bnlearn/examples/benchmark_hnet_skeleton.py --n_networks 20 --n_samples 500 1000 10000 --seed 0

The tables below are the output of this script (default settings of ``params_hnet``), obtained with pgmpy 1.1.2 and a version of hnet with the corrected ``simmatLogP_cat``. The column *Change* is the difference between hc + hnet and hc (the runtime is the ratio).

.. list-table:: 500 samples (20 random networks)
   :header-rows: 1
   :widths: 40 20 20 20

   * - Metric
     - hc
     - hc + hnet
     - Change
   * - Undirected precision
     - 0.92
     - 0.99
     - +0.07
   * - Undirected recall
     - 0.72
     - 0.63
     - -0.09
   * - Directed precision
     - 0.56
     - 0.59
     - +0.03
   * - Directed recall
     - 0.43
     - 0.38
     - -0.05
   * - Runtime (seconds)
     - 0.27
     - 1.24
     - x4.5

.. list-table:: 1,000 samples (20 random networks)
   :header-rows: 1
   :widths: 40 20 20 20

   * - Metric
     - hc
     - hc + hnet
     - Change
   * - Undirected precision
     - 0.92
     - 0.98
     - +0.05
   * - Undirected recall
     - 0.80
     - 0.71
     - -0.09
   * - Directed precision
     - 0.65
     - 0.68
     - +0.03
   * - Directed recall
     - 0.57
     - 0.50
     - -0.06
   * - Runtime (seconds)
     - 0.30
     - 1.43
     - x4.8

.. list-table:: 10,000 samples (20 random networks)
   :header-rows: 1
   :widths: 40 20 20 20

   * - Metric
     - hc
     - hc + hnet
     - Change
   * - Undirected precision
     - 0.90
     - 0.97
     - +0.08
   * - Undirected recall
     - 0.92
     - 0.89
     - -0.02
   * - Directed precision
     - 0.65
     - 0.70
     - +0.05
   * - Directed recall
     - 0.67
     - 0.66
     - -0.01
   * - Runtime (seconds)
     - 0.38
     - 3.31
     - x8.7

**Results**

* **Fewer false edges.** The undirected precision is higher with the skeleton for all sample sizes (+0.05 to +0.08, which is about 3 to 4 standard errors of the mean change).
* **More missed edges.** The undirected recall is lower: -0.09 for 500 and 1,000 samples, and -0.02 for 10,000 samples. The recall of the skeleton itself is 0.66, 0.75 and 0.90, and an edge that is not in the skeleton can not be found. The recall is not higher in any of the networks at 500 and 1,000 samples (it is equal or lower in all 20 networks). A true edge is missed if the pair shows no or only a weak marginal association, for example when the dependency is weak or cancels out marginally.
* **Directed edges.** The directed precision is on average +0.03, +0.03 and +0.05 higher, but the change varies from network to network (it is higher in 10, 10 and 10 of the 20 networks, and lower in 7, 5 and 4). This is a small effect compared to the variation between networks, and only for 10,000 samples is it more than about two standard errors. The directed recall is lower (-0.05, -0.06 and -0.01).
* **The skeleton is a superset.** The precision of the skeleton itself is 0.57 to 0.77, because of the indirect associations described above. The hill climbing removes most of these.
* **Runtime.** For networks of 8 to 12 variables, hc + hnet is 4.5 to 8.7 times slower, because the HNET skeleton has to be computed in addition to the hill climbing. Whether the skeleton saves time for much larger networks was not evaluated.

**Choosing the multiple test correction**

The default ``multtest='holm'`` is conservative. Using a less conservative correction increases the recall of the skeleton, at the cost of precision. The table shows this for 1,000 samples (``--multtest fdr_bh`` and ``--multtest none`` of the benchmark script).

.. list-table:: Multiple test correction in hnet, 1,000 samples (20 random networks)
   :header-rows: 1
   :widths: 34 14 18 18 16

   * - Metric
     - hc
     - hc + hnet, ``multtest='holm'`` (default)
     - hc + hnet, ``multtest='fdr_bh'``
     - hc + hnet, ``multtest=None``
   * - Undirected precision
     - 0.92
     - 0.98
     - 0.97
     - 0.97
   * - Undirected recall
     - 0.80
     - 0.71
     - 0.77
     - 0.78
   * - Directed precision
     - 0.65
     - 0.68
     - 0.69
     - 0.68
   * - Directed recall
     - 0.57
     - 0.50
     - 0.56
     - 0.55
   * - Skeleton recall
     - 
     - 0.75
     - 0.81
     - 0.84
   * - Skeleton precision
     - 
     - 0.74
     - 0.69
     - 0.64
   * - Runtime (seconds)
     - 0.30
     - 1.43
     - 1.32
     - 1.36

With ``multtest='fdr_bh'`` the loss in undirected recall is -0.03 instead of -0.09, while the undirected precision remains higher than without the skeleton.

When to use it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* The skeleton is useful if false edges are more costly than missed edges: in this benchmark it gives fewer false edges and, for the default settings, more missed edges.
* If recall is most important, the hill climbing without a skeleton is faster and has the higher recall in this benchmark.
* The benchmark uses discrete networks with 8 to 12 variables. The results for continuous and mixed data, and for larger networks, may differ.

Limitations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Pairs are tested marginally. Dependencies that only appear after conditioning on other variables (for example an exclusive-or relation) are not detected and the edge can not be found.
* Numeric-numeric pairs are tested for linear associations (Pearson).
* Categories with fewer than ``y_min`` samples are not tested by hnet.
* hnet and the numeric-numeric tests are computed for all pairs of variables, which becomes slower with many variables.

References
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* HNET: https://github.com/erdogant/hnet
* HNET article: https://arxiv.org/abs/2005.04679



.. include:: add_bottom.add
