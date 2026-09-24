Inference
=========

The basic concept of variable elimination is same as doing marginalization over Joint Distribution.
But variable elimination avoids computing the Joint Distribution by doing marginalization over much smaller factors.
So basically if we want to eliminate **X** from our distribution, then we compute
the product of all the factors involving **X** and marginalize over them,
thus allowing us to work on much smaller factors.


Inference Algorithms
======================================

The main categories for inference algorithms:
  1. Exact Inference: These algorithms find the exact probability values for our queries.
  2. Approximate Inference: These algorithms try to find approximate values by saving on computation.

**Two common Inference algorithms with variable Elimination**
  a. Clique Tree Belief Propagation
  b. Variable Elimination


Examples Inference
======================================


Example (1)
^^^^^^^^^^^^^^^^^^^

Lets load the Sprinkler data set and make some inferences.


What is the probability of *wet grass* given that it *Rains*, and the *sprinkler* is off and its *cloudy*: P(wet grass | rain=1, sprinkler=0, cloudy=1)?

.. code-block:: python

   # Import library
   import bnlearn as bn

   model = bn.import_DAG('sprinkler')
   q1 = bn.inference.fit(model, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})


The probability having wet grass is 0.9 and not-wet-gras is 0.1.

  +--------------+------------------+
  | Wet_Grass    |   phi(Wet_Grass) |
  +==============+==================+
  | Wet_Grass(0) |           0.1000 |
  +--------------+------------------+
  | Wet_Grass(1) |           0.9000 |
  +--------------+------------------+


Example (2)
^^^^^^^^^^^^^^^^^^^

What is the probability of wet grass given and Rain given that the *Sprinkler* is on?

.. code-block:: python

   q2 = bn.inference.fit(model, variables=['Wet_Grass','Rain'], evidence={'Sprinkler':1})


The highest probability is that in these condition, there is wet grass and no rain (P=0.63)

  +--------------+---------+-----------------------+
  | Wet_Grass    | Rain    |   phi(Wet_Grass,Rain) |
  +==============+=========+=======================+
  | Wet_Grass(0) | Rain(0) |                0.0700 |
  +--------------+---------+-----------------------+
  | Wet_Grass(0) | Rain(1) |                0.0030 |
  +--------------+---------+-----------------------+
  | Wet_Grass(1) | Rain(0) |                0.6300 |
  +--------------+---------+-----------------------+
  | Wet_Grass(1) | Rain(1) |                0.2970 |
  +--------------+---------+-----------------------+


Example (3)
^^^^^^^^^^^^^^^^^^^

Given our model, what is the probability on lung cancer given that the person is a smoker and xray is negative?
P(lung | smoker=1, xray=0)

.. code-block:: python

   # Import library
   import bnlearn as bn

   # Lets create the dataset
   model = bn.import_DAG('asia')

Lets make the inference:

.. code-block:: python

   q1 = bn.inference.fit(model, variables=['lung'], evidence={'xray':0, 'smoke':1})

  +---------+-------------+
  | lung    |   phi(lung) |
  +=========+=============+
  | lung(0) |      0.1423 |
  +---------+-------------+
  | lung(1) |      0.8577 |
  +---------+-------------+

  # Do operator
  q2 = bn.inference.fit(model, variables=['lung'], do={'xray':0, 'smoke':1})

  +----+--------+------+
  |    |   lung |    p |
  +====+========+======+
  |  0 |      0 | 0.01 |
  +----+--------+------+
  |  1 |      1 | 0.99 |
  +----+--------+------+


Do-calculus (Intervention)
======================================

Observational inference answers ``P(Y | X=x)`` (conditioning on evidence).
Intervention answers a different question: ``P(Y | do(X=x))`` — what happens
to **Y** if we *actively set* **X** to **x**.

In ``bnlearn``, Pearl's do-operator is available via the ``do`` argument of
``bn.inference.fit``. Internally the query runs Variable Elimination on the
**mutilated** network (incoming edges of the intervened nodes are cut), with
the intervened values fixed as evidence. That is exact and combines freely
with ordinary ``evidence``.

A variable cannot appear in both ``do`` and ``evidence``.


Example (4) — Observational vs interventional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On the sprinkler network, observing that the sprinkler is on is evidence that
it is probably not cloudy. Setting the sprinkler on by intervention cuts the
``Cloudy → Sprinkler`` edge, so the weather is unaffected. The two queries
therefore differ:

.. code-block:: python

   import bnlearn as bn

   model = bn.import_DAG('sprinkler')

   # Observational: P(Wet_Grass | Sprinkler=1) ≈ 0.927
   q_obs = bn.inference.fit(
       model,
       variables=['Wet_Grass'],
       evidence={'Sprinkler': 1},
   )

   # Interventional: P(Wet_Grass | do(Sprinkler=1)) ≈ 0.945
   q_do = bn.inference.fit(
       model,
       variables=['Wet_Grass'],
       do={'Sprinkler': 1},
   )


Example (5) — Combining do and evidence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Interventions and ordinary evidence can be used together:

.. code-block:: python

   # P(Wet_Grass | do(Sprinkler=1), Rain=1)
   q_mix = bn.inference.fit(
       model,
       variables=['Wet_Grass'],
       do={'Sprinkler': 1},
       evidence={'Rain': 1},
   )


Example (6) — Multiple interventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   q_multi = bn.inference.fit(
       model,
       variables=['Wet_Grass'],
       do={'Sprinkler': 1, 'Rain': 0},
   )


When ``to_df=True`` (the default), interventions are labeled ``do(X)=…`` in
``query.text``. ``do`` is the last parameter of ``fit``, so existing positional
calls remain compatible.


Continuous and hybrid inference
=========================================

``bn.inference.fit`` detects the fitted model type and routes accordingly:

* **Discrete** Bayesian networks → ``fit_discrete`` (Variable Elimination; original behaviour).
* **Linear-Gaussian** networks → ``fit_continuous`` (conditional means via predict / simulate).
* **Conditional-Gaussian** (mixed) networks → discrete queries via Variable Elimination and continuous queries via the fitted CG local regressions.

You can also call ``fit_discrete`` or ``fit_continuous`` directly. The do-operator works
for discrete, continuous, and CG interventions within the type combinations below.

.. code-block:: python

    import bnlearn as bn
    import numpy as np
    import pandas as pd
    bn.set_logger('info')   # or 'info', 'warning', 'error', 'trace', None

    rng = np.random.default_rng(5)
    n = 300
    x = rng.normal(size=n)
    y = 1.5 * x + rng.normal(scale=0.4, size=n)
    df = pd.DataFrame({'X': x, 'Y': y})

    model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic-g')
    model = bn.parameter_learning.fit(model, df, methodtype='linear-gaussian')

    # Conditional mean of Y given X
    q = bn.inference.fit(model, variables=['Y'], evidence={'X': 0.0})
    print(q.means)

    # Intervention do(X=1)
    q_do = bn.inference.fit(model, variables=['Y'], do={'X': 1.0})
    print(q_do.means)

Continuous results are returned as a ``ContinuousQueryResult`` with attributes
``.means``, ``.variances`` (when available), ``.df``, and ``.text``.

For Conditional-Gaussian models, continuous query variables produce means (and local
residual std) given discrete configurations and continuous parents in the evidence.
Discrete query variables produce probability tables from the discrete sub-model.

.. _cg-inference-limitation-inference:

CG inference limitation
^^^^^^^^^^^^^^^^^^^^^^^^^

The CG engine keeps discrete and continuous inference separate. There is **no bridge**
that evaluates continuous evidence and injects it into a discrete CPT.

* Continuous query + discrete/continuous evidence → conditional local Gaussian (correct).
* Discrete query + discrete evidence → Variable Elimination (correct).
* Discrete query + continuous evidence → **marginal** of the discrete node; continuous
  evidence is not applied.

Example: ``P(Machine failure | Torque = 40)`` in a CG model with binary failure and
continuous torque is **not** conditioned on torque. Use a discretized discrete BN when
the target is a failure-style event that must depend on sensor thresholds
(see :doc:`Discretizing` and :doc:`Continuous Data`).

See :doc:`Continuous Data` for more hybrid examples.




.. include:: add_bottom.add
