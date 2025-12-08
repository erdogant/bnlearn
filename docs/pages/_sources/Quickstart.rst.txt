Installation
============

.. _quickstart-installation:

Installing bnlearn is straightforward. 
It is recommended to create a new environment for the installation.

.. code-block:: console

   conda create -n env_bnlearn python=3.8
   conda activate env_bnlearn
   pip install bnlearn

Quick Examples
========================

Let's start by importing some data. We need a DAG and CPD (Conditional Probability Distribution).

.. code:: python

    import bnlearn as bn

    # Import example dataset
    df = bn.import_example()

    # Learn the structure from data
    model = bn.structure_learning.fit(df)

    # Perform independence tests
    model = bn.independence_test(model, df)

    # Visualize the network
    G = bn.plot(model)

Here's another example demonstrating a complete workflow:

.. code:: python

    import bnlearn as bn

    # Import a predefined DAG (Sprinkler network)
    model = bn.import_DAG('sprinkler')

    # Import example dataset
    df = bn.import_example()

    # Generate samples from the model
    df = bn.sampling(model)

    # Perform inference
    query = bn.inference.fit(model, variables=['Rain'], evidence={'Cloudy':1, 'Wet_Grass':1})
    print(query.df)

    # Learn structure from data
    model_sl = bn.structure_learning.fit(df)

    # Learn parameters
    model_pl = bn.parameter_learning.fit(model_sl, df)

    # Compare networks
    scores, adjmat = bn.compare_networks(model_sl, model)



Comparison between Bayesian Libraries
================================================

+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Feature**                   | **bnlearn**                                                              | **pgmpy**                                    | **CausalNex**                              | **DoWhy**                                    | **pyAgrum**                                | **CausalImpact**                           |
+===============================+==========================================================================+==============================================+============================================+==============================================+============================================+============================================+
| **Primary Purpose**           | End-to-end Bayesian network modeling, structure learning, parameter      | Low-level probabilistic graphical model      | Causal discovery and reasoning             | Causal inference with explicit               | General Bayesian/Markov model creation,    | Bayesian structural time-series for        |
|                               | learning, and inference                                                  | toolkit                                      | (discrete data)                            | treatment/outcome                            | inference, and visualization               | causal impact estimation                   |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Structure Learning**        | [x] Multiple methods (Hillclimb, Constraint, Chow-Liu, Naive Bayes, TAN) | [x] Multiple estimators (HillClimb, PC, etc. | [x] NOTEARS algorithm (score-based)        | [ ] Requires DAG as input                    | [x] Greedy or score-based approaches       | [ ] Not applicable                         |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Parameter Learning**        | [x] One-line CPD estimation                                              | [x] Requires explicit setup                  | [x] Supported but requires preprocessing   | [-] Limited; works with provided DAG         | [x] Fully supported                        | [-] Limited; implicit (time-series reg.)   |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Inference / Do-calculus**   | [x] Built-in, intuitive syntax                                           | [x] Variable elimination, sampling           | [x] Supported (fitted CPDs)                | [-] Treatment-outcome inference only         | [x] Multiple algorithms                    | [x] Bayesian inference for intervention    |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Synthetic Data Generation** | [x] Built-in (`bn.sampling`, `bn.forward_sample`)                        | [-] Possible manually                        | [ ] Not supported                          | [ ] Not supported                            | [x] Supported                              | [-] Possible via simulation only           |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Input Data Support**        | Discrete, continuous, or mixed (string labels allowed)                   | Discrete only                                | Discrete only                              | Encoded numeric (binary treatment)           | Discrete or continuous                     | Continuous time series                     |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Causal DAG Learning**       | [x] Fully unsupervised (independence tests, whitelist/blacklist edges)   | [x] Requires explicit setup                  | [x] NOTEARS-based (numeric)                | [ ] Requires predefined DAG                  | [x] Supported                              | [ ] Not applicable                         |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Best For**                  | [x] Practical causal modeling, exploratory discovery, education          | [x] Developers needing flexibility           | [x] Categorical causal discovery           | [x] Econometric and policy impact estimation | [x] Education and simulation               | [x] Intervention analysis and time-series  |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Ease of Use**               | Very high (auto-handles preprocessing, pipelines)                        | Moderate to low (manual setup required)      | Moderate (needs encoding, tuning)          | Low (requires causal knowledge)              | Moderate                                   | High (simple interface)                    |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Learning Curve**            | Easy                                                                     | Moderate                                     | Moderate                                   | Steep                                        | Moderate                                   | Easy                                       |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Visualization**             | [x] Static + interactive (Plotly/Graphviz)                               | [-] Basic plotting                           | [x] NetworkX-based graph                   | [-] Limited                                  | [x] Interactive GUI                        | [-] Time-series plots only                 |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Dependencies**              | Lightweight (`bnlearn`, `pgmpy`, `pandas`, `numpy`)                      | `pgmpy`, `pandas`, `numpy`                   | `pgmpy`, `scikit-learn`, `networkx`        | `dowhy`, `econml`                            | `aGrUM` backend                            | `CausalImpact`, `statsmodels`, `pandas`    |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Causal Effect Estimation**  | [x] Via inference/intervention                                           | [x] Via queries                              | [x] Via CPD inspection                     | [x] Core functionality (ATE, backdoor, IV)   | [x] Via intervention modeling              | [x] Core functionality (timeseries effects)|
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Preprocessing Effort**      | Low (automatic handling, independence testing)                           | Medium (manual setup)                        | High (encoding, state definition)          | High (requires treatment/outcome encoding)   | Medium                                     | Low (requires clean time series)           |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Typical Use Cases**         | Exploratory causal discovery, inference, simulation, education           | Custom Bayesian model design                 | Discrete causal discovery, NOTEARS DAGs    | Estimating treatment effects (A/B tests)     | Teaching, simulation                       | Program evaluation, A/B test               |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Advantages**                | End-to-end pipeline, supports string labels, whitelist/blacklist,        | Modular and flexible                         | NOTEARS algorithm, explicit DAG control    | Rigorous causal framework, ATE estimation    | Strong visualization tools                 | Simple time-series causality with Bayesian |
|                               | independence tests, modern compatibility                                 |                                              |                                            |                                              |                                            | backbone                                   |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Disadvantages**             | Limited to moderate-scale DAGs                                           | Verbose code, no auto-pipeline               | No continuous data, high preprocessing     | No structure learning, binary treatments     | Verbose syntax                             | Only handles time-series, not generic DAGs |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+

.. include:: add_bottom.add
