.. _installation-quickstart:

Installation
============

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

.. list-table:: Comparison between Bayesian Libraries
   :header-rows: 1
   :widths: 22 18 16 16 16 16 16
   :align: center

   * - **Feature**
     - **bnlearn**
     - **pgmpy**
     - **CausalNex**
     - **DoWhy**
     - **pyAgrum**
     - **CausalImpact**

   * - **Primary Purpose**
     - ✅ End-to-end Bayesian network modeling
     - Probabilistic graphical models
     - Causal discovery
     - Causal inference
     - Bayesian & Markov networks
     - Bayesian time-series

   * - **Structure Learning**
     - ✅ HillClimb, PC, Chow-Liu, Naive Bayes, TAN
     - ✅ HillClimb, PC, Exhaustive, etc.
     - ✅ NOTEARS
     - ❌ Requires DAG
     - ✅ Greedy & score-based
     - ❌

   * - **Parameter Learning**
     - ✅ One-line CPD estimation
     - ✅ Explicit API
     - ✅ Supported
     - ➖ Limited
     - ✅ Fully supported
     - ➖ Implicit

   * - **Inference / Do-calculus**
     - ✅ Built-in inference & interventions
     - ✅ Variable elimination & sampling
     - ✅ CPD inference
     - ➖ Treatment inference
     - ✅ Multiple algorithms
     - ✅ Bayesian inference

   * - **Synthetic Data Generation**
     - ✅ Built-in sampling
     - ➖ Manual
     - ❌
     - ❌
     - ✅
     - ➖ Simulation

   * - **Input Data**
     - ✅ Discrete, continuous & hybrid
     - Discrete & continuous
     - Discrete
     - Numeric
     - Discrete & continuous
     - Continuous time-series

   * - **Causal DAG Learning**
     - ✅ Unsupervised
     - ✅ Supported
     - ✅ NOTEARS
     - ❌ Requires DAG
     - ✅ Supported
     - ❌

   * - **Best For**
     - ✅ Practical Bayesian workflows
     - Flexible research
     - Causal discovery
     - Treatment effects
     - Teaching & simulation
     - Intervention analysis

   * - **Ease of Use**
     - ✅ Very high
     - Moderate
     - Moderate
     - Low
     - Moderate
     - High

   * - **Learning Curve**
     - ✅ Easy
     - Moderate
     - Moderate
     - Steep
     - Moderate
     - Easy

   * - **Visualization**
     - ✅ Graphviz & interactive HTML
     - ➖ Basic
     - ✅ NetworkX
     - ➖ Limited
     - ✅ GUI
     - ➖ Time-series

   * - **Dependencies**
     - ✅ Lightweight
     - pgmpy ecosystem
     - pgmpy + sklearn
     - dowhy + econml
     - aGrUM
     - statsmodels

   * - **Causal Effect Estimation**
     - ✅ Intervention queries
     - ✅ Query based
     - ✅ CPD inspection
     - ✅ Core functionality
     - ✅ Intervention models
     - ✅ Core functionality

   * - **Preprocessing**
     - ✅ Automatic
     - Moderate
     - High
     - High
     - Moderate
     - Low

   * - **Typical Use Cases**
     - ✅ Discovery, inference & simulation
     - Custom Bayesian models
     - NOTEARS DAG learning
     - ATE estimation
     - Teaching
     - Policy evaluation

   * - **Advantages**
     - ✅ End-to-end pipelines, intuitive API
     - Flexible & modular
     - NOTEARS implementation
     - Causal framework
     - GUI
     - Bayesian time-series

   * - **Disadvantages**
     - ➖ Moderate-sized DAGs
     - Verbose API
     - Limited continuous support
     - No structure learning
     - Verbose syntax
     - Time-series only
     
     
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Feature**                   | **bnlearn**                                                              | **pgmpy**                                    | **CausalNex**                              | **DoWhy**                                    | **pyAgrum**                                | **CausalImpact**                           |
+===============================+==========================================================================+==============================================+============================================+==============================================+============================================+============================================+
| **Primary Purpose**           | [✓] End-to-end Bayesian network modeling, structure learning, parameter      | Low-level probabilistic graphical model      | Causal discovery and reasoning             | Causal inference with explicit               | General Bayesian/Markov model creation,    | Bayesian structural time-series for        |
|                               | learning, and inference                                                  | toolkit                                      | (discrete data)                            | treatment/outcome                            | inference, and visualization               | causal impact estimation                   |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Structure Learning**        | [✓] Multiple methods (Hillclimb, Constraint, Chow-Liu, Naive Bayes, TAN)| [✓] Multiple estimators (HillClimb, PC, etc. | [✓] NOTEARS algorithm (score-based)        | [ ] Requires DAG as input                    | [✓] Greedy or score-based approaches       | [ ] Not applicable                         |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Parameter Learning**        | [✓] One-line CPD estimation                                             | [✓] Requires explicit setup                  | [✓] Supported but requires preprocessing   | [-] Limited; works with provided DAG         | [✓] Fully supported                        | [-] Limited; implicit (time-series reg.)   |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Inference / Do-calculus**   | [✓] Built-in, intuitive syntax                                          | [✓] Variable elimination, sampling           | [✓] Supported (fitted CPDs)                | [-] Treatment-outcome inference only         | [✓] Multiple algorithms                    | [✓] Bayesian inference for intervention    |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Synthetic Data Generation** | [✓] Built-in (`bn.sampling`, `bn.forward_sample`)                       | [-] Possible manually                        | [ ] Not supported                          | [ ] Not supported                            | [✓] Supported                              | [-] Possible via simulation only           |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Input Data Support**        | [✓] Discrete, continuous, or mixed (string labels allowed)              | Discrete and continuous                      | Discrete only                              | Encoded numeric (binary treatment)           | Discrete or continuous                     | Continuous time series                     |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Causal DAG Learning**       | [✓] Fully unsupervised (independence tests, whitelist/blacklist edges)  | [✓] Requires explicit setup                  | [✓] NOTEARS-based (numeric)                | [ ] Requires predefined DAG                  | [✓] Supported                              | [ ] Not applicable                         |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Best For**                  | [✓] Practical causal modeling, exploratory discovery, education         | [✓] Developers needing flexibility           | [✓] Categorical causal discovery           | [✓] Econometric and policy impact estimation | [✓] Education and simulation               | [✓] Intervention analysis and time-series  |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Ease of Use**               | [✓] Very high (auto-handles preprocessing, pipelines)                   | Moderate to low (manual setup required)      | Moderate (needs encoding, tuning)          | Low (requires causal knowledge)              | Moderate                                   | High (simple interface)                    |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Learning Curve**            | [✓] Easy                                                                | Moderate                                     | Moderate                                   | Steep                                        | Moderate                                   | Easy                                       |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Visualization**             | [✓] Static + interactive (Plotly/Graphviz)                              | [-] Basic plotting                           | [✓] NetworkX-based graph                   | [-] Limited                                  | [✓] Interactive GUI                        | [-] Time-series plots only                 |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Dependencies**              | [✓] Lightweight (`bnlearn`, `pgmpy`, `pandas`, `numpy`)                 | `pgmpy`, `pandas`, `numpy`                   | `pgmpy`, `scikit-learn`, `networkx`        | `dowhy`, `econml`                            | `aGrUM` backend                            | `CausalImpact`, `statsmodels`, `pandas`    |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Causal Effect Estimation**  | [✓] Via inference/intervention                                          | [✓] Via queries                              | [✓] Via CPD inspection                     | [✓] Core functionality (ATE, backdoor, IV)   | [✓] Via intervention modeling              | [✓] Core functionality (timeseries effects)|
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Preprocessing Effort**      | [✓] Low (automatic handling, independence testing)                      | Medium (manual setup)                        | High (encoding, state definition)          | High (requires treatment/outcome encoding)   | Medium                                     | Low (requires clean time series)           |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Typical Use Cases**         | [✓] Exploratory causal discovery, inference, simulation, education      | Custom Bayesian model design                 | Discrete causal discovery, NOTEARS DAGs    | Estimating treatment effects (A/B tests)     | Teaching, simulation                       | Program evaluation, A/B test               |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Advantages**                | [✓] End-to-end pipeline, supports string labels, whitelist/blacklist,   | Modular and flexible                         | NOTEARS algorithm, explicit DAG control    | Rigorous causal framework, ATE estimation    | Strong visualization tools                 | Simple time-series causality with Bayesian |
|                               | independence tests, modern compatibility                                 |                                              |                                            |                                              |                                            | backbone                                   |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+
| **Disadvantages**             | [✓] Limited to moderate-scale DAGs                                      | Verbose code, no auto-pipeline               | No continuous data, high preprocessing     | No structure learning, binary treatments     | Verbose syntax                             | Only handles time-series, not generic DAGs |
+-------------------------------+--------------------------------------------------------------------------+----------------------------------------------+--------------------------------------------+----------------------------------------------+--------------------------------------------+--------------------------------------------+

.. include:: add_bottom.add
