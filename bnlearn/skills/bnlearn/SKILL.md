---
name: bnlearn
description: Use this skill when working with bnlearn, a Python library for Bayesian modeling, probabilistic graphical models, causal discovery, parameter learning, structure learning, inference, do-why calculus (dowhy), sampling, and directed acyclic graphs (DAG).
---

# bnlearn

## Purpose

Use this skill when working with Bayesian Networks, probabilistic graphical
models, structure learning, parameter learning, inference, prediction,
sampling, and causal discovery using the Python `bnlearn` library.

The goal of this skill is to help an AI agent:

1. Understand the statistical problem.
2. Select an appropriate Bayesian Network approach.
3. Select an appropriate bnlearn algorithm, score, or test.
4. Prepare and validate the data.
5. Execute the analysis using the bnlearn API.
6. Validate the resulting network.
7. Interpret the results without making unsupported causal claims.

---

# 1. Core Concepts

A Bayesian Network (BN) is a directed acyclic graph (DAG) in which:

* Nodes represent random variables.
* Directed edges represent conditional dependencies.
* Each node has an associated conditional probability distribution (CPD).
* The graph encodes conditional independence relationships.

The main Bayesian Network workflow is:

```text
Data
  │
  ▼
Data inspection
  │
  ▼
Structure learning
  │
  ▼
Network structure
  │
  ▼
Parameter learning
  │
  ▼
Bayesian Network
  │
  ├── Inference
  ├── Prediction
  ├── Sampling
  ├── Intervention
  └── Visualization
```

Do not confuse:

```text
association
    ≠
conditional dependence
    ≠
directed edge
    ≠
causal relationship
```

A learned DAG should not automatically be interpreted as a causal graph.

---

# 2. When to Use bnlearn skill

Use bnlearn when the task involves:

* Learning dependencies between variables.
* Learning a Bayesian Network from observational data.
* Learning a DAG structure.
* Estimating Bayesian Network parameters.
* Performing probabilistic inference.
* Predicting variables using a Bayesian Network.
* Sampling from a Bayesian Network.
* Finding conditional dependencies.
* Testing conditional independence.
* Comparing Bayesian Network structures.
* Discovering potentially causal structures.
* Performing interventions.
* Visualizing Bayesian Networks.

Do not automatically use bnlearn for:

* Ordinary regression.
* Ordinary clustering.
* Dimensionality reduction.
* Classification when a Bayesian Network is not required.
* Time-series forecasting when temporal structure is the primary problem.
* Deep learning problems.

---


# 3. Identify the User's Task

Before selecting a bnlearn function, determine what the user is trying to accomplish.

| User goal                       | bnlearn workflow                |
| ------------------------------- | ------------------------------- |
| Discover relationships          | Structure learning              |
| Find a DAG                      | Structure learning              |
| Test conditional independence   | Independence testing            |
| Estimate probabilities          | Parameter learning              |
| Ask probability questions       | Inference                       |
| Predict a variable              | Prediction                      |
| Generate synthetic observations | Sampling                        |
| Compare networks                | Structure scoring               |
| Ask whether X causes Y          | Causal discovery / intervention |
| Visualize a network             | Network visualization           |

Do not select an algorithm before identifying the problem type.

---

# 4. Standard Workflow

For a new Bayesian Network problem, follow this workflow:

1. Identify the user's goal.
2. Select the appropriate method.
3. Consult the relevant reference documentation.
4. Prefer an existing example when one matches the task.
5. Adapt the example to the user's data.

Now we move to the data set:

1. Inspect the dataset.
2. Identify variable types.
3. Check missing values.
4. Check categorical cardinalities.
5. Inspect continuous-variable distributions.
6. Determine whether preprocessing is required.
7. Determine whether the data are discrete, continuous, or mixed.
8. Select a structure-learning strategy.
9. Select an appropriate score or independence test.
10. Apply domain constraints when available.
11. Learn the network structure.
12. Inspect the learned structure.
13. Validate the structure.
14. Learn the network parameters.
15. Validate the fitted model.
16. Perform inference, prediction, sampling, or intervention.
17. Interpret the results.
18. Clearly distinguish statistical dependency from causal interpretation.


---

# 5. Data Preparation

Before structure learning, inspect:

* Number of observations.
* Number of variables.
* Variable types.
* Missing values.
* Constant variables.
* Duplicate variables.
* Categorical cardinality.
* Continuous-variable distributions.
* Outliers.
* Highly correlated variables.
* Sample size relative to network complexity.

Do not silently modify the dataset.

If preprocessing is required, explain why it is being performed.

---

## 5.1 Discrete Variables

Discrete variables may represent:

* Categories.
* Binary states.
* Ordinal states.
* Discretized measurements.

Check whether categorical variables are represented consistently.

Avoid unnecessarily creating a very large number of states because this can make
parameter estimation unreliable.

**High cardinality:** If a categorical column has many rare levels (e.g. job
titles, product IDs, free-text codes), group rare levels into an `"Other"`
category (or a domain-sensible bin) *before* structure learning. Large
cardinality explodes CPD size and makes scores/CI tests unstable. See §18.10 D.

---

## 5.2 Continuous Variables

Do not automatically discretize continuous variables.

First determine whether a continuous Bayesian Network is appropriate.

Inspect:

* Distribution shape.
* Outliers.
* Transformations.
* Approximate Gaussianity.
* Sample size.
* Nonlinear relationships.
* Measurement scale.

When the assumptions are appropriate, prefer continuous structure learning
(`scoretype='bic-g'` / `'aic-g'` / `'loglik-g'`, or LiNGAM methods) rather
than discarding information through discretization.

**Full continuous pipeline** (structure → parameters → inference / sampling):

```python
model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic-g')
model = bn.parameter_learning.fit(model, df, methodtype='linear-gaussian')
q = bn.inference.fit(model, variables=['Y'], evidence={'X': 0.5})
df_s = bn.sampling(model, n=200, methodtype='linear-gaussian')
```

Discretize only when a discrete BN is explicitly required (`bn.discretize`).

See:

`references/continuous_models.md`

---

## 5.3 Mixed Variables

When variables contain a mixture of discrete and continuous values:

1. Identify the variable types (`utils.infer_data_type` / automatic detection).
2. Choose the model from the **query target**, not only from the data types
   (see CG inference limitation below).
3. Structure: `scoretype='bic-cg'` (or `'auto'`) for a Conditional Gaussian graph.
4. Parameters: `methodtype='cg'` (or `'auto'`).

```python
model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic-cg')
model = bn.parameter_learning.fit(model, df, methodtype='cg')
# Supported: continuous query given discrete (or continuous) evidence
q = bn.inference.fit(model, variables=['torque'], evidence={'fail': 1})
df_s = bn.sampling(model, n=200, methodtype='cg')
```

### CG inference limitation (important)

The CG engine is a **hybrid** that does not fully couple the two sides:

* Discrete nodes → Variable Elimination on the discrete sub-network (TabularCPDs).
* Continuous nodes → local Gaussian CPDs (`continuous_cpds`).
* There is **no bridge** that takes a continuous parent value, evaluates a
  Gaussian CPD, and propagates that result into a **discrete** target.

| Query | Evidence | Result |
| ----- | -------- | ------ |
| Continuous | Discrete and/or continuous | Conditional mean / local Gaussian (correct) |
| Discrete | Discrete only | CPT / VE (correct) |
| Discrete | Continuous | **Marginal** of the discrete node — **not** conditioned on the continuous value |

Example of what is **not** supported:  
`P(Machine failure | Torque [Nm] = 40)` in a CG model where failure is discrete
and torque is continuous. The continuous evidence is ignored for the discrete
query; you get the marginal failure distribution.

This is an **architectural constraint**, not a temporary bug. It is also an
argument for discretization when the decision target is discrete (e.g. binary
failure driven by operational thresholds):

* **CG** — right tool when you need continuous sensor dynamics and continuous
  queries (how torque / temperature propagate given discrete regimes).
* **Discrete BN** (after `bn.discretize`) — right tool when the target is a
  binary / categorical event and you must condition on sensor-like evidence.
* The **target of the analysis** decides the approach; discretization is not
  merely a weaker version of CG for failure-prediction questions.

Do not silently convert variables without explaining the consequences.

See:

`references/continuous_models.md` and `references/inference.md`

---

# 6. Structure Learning

Structure learning determines the dependency structure of the Bayesian Network.

**Exact call** (see §18 for full signature):

```python
model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
```

The main approaches are:

```text
Structure Learning
│
├── Score-based (methodtype='hc' | 'ex')
│   └── Search for a graph that optimizes a score
│
├── Constraint-based (methodtype='pc' | 'cs')
│   └── Infer structure using conditional independence tests
│
├── Tree / classification (methodtype='cl' | 'tan' | 'nb')
│
└── Causal discovery (methodtype='direct-lingam' | 'ica-lingam')
```

---

## 6.1 Score-Based Learning

Use score-based learning when the goal is to find a graph that optimizes a
statistical score.

Typical approaches:

* `methodtype='hc'` — Hill Climbing (default, recommended starting point)
* `methodtype='ex'` — Exhaustive Search (only for very small networks)

Typical scores:

* Discrete: `bic`, `aic`, `k2`, `bdeu`, `bds`
* Continuous (Gaussian): `bic-g`, `aic-g`, `loglik-g`
* Hybrid / Conditional Gaussian: `bic-cg`, `aic-cg`, `loglik-cg`
* `auto` — detect data type and select `bic` / `bic-g` / `bic-cg`

See:

`references/structure_learning.md`

and:

`references/scoring.md`

---

## 6.2 Constraint-Based Learning

Constraint-based methods infer graph structure from conditional independence
relationships.

```python
model = bn.structure_learning.fit(
    df,
    methodtype='pc',                 # or 'cs'
    params_pc={'ci_test': 'chi_square', 'alpha': 0.05},
)
```

Use this approach when conditional independence testing is central to the
analysis.

Consider:

* Variable type
* Independence test (`params_pc['ci_test']`)
* Significance level (`params_pc['alpha']`)
* Sample size
* Multiple testing
* Causal assumptions

See:

`references/structure_learning.md`

---

## 6.3 Tree, classification and causal methods

* Tree-structured: `methodtype='cl'` / `'chow-liu'` (requires `root_node`)
* Tree-Augmented Naive Bayes: `methodtype='tan'` (requires `root_node` + `class_node`)
* Naive Bayes: `methodtype='nb'` / `'naivebayes'`
* Causal discovery (continuous): `methodtype='direct-lingam'` or `'ica-lingam'`

See:

`references/structure_learning.md` and `references/causal_discovery.md`

---

# 7. Choosing a Structure-Learning Method

Use the following decision logic:

```text
Is the network structure already known?
│
├── Yes → bn.make_DAG / bn.import_DAG → parameter_learning.fit
│
└── No
    │
    ├── Continuous data + causal discovery goal?
    │      └── methodtype='direct-lingam' or 'ica-lingam'
    │
    ├── Conditional independence testing is central?
    │      └── methodtype='pc' (or 'cs')
    │
    ├── Tree / classification structure desired?
    │      ├── methodtype='cl' / 'chow-liu'   (need root_node)
    │      ├── methodtype='tan'              (need root_node + class_node)
    │      └── methodtype='nb' / 'naivebayes'
    │
    ├── Very small number of variables (≤ ~6) and exhaustive OK?
    │      └── methodtype='ex'
    │
    └── Default / general score-based search
           └── methodtype='hc'  + appropriate scoretype
```

Do not choose an algorithm solely because it is the default.

Also consider:

* Data type (discrete scores vs `*-g` Gaussian scores)
* Sample size vs. number of variables
* Expected graph complexity → use `max_indegree` or black/white lists
* Computational budget
* Causal assumptions
* Available domain knowledge (`black_list`, `white_list`, `fixed_edges`)

---

# 8. Structure Constraints

When domain knowledge is available, use it to constrain the search space.

```python
DAG = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic',
    black_list=[('A', 'B'), ('C', 'D')],   # forbidden edges
    white_list=[('X', 'Y')],               # allowed edges (or nodes)
    bw_list_method='edges',                # REQUIRED with black/white lists
    max_indegree=3,                        # max parents per node
    fixed_edges=[('Temp', 'Pressure')],    # edges that must be present
)
```

**Required:** if you pass `black_list` or `white_list`, you **must** also set
`bw_list_method` to `'edges'` or `'nodes'`. Omitting it raises an Exception.

**start_dag:** must include the **same variables as the dataset**. A partial
DAG that omits nodes present in `df` raises `ValueError`.

Possible constraints:

* `black_list` / `white_list` (+ required `bw_list_method`)
* `max_indegree`
* `fixed_edges`
* Starting structure via `start_dag` (full variable set)
* Temporal / domain ordering encoded as black/white lists

Constraints can substantially improve both computational efficiency and
interpretability.

Do not invent domain constraints.

Only apply constraints supplied by the user or justified by the problem domain.

---

# 9. Parameter Learning

Parameter learning estimates the probability distributions associated with a
known network structure.

```python
model = bn.parameter_learning.fit(DAG, df, methodtype='bayes')
# methodtype: 'bayes' (recommended when sparse) | 'ml' | 'maximumlikelihood'
```

The typical workflow is:

```text
Data
  │
  ▼
Learn or define DAG
  │
  ▼
Parameter learning
  │
  ▼
Fitted Bayesian Network
```

Do not use parameter learning as a substitute for structure learning when the
network structure is unknown. Always run it **after** `structure_learning.fit`
(or after `make_DAG` / `import_DAG`) before inference, prediction, or sampling.

See:

`references/parameter_learning.md`

---

# 10. Bayesian Inference

Use inference when the user wants to calculate probabilities conditioned on
observed evidence, or (with the `do` argument) interventional probabilities.

```python
query = bn.inference.fit(
    model,
    variables=['Y'],
    evidence={'X': 1},
    to_df=True,
)
# query.df has columns = variables + 'p'
```

Typical observational problem:

```text
Given:
    X = x

Calculate:
    P(Y | X = x)
```

Interventional problem (Pearl's do-operator):

```python
query = bn.inference.fit(
    model,
    variables=['Y'],
    do={'X': 1},
)
# P(Y | do(X = 1))
```

`do` and `evidence` may be combined. A variable cannot appear in both.

General workflow:

```text
Fitted Bayesian Network
        │
        ▼
Evidence and/or interventions (do)
        │
        ▼
Inference (Variable Elimination on the
possibly mutilated network)
        │
        ▼
Posterior / interventional distribution
```

Clearly distinguish:

* Prior probability
* Likelihood
* Posterior probability
* Marginal probability
* Conditional probability
* Interventional probability `P(Y | do(X=x))`

See:

`references/inference.md`

---

# 11. Prediction

Use prediction when the user wants to estimate an unknown variable from
observed variables.

Before prediction:

1. Verify that the target variable exists.
2. Verify that the model has been fitted.
3. Identify available evidence variables.
4. Determine whether the target is discrete or continuous.
5. Report uncertainty where available.

Do not describe prediction as causal intervention.

---

# 12. Sampling

Use sampling when the user wants to:

* Generate synthetic observations.
* Explore the model distribution.
* Perform simulation.
* Generate conditional samples.

```python
samples = bn.sampling(model, n=1000, methodtype='bayes')
# Conditional: bn.sampling(model, n=100, methodtype='bayes', evidence={'Rain': 1})
# Gibbs does NOT support evidence
```

Sampling should use the fitted Bayesian Network rather than independently
sampling each variable.

See:

`references/sampling.md`

---

# 13. Causal Discovery

Causal interpretation requires additional assumptions.

Do not automatically interpret:

```text
X → Y
```

as:

```text
X causes Y
```

Before making a causal claim, consider:

* Causal sufficiency.
* Hidden confounding.
* Measurement quality.
* Sampling assumptions.
* Temporal ordering.
* Faithfulness.
* Markov assumptions.
* Correct specification of the causal graph.
* Intervention semantics.

When assumptions are not established, describe the result as a dependency
structure.

See:

`references/causal_discovery.md`

---

# 14. Intervention (do-why calculus)

An intervention asks a different question from ordinary inference.

Observational question:

```text
P(Y | X = x)          # evidence={'X': x}
```

Interventional question:

```text
P(Y | do(X = x))      # do={'X': x}
```

bnlearn implements Pearl's do-operator via the `do` argument of
`bn.inference.fit`. Internally the query runs Variable Elimination on the
**mutilated** network (`model.do(...)`): incoming edges of the intervened
variables are cut, then the intervened values are fixed as evidence. That is
exact and works together with ordinary evidence, `elimination_order`, and
`joint`.

```python
# Import bnlearn library
import bnlearn as bn

# Example data
model = bn.import_DAG('sprinkler')

#  print(model)
#  {'model': <pgmpy.models.DiscreteBayesianNetwork.DiscreteBayesianNetwork at 0x2cf90df3410>,
#  'adjmat': target     Cloudy  Sprinkler   Rain  Wet_Grass
#  source
# Cloudy      False       True   True      False
#  Sprinkler   False      False  False       True
#  Rain        False      False  False       True
#  Wet_Grass   False      False  False      False}

# Observational
q_obs = bn.inference.fit(model, variables=['Wet_Grass'], evidence={'Sprinkler': 1})
# P(Wet_Grass=1 | Sprinkler=1) ≈ 0.927

# Interventional
q_do = bn.inference.fit(model, variables=['Wet_Grass'], do={'Sprinkler': 1})
# P(Wet_Grass=1 | do(Sprinkler=1)) ≈ 0.945

# Combined
q_mix = bn.inference.fit(
    model,
    variables=['Wet_Grass'],
    do={'Sprinkler': 1},
    evidence={'Rain': 1},
)
```

Rules:

* A variable cannot appear in both `do` and `evidence` (raises).
* Unknown node names in `do` raise.
* Interventions are labeled `do(X)=…` in `query.text`.
* `do` is the last parameter of `fit`, so existing positional calls stay compatible.

Do not replace an intervention with ordinary conditioning.

When the user asks:

> What happens if I force / set / change X to a particular value?

interpret this as an intervention problem and use `do=`.

Causal interpretation of the underlying DAG still requires the usual
assumptions (no hidden confounding, correct causal graph, etc.). See
`references/causal_discovery.md` and `references/inference.md`.

---

# 15. Model Scoring

Use structure scores to compare candidate Bayesian Network structures.

Common scores include:

* BIC.
* AIC.
* K2.
* BDeu.
* BDs.
* Gaussian scores for continuous data where supported.

When comparing scores:

* Use compatible data types.
* Consider model complexity.
* Consider the assumptions behind each score.
* Do not compare scores calculated under incompatible conditions without
  explaining the limitation.

See:

`references/scoring.md`

---

# 16. Model Validation

Do not assume that a learned network is correct merely because the algorithm
successfully completed.

Evaluate:

* Structural plausibility.
* Stability.
* Predictive performance where appropriate.
* Parameter estimates.
* Conditional independence relationships.
* Sensitivity to preprocessing.
* Sensitivity to algorithm settings.
* Sensitivity to sample variation.

Where possible, assess whether important edges are stable under resampling.

---

# 17. Visualization

Use visualization to inspect:

* Nodes.
* Directed edges.
* Network structure.
* Parent-child relationships.
* Markov blankets.
* Edge strengths where available.

Visualization is an exploratory and communication tool.

Do not infer statistical significance merely from visual appearance.

---

# 18. Exact API Cheat Sheet (bnlearn ≥ 0.14)

**Critical rule:** Use the exact parameter names below. Never invent aliases
(`method=` is wrong; the correct name is `methodtype=`). Never borrow
signatures from pgmpy, DoWhy, CausalNex, or other libraries.

All signatures below were verified against the installed bnlearn API.

---

## 18.1 Structure learning

```python
model = bn.structure_learning.fit(
    df,                          # pandas DataFrame (required)
    methodtype='hc',             # see table below
    scoretype='bic',             # see table below
    black_list=None,             # list of edges or nodes to forbid
    white_list=None,             # list of edges or nodes to allow
    bw_list_method=None,         # 'edges' | 'nodes' | None
    max_indegree=None,           # int, limit number of parents
    tabu_length=100,             # Hill-Climbing tabu list length
    epsilon=1e-4,
    max_iter=1e6,
    root_node=None,              # required for 'cl' / 'chow-liu'
    class_node=None,             # required for 'tan'
    fixed_edges=None,            # iterable of edges that must be present
    start_dag=None,              # optional starting DAG
    params_pc={'ci_test': 'chi_square', 'alpha': 0.05},
    params_lingam={'random_state': None, 'prior_knowledge': None,
                   'apply_prior_knowledge_softly': False, 'measure': 'pwling'},
    n_jobs=-1,
    verbose=3,
)
```

### Supported `methodtype` values

| methodtype (aliases)              | Family              | Typical data          | Notes |
|-----------------------------------|---------------------|-----------------------|-------|
| `hc` / `hillclimbsearch`          | Score-based         | Discrete / continuous | Default. Hill Climbing |
| `ex` / `exhaustivesearch`         | Score-based         | Discrete (tiny nets)  | Exhaustive search |
| `pc` / `cs` / `constraintsearch`  | Constraint-based    | Discrete / continuous | PC algorithm |
| `cl` / `chow-liu`                 | Tree                | Discrete              | Requires `root_node` |
| `tan`                             | Tree-Augmented NB   | Discrete              | Requires `root_node` + `class_node` |
| `nb` / `naivebayes`               | Naive Bayes         | Discrete              | Classification |
| `direct-lingam`                   | Causal (LiNGAM)     | Continuous / mixed    | Causal discovery |
| `ica-lingam`                      | Causal (LiNGAM)     | Continuous            | Causal discovery |

Prefer the short forms (`hc`, `ex`, `pc`, `cl`) in new code. The long aliases
are accepted by bnlearn and appear in older examples/blogs.

### Supported `scoretype` values

| scoretype   | Data type   | Notes |
|-------------|-------------|-------|
| `bic`       | Discrete    | Default, good general choice |
| `aic`       | Discrete    | Less penalization than BIC |
| `k2`        | Discrete    | Dirichlet prior |
| `bdeu`      | Discrete    | BDeu score |
| `bds`       | Discrete    | BDs score |
| `bic-g`     | Continuous  | Gaussian BIC |
| `aic-g`     | Continuous  | Gaussian AIC |
| `loglik-g`  | Continuous  | Gaussian log-likelihood |

**Never** pass a discrete score to continuous data (or vice-versa) without
explicit conversion/discretization.

### Return value (structure learning)

Common keys (always present for score-based methods):

```text
model            # pgmpy model
model_edges      # list of edges
adjmat           # adjacency matrix (DataFrame)
config           # configuration dict
structure_scores # scores of the learned structure
```

Constraint-based (`pc` / `cs`) additionally returns:

```text
undirected, undirected_edges, pdag, pdag_edges, dag, dag_edges
```

Do not assume attributes that are not in the returned dictionary.

---

## 18.2 Parameter learning

```python
model = bn.parameter_learning.fit(
    model,                 # output of structure_learning.fit or make_DAG / import_DAG
    df,                    # same DataFrame (or compatible)
    methodtype='bayes',    # 'bayes' | 'ml' | 'maximumlikelihood'
    scoretype='bdeu',      # used by some estimators
    smooth=None,
    n_jobs=-1,
    verbose=3,
)
```

- `methodtype='bayes'` → Bayesian parameter estimation (recommended when counts are low).
- `methodtype='ml'` / `'maximumlikelihood'` → Maximum-likelihood estimates.

After this call the model contains CPDs and can be used for inference,
prediction and sampling.

---

## 18.3 Inference (including do-why calculus)

```python
query = bn.inference.fit(
    model,                     # fitted model (after parameter learning) or DAG with CPDs
    variables=['Target'],      # variables to query (list)
    evidence={'A': 1, 'B': 0}, # observational evidence (state values)
    to_df=True,                # return a DataFrame-friendly object
    elimination_order='greedy',
    joint=True,
    groupby=None,
    plot=False,
    verbose=3,
    do=None,                   # interventional assignment, e.g. {'Sprinkler': 1}
)
```

- Evidence values must be valid states of the variables.
- For a pure marginal (no evidence), pass `evidence={}` — do **not** omit the
  argument or pass `None` (current bnlearn calls `evidence.keys()` and will
  raise `AttributeError`).
- When `to_df=True` the result has a `.df` attribute with columns = variables + `p`.
- Use `bn.query2df(query, variables=[...])` to reshape.
- **`do`**: Pearl's do-operator. Cuts incoming edges of the intervened nodes
  (mutilated network) and runs Variable Elimination with those values fixed.
  Combines freely with `evidence`. A variable must not appear in both `do` and
  `evidence`. Interventions appear as `do(X)=…` in `query.text`.

**Observational vs interventional (sprinkler)**

```python
model = bn.import_DAG('sprinkler')

# P(Wet_Grass | Sprinkler=1)  ≈ 0.927
bn.inference.fit(model, variables=['Wet_Grass'], evidence={'Sprinkler': 1})

# P(Wet_Grass | do(Sprinkler=1)) ≈ 0.945
bn.inference.fit(model, variables=['Wet_Grass'], do={'Sprinkler': 1})
```

**Pattern: inference on a hand-built DAG with placeholder CPDs**

```python
edges = [
    ('Cloudy', 'Sprinkler'),
    ('Cloudy', 'Rain'),
    ('Sprinkler', 'Wet_Grass'),
    ('Rain', 'Wet_Grass'),
]

# Generate placeholder CPDs (uniform by default)
CPD = bn.build_cpts_from_structure(edges, variable_card=2)

# Create DAG with those CPDs
DAG = bn.make_DAG(edges, CPD=CPD)

query = bn.inference.fit(
    DAG,
    variables=['Wet_Grass'],
    evidence={'Rain': 1, 'Sprinkler': 0, 'Cloudy': 1},
)
```

---

## 18.4 Sampling

```python
df_samples = bn.sampling(
    DAG,                       # fitted model or DAG dict
    n=1000,
    methodtype='bayes',        # 'bayes' | 'gibbs'
    evidence=None,             # optional conditioning dict
    verbose=0,
)
```

- `methodtype='gibbs'` does **not** support `evidence`.
- Evidence that has zero probability under the model raises `ValueError`.
- Evidence variable names and states are validated; unknown names raise `ValueError`.

---

## 18.5 Prediction

```python
yhat = bn.predict(
    model,           # fitted model
    df,              # DataFrame with evidence columns
    variables,       # target variable name(s) but must be in of df.columns
    to_df=True,
    method='max',    # 'max' (MAP) or probability
    verbose=3,
)
```

---

## 18.6 Building / loading a DAG by hand

```python
# From edge list (optionally with CPDs)
DAG = bn.make_DAG(
    edges,                     # list of (parent, child) tuples
    CPD=None,                  # optional list of TabularCPD
    methodtype='bayes',        # 'bayes' | 'naivebayes' | 'DBN' | 'markov'
    checkmodel=True,
    verbose=3,
)

# Generate uniform placeholder CPDs for a structure
CPDs = bn.build_cpts_from_structure(edges, variable_card=2)
DAG = bn.make_DAG(edges, CPD=CPDs)

# Single-node CPT (optionally with a rulebook)
cpt = bn.generate_cpt('Y', parents=['X'], variable_card=2)

# From built-in example network
DAG = bn.import_DAG('sprinkler')   # also: 'asia', 'alarm', ...
df  = bn.import_example('sprinkler')
```

---

## 18.7 Independence tests, discretization, one-hot

```python
# Conditional independence test on a learned model
# prune=True removes edges that fail the significance test (common post-processing step)
model = bn.independence_test(
    model, df,
    test='chi_square',   # also: 'pearsonr', 'g_sq', 'log_likelihood', ...
    alpha=0.05,
    prune=True,
)

# Discretize continuous columns given a structure
# continuous_columns must be a list of column names that are continuous
df_disc = bn.discretize(
    df,
    edges,
    continuous_columns=['Age', 'Income'],
    max_iterations=8,
)

# One-hot / numeric encoding helper
dfhot, dfnum = bn.df2onehot(df)
```

---

## 18.8 Visualization, CPD inspection, scores, I/O

```python
bn.plot(model)                    # static matplotlib plot
bn.plot_graphviz(model)           # graphviz
bn.print_CPD(model)               # print all CPDs
bn.structure_scores(model, df, scoring_method=['bic', 'k2', 'bdeu', 'bds'])
bn.save(model, filepath='bn')
model = bn.load(filepath='bn')
```

---

## 18.9 Minimal end-to-end pattern (discrete)

```python
import bnlearn as bn

df = bn.import_example('sprinkler')

# 1. Structure
DAG = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')

# 2. Parameters
model = bn.parameter_learning.fit(DAG, df, methodtype='bayes')

# 3. Inference
q = bn.inference.fit(model, variables=['Wet_Grass'],
                     evidence={'Rain': 1, 'Sprinkler': 0})

# 4. Sampling
samples = bn.sampling(model, n=1000)
```

Always keep structure learning and parameter learning as two distinct steps
unless the user already supplies a fully specified DAG with CPDs.

---

## 18.10 Canonical workflows from practice (blog patterns)

These patterns appear repeatedly in real bnlearn tutorials and should be
preferred when they match the user’s goal.

### A. Expert-knowledge-first (no structure learning)

When the user (or domain expert) already knows the causal / dependency
structure:

```python
edges = [
    ('Cloudy', 'Sprinkler'),
    ('Cloudy', 'Rain'),
    ('Sprinkler', 'Wet_Grass'),
    ('Rain', 'Wet_Grass'),
]
DAG = bn.make_DAG(edges)
model = bn.parameter_learning.fit(DAG, df, methodtype='bayes')
bn.print_CPD(model)
q = bn.inference.fit(model, variables=['Wet_Grass'],
                     evidence={'Rain': 1, 'Sprinkler': 0})
```

Do **not** run `structure_learning.fit` when the structure is already given.

### B. Data-driven structure → prune → parameters → “chat with the data”

```python
model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
# Optional but recommended: drop edges that fail a CI test
model = bn.independence_test(model, df, test='chi_square', alpha=0.05, prune=True)
model = bn.parameter_learning.fit(model, df, methodtype='bayes')

# Ask probability questions (the “chat with your dataset” pattern)
q1 = bn.inference.fit(model, variables=['salary'], evidence={'education': 'Doctorate'})
q2 = bn.inference.fit(model, variables=['salary'], evidence={'education': 'HS-grad'})
```

### C. Compare several structure learners / scores

When the user asks which method or score is better, or wants robustness:

```python
models = {
    'hc_bic':  bn.structure_learning.fit(df, methodtype='hc', scoretype='bic'),
    'hc_k2':   bn.structure_learning.fit(df, methodtype='hc', scoretype='k2'),
    'hc_bdeu': bn.structure_learning.fit(df, methodtype='hc', scoretype='bdeu'),
    'cs_bic':  bn.structure_learning.fit(df, methodtype='cs', scoretype='bic'),
    'cl':      bn.structure_learning.fit(df, methodtype='cl', root_node='Cloudy'),
}
for name, m in models.items():
    print(name, m['model_edges'])
```

### D. High-cardinality / messy categoricals before structure learning

Structure learning degrades when categorical variables have dozens of rare
levels. From practice:

1. Inspect `df[col].value_counts()`.
2. Group rare levels into an `"Other"` (or domain-sensible) category.
3. Drop or bin continuous columns that should not enter a discrete BN
   (or switch to continuous scores / LiNGAM instead of discretizing blindly).
4. Only then call `structure_learning.fit`.

```python
# Example: collapse rare job titles
counts = df['job_title'].value_counts()
rare = counts[counts < 30].index
df['job_title'] = df['job_title'].where(~df['job_title'].isin(rare), 'Other')
```

### E. Prediction vs intervention (prescriptive framing)

- **Prediction / inference:** `P(Y | X = x)` → `bn.inference.fit(..., evidence={'X': x})` or `bn.predict`
- **Intervention:** `P(Y | do(X = x))` → `bn.inference.fit(..., do={'X': x})`

If the user asks “what happens if we *force* / *set* / *change* X?”, use the
`do=` argument. The underlying DAG still needs a causal interpretation
(no unmeasured confounding, correct structure, etc.); state those assumptions
(see §13–14 and `references/causal_discovery.md`).

---

# 19. Common Mistakes (agent failure modes)

These are the mistakes that appear most frequently when an agent uses bnlearn.

---

## Mistake: Wrong parameter name (`method` instead of `methodtype`)

Incorrect:

```python
bn.structure_learning.fit(df, method='hc', scoretype='bic')
```

Correct:

```python
bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
```

The same rule applies to `parameter_learning.fit`, `make_DAG`, `sampling`, etc.
Always use `methodtype=`.

---

## Mistake: Treating every edge as causal

Incorrect:

```text
X → Y
therefore X causes Y
```

Correct:

```text
X → Y
is evidence of a learned dependency structure.
```

Causal interpretation requires additional assumptions (see §13 and
`references/causal_discovery.md`).

---

## Mistake: Discretizing continuous variables automatically

Do not discretize continuous variables simply because discrete Bayesian
Networks are easier to use.

First consider whether a continuous/Gaussian model is appropriate
(`scoretype='bic-g'`, `methodtype='hc'` or LiNGAM methods).

Only call `bn.discretize(...)` when the user explicitly requests
discretization or when a continuous method is demonstrably unsuitable.

---

## Mistake: Confusing inference and intervention

These are different operations:

```text
P(Y | X = x)          # evidence={'X': x}
```

versus:

```text
P(Y | do(X = x))      # do={'X': x}
```

Use the `do` argument of `bn.inference.fit` for interventions. Do not answer
an intervention question with ordinary `evidence`. A variable cannot appear in
both `do` and `evidence`.

---

## Mistake: Ignoring variable types

Do not select a score or independence test without considering whether the
variables are:

* Discrete.
* Continuous.
* Mixed.

Discrete scores (`bic`, `k2`, `bdeu`, …) on continuous data, or Gaussian
scores (`bic-g`, …) on discrete data, produce incorrect or failing results.

---

## Mistake: Calling parameter learning before structure learning

When the structure is unknown:

```text
structure_learning.fit  →  parameter_learning.fit  →  inference / sampling
```

Do not call `parameter_learning.fit` on a raw DataFrame; it expects a model
dict that already contains a DAG.

---

## Mistake: Inventing function names or arguments

Never invent:

* `bn.learn_structure`, `bn.fit_bn`, `bn.build_network`, …
* Arguments such as `algorithm=`, `scoring=`, `ci_test=` at the top level
  of `structure_learning.fit` (CI-test settings belong inside `params_pc`).

Stick to the functions listed in §18 and the reference files.

---

## Mistake: Passing invalid evidence

* Variable names that do not exist in the model → `ValueError`.
* State values outside the variable’s cardinality → `ValueError`.
* Evidence with zero probability under the model → `ValueError` (sampling).
* `methodtype='gibbs'` together with `evidence=` → `ValueError`.

Always validate evidence against the model’s nodes and CPDs.

---

## Mistake: Ignoring sample size

A network with many parameters can be poorly estimated even when structure
learning succeeds computationally.

Prefer simpler structures (`max_indegree`, stronger scores, domain constraints)
when the available data cannot support a highly complex model.

---

## Mistake: Treating a single learned DAG as ground truth

Structure learning is sensitive to:

* Data
* Algorithm (`methodtype`)
* Score / test (`scoretype`, `params_pc`)
* Hyperparameters
* Sampling variation
* Model assumptions

Consider stability (bootstrap / resampling) and uncertainty when the user
asks for confidence in the structure.

---

## Mistake: Using the wrong return-value keys

After `structure_learning.fit`:

* Always present: `model`, `model_edges`, `adjmat`, `config`, `structure_scores`
* Only for `pc`/`cs`: also `undirected`, `pdag`, `dag`, …

Do not write `model.edges` or `model.cpds` on the raw return dict; use
`model['model_edges']` and, after parameter learning, `model['model'].get_cpds()`.

---

# 20. Troubleshooting

## Unexpected network structure

Check:

* Variable types.
* Missing values.
* Score selection.
* Independence test / significance level.
* Search constraints (`black_list`, `white_list`, `max_indegree`).
* Maximum parent count.
* Sample size.
* Outliers / transformations.
* Algorithm assumptions.
* High-cardinality categoricals (group rare levels).

**Too many edges after structure learning:** run

```python
model = bn.independence_test(model, df, test='chi_square', alpha=0.05, prune=True)
```

before parameter learning. This is a standard post-processing step in practice.

---

## Poor continuous-variable results

Check:

* Gaussian assumptions.
* Outliers.
* Transformations.
* Nonlinear relationships.
* Sample size.
* Score selection.
* Whether discretization is appropriate.

---

## Too many edges

Consider:

* Stronger model-selection criteria.
* Search constraints.
* Maximum parent count.
* Domain knowledge.
* Larger sample size.
* Stability analysis.

Do not arbitrarily delete edges merely to make the graph look simpler.

---

## Too few edges

Check:

* Independence-test threshold.
* Score selection.
* Sample size.
* Variable preprocessing.
* Search constraints.
* Whether the underlying relationships are detectable with the available
  data.

---

# 21. Reproducibility

When applicable:

* Set random seeds.
* Record algorithm settings.
* Record score/test settings.
* Record preprocessing steps.
* Record bnlearn and dependency versions.
* Record structural constraints.
* Record the input variables.
* Save the learned network.

A Bayesian Network result should be reproducible from the documented
configuration and input data.

---

# 22. Performance

For large datasets or many variables:

1. Reduce unnecessary variables.
2. Remove constant variables.
3. Use domain constraints where justified.
4. Limit graph complexity where appropriate.
5. Select computationally appropriate algorithms.
6. Avoid unnecessarily expensive exhaustive searches.
7. Validate the result after optimization.

Do not sacrifice statistical validity solely for speed.

---

# 23. Recommended Response Pattern

When solving a bnlearn problem, structure the response as:

### 1. Problem identification

Explain what Bayesian Network task is being solved.

### 2. Data assessment

Identify:

* Variable types.
* Sample size.
* Missing values.
* Relevant assumptions.

### 3. Method selection

Explain:

* Structure-learning algorithm.
* Score or independence test.
* Constraints.
* Continuous/discrete strategy.

### 4. Implementation

Provide the relevant bnlearn code.

### 5. Validation

Explain how to evaluate the learned model.

### 6. Interpretation

Explain what the network means and explicitly state limitations.

---

# 24. Reference Documentation

Use the following references for detailed guidance:

* `references/structure_learning.md`
* `references/continuous_models.md`
* `references/scoring.md`
* `references/parameter_learning.md`
* `references/inference.md`
* `references/sampling.md`
* `references/causal_discovery.md`
* `references/troubleshooting.md`

Examples:

* `examples/discrete_bn.py`
* `examples/continuous_bn.py`
* `examples/hill_climbing.py`
* `examples/pc_algorithm.py`
* `examples/inference.py`
* `examples/sampling.py`

---

# 25. API Accuracy (non-negotiable rules)

The bnlearn API can evolve. When implementing a solution:

1. Prefer the API exposed by the **installed** bnlearn version (currently documented for ≥ 0.14).
2. Do **not** assume that APIs from other Bayesian Network libraries are interchangeable.
3. Use only functions that exist in bnlearn. Do not invent new function names.
4. Do not invent function arguments.
5. The structure-learning keyword is **`methodtype`**, never `method`, `algorithm`, or `algo`.
6. Verify estimator, score, test, and inference arguments before using them.
7. Prefer the current bnlearn documentation, the unit tests under `tests/`, and the
   reference files over outdated blog snippets or memory of other libraries.
8. If an API differs between versions, explicitly state the version-specific behavior.

**Never** silently substitute APIs from pgmpy, pomegranate, DoWhy, CausalNex,
Lingam (standalone), or any other library. bnlearn already wraps the needed
functionality; call the bnlearn surface.

When in doubt, open the corresponding file in `references/` or the matching
example in `examples/` before writing code.

---

# 26. Final Decision Checklist

Before returning a bnlearn solution, verify:

* [ ] The Bayesian Network task has been identified (structure / parameters / inference / sampling / causal).
* [ ] Variable types (discrete / continuous / mixed) have been determined.
* [ ] Missing values have been considered.
* [ ] Continuous variables have **not** been discretized unnecessarily.
* [ ] `methodtype` (not `method`) is used for structure / parameter / sampling calls.
* [ ] The chosen `methodtype` and `scoretype` are compatible with the data type.
* [ ] Domain constraints (`black_list` / `white_list` / `max_indegree` / `fixed_edges`) have been considered when knowledge exists.
* [ ] Sample size versus model complexity has been considered.
* [ ] Structure learning and parameter learning are performed as **separate** steps when the DAG is unknown.
* [ ] Return-value keys (`model`, `model_edges`, `adjmat`, …) are used correctly.
* [ ] Evidence passed to inference / sampling uses valid variable names and states.
* [ ] Inference and intervention (`do`) have not been confused.
* [ ] Causal claims are supported by appropriate assumptions (or are explicitly withheld).
* [ ] Only real bnlearn functions and arguments are used (no invented APIs).
* [ ] The result is reproducible where appropriate (seeds, settings, versions).
* [ ] Limitations are clearly communicated to the user.
