# Structure Learning

> **Quick API (verified bnlearn ≥ 0.14)**
>
> ```python
> model = bn.structure_learning.fit(
>     df,
>     methodtype='hc',          # NEVER use method= ; correct name is methodtype
>     scoretype='bic',          # discrete: bic|aic|k2|bdeu|bds
>                               # continuous: bic-g|aic-g|loglik-g
>                               # hybrid CG: bic-cg|aic-cg|loglik-cg
>                               # auto: bic / bic-g / bic-cg from data types
>     black_list=None,
>     white_list=None,
>     bw_list_method=None,      # 'edges' | 'nodes' | None
>     max_indegree=None,
>     root_node=None,           # required for cl / chow-liu
>     class_node=None,          # required for tan
>     fixed_edges=None,
>     params_pc={'ci_test': 'chi_square', 'alpha': 0.05},
>     params_lingam={...},
>     verbose=3,
> )
> ```
>
> **Return keys (score-based):** `model`, `model_edges`, `adjmat`, `config`, `structure_scores`  
> **Extra keys (pc/cs):** `undirected`, `undirected_edges`, `pdag`, `pdag_edges`, `dag`, `dag_edges`
>
> After structure learning always call `bn.parameter_learning.fit(...)` before inference/sampling.

---

Structure learning estimates a Directed Acyclic Graph (DAG) from a dataset.
The goal is to identify dependencies between variables without requiring the
network structure to be specified in advance.

In `bnlearn`, structure learning is exposed through:

```python
bn.structure_learning.fit()
```

The implementation supports several families of structure-learning methods:

* Score-based learning (`hc`, `ex`)
* Constraint-based learning (`pc`, `cs`)
* Tree / classification (`cl`/`chow-liu`, `tan`, `nb`/`naivebayes`)
* LiNGAM-based causal discovery (`direct-lingam`, `ica-lingam`)

The appropriate method depends primarily on:

* Variable type
* Number of variables
* Number of observations
* Whether conditional independence testing is desired
* Whether causal assumptions are appropriate
* Whether domain constraints are available

---

# 1. Basic Workflow

The standard workflow is:

```text
Dataset
   │
   ▼
Inspect variables
   │
   ├── Discrete
   │
   ├── Continuous
   │
   └── Mixed
   │
   ▼
Select structure-learning method
   │
   ▼
Select score / independence test
   │
   ▼
Apply domain constraints
   │
   ▼
Learn DAG
   │
   ▼
Inspect / validate DAG
```

Structure learning only estimates the network structure. It does not estimate
the CPDs/parameters of the Bayesian Network.

After structure learning, use parameter learning when a fitted probabilistic
model is required.

---

# 2. Main API

The main entry point is:

```python
import bnlearn as bn

model = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic'
)
```

The input `df` must be a pandas DataFrame.

The function returns a dictionary containing:

```text
model
model_edges
adjmat
config
structure_scores
```

where:

* `model` is the learned pgmpy model.
* `model_edges` contains the learned edges.
* `adjmat` is the adjacency matrix.
* `config` contains the structure-learning configuration.
* `structure_scores` contains structure scores.

The implementation validates the DataFrame and supported method and score
types before starting the search.

---

# 3. Selecting a Method

`methodtype` determines how the DAG is learned.

Supported methods are:

| `methodtype` (aliases)                | Method                     | Data / use case                   |
| ------------------------------------- | -------------------------- | --------------------------------- |
| `hc` / `hillclimbsearch`              | Hill Climbing              | General score-based learning      |
| `ex` / `exhaustivesearch`             | Exhaustive Search          | Very small networks               |
| `pc` / `cs` / `constraintsearch`      | PC / Constraint Search     | Constraint-based learning         |
| `cl` / `chow-liu`                     | Chow-Liu                   | Tree-structured networks          |
| `tan`                                 | Tree-Augmented Naive Bayes | Classification                    |
| `nb` / `naivebayes`                   | Naive Bayes                | Classification                    |
| `direct-lingam`                       | DirectLiNGAM               | Continuous/mixed causal discovery |
| `ica-lingam`                          | ICA-LiNGAM                 | Causal discovery                  |

These method names (including long aliases) are explicitly validated by `bnlearn`.
Prefer the short forms in new code.

---

# 4. Method Selection Decision Tree

Use the following decision process.

```text
What is the objective?
│
├── General DAG structure
│   │
│   ├── Score-based search
│   │      ├── hc → Hill Climbing
│   │      └── ex → Exhaustive Search
│   │
│   ├── Conditional independence
│   │      └── pc → PC
│   │
│   └── Hybrid / specialized
│
├── Tree structure required
│   ├── chow-liu
│   └── tan
│
├── Classification structure
│   └── naivebayes
│
└── Continuous / mixed causal discovery
    ├── direct-lingam
    └── ica-lingam
```

For a general-purpose Bayesian Network, `hc` is the default and should usually
be the first score-based method to consider.

For very small networks where computational cost is acceptable, exhaustive
search can be used.

For conditional-independence-driven discovery, use `pc`.

For continuous or mixed data where LiNGAM assumptions are appropriate, consider
`direct-lingam` or `ica-lingam`.

---

# 5. Score-Based Structure Learning

Score-based learning treats structure learning as an optimization problem.

A scoring function assigns a numerical score to a candidate DAG, while a
search algorithm explores possible DAGs.

Conceptually:

```text
Candidate DAG
     │
     ▼
Score(DAG | data)
     │
     ▼
Search for better DAG
     │
     ▼
Best-scoring DAG found
```

The implementation supports:

* Hill Climbing.
* Exhaustive Search.
* TreeSearch-based methods.

Supported scores include:

```text
bic
k2
bdeu
bds
aic
loglik-g
aic-g
bic-g
```

---

# 6. Hill Climbing

## Method

```python
methodtype='hc'
```

or:

```python
methodtype='hillclimbsearch'
```

Hill Climbing performs a heuristic local search over DAG structures.

The search starts from a starting DAG, which by default is a disconnected
network, and repeatedly applies single-edge modifications that improve the
score.

The search terminates when a local maximum is reached.

Basic example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic'
)
```

### When to use

Use Hill Climbing when:

* The number of variables is too large for exhaustive search.
* A practical score-based solution is required.
* The user wants to optimize a Bayesian Network score.
* The graph does not need to be restricted to a tree structure.

Hill Climbing is generally the preferred starting point for general-purpose
structure learning.

---

# 7. Hill Climbing Parameters

Hill Climbing supports additional search controls.

## `start_dag`

Specifies the starting DAG.

```python
model = bn.structure_learning.fit(
    df,
    methodtype='hc',
    start_dag=start_model
)
```

By default, a disconnected DAG is used.

A dictionary containing a `model` can also be supplied; the implementation
extracts the model from it. The starting object must be a Bayesian Network;
otherwise it is ignored.

---

## `max_indegree`

Limits the maximum number of parents a node may have.

```python
model = bn.structure_learning.fit(
    df,
    methodtype='hc',
    max_indegree=3
)
```

This can be useful for:

* Controlling model complexity.
* Reducing the search space.
* Preventing unrealistic highly connected nodes.
* Improving computational performance.

`max_indegree` is supported only for Hill Climbing.

---

## `tabu_length`

Controls the tabu-search memory used during Hill Climbing.

Default:

```python
tabu_length=100
```

Use this when additional control over the local search behavior is required.

---

## `epsilon`

Controls the minimum score improvement required to continue the search.

Default:

```python
epsilon=1e-4
```

The search terminates when the improvement falls below this threshold.

---

## `max_iter`

Maximum number of iterations.

Default:

```python
max_iter=1e6
```

Use this to limit computation for difficult search spaces.

---

# 8. Exhaustive Search

## Method

```python
methodtype='ex'
```

or:

```python
methodtype='exhaustivesearch'
```

Exhaustive Search evaluates all possible DAG structures and identifies the
best-scoring structure.

Example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='ex',
    scoretype='bic'
)
```

Exhaustive search becomes computationally infeasible very quickly because the
number of possible DAGs grows super-exponentially with the number of
variables. The implementation warns when more than 15 variables are supplied
and recommends Hill Climbing or Constraint Search instead.

### Recommended use

Use exhaustive search only for very small networks.

As a practical rule:

```text
< 5 variables
    → exhaustive search can be useful

larger networks
    → prefer hc or pc
```

The source specifically notes that heuristic approaches are useful when only a
few nodes are involved, particularly fewer than approximately five.

---

## `return_all_dags`

Set:

```python
return_all_dags=True
```

to return all candidate DAGs and their scores.

Example:

```python
result = bn.structure_learning.fit(
    df,
    methodtype='ex',
    scoretype='bic',
    return_all_dags=True
)
```

The returned dictionary then contains:

```text
scores
dag
```

in addition to the normal structure-learning output.

This option should only be used for very small networks.

---

# 9. Constraint-Based Structure Learning: PC

## Method

```python
methodtype='pc'
```

The aliases:

```python
methodtype='cs'
methodtype='constraintsearch'
```

also invoke constraint-based structure learning.

PC learns a graph using conditional independence tests rather than directly
optimizing a structure score.

Conceptually:

```text
Data
 │
 ▼
Conditional independence tests
 │
 ▼
Undirected skeleton
 │
 ▼
PDAG
 │
 ▼
DAG
```

The implementation first estimates a PDAG and then converts it to a DAG. The
result also retains the undirected skeleton and PDAG.

---

# 10. PC Conditional Independence Tests

The `params_pc` argument controls the conditional independence test and
significance level.

Default:

```python
params_pc = {
    'ci_test': 'chi_square',
    'alpha': 0.05
}
```

Supported tests include:

```text
chi_square
independence_match
pearsonr
g_sq
log_likelihood
freeman_tuckey
modified_log_likelihood
neyman
cressie_read
power_divergence
```

These are passed to the underlying pgmpy PC implementation.

Example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='pc',
    params_pc={
        'ci_test': 'chi_square',
        'alpha': 0.05
    }
)
```

---

# 11. Interpreting `alpha`

`alpha` determines the significance threshold for conditional independence
testing.

Default:

```python
alpha=0.05
```

The conditional independence null hypothesis is:

```text
X ⟂ Y | Z
```

A high p-value provides insufficient evidence to reject conditional
independence at the selected significance level.

Do not interpret `alpha` as an edge-strength parameter.

It controls the hypothesis-testing threshold used during structure discovery.

---

# 12. PC and Faithfulness

PC relies on assumptions connecting conditional independencies in the data to
the underlying DAG.

The implementation explicitly notes the importance of the faithfulness
assumption.

Spurious dependencies or independencies can result in structures that do not
correspond cleanly to a faithful DAG.

Therefore:

* Do not treat every PC orientation as established causality.
* Consider sampling variability.
* Consider statistical power.
* Consider whether the faithfulness assumption is plausible.
* Treat the learned DAG as model-dependent.

---

# 13. Chow-Liu

## Method

```python
methodtype='chow-liu'
```

or:

```python
methodtype='cl'
```

Chow-Liu searches for a tree-structured Bayesian Network.

The method requires:

```python
root_node='variable'
```

Example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='chow-liu',
    root_node='target'
)
```

The `cl` alias is internally converted to `chow-liu`.

Use Chow-Liu when a tree structure is desirable or required rather than a
general DAG.

---

# 14. Tree-Augmented Naive Bayes (TAN)

## Method

```python
methodtype='tan'
```

TAN is intended for classification and requires both:

```python
root_node
class_node
```

Example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='tan',
    root_node='feature_1',
    class_node='target'
)
```

The implementation explicitly requires `class_node` for TAN.

Use TAN when:

* There is a designated class/target variable.
* A tree-augmented Naive Bayes structure is appropriate.
* The user wants a classification-oriented Bayesian Network.

---

# 15. Naive Bayes

## Method

Use:

```python
methodtype='nb'
```

or:

```python
methodtype='naivebayes'
```

Naive Bayes is a special Bayesian Network structure in which the feature
variables are connected to a dependent/class variable.

Example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='nb',
    root_node='target'
)
```

`root_node` is required.

The implementation constructs a `NaiveBayes` model and fits it using the
specified root node.

---

# 16. Blacklists and Whitelists

Domain knowledge can be incorporated through:

```python
black_list
white_list
```

These constraints can operate at either the edge or node level depending on
`bw_list_method`.

---

## 16.1 Edge Constraints

Use:

```python
bw_list_method='edges'
```

with lists such as:

```python
black_list=[
    ('A', 'B'),
    ('C', 'D')
]
```

A blacklist excludes specified edges.

A whitelist restricts the search space to specified edges.

Edge-based blacklists and whitelists are supported only for Hill Climbing. The
implementation maps these constraints to pgmpy's `ExpertKnowledge`.

---

## 16.2 Node Constraints

Use:

```python
bw_list_method='nodes'
```

Example:

```python
white_list=[
    'A',
    'B',
    'C'
]
```

This filters the DataFrame to the specified nodes.

A node blacklist removes the specified variables.

Node filtering can be used across the supported structure-learning methods.

The implementation performs case-insensitive node matching.

---

## 16.3 Constraint Selection

Use:

```text
Need to control specific edges?
    → bw_list_method='edges'

Need to select variables?
    → bw_list_method='nodes'
```

When `black_list` or `white_list` is supplied, `bw_list_method` must also be
specified.

---

# 17. Fixed Edges

Hill Climbing supports:

```python
fixed_edges
```

Example:

```python
fixed_edges=[
    ('A', 'B')
]
```

Fixed edges are required to remain in the final learned model.

The implementation maps these to pgmpy's `ExpertKnowledge.required_edges`.
They are inserted at the beginning of the search and are not modified by the
algorithm.

Use fixed edges only when there is strong domain knowledge supporting the
relationship.

---

# 18. Scoring Methods

The following `scoretype` values are supported:

```text
bic
k2
bdeu
bds
aic
loglik-g
aic-g
bic-g
```

They are mapped internally as follows:

| `scoretype` | Score                   |
| ----------- | ----------------------- |
| `bic`       | BIC                     |
| `k2`        | K2                      |
| `bdeu`      | BDeu                    |
| `bds`       | BDs                     |
| `aic`       | AIC                     |
| `loglik-g`  | Gaussian log-likelihood |
| `aic-g`     | Gaussian AIC            |
| `bic-g`     | Gaussian BIC            |

The mapping is implemented in `_SetScoringType()`.

---

# 19. Choosing a Score

For discrete Bayesian Networks:

```text
General model selection
    → BIC

Bayesian Dirichlet scoring
    → BDeu / BDs

K2-compatible scoring
    → K2

AIC-based selection
    → AIC
```

For continuous variables:

```text
Continuous Gaussian model
    → loglik-g

Continuous Gaussian model with AIC penalty
    → aic-g

Continuous Gaussian model with BIC penalty
    → bic-g
```

Do not use a discrete score simply because it is the default when the dataset
contains continuous measurements.

---

# 20. Gaussian Structure Scores

`bnlearn` implements three Gaussian scores specifically for continuous data:

```text
loglik-g
aic-g
bic-g
```

The Gaussian model assumes each node follows a linear Gaussian regression on
its parents.

The total DAG score is decomposable into local node scores.

---

## 20.1 `loglik-g`

Use:

```python
scoretype='loglik-g'
```

This computes the Gaussian log-likelihood.

Higher scores are better.

The local model is:

```text
Y = β₀ + β₁X₁ + ... + βₚXₚ + ε
```

where the residual variance is estimated from the regression residuals.

---

## 20.2 `aic-g`

Use:

```python
scoretype='aic-g'
```

The Gaussian AIC score is:

```text
log-likelihood - number_of_parameters
```

The implementation counts:

```text
intercept
+ regression coefficients
+ variance
```

as model parameters.

---

## 20.3 `bic-g`

Use:

```python
scoretype='bic-g'
```

The Gaussian BIC score is:

```text
log-likelihood
- 0.5 * number_of_parameters * log(n)
```

where `n` is the number of observations.

BIC therefore applies a stronger complexity penalty as the sample size
increases.

---

# 21. Gaussian Data Requirements

The Gaussian scoring classes require:

* A pandas DataFrame.
* At least one observation.
* Numeric columns only.
* Finite values.
* No missing values.
* No infinite values.

The implementation explicitly raises an error when non-numeric columns,
missing values, or infinite values are detected.

Therefore:

```text
Continuous data
     │
     ├── Numeric?
     │      └── No → preprocess
     │
     ├── Missing values?
     │      └── Yes → handle before Gaussian scoring
     │
     └── Infinite values?
            └── Yes → handle before Gaussian scoring
```

---

# 22. Variance Floor

The Gaussian scores support:

```python
variance_floor=1e-12
```

This prevents estimated residual variances from reaching zero.

It must be greater than zero.

Example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic-g'
)
```

The current `fit()` API does not expose `variance_floor` directly, but the
Gaussian scoring implementation accepts it internally.

---

# 23. LiNGAM Structure Learning

`bnlearn` supports:

```text
direct-lingam
ica-lingam
```

These methods are particularly relevant when working with continuous or mixed
datasets.

Unlike score-based Bayesian Network learning, LiNGAM is a causal-discovery
approach based on assumptions about the structural relationships and error
distributions.

---

# 24. DirectLiNGAM

Use:

```python
methodtype='direct-lingam'
```

DirectLiNGAM assumes:

* Linear relationships.
* Non-Gaussian error terms.
* An acyclic graph.

The method attempts to infer causal ordering from independence relationships
between explanatory variables and regression residuals.

Example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='direct-lingam'
)
```

The returned object contains:

```text
model
model_edges
adjmat
causal_order
structure_scores
config
```

For LiNGAM, `structure_scores` is `None`.

---

# 25. ICA-LiNGAM

Use:

```python
methodtype='ica-lingam'
```

The implementation constructs an `ICALiNGAM` model and fits it directly to the
input DataFrame.

Use ICA-LiNGAM when the assumptions of ICA-LiNGAM are appropriate.

Do not automatically choose LiNGAM merely because the data are continuous.

The method makes stronger structural assumptions than generic score-based
learning.

---

# 26. LiNGAM Prior Knowledge

LiNGAM supports:

```python
params_lingam={
    'random_state': None,
    'prior_knowledge': None,
    'apply_prior_knowledge_softly': False,
    'measure': 'pwling'
}
```

`prior_knowledge` can encode known causal relationships.

The prior-knowledge matrix uses:

```text
0  → xi does not have a directed path to xj
1  → xi has a directed path to xj
-1 → no prior knowledge
```

This is particularly useful when domain knowledge constrains the possible
causal ordering.

---

# 27. LiNGAM Independence Measures

Supported measures are:

```text
pwling
kernel
pwling_fast
```

Default:

```python
measure='pwling'
```

For GPU-accelerated execution:

```text
pwling_fast
```

can be used when the required `culingam` dependency is available.

---

# 28. Causal Interpretation Warning

LiNGAM methods are explicitly intended for causal discovery, but causal
interpretation still depends on the assumptions of the method and the data.

Do not state:

> X causes Y

solely because an edge appears in a learned graph.

Instead consider:

* Whether the linearity assumption is reasonable.
* Whether errors are sufficiently non-Gaussian.
* Whether relevant variables are observed.
* Whether hidden confounding is plausible.
* Whether the causal ordering is physically meaningful.
* Whether the data-generating process is compatible with LiNGAM.

---

# 29. Parallel Processing

The main API exposes:

```python
n_jobs=-1
```

However, not every underlying method supports parallel processing.

In the current implementation:

* Hill Climbing does not use `n_jobs`.
* Exhaustive Search does not use `n_jobs`.
* Naive Bayes does not use `n_jobs`.
* Constraint Search does not use `n_jobs` directly.
* TreeSearch passes `n_jobs` to the underlying estimator.

Therefore, do not assume that setting:

```python
n_jobs=-1
```

will make every structure-learning method parallel.

---

# 30. Verbosity

The `verbose` parameter controls output:

```text
0 → None
1 → Error
2 → Warning
3 → Info
4 → Debug
5 → Trace
```

Default:

```python
verbose=3
```

For normal use:

```python
verbose=3
```

is appropriate.

For debugging:

```python
verbose=4
```

or:

```python
verbose=5
```

can provide additional information.

---

# 31. Inspecting the Result

After structure learning:

```python
result = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic'
)
```

Inspect:

```python
result['model']
result['model_edges']
result['adjmat']
result['config']
result['structure_scores']
```

For example:

```python
print(result['model_edges'])
```

The adjacency matrix can be inspected with:

```python
result['adjmat']
```

The learned model can be visualized using:

```python
bn.plot(result)
```

---

# 32. Structure Scores

After structure learning, `bnlearn` computes structure scores and stores them
in:

```python
result['structure_scores']
```

The returned `structure_scores` should be interpreted according to the
selected scoring convention.

When comparing models, always compare models evaluated under the same score
and compatible data assumptions.

Do not compare BIC, AIC, BDeu, and Gaussian likelihood values as if they were
the same numerical quantity.

---

# 33. Post-Learning Validation

A learned DAG should be inspected before using it for inference or causal
interpretation.

Recommended checks:

```text
1. Inspect nodes
2. Inspect edges
3. Inspect adjacency matrix
4. Check domain plausibility
5. Check graph complexity
6. Evaluate score
7. Test stability where appropriate
8. Check sensitivity to algorithm settings
```

For discrete networks, consider conditional-independence or edge-strength
analysis after learning.

For continuous networks, inspect:

* residual behavior,
* linearity assumptions,
* outliers,
* influential observations,
* stability of learned edges.

---

# 34. Recommended Method Selection

Use this practical guide:

| Situation                                    | Recommended method                    |
| -------------------------------------------- | ------------------------------------- |
| General Bayesian Network                     | `hc`                                  |
| Very small network                           | `ex`                                  |
| Conditional-independence discovery           | `pc`                                  |
| Tree-structured BN                           | `chow-liu`                            |
| Classification with Naive Bayes structure    | `nb`                                  |
| Classification with tree-augmented structure | `tan`                                 |
| Continuous Gaussian BN                       | `hc` + `bic-g` / `aic-g` / `loglik-g` |
| Continuous/mixed causal discovery            | `direct-lingam`                       |
| ICA-based causal discovery                   | `ica-lingam`                          |

This is a starting point, not an automatic rule. The data characteristics and
scientific objective should determine the final choice.

---

# 35. Recommended Defaults

For a general discrete Bayesian Network:

```python
result = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic'
)
```

For a continuous Gaussian Bayesian Network:

```python
result = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic-g'
)
```

For constraint-based discovery:

```python
result = bn.structure_learning.fit(
    df,
    methodtype='pc',
    params_pc={
        'ci_test': 'chi_square',
        'alpha': 0.05
    }
)
```

For LiNGAM causal discovery:

```python
result = bn.structure_learning.fit(
    df,
    methodtype='direct-lingam'
)
```

---

# 36. Common Mistakes

### Do not use exhaustive search on large networks

The number of possible DAGs grows super-exponentially.

Prefer:

```text
hc
pc
```

for larger networks.

---

### Do not use discrete scores for continuous data

For continuous Gaussian data, consider:

```text
loglik-g
aic-g
bic-g
```

instead of automatically discretizing the data.

---

### Do not interpret every edge as causal

A learned DAG represents a model of dependencies unless the assumptions
required for causal interpretation are satisfied.

---

### Do not ignore domain knowledge

If known relationships exist, use:

```text
black_list
white_list
fixed_edges
```

where appropriate.

---

### Do not use edge constraints without specifying `bw_list_method`

This is invalid:

```python
bn.structure_learning.fit(
    df,
    methodtype='hc',
    black_list=[('A', 'B')]
)
```

Specify:

```python
bw_list_method='edges'
```

---

### Do not use `max_indegree` with PC

`max_indegree` is only supported for Hill Climbing.

---

### Do not assume LiNGAM is simply another Gaussian BN algorithm

LiNGAM uses different structural assumptions and is intended for causal
discovery.

---

# 37. Minimal End-to-End Example

```python
import bnlearn as bn

# Learn structure
result = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic'
)

# Inspect learned edges
print(result['model_edges'])

# Inspect adjacency matrix
print(result['adjmat'])

# Visualize
bn.plot(result)
```

**Recommended post-processing (from practice):** prune edges that fail a
conditional independence test before estimating parameters:

```python
result = bn.independence_test(
    result, df,
    test='chi_square',
    alpha=0.05,
    prune=True,
)
```

Then pass the (possibly pruned) structure to parameter learning:

```python
model = bn.parameter_learning.fit(
    result,
    df
)
```

After parameter learning, the model can be used for inference, prediction, or
sampling.

---

# 38. AI Decision Rules

When an AI agent is asked to perform structure learning with bnlearn, follow
these rules:

1. Identify whether the data are discrete, continuous, or mixed.
2. Identify whether the objective is dependency discovery, classification, or
   causal discovery.
3. Prefer `hc` for general score-based Bayesian Network structure learning.
4. Prefer `pc` when conditional independence testing is central.
5. Use `ex` only for very small networks.
6. Use `chow-liu` when a tree structure is required.
7. Use `nb` for a Naive Bayes structure.
8. Use `tan` for a Tree-Augmented Naive Bayes structure.
9. For continuous Gaussian data, consider `bic-g`, `aic-g`, or `loglik-g`.
10. Do not automatically discretize continuous variables.
11. Consider LiNGAM for continuous/mixed causal discovery when its assumptions
    are appropriate.
12. Use domain constraints when they are known and justified.
13. Validate the learned structure before interpreting it.
14. Consider `bn.independence_test(..., prune=True)` after structure learning
    when the graph looks overly dense.
15. Never equate a learned edge with causality without appropriate assumptions.
16. Never invent bnlearn parameters or use an API from another Bayesian Network
    library.
17. When the user already supplies the edge list, use `make_DAG` +
    `parameter_learning.fit` — do not run structure learning.
16. Check the installed bnlearn version when API behavior may differ.

---

# 39. Related References

For additional guidance, see:

* `parameter_learning.md`
* `continuous_models.md`
* `scoring.md`
* `causal_discovery.md`
* `inference.md`
* `troubleshooting.md`
