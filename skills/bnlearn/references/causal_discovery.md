# Causal Discovery

> **Quick API (verified bnlearn ≥ 0.14)**
>
> **LiNGAM (continuous / mixed, causal orientation)**:
> ```python
> DAG = bn.structure_learning.fit(
>     df,
>     methodtype='direct-lingam',   # or 'ica-lingam'
>     params_lingam={
>         'random_state': None,
>         'prior_knowledge': None,
>         'apply_prior_knowledge_softly': False,
>         'measure': 'pwling',
>     },
> )
> ```
>
> **Constraint-based (PC)**:
> ```python
> DAG = bn.structure_learning.fit(
>     df,
>     methodtype='pc',
>     params_pc={'ci_test': 'chi_square', 'alpha': 0.05},
> )
> ```
>
> A learned edge `X → Y` is **not** automatically a causal claim. Causal
> interpretation requires additional assumptions (sufficiency, faithfulness,
> no unmeasured confounding, etc.). Prefer the language “learned dependency
> structure” unless those assumptions are stated.

---

Causal discovery is the process of using observational data and domain
knowledge to estimate a directed graphical structure that represents potential
causal relationships between variables.

In `bnlearn`, causal discovery is primarily implemented through
`bn.structure_learning.fit()`.

The resulting graph can subsequently be used for:

* Parameter learning.
* Probabilistic inference.
* Conditional probability queries.
* Synthetic data generation.
* Causal reasoning and intervention analysis.

The central workflow is:

```text
Observed Data
     │
     ▼
Causal Discovery
     │
     ▼
Estimated DAG
     │
     ▼
Parameter Learning
     │
     ▼
Bayesian Network
     │
     ├── Inference
     │
     └── Sampling
```

The important distinction is that **causal discovery estimates the structure**,
while parameter learning estimates the probability distributions associated
with that structure.

---

# 1. What Is Causal Discovery?

Suppose a dataset contains:

```text
Temperature
Pressure
Torque
Tool Wear
Machine Failure
```

A conventional supervised learning approach might define:

```text
Machine Failure
```

as the target variable.

Causal discovery takes a different approach.

Instead of specifying a target, the algorithm attempts to discover relationships
between the variables themselves.

For example, it may estimate a structure such as:

```text
Temperature ─────► Heat Failure
                      │
Torque ──────────────►│
                      ▼
                Machine Failure
```

The goal is to determine which relationships are supported by the data and
the assumptions of the selected discovery algorithm.

---

# 2. Causal Discovery Is Not Correlation

A statistical association does not automatically establish causality.

For example:

```text
X ───► Y
```

and:

```text
X ◄───► Y
```

are not equivalent causal statements.

Observed association may arise from:

* Direct causal relationships.
* Indirect causal relationships.
* Common causes.
* Selection effects.
* Measurement effects.
* Confounding.
* Sampling variation.

Therefore, a discovered graph should be interpreted together with the
assumptions of the discovery method and domain knowledge.

---

# 3. Causal Discovery in bnlearn

The primary interface is:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic'
)
```

The function receives a pandas DataFrame and estimates a graphical structure.

The implementation supports several broad families:

```text
1. Score-based
2. Constraint-based
3. Tree-search based
4. LiNGAM-based
```

The implementation explicitly describes score-based, constraint-based, and
hybrid structure learning as the main broad categories.

---

# 4. The Causal Discovery Pipeline

A typical causal discovery pipeline is:

```text
                 Dataset
                    │
                    ▼
          Data preparation
                    │
                    ▼
        Select discovery method
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Score      PC       LiNGAM
       based   constraint   based
          │         │         │
          └─────────┼─────────┘
                    ▼
              Estimated graph
                    │
                    ▼
             Evaluate structure
                    │
                    ▼
             Learn parameters
                    │
                    ▼
           Bayesian Network
```

The appropriate branch depends strongly on the type of data and assumptions
that can reasonably be made about the generating process.

---

# 5. Structure Discovery Interface

The main function is:

```python
bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic',
    black_list=None,
    white_list=None,
    bw_list_method=None,
    max_indegree=None,
    tabu_length=100,
    epsilon=1e-4,
    max_iter=1e6,
    root_node=None,
    start_dag=None,
    class_node=None,
    fixed_edges=None,
    return_all_dags=False,
    params_lingam={
        'random_state': None,
        'prior_knowledge': None,
        'apply_prior_knowledge_softly': False,
        'measure': 'pwling'
    },
    params_pc={
        'ci_test': 'chi_square',
        'alpha': 0.05
    },
    n_jobs=-1
)
```

The exact implementation supports the following discovery methods:

```text
pc
cs
constraintsearch

hc
hillclimbsearch

ex
exhaustivesearch

cl
chow-liu

nb
naivebayes

tan

direct-lingam
ica-lingam
```

The aliases are normalized internally where required.

---

# 6. Method Selection

A useful decision rule is:

```text
Discrete data
    │
    ├── Score-based
    │      ├── hc
    │      └── ex
    │
    └── Constraint-based
           └── pc

Continuous / mixed data
    │
    └── LiNGAM
           ├── direct-lingam
           └── ica-lingam

Tree-constrained models
    │
    ├── chow-liu
    └── tan

Classification-oriented structure
    │
    └── naivebayes
```

This is not simply a performance choice. Each method encodes different
assumptions about how the causal structure can be identified.

---

# 7. Score-Based Causal Discovery

Score-based discovery treats structure learning as an optimization problem.

There are two components:

```text
Scoring function
       +
Search strategy
```

The scoring function evaluates how well a candidate DAG explains the observed
data.

The search strategy explores the enormous space of possible DAGs.

Conceptually:

```text
Candidate DAG
     │
     ▼
Score DAG
     │
     ▼
Modify DAG
     │
     ▼
Score new DAG
     │
     ▼
Keep better structure
     │
     ▼
Repeat
```

The search space of DAGs grows super-exponentially with the number of
variables, which is why heuristic search is important for larger networks.

---

# 8. Hill Climb Search

The default method is:

```python
methodtype='hc'
```

or:

```python
methodtype='hillclimbsearch'
```

Example:

```python
import bnlearn as bn

df = bn.import_example('sprinkler')

model = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic'
)
```

Hill climbing starts from a DAG and repeatedly applies local modifications
that improve the score.

The default starting structure is a disconnected DAG.

The algorithm performs single-edge operations and terminates when it reaches a
local maximum.

---

# 9. Hill Climbing Is a Local Search

Hill climbing does not enumerate every possible DAG.

Instead:

```text
Current DAG
    │
    ├── Add edge
    ├── Remove edge
    └── Reverse edge
         │
         ▼
    Evaluate candidates
         │
         ▼
    Select improvement
         │
         ▼
    New DAG
```

This makes it substantially more practical than exhaustive search for larger
networks.

However, because the optimization is local, it can converge to a local
maximum rather than the globally optimal DAG.

---

# 10. Starting DAG

A starting DAG can be supplied:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic',
    start_dag=my_dag
)
```

The starting structure provides the initial point for the local search.

If a dictionary returned by `bnlearn` is supplied, the implementation extracts:

```python
start_dag['model']
```

before starting the search.

This makes `start_dag` useful when incorporating an existing structure into
the search.

---

# 11. Fixed Edges

Known relationships can be enforced using:

```python
fixed_edges=[
    ('A', 'B'),
    ('C', 'D')
]
```

These are mapped internally to pgmpy's:

```text
required_edges
```

The resulting edges are required to remain in the learned structure.

Use this when domain knowledge establishes that an edge must be present.

For example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic',
    fixed_edges=[
        ('Temperature', 'Heat_Failure')
    ]
)
```

---

# 12. Blacklist

A blacklist prevents specific edges from being considered.

Example:

```python
black_list=[
    ('Machine Failure', 'Torque')
]
```

Conceptually:

```text
Machine Failure ──X──► Torque
```

The edge is forbidden during the search when edge-based filtering is used.

Blacklists are useful when domain knowledge says that a particular causal
direction is not allowed.

---

# 13. Whitelist

A whitelist restricts the allowed search space.

Example:

```python
white_list=[
    ('Temperature', 'Heat_Failure'),
    ('Torque', 'Overstrain_Failure')
]
```

Conceptually, the search is constrained to the supplied relationships.

For edge-based constraints, `white_list` is supported by HillClimbSearch through
pgmpy's `search_space` mechanism.

---

# 14. Blacklist and Whitelist Modes

The parameter:

```python
bw_list_method
```

controls how blacklists and whitelists are interpreted.

Supported modes are:

```text
edges
nodes
```

### `edges`

The lists describe edges:

```python
[
    ('A', 'B'),
    ('C', 'D')
]
```

This mode is restricted to:

```text
methodtype='hc'
```

### `nodes`

The lists describe variables:

```python
[
    'A',
    'B',
    'C'
]
```

The DataFrame is filtered to include or exclude those variables.

The implementation performs node filtering independently of the structure
learning algorithm.

---

# 15. Maximum Number of Parents

Hill climbing can be constrained using:

```python
max_indegree
```

For example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='hc',
    max_indegree=3
)
```

This restricts the search to models where each node has at most three parents.

This can substantially reduce the search space for larger networks.

The parameter is only effective for:

```text
methodtype='hc'
```

according to the implementation.

---

# 16. Tabu Search

Hill climbing supports:

```python
tabu_length=100
```

Tabu search prevents recently explored operations from immediately being
revisited.

This can help the search move away from local structures that would otherwise
cause cycling or premature convergence.

The value is passed directly to the underlying HillClimbSearch estimator.

---

# 17. Search Termination

Two important termination parameters are:

```python
epsilon=1e-4
max_iter=1e6
```

`epsilon` defines the minimum score improvement required to continue.

If the improvement is smaller than `epsilon`, the search terminates.

`max_iter` provides an upper bound on the number of iterations.

Therefore:

```text
score improvement < epsilon
          OR
iterations > max_iter
          ↓
       terminate
```

---

# 18. Exhaustive Search

The alternative score-based method is:

```python
methodtype='ex'
```

or:

```python
methodtype='exhaustivesearch'
```

Example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='ex',
    scoretype='bic'
)
```

Exhaustive search evaluates the possible DAG structures and selects the
best-scoring structure.

This is useful for small networks where the complete search space is
manageable.

---

# 19. Exhaustive Search Does Not Scale

The number of possible DAGs increases extremely rapidly with the number of
variables.

The implementation therefore warns when:

```text
number of nodes > 15
```

and recommends HillClimbSearch or constraint-based search instead.

A practical rule is:

```text
Small network
    → Exhaustive search

Larger network
    → Hill climbing or PC
```

Do not interpret exhaustive search as the default method for large datasets.

---

# 20. Returning All DAGs

For exhaustive search:

```python
return_all_dags=True
```

requests all candidate DAGs and their scores.

Example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='ex',
    scoretype='bic',
    return_all_dags=True
)
```

The implementation stores:

```python
model['scores']
model['dag']
```

and plots the scores.

This option is intended for small networks because enumerating all DAGs becomes
intractable rapidly.

---

# 21. Constraint-Based Causal Discovery

Constraint-based discovery uses statistical independence tests instead of
directly optimizing a score.

The fundamental question is:

```text
Are X and Y independent given Z?
```

Formally:

```text
X ⟂ Y | Z
```

If the data provide evidence for conditional independence, an edge can be
removed or an edge orientation can be constrained.

---

# 22. PC Algorithm

The primary constraint-based method is:

```python
methodtype='pc'
```

with aliases:

```text
cs
constraintsearch
```

Example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='pc'
)
```

The PC algorithm proceeds conceptually through:

```text
Complete graph
      │
      ▼
Conditional independence tests
      │
      ▼
Remove independent edges
      │
      ▼
Undirected skeleton
      │
      ▼
Orient compelled edges
      │
      ▼
PDAG
      │
      ▼
DAG completion
```

The implementation explicitly constructs the skeleton, PDAG, and final DAG.

---

# 23. Conditional Independence

A conditional independence test evaluates:

```text
H0: X ⟂ Y | Z
```

The resulting p-value is interpreted under the null hypothesis that `X` and
`Y` are independent given `Z`.

The PC algorithm uses these tests to determine which relationships should
remain in the graph.

---

# 24. Significance Level

The PC algorithm uses:

```python
params_pc={
    'alpha': 0.05
}
```

The value controls the significance threshold for conditional independence
tests.

For example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='pc',
    params_pc={
        'ci_test': 'chi_square',
        'alpha': 0.01
    }
)
```

A smaller alpha makes the independence decision more conservative.

The default implementation uses:

```text
alpha = 0.05
```

---

# 25. PC Conditional Independence Tests

The implementation exposes the following `ci_test` options:

```text
chi_square
pearsonr
g_sq
log_likelihood
freeman_tuckey
modified_log_likelihood
neyman
cressie_read
power_divergence
```

These are passed to the underlying pgmpy constraint-based estimator.

The appropriate test depends on the data type and assumptions of the selected
test.

---

# 26. PC Skeleton

The first important output of PC is the undirected skeleton.

For example:

```text
A ─── B
│
C ─── D
```

The skeleton tells you which pairs of variables remain connected.

It does not yet specify every causal direction.

In the implementation:

```python
skel = pdag.to_undirected()
```

and it is returned as:

```python
out['undirected']
out['undirected_edges']
```

---

# 27. PC PDAG

The next representation is a partially directed acyclic graph:

```text
PDAG
```

It contains directed and potentially undirected relationships.

The PDAG represents the structure implied by the conditional independence
information before all remaining edges are oriented.

The implementation returns:

```python
out['pdag']
out['pdag_edges']
```

---

# 28. PC DAG

The final graph is obtained through:

```python
dag = pdag.to_dag()
```

and is stored as:

```python
out['dag']
out['dag_edges']
out['model']
```

Therefore, the model returned by `structure_learning.fit()` is the final DAG
used by bnlearn.

---

# 29. Important PC Limitation

PC relies on assumptions about the relationship between the observed
conditional independencies and an underlying DAG.

The implementation notes the importance of **faithfulness**.

If the estimated independencies violate faithfulness, the resulting PDAG may
not have a valid faithful DAG completion.

Therefore, PC output should not be interpreted as an unquestionable causal
truth.

It is an estimated structure under the assumptions of the algorithm.

---

# 30. Score-Based vs Constraint-Based

The two approaches answer different computational questions.

### Score-based

```text
Which DAG gives the best score?
```

### Constraint-based

```text
Which relationships are incompatible with
the conditional independencies in the data?
```

Conceptually:

```text
Score-based
Data → candidate DAGs → scores → best DAG

Constraint-based
Data → independence tests → skeleton → orientations → DAG
```

---

# 31. Hybrid Causal Discovery

Hybrid structure learning combines the ideas of:

```text
constraint-based discovery
+
score-based optimization
```

The general motivation is to use statistical independence information to
reduce the search space and then use scoring to select among plausible
structures.

In the current `structure_learning.py` implementation, the main explicit
constraint-based interface is PC and the main score-based interfaces are
HillClimbSearch and ExhaustiveSearch. The documentation should therefore not
claim that `bnlearn` exposes a separate generic `"hybrid"` method unless the
installed version contains such an interface.

---

# 32. Tree-Based Structure Learning

`bnlearn` also supports tree-constrained structures through:

```text
chow-liu
tan
```

These methods use pgmpy's `TreeSearch`.

The implementation calls:

```python
TreeSearch(
    df,
    root_node=root_node,
    n_jobs=n_jobs
)
```

and then estimates the selected tree structure.

---

# 33. Chow-Liu

Chow-Liu is selected using:

```python
methodtype='cl'
```

or:

```python
methodtype='chow-liu'
```

A root node is required.

Example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='chow-liu',
    root_node='Cloudy'
)
```

The resulting structure is constrained to a tree.

This is useful when a tree-structured approximation is desired.

---

# 34. Tree-Augmented Naive Bayes

TAN is selected using:

```python
methodtype='tan'
```

TAN requires:

```python
root_node
class_node
```

Example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='tan',
    root_node='Cloudy',
    class_node='Wet_Grass'
)
```

The implementation explicitly checks that `class_node` is supplied when using
TAN.

---

# 35. Naive Bayes

Naive Bayes is selected using:

```python
methodtype='nb'
```

or:

```python
methodtype='naivebayes'
```

It requires:

```python
root_node
```

The resulting structure is a special Bayesian model in which feature variables
are parents of the dependent variable.

Example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='naivebayes',
    root_node='Target'
)
```

---

# 36. LiNGAM-Based Causal Discovery

For continuous and mixed datasets, `bnlearn` supports:

```text
direct-lingam
ica-lingam
```

These methods are fundamentally different from ordinary score-based discrete
Bayesian-network structure learning.

The implementation uses the `lingam` package.

The DirectLiNGAM approach assumes:

```text
linear relationships
+
non-Gaussian error terms
+
acyclic structure
```

The implementation describes it as a method suitable for continuous and mixed
datasets.

---

# 37. DirectLiNGAM

Use:

```python
methodtype='direct-lingam'
```

Example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='direct-lingam'
)
```

The method estimates both:

```text
causal ordering
+
causal adjacency
```

The implementation stores the resulting causal order in:

```python
model['causal_order']
```

and the adjacency matrix in:

```python
model['adjmat']
```

---

# 38. DirectLiNGAM Causal Order

A key output is:

```python
model['causal_order']
```

For example:

```text
[
    'Temperature',
    'Torque',
    'Tool Wear',
    'Machine Failure'
]
```

The ordering represents the causal ordering estimated by LiNGAM.

This is particularly useful when the direction of relationships is important.

---

# 39. DirectLiNGAM Prior Knowledge

DirectLiNGAM supports:

```python
prior_knowledge
```

as an array-like matrix.

The implementation documents the values as:

```text
 0 → xi does not have a directed path to xj

 1 → xi has a directed path to xj

-1 → no prior knowledge
```

The matrix therefore allows prior causal knowledge to be incorporated into
causal discovery.

---

# 40. Soft Prior Knowledge

The parameter:

```python
apply_prior_knowledge_softly=False
```

controls whether the supplied prior knowledge should be applied softly.

Example:

```python
params_lingam={
    'prior_knowledge': prior_matrix,
    'apply_prior_knowledge_softly': True
}
```

This is only relevant to LiNGAM-based discovery.

---

# 41. LiNGAM Independence Measure

DirectLiNGAM supports:

```text
pwling
kernel
pwling_fast
```

The default is:

```text
pwling
```

The implementation notes that:

```text
pwling_fast
```

can be used for faster GPU execution when the required `culingam` dependency
is available.

---

# 42. ICA-LiNGAM

The alternative LiNGAM method is:

```python
methodtype='ica-lingam'
```

The implementation creates:

```python
lingam.ICALiNGAM(
    random_state=random_state
)
```

and fits it to the DataFrame.

The resulting causal adjacency matrix is exposed through:

```python
model.adjacency_matrix_
```

and the bnlearn output contains:

```python
model['adjmat']
model['model_edges']
model['causal_order']
```

---

# 43. Continuous Data and Causal Discovery

The choice of causal discovery method should follow the data type.

For discrete variables:

```text
hc + bic
hc + bdeu
hc + k2
pc
```

are natural choices.

For continuous or mixed variables:

```text
direct-lingam
ica-lingam
```

are specifically supported.

Alternatively, continuous variables can be transformed into discrete states
before using discrete Bayesian-network structure learning.

The continuous-model documentation should be consulted before discretizing
continuous measurements because discretization changes the representation of
the original variables.

---

# 44. Gaussian Score-Based Discovery

`bnlearn` also implements Gaussian score functions:

```text
loglik-g
aic-g
bic-g
```

These are designed for numeric continuous data.

The Gaussian local model treats each node as a linear Gaussian regression on
its parents.

The local scores are decomposable, making them compatible with
HillClimbSearch and ExhaustiveSearch.

Example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic-g'
)
```

This should not be confused with LiNGAM.

```text
Gaussian score
    → linear Gaussian probabilistic model

LiNGAM
    → linear non-Gaussian causal model
```

The assumptions are different.

---

# 45. Causal Discovery With Mixed Data

Mixed datasets contain combinations such as:

```text
continuous
categorical
binary
```

For example:

```text
Temperature       continuous
Torque            continuous
Machine Type      categorical
Failure           binary
```

Not every discovery algorithm handles this representation identically.

DirectLiNGAM is explicitly supported for continuous and mixed datasets in the
implementation.

Another workflow is to transform continuous variables into categorical states
and then use discrete Bayesian-network structure learning.

---

# 46. Causal Discovery With Domain Knowledge

Purely data-driven discovery is not always appropriate.

Domain knowledge can be incorporated through:

```text
fixed_edges
black_list
white_list
start_dag
prior_knowledge
```

These mechanisms have different semantics.

```text
fixed_edges
    → edges that must exist

black_list
    → edges that are forbidden

white_list
    → allowed search space

start_dag
    → starting structure

prior_knowledge
    → LiNGAM causal constraints
```

Use the mechanism that corresponds to the actual strength of your prior
knowledge.

---

# 47. Do Not Confuse Whitelist With Fixed Edges

These are different.

A whitelist:

```python
white_list=[
    ('A', 'B')
]
```

restricts the search space.

A fixed edge:

```python
fixed_edges=[
    ('A', 'B')
]
```

requires the edge to remain in the final model.

Therefore:

```text
white_list
    = permitted search relationship

fixed_edges
    = required relationship
```

This distinction is important when encoding expert knowledge.

---

# 48. Structure Scores After Discovery

For score-based discovery, bnlearn computes structure scores after the model
has been learned.

The returned dictionary contains:

```python
model['structure_scores']
```

The implementation calls:

```python
bnlearn.structure_scores(
    out,
    df
)
```

after constructing the model and adjacency matrix.

Available scoring methods include:

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

# 49. Comparing Candidate Structures

Structure scores should primarily be used for **relative comparison** between
candidate models under the same scoring framework.

For example:

```python
score_bic = bn.structure_scores(
    model,
    df,
    scoring_method='bic'
)
```

A higher score represents a better model under the higher-is-better convention
used by the underlying scoring implementation.

Do not compare scores from fundamentally different datasets as if they were
directly comparable.

---

# 50. Independence Testing After Discovery

After structure learning, edge strength can be assessed using:

```python
model = bn.independence_test(
    model,
    df
)
```

For example, the common workflow is:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic'
)

model = bn.independence_test(
    model,
    df
)
```

The resulting statistical information can then be used when inspecting or
visualizing the learned network.

This is especially useful because structure learning and statistical edge
assessment answer different questions:

```text
Structure learning
    → Which edges form the candidate network?

Independence testing
    → How strongly is an edge supported statistically?
```

A causal graph should therefore not be judged solely by whether an edge exists.

---

# 51. Causal Discovery Does Not Automatically Learn CPDs

`structure_learning.fit()` learns structure.

It does not replace parameter learning.

The workflow is:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic'
)

model = bn.parameter_learning.fit(
    model,
    df
)
```

After parameter learning, the model contains the probability distributions
required for inference and Bayesian sampling.

This distinction is fundamental:

```text
Structure learning
    → DAG

Parameter learning
    → CPDs
```

---

# 52. From Causal Discovery to Inference

Once the DAG and CPDs are available:

```python
query = bn.inference.fit(
    model,
    variables=['Machine Failure'],
    evidence={
        'Tool Wear': 'high'
    }
)
```

Inference calculates conditional probabilities using the learned Bayesian
Network.

The inference implementation uses Variable Elimination and computes:

```text
P(variables | evidence)
```

from the Bayesian Network.

Thus:

```text
Causal discovery
       ↓
DAG
       ↓
Parameter learning
       ↓
CPDs
       ↓
Inference
       ↓
P(Y | X)
```

---

# 53. From Causal Discovery to Sampling

The learned Bayesian Network can also generate synthetic data:

```python
df_synthetic = bn.sampling(
    model,
    n=1000,
    methodtype='bayes'
)
```

Conditional sampling can be performed with evidence:

```python
df_synthetic = bn.sampling(
    model,
    n=1000,
    methodtype='bayes',
    evidence={
        'Rain': 1
    }
)
```

This uses rejection sampling when evidence is supplied.

The important conceptual distinction is:

```text
Inference
    → calculates probabilities

Sampling
    → generates observations
```

---

# 54. Causal Discovery Is Not Intervention

A learned causal graph can support causal reasoning, but merely discovering:

```text
X → Y
```

does not mean that every observational conditional probability can be
interpreted as an intervention.

For example:

```text
P(Y | X=x)
```

is a conditional probability.

It is not automatically equivalent to:

```text
P(Y | do(X=x))
```

Interventional reasoning requires appropriate causal assumptions and a valid
causal model.

---

# 55. Observational Data and Causal Interpretation

The phrase "causal discovery" should therefore be interpreted carefully.

The algorithms infer causal structure from observational relationships under
their respective assumptions.

For example:

```text
PC
    → relies on conditional independencies
       and assumptions such as faithfulness.

LiNGAM
    → relies on linearity, non-Gaussian errors,
       and acyclicity.

Score-based methods
    → select structures according to
       probabilistic scores and search strategy.
```

A discovered DAG is therefore a **model of the causal structure**, not a
guarantee that the estimated arrows represent physical causation.

Domain knowledge and experimental validation remain important.

---

# 56. Causal Discovery With Predictive Maintenance

A typical use case is a machine dataset containing:

```text
Air Temperature
Process Temperature
Rotational Speed
Torque
Tool Wear
Failure Modes
Machine Failure
```

A supervised model might ask:

```text
Can we predict Machine Failure?
```

Causal discovery instead asks:

```text
Which variables are connected?
Which variables potentially influence others?
Which causal pathways explain the observed failure?
```

This is particularly useful when the objective is not merely prediction but
understanding which variables could potentially be changed to influence an
outcome.

---

# 57. Example: Complete Causal Discovery Pipeline

```python
import bnlearn as bn

# Load data
df = bn.import_example('predictive_maintenance')

# Learn structure
model = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic'
)

# Assess edge relationships
model = bn.independence_test(
    model,
    df
)

# Inspect the learned graph
print(model['model_edges'])

# Learn CPDs
model = bn.parameter_learning.fit(
    model,
    df
)

# Query the model
query = bn.inference.fit(
    model,
    variables=['Machine Failure'],
    evidence={
        'Tool Wear': 'high'
    }
)

print(query.df)
```

The complete pipeline is:

```text
Data
 │
 ▼
Structure learning
 │
 ▼
Candidate causal DAG
 │
 ▼
Independence assessment
 │
 ▼
Parameter learning
 │
 ▼
Bayesian Network
 │
 ▼
Inference
```

---

# 58. Example: PC-Based Discovery

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

Use PC when conditional independence testing is the intended discovery
framework.

Inspect:

```python
model['undirected']
model['pdag']
model['dag']
model['model_edges']
```

to understand the different stages of the discovered structure.

---

# 59. Example: DirectLiNGAM Discovery

For continuous or mixed data:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='direct-lingam'
)
```

Inspect:

```python
print(model['causal_order'])
print(model['model_edges'])
print(model['adjmat'])
```

The result contains the causal ordering and estimated adjacency structure.

---

# 60. Example: Constrained Hill Climbing

```python
model = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic',
    fixed_edges=[
        ('Temperature', 'Heat_Failure')
    ],
    black_list=[
        ('Machine Failure', 'Temperature')
    ]
)
```

This combines:

```text
Required knowledge
    +
Forbidden knowledge
    +
Data-driven optimization
```

This is often preferable to completely unconstrained structure learning when
strong domain knowledge exists.

---

# 61. Choosing a Discovery Method

A practical decision table is:

| Situation                              | Recommended method |
| -------------------------------------- | ------------------ |
| Small discrete network                 | `ex`               |
| General discrete network               | `hc` + `bic`       |
| Want Bayesian Dirichlet scoring        | `hc` + `bdeu`      |
| Conditional-independence approach      | `pc`               |
| Continuous linear non-Gaussian data    | `direct-lingam`    |
| Alternative LiNGAM approach            | `ica-lingam`       |
| Tree-structured approximation          | `chow-liu`         |
| Tree-augmented classification          | `tan`              |
| Naive Bayes structure                  | `naivebayes`       |
| Continuous Gaussian score-based search | `hc` + `bic-g`     |

This should be treated as a starting point rather than an automatic method
selection rule.

---

# 62. Computational Complexity

Causal discovery becomes increasingly difficult as the number of variables
increases.

The main reason is the enormous number of possible DAG structures.

Consequently:

```text
Number of variables
       ↑
       │
       ▼
Search space
       ↑↑↑
```

For small networks, exhaustive search may be useful.

For larger networks:

```text
HillClimbSearch
```

or:

```text
PC
```

is generally more practical.

For continuous/mixed data where LiNGAM assumptions are appropriate:

```text
DirectLiNGAM
```

provides a different route to causal discovery.

---

# 63. Parallelization

The main interface exposes:

```python
n_jobs=-1
```

which allows supported underlying algorithms to use parallel processing.

However, not every algorithm uses this parameter.

The implementation explicitly notes that `n_jobs` is not supported for:

```text
NaiveBayes
HillClimbSearch
ExhaustiveSearch
ConstraintSearch
```

in the corresponding internal implementations.
Do not assume that:

```python
n_jobs=-1
```

means every discovery method will run in parallel.

---

# 64. Returned Model Structure

For most structure-learning methods, the returned dictionary contains:

```python
model['model']
model['model_edges']
model['adjmat']
model['config']
model['structure_scores']
```

The implementation constructs these fields after discovery.

For PC, additional outputs are available:

```python
model['undirected']
model['undirected_edges']
model['pdag']
model['pdag_edges']
model['dag']
model['dag_edges']
```

For LiNGAM, additional information includes:

```python
model['causal_order']
```

---

# 65. Adjacency Matrix

The adjacency matrix is stored as:

```python
model['adjmat']
```

For ordinary DAG structure learning, it is generated using:

```python
bnlearn.dag2adjmat(model['model'])
```

For LiNGAM, the adjacency matrix comes from the estimated LiNGAM model.

The matrix represents the directed relationships between source and target
variables.

---

# 66. Model Edges

The learned edges can be inspected using:

```python
model['model_edges']
```

For example:

```python
print(model['model_edges'])
```

might return:

```text
[
    ('Cloudy', 'Rain'),
    ('Cloudy', 'Sprinkler'),
    ('Rain', 'Wet_Grass'),
    ('Sprinkler', 'Wet_Grass')
]
```

This is often the easiest representation for inspecting a discovered graph.

---

# 67. Verbosity

The default is:


The supported levels are:

```text
0: None
1: Error
2: Warning
3: Info
4: Debug
5: Trace
```

At information level, bnlearn reports the selected discovery method and
important processing steps.

For example:

```text
[bnlearn] >Computing best DAG using [hc]
```

The exact diagnostic output depends on the selected algorithm.

---

# 68. Data Requirements

The input must be:

```python
pandas.DataFrame
```

Each column represents a variable.

The implementation verifies the DataFrame type before structure learning.

Column names are converted to strings:

```python
df.columns = df.columns.astype(str)
```

Therefore, downstream variable references should use the resulting string
column names.

---

# 69. Avoid Identifier Columns

Causal discovery should generally operate on meaningful variables.

Do not blindly include identifiers such as:

```text
UDI
row_id
transaction_id
UUID
```

unless they represent meaningful variables in the causal system.

Identifier columns can create meaningless associations and unnecessarily
increase the search space.

---

# 70. Avoid Leakage

Variables that contain information derived directly from the outcome can create
misleading structures.

For example, if:

```text
Machine Failure
```

is constructed from:

```text
Failure A
Failure B
Failure C
```

then the relationship is partly definitional rather than an independently
discovered causal relationship.

Before causal discovery, determine whether variables are:

```text
cause
effect
proxy
derived variable
identifier
composite outcome
```

This is a modeling decision, not something the structure-learning algorithm
can determine automatically.

---

# 71. Causal Discovery Checklist

Before running causal discovery:

* [ ] Understand what each variable represents.
* [ ] Remove irrelevant identifiers.
* [ ] Check missing values and data quality.
* [ ] Determine whether variables are discrete, continuous, or mixed.
* [ ] Decide whether discretization is appropriate.
* [ ] Consider known causal constraints.
* [ ] Decide whether score-based or constraint-based discovery is appropriate.
* [ ] Consider LiNGAM when its assumptions are appropriate.
* [ ] Select a suitable scoring function or independence test.
* [ ] Consider limiting the maximum indegree for large networks.
* [ ] Consider blacklists, whitelists, or fixed edges when domain knowledge is
  available.
* [ ] Inspect the resulting graph.
* [ ] Evaluate the discovered structure rather than blindly accepting it.
* [ ] Learn CPDs separately.
* [ ] Use inference or sampling only after the Bayesian Network has been
  parameterized.

---

# 72. AI Decision Rules

When an AI agent needs to perform causal discovery with bnlearn:

1. Start with a pandas DataFrame.
2. Identify the data type of each variable.
3. Do not automatically designate a target variable.
4. Use `hc` as the general score-based starting point for discrete data.
5. Use `bic` when a penalized score-based structure is desired.
6. Use `ex` only for sufficiently small networks.
7. Use `pc` when conditional-independence-based discovery is desired.
8. Use `direct-lingam` for continuous or mixed data when its linear,
   non-Gaussian, acyclic assumptions are appropriate.
9. Use `ica-lingam` when that LiNGAM variant is specifically desired.
10. Use `chow-liu` when a tree-structured model is appropriate.
11. Use `tan` only when the required `root_node` and `class_node` are
    specified.
12. Use `naivebayes` only when the Naive Bayes structure is appropriate.
13. Use `fixed_edges` for relationships that must be present.
14. Use `black_list` for relationships that must be excluded.
15. Use `white_list` to restrict the search space.
16. Do not confuse a whitelist with a set of mandatory edges.
17. Use `start_dag` when an existing structure should seed HillClimbSearch.
18. Use `max_indegree` to constrain HillClimbSearch for larger networks.
19. Do not assume `n_jobs` is supported equally by every method.
20. Inspect `model['model_edges']` and `model['adjmat']` after discovery.
21. For PC, inspect `undirected`, `pdag`, and `dag` when understanding the
    discovery process.
22. For LiNGAM, inspect `causal_order`.
23. Remember that structure learning does not learn CPDs.
24. Run parameter learning before probabilistic inference or Bayesian sampling.
25. Do not interpret conditional probability as causal intervention.
26. Treat a discovered DAG as a model under the assumptions of the selected
    discovery method, not as automatically verified ground truth.
27. Use domain knowledge and, where possible, experimental or external
    validation to support causal interpretation.

---

# 73. Complete Causal Modeling Workflow

The complete bnlearn causal workflow is:

```text
                         DATA
                          │
                          ▼
                  Data Understanding
                          │
                          ▼
                  Data Preparation
                          │
                          ▼
                 Causal Discovery
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
      Score-based    Constraint-based   LiNGAM
          │               │                │
          └───────────────┼────────────────┘
                          ▼
                    Estimated DAG
                          │
                          ▼
                 Structure Evaluation
                          │
                          ▼
                  Parameter Learning
                          │
                          ▼
                Bayesian Network + CPDs
                          │
              ┌───────────┼───────────┐
              │           │           │
              ▼           ▼           ▼
          Inference    Sampling    Prediction
              │
              ▼
       Causal reasoning
```

The important conceptual separation is:

```text
Causal discovery
    = learn relationships

Parameter learning
    = learn probabilities

Inference
    = ask probability questions

Sampling
    = generate synthetic observations
```

---

# 74. The Core Principle

The most important concept for using `bnlearn` for causal discovery is that
there is no universally correct structure-learning algorithm.

Different algorithms identify structure using different assumptions:

```text
HillClimbSearch
    → optimize a structure score

ExhaustiveSearch
    → evaluate the complete feasible search space

PC
    → exploit conditional independencies

Chow-Liu / TAN
    → impose tree-based structural constraints

DirectLiNGAM / ICA-LiNGAM
    → exploit assumptions about linearity,
      non-Gaussianity and causal ordering
```

Therefore, causal discovery should be treated as a **modeling process**:

```text
Data
 +
Assumptions
 +
Domain knowledge
 +
Discovery algorithm
 +
Model evaluation
 =
Causal structure hypothesis
```

That causal structure can then become the foundation for parameter learning,
probabilistic inference, synthetic data generation, and—when the causal
assumptions are justified—causal reasoning about potential interventions.
