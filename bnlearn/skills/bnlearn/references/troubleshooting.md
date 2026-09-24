# Troubleshooting

> **Most common agent errors (check these first)**
>
> | Symptom | Likely cause | Fix |
> |---------|--------------|-----|
> | `TypeError` / unexpected keyword | Used `method=` instead of `methodtype=` | Always `methodtype=` |
> | Wrong / empty graph | Discrete score on continuous data (or vice-versa) | Use `bic-g` / `aic-g` / `loglik-g` for continuous |
> | `ValueError` on inference/sampling | Invalid evidence variable or state | Check node names and CPD cardinalities |
> | `ValueError` “gibbs … evidence” | `methodtype='gibbs'` + `evidence=` | Use `methodtype='bayes'` for conditional sampling |
> | AttributeError on return value | Accessing non-existent keys | Use `model['model_edges']`, `model['adjmat']`, etc. |
> | Inference fails | Parameter learning was skipped | Call `parameter_learning.fit` after structure learning |

---

This document provides a systematic troubleshooting guide for `bnlearn`.

The objective is to diagnose problems by inspecting:

```text
Input data
    ↓
Structure
    ↓
Parameters / CPDs
    ↓
Inference / Sampling
```

Most `bnlearn` problems occur because one of these four layers is missing,
incompatible, incorrectly specified, or inconsistent with the selected method.

---

# 1. First Diagnostic Step

Before changing parameters, inspect the object and data:

```python
print(type(df))
print(df.shape)
print(df.dtypes)
print(df.head())
```

For a learned model:

```python
print(type(model))
print(model.keys())
print(model['model'])
print(model['model_edges'])
print(model['adjmat'])
```

For a Bayesian Network:

```python
print(model['model'].nodes())
print(model['model'].edges())
print(model['model'].get_cpds())
```

The most important question is:

> At which stage does the problem occur?

```text
Data
  ↓
Structure learning
  ↓
Parameter learning
  ↓
Inference
  ↓
Sampling
```

Do not troubleshoot inference if parameter learning has not successfully
completed.

---

# 2. Model Dictionary vs BayesianNetwork

A common source of confusion is that `bnlearn` usually returns a dictionary-like
model object containing several components.

For example:

```python
model['model']
model['adjmat']
model['model_edges']
```

The actual pgmpy Bayesian Network is:

```python
model['model']
```

Therefore:

```python
type(model)
```

and:

```python
type(model['model'])
```

are not necessarily the same.

---

# 3. Check the Bayesian Network

Many downstream functions require a Bayesian Network rather than a plain DAG.

Check:

```python
print(type(model['model']))
```

A valid Bayesian Network should expose:

```python
model['model'].nodes()
model['model'].edges()
model['model'].get_cpds()
```

If CPDs are absent:

```python
print(model['model'].get_cpds())
```

and the result is empty, parameter learning has not yet been performed.

---

# 4a. Error: unhashable type: 'list'

This is wrong:
```python

DAG = [
    ["A", "C"],
    ["B", "C"],
    ["C", "D"],
]
```

This is correct:
```python

DAG = [
    ("A", "C"),
    ("B", "C"),
    ("C", "D"),
]

```

# 4b. Error: Input Must Be a DataFrame

Structure learning expects:

```python
pandas.DataFrame
```

Check:

```python
import pandas as pd

print(isinstance(df, pd.DataFrame))
```

If this returns:

```text
False
```

convert the data:

```python
df = pd.DataFrame(df)
```

Then inspect:

```python
print(df.shape)
print(df.dtypes)
```

---

# 5. Error: Empty DataFrame

Check:

```python
print(df.shape)
```

A DataFrame with zero rows or zero columns cannot provide useful evidence for
structure learning.

Use:

```python
if df.empty:
    raise ValueError("DataFrame is empty.")
```

Also check after preprocessing:

```python
print(df.shape)
```

A filtering or `dropna()` operation may have removed all observations.

---

# 6. Error: Column Names Do Not Match

`bnlearn` uses variable names directly from the DataFrame.

Inspect:

```python
print(df.columns.tolist())
```

Column names are converted to strings during structure learning.

Therefore use:

```python
df.columns = df.columns.astype(str)
```

before referencing variables.

Variable names are case-sensitive.

These are different:

```text
Temperature
temperature
TEMPERATURE
```

---

# 7. Missing Values

Check:

```python
print(df.isna().sum())
```

and:

```python
print(df.isnull().sum())
```

Also inspect the total:

```python
print(df.isna().sum().sum())
```

If missing values are present, decide how they should be handled before
structure learning.

Typical approaches include:

```python
df = df.dropna()
```

or explicit imputation.

Do not blindly drop rows when the missingness mechanism is important to the
analysis.

---

# 8. Constant Variables

A variable with only one unique value contains no useful variation.

Check:

```python
for col in df.columns:
    print(col, df[col].nunique())
```

Look for:

```text
nunique == 1
```

Such columns should normally be removed:

```python
constant_columns = [
    col for col in df.columns
    if df[col].nunique() <= 1
]

df = df.drop(columns=constant_columns)
```

Constant variables can cause problems in statistical tests and parameter
estimation.

---

# 9. Duplicate Columns

Check:

```python
print(df.columns.duplicated())
```

Duplicate variable names can make model interpretation ambiguous.

Use:

```python
df.columns[df.columns.duplicated()]
```

to identify duplicates.

Ensure every model variable has a unique name.

---

# 10. Wrong Data Type

Inspect:

```python
print(df.dtypes)
```

Pay particular attention to:

```text
object
category
bool
int
float
```

Discrete Bayesian-network methods generally expect categorical states.

Continuous scoring methods expect numeric variables.

LiNGAM methods require numeric data.

---

# 11. Discrete vs Continuous Data

A frequent source of errors is selecting a method inconsistent with the data.

Conceptually:

```text
Discrete data
    → discrete Bayesian-network scores
    → conditional-independence tests

Continuous numeric data
    → Gaussian scores
    → LiNGAM

Mixed data
    → carefully select a method supporting the representation
```

Do not automatically use:

```python
scoretype='bic'
```

for continuous variables and assume it means Gaussian BIC.

For Gaussian structure learning use:

```python
scoretype='bic-g'
```

when appropriate.

---

# 12. Gaussian Score Errors

For continuous data, the Gaussian scores are:

```text
loglik-g
aic-g
bic-g
```

Example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic-g'
)
```

Check that the relevant columns are numeric:

```python
print(df.dtypes)
```

Convert appropriate columns if necessary:

```python
df = df.astype(float)
```

Do not use categorical strings with Gaussian scores.

---

# 13. Error: Unknown Structure-Learning Method

Check the requested method:

```python
methodtype
```

Supported method families include:

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

Use the exact supported spelling or alias.

For example:

```python
methodtype='hc'
```

is valid.

---

# 14. Error: Unknown Score Type

For discrete structure learning, supported scoring methods include:

```text
bic
k2
bdeu
bds
aic
```

For Gaussian structure learning:

```text
loglik-g
aic-g
bic-g
```

Example:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic-g'
)
```

Do not invent score names such as:

```text
gaussian-bic
continuous-bic
```

unless supported by the installed version.

---

# 15. Hill Climbing Does Not Improve

If HillClimbSearch returns a structure that looks unexpected, inspect:

```python
print(model['model_edges'])
print(model['structure_scores'])
```

Possible causes include:

* Local optimum.
* Weak statistical signal.
* Too many variables.
* Incorrect data type.
* Poor score selection.
* Missing domain constraints.
* Sampling noise.
* Strong correlations that create multiple plausible structures.

Try:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic',
    tabu_length=100
)
```

and compare results across reasonable configurations.

---

# 16. Hill Climbing Gets Stuck

Hill climbing is a local optimization algorithm.

It is not guaranteed to find the globally optimal DAG.

If the result is unstable:

```text
Run A → DAG A
Run B → DAG B
Run C → DAG C
```

do not immediately assume that one run is correct.

Instead:

1. Compare structure scores.
2. Inspect common edges.
3. Check domain knowledge.
4. Use constraints where justified.
5. Compare with another discovery method.
6. Increase data quality or sample size where possible.

---

# 17. Use `start_dag`

If an existing DAG provides a reasonable starting point:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic',
    start_dag=start_dag
)
```

This can be useful when domain knowledge provides an initial structure.

Verify:

```python
start_dag['model']
```

is a valid graph before passing it to structure learning.

---

# 18. Fixed Edges Are Not a Whitelist

These two mechanisms are different.

Required edges:

```python
fixed_edges=[
    ('A', 'B')
]
```

Allowed search relationships:

```python
white_list=[
    ('A', 'B')
]
```

If you need to guarantee that an edge exists, use:

```python
fixed_edges
```

If you want to restrict the search space, use:

```python
white_list
```

Do not substitute one for the other.

---

# 19. Blacklist Direction

Edges are directional.

These are different constraints:

```python
('A', 'B')
```

and:

```python
('B', 'A')
```

Therefore:

```python
black_list=[('A', 'B')]
```

does not necessarily mean:

```text
A ↔ B forbidden
```

It specifies the relationship in the supplied direction.

When troubleshooting constraints, print them explicitly:

```python
print(black_list)
print(white_list)
print(fixed_edges)
```

---

# 20. `bw_list_method` Problems

The parameter supports:

```text
edges
nodes
```

### Edge mode

Use:

```python
bw_list_method='edges'
```

with entries such as:

```python
[
    ('A', 'B'),
    ('C', 'D')
]
```

### Node mode

Use:

```python
bw_list_method='nodes'
```

with variable names:

```python
[
    'A',
    'B'
]
```

Do not mix node names and edge tuples in the same mode.

---

# 21. Maximum Indegree

If HillClimbSearch produces overly complex structures, constrain parent count:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='hc',
    max_indegree=3
)
```

Inspect the result:

```python
for node in model['model'].nodes():
    print(node, model['model'].get_parents(node))
```

Do not choose an arbitrarily small value because it can prevent important
relationships from being discovered.

---

# 22. Exhaustive Search Is Too Slow

Exhaustive search is appropriate only for relatively small networks.

Do not use:

```python
methodtype='ex'
```

for a large number of variables.

The search space grows extremely rapidly.

For larger networks prefer:

```python
methodtype='hc'
```

or:

```python
methodtype='pc'
```

when appropriate.

---

# 23. PC Produces Unexpected Results

PC is based on conditional independence tests.

Check:

```python
params_pc={
    'ci_test': 'chi_square',
    'alpha': 0.05
}
```

Important parameters are:

```text
ci_test
alpha
```

If the structure changes substantially when alpha changes, the discovered
relationships may be sensitive to the statistical threshold.

Compare, for example:

```python
alpha=0.01
alpha=0.05
alpha=0.10
```

and inspect the stability of important edges.

---

# 24. PC Independence-Test Problems

The implementation supports:

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

Choose a test appropriate for the data representation.

Do not use a categorical chi-square test simply because it is the default if
the variables are genuinely continuous.

For continuous data, investigate whether a continuous-data discovery approach
is more appropriate.

---

# 25. PC Alpha Too Strict or Too Liberal

The default is:

```python
alpha=0.05
```

If alpha is reduced:

```python
alpha=0.01
```

the independence decisions become more conservative.

If alpha is increased:

```python
alpha=0.10
```

more relationships may be retained.

Do not choose alpha solely to obtain a desired graph.

Choose it according to the statistical context and evaluate the resulting
structure for stability.

---

# 26. PC Skeleton vs DAG

When troubleshooting PC, do not inspect only:

```python
model['model_edges']
```

Inspect all intermediate representations:

```python
print(model['undirected_edges'])
print(model['pdag_edges'])
print(model['dag_edges'])
```

These represent different stages:

```text
undirected
    ↓
PDAG
    ↓
DAG
```

An unexpected direction may arise during orientation/completion even when the
undirected skeleton looks reasonable.

---

# 27. LiNGAM Fails

For:

```python
methodtype='direct-lingam'
```

or:

```python
methodtype='ica-lingam'
```

first check:

```python
print(df.dtypes)
```

The variables must be represented numerically.

Convert appropriate columns:

```python
df = df.astype(float)
```

Do not pass arbitrary strings or categorical labels directly.

---

# 28. LiNGAM Assumptions

LiNGAM should not be selected simply because the data are continuous.

The method is based on assumptions including:

```text
linear relationships
non-Gaussian disturbances
acyclic causal structure
```

If these assumptions are inappropriate, the resulting causal ordering may not
be meaningful.

---

# 29. DirectLiNGAM Causal Order

Inspect:

```python
print(model['causal_order'])
```

If the order appears unreasonable, investigate:

* Data preprocessing.
* Strong measurement noise.
* Hidden confounding.
* Nonlinear relationships.
* Gaussian error distributions.
* Model assumptions.

Do not automatically force the expected order unless domain knowledge justifies
it.

---

# 30. LiNGAM Prior Knowledge Errors

`prior_knowledge` must represent the expected matrix structure.

The documented values are:

```text
 0  → xi does not have a directed path to xj
 1  → xi has a directed path to xj
-1  → unknown
```

Before fitting, verify:

```python
print(prior_knowledge)
```

Check:

```text
number of rows == number of variables
number of columns == number of variables
```

and ensure the variable ordering matches the DataFrame column ordering expected
by the LiNGAM model.

---

# 31. Missing `root_node`

Tree-based methods and Naive Bayes require specific configuration.

For Chow-Liu:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='chow-liu',
    root_node='Cloudy'
)
```

For Naive Bayes:

```python
model = bn.structure_learning.fit(
    df,
    methodtype='naivebayes',
    root_node='Target'
)
```

If `root_node` is missing, specify it explicitly.

---

# 32. Missing `class_node` for TAN

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
    class_node='Target'
)
```

If `class_node` is omitted, TAN cannot construct the intended tree-augmented
classification structure.

---

# 33. Inference Fails: No CPDs

A common inference error occurs when the model contains a DAG but no learned
CPDs.

Check:

```python
print(model['model'].get_cpds())
```

If there are no CPDs:

```python
model = bn.parameter_learning.fit(
    model,
    df
)
```

Then retry:

```python
query = bn.inference.fit(
    model,
    variables=['Target']
)
```

The required workflow is:

```text
DAG
 ↓
Parameter learning
 ↓
CPDs
 ↓
Inference
```

---

# 34. Inference Fails: Wrong Variable Name

Before querying:

```python
print(model['model'].nodes())
```

Then check:

```python
variables=['Target']
```

and:

```python
evidence={'Temperature': 'high'}
```

Every variable must exist in the model.

Names are case-sensitive.

---

# 35. Inference Fails: Invalid Evidence State

For example:

```python
evidence={
    'Rain': 2
}
```

will fail if the model only contains:

```text
Rain = 0
Rain = 1
```

Inspect CPD states:

```python
cpd = model['model'].get_cpds('Rain')
print(cpd.state_names)
```

Then use a valid state.

---

# 36. Inference Fails: Wrong Model Type

Inference requires a Bayesian Network with learned parameters.

Check:

```python
print(type(model['model']))
```

and:

```python
print(model['model'].get_cpds())
```

If the object is only a DAG structure, perform parameter learning first.

---

# 37. Inference Returns Unexpected Probabilities

First verify:

```python
print(query.df)
```

Then inspect the evidence:

```python
print(evidence)
```

Remember that:

```text
P(Y | X)
```

is a conditional probability.

It is not automatically:

```text
P(Y | do(X))
```

Do not interpret an ordinary inference query as an intervention without an
appropriate causal framework.

---

# 38. `to_df=False`

Inference can avoid DataFrame conversion:

```python
query = bn.inference.fit(
    model,
    variables=['Target'],
    to_df=False
)
```

This can be useful when performance is important.

If:

```python
query.df
```

is `None`, check whether:

```python
to_df=False
```

was used.

---

# 39. Sampling Fails: Invalid `n`

Sampling requires:

```python
n >= 1
```

Invalid:

```python
bn.sampling(model, n=0)
```

Valid:

```python
bn.sampling(model, n=1000)
```

---

# 40. Sampling Fails: No CPDs

Synthetic sampling requires a Bayesian Network with CPDs.

Check:

```python
print(model['model'].get_cpds())
```

If empty:

```python
model = bn.parameter_learning.fit(
    model,
    df
)
```

Then:

```python
samples = bn.sampling(
    model,
    n=1000
)
```

---

# 41. Sampling Fails With Evidence

Conditional evidence is supported for:

```python
methodtype='bayes'
```

Example:

```python
samples = bn.sampling(
    model,
    n=1000,
    methodtype='bayes',
    evidence={
        'Rain': 1
    }
)
```

Gibbs sampling does not support the `evidence` parameter in the current
implementation.

Do not use:

```python
methodtype='gibbs',
evidence={'Rain': 1}
```

Instead use Bayesian rejection sampling:

```python
methodtype='bayes'
```

---

# 42. Sampling Fails: Impossible Evidence

The sampler checks whether the supplied evidence has non-zero probability.

For example:

```python
evidence={
    'Rain': 1,
    'Cloudy': 999
}
```

can fail because the state is invalid.

More subtly, a combination can be individually valid but jointly impossible
under the model.

The implementation checks the evidence before rejection sampling to avoid
waiting indefinitely for samples that cannot occur.

---

# 43. Sampling Produces Too Few Accepted Samples

Conditional rejection sampling can become inefficient when:

```text
P(evidence)
```

is very small.

For example:

```python
evidence={
    'A': rare_state,
    'B': rare_state,
    'C': rare_state
}
```

may require many generated observations before enough samples satisfy the
evidence.

This is a computational issue rather than necessarily a model error.

Consider whether the conditioning event is realistically common enough for
rejection sampling.

---

# 44. Gibbs Sampling

Use:

```python
samples = bn.sampling(
    model,
    n=1000,
    methodtype='gibbs'
)
```

Do not supply evidence.

If conditional samples are required:

```python
samples = bn.sampling(
    model,
    n=1000,
    methodtype='bayes',
    evidence=evidence
)
```

---

# 45. Sampling Output Looks Wrong

Inspect:

```python
print(samples.head())
print(samples.shape)
print(samples.dtypes)
```

Compare empirical distributions:

```python
print(samples['Target'].value_counts(normalize=True))
```

with the corresponding model probabilities.

Large differences can occur with small sample sizes.

Increase:

```python
n
```

and compare again.

---

# 46. Structure Looks Wrong but Code Runs

A successful execution does not imply a valid causal model.

Check:

```python
print(model['model_edges'])
```

Then ask:

```text
Does the structure make domain sense?
Are directions plausible?
Are important variables missing?
Are there obvious confounders?
Are there deterministic relationships?
```

Also compare alternative discovery approaches where appropriate.

For example:

```text
Hill climbing
vs
PC
vs
LiNGAM
```

Differences are informative.

---

# 47. Too Many Edges

If the graph is overly dense:

### Hill climbing

Try:

```python
max_indegree=...
```

or stronger structural constraints.

### PC

Inspect:

```python
alpha
```

and the conditional independence test.

### All methods

Check whether:

* Variables are redundant.
* Variables are strongly correlated.
* Sample size is sufficient.
* Irrelevant variables were included.
* Measurement variables encode the same underlying quantity.

Do not simply remove edges after discovery without documenting the reason.

---

# 48. Too Few Edges

If the graph is unexpectedly sparse:

Check:

```text
sample size
variable quality
missing values
constant variables
independence-test threshold
score selection
domain constraints
```

For PC, inspect:

```python
params_pc['alpha']
```

For Hill climbing, compare:

```python
scoretype='bic'
```

with another appropriate score.

---

# 49. Network Contains Cycles

A Bayesian Network DAG must be acyclic.

If an expected relationship appears to require:

```text
A → B
B → C
C → A
```

then a static DAG cannot represent that feedback loop directly.

Investigate whether:

* Variables should be time-indexed.
* The process is dynamic.
* One relationship represents a lagged effect.
* A different modeling framework is required.

Do not simply remove an edge to hide a genuine feedback process.

---

# 50. Temporal Data

If the dataset contains time-dependent variables such as:

```text
Temperature_t
Temperature_t+1
Pressure_t
Pressure_t+1
```

be explicit about temporal ordering.

A static Bayesian Network assumes an acyclic graph.

Feedback can often be represented by lagging variables:

```text
Temperature_t → Pressure_t+1
Pressure_t → Temperature_t+1
```

rather than creating a same-time cycle.

---

# 51. Highly Correlated Variables

Strongly correlated variables can make causal discovery difficult.

For example:

```text
Temperature
Heat Energy
Thermal Load
```

may contain overlapping information.

A discovery algorithm may choose one relationship over another because of
statistical or scoring differences.

Inspect correlations:

```python
print(df.corr(numeric_only=True))
```

but remember:

```text
correlation ≠ causality
```

Correlation is a diagnostic, not a causal decision rule.

---

# 52. Deterministic Relationships

If:

```text
C = A + B
```

then `C` is not an independent measurement.

Deterministic or nearly deterministic relationships can cause problems in
statistical modeling and may produce unstable structures.

Identify derived variables before causal discovery.

---

# 53. Multicollinearity in Continuous Models

For continuous Gaussian models and LiNGAM, strongly dependent predictors can
make parameter estimation and causal orientation difficult.

Inspect:

```python
df.corr(numeric_only=True)
```

and consider whether highly redundant variables should remain in the model.

Do not remove variables solely because they are correlated if they represent
meaningfully different causal mechanisms.

---

# 54. Model Is Sensitive to Randomness

Some algorithms or preprocessing steps involve randomness.

For LiNGAM, use:

```python
params_lingam={
    'random_state': 42
}
```

when supported.

For reproducible experiments, record:

```text
random seed
bnlearn version
pgmpy version
dataset version
algorithm
score/test
hyperparameters
constraints
```

Reproducibility is especially important when comparing discovered structures.

---

# 55. Dependency Errors

If an algorithm fails during import or execution, inspect the relevant
dependency.

For example, LiNGAM methods require the appropriate `lingam` installation.

Do not troubleshoot a model parameter when the actual problem is a missing
Python dependency.

Check:

```python
import lingam
print(lingam.__version__)
```

Similarly, verify core dependencies used by the selected method.

---

# 56. Version Compatibility

If code worked previously but fails after upgrading packages, inspect:

```python
import bnlearn
import pgmpy

print(bnlearn.__version__)
print(pgmpy.__version__)
```

Version changes can alter:

* Estimator APIs.
* Model classes.
* CPD behavior.
* Sampling behavior.
* Conditional independence tests.
* Search algorithms.

When reporting a reproducible bug, always include package versions.

---

# 57. Debugging With the Logger

Use:

```python
import bnlearn as bn
bn.set_logger('info')
```

for normal diagnostic output.

Increase verbosity when deeper debugging is required:

```python
bn.set_logger('debug')
```

or:

```python
bn.set_logger('trace')
```

Use:

```python
bn.set_logger('silent')
```

only when diagnostic output is intentionally suppressed.

---

# 58. Minimal Reproducible Example

When a problem cannot be isolated, reduce the dataset.

Start with:

```python
df_small = df.iloc[:100].copy()
```

and a small number of variables:

```python
df_small = df[
    [
        'A',
        'B',
        'C'
    ]
]
```

Then run:

```python
model = bn.structure_learning.fit(
    df_small,
    methodtype='hc',
    scoretype='bic'
)
```

If the small example works, progressively add:

```text
variables
rows
constraints
different scores
different algorithms
```

until the problem reappears.

---

# 59. Recommended Diagnostic Order

When debugging a complete workflow, use this order:

### Step 1 — Data

```python
print(df.shape)
print(df.dtypes)
print(df.isna().sum())
print(df.columns.tolist())
```

### Step 2 — Structure

```python
print(model['model_edges'])
print(model['adjmat'])
```

### Step 3 — CPDs

```python
print(model['model'].get_cpds())
```

### Step 4 — Inference

```python
query = bn.inference.fit(
    model,
    variables=['Target']
)
```

### Step 5 — Sampling

```python
samples = bn.sampling(
    model,
    n=100
)
```

This isolates the first failing layer.

---

# 60. Common Error → Likely Cause

| Symptom                 | Likely cause                                           |
| ----------------------- | ------------------------------------------------------ |
| Input type error        | `df` is not a DataFrame                                |
| Unknown variable        | Column/model name mismatch                             |
| Inference without CPDs  | Parameter learning not performed                       |
| Invalid evidence        | Unknown state or variable                              |
| Gibbs + evidence error  | Gibbs does not support evidence                        |
| Sampling hangs/fails    | Evidence has very low/zero probability                 |
| Gaussian score error    | Non-numeric data                                       |
| LiNGAM error            | Invalid/non-numeric input or dependency                |
| TAN error               | Missing `class_node`                                   |
| Chow-Liu error          | Missing `root_node`                                    |
| Huge runtime            | Too many variables / exhaustive search                 |
| Too many edges          | Weak constraints or dense statistical structure        |
| Too few edges           | Strong constraints, weak signal, or strict test        |
| Unexpected PC direction | PDAG orientation/completion                            |
| Unstable DAG            | Multiple structures fit similarly well                 |
| Missing CPDs            | Structure learned but parameters not learned           |
| Wrong probability       | Wrong evidence/state or incorrect model interpretation |

---

# 61. Never Fix the Symptom First

When an error occurs, avoid immediately changing multiple parameters.

Bad troubleshooting:

```text
Change alpha
Change score
Change method
Drop columns
Increase n
Disable constraints
```

all at once.

Instead:

```text
1. Reproduce error
2. Identify failing stage
3. Inspect inputs
4. Inspect model state
5. Change one thing
6. Re-run
7. Compare result
```

This makes the cause identifiable.

---

# 62. Troubleshooting Decision Tree

Use the following decision process:

```text
Does the DataFrame load?
        │
        ├── NO → Fix data loading
        │
        ▼
Are columns valid?
        │
        ├── NO → Fix names/types
        │
        ▼
Does structure learning run?
        │
        ├── NO → Check method/score/data/dependencies
        │
        ▼
Is the DAG reasonable?
        │
        ├── NO → Check assumptions/constraints/data
        │
        ▼
Are CPDs present?
        │
        ├── NO → Run parameter_learning.fit()
        │
        ▼
Does inference run?
        │
        ├── NO → Check variables/evidence/model type
        │
        ▼
Does sampling run?
        │
        ├── NO → Check method/evidence/CPDs
        │
        ▼
Validate probabilities and generated data
```

---

# 63. AI Troubleshooting Rules

When an AI agent encounters a `bnlearn` error:

1. Identify the failing function.
2. Identify the stage:

   * data
   * structure
   * parameters
   * inference
   * sampling
3. Inspect the exact input types.
4. Inspect DataFrame columns and dtypes.
5. Inspect missing values.
6. Inspect model keys.
7. Inspect `model['model']`.
8. Check whether CPDs exist before troubleshooting inference or sampling.
9. Check variable names exactly, including capitalization.
10. Check evidence variables and states.
11. Check method-specific parameters.
12. Check score/test compatibility with the data.
13. Check dependency availability for LiNGAM.
14. Check whether the selected method supports the requested operation.
15. Do not use Gibbs sampling with evidence.
16. Do not use inference on a DAG that has not been parameterized.
17. Do not use Gaussian scores on categorical string data.
18. Do not use exhaustive search on large networks.
19. Do not interpret a conditional probability as an intervention.
20. Change one parameter at a time when diagnosing behavior.
21. Reduce the problem to a minimal reproducible example when necessary.
22. Record package versions when reporting a reproducibility problem.

---

# 64. Complete Health Check

A simple model health check can be performed with:

```python
print('Model type:', type(model['model']))
print('Nodes:', list(model['model'].nodes()))
print('Edges:', list(model['model'].edges()))
print('CPDs:', len(model['model'].get_cpds()))
print('Adjacency matrix:')
print(model['adjmat'])
```

For the data:

```python
print('Shape:', df.shape)
print('Missing values:')
print(df.isna().sum())
print('Data types:')
print(df.dtypes)
```

For inference:

```python
print('Query variables:', variables)
print('Evidence:', evidence)
```

This provides a compact diagnostic snapshot.

---

# 65. Final Troubleshooting Principle

Most `bnlearn` problems can be reduced to one of five categories:

```text
1. Invalid data
2. Incompatible algorithm
3. Incorrect model structure
4. Missing or invalid parameters
5. Invalid inference/sampling request
```

The correct debugging strategy is therefore:

```text
                 DATA
                   │
             Is it valid?
                   │
                   ▼
             STRUCTURE
                   │
          Is the DAG valid?
                   │
                   ▼
             PARAMETERS
                   │
            Are CPDs learned?
                   │
                   ▼
              INFERENCE
                   │
       Are variables/evidence valid?
                   │
                   ▼
               SAMPLING
                   │
        Is the requested sampling
             method supported?
```

Do not treat a Bayesian Network as a single object with a single failure mode.

A reliable diagnosis always identifies **which layer of the Bayesian modeling
pipeline failed first**.
