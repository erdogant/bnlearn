# Inference

> **Quick API (verified bnlearn ≥ 0.14)**
>
> ```python
> query = bn.inference.fit(
>     model,                       # discrete CPDs, LinearGaussian, or CG model
>     variables=['Target'],        # list of query variables
>     evidence={'A': 1, 'B': 0},   # discrete states and/or continuous numbers
>     to_df=True,
>     elimination_order='greedy',
>     joint=True,
>     verbose=3,
>     do=None,                     # interventional assignment, e.g. {'X': 1}
> )
> # Discrete: query.df → columns = variables + 'p'
> # Continuous/CG continuous: ContinuousQueryResult with .means / .df / .text
> ```
>
> `fit()` routes to `fit_discrete` (Variable Elimination) or `fit_continuous`
> (linear-Gaussian / Conditional Gaussian) from the model type.
>
> Evidence / `do` variable names must exist in the model. A variable cannot appear
> in both `evidence` and `do`.
>
> - `evidence={'X': x}` → observational `P(Y | X = x)`
> - `do={'X': x}` → interventional `P(Y | do(X = x))` (Pearl's do-operator)
> - Continuous evidence example: `evidence={'X': 0.5}` on a linear-Gaussian model

---

Inference is the process of asking conditional (and interventional) probability
questions to a Bayesian Network.

In `bnlearn`, exact inference is performed using **Variable Elimination**
through pgmpy's `VariableElimination`. Interventions (`do`) are implemented by
running the same engine on the mutilated network.

The basic observational question is:

```text
P(Query variables | Evidence)
```

For example:

```text
P(Wet_Grass | Rain=1, Sprinkler=0, Cloudy=1)
```

The corresponding interventional question uses the `do` argument:

```text
P(Wet_Grass | do(Sprinkler=1))
```

Inference does not learn a new network.

Instead, it uses an existing Bayesian Network and its learned Conditional
Probability Distributions (CPDs) to calculate probabilities for queried
variables.

---

# 1. Inference Workflow

The complete workflow is:

```text
Data
 │
 ▼
Structure Learning
 │
 ▼
DAG
 │
 ▼
Parameter Learning
 │
 ▼
Bayesian Network + CPDs
 │
 ▼
Inference
 │
 ├── Query variables
 │
 └── Evidence
 │
 ▼
Variable Elimination
 │
 ▼
Probability distribution
 │
 ├── query
 ├── query.df
 └── query.text
```

Inference therefore normally happens **after parameter learning**.

A structure alone is not sufficient to calculate probabilities.

---

# 2. Basic Inference

The main interface is:

```python
bn.inference.fit(
    model,
    variables=None,
    evidence=None,
    do=None,
)
```

The most important arguments are:

```text
model
variables
evidence   # observational conditioning
do         # interventional assignment (Pearl's do-operator)
```

For example (observational):

```python
query = bn.inference.fit(
    model,
    variables=['Wet_Grass'],
    evidence={
        'Rain': 1,
        'Sprinkler': 0,
        'Cloudy': 1
    }
)
```

This asks:

```text
P(Wet_Grass | Rain=1, Sprinkler=0, Cloudy=1)
```

For an intervention, pass `do` instead of (or in addition to) `evidence`:

```python
query = bn.inference.fit(
    model,
    variables=['Wet_Grass'],
    do={'Sprinkler': 1},
)
# P(Wet_Grass | do(Sprinkler=1))
```

---

# 3. The Model

`model` must be a dictionary containing the Bayesian Network model.

The implementation first checks:

```python
isinstance(model, dict)
```

and expects:

```python
model['adjmat']
```

and:

```python
model['model']
```

to be available.

The adjacency matrix is used to validate variable names.

The actual Bayesian Network is extracted from:

```python
model['model']
```

---

# 4. BayesianNetwork Requirement

Inference requires a pgmpy:

```text
BayesianNetwork
```

The implementation checks whether the supplied model is a Bayesian Network.

If it is not, a warning is printed:

```text
Inference requires BayesianNetwork.
```

The warning suggests fitting the parameters first:

```python
bn.parameter_learning.fit(
    DAG,
    df,
    methodtype='bayes'
)
```

The important distinction is:

```text
DAG
    ≠
fully parameterized Bayesian Network
```

A DAG describes structure.

A Bayesian Network used for probability inference must also contain the
parameters required to define the probability distributions.

---

# 5. Structure vs Parameters

A learned structure might contain:

```text
Rain → Wet_Grass
Sprinkler → Wet_Grass
```

but inference requires the corresponding CPDs.

Conceptually:

```text
Structure
    +
CPDs
    =
Probabilistic Model
```

Inference operates on the probabilistic model.

Therefore, if the model does not contain valid CPDs, inference can fail when
`VariableElimination` is initialized.

---

# 6. Parameter Learning Before Inference

A typical workflow is:

```python
import bnlearn as bn

# Learn the structure
DAG = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic'
)

# Learn the parameters
model = bn.parameter_learning.fit(
    DAG,
    df,
    methodtype='bayes'
)

# Perform inference
query = bn.inference.fit(
    model,
    variables=['Wet_Grass'],
    evidence={'Rain': 1}
)
```

The exact parameter-learning configuration depends on the data and model.

The important requirement is that the final model contains a valid
BayesianNetwork with learned CPDs.

---

# 7. Query Variables

The `variables` argument specifies the variables whose probability
distribution should be returned.

Example:

```python
variables=['Wet_Grass']
```

asks for:

```text
P(Wet_Grass | evidence)
```

Multiple variables can also be queried:

```python
variables=[
    'Wet_Grass',
    'Rain'
]
```

This asks for:

```text
P(Wet_Grass, Rain | evidence)
```

when:

```python
joint=True
```

---

# 8. Variable Names Are Case Sensitive

The implementation validates query variables against:

```python
model['adjmat'].columns
```

Therefore:

```text
Wet_Grass
```

and:

```text
wet_grass
```

are different names.

For example:

```python
variables=['Wet_Grass']
```

is valid only if:

```text
Wet_Grass
```

exists in the model.

If a query variable does not exist, inference raises an error indicating that
the variable names must match the model.

---

# 9. Evidence

Evidence specifies observed values for variables in the network.

Example:

```python
evidence={
    'Rain': 1
}
```

represents:

```text
Rain = 1
```

The query then becomes:

```text
P(Wet_Grass | Rain=1)
```

Multiple evidence variables are supported:

```python
evidence={
    'Rain': 1,
    'Sprinkler': 0,
    'Cloudy': 1
}
```

which represents:

```text
P(Wet_Grass |
  Rain=1,
  Sprinkler=0,
  Cloudy=1)
```

---

# 10. Evidence Variable Names

Evidence variable names are also checked against the model.

The implementation validates:

```python
[*evidence.keys()]
```

against:

```python
model['adjmat'].columns
```

Therefore all evidence variable names must exactly match model variable names.

For example:

```python
evidence={'Rain': 1}
```

is valid only when `Rain` exists in the network.

---

# 11. Evidence Values

The evidence value must correspond to a valid state of the variable's CPD.

For a binary variable:

```text
Rain = 0
Rain = 1
```

might be valid.

For a categorical variable:

```text
Weather = 'sunny'
```

might be valid if the corresponding CPD uses that state.

Inference does not convert arbitrary evidence values into valid states.

The evidence must be compatible with the learned model.

---

# 12. Conditional Probability

The fundamental operation is:

```text
P(X | E)
```

where:

```text
X = queried variables
E = evidence
```

For example:

```python
query = bn.inference.fit(
    model,
    variables=['Wet_Grass'],
    evidence={'Rain': 1}
)
```

represents:

```text
P(Wet_Grass | Rain=1)
```

Variable Elimination calculates this conditional distribution from the
Bayesian Network.

---

# 13. Joint Queries

By default:

```python
joint=True
```

When multiple variables are queried, the result is a joint distribution.

Example:

```python
query = bn.inference.fit(
    model,
    variables=['Wet_Grass', 'Rain'],
    evidence={'Sprinkler': 1},
    joint=True
)
```

This represents:

```text
P(Wet_Grass, Rain | Sprinkler=1)
```

The returned distribution contains combinations of states of the queried
variables.

---

# 14. Separate Queries

Set:

```python
joint=False
```

to request separate distributions over the queried variables.

Example:

```python
query = bn.inference.fit(
    model,
    variables=['Wet_Grass', 'Rain'],
    evidence={'Sprinkler': 1},
    joint=False
)
```

Conceptually this produces separate distributions:

```text
P(Wet_Grass | Sprinkler=1)

P(Rain | Sprinkler=1)
```

rather than a joint:

```text
P(Wet_Grass, Rain | Sprinkler=1)
```

This distinction is important.

---

# 15. Joint vs Separate Distribution

Use:

```text
joint=True
```

when the relationship between multiple queried variables is required.

Use:

```text
joint=False
```

when independent marginal distributions for the queried variables are
sufficient.

Do not interpret separate marginal distributions as a joint distribution.

---

# 16. Variable Elimination

`bnlearn` uses:

```python
from pgmpy.inference import VariableElimination
```

The inference object is created as:

```python
model_infer = VariableElimination(model)
```

The query is then executed using:

```python
model_infer.query(
    variables=variables,
    evidence=evidence,
    elimination_order=elimination_order,
    joint=joint,
    show_progress=(verbose >= 3)
)
```

---

# 17. What Variable Elimination Does

A Bayesian Network represents a factorized joint distribution.

Instead of explicitly constructing the complete joint distribution, Variable
Elimination works with smaller factors.

Conceptually:

```text
CPDs
 │
 ▼
Relevant factors
 │
 ▼
Multiply factors
 │
 ▼
Eliminate hidden variables
 │
 ▼
Normalize
 │
 ▼
P(Query | Evidence)
```

This avoids explicitly constructing the complete joint distribution in many
cases.

The method therefore performs marginalization through intermediate factors.

---

# 18. Elimination Order

The argument:

```python
elimination_order='greedy'
```

controls the order in which non-query variables are eliminated.

Supported string options are:

```text
greedy
WeightedMinFill
MinNeighbors
MinWeight
MinFill
```

A list can also be supplied.

---

# 19. Default Elimination Order

The default is:

```python
elimination_order='greedy'
```

Example:

```python
query = bn.inference.fit(
    model,
    variables=['Wet_Grass'],
    evidence={'Rain': 1},
    elimination_order='greedy'
)
```

The elimination strategy affects computational efficiency.

It does not change the underlying probability represented by a correctly
executed exact inference operation.

---

# 20. Explicit Elimination Order

A list can be supplied instead of a strategy name.

When a list is supplied, it should contain all variables in the model except
the queried variables.

For example, conceptually:

```python
elimination_order=[
    'Cloudy',
    'Sprinkler'
]
```

when those are the variables that need to be eliminated.

The order can have a substantial impact on computational complexity.

---

# 21. Choosing an Elimination Order

For most use cases:

```python
elimination_order='greedy'
```

is a good default.

For performance-sensitive inference, alternative elimination-order strategies
can be evaluated.

The important principle is:

```text
Same model
+
Same query
+
Same evidence
=
same exact probability
```

while:

```text
different elimination order
```

can result in different computational cost.

---

# 22. Computational Complexity

Variable Elimination can become expensive when the network contains structures
that create large intermediate factors.

The important concept is not simply:

```text
number of nodes
```

but also the connectivity of the network and the resulting intermediate
factor sizes.

A poor elimination order can create very large factors.

Therefore, elimination order is an important performance parameter.

---

# 23. `to_df`

The inference function provides:

```python
to_df=True
```

by default.

When enabled, the inference result is converted into a pandas DataFrame:

```python
query.df
```

The conversion is performed using:

```python
bnlearn.query2df(...)
```

---

# 24. Why `to_df` Exists

The native pgmpy inference result is useful for probabilistic operations.

The DataFrame representation is convenient for:

* Inspecting probabilities.
* Filtering results.
* Plotting.
* Grouping.
* Exporting.
* Downstream pandas operations.

Example:

```python
query = bn.inference.fit(
    model,
    variables=['Wet_Grass'],
    evidence={'Rain': 1},
    to_df=True
)

print(query.df)
```

---

# 25. Performance Consideration of `to_df`

Converting the result to a DataFrame adds processing overhead.

The implementation therefore allows:

```python
to_df=False
```

When `to_df=False` and `plot=False`:

```text
query.df = None
query.text = None
```

This can be useful when only the native inference object is required.

---

# 26. `plot`

The argument:

```python
plot=False
```

controls whether an inference summary is plotted.

When:

```python
plot=True
```

the implementation also creates:

```python
query.df
```

and:

```python
query.text
```

even if:

```python
to_df=False
```

because the plotting and summary functions require the DataFrame.

Therefore:

```text
plot=True
    →
DataFrame conversion is performed
```

---

# 27. Inference Summary

When `to_df=True` or `plot=True`, the implementation calls:

```python
summarize_inference(...)
```

The resulting text is stored as:

```python
query.text
```

For example:

```python
print(query.text)
```

produces a readable summary containing:

```text
Summary for variables: [...]
Given evidence: ...

Variable outcomes:
- Variable: state (percentage)
```

---

# 28. Probability Normalization in the Summary

The summary groups:

```python
query.df
```

by each queried variable and sums the `p` column.

Conceptually:

```text
grouped = Σ p
```

for each state.

Then:

```text
percentage = grouped / total
```

where:

```text
total = Σ grouped
```

This converts the probabilities into percentages for display.

---

# 29. Multiple Query Variables in the Summary

If:

```python
variables=[
    'Wet_Grass',
    'Rain'
]
```

the summary processes each variable separately.

For each variable:

```text
group by variable state
    ↓
sum probability
    ↓
normalize
    ↓
display percentage
```

The summary therefore provides per-variable marginal summaries even when the
underlying query is joint.

---

# 30. `groupby`

The `groupby` argument is:

```python
groupby=None
```

and is passed to:

```python
bnlearn.query2df(...)
```

It is intended to group the query output by specified variable names by taking
the maximum probability for each category.

Use it when a grouped representation of the query output is desired.

Example:

```python
query = bn.inference.fit(
    model,
    variables=['Wet_Grass', 'Rain'],
    evidence={'Sprinkler': 1},
    groupby=['Wet_Grass']
)
```

The exact resulting DataFrame representation is determined by
`bnlearn.query2df`.

---

# 31. Plotting Inference Results

Set:

```python
plot=True
```

to display a bar plot for each queried variable.

Example:

```python
query = bn.inference.fit(
    model,
    variables=['Wet_Grass'],
    evidence={'Rain': 1},
    plot=True
)
```

The plot displays the normalized probability of each state.

---

# 32. Plot Interpretation

The plot uses:

```text
horizontal bars
```

with:

```text
x-axis = Percentage (%)
y-axis = variable states
```

The state labels are generated as:

```text
state_<value>
```

For example:

```text
state_0
state_1
```

for a binary variable.

Percentages are displayed at the end of each bar.

---

# 33. Important Plotting Detail

The plotting code creates a separate figure for each queried variable.

Therefore, querying:

```python
variables=[
    'Wet_Grass',
    'Rain'
]
```

with:

```python
plot=True
```

produces separate plots for the two variables.

The plot title includes:

```text
Inference Summary: <variable>
```

and the supplied evidence.

---

# 34. Verbosity

The default is:

```python
verbose=3
```

The levels are:

```text
0: None
1: ERROR
2: WARN
3: INFO
4: DEBUG
5: TRACE
```

At:

```python
verbose >= 3
```

the implementation prints:

```text
[bnlearn] >Variable Elimination.
```

and enables pgmpy's query progress display.

---

# 35. Error Handling

Inference validates the model and query before executing Variable Elimination.

Potential errors include:

### Invalid model object

```text
Input requires an object that contains the key: model.
```

### Unknown query variable

```text
variables should match names in the model
```

### Unknown evidence variable

```text
evidence should match names in the model
```

### Invalid Bayesian Network

A warning is emitted when the supplied model is not a Bayesian Network.

### Missing CPDs

Initializing `VariableElimination` can raise a `ValueError`, which is
converted into a `bnlearn` exception.

This typically indicates that the Bayesian Network does not contain the
required learned CPDs.

---

# 36. Minimal Inference Example

```python
import bnlearn as bn

model = bn.import_DAG('sprinkler')

query = bn.inference.fit(
    model,
    variables=['Wet_Grass'],
    evidence={
        'Rain': 1,
        'Sprinkler': 0,
        'Cloudy': 1
    }
)

print(query)
print(query.df)
```

This asks:

```text
P(Wet_Grass |
  Rain=1,
  Sprinkler=0,
  Cloudy=1)
```

---

# 37. Multiple-Variable Example

```python
query = bn.inference.fit(
    model,
    variables=[
        'Wet_Grass',
        'Rain'
    ],
    evidence={
        'Sprinkler': 1
    }
)

print(query)
print(query.df)
```

With the default:

```python
joint=True
```

this requests the joint distribution:

```text
P(Wet_Grass, Rain | Sprinkler=1)
```

---

# 38. Separate Distributions

```python
query = bn.inference.fit(
    model,
    variables=[
        'Wet_Grass',
        'Rain'
    ],
    evidence={
        'Sprinkler': 1
    },
    joint=False
)
```

This requests separate distributions for the queried variables.

Use this when the joint relationship between the queried variables is not
needed.

---

# 39. No DataFrame

When only the native inference object is required:

```python
query = bn.inference.fit(
    model,
    variables=['Wet_Grass'],
    evidence={'Rain': 1},
    to_df=False
)
```

The result will contain:

```python
query.df is None
query.text is None
```

unless plotting requires conversion.

---

# 40. Custom Elimination Strategy

Example:

```python
query = bn.inference.fit(
    model,
    variables=['Wet_Grass'],
    evidence={'Rain': 1},
    elimination_order='MinFill'
)
```

Available strategies include:

```text
greedy
WeightedMinFill
MinNeighbors
MinWeight
MinFill
```

Use alternative strategies when inference performance needs to be optimized
or explicitly controlled.

---

# 41. Conditional Probability Questions

The inference API can answer questions of the form:

```text
P(X)
P(X | Y)
P(X | Y, Z)
P(X, Y | Z)
```

Examples:

```python
# Marginal
bn.inference.fit(
    model,
    variables=['Wet_Grass']
)

# Conditional
bn.inference.fit(
    model,
    variables=['Wet_Grass'],
    evidence={'Rain': 1}
)

# Multiple evidence variables
bn.inference.fit(
    model,
    variables=['Wet_Grass'],
    evidence={
        'Rain': 1,
        'Sprinkler': 0
    }
)

# Joint query
bn.inference.fit(
    model,
    variables=['Wet_Grass', 'Rain'],
    evidence={'Sprinkler': 1}
)
```

---

# 42. Inference vs Intervention (do-calculus)

This distinction is essential.

**Observational inference** asks:

```text
P(Y | X=x)          # evidence={'X': x}
```

Meaning: given that we *observe* X=x, what is the probability of Y?

**Interventional inference** (Pearl's do-operator) asks:

```text
P(Y | do(X=x))      # do={'X': x}
```

Meaning: what happens to Y if we *actively set* X=x (incoming edges of X are
cut)?

These are not generally the same quantity.

## 42.1 Using `do` in bnlearn

`bn.inference.fit` accepts a `do` argument:

```python
query = bn.inference.fit(
    model,
    variables=['Wet_Grass'],
    do={'Sprinkler': 1},
)
```

Internally the code:

1. Builds the **mutilated** network with `model.do(list(do.keys()))` (incoming
   edges of the intervened nodes are removed).
2. Runs ordinary Variable Elimination on that network with the intervened
   values fixed as evidence.

Because conditioning on a mutilated graph *is* intervening, the result is exact
and works with `evidence`, `elimination_order`, and `joint`.

## 42.2 Observational vs interventional example (sprinkler)

```python
import bnlearn as bn

model = bn.import_DAG('sprinkler')

# Observational: P(Wet_Grass | Sprinkler=1) ≈ 0.927
q_obs = bn.inference.fit(
    model, variables=['Wet_Grass'], evidence={'Sprinkler': 1}
)

# Interventional: P(Wet_Grass | do(Sprinkler=1)) ≈ 0.945
q_do = bn.inference.fit(
    model, variables=['Wet_Grass'], do={'Sprinkler': 1}
)
```

Observing Sprinkler=1 is evidence that it was probably not cloudy; setting it
by intervention says nothing about the weather (Cloudy → Sprinkler is cut).

## 42.3 Combining `do` and `evidence`

```python
# P(Wet_Grass | do(Sprinkler=1), Rain=1)
q_mix = bn.inference.fit(
    model,
    variables=['Wet_Grass'],
    do={'Sprinkler': 1},
    evidence={'Rain': 1},
)
```

Rules:

* A variable **cannot** appear in both `do` and `evidence` (raises).
* Unknown node names in `do` raise.
* Interventions are labeled `do(X)=…` in `query.text`.
* `do` is the **last** parameter of `fit`, so existing positional calls stay
  compatible.

## 42.4 `evidence` alone is never an intervention

```python
evidence={'X': 1}   # observe X=1
do={'X': 1}         # do(X=1)
```

Supplying a value only in `evidence` never performs an intervention.

---

# 43. Inference Does Not Learn Causality

Inference uses the structure and CPDs already present in the Bayesian Network.

It does not determine whether an edge is causal.

For example:

```text
X → Y
```

in the model does not mean that the inference function has established that
`X` causes `Y`.

Inference answers probability questions **conditional on the supplied model**.

---

# 44. Evidence vs Prediction

Evidence can be used to update beliefs.

For example:

```text
P(Machine_Failure | Temperature=high)
```

asks for the probability of failure conditional on observing high temperature.

This is different from:

```text
P(Machine_Failure)
```

which is the marginal probability without that observation.

The Bayesian Network propagates the evidence through the network to obtain the
conditional distribution.

---

# 45. Evidence Propagation

Consider:

```text
Rain → Wet_Grass
Sprinkler → Wet_Grass
```

Observing:

```text
Rain=1
```

changes the distribution of downstream variables.

Variable Elimination propagates the evidence through the model's factors and
calculates the resulting conditional distribution.

This is the central purpose of Bayesian Network inference.

---

# 46. Exact Inference

The implementation uses exact Variable Elimination.

Therefore, unlike approximate inference methods, the result is intended to be
the exact probability implied by the supplied model, subject to numerical
precision and the capabilities of the underlying inference implementation.

The computational cost can nevertheless become large for complex networks.

---

# 47. Performance Considerations

Inference performance depends strongly on:

```text
number of variables
+
network connectivity
+
query variables
+
evidence
+
elimination order
```

For larger or densely connected networks, intermediate factors can become
large.

For performance-sensitive applications:

1. Keep the network appropriately constrained.
2. Query only the variables required.
3. Use an appropriate elimination strategy.
4. Avoid unnecessary DataFrame conversion with `to_df=False`.
5. Avoid plotting when it is not required.

---

# 48. Output Object

The return value is the pgmpy inference query object.

When DataFrame conversion is enabled, `bnlearn` attaches:

```python
query.df
```

When summary generation is enabled, it also attaches:

```python
query.text
```

Therefore the result can be inspected in multiple forms:

```python
print(query)
```

```python
print(query.df)
```

```python
print(query.text)
```

---

# 49. Recommended Workflow

For normal use:

```python
query = bn.inference.fit(
    model,
    variables=['Target'],
    evidence={'Observed_Variable': value}
)

print(query.df)
```

For human-readable output:

```python
query = bn.inference.fit(
    model,
    variables=['Target'],
    evidence={'Observed_Variable': value},
    plot=True
)

print(query.text)
```

For performance-sensitive programmatic use:

```python
query = bn.inference.fit(
    model,
    variables=['Target'],
    evidence={'Observed_Variable': value},
    to_df=False,
    plot=False
)
```

---

# 50. AI Decision Rules

When an AI agent needs to perform inference:

1. Confirm that `model` is a dictionary containing the required model object.
2. Confirm that the underlying model is a `BayesianNetwork`.
3. Confirm that the network contains learned CPDs.
4. Use `variables` to specify the query variables.
5. Use `evidence` to specify observed variable values (conditioning).
6. Use `do` to specify interventions (Pearl's do-operator).
7. Treat variable names as case-sensitive.
8. Ensure every query variable exists in the model.
9. Ensure every evidence and `do` variable exists in the model.
10. Ensure no variable appears in both `evidence` and `do`.
11. Use `joint=True` when a joint distribution over multiple query variables is
    required.
12. Use `joint=False` when separate distributions are required.
13. Use `elimination_order='greedy'` as the default unless there is a reason to
    change it.
14. Use an explicit elimination-order list only when it contains the required
    non-query variables.
15. Use `to_df=True` when a pandas representation is useful.
16. Use `to_df=False` when DataFrame conversion is unnecessary.
17. Remember that `plot=True` requires DataFrame conversion.
18. Use `query.df` for tabular probability results.
19. Use `query.text` for the generated human-readable summary (interventions
    appear as `do(X)=…`).
20. Remember that `evidence={'X': value}` represents observation, not
    intervention; use `do={'X': value}` for intervention.
21. Do not interpret conditional inference as causal intervention.
22. Do not use inference to establish causal direction.
23. Remember that inference answers questions under the supplied Bayesian
    Network and CPDs (and the mutilated graph when `do` is used).
24. For large networks, consider elimination-order and factor-size
    implications.
25. Do not compare separate marginal distributions with a joint distribution.
26. Validate the probability results before making domain-level conclusions.

---

# 51. Quick Reference

### Marginal probability

```python
query = bn.inference.fit(
    model,
    variables=['Target']
)
```

Represents:

```text
P(Target)
```

### Conditional probability

```python
query = bn.inference.fit(
    model,
    variables=['Target'],
    evidence={'Evidence': value}
)
```

Represents:

```text
P(Target | Evidence=value)
```

### Multiple evidence variables

```python
query = bn.inference.fit(
    model,
    variables=['Target'],
    evidence={
        'A': value_a,
        'B': value_b
    }
)
```

Represents:

```text
P(Target | A=value_a, B=value_b)
```

### Joint probability

```python
query = bn.inference.fit(
    model,
    variables=['A', 'B'],
    evidence={'C': value},
    joint=True
)
```

Represents:

```text
P(A, B | C=value)
```

### Separate distributions

```python
query = bn.inference.fit(
    model,
    variables=['A', 'B'],
    evidence={'C': value},
    joint=False
)
```

Returns separate distributions for `A` and `B`.

### DataFrame output

```python
query.df
```

### Text summary

```python
query.text
```

### Plot

```python
query = bn.inference.fit(
    model,
    variables=['Target'],
    evidence={'Evidence': value},
    plot=True
)
```

### Intervention (do-calculus)

```python
# P(Target | do(X=1))
query = bn.inference.fit(
    model,
    variables=['Target'],
    do={'X': 1},
)

# Combined with evidence
query = bn.inference.fit(
    model,
    variables=['Target'],
    do={'X': 1},
    evidence={'Z': 0},
)
```

---

# 52. Final Checklist

Before running inference:

* [ ] Is the model a dictionary?
* [ ] Does it contain `model`?
* [ ] Does it contain `adjmat`?
* [ ] Is the underlying model a `BayesianNetwork`?
* [ ] Does the Bayesian Network contain learned CPDs?
* [ ] Do all query variables exist?
* [ ] Do all evidence variables exist?
* [ ] Do all `do` variables exist (if interventions are used)?
* [ ] Are variable names spelled exactly and with correct case?
* [ ] Are evidence / `do` values valid states for their variables?
* [ ] Is no variable present in both `evidence` and `do`?
* [ ] Is a joint distribution required?
* [ ] Is DataFrame output required?
* [ ] Is plotting required?
* [ ] Is the default elimination order sufficient?
* [ ] Is the network large enough that elimination order may affect performance?
* [ ] Are conditional probabilities being distinguished from interventions?
* [ ] For interventions: is `do=` used (not only `evidence=`)?
* [ ] Are causal conclusions being kept separate from ordinary inference?

The central principle is:

```text
Bayesian Network + CPDs
        +
Query variables
        +
Evidence and/or do-interventions
        ↓
Variable Elimination
  (on the possibly mutilated network)
        ↓
P(Query | Evidence, do(...))
```

`bnlearn.inference.fit()` answers conditional *and* interventional probability
questions about the model you provide. Use `evidence` for observation and
`do` for intervention. It does not learn new parameters or invent causal
structure.

---

# Continuous and hybrid inference

```python
# Linear-Gaussian conditional mean
q = bn.inference.fit(model_lg, variables=['Y'], evidence={'X': 0.0})
print(q.means)

# Intervention on continuous variable
q = bn.inference.fit(model_lg, variables=['Y'], do={'X': 1.0})

# CG: continuous node given discrete evidence
q = bn.inference.fit(model_cg, variables=['torque'], evidence={'fail': 1})

# Explicit backends
q = bn.inference.fit_discrete(model_disc, variables=['Wet_Grass'], evidence={'Rain': 1})
q = bn.inference.fit_continuous(model_lg, variables=['Y'], evidence={'X': 0.5})
```
