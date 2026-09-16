# Parameter Learning

> **Quick API (verified bnlearn ≥ 0.14)**
>
> ```python
> model = bn.parameter_learning.fit(
>     model,                  # output of structure_learning.fit / make_DAG / import_DAG
>     df,                     # DataFrame
>     methodtype='bayes',     # discrete: 'bayes' | 'ml' | 'maximumlikelihood'
>                             # continuous: 'linear-gaussian' | 'lg'
>                             # mixed: 'cg' | 'conditional-gaussian'
>                             # or 'auto' / 'DBN'
>     scoretype='bdeu',       # discrete Bayesian prior (bdeu | k2 | ...)
>     smooth=None,
>     n_jobs=-1,
>     verbose=3,
> )
> ```
>
> - Always run **after** structure learning (or after supplying a DAG).
> - Discrete: `methodtype='bayes'` is preferred when counts are sparse.
> - Continuous: `methodtype='linear-gaussian'` fits a `LinearGaussianBayesianNetwork`.
> - Mixed: `methodtype='cg'` fits discrete CPTs + `continuous_cpds` (CG locals).
> - `methodtype='auto'` chooses bayes / linear-gaussian / cg from column types.
> - After this call the model is ready for inference, prediction, sampling.

---

Parameter learning estimates the parameters of the conditional probability
distributions (CPDs) for a given Bayesian Network structure.

Structure learning determines:

```text
Which variables are connected?
```

Parameter learning determines:

```text
What probabilities describe those connections?
```

The workflow is therefore:

```text
Data
  +
DAG structure
  ↓
Parameter learning
  ↓
CPDs
  ↓
Parameterized Bayesian Network
```

The implementation supports:

```text
Discrete: Maximum Likelihood (ml) and Bayesian Estimation (bayes)
Continuous: Linear-Gaussian (linear-gaussian / lg)
Hybrid: Conditional Gaussian (cg / conditional-gaussian)
Auto: choose from detected data types (auto)
Dynamic Bayesian Networks (DBNs)
```

---

# 1. Main Interface

Parameter learning is exposed through:

```python
bnlearn.parameter_learning.fit()
```

The function signature is:

```python
fit(
    model,
    df,
    methodtype='bayes',
    scoretype='bdeu',
    smooth=None,
    n_jobs=-1,
    verbose=3
)
```

The parameters are:

| Parameter    |   Default | Description                         |
| ------------ | --------: | ----------------------------------- |
| `model`      |  required | Existing Bayesian Network structure |
| `df`         |  required | DataFrame containing observations   |
| `methodtype` | `'bayes'` | Parameter-learning method           |
| `scoretype`  |  `'bdeu'` | Bayesian prior type                 |
| `smooth`     |    `None` | Pseudo-count specification          |
| `n_jobs`     |      `-1` | Parallelization setting             |
| `verbose`    |       `3` | Logging level                       |

---

# 2. Required Input

The two primary inputs are:

```python
model
df
```

The `model` provides the network structure.

The `df` provides the observations from which the CPDs are estimated.

Conceptually:

```text
model:
    DAG structure

df:
    observed variable states
```

Parameter learning combines both:

```text
DAG + data
    ↓
CPDs
```

---

# 3. The Model Must Contain an Adjacency Matrix

The implementation obtains:

```python
adjmat = model['adjmat']
```

Therefore the supplied model dictionary must contain:

```text
'adjmat'
```

The adjacency matrix defines the variables and their relationships.

The DataFrame is subsequently filtered against the variables represented in
the adjacency matrix.

---

# 4. DataFrame Filtering

For standard Bayesian Networks, the implementation calls:

```python
df = bnlearn._filter_df(
    adjmat,
    copy.deepcopy(df),
    verbose=config['verbose']
)
```

This means the parameter-learning routine does not simply use every column
from the supplied DataFrame.

It filters the data according to the variables represented by the model
structure.

This is important when the DataFrame contains additional columns that are not
part of the Bayesian Network.

---

# 5. DBN Special Case

Dynamic Bayesian Networks are handled differently.

When the method is:

```text
DBN
```

the implementation sets:

```python
df = adjmat
```

rather than filtering the DataFrame using `_filter_df()`.

Therefore, the DBN pathway should not be treated as identical to ordinary
Bayesian Network parameter learning.

---

# 6. Automatic DBN Detection

The implementation automatically changes the method to:

```text
DBN
```

when either of the following is detected:

```python
model.get('config', {}).get('method') == 'DBN'
```

or:

```python
model.get('methodtype', {}) == 'DBN'
```

The effective method therefore takes precedence over the supplied
`methodtype`.

Conceptually:

```text
User specifies methodtype
        ↓
Inspect model
        ↓
Model indicates DBN?
    ├── Yes → method = DBN
    └── No  → use requested method
```

An AI agent should check the model configuration before assuming that the
requested `methodtype` will be used.

---

# 7. Model Object Conversion

The implementation accepts a model dictionary.

If the supplied model is a dictionary:

```python
model = model['model']
```

The resulting model object is then checked.

If it is not already a pgmpy `BayesianNetwork`, it is converted using:

```python
bnlearn.to_bayesiannetwork(
    adjmat,
    verbose=config['verbose']
)
```

Therefore the workflow is:

```text
bnlearn model dictionary
        ↓
extract model
        ↓
Is it a BayesianNetwork?
    ├── Yes → use it
    └── No  → convert it
```

---

# 8. Discrete Parameter Learning

The implementation currently learns CPDs for discrete variables.

The two principal methods are:

```text
ml
maximumlikelihood
bayes
```

There is also:

```text
DBN
```

for Dynamic Bayesian Networks.

---

# 9. Maximum Likelihood Estimation

Maximum Likelihood Estimation is selected with:

```python
methodtype='ml'
```

or:

```python
methodtype='maximumlikelihood'
```

Both values select the same implementation.

The model is fitted using:

```python
model.fit(
    df,
    estimator=None
)
```

In this implementation, `estimator=None` is used to invoke the maximum
likelihood estimator according to pgmpy's API.

---

# 10. MLE Concept

For a discrete variable, MLE estimates probabilities from relative
frequencies.

Suppose a variable has states:

```text
sunny
cloudy
rainy
```

and the observed counts are:

```text
sunny   = 50
cloudy  = 30
rainy   = 20
```

Then the estimated probabilities are:

```text
P(sunny)  = 50 / 100 = 0.50
P(cloudy) = 30 / 100 = 0.30
P(rainy)  = 20 / 100 = 0.20
```

The probabilities are therefore based directly on observed counts.

---

# 11. Conditional MLE

For a node with parents, the counts are calculated separately for each
parent configuration.

Suppose:

```text
Rain → Sprinkler
```

Then the probability of `Sprinkler` is estimated conditionally:

```text
P(Sprinkler | Rain)
```

For example:

```text
Rain = True
    → count sprinkler states

Rain = False
    → count sprinkler states
```

The resulting CPD contains a separate probability distribution for each
parent configuration.

---

# 12. Multiple Parents

Suppose:

```text
A → C
B → C
```

Then parameter learning estimates:

```text
P(C | A, B)
```

separately for every combination of states of `A` and `B`.

If:

```text
A = 2 states
B = 3 states
```

then there are:

```text
2 × 3 = 6
```

possible parent configurations.

This fragmentation of the data is one of the principal weaknesses of MLE for
Bayesian Network parameter learning.

---

# 13. Sparse Parent Configurations

Even when the overall dataset is large, individual parent configurations
may contain very few observations.

For example:

```text
100,000 observations
```

may sound like a large dataset.

However, if a node has several parents with many states, the observations can
be distributed across a large number of configurations.

Therefore:

```text
large total sample
    ≠
large sample for every CPD cell
```

This is one of the reasons Bayesian parameter estimation can be useful.

---

# 14. MLE and Zero Probabilities

MLE can produce zero probabilities.

For example:

```text
Parent configuration:
    A=True
    B=False

Observed:
    C=0 → 20
    C=1 → 0
```

MLE gives:

```text
P(C=0 | A=True,B=False) = 1
P(C=1 | A=True,B=False) = 0
```

The zero is a direct consequence of the observed counts.

This can become problematic when a configuration has not been observed even
though the corresponding event is possible.

---

# 15. Bayesian Parameter Estimation

Bayesian estimation is selected using:

```python
methodtype='bayes'
```

This is the default method.

The implementation creates:

```python
DiscreteBayesianEstimator(
    prior_type=scoretype,
    equivalent_sample_size=1000,
    pseudo_counts=smooth,
    n_jobs=n_jobs
)
```

and passes the estimator to:

```python
model.fit(
    df,
    estimator=estimator
)
```

---

# 16. Bayesian Estimation Concept

Bayesian parameter estimation combines:

```text
prior information
+
observed data
```

to estimate the CPDs.

Conceptually:

```text
Prior
  ↓
Pseudo-counts
  +
Observed counts
  ↓
Posterior CPD
```

This prevents parameter estimates from depending exclusively on sparse
observations.

---

# 17. Pseudo-Counts

A useful interpretation of Bayesian estimation is that the prior contributes
pseudo-counts.

Suppose:

```text
Observed counts:

A = 10
B = 0
```

A prior can contribute additional counts:

```text
Prior:

A = 1
B = 1
```

giving effective counts:

```text
A = 11
B = 1
```

before normalization.

The exact interpretation depends on the selected prior and its configuration.

---

# 18. Bayesian vs MLE

The conceptual difference is:

```text
MLE:

observed counts
      ↓
probabilities
```

versus:

```text
Bayesian:

prior counts
      +
observed counts
      ↓
probabilities
```

Therefore Bayesian estimation can be more stable when some CPD cells contain
few observations.

---

# 19. `scoretype`

For Bayesian parameter estimation, the implementation supports:

```text
bdeu
dirichlet
k2
```

These are passed directly to:

```python
DiscreteBayesianEstimator(
    prior_type=scoretype,
    ...
)
```

The default is:

```python
scoretype='bdeu'
```

---

# 20. BDeu Parameter Estimation

Use:

```python
methodtype='bayes'
scoretype='bdeu'
```

This selects the Bayesian Dirichlet equivalent uniform prior.

Example:

```python
model_update = bn.parameter_learning.fit(
    model,
    df,
    methodtype='bayes',
    scoretype='bdeu'
)
```

The implementation uses:

```python
equivalent_sample_size=1000
```

for this Bayesian estimator.

---

# 21. Equivalent Sample Size

The Bayesian estimator is initialized with:

```python
equivalent_sample_size=1000
```

This value is fixed in the current implementation.

It represents the strength of the equivalent prior sample used by the Bayesian
estimator.

Therefore, in the current `parameter_learning.fit()` interface, there is no
separate public parameter for changing the equivalent sample size.

The exposed Bayesian controls are:

```text
scoretype
smooth
```

while the equivalent sample size remains:

```text
1000
```

---

# 22. K2 Prior

Use:

```python
methodtype='bayes'
scoretype='k2'
```

to select the K2 prior.

Conceptually, the K2 prior can be viewed as contributing a uniform
pseudo-count of one to the relevant states.

Example:

```python
model_update = bn.parameter_learning.fit(
    model,
    df,
    methodtype='bayes',
    scoretype='k2'
)
```

---

# 23. Dirichlet Prior

Use:

```python
methodtype='bayes'
scoretype='dirichlet'
```

when an explicit Dirichlet prior is desired.

Unlike BDeu and K2, the implementation requires:

```python
smooth
```

to be supplied.

If:

```python
scoretype == 'dirichlet'
```

and:

```python
smooth is None
```

the function raises:

```text
[bnlearn] >dirichlet requires "smooth" to be not None
```

---

# 24. `smooth`

The parameter:

```python
smooth=None
```

is passed to the Bayesian estimator as:

```python
pseudo_counts=smooth
```

This provides explicit pseudo-count information.

For example:

```python
model_update = bn.parameter_learning.fit(
    model,
    df,
    methodtype='bayes',
    scoretype='dirichlet',
    smooth=1
)
```

The exact shape and interpretation of `smooth` depend on the underlying
pgmpy `DiscreteBayesianEstimator` API.

The `bnlearn` implementation itself does not transform the supplied
`smooth` value before passing it to the estimator.

---

# 25. `smooth` Is Not Equivalent to Equivalent Sample Size

These are separate concepts.

The implementation supplies:

```text
equivalent_sample_size=1000
```

and:

```text
pseudo_counts=smooth
```

to the estimator.

Therefore:

```text
smooth
    ≠
equivalent_sample_size
```

Do not describe them as interchangeable parameters.

---

# 26. Dirichlet Requirement

This is an explicit validation rule:

```python
if (scoretype == 'dirichlet') and (smooth is None):
    raise Exception(...)
```

Therefore:

```python
scoretype='dirichlet'
smooth=None
```

is invalid.

An AI agent should detect this before calling the function.

---

# 27. Maximum Likelihood Does Not Use `scoretype`

When:

```python
methodtype='ml'
```

the implementation calls:

```python
model.fit(
    df,
    estimator=None
)
```

The `scoretype` parameter is therefore not used to select an MLE prior.

Do not tell users that:

```python
scoretype='bdeu'
```

changes the MLE calculation.

It only affects the Bayesian pathway.

---

# 28. Dynamic Bayesian Networks

The implementation contains a separate:

```text
DBN
```

parameter-learning path.

If the model indicates a Dynamic Bayesian Network, the method is automatically
set to:

```text
DBN
```

The implementation then performs:

```python
model.fit(
    df,
    estimator='MLE'
)
```

---

# 29. DBN Estimation

The DBN pathway uses:

```text
MLE
```

rather than the `DiscreteBayesianEstimator` used by the ordinary Bayesian
parameter-learning path.

Therefore:

```text
ordinary BN + bayes
    →
DiscreteBayesianEstimator

DBN
    →
MLE
```

Do not assume that:

```python
methodtype='bayes'
```

means the same thing when the supplied model is configured as a DBN.

The model configuration can override the requested method.

---

# 30. CPDs

The output of parameter learning is a Bayesian Network containing learned
CPDs.

The implementation iterates through:

```python
model.get_cpds()
```

after fitting.

The CPDs are printed when:

```text
verbose >= 2
```

For example:

```text
CPD of A:
...

CPD of B:
...

CPD of C:
...
```

---

# 31. CPD Interpretation

For a root node:

```text
A
```

the CPD represents:

```text
P(A)
```

For a node with parents:

```text
A → B
```

the CPD represents:

```text
P(B | A)
```

For:

```text
A → C
B → C
```

the CPD represents:

```text
P(C | A, B)
```

Together, all CPDs define the Bayesian Network distribution:

```text
P(X₁, ..., Xₙ)
    =
Πᵢ P(Xᵢ | Parents(Xᵢ))
```

---

# 32. Parameter Learning Does Not Change the Intended Structure

Parameter learning receives an existing structure.

Its primary task is:

```text
estimate parameters
```

not:

```text
discover edges
```

The adjacency matrix is retained:

```python
adjmat = model['adjmat']
```

and returned unchanged as:

```python
out['adjmat'] = adjmat
```

Therefore the conceptual workflow is:

```text
Structure learning
       ↓
DAG
       ↓
Parameter learning
       ↓
CPDs
```

Do not confuse parameter learning with structure learning.

---

# 33. Output

The function returns a dictionary:

```python
out = {}
```

with the following keys:

```text
model
adjmat
config
model_edges
structure_scores
independence_test
```

---

# 34. `model`

The returned:

```python
out['model']
```

contains the fitted Bayesian Network model.

This is the most important output because it contains the learned CPDs.

For example:

```python
model_update = bn.parameter_learning.fit(
    model,
    df
)

fitted_model = model_update['model']
```

---

# 35. `adjmat`

The returned:

```python
out['adjmat']
```

contains the original adjacency matrix:

```python
out['adjmat'] = adjmat
```

This represents the network structure used for parameter learning.

---

# 36. `config`

The returned configuration contains:

```python
config['verbose'] = verbose
config['method'] = methodtype
config['n_jobs'] = n_jobs
```

Therefore:

```python
out['config']
```

records the effective parameter-learning configuration.

For DBNs, the effective method can be overwritten to:

```text
DBN
```

---

# 37. `model_edges`

The fitted network edges are returned as:

```python
out['model_edges'] = list(model.edges())
```

This provides a convenient representation of the learned network structure.

Example:

```python
model_update['model_edges']
```

might return:

```python
[
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'D')
]
```

---

# 38. `structure_scores`

After CPD estimation, the implementation computes:

```python
out['structure_scores'] = bnlearn.structure_scores(
    out,
    df,
    verbose=verbose
)
```

Therefore parameter learning also returns structure-score information for the
resulting model.

This is a post-fitting evaluation step.

It should not be interpreted as the parameter-learning algorithm itself.

---

# 39. `independence_test`

The implementation preserves:

```python
independence_test = model.get(
    'independence_test',
    None
)
```

and returns:

```python
out['independence_test'] = independence_test
```

This means information about an independence test associated with the input
model is carried through to the output.

If no such value exists, it is:

```text
None
```

---

# 40. Verbosity

The `verbose` parameter controls output.

The implementation creates:

```python
config['verbose'] = verbose
```

The documented levels are:

```text
0: None
1: ERROR
2: WARN
3: INFO
4: DEBUG
5: TRACE
```

The default is:

```python
verbose=3
```

At `verbose >= 3`, progress messages are printed.

At `verbose >= 2`, learned CPDs are printed.

---

# 41. Parallelization

The parameter:

```python
n_jobs=-1
```

is passed to:

```python
DiscreteBayesianEstimator(
    ...
    n_jobs=config['n_jobs']
)
```

This applies to Bayesian parameter estimation.

The MLE pathway does not pass `n_jobs` to `model.fit()`.

Therefore, do not claim that `n_jobs` controls every operation performed by
`parameter_learning.fit()`.

---

# 42. Complete MLE Example

```python
import bnlearn as bn

df = bn.import_example()

model = bn.import_DAG(
    'sprinkler',
    CPD=False
)

model_update = bn.parameter_learning.fit(
    model,
    df,
    methodtype='ml'
)
```

The CPDs are estimated using maximum likelihood.

---

# 43. Complete Bayesian Example

```python
import bnlearn as bn

df = bn.import_example()

model = bn.import_DAG(
    'sprinkler',
    CPD=False
)

model_update = bn.parameter_learning.fit(
    model,
    df,
    methodtype='bayes',
    scoretype='bdeu'
)
```

This uses Bayesian parameter estimation with:

```text
prior_type = bdeu
equivalent_sample_size = 1000
pseudo_counts = None
```

---

# 44. Dirichlet Example

```python
model_update = bn.parameter_learning.fit(
    model,
    df,
    methodtype='bayes',
    scoretype='dirichlet',
    smooth=1
)
```

The `smooth` value is passed as:

```python
pseudo_counts=1
```

Without `smooth`, the function raises an exception.

---

# 45. K2 Example

```python
model_update = bn.parameter_learning.fit(
    model,
    df,
    methodtype='bayes',
    scoretype='k2'
)
```

This selects the K2 prior through:

```python
DiscreteBayesianEstimator(
    prior_type='k2',
    equivalent_sample_size=1000,
    ...
)
```

---

# 46. Inspect the CPDs

After fitting:

```python
model_update = bn.parameter_learning.fit(
    model,
    df
)
```

retrieve the fitted model:

```python
fitted_model = model_update['model']
```

Then inspect:

```python
fitted_model.get_cpds()
```

or individual CPDs through the Bayesian Network API.

---

# 47. Complete Workflow

The intended workflow is:

```text
1. Obtain data
        ↓
2. Define or learn DAG
        ↓
3. Provide DAG + data
        ↓
4. Select parameter-learning method
        ↓
5. Estimate CPDs
        ↓
6. Inspect CPDs
        ↓
7. Evaluate resulting model
```

For example:

```python
import bnlearn as bn

df = bn.import_example()

model = bn.import_DAG(
    'sprinkler',
    CPD=False
)

model_update = bn.parameter_learning.fit(
    model,
    df,
    methodtype='bayes',
    scoretype='bdeu'
)

bn.plot(model_update)
```

---

# 48. MLE vs Bayesian Estimation

| Property               | MLE                        | Bayesian                         |
| ---------------------- | -------------------------- | -------------------------------- |
| Method                 | `ml` / `maximumlikelihood` | `bayes`                          |
| Uses prior             | No                         | Yes                              |
| Based on counts        | Yes                        | Yes                              |
| Pseudo-counts          | No                         | Yes                              |
| Sparse CPD cells       | Can be unstable            | More robust                      |
| `scoretype`            | Not used                   | Used                             |
| `smooth`               | Not used                   | Passed as `pseudo_counts`        |
| Equivalent sample size | Not used                   | `1000` in current implementation |

---

# 49. Choosing the Method

Use:

```text
methodtype='ml'
```

when:

* Maximum likelihood estimation is desired.
* You want probabilities based directly on observed frequencies.
* Sufficient observations exist for the relevant parent configurations.

Use:

```text
methodtype='bayes'
```

when:

* Sparse conditional counts are a concern.
* Prior information is desired.
* Smoothing is useful.
* A Bayesian Dirichlet prior is appropriate.

Use:

```text
DBN
```

when the supplied model is a Dynamic Bayesian Network.

---

# 50. Choosing the Bayesian Prior

For:

```text
methodtype='bayes'
```

choose:

```text
bdeu
```

for the default Bayesian Dirichlet equivalent uniform approach.

Choose:

```text
k2
```

for the K2 prior.

Choose:

```text
dirichlet
```

when explicit pseudo-count specification is required.

For `dirichlet`, always provide:

```python
smooth=...
```

---

# 51. Common Mistakes

### Mistake 1 — Calling parameter learning without a structure

Parameter learning requires a model structure.

Incorrect:

```python
bn.parameter_learning.fit(
    None,
    df
)
```

Correct:

```python
model = bn.import_DAG(
    'sprinkler',
    CPD=False
)

bn.parameter_learning.fit(
    model,
    df
)
```

---

### Mistake 2 — Assuming parameter learning discovers edges

It does not.

The supplied adjacency matrix defines the structure.

Parameter learning estimates the CPDs associated with that structure.

---

### Mistake 3 — Using `scoretype` with MLE

`scoretype` controls the Bayesian estimator.

It does not alter:

```python
methodtype='ml'
```

---

### Mistake 4 — Using Dirichlet without `smooth`

This is explicitly invalid:

```python
scoretype='dirichlet'
smooth=None
```

Use:

```python
scoretype='dirichlet'
smooth=<value>
```

---

### Mistake 5 — Confusing `smooth` and equivalent sample size

The implementation passes:

```text
equivalent_sample_size=1000
```

and:

```text
pseudo_counts=smooth
```

These are distinct estimator inputs.

---

### Mistake 6 — Assuming `n_jobs` affects MLE

`n_jobs` is supplied to the Bayesian estimator.

It is not passed to the MLE `model.fit()` call.

---

### Mistake 7 — Assuming DBN uses the same Bayesian estimator

The DBN pathway uses:

```python
model.fit(
    df,
    estimator='MLE'
)
```

rather than `DiscreteBayesianEstimator`.

---

# 52. AI Decision Rules

When an AI agent needs to perform parameter learning:

1. Confirm that a model structure is available.
2. Confirm that the model contains an `adjmat`.
3. Treat the adjacency matrix as the structure to parameterize.
4. Do not perform structure discovery during parameter learning.
5. Check whether the model is configured as a DBN.
6. If it is a DBN, recognize that the implementation overrides the method to
   `DBN`.
7. For ordinary discrete networks, use `ml`, `maximumlikelihood`, or `bayes`.
8. Use `ml` when direct relative-frequency estimation is desired.
9. Use `bayes` when prior-based estimation or smoothing is desired.
10. For Bayesian estimation, use `bdeu`, `k2`, or `dirichlet`.
11. Remember that `bdeu` is the default Bayesian prior.
12. Remember that the implementation uses `equivalent_sample_size=1000`.
13. Remember that `smooth` is passed as `pseudo_counts`.
14. If `scoretype='dirichlet'`, require `smooth` to be non-None.
15. Do not interpret `scoretype` as relevant to MLE.
16. Do not confuse `smooth` with equivalent sample size.
17. Remember that DBN parameter learning uses MLE in this implementation.
18. Inspect the learned CPDs after fitting.
19. Use `model_update['model']` to access the fitted Bayesian Network.
20. Use `model_update['model_edges']` to inspect the resulting edges.
21. Remember that the output also contains `adjmat`, `config`,
    `structure_scores`, and `independence_test`.
22. Do not claim that `n_jobs` parallelizes the MLE pathway.
23. Treat sparse parent configurations as an important reason to consider
    Bayesian estimation.
24. Do not interpret Bayesian smoothing as additional observed data.
25. Keep parameter estimation conceptually separate from causal interpretation.

---

# 53. Quick Reference

### Maximum Likelihood

```python
model_update = bn.parameter_learning.fit(
    model,
    df,
    methodtype='ml'
)
```

### Maximum Likelihood — long name

```python
model_update = bn.parameter_learning.fit(
    model,
    df,
    methodtype='maximumlikelihood'
)
```

### Bayesian BDeu

```python
model_update = bn.parameter_learning.fit(
    model,
    df,
    methodtype='bayes',
    scoretype='bdeu'
)
```

### Bayesian K2

```python
model_update = bn.parameter_learning.fit(
    model,
    df,
    methodtype='bayes',
    scoretype='k2'
)
```

### Bayesian Dirichlet

```python
model_update = bn.parameter_learning.fit(
    model,
    df,
    methodtype='bayes',
    scoretype='dirichlet',
    smooth=1
)
```

### Inspect CPDs

```python
model_update['model'].get_cpds()
```

### Inspect edges

```python
model_update['model_edges']
```

---

# 54. Final Checklist

Before parameter learning:

* [ ] Is a Bayesian Network structure available?
* [ ] Does the model contain `adjmat`?
* [ ] Does the DataFrame contain the required variables?
* [ ] Is the network discrete?
* [ ] Is the model actually a DBN?
* [ ] Should MLE or Bayesian estimation be used?
* [ ] If Bayesian, should the prior be `bdeu`, `k2`, or `dirichlet`?
* [ ] If using `dirichlet`, is `smooth` supplied?
* [ ] Are sparse parent configurations likely?
* [ ] Is `n_jobs` being used with Bayesian estimation?
* [ ] Have the resulting CPDs been inspected?
* [ ] Has the fitted model been retrieved from `model_update['model']`?
* [ ] Are parameter estimation and structure learning being treated as
  separate steps?
* [ ] Are the learned parameters being distinguished from causal claims?

---

# Linear-Gaussian and Conditional-Gaussian parameter learning

```python
# Pure continuous
model = bn.parameter_learning.fit(DAG, df, methodtype='linear-gaussian')
# model['model'] is LinearGaussianBayesianNetwork

# Mixed / Conditional Gaussian
model = bn.parameter_learning.fit(DAG, df, methodtype='cg')
# model['continuous_cpds'] holds configuration-specific linear Gaussians
# model['model'] is the discrete DiscreteBayesianNetwork sub-model (or None)

# Automatic selection
model = bn.parameter_learning.fit(DAG, df, methodtype='auto')
```

Discrete result keys remain stable: `continuous_cpds` is only attached for CG fits.
