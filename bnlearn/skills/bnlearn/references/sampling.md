# Sampling

> **Quick API (verified bnlearn ≥ 0.14)**
>
> ```python
> df_samples = bn.sampling(
>     DAG,                     # model that already contains CPDs / LG / CG params
>     n=1000,
>     methodtype='bayes',      # discrete: 'bayes' | 'gibbs'
>                              # continuous: 'linear-gaussian' | 'lg'
>                              # mixed: 'cg' | 'conditional-gaussian'
>                              # or 'auto'
>     evidence=None,           # optional conditioning dict
>     do=None,                 # optional interventions (LG / CG)
>     seed=None,               # continuous / CG sampling
> )
> ```
>
> Rules enforced by the library:
> - `methodtype='gibbs'` does **not** accept `evidence`.
> - Discrete: unknown evidence variables / invalid states / impossible evidence → `ValueError`.
> - Continuous / CG: use `methodtype='linear-gaussian'`, `'cg'`, or `'auto'`.

---

Sampling generates synthetic data from a Bayesian Network.

In `bnlearn`, sampling uses the probability distributions encoded in the
network's Conditional Probability Distributions (CPDs) to generate new
observations.

The function supports:

```text
bayes            – discrete forward / rejection sampling (optional evidence)
gibbs            – discrete Gibbs (no evidence)
linear-gaussian  – LinearGaussianBayesianNetwork.simulate (optional do / evidence)
cg               – Conditional Gaussian hybrid sampling
auto             – detect model type
```

---

# 1. Sampling Workflow

Sampling operates on a Bayesian Network that already contains CPDs.

The typical workflow is:

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
Sampling
 │
 ├── Forward Sampling
 │
 ├── Rejection Sampling
 │
 └── Gibbs Sampling
 │
 ▼
Synthetic DataFrame
```

The output is a pandas DataFrame containing the generated samples.

---

# 2. Basic Usage

The simplest example is:

```python
import bnlearn as bn

DAG = bn.import_DAG('sprinkler')

df = bn.sampling(
    DAG,
    n=1000,
    methodtype='bayes'
)
```

This generates:

```text
1000 synthetic observations
```

from the joint distribution represented by the Bayesian Network.

---

# 3. Function Signature

```python
bn.sampling(
    DAG,
    n=1000,
    methodtype='bayes',
    evidence=None
)
```

Parameters:

| Parameter    | Type             |   Default | Description                                                            |
| ------------ | ---------------- | --------: | ---------------------------------------------------------------------- |
| `DAG`        | `dict`           |  required | Dictionary containing the Bayesian Network and adjacency matrix.       |
| `n`          | `int`            |    `1000` | Number of samples to generate.                                         |
| `methodtype` | `str`            | `'bayes'` | Sampling method: `'bayes'` or `'gibbs'`.                               |
| `evidence`   | `dict` or `None` |    `None` | Evidence used for conditional sampling. Supported only with `'bayes'`. |
---

# 4. Input Model Requirement

The input must contain a Bayesian Network.

The implementation checks:

```python
DAG['model']
```

and requires its type to contain:

```text
BayesianNetwork
```

If the model does not contain a Bayesian Network, sampling raises:

```text
The input model (DAG) must contain BayesianNetwork.
```

Therefore, a DAG containing only edges is not sufficient.

---

# 5. CPDs Are Required

The Bayesian Network must contain Conditional Probability Distributions.

The implementation explicitly checks:

```python
len(DAG['model'].get_cpds())
```

If no CPDs are present, sampling fails.

The error explains that the supplied DAG contains only edges and recommends
learning or specifying the CPDs.

For example:

```python
DAG = bn.parameter_learning.fit(
    DAG,
    df
)
```

should be performed before sampling when the CPDs have not yet been learned.

The fundamental requirement is:

```text
Bayesian Network
        +
CPDs
        ↓
Sampling
```

---

# 6. Sampling Is Not Structure Learning

Sampling does not learn the structure of a Bayesian Network.

It does not determine:

```text
X → Y
```

or:

```text
Y → X
```

Instead, it uses the structure and CPDs already contained in the model.

Sampling therefore answers:

```text
What synthetic observations could be generated
from this probabilistic model?
```

It does not answer:

```text
What is the correct network structure?
```

---

# 7. Sampling Is Not Inference

Sampling and inference have different purposes.

Inference calculates probabilities such as:

```text
P(Y | X=x)
```

Sampling generates observations from the model.

For example:

```python
df = bn.sampling(
    model,
    n=1000
)
```

generates synthetic observations.

By contrast:

```python
query = bn.inference.fit(
    model,
    variables=['Y'],
    evidence={'X': x}
)
```

calculates a probability distribution.

Sampling therefore produces **data**, while inference produces **probability
distributions**.

---

# 8. Sampling Methods

Two methods are supported:

```text
methodtype='bayes'
methodtype='gibbs'
```

They use different sampling procedures.

## Bayesian sampling

```text
bayes
```

uses:

```python
BayesianModelSampling
```

and supports:

* Forward sampling.
* Rejection sampling when evidence is supplied.

## Gibbs sampling

```text
gibbs
```

uses:

```python
GibbsSampling
```

and does not support evidence.

---

# 9. Bayesian Forward Sampling

When:

```python
methodtype='bayes'
```

and:

```python
evidence=None
```

the implementation uses:

```python
BayesianModelSampling(DAG['model'])
```

followed by:

```python
forward_sample(
    size=n,
    seed=None,
    show_progress=logger.isEnabledFor(logging.INFO)
)
```

This is unconditional forward sampling.

---

# 10. Forward Sampling

Forward sampling generates observations according to the Bayesian Network's
factorization.

Conceptually:

```text
Root variables
      │
      ▼
Sample root states
      │
      ▼
Sample child variables
conditioned on their parents
      │
      ▼
Continue through the network
      │
      ▼
Complete synthetic observation
```

The generated rows therefore follow the probability distributions represented
by the model.

---

# 11. Forward Sampling Example

```python
import bnlearn as bn

DAG = bn.import_DAG('sprinkler')

df = bn.sampling(
    DAG,
    n=1000,
    methodtype='bayes'
)
```

The result:

```python
type(df)
```

is:

```text
pandas.DataFrame
```

and contains synthetic observations for the variables in the model.

---

# 12. Number of Samples

The number of requested observations is controlled by:

```python
n
```

For example:

```python
df = bn.sampling(
    DAG,
    n=100
)
```

requests 100 samples.

Similarly:

```python
df = bn.sampling(
    DAG,
    n=10000
)
```

requests 10,000 samples.

The implementation requires:

```text
n >= 1
```

---

# 13. Invalid Sample Size

If:

```python
n <= 0
```

the function raises:

```text
Number of samples (n) must be 1 or larger!
```

Therefore:

```python
n=0
```

and:

```python
n=-1
```

are invalid.

---

# 14. Conditional Sampling

Conditional sampling is activated by supplying:

```python
evidence
```

with:

```python
methodtype='bayes'
```

For example:

```python
df = bn.sampling(
    model,
    n=100,
    methodtype='bayes',
    evidence={
        'Rain': 1,
        'Cloudy': 0
    }
)
```

This requests samples conditioned on:

```text
Rain = 1
Cloudy = 0
```

Every returned sample is expected to be consistent with the supplied evidence.

---

# 15. Conditional Sampling Uses Rejection Sampling

When evidence is supplied, the `bayes` method does **not** use ordinary
forward sampling.

Instead, it uses:

```python
rejection_sample(...)
```

The process is:

```text
Bayesian Network + CPDs
          │
          ▼
Generate candidate samples
          │
          ▼
Check evidence
          │
      ┌───┴───┐
      │       │
    match   reject
      │
      ▼
keep sample
      │
      ▼
repeat until n samples
```

Therefore the returned samples satisfy the requested evidence.

---

# 16. Evidence Format

Evidence must be a dictionary:

```python
{
    'variable': state
}
```

For example:

```python
evidence={
    'Rain': 1
}
```

Multiple variables are supported:

```python
evidence={
    'Rain': 1,
    'Cloudy': 0
}
```

The keys must correspond exactly to variables in the model.

Variable names are case-sensitive.

---

# 17. Evidence Validation

Before rejection sampling begins, the evidence is converted using:

```python
_evidence_as_states(
    evidence,
    DAG['model']
)
```

This performs several checks.

First, `evidence` must be a dictionary.

For example:

```python
evidence=['Rain', 1]
```

is invalid.

The expected form is:

```python
evidence={'Rain': 1}
```

---

# 18. Evidence Variable Validation

Every evidence variable must exist in the Bayesian Network.

For example, if the model contains:

```text
Rain
Cloudy
Sprinkler
Wet_Grass
```

then:

```python
evidence={
    'Rain': 1
}
```

is valid.

But:

```python
evidence={
    'Temperature': 'high'
}
```

fails if `Temperature` is not a model node.

The implementation raises an error containing the unknown variables and
available nodes.

---

# 19. Evidence State Validation

The evidence state must also exist in the variable's CPD state space.

The implementation obtains valid states from:

```python
model.get_cpds(var).state_names[var]
```

For example, if:

```text
Rain = 0
Rain = 1
```

are the valid states, then:

```python
evidence={'Rain': 1}
```

is valid.

But:

```python
evidence={'Rain': 2}
```

is invalid if state `2` does not exist.

---

# 20. Why Evidence Validation Matters

Rejection sampling can become problematic when the requested evidence can never
occur.

For example:

```text
P(Rain=1, Cloudy=0) = 0
```

means that no valid sample can satisfy the requested evidence.

Without detecting this condition, rejection sampling could continue indefinitely
waiting for a sample that cannot occur.

`bnlearn` therefore checks whether the evidence has non-zero probability before
starting rejection sampling.

---

# 21. Impossible Evidence

The implementation uses:

```python
_evidence_is_possible(
    evidence,
    DAG['model']
)
```

to check whether the evidence is possible under the model.

If the evidence has zero probability, sampling raises:

```text
evidence ... has zero probability under the model.
Rejection sampling cannot produce matching samples.
```

This is an important safety mechanism.

It prevents the sampler from waiting indefinitely for impossible evidence.

---

# 22. How Evidence Possibility Is Checked

The implementation creates:

```python
VariableElimination(model)
```

and evaluates the evidence incrementally.

For each evidence variable:

```text
query variable
      │
      ▼
condition on previously observed evidence
      │
      ▼
obtain distribution
      │
      ▼
check requested state probability
```

If a requested state has probability zero at any stage, the complete evidence
is considered impossible.

---

# 23. Avoiding Numerical Underflow

The implementation deliberately checks individual conditional factors rather
than multiplying probabilities of all evidence states directly.

The reason is that multiplying many very small probabilities can result in
numerical underflow.

For example, a probability can be extremely small without being zero.

The implementation therefore distinguishes:

```text
very unlikely
```

from:

```text
impossible
```

by checking the relevant conditional distribution directly.

---

# 24. Conditional Sampling Example

```python
import bnlearn as bn

model = bn.import_DAG('sprinkler')

df = bn.sampling(
    model,
    n=100,
    methodtype='bayes',
    evidence={
        'Rain': 1,
        'Cloudy': 0
    }
)
```

The resulting DataFrame contains samples consistent with:

```text
Rain = 1
Cloudy = 0
```

The remaining variables are sampled according to the conditional distribution
implied by the model.

---

# 25. Gibbs Sampling

The alternative method is:

```python
methodtype='gibbs'
```

For example:

```python
df = bn.sampling(
    model,
    n=100,
    methodtype='gibbs'
)
```

The implementation creates:

```python
GibbsSampling(DAG['model'])
```

and calls:

```python
gibbs.sample(
    size=n,
    seed=None
)
```

---

# 26. Gibbs Sampling Characteristics

Gibbs sampling generates samples using a Markov Chain Monte Carlo approach.

The important implementation-level distinction is:

```text
bayes
    → BayesianModelSampling
    → forward sampling
    → rejection sampling with evidence

gibbs
    → GibbsSampling
    → Gibbs sampling
    → no evidence support
```

---

# 27. Gibbs Sampling Does Not Support Evidence

This is an explicit restriction in the implementation.

The following is invalid:

```python
df = bn.sampling(
    model,
    n=100,
    methodtype='gibbs',
    evidence={'Rain': 1}
)
```

The function raises:

```text
Gibbs sampling does not support conditioning on evidence.
```

The error explicitly recommends using:

```python
methodtype='bayes'
```

with:

```python
evidence=...
```

for conditional sampling.

---

# 28. Choosing Between Bayesian and Gibbs Sampling

Use:

```python
methodtype='bayes'
```

when:

* Standard synthetic data generation is required.
* Evidence conditioning is required.
* Direct Bayesian forward sampling is appropriate.

Use:

```python
methodtype='gibbs'
```

when:

* Gibbs sampling is specifically desired.
* No evidence conditioning is required.

For conditional sampling, the correct implementation choice is:

```python
methodtype='bayes'
evidence={...}
```

not:

```python
methodtype='gibbs'
evidence={...}
```

---

# 29. Sampling From a Learned Model

A complete workflow is:

```python
import bnlearn as bn

# Load example data
df = bn.import_example('sprinkler')

# Define structure
edges = [
    ('Cloudy', 'Sprinkler'),
    ('Cloudy', 'Rain'),
    ('Sprinkler', 'Wet_Grass'),
    ('Rain', 'Wet_Grass')
]

# Create Bayesian DAG
DAG = bn.make_DAG(
    edges,
    methodtype='bayes'
)

# Learn CPDs
model = bn.parameter_learning.fit(
    DAG,
    df,
    methodtype='bayes'
)

# Generate synthetic data
synthetic_df = bn.sampling(
    model,
    n=1000,
    methodtype='bayes'
)
```

The critical sequence is:

```text
DAG
 ↓
parameter_learning.fit()
 ↓
Bayesian Network + CPDs
 ↓
sampling()
```

---

# 30. Sampling From an Imported Model

If an imported model already contains CPDs:

```python
model = bn.import_DAG('sprinkler')
```

can be passed directly to sampling:

```python
df = bn.sampling(
    model,
    n=1000,
    methodtype='bayes'
)
```

The requirement remains that the model must contain the required Bayesian
Network and CPDs.

---

# 31. Returned DataFrame

The function returns:

```python
pd.DataFrame
```

The result is the synthetic dataset generated by the selected sampler.

For example:

```python
df = bn.sampling(
    model,
    n=1000
)

print(df.head())
```

The DataFrame contains columns corresponding to variables in the Bayesian
Network.

---

# 32. Sample Size and Returned Rows

The requested sample count is:

```python
n
```

For example:

```python
n=1000
```

requests 1000 samples.

For unconditional forward sampling and Gibbs sampling, the sampler is called
with:

```python
size=n
```

For conditional Bayesian sampling, rejection sampling continues until the
requested number of matching samples is obtained.

Therefore, conditional sampling can require substantially more candidate
samples internally than the number of rows returned.

---

# 33. Conditional Sampling Efficiency

Rejection sampling can be inefficient when the evidence is unlikely.

Suppose:

```text
P(Evidence) = 0.001
```

Only a small fraction of candidate samples will satisfy the evidence.

The sampler may therefore need to generate many candidates before obtaining the
requested number of accepted samples.

This means:

```text
rare evidence
    →
more rejected samples
    →
longer sampling time
```

The implementation checks that the probability is non-zero, but non-zero does
not mean computationally efficient.

---

# 34. Evidence With Multiple Variables

Multiple evidence variables can be supplied:

```python
df = bn.sampling(
    model,
    n=1000,
    evidence={
        'Rain': 1,
        'Cloudy': 0,
        'Sprinkler': 1
    }
)
```

The joint evidence must be possible under the model.

The sampler then returns samples satisfying all specified conditions.

---

# 35. Sampling and Probability Distributions

The synthetic data is generated from the distributions encoded in the CPDs.

Conceptually:

```text
CPDs
 │
 ▼
Joint probability distribution
 │
 ▼
Sampling
 │
 ▼
Synthetic observations
```

With evidence:

```text
CPDs
 │
 ▼
Conditional distribution
 │
 ▼
Rejection sampling
 │
 ▼
Synthetic observations satisfying evidence
```

---

# 36. Synthetic Does Not Mean Independent

The generated observations are not intended to destroy the dependencies in
the Bayesian Network.

For example:

```text
Cloudy → Rain
Cloudy → Sprinkler
Rain → Wet_Grass
Sprinkler → Wet_Grass
```

The generated data should reflect the dependencies encoded by these edges and
their CPDs.

Therefore, sampling from a Bayesian Network is useful for generating synthetic
datasets that preserve the probabilistic structure represented by the model.

---

# 37. Sampling Does Not Validate the Model

Sampling from a model successfully does not prove that the model is correct.

A Bayesian Network can generate synthetic observations even if:

* The structure is incorrect.
* The CPDs are poorly estimated.
* The original data does not adequately represent the domain.

Sampling reproduces the behavior of the supplied model.

It does not independently validate that model.

---

# 38. Randomness and Reproducibility

The implementation passes:

```python
seed=None
```

to Bayesian forward sampling and rejection sampling.

Gibbs sampling is also called with:

```python
seed=None
```

Therefore, the current implementation does not expose a public `seed`
parameter through `bn.sampling()`.

Consequently, users should not assume that repeated calls will return identical
samples.

---

# 39. Logging

Control diagnostic output with:

```python
import bnlearn as bn
bn.set_logger('info')     # default — sampling progress messages
bn.set_logger('silent')   # suppress log output
```

At INFO level the implementation logs a description of the sampling operation, for example:

```text
Bayesian forward sampling for 1000 samples..
```

or:

```text
Bayesian rejection sampling for 100 samples conditioned on 2 evidence variable(s)..
```

or:

```text
Gibbs sampling for 100 samples..
```

---

# 40. Progress Display

For Bayesian forward sampling:

```python
show_progress=logger.isEnabledFor(logging.INFO)
```

is passed to pgmpy.

The same applies to Bayesian rejection sampling.

Gibbs sampling does not pass a progress-display argument in this
implementation.

---

# 41. Unsupported Sampling Method

Only these methods are supported:

```text
bayes
gibbs
```

For example:

```python
df = bn.sampling(
    model,
    n=100,
    methodtype='mcmc'
)
```

is invalid.

The function raises an error indicating that the method is unknown and
recommends:

```text
bayes
```

or:

```text
gibbs
```

---

# 42. Internal Evidence Representation

The public API uses:

```python
evidence={
    'Rain': 1
}
```

Internally this is converted to pgmpy's `State` representation.

The implementation creates:

```python
State(
    variable,
    state
)
```

for each evidence pair.

Therefore, users should provide the simple dictionary interface and should not
need to construct `State` objects themselves.

---

# 43. Internal Evidence Validation Sequence

Conditional Bayesian sampling follows this sequence:

```text
evidence dictionary
        │
        ▼
_evidence_as_states()
        │
        ├── Is evidence a dict?
        │
        ├── Do variables exist?
        │
        └── Are states valid?
        │
        ▼
_evidence_is_possible()
        │
        ├── Is evidence probability zero?
        │
        └── Is evidence possible?
        │
        ▼
rejection_sample()
        │
        ▼
DataFrame
```

This validation happens before rejection sampling.

---

# 44. Impossible vs Rare Evidence

These cases must be distinguished.

### Impossible

```text
P(Evidence) = 0
```

The implementation raises an error.

### Rare but possible

```text
P(Evidence) > 0
```

The implementation proceeds with rejection sampling.

However, rare evidence can make rejection sampling slow.

Therefore:

```text
zero probability
    →
error

very small probability
    →
valid but potentially expensive
```

---

# 45. DataFrame Compatibility Patch

The module contains a patch for:

```python
pd.DataFrame.from_records
```

The patched implementation checks whether the supplied data is already a
DataFrame.

If it is:

```python
isinstance(data, pd.DataFrame)
```

the DataFrame is returned directly.

Otherwise, the original pandas `from_records` implementation is called.

This patch exists at module level and is used to ensure that DataFrame inputs
are handled correctly through the relevant pgmpy sampling code paths.

---

# 46. Practical Examples

## Unconditional Bayesian sampling

```python
df = bn.sampling(
    model,
    n=1000,
    methodtype='bayes'
)
```

Use this for ordinary synthetic data generation.

---

## Conditional Bayesian sampling

```python
df = bn.sampling(
    model,
    n=1000,
    methodtype='bayes',
    evidence={
        'Rain': 1,
        'Cloudy': 0
    }
)
```

Use this when every generated observation should satisfy the specified
evidence.

---

## Gibbs sampling

```python
df = bn.sampling(
    model,
    n=1000,
    methodtype='gibbs'
)
```

Use this when Gibbs sampling is desired and no evidence is required.

---

# 47. Common Mistakes

### Mistake 1: Sampling from a DAG without CPDs

```python
DAG = bn.make_DAG(edges)

df = bn.sampling(DAG)
```

This fails if the resulting Bayesian Network has no CPDs.

Learn or specify the CPDs first.

---

### Mistake 2: Using evidence with Gibbs

```python
bn.sampling(
    model,
    methodtype='gibbs',
    evidence={'Rain': 1}
)
```

This is not supported.

Use:

```python
bn.sampling(
    model,
    methodtype='bayes',
    evidence={'Rain': 1}
)
```

instead.

---

### Mistake 3: Unknown evidence variable

```python
evidence={
    'UnknownVariable': 1
}
```

This fails because the variable does not exist in the model.

---

### Mistake 4: Invalid state

If the model only supports:

```text
Rain = 0
Rain = 1
```

then:

```python
evidence={
    'Rain': 2
}
```

is invalid.

---

### Mistake 5: Impossible evidence

Evidence with zero probability cannot produce samples.

The implementation detects this before rejection sampling.

---

### Mistake 6: Expecting conditional inference from ordinary sampling

This:

```python
bn.sampling(
    model,
    n=1000
)
```

does not calculate:

```text
P(X | E)
```

It generates unconditional synthetic observations.

For conditional samples, explicitly provide:

```python
evidence=...
```

---

# 48. Sampling vs Conditional Sampling

The distinction is:

### Unconditional

```python
bn.sampling(
    model,
    n=1000,
    methodtype='bayes'
)
```

Conceptually:

```text
X ~ P(X)
```

### Conditional

```python
bn.sampling(
    model,
    n=1000,
    methodtype='bayes',
    evidence={'E': value}
)
```

Conceptually:

```text
X ~ P(X | E=value)
```

The conditional version is implemented using rejection sampling.

---

# 49. Sampling vs Intervention

Sampling with evidence should not be interpreted as causal intervention.

For example:

```python
bn.sampling(
    model,
    evidence={'Temperature': 'high'}
)
```

conditions the generated samples on:

```text
Temperature = high
```

It does not mean:

```text
do(Temperature = high)
```

The sampling implementation is conditioning on evidence.

It does not implement causal intervention semantics.

---

# 50. AI Decision Rules

When an AI agent needs to generate synthetic data:

1. Confirm that the supplied object contains a Bayesian Network.
2. Confirm that the Bayesian Network contains CPDs.
3. Validate that `n >= 1`.
4. Use `methodtype='bayes'` for Bayesian sampling.
5. Use `methodtype='gibbs'` for Gibbs sampling.
6. Use no evidence for unconditional sampling.
7. Use `evidence={...}` with `methodtype='bayes'` for conditional sampling.
8. Never pass evidence to `methodtype='gibbs'`.
9. Ensure evidence is a dictionary.
10. Ensure every evidence variable exists in the model.
11. Treat evidence variable names as case-sensitive.
12. Ensure every evidence state exists in the variable's CPD state space.
13. Remember that impossible evidence is rejected before sampling.
14. Distinguish rare evidence from impossible evidence.
15. Expect rejection sampling to become slower as evidence becomes less
    probable.
16. Remember that `n` is the number of accepted/returned samples.
17. Remember that conditional rejection sampling may generate many more
    candidate samples internally.
18. Do not interpret evidence as a causal intervention.
19. Do not use sampling success as evidence that the Bayesian Network itself
    is correct.
20. Use the returned DataFrame as synthetic data generated according to the
    supplied probabilistic model.

---

# 51. Quick Reference

### Forward sampling

```python
df = bn.sampling(
    model,
    n=1000,
    methodtype='bayes'
)
```

### Conditional sampling

```python
df = bn.sampling(
    model,
    n=1000,
    methodtype='bayes',
    evidence={
        'Rain': 1
    }
)
```

### Gibbs sampling

```python
df = bn.sampling(
    model,
    n=1000,
    methodtype='gibbs'
)
```

### Invalid: Gibbs + evidence

```python
df = bn.sampling(
    model,
    n=1000,
    methodtype='gibbs',
    evidence={'Rain': 1}
)
```

Do not use this combination.

### Result

```python
df
```

is a:

```text
pandas.DataFrame
```

containing synthetic observations.

---

# 52. Final Checklist

Before sampling:

* [ ] Is `DAG` provided?
* [ ] Does `DAG['model']` contain a `BayesianNetwork`?
* [ ] Does the Bayesian Network contain CPDs?
* [ ] Is `n >= 1`?
* [ ] Is `methodtype` either `'bayes'` or `'gibbs'`?
* [ ] If using evidence, is `methodtype='bayes'`?
* [ ] Is `evidence` a dictionary?
* [ ] Do all evidence variables exist?
* [ ] Are variable names correctly capitalized?
* [ ] Are all evidence states valid CPD states?
* [ ] Is the evidence possible under the model?
* [ ] If evidence is rare, is rejection-sampling performance acceptable?
* [ ] Is the intended operation conditioning rather than intervention?

The central sampling logic is:

```text
Bayesian Network + CPDs
          │
          ├───────────────┐
          │               │
     No evidence      Evidence
          │               │
          ▼               ▼
Forward sampling    Validate evidence
                          │
                          ▼
                    Check P(E) > 0
                          │
                          ▼
                  Rejection sampling
                          │
                          ▼
                    Synthetic Data
```

The essential rule is:

```text
bayes + no evidence
    → forward sampling

bayes + evidence
    → rejection sampling

gibbs + no evidence
    → Gibbs sampling

gibbs + evidence
    → unsupported
```

`bnlearn.sampling()` therefore provides synthetic-data generation directly from
a parameterized Bayesian Network, with conditional generation available
through Bayesian rejection sampling.

---

# Sampling continuous and hybrid models

```python
df_s = bn.sampling(model_lg, n=200, methodtype='linear-gaussian', seed=0)
df_s = bn.sampling(model_lg, n=200, methodtype='linear-gaussian', do={'X': 0.0}, seed=0)
df_s = bn.sampling(model_cg, n=200, methodtype='cg', evidence={'fail': 0}, seed=0)
df_s = bn.sampling(model, n=200, methodtype='auto')
# convenience wrapper
df_s = bn.inference.sample(model, n=200, seed=0)
```
