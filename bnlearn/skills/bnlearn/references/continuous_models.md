# Continuous Models

> **Quick API (verified bnlearn ≥ 0.14, continuous / hybrid pipeline)**
>
> **Gaussian structure + linear-Gaussian parameters + inference / sampling**
> ```python
> DAG = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic-g')  # or aic-g | loglik-g | auto
> model = bn.parameter_learning.fit(DAG, df, methodtype='linear-gaussian')  # alias: 'lg'
> q = bn.inference.fit(model, variables=['Y'], evidence={'X': 0.5})
> df_s = bn.sampling(model, n=200, methodtype='linear-gaussian', seed=0)
> ```
>
> **Conditional Gaussian (mixed discrete + continuous)**
> ```python
> DAG = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic-cg')  # or aic-cg | loglik-cg | auto
> model = bn.parameter_learning.fit(DAG, df, methodtype='cg')  # alias: 'conditional-gaussian'
> q = bn.inference.fit(model, variables=['torque'], evidence={'fail': 1})
> df_s = bn.sampling(model, n=200, methodtype='cg', seed=0)
> ```
>
> **LiNGAM causal discovery** (structure only)
> ```python
> DAG = bn.structure_learning.fit(df, methodtype='direct-lingam')  # or 'ica-lingam'
> ```
>
> Do **not** use discrete scores (`bic`, `k2`, …) on continuous data.
> Do **not** discretize automatically; prefer the continuous / CG pathway first.

---

`bnlearn` supports continuous and hybrid data through several approaches:

1. **Linear Gaussian Bayesian Network scoring** (structure)

   * `loglik-g`, `aic-g`, `bic-g`

2. **Conditional Gaussian (hybrid) scoring** (structure on mixed data)

   * `loglik-cg`, `aic-cg`, `bic-cg`
   * `scoretype='auto'` selects `bic` / `bic-g` / `bic-cg` from column types

3. **Parameter learning**

   * `methodtype='linear-gaussian'` / `'lg'` — pure continuous
   * `methodtype='cg'` / `'conditional-gaussian'` — mixed
   * `methodtype='auto'` — choose from data types

4. **Inference and sampling** on fitted LG / CG models

5. **LiNGAM causal discovery** (structure only)

   * `direct-lingam`, `ica-lingam`

These approaches should not be treated as equivalent.

The Gaussian scores evaluate DAG structures under a linear Gaussian model.
Conditional Gaussian scores handle mixed discrete/continuous data.
LiNGAM uses linear structural relationships with non-Gaussian errors to orient edges.

---

# 1. Continuous Modeling Overview

The continuous-data workflow is:

```text
Continuous / mixed data
      │
      ├── Pure continuous → Gaussian BN pipeline
      │       structure: hc/ex + bic-g | aic-g | loglik-g | auto
      │       parameters: methodtype='linear-gaussian'
      │       inference / sampling: methodtype='linear-gaussian' | auto
      │
      ├── Mixed discrete + continuous → Conditional Gaussian pipeline
      │       structure: hc/ex + bic-cg | aic-cg | loglik-cg | auto
      │       parameters: methodtype='cg'
      │       inference / sampling: methodtype='cg' | auto
      │
      └── Causal orientation only (structure endpoint)
              ├── direct-lingam
              └── ica-lingam
```

Do not select LiNGAM merely because the variables are continuous.
Prefer the Gaussian or CG pipeline when parameters, inference, or sampling are required.

---

# 2. Linear Gaussian Bayesian Networks

A linear Gaussian Bayesian Network models each variable as a linear function
of its parents plus Gaussian noise.

For a node `Y` with parents:

```text
X₁, X₂, ..., Xₚ
```

the local model is:

```text
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε
```

with:

```text
ε ~ N(0, σ²)
```

where:

* `β₀` is the intercept.
* `β₁ ... βₚ` are regression coefficients.
* `σ²` is the residual variance.

The DAG score is decomposable:

```text
Score(DAG)
    =
Σ Score(node | parents)
```

This decomposability is what allows the Gaussian score classes to be used
with structure-search algorithms such as Hill Climbing and Exhaustive Search.
The implementation explicitly subclasses pgmpy's `StructureScore` for this
reason.

---

# 3. Gaussian Scores

Three Gaussian scores are implemented:

```python
scoretype='loglik-g'
scoretype='aic-g'
scoretype='bic-g'
```

They are implemented as:

```text
LogLikelihoodGauss
AICGauss
BICGauss
```

The scoring-type dispatcher maps these names directly to the corresponding
classes.

---

# 4. Data Requirements

The Gaussian scoring classes require a pandas DataFrame:

```python
import pandas as pd

df = pd.DataFrame(...)
```

The following requirements are enforced:

```text
DataFrame
    +
at least one sample
    +
numeric columns
    +
finite values
```

The implementation rejects:

* Non-DataFrame input.
* Empty datasets.
* Non-numeric columns.
* Missing values (`NaN`).
* Infinite values (`+inf`, `-inf`).

These checks are performed when the Gaussian score object is initialized.

---

# 5. Numeric Columns

Every column must have a numeric pandas dtype.

For example:

```python
df.dtypes
```

should contain numeric types for every variable.

This is valid:

```text
temperature    float64
pressure       float64
rpm            float64
torque         float64
```

This is not valid for Gaussian scoring:

```text
temperature    float64
machine        object
pressure       float64
```

Do not encode categorical variables as arbitrary integers and then treat those
integers as Gaussian measurements.

For example:

```text
low     → 0
medium  → 1
high    → 2
```

does not make `low`, `medium`, and `high` a continuous Gaussian variable.

---

# 6. Missing Values

Gaussian scores do not support missing values.

This will fail:

```python
df = pd.DataFrame({
    'A': [1.0, 2.0, np.nan],
    'B': [2.0, 3.0, 4.0]
})
```

Before structure learning, missing values must be handled.

Possible preprocessing strategies include:

```text
imputation
row removal
model-based missing-data handling
```

The appropriate strategy depends on the application.

The Gaussian scoring implementation itself does not perform imputation.

---

# 7. Infinite Values

Infinite values are also rejected.

For example:

```python
df = pd.DataFrame({
    'A': [1.0, 2.0, np.inf]
})
```

is invalid.

Check the data with:

```python
np.isfinite(df.to_numpy(dtype=float)).all()
```

before using a Gaussian score.

The implementation explicitly checks all values for finiteness.

---

# 8. Local Gaussian Regression

For each node, the implementation constructs a regression model using its
parents.

Suppose:

```text
A → C
B → C
```

Then the local model for `C` is:

```text
C = β₀ + β₁A + β₂B + ε
```

The design matrix is constructed as:

```text
[1, A, B]
```

where the first column contains ones for the intercept.

The regression coefficients are estimated using least squares:

```python
coefficients, _, _, _ = np.linalg.lstsq(
    design,
    y,
    rcond=None
)
```

The residuals are then:

```text
residual = observed - predicted
```

The implementation uses this least-squares regression for every local node
model.

---

# 9. Root Nodes

A node without parents is modeled using its mean.

For:

```text
A
```

with no parents:

```text
A = β₀ + ε
```

the implementation computes residuals as:

```text
residual = A - mean(A)
```

Thus the intercept corresponds to the sample mean when a node has no parents.

---

# 10. Residual Sum of Squares

After fitting the local regression, the implementation computes:

```text
RSS = Σ residualᵢ²
```

or equivalently:

```text
RSS = residualᵀ residual
```

The implementation then estimates the Gaussian residual variance as:

```text
variance = RSS / n
```

where `n` is the number of observations.

This is important:

```text
variance = RSS / n
```

not:

```text
RSS / (n - p)
```

and not:

```text
RSS / (n - p - 1)
```

The implementation uses the maximum-likelihood-style variance estimate based
on `n`.

---

# 11. Variance Floor

The Gaussian score uses:

```python
variance_floor=1e-12
```

by default.

The actual variance used in scoring is:

```text
max(RSS / n, variance_floor)
```

Therefore:

```text
RSS / n < variance_floor
        ↓
variance = variance_floor
```

This prevents numerical problems caused by an estimated residual variance of
zero.

The value must be strictly greater than zero.

---

# 12. Why a Variance Floor Is Needed

A zero residual variance would result in:

```text
log(variance)
```

becoming:

```text
log(0)
```

which is undefined.

A perfect or nearly perfect linear relationship can produce extremely small
residual variance.

The floor therefore acts as a numerical safeguard:

```text
variance → max(estimated_variance, variance_floor)
```

It should not be interpreted as a substantive prior about the variance.

---

# 13. Gaussian Log-Likelihood

The local Gaussian log-likelihood implemented by `LogLikelihoodGauss` is:

```text
L = -0.5 n [
        log(2π)
        + 1
        + log(σ²)
    ]
```

where:

```text
σ² = max(RSS / n, variance_floor)
```

The implementation calculates this directly as:

```text
-0.5 * n * (
    log(2π)
    + 1
    + log(variance)
)
```

The result is returned as a floating-point value.

Higher scores are better.

---

# 14. `loglik-g`

Use:

```python
scoretype='loglik-g'
```

This score represents the Gaussian log-likelihood without an explicit AIC or
BIC complexity penalty.

Example:

```python
result = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='loglik-g'
)
```

The local score is:

```text
loglik-g(node | parents)
    =
-0.5 n [
    log(2π)
    + 1
    + log(σ²)
]
```

The total DAG score is the sum over all nodes.

---

# 15. Important Property of `loglik-g`

`loglik-g` measures model fit but does not subtract an explicit complexity
penalty.

Adding a parent can improve the fit by reducing residual variance.

Therefore, when using unrestricted structure search, `loglik-g` should not be
treated as equivalent to a penalized criterion such as `aic-g` or `bic-g`.

Use it when Gaussian likelihood itself is the desired scoring objective.

For structure selection where model complexity should explicitly be penalized,
prefer:

```text
aic-g
```

or:

```text
bic-g
```

depending on the objective.

---

# 16. Number of Gaussian Parameters

For a node with `p` parents, the implementation counts:

```text
1 intercept
+
p regression coefficients
+
1 variance parameter
```

Therefore:

```text
k = p + 2
```

Examples:

| Parents | Parameters |
| ------: | ---------: |
|       0 |          2 |
|       1 |          3 |
|       2 |          4 |
|       5 |          7 |
|      10 |         12 |

This parameter count is used by both `AICGauss` and `BICGauss`.

---

# 17. `aic-g`

Use:

```python
scoretype='aic-g'
```

The Gaussian AIC local score is:

```text
AIC = loglikelihood - k
```

where:

```text
k = p + 2
```

Therefore:

```text
AIC = loglikelihood - (p + 2)
```

for a node with `p` parents.

The implementation directly subtracts the parameter count from the local
Gaussian log-likelihood.

Higher scores are better.

---

# 18. Interpretation of `aic-g`

`aic-g` balances:

```text
Gaussian model fit
+
parameter complexity
```

Each additional parent introduces an additional regression coefficient and
therefore increases the parameter count.

For a node:

```text
Y
```

with:

```text
p = 2
```

parents:

```text
k = 4
```

and therefore:

```text
AIC = loglikelihood - 4
```

---

# 19. `bic-g`

Use:

```python
scoretype='bic-g'
```

The Gaussian BIC local score is:

```text
BIC =
    loglikelihood
    - 0.5 * k * log(n)
```

where:

```text
k = p + 2
```

Therefore:

```text
BIC =
    loglikelihood
    - 0.5 * (p + 2) * log(n)
```

The implementation uses the number of observations in the DataFrame for `n`.

Higher scores are better.

---

# 20. `aic-g` vs `bic-g`

The difference is entirely in the complexity penalty.

```text
aic-g:

loglikelihood - k
```

versus:

```text
bic-g:

loglikelihood - 0.5*k*log(n)
```

Therefore:

```text
n small
    → penalties can be relatively similar

n large
    → BIC increasingly penalizes complexity
```

Consequently, BIC will generally favor more parsimonious structures as the
sample size grows.

---

# 21. Gaussian Score Summary

| Score      | Formula             | Complexity penalty |
| ---------- | ------------------- | ------------------ |
| `loglik-g` | `LL`                | None               |
| `aic-g`    | `LL - k`            | AIC                |
| `bic-g`    | `LL - 0.5*k*log(n)` | BIC                |

where:

```text
LL = Gaussian log-likelihood
k  = p + 2
n  = number of observations
p  = number of parents
```

All three use the same underlying Gaussian regression and variance estimate.

---

# 22. Complete DAG Score

Because the Gaussian scores are decomposable, the score for an entire DAG is:

```text
Score(G)
    =
Σᵢ Score(Xᵢ | Parents(Xᵢ))
```

For example:

```text
A → C
B → C
```

gives:

```text
Score(G)
    =
Score(A | ∅)
+
Score(B | ∅)
+
Score(C | A, B)
```

This local decomposition makes it possible for structure-learning algorithms
to efficiently evaluate candidate edge modifications.

The implementation also caches local Gaussian scores by:

```text
(variable, sorted(parents))
```

to avoid repeatedly recalculating identical local models.

---

# 23. Parent Order Does Not Affect the Cache

The implementation sorts the parent tuple before constructing the cache key:

```text
parents = tuple(sorted(parents))
```

Therefore:

```text
(A, B)
```

and:

```text
(B, A)
```

refer to the same parent set for the purpose of local Gaussian scoring.

This is appropriate because the regression contains the same parent variables
regardless of their ordering.

---

# 24. Continuous Structure Learning with Hill Climbing

For a continuous Gaussian Bayesian Network, use:

```python
result = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic-g'
)
```

The process is:

```text
Data
 │
 ▼
Gaussian score
 │
 ▼
Initial DAG
 │
 ▼
Hill Climbing
 │
 ├── add edge
 ├── remove edge
 └── reverse edge
 │
 ▼
Evaluate Gaussian score
 │
 ▼
Accept improvement
 │
 ▼
Repeat
 │
 ▼
Final DAG
```

Hill Climbing uses the selected Gaussian score to determine which candidate
structure is preferred.

The implementation passes the selected scoring object directly into
`HillClimbSearch`.

---

# 25. Continuous Structure Learning with Exhaustive Search

For very small continuous networks:

```python
result = bn.structure_learning.fit(
    df,
    methodtype='ex',
    scoretype='bic-g'
)
```

Exhaustive Search evaluates possible DAGs using the selected Gaussian score.

The implementation constructs:

```python
ExhaustiveSearch(
    df,
    scoring_method=scoring_method
)
```

and estimates the best model.

Exhaustive search should only be considered for very small networks because
the DAG search space grows extremely rapidly.

---

# 26. Choosing the Gaussian Score

Use this rule:

```text
Need pure Gaussian likelihood?
    → loglik-g

Need Gaussian likelihood + AIC penalty?
    → aic-g

Need Gaussian likelihood + stronger sample-size-dependent penalty?
    → bic-g
```

For general structure learning:

```python
scoretype='bic-g'
```

is a sensible starting point when the linear Gaussian assumptions are
appropriate.

---

# 27. Continuous Does Not Automatically Mean Gaussian

This distinction is critical:

```text
continuous
    ≠
Gaussian
```

A continuous variable may have:

* Strong skewness.
* Heavy tails.
* Nonlinear relationships.
* Heteroscedasticity.
* Non-Gaussian errors.

The Gaussian score implementation assumes the local conditional model can be
represented using linear regression with Gaussian residuals.

Therefore, simply seeing numerical columns is not sufficient justification for
using:

```text
loglik-g
aic-g
bic-g
```

---

# 28. Linear Does Not Automatically Mean Causal

The Gaussian BN model:

```text
Y = β₀ + β₁X + ε
```

describes a conditional relationship within the Bayesian Network.

It does not by itself establish:

```text
X causes Y
```

A learned edge:

```text
X → Y
```

should therefore not automatically be interpreted as causal.

If causal discovery is the objective, consider whether a method such as
LiNGAM is appropriate.

---

# 29. Gaussian BN vs LiNGAM

These two approaches should be clearly separated.

| Property              | Gaussian BN scores     | LiNGAM                                          |
| --------------------- | ---------------------- | ----------------------------------------------- |
| `bnlearn` interface   | `scoretype='*-g'`      | `methodtype='*-lingam'`                         |
| Model                 | Linear Gaussian BN     | Linear non-Gaussian SEM                         |
| Error assumption      | Gaussian               | Non-Gaussian                                    |
| Main objective        | Structure scoring      | Causal discovery                                |
| Structure search      | Score-based            | LiNGAM algorithm                                |
| Continuous data       | Yes                    | Yes                                             |
| Mixed data            | No for Gaussian scores | DirectLiNGAM supports mixed data                |
| Causal interpretation | Not automatically      | Intended for causal discovery under assumptions |

Do not substitute one for the other.

---

# 30. DirectLiNGAM

Use:

```python
methodtype='direct-lingam'
```

Example:

```python
result = bn.structure_learning.fit(
    df,
    methodtype='direct-lingam'
)
```

The implementation describes DirectLiNGAM as a semi-parametric approach that
assumes:

```text
linear relationships
+
non-Gaussian error terms
+
acyclic graph
```

It uses repeated regression and independence assessments to infer causal
ordering.

---

# 31. Why Non-Gaussian Errors Matter for LiNGAM

LiNGAM relies on a property that distinguishes the correct causal direction
from the reverse direction.

In the implementation's description:

```text
correct causal order
    →
explanatory variable and residual become independent
```

whereas an incorrect order does not produce the same independence behavior.

This is a key difference from ordinary linear Gaussian models.

For Gaussian linear models, the direction of a regression cannot generally be
identified from observational data by likelihood alone.

LiNGAM introduces the non-Gaussian assumption to obtain additional
identifiability information.

The source explicitly describes this residual/explanatory-variable independence
mechanism.

---

# 32. DirectLiNGAM Configuration

The `fit()` API provides:

```python
params_lingam = {
    'random_state': None,
    'prior_knowledge': None,
    'apply_prior_knowledge_softly': False,
    'measure': 'pwling'
}
```

These parameters are passed to the LiNGAM implementation.

The defaults are defined directly in the structure-learning API.

---

# 33. `random_state`

Use:

```python
params_lingam={
    'random_state': 42
}
```

when deterministic behavior from the random-number generator is desired.

Example:

```python
result = bn.structure_learning.fit(
    df,
    methodtype='direct-lingam',
    params_lingam={
        'random_state': 42
    }
)
```

The default is:

```python
random_state=None
```

---

# 34. LiNGAM Prior Knowledge

LiNGAM supports:

```python
prior_knowledge
```

with shape:

```text
(n_features, n_features)
```

The implementation documents the matrix as encoding prior causal knowledge.

Use prior knowledge when the domain provides constraints that should influence
causal discovery.

For example, prior knowledge can distinguish variables that cannot have
directed paths to other variables.

Do not construct prior knowledge merely to force a desired result.

The documented configuration supports both hard and soft application of prior
knowledge.

---

# 35. `apply_prior_knowledge_softly`

The configuration contains:

```python
apply_prior_knowledge_softly=False
```

This controls whether the supplied prior knowledge is applied softly.

When prior knowledge is supplied, document whether it represents:

```text
hard domain constraints
```

or:

```text
soft prior information
```

because these have different interpretations.

---

# 36. LiNGAM Independence Measures

The supported measures are:

```text
pwling
kernel
pwling_fast
```

The default is:

```python
measure='pwling'
```

For GPU-enabled execution, the implementation documentation identifies:

```text
pwling_fast
```

as the accelerated option when the required dependency is available.

Use the default unless there is a specific reason to select another measure.

---

# 37. ICA-LiNGAM

The second LiNGAM method is:

```python
methodtype='ica-lingam'
```

Example:

```python
result = bn.structure_learning.fit(
    df,
    methodtype='ica-lingam'
)
```

The implementation constructs an `ICALiNGAM` model and fits it to the supplied
DataFrame.

ICA-LiNGAM is therefore a separate causal-discovery route from
DirectLiNGAM and should not be described as another Gaussian scoring method.

---

# 38. LiNGAM and Mixed Data

The `bnlearn` structure-learning documentation explicitly describes
DirectLiNGAM as applicable to:

```text
continuous
+
mixed
```

datasets.

This should not be confused with the Gaussian score classes.

The Gaussian score classes explicitly require:

```text
numeric columns only
```

Therefore:

```text
Mixed data
    → DirectLiNGAM can be considered

Mixed data
    → loglik-g / aic-g / bic-g are not directly applicable
```

---

# 39. Do Not Use Gaussian Scores for Mixed Data

This is invalid:

```python
result = bn.structure_learning.fit(
    mixed_df,
    methodtype='hc',
    scoretype='bic-g'
)
```

when `mixed_df` contains categorical/object columns.

The Gaussian scoring implementation will reject non-numeric columns.

If mixed data are present and causal discovery is the objective, consider:

```python
methodtype='direct-lingam'
```

provided its assumptions are appropriate.

---

# 40. Continuous Workflow

A robust workflow for continuous data is:

```text
1. Inspect variable types
        ↓
2. Check missing values
        ↓
3. Check infinite values
        ↓
4. Examine distributions
        ↓
5. Assess linearity
        ↓
6. Decide:
       Gaussian BN
       or
       LiNGAM
        ↓
7. Select method
        ↓
8. Learn structure
        ↓
9. Inspect learned DAG
        ↓
10. Validate stability
        ↓
11. Interpret cautiously
```

Do not jump directly from:

```text
numeric DataFrame
```

to:

```text
bic-g
```

without considering the model assumptions.

---

# 41. Example: Gaussian Bayesian Network

For continuous sensor variables:

```python
import bnlearn as bn

result = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic-g'
)
```

Inspect the structure:

```python
print(result['model_edges'])
```

The resulting graph represents the highest-scoring structure found by the
selected search algorithm under the Gaussian BIC objective.

---

# 42. Example: Compare Gaussian Scores

To evaluate sensitivity to the complexity criterion:

```python
result_loglik = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='loglik-g'
)

result_aic = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='aic-g'
)

result_bic = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic-g'
)
```

Compare:

```text
number of edges
edge identities
parent sets
model stability
```

Do not compare the raw numerical values as if:

```text
loglik-g
aic-g
bic-g
```

were the same metric.

They are not.

---

# 43. Example: DirectLiNGAM

For causal discovery with continuous data:

```python
result = bn.structure_learning.fit(
    df,
    methodtype='direct-lingam'
)
```

With a fixed random state:

```python
result = bn.structure_learning.fit(
    df,
    methodtype='direct-lingam',
    params_lingam={
        'random_state': 42
    }
)
```

Use this when the linear non-Gaussian assumptions underlying LiNGAM are
appropriate.

---

# 44. Gaussian Scores vs DirectLiNGAM

The choice should be driven by the question.

### Question:

> Which DAG provides a good linear Gaussian representation of the data?

Consider:

```text
hc + bic-g
```

### Question:

> Can causal ordering be identified under linear non-Gaussian assumptions?

Consider:

```text
direct-lingam
```

These are different scientific questions.

---

# 45. Common Mistakes

### Mistake 1 — Treating numeric data as automatically Gaussian

Incorrect:

```text
dtype=float
    →
Gaussian model
```

Correct:

```text
dtype=float
+
linear Gaussian assumptions reasonable
    →
Gaussian model
```

---

### Mistake 2 — Using discrete BIC on continuous measurements

Do not assume:

```python
scoretype='bic'
```

is appropriate simply because the data are Bayesian Network data.

For continuous Gaussian variables:

```python
scoretype='bic-g'
```

is the corresponding Gaussian criterion.

---

### Mistake 3 — Encoding categories as numbers

Do not convert:

```text
low
medium
high
```

to:

```text
0
1
2
```

and then use Gaussian scoring unless those values genuinely represent a
continuous numerical scale.

---

### Mistake 4 — Using `loglik-g` to avoid complexity penalties

`loglik-g` is an unpenalized Gaussian likelihood criterion.

For model selection, consider:

```text
aic-g
```

or:

```text
bic-g
```

when a complexity penalty is desired.

---

### Mistake 5 — Assuming BIC and AIC are equivalent

They use different complexity penalties:

```text
AIC:
    k

BIC:
    0.5*k*log(n)
```

The difference becomes increasingly important with larger datasets.

---

### Mistake 6 — Assuming Gaussian BN edges are causal

A Gaussian BN score evaluates statistical model fit.

It does not by itself establish causality.

---

### Mistake 7 — Using LiNGAM without checking its assumptions

LiNGAM relies on linearity and non-Gaussian errors.

Continuous data alone are not sufficient.

---

### Mistake 8 — Ignoring residual variance

The Gaussian implementation estimates:

```text
variance = max(RSS / n, variance_floor)
```

Therefore, extremely small residual variance can materially affect the
likelihood.

---

# 46. Important Implementation Detail

The Gaussian score classes use:

```text
RSS / n
```

for residual variance.

This matters when reproducing or validating `bnlearn` scores externally.

A separate implementation that uses:

```text
RSS / (n - p)
```

will generally produce different likelihood values.

Therefore, when reproducing `bnlearn` Gaussian scores, reproduce the exact
variance convention used by the implementation.

---

# 47. Important Implementation Detail: Parameter Count

The implementation uses:

```text
k = number_of_parents + 2
```

where the two additional parameters are:

```text
intercept
variance
```

Therefore:

```text
AIC-G:
    LL - (p + 2)

BIC-G:
    LL - 0.5 * (p + 2) * log(n)
```

Do not use:

```text
k = p
```

when reproducing the implemented scores.

The intercept and variance are explicitly counted.

---

# 48. Important Implementation Detail: Score Cache

The Gaussian implementation maintains:

```python
self._gaussian_score_cache
```

and caches local scores using:

```text
(variable, sorted(parents))
```

This means repeated evaluation of the same local family does not require
re-fitting the regression.

This is particularly useful during Hill Climbing, where many candidate DAGs
share the same local parent configurations.

---

# 49. Practical Recommendations

For a continuous dataset:

```text
If the goal is general BN structure learning
    →
    hc + bic-g

If AIC-style model selection is preferred
    →
    hc + aic-g

If Gaussian likelihood itself is the objective
    →
    hc + loglik-g

If causal discovery is the objective and
linear non-Gaussian assumptions are reasonable
    →
    direct-lingam

If ICA-based LiNGAM is specifically desired
    →
    ica-lingam
```

---

# 50. AI Decision Rules

When an AI agent encounters continuous data:

1. Determine whether all variables are numeric.
2. Check for missing values.
3. Check for infinite values.
4. Do not assume that numeric variables are Gaussian.
5. Determine whether a linear Gaussian conditional model is reasonable.
6. If yes and the objective is Bayesian Network structure learning, consider
   `hc` with `bic-g`.
7. Use `aic-g` when an AIC complexity penalty is specifically desired.
8. Use `loglik-g` when an unpenalized Gaussian likelihood objective is desired.
9. Remember that the Gaussian implementation uses `RSS / n` for residual
   variance.
10. Remember that Gaussian AIC/BIC count `p + 2` parameters for a node with
    `p` parents.
11. Do not use Gaussian scores on DataFrames containing non-numeric columns.
12. Do not use Gaussian scores with missing or infinite values.
13. Do not interpret Gaussian BN edges as automatically causal.
14. If causal discovery is the objective, evaluate whether LiNGAM assumptions
    are appropriate.
15. Use `direct-lingam` for DirectLiNGAM and `ica-lingam` for ICA-LiNGAM.
16. Remember that DirectLiNGAM is documented as supporting continuous and mixed
    datasets.
17. Do not describe LiNGAM as a Gaussian Bayesian Network scoring method.
18. When reproducing `bnlearn` Gaussian scores, reproduce the exact implemented
    variance and parameter-count conventions.
19. Compare models using the same scoring criterion.
20. Validate learned structures rather than treating the highest-scoring DAG as
    ground truth.

---

# 51. Minimal Reference

### Gaussian Bayesian Network

```python
result = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic-g'
)
```

### Gaussian AIC

```python
result = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='aic-g'
)
```

### Gaussian likelihood

```python
result = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='loglik-g'
)
```

### DirectLiNGAM

```python
result = bn.structure_learning.fit(
    df,
    methodtype='direct-lingam'
)
```

### ICA-LiNGAM

```python
result = bn.structure_learning.fit(
    df,
    methodtype='ica-lingam'
)
```

---

# 52. Final Checklist

Before using a continuous model:

* [ ] Are all variables numeric?
* [ ] Are there missing values?
* [ ] Are there infinite values?
* [ ] Are the relationships approximately linear?
* [ ] Is a Gaussian error model reasonable?
* [ ] Is the objective Bayesian Network structure learning?
* [ ] Or is the objective causal discovery?
* [ ] If Bayesian Network structure learning, should complexity be penalized?
* [ ] If yes, should AIC or BIC be used?
* [ ] If causal discovery, are LiNGAM's non-Gaussian assumptions reasonable?
* [ ] Are mixed data present?
* [ ] If mixed data are present, avoid the Gaussian score classes.
* [ ] Are learned edges being interpreted appropriately?
* [ ] Has structure stability been evaluated?
* [ ] If reproducing scores, are `RSS / n` and `k = p + 2` being used?

---

# Parameter learning for continuous and hybrid models

After structure learning, estimate parameters with a matching method:

| `methodtype` | Data | Result |
|---|---|---|
| `linear-gaussian` / `lg` | continuous | `LinearGaussianBayesianNetwork` |
| `cg` / `conditional-gaussian` | mixed | discrete CPTs + `continuous_cpds` |
| `auto` | any | chooses bayes / linear-gaussian / cg |

```python
model = bn.parameter_learning.fit(DAG, df, methodtype='linear-gaussian')
model = bn.parameter_learning.fit(DAG, df, methodtype='cg')
model = bn.parameter_learning.fit(DAG, df, methodtype='auto')
```

For CG fits, local continuous regressions are stored in `model['continuous_cpds']`.
Discrete-only fits do **not** include that key (stable result keys for existing tests).

---

# Inference on continuous and hybrid models

`bn.inference.fit` routes by model type:

* Discrete → `fit_discrete` (Variable Elimination; original behaviour)
* Linear-Gaussian → `fit_continuous` (conditional means)
* CG → discrete VE for discrete queries; CG local means/std for continuous queries

Evidence may mix discrete states and continuous numbers. Use `do={...}` for interventions.

```python
q = bn.inference.fit(model, variables=['Y'], evidence={'X': 0.5})
q = bn.inference.fit(model, variables=['Y'], do={'X': 1.0})
print(q.means)  # ContinuousQueryResult
```

---

# Sampling continuous and hybrid models

```python
df_s = bn.sampling(model, n=200, methodtype='linear-gaussian', seed=0)
df_s = bn.sampling(model, n=200, methodtype='cg', do={'fail': 0}, seed=0)
df_s = bn.sampling(model, n=200, methodtype='auto')
# or
df_s = bn.inference.sample(model, n=200, seed=0)
```

---

# Conditional Gaussian (hybrid) networks

A CG network models mixed data as:

* **Discrete nodes** — standard TabularCPDs
* **Continuous nodes** — linear Gaussian regressions; if discrete parents exist, one coefficient vector and residual std **per discrete parent configuration**

Use CG when you need both probability-level answers for discrete outcomes and coefficient-level insight among continuous sensors in one graph.
