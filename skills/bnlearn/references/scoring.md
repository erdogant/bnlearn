# Structure Scoring

> **Quick API (verified bnlearn ≥ 0.14)**
>
> **During structure learning** (preferred):
> ```python
> DAG = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
> # discrete:   bic | aic | k2 | bdeu | bds
> # continuous: bic-g | aic-g | loglik-g
> ```
>
> **Score an existing model**:
> ```python
> scores = bn.structure_scores(
>     model, df,
>     scoring_method=['bic', 'k2', 'bdeu', 'bds'],
> )
> ```
>
> Never mix discrete scores with continuous data (or Gaussian scores with
> discrete data) without an explicit, justified conversion step.

---

Structure scores quantify how well a candidate Bayesian Network structure
explains the observed data.

In `bnlearn`, scores are primarily used by score-based structure-learning
methods such as Hill Climbing.

The general optimization problem is:

```text
Given data D

Find DAG G that maximizes:

    Score(G | D)
```

The score must therefore be interpreted together with:

* The data type.
* The probabilistic assumptions.
* The likelihood model.
* The complexity penalty.
* The structure-learning algorithm.

Do not select a score independently of the data-generating assumptions.

---

# 1. Supported Scores

The current `bnlearn` structure-learning implementation supports:

```text
bic
aic
k2
bdeu
bds
loglik-g
aic-g
bic-g
```

The scores fall into two main groups:

```text
Structure scores
│
├── Discrete
│   ├── bic
│   ├── aic
│   ├── k2
│   ├── bdeu
│   └── bds
│
└── Continuous Gaussian
    ├── loglik-g
    ├── aic-g
    └── bic-g
```

The Gaussian scores are separate from the standard discrete Bayesian Network
scores and should not be treated as interchangeable.

---

# 2. Most Important Rule

The first question when selecting a score is:

```text
What type of variables are being modeled?
```

Use:

```text
Discrete variables
    → bic / aic / k2 / bdeu / bds

Continuous Gaussian variables
    → loglik-g / aic-g / bic-g
```

Do not use a discrete score simply because it is commonly used.

Do not automatically discretize continuous variables merely to make a
discrete score applicable.

---

# 3. Score Optimization Direction

`bnlearn` uses scores where a **higher score is better**.

Therefore:

```text
Higher score
    →
Better candidate structure
```

When comparing two candidate DAGs under the same score:

```text
Score(G1) > Score(G2)
```

means `G1` is preferred according to that scoring criterion.

Do not compare numerical scores from different scoring families as if the
values were directly comparable.

For example:

```text
BIC = -120
AIC = -100
```

does not mean AIC produced a better model than BIC.

The two scores have different definitions.

---

# 4. Score Selection Decision Tree

Use this decision process:

```text
What type of data?

├── Discrete
│   │
│   ├── General model selection
│   │      └── BIC
│   │
│   ├── Bayesian Dirichlet scoring
│   │      ├── BDeu
│   │      └── BDs
│   │
│   ├── K2-compatible scoring
│   │      └── K2
│   │
│   └── AIC-based selection
│          └── AIC
│
└── Continuous
    │
    └── Approximately Gaussian / linear-Gaussian model
           ├── loglik-g
           ├── aic-g
           └── bic-g
```

When uncertain, `bic` is a reasonable default for discrete Bayesian Network
structure learning, while `bic-g` is a reasonable default for continuous
Gaussian structure learning.

These defaults should not override domain-specific assumptions.

---

# 5. BIC

## Score type

```python
scoretype='bic'
```

BIC stands for Bayesian Information Criterion.

The general form is:

```text
BIC = log-likelihood - 0.5 * k * log(n)
```

where:

* `log-likelihood` measures how well the model explains the data.
* `k` is the number of estimated parameters.
* `n` is the number of observations.

Higher BIC is better.

The penalty for additional parameters increases with sample size.

---

# 6. Why BIC Is a Strong Default

BIC balances:

```text
Model fit
    +
Model complexity
```

A more complex DAG can improve likelihood by adding edges, but additional
parameters are penalized.

This makes BIC useful when the goal is to identify a reasonably parsimonious
network rather than simply maximizing training-data fit.

For general discrete Bayesian Network structure learning:

```python
result = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic'
)
```

is a strong starting point.

---

# 7. BIC and Network Complexity

Adding an edge generally increases model flexibility.

For example:

```text
A → B
```

has fewer parameters than:

```text
A → B
C → B
D → B
```

BIC asks whether the additional explanatory power justifies the additional
complexity.

Therefore BIC tends to favor simpler structures when additional edges do not
provide sufficient improvement in fit.

This is particularly important when structure learning is performed with
Hill Climbing.

---

# 8. AIC

## Score type

```python
scoretype='aic'
```

AIC stands for Akaike Information Criterion.

The general form is:

```text
AIC = log-likelihood - k
```

where:

* `log-likelihood` measures model fit.
* `k` is the number of estimated parameters.

Higher AIC is better under the `bnlearn` scoring convention.

AIC imposes a weaker complexity penalty than BIC for sufficiently large
datasets.

Therefore:

```text
AIC
    → generally more tolerant of model complexity

BIC
    → generally stronger complexity penalty
```

---

# 9. AIC vs BIC

The main distinction is the complexity penalty.

```text
AIC:
    log-likelihood - k

BIC:
    log-likelihood - 0.5*k*log(n)
```

As `n` increases, the BIC penalty increases relative to AIC.

Therefore, when comparing structures:

```text
AIC
    may retain additional edges

BIC
    may prefer a more parsimonious structure
```

Do not select AIC or BIC solely from which one produces a visually appealing
network.

The choice should reflect the modeling objective.

---

# 10. K2

## Score type

```python
scoretype='k2'
```

K2 is a Bayesian Network scoring criterion for discrete variables.

It is associated with Bayesian parameter estimation using a particular prior
structure.

Use K2 when:

* The variables are discrete.
* K2 is specifically desired.
* Compatibility with a K2-based Bayesian Network workflow is important.

K2 should not be treated as a generic replacement for BIC.

---

# 11. K2 and Node Ordering

K2-style Bayesian Network structure learning is traditionally associated with
a predefined ordering of variables.

When using K2, consider whether the required variable ordering is meaningful
for the problem.

Do not invent a causal or temporal ordering merely because an algorithm can
use one.

If the ordering is unknown, prefer a structure-learning approach whose
assumptions better match the problem.

---

# 12. BDeu

## Score type

```python
scoretype='bdeu'
```

BDeu stands for Bayesian Dirichlet equivalent uniform.

It is a Bayesian score for discrete Bayesian Networks.

BDeu uses a Dirichlet prior and is designed to provide score equivalence
under its assumptions.

Score equivalence is important because Markov-equivalent DAGs can represent
the same conditional independence structure.

BDeu is therefore useful when Bayesian scoring and score-equivalent treatment
of equivalent structures are desired.

---

# 13. Equivalent DAGs

Different DAGs can encode the same set of conditional independence
relationships.

For example, under appropriate conditions, different orientations can belong
to the same Markov equivalence class.

Therefore:

```text
Different DAG
    ≠
Different statistical model
```

when the DAGs are Markov equivalent.

This is particularly important when interpreting structure-learning output.

A score-equivalent scoring criterion can treat equivalent structures
consistently, but this does not identify a unique causal DAG.

---

# 14. Equivalent Sample Size

BDeu uses a prior whose behavior depends on the equivalent sample size.

The equivalent sample size controls the strength of the prior relative to the
observed data.

Therefore, BDeu results can depend on prior settings.

When comparing networks using BDeu:

* Keep the equivalent sample size consistent.
* Report it when it materially affects the analysis.
* Avoid treating a single equivalent sample size as universally optimal.

Do not interpret changes caused by the prior as purely data-driven effects.

---

# 15. BDs

## Score type

```python
scoretype='bds'
```

BDs is another Bayesian Dirichlet score for discrete Bayesian Networks.

It is related to BDeu but uses a different treatment of the Dirichlet prior,
particularly for parent configurations that are not observed.

Use BDs when the corresponding Bayesian scoring assumptions are appropriate.

Do not assume:

```text
BDeu == BDs
```

They are different scoring criteria and can produce different preferred
network structures.

---

# 16. Discrete Bayesian Scores Compared

| Score | Data     | Main characteristic                  | Complexity     |
| ----- | -------- | ------------------------------------ | -------------- |
| BIC   | Discrete | Likelihood + complexity penalty      | Penalized      |
| AIC   | Discrete | Likelihood + complexity penalty      | Weaker penalty |
| K2    | Discrete | Bayesian score                       | Prior-based    |
| BDeu  | Discrete | Bayesian Dirichlet, score-equivalent | Prior-based    |
| BDs   | Discrete | Bayesian Dirichlet variant           | Prior-based    |

The score should be selected based on the statistical objective rather than
the name of the algorithm alone.

---

# 17. Gaussian Scores

For continuous data, `bnlearn` provides:

```text
loglik-g
aic-g
bic-g
```

The `-g` suffix indicates Gaussian scoring.

These scores assume a linear Gaussian Bayesian Network.

For a node `Y` with parents:

```text
X₁, X₂, ..., Xₚ
```

the local model is:

```text
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε
```

where:

```text
ε ~ Gaussian(0, σ²)
```

The complete DAG score is decomposable into the sum of local node scores.

---

# 18. `loglik-g`

## Score type

```python
scoretype='loglik-g'
```

This score uses the Gaussian log-likelihood.

Conceptually:

```text
Score(G)
    =
Σ local_log_likelihood(node | parents)
```

No explicit AIC/BIC complexity penalty is applied.

Therefore, adding parents can improve the score by improving model fit.

This makes `loglik-g` useful for evaluating Gaussian model fit, but it should
be used with care for unconstrained structure search because a pure
likelihood criterion does not provide the same complexity control as AIC or
BIC.

---

# 19. `aic-g`

## Score type

```python
scoretype='aic-g'
```

`aic-g` is the Gaussian version of AIC.

For each local Gaussian regression:

```text
AIC = log-likelihood - k
```

where `k` is the number of estimated model parameters.

The implementation counts:

```text
1 intercept
+
number of parent coefficients
+
1 variance parameter
```

as the local number of parameters.

Therefore, with `p` parents:

```text
k = p + 2
```

The complete network score is obtained by summing the local scores.

---

# 20. `bic-g`

## Score type

```python
scoretype='bic-g'
```

`bic-g` is the Gaussian version of BIC.

The local score is:

```text
BIC = log-likelihood
      - 0.5 * k * log(n)
```

where:

* `k` is the number of local parameters.
* `n` is the number of observations.

The implementation counts:

```text
intercept
+
parent coefficients
+
variance
```

as model parameters.

Therefore, with `p` parents:

```text
k = p + 2
```

The complete DAG score is the sum of the local Gaussian BIC scores.

---

# 21. Gaussian Score Comparison

For continuous Gaussian data:

| Score      | Fit                 | Complexity penalty | Recommended use        |
| ---------- | ------------------- | ------------------ | ---------------------- |
| `loglik-g` | Gaussian likelihood | None               | Likelihood evaluation  |
| `aic-g`    | Gaussian likelihood | AIC                | Fit vs complexity      |
| `bic-g`    | Gaussian likelihood | BIC                | Parsimonious structure |

A useful default for structure learning is:

```python
scoretype='bic-g'
```

when a linear Gaussian model is scientifically appropriate.

---

# 22. Gaussian Score Requirements

The Gaussian scoring implementation expects numeric data.

The data should contain:

* Numeric columns.
* Finite values.
* No missing values.
* No infinite values.

Before using:

```python
scoretype='loglik-g'
scoretype='aic-g'
scoretype='bic-g'
```

inspect the DataFrame.

For example:

```python
df.dtypes
df.isna().sum()
```

Also check:

```python
import numpy as np

np.isfinite(df.to_numpy()).all()
```

Do not silently convert categorical variables to integer codes and treat those
codes as continuous measurements.

---

# 23. Gaussian Assumptions

Gaussian scores are based on a linear Gaussian model.

This means the conditional relationship is modeled as:

```text
Y | Parents(Y)
```

being Gaussian with a mean that is linear in the parent variables.

Consider whether this is reasonable for the data.

Potential problems include:

* Strong nonlinear relationships.
* Severe outliers.
* Highly skewed distributions.
* Heteroscedasticity.
* Non-Gaussian residuals.

A variable being numeric does not automatically make a Gaussian Bayesian
Network appropriate.

---

# 24. Continuous Does Not Mean Gaussian

This distinction is critical.

```text
continuous
    ≠
Gaussian
```

A dataset can contain continuous measurements while violating the assumptions
of a linear Gaussian model.

For example:

```text
Temperature
Pressure
Vibration
Torque
```

are continuous variables, but their conditional relationships may be
nonlinear or non-Gaussian.

Before selecting `bic-g`, consider whether the Gaussian assumptions are
reasonable.

---

# 25. Do Not Automatically Discretize

A common mistake is:

```text
Continuous data
      ↓
Discretize
      ↓
Use BIC
```

This is not automatically preferable.

Discretization can:

* Remove information.
* Introduce arbitrary thresholds.
* Change conditional relationships.
* Reduce statistical resolution.

Instead:

```text
Continuous data
      │
      ├── Linear Gaussian assumptions reasonable?
      │       └── Yes → Gaussian score
      │
      └── No
          ├── Transform variables
          ├── Consider alternative continuous methods
          └── Consider discretization when justified
```

---

# 26. Score and Method Must Match

The structure-learning algorithm and score solve different parts of the
problem.

For example:

```python
methodtype='hc'
scoretype='bic'
```

means:

```text
Hill Climbing
+
BIC objective
```

while:

```python
methodtype='hc'
scoretype='bic-g'
```

means:

```text
Hill Climbing
+
Gaussian BIC objective
```

Do not confuse:

```text
methodtype
```

with:

```text
scoretype
```

The method defines **how the search is performed**.

The score defines **what makes one candidate DAG better than another**.

---

# 27. Score Selection Examples

## Example 1 — Discrete data

Suppose:

```text
age_group
smoking
exercise
disease
```

are categorical variables.

A reasonable starting point is:

```python
result = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic'
)
```

---

## Example 2 — Continuous sensor data

Suppose:

```text
temperature
pressure
rpm
torque
vibration
```

are continuous measurements.

Do not automatically discretize them.

If a linear Gaussian model is appropriate:

```python
result = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic-g'
)
```

---

## Example 3 — Comparing Gaussian criteria

Run multiple structure-learning analyses:

```python
model_loglik = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='loglik-g'
)

model_aic = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='aic-g'
)

model_bic = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic-g'
)
```

Compare:

* Network structure.
* Number of edges.
* Score.
* Stability.
* Predictive or inferential performance where appropriate.

Do not compare the raw numerical scores across the three criteria directly.

---

# 28. Score Sensitivity

Structure learning can be sensitive to the selected score.

For example:

```text
Dataset
   │
   ├── BIC
   │     → DAG₁
   │
   ├── AIC
   │     → DAG₂
   │
   └── BDeu
         → DAG₃
```

This does not automatically mean one result is "wrong".

The scoring criteria optimize different objectives and encode different
assumptions.

When structure is scientifically important, consider sensitivity analysis.

A robust workflow can evaluate whether important edges persist across:

* Scores.
* Resampled datasets.
* Reasonable hyperparameter settings.
* Alternative algorithms.

---

# 29. Score Stability

A single best-scoring DAG should not automatically be treated as the true
network.

Evaluate whether important edges are stable.

For example:

```text
Bootstrap dataset 1 → A → B
Bootstrap dataset 2 → A → B
Bootstrap dataset 3 → A → B
...
```

If an edge appears consistently, it provides stronger evidence of structural
stability than an edge appearing in only one fitted network.

Score optimization and structural stability are different concepts.

---

# 30. Complexity and Overfitting

Structure learning can overfit when the search space is large relative to
the available data.

Signs include:

* Very dense networks.
* Many weakly supported edges.
* Large changes in structure under resampling.
* Strong dependence on score settings.
* Poor generalization.

Possible responses include:

* Prefer BIC over an unpenalized likelihood criterion.
* Limit maximum indegree.
* Apply justified blacklist/whitelist constraints.
* Increase sample size.
* Reduce unnecessary variables.
* Perform stability analysis.

Do not remove edges solely because the graph looks complicated.

---

# 31. Gaussian Variance Floor

The Gaussian scoring implementation uses a variance floor to prevent
numerically zero residual variance.

The default is:

```python
variance_floor=1e-12
```

The variance floor must be strictly greater than zero.

This is primarily a numerical safeguard.

Do not interpret the variance floor as a statistical regularization parameter
that determines the network structure.

---

# 32. What Not to Do

### Do not compare raw scores across score families

Incorrect:

```text
BIC = -100
AIC = -90

Therefore AIC is better.
```

Correct:

Compare structures under the same scoring criterion, or compare their
scientific/modeling consequences using an appropriate evaluation strategy.

---

### Do not use `loglik-g` simply because it gives a higher likelihood

Likelihood rewards model fit but does not provide the same complexity penalty as
AIC or BIC.

---

### Do not assume BIC is always correct

BIC is a strong general-purpose choice, but its assumptions and modeling
objective must still be considered.

---

### Do not use Gaussian scores for categorical data

Do not encode:

```text
red   → 0
green → 1
blue  → 2
```

and then treat the result as a continuous Gaussian measurement.

---

### Do not discretize without justification

Preserve continuous information when an appropriate continuous model is
available.

---

### Do not treat the score as causal evidence

A higher-scoring DAG is preferred by the scoring criterion.

That does not establish that its directed edges are causal.

---

# 33. AI Decision Rules

When selecting a bnlearn score:

1. Identify whether the variables are discrete or continuous.
2. If variables are discrete, consider `bic` as the general-purpose default.
3. Consider `aic` when a weaker complexity penalty is desired.
4. Consider `k2` for K2-style Bayesian scoring.
5. Consider `bdeu` when Bayesian Dirichlet score-equivalent scoring is desired.
6. Consider `bds` when its alternative Dirichlet prior treatment is desired.
7. If variables are continuous and a linear Gaussian model is appropriate,
   consider `bic-g`.
8. Use `aic-g` when AIC-style Gaussian model selection is desired.
9. Use `loglik-g` when Gaussian likelihood without AIC/BIC complexity
   penalization is specifically desired.
10. Do not use discrete scores on continuous variables merely for convenience.
11. Do not automatically discretize continuous variables.
12. Keep the score fixed when comparing candidate DAGs.
13. Consider structural stability in addition to the best score.
14. Do not interpret a high score as evidence of causality.
15. Do not compare raw score magnitudes across different scoring families.
16. Ensure Gaussian data are numeric, finite, and free of missing values.
17. Report the selected score when communicating the learned network.

---

# 34. Recommended Defaults

## General discrete Bayesian Network

```python
result = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic'
)
```

Use this as the default starting point unless there is a reason to select
another criterion.

---

## Continuous Gaussian Bayesian Network

```python
result = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic-g'
)
```

Use this when a linear Gaussian model is appropriate.

---

## Bayesian Dirichlet scoring

```python
result = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bdeu'
)
```

Use when the Bayesian Dirichlet scoring assumptions are desired.

---

# 35. Final Checklist

Before selecting a score, verify:

* [ ] Are the variables discrete or continuous?
* [ ] If continuous, is a Gaussian model appropriate?
* [ ] Is the likelihood model appropriate?
* [ ] Is model complexity important?
* [ ] Is AIC or BIC more appropriate for the objective?
* [ ] Is a Bayesian Dirichlet score desired?
* [ ] Is K2 specifically required?
* [ ] Are prior assumptions relevant?
* [ ] Are missing or infinite values present?
* [ ] Are categorical values incorrectly encoded as numbers?
* [ ] Is the same score used when comparing structures?
* [ ] Has score sensitivity been considered?
* [ ] Has structural stability been considered?
* [ ] Are causal claims being kept separate from score optimization?

The score is one component of the modeling workflow. A statistically
appropriate score does not by itself guarantee a correct network.
