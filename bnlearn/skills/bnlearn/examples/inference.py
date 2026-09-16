# -*- coding: utf-8 -*-
"""
Bayesian Network inference example
==================================

Shows how to answer P(query | evidence) with bn.inference.fit.

Important
---------
* The model must already contain CPDs (from parameter_learning.fit, or from
  make_DAG which installs default uniform CPDs when CPD=None).
* Evidence keys must be variable names present in the model.
* Evidence values must be valid states for those variables.

Workflow
--------
1. Build a small DAG with default CPDs via make_DAG.
2. Query a marginal.
3. Query with evidence.
4. Optionally refit CPDs from sampled data and query again.
"""

import matplotlib
matplotlib.use('Agg')
import bnlearn as bn


# %% Network structure
edges = [
    ('A', 'C'),
    ('B', 'C'),
    ('C', 'D'),
]

# make_DAG installs default (uniform) TabularCPDs when CPD is omitted.
model = bn.make_DAG(edges, verbose=0)
print('[bnlearn] > Nodes:', list(model['model'].nodes()))
print('[bnlearn] > Edges:', list(model['model'].edges()))
print('[bnlearn] > CPDs:', len(model['model'].get_cpds()))


# %% Marginal query: P(D)
# Note: pass evidence={} for a pure marginal. evidence=None (or omitting it)
# raises AttributeError in current bnlearn (it calls evidence.keys()).
query = bn.inference.fit(
    model,
    variables=['D'],
    evidence={},
    verbose=0,
)
print('\nP(D):')
print(query.df)


# %% Conditional query: P(D | A=1)
query = bn.inference.fit(
    model,
    variables=['D'],
    evidence={'A': 1},
    verbose=0,
)
print('\nP(D | A=1):')
print(query.df)


# %% Conditional query: P(D | A=1, B=1)
query = bn.inference.fit(
    model,
    variables=['D'],
    evidence={'A': 1, 'B': 1},
    verbose=0,
)
print('\nP(D | A=1, B=1):')
print(query.df)


# %% Note
# To estimate CPDs from data instead of default uniforms:
#   df = bn.sampling(model, n=2000, methodtype='bayes', verbose=0)
#   model = bn.parameter_learning.fit(model, df, methodtype='bayes', verbose=0)
#   query = bn.inference.fit(model, variables=['D'], evidence={'A': 1}, verbose=0)

# %% Continuous (linear-Gaussian) inference
import numpy as np
import pandas as pd

rng = np.random.default_rng(5)
n = 300
x = rng.normal(size=n)
y = 1.5 * x + rng.normal(scale=0.4, size=n)
df_c = pd.DataFrame({'X': x, 'Y': y})
DAG_c = bn.structure_learning.fit(df_c, methodtype='hc', scoretype='bic-g', verbose=0)
model_c = bn.parameter_learning.fit(DAG_c, df_c, methodtype='linear-gaussian', verbose=0)
q_c = bn.inference.fit(model_c, variables=['Y'], evidence={'X': 0.0}, verbose=0)
print('\nContinuous P-mean(Y | X=0):', getattr(q_c, 'means', q_c))
q_do = bn.inference.fit(model_c, variables=['Y'], do={'X': 1.0}, verbose=0)
print('Continuous P-mean(Y | do(X=1)):', getattr(q_do, 'means', q_do))


print('\n' + '=' * 70)
print('Inference example completed successfully.')
print('=' * 70)
