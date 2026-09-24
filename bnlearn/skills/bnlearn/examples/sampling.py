# -*- coding: utf-8 -*-
"""
Sampling from a Bayesian Network
================================

Generate synthetic observations with bn.sampling.

Rules enforced by bnlearn
-------------------------
* methodtype='bayes' supports optional evidence (rejection sampling).
* methodtype='gibbs' does NOT support evidence.
* Evidence variable names and states must exist in the model.
* Jointly impossible evidence raises ValueError.

Workflow
--------
1. Build a DAG (default uniform CPDs via make_DAG).
2. Unconditional sampling.
3. Conditional sampling with evidence.
4. Gibbs sampling (no evidence).
"""

import matplotlib
matplotlib.use('Agg')
import bnlearn as bn


# %% Network
edges = [
    ('A', 'C'),
    ('B', 'C'),
    ('C', 'D'),
]
model = bn.make_DAG(edges)
print('[bnlearn] > Nodes:', list(model['model'].nodes()))
print('[bnlearn] > Edges:', list(model['model'].edges()))
print('[bnlearn] > CPDs:', len(model['model'].get_cpds()))


# %% Unconditional sampling
df = bn.sampling(model, n=1000, methodtype='bayes')
print('\n[bnlearn] > Unconditional samples:', df.shape)
print(df.head())
print('\nValue counts:')
for col in df.columns:
    print(f'\n{col}:\n{df[col].value_counts()}')


# %% Conditional sampling (rejection) — methodtype must be 'bayes'
cond = bn.sampling(
    model,
    n=200,
    methodtype='bayes',
    evidence={'A': 1, 'B': 0})
print('\n[bnlearn] > Conditional samples (A=1, B=0):', cond.shape)
assert (cond['A'] == 1).all()
assert (cond['B'] == 0).all()
print(cond.head())


# %% Gibbs sampling (no evidence allowed)
gibbs = bn.sampling(model, n=200, methodtype='gibbs')
print('\n[bnlearn] > Gibbs samples:', gibbs.shape)
print(gibbs.head())

# %% Continuous (linear-Gaussian) sampling
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 200
x = rng.normal(size=n)
y = 0.5 * x + rng.normal(scale=0.5, size=n)
df_c = pd.DataFrame({'X': x, 'Y': y})
DAG_c = bn.structure_learning.fit(df_c, methodtype='hc', scoretype='bic-g')
model_c = bn.parameter_learning.fit(DAG_c, df_c, methodtype='linear-gaussian')
df_lg = bn.sampling(model_c, n=50, methodtype='linear-gaussian', seed=0)
print('\n[bnlearn] > Linear-Gaussian samples:', df_lg.shape)
print(df_lg.head())


print('\n' + '=' * 70)
print('Sampling example completed successfully.')
print('=' * 70)
