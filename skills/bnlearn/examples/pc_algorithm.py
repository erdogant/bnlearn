# -*- coding: utf-8 -*-
"""
PC algorithm for constraint-based structure learning
====================================================

The PC algorithm learns a graph from conditional independence tests.
In bnlearn it is exposed as methodtype='pc' (alias: 'cs' / 'constraintsearch').

CI-test settings belong in params_pc, not in scoretype.

Workflow
--------
1. Load discrete data with known dependencies (sprinkler).
2. Run PC with chi_square CI test.
3. Inspect the returned keys (PC returns extra CPDAG-related fields).
4. Compare with a score-based Hill Climb baseline.
"""

import matplotlib
matplotlib.use('Agg')
import bnlearn as bn


# %% Data — use a dataset with real dependencies
df = bn.import_example('sprinkler')
print('\n[bnlearn] > Data shape:', df.shape)
print(df.head())


# %% PC algorithm
# Constraint-based: relies on conditional independence tests, not a score.
model_pc = bn.structure_learning.fit(
    df,
    methodtype='pc',
    params_pc={'ci_test': 'chi_square', 'alpha': 0.05})

print('\n[bnlearn] > PC return keys:', sorted(model_pc.keys()))
print('[bnlearn] > model_edges:', model_pc.get('model_edges'))
print('[bnlearn] > dag_edges:  ', model_pc.get('dag_edges'))
print('[bnlearn] > pdag_edges: ', model_pc.get('pdag_edges'))
print('[bnlearn] > undirected_edges:', model_pc.get('undirected_edges'))


# %% Compare with Hill Climbing (score-based)
model_hc = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic')
print('\n[bnlearn] > HC (bic) edges:', model_hc['model_edges'])
print('[bnlearn] > PC edges:      ', model_pc.get('model_edges'))


# %% Optional: prune edges that fail an independence test
model_pruned = bn.independence_test(
    model_hc, df,
    test='chi_square',
    alpha=0.05,
    prune=True)
print('\n[bnlearn] > HC edges after independence_test(prune=True):')
print(model_pruned.get('model_edges'))

print('\n' + '=' * 70)
print('PC algorithm example completed successfully.')
print('=' * 70)
