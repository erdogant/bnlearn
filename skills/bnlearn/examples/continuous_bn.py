# -*- coding: utf-8 -*-
"""
Continuous and hybrid Bayesian Network example
==============================================

Full pipeline for continuous (linear-Gaussian) and mixed (Conditional Gaussian)
data with bnlearn:

    structure learning → parameter learning → inference → sampling

Workflow
--------
1. Continuous linear-Gaussian data: HC + bic-g → linear-gaussian params → query / sample.
2. Mixed discrete + continuous data: HC + bic-cg → cg params → query / sample.
3. Optional: DirectLiNGAM for causal orientation only (structure endpoint).
"""

# %% Libraries
import matplotlib
matplotlib.use('Agg')
import bnlearn as bn
import pandas as pd
import numpy as np


# %% Continuous linear-Gaussian data
#
# True process:
#     X1 → X2
#     X1 → X3
#     X2 → X3

np.random.seed(42)
n = 500
X1 = np.random.normal(0, 1, n)
X2 = 2.0 * X1 + np.random.normal(0, 0.5, n)
X3 = 1.0 * X1 + 1.5 * X2 + np.random.normal(0, 0.5, n)
df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3})

print('\n[bnlearn] > Continuous data shape:', df.shape)


# %% Structure learning (Gaussian BIC)
DAG = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic-g')
print('[bnlearn] > Continuous structure edges:', DAG['model_edges'])
print('[bnlearn] > data_type:', DAG['config'].get('data_type'))


# %% Parameter learning (linear-Gaussian)
model = bn.parameter_learning.fit(DAG, df, methodtype='linear-gaussian')
print('[bnlearn] > LG model type:', type(model['model']).__name__)
print('[bnlearn] > LG CPDs:', len(model['model'].get_cpds()))


# %% Inference (conditional mean)
q = bn.inference.fit(model, variables=['X3'], evidence={'X1': 0.0})
print('[bnlearn] > P-mean(X3 | X1=0):', getattr(q, 'means', q))

q_do = bn.inference.fit(model, variables=['X3'], do={'X1': 1.0})
print('[bnlearn] > P-mean(X3 | do(X1=1)):', getattr(q_do, 'means', q_do))


# %% Sampling
df_s = bn.sampling(model, n=100, methodtype='linear-gaussian', seed=0)
print('[bnlearn] > LG samples shape:', df_s.shape)


# %% Mixed / Conditional Gaussian data
rng = np.random.default_rng(0)
fail = rng.integers(0, 2, size=n)
torque = rng.normal(size=n) + fail * 2.0
df_mix = pd.DataFrame({'fail': fail, 'torque': torque})

DAG_mix = bn.structure_learning.fit(df_mix, methodtype='hc', scoretype='bic-cg')
print('\n[bnlearn] > Mixed structure edges:', DAG_mix['model_edges'])
print('[bnlearn] > data_type:', DAG_mix['config'].get('data_type'))

model_mix = bn.parameter_learning.fit(DAG_mix, df_mix, methodtype='cg')
print('[bnlearn] > CG continuous_cpds nodes:',
      [c['variable'] for c in (model_mix.get('continuous_cpds') or [])])

q_cg = bn.inference.fit(model_mix, variables=['torque'], evidence={'fail': 1})
print('[bnlearn] > CG torque | fail=1:', getattr(q_cg, 'means', q_cg))

df_cg = bn.sampling(model_mix, n=100, methodtype='cg', seed=0)
print('[bnlearn] > CG samples shape:', df_cg.shape)


# %% Optional: LiNGAM (structure only)
try:
    DAG_lingam = bn.structure_learning.fit(df, methodtype='direct-lingam')
    print('\n[bnlearn] > DirectLiNGAM edges:', DAG_lingam.get('model_edges'))
except Exception as exc:
    print('\n[bnlearn] > DirectLiNGAM skipped:', type(exc).__name__, exc)


# %% Summary
print('\n' + '=' * 70)
print('Continuous / hybrid pipeline completed.')
print('  linear-gaussian: structure(bic-g) → params(lg) → inference → sampling')
print('  conditional-gaussian: structure(bic-cg) → params(cg) → inference → sampling')
print('=' * 70)
