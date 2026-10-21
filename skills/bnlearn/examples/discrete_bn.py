"""
Discrete Bayesian Network Example
=================================

This example demonstrates the complete workflow for learning and using
a discrete Bayesian Network with bnlearn.

Workflow
--------
1. Load a discrete example dataset.
2. Learn the Bayesian Network structure.
3. Learn the parameters (CPDs).
4. Inspect the learned network.
5. Perform probabilistic inference.
6. Generate synthetic data.

"""

# %% Libraries
import matplotlib
matplotlib.use('Agg')
import bnlearn as bn


# %% Load example dataset
df = bn.import_example('sprinkler')

print('\n[bnlearn] > Example data:')
print(df.head())

print('\n[bnlearn] > Data types:')
print(df.dtypes)


# %% Structure learning
# Learn the DAG from the discrete observations.
#
# Hill Climbing with BIC is a standard score-based approach for
# discrete Bayesian Networks.

DAG = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic')

print('\n[bnlearn] > Learned DAG:')
print(DAG['model_edges'])


# %% Plot learned structure (static; avoid interactive backends in scripts)
bn.plot(DAG, params_static={'showplot': False})


# %% Parameter learning
# Learn the Conditional Probability Distributions (CPDs)
# for all variables in the DAG.

model = bn.parameter_learning.fit(
    DAG,
    df,
    methodtype='bayes')

print('\n[bnlearn] > Bayesian Network:')
print(model['model'])


# %% Inspect CPDs
# Each node in a discrete Bayesian Network has a Conditional
# Probability Distribution.

print('\n[bnlearn] > Learned CPDs:')
for cpd in model['model'].get_cpds():
    print(cpd)


# %% Plot Bayesian Network
bn.plot(model, params_static={'showplot': False})


# %% Exact inference
# Calculate:
#
#     P(Wet_Grass | Rain=1, Sprinkler=0)
#
# The result contains the probability distribution over Wet_Grass.

query = bn.inference.fit(
    model,
    variables=['Wet_Grass'],
    evidence={
        'Rain': 1,
        'Sprinkler': 0,
    })

print('\n[bnlearn] > Inference result:')
print(query)

print('\n[bnlearn] > Inference DataFrame:')
print(query.df)


# %% Joint inference
# Query multiple variables simultaneously.

query_joint = bn.inference.fit(
    model,
    variables=['Wet_Grass', 'Rain'],
    evidence={
        'Sprinkler': 1,
    })

print('\n[bnlearn] > Joint inference:')
print(query_joint.df)


# %% Conditional sampling
# Generate synthetic observations from the learned Bayesian Network.
#
# Without evidence, sampling generates observations from the complete
# joint distribution.

samples = bn.sampling(
    model,
    n=1000,
    methodtype='bayes')

print('\n[bnlearn] > Synthetic samples:')
print(samples.head())

print('\n[bnlearn] > Sample shape:')
print(samples.shape)


# %% Sampling conditioned on evidence
# Generate observations conditioned on:
#
#     Rain=1
#     Cloudy=0
#
# For Bayesian sampling, evidence uses rejection sampling.

conditional_samples = bn.sampling(
    model,
    n=100,
    methodtype='bayes',
    evidence={
        'Rain': 1,
        'Cloudy': 0,
    })

print('\n[bnlearn] > Conditional synthetic samples:')
print(conditional_samples.head())


# %% Gibbs sampling
# Gibbs sampling generates observations from the learned Bayesian Network.
#
# The current bnlearn sampling implementation does not support evidence
# with Gibbs sampling.

gibbs_samples = bn.sampling(
    model,
    n=100,
    methodtype='gibbs')

print('\n[bnlearn] > Gibbs samples:')
print(gibbs_samples.head())


# %% Summary
print('\n' + '=' * 70)
print('Discrete Bayesian Network example completed successfully.')
print('=' * 70)

print('\nNodes:')
print(list(model['model'].nodes()))

print('\nEdges:')
print(list(model['model'].edges()))

print('\nNumber of CPDs:')
print(len(model['model'].get_cpds()))

print('\nSynthetic data:')
print(samples.shape)

print('\nConditional synthetic data:')
print(conditional_samples.shape)

print('\nGibbs samples:')
print(gibbs_samples.shape)