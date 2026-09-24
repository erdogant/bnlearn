# -*- coding: utf-8 -*-
"""
Hill Climbing Structure Learning Example
=========================================

Demonstrates score-based Bayesian Network structure learning with
Hill Climb Search in bnlearn, including constraints and score comparison.

Workflow
--------
1. Load a discrete example dataset.
2. Learn a DAG using Hill Climbing.
3. Inspect the learned structure.
4. Constrain the search (max_indegree, fixed_edges, white/black lists).
5. Compare different discrete scoring methods.
6. Fit parameters and run a simple inference query.

Important API notes
-------------------
* Use methodtype= (not method=).
* white_list / black_list require bw_list_method='edges' or 'nodes'.
* start_dag must contain the same variables as the data (all nodes).
"""

# %% Libraries
import matplotlib
matplotlib.use('Agg')
import bnlearn as bn


# %% Load example dataset
df = bn.import_example('sprinkler')

print('\n[bnlearn] > Example data:')
print(df.head())
print('\n[bnlearn] > Shape:', df.shape)
print('\n[bnlearn] > Data types:')
print(df.dtypes)


# %% Basic Hill Climbing
DAG = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic')

print('\n[bnlearn] > Hill Climbing DAG:')
print(DAG['model_edges'])
print('\n[bnlearn] > Adjacency matrix:')
print(DAG['adjmat'])

if 'structure_scores' in DAG:
    print('\n[bnlearn] > Structure scores:')
    print(DAG['structure_scores'])


# %% Tabu list
DAG_tabu = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic',
    tabu_length=10)
print('\n[bnlearn] > Hill Climbing with tabu_length=10:')
print(DAG_tabu['model_edges'])


# %% Limit maximum number of parents
DAG_indegree = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic',
    max_indegree=2)
print('\n[bnlearn] > Hill Climbing with max_indegree=2:')
print(DAG_indegree['model_edges'])


# %% Start from an existing DAG
# start_dag MUST include the same variables as the dataset.
# A partial edge list that omits nodes (e.g. Wet_Grass) raises ValueError.
start_dag = bn.make_DAG(
    [
        ('Cloudy', 'Sprinkler'),
        ('Cloudy', 'Rain'),
        ('Sprinkler', 'Wet_Grass'),
        ('Rain', 'Wet_Grass'),
    ])

DAG_start = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic',
    start_dag=start_dag)
print('\n[bnlearn] > Hill Climbing from start DAG:')
print(DAG_start['model_edges'])


# %% Required edges (must be present)
DAG_fixed = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic',
    fixed_edges=[('Cloudy', 'Rain')])
print('\n[bnlearn] > Hill Climbing with fixed_edges=[(Cloudy, Rain)]:')
print(DAG_fixed['model_edges'])


# %% White list — requires bw_list_method
# white_list = edges that are *allowed* during search (not the same as fixed_edges)
DAG_white = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic',
    white_list=[
        ('Cloudy', 'Sprinkler'),
        ('Cloudy', 'Rain'),
        ('Sprinkler', 'Wet_Grass'),
        ('Rain', 'Wet_Grass'),
    ],
    bw_list_method='edges')
print('\n[bnlearn] > Hill Climbing with white_list:')
print(DAG_white['model_edges'])


# %% Black list — requires bw_list_method
DAG_black = bn.structure_learning.fit(
    df,
    methodtype='hc',
    scoretype='bic',
    black_list=[('Wet_Grass', 'Cloudy')],
    bw_list_method='edges')
print('\n[bnlearn] > Hill Climbing with black_list:')
print(DAG_black['model_edges'])


# %% Compare discrete scoring methods
scoretypes = ['bic', 'k2', 'bdeu']  # subset for a faster demo; aic/bds also valid
results = {}

for scoretype in scoretypes:
    print(f'\n[bnlearn] > Running Hill Climbing with score: {scoretype}')
    result = bn.structure_learning.fit(
        df,
        methodtype='hc',
        scoretype=scoretype)
    results[scoretype] = result
    print(f'{scoretype}: {result["model_edges"]}')

print('\n[bnlearn] > Structure comparison:')
for scoretype, result in results.items():
    print(f'  {scoretype}: {len(result["model_edges"])} edges -> {result["model_edges"]}')


# %% Parameter learning + inference on the first DAG
model = bn.parameter_learning.fit(
    DAG,
    df,
    methodtype='bayes')

print('\n[bnlearn] > Learned CPDs:')
for cpd in model['model'].get_cpds():
    print(cpd)

query = bn.inference.fit(
    model,
    variables=['Wet_Grass'],
    evidence={'Rain': 1})
print('\n[bnlearn] > P(Wet_Grass | Rain=1):')
print(query.df)

samples = bn.sampling(model, n=100, methodtype='bayes')
print('\n[bnlearn] > Synthetic sample shape:', samples.shape)


# %% Summary
print('\n' + '=' * 70)
print('Hill Climbing example completed successfully.')
print('=' * 70)
print('Nodes:', list(model['model'].nodes()))
print('Edges:', list(model['model'].edges()))
print('CPDs:', len(model['model'].get_cpds()))
