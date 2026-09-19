import bnlearn as bn

# Load the predictive maintenance dataset
df = bn.import_example('predictive_maintenance')
print(df.head())

del df['UDI']
del df['Product ID']
del df['Type']

# Select only continuous columns
continuous_cols = ['Air temperature [K]', 'Process temperature [K]',
                   'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

df_continuous = df[continuous_cols].copy()

# Learn causal structure using Direct LiNGAM
model = bn.structure_learning.fit(df_continuous, methodtype='direct-lingam')

# Examine the edge weights (causal coefficients)
print(model['adjmat'])
# Plot the DAG
bn.plot(model, title='Predictive Maintenance: Continuous Sensor DAG')

# %%
# Visualize the learned DAG
bn.plot(model, title='direct-lingam recovered DAG')
dotgraph = bn.plot_graphviz(model, edge_labels='weight')
dotgraph


# %%


# Test edge significance
model = bn.independence_test(model, df_continuous, prune=True)
print(model['independence_test'])

# [bnlearn] >Compute edge strength with direct-lingam
#                      source                   target  p_value  dof  stat_test
# 0       Air temperature [K]  Process temperature [K]      1.0    1      False
# 1       Air temperature [K]   Rotational speed [rpm]      1.0    1      False
# 2       Air temperature [K]              Torque [Nm]      1.0    1      False
# 3       Air temperature [K]          Tool wear [min]      1.0    1      False
# 4   Process temperature [K]      Air temperature [K]      1.0    1      False
# 5   Process temperature [K]   Rotational speed [rpm]      1.0    1      False
# 6   Process temperature [K]              Torque [Nm]      1.0    1      False
# 7   Process temperature [K]          Tool wear [min]      1.0    1      False
# 8    Rotational speed [rpm]      Air temperature [K]      1.0    1      False
# 9    Rotational speed [rpm]  Process temperature [K]      1.0    1      False
# 10   Rotational speed [rpm]              Torque [Nm]      1.0    1      False
# 11   Rotational speed [rpm]          Tool wear [min]      1.0    1      False
# 12              Torque [Nm]      Air temperature [K]      1.0    1      False
# 13              Torque [Nm]  Process temperature [K]      1.0    1      False
# 14              Torque [Nm]   Rotational speed [rpm]      1.0    1      False
# 15              Torque [Nm]          Tool wear [min]      1.0    1      False
# 16          Tool wear [min]      Air temperature [K]      1.0    1      False
# 17          Tool wear [min]  Process temperature [K]      1.0    1      False
# 18          Tool wear [min]   Rotational speed [rpm]      1.0    1      False
# 19          Tool wear [min]              Torque [Nm]      1.0    1      False

# %%
import pandas as pd
import bnlearn as bn

# Load the predictive maintenance dataset
df = bn.import_example('predictive_maintenance')
print(df.head())

del df['UDI']
del df['Product ID']
del df['Type']

# Learn causal structure on mixed dataset
model1 = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic-cg')
model2 = bn.structure_learning.fit(df, methodtype='hc', scoretype='aic-cg')
model3 = bn.structure_learning.fit(df, methodtype='hc', scoretype='loglik-cg')
model1 = bn.structure_learning.fit(df, methodtype='hc', scoretype='auto')

# bn.plot(model, title='Predictive Maintenance: Mixed Dataset DAG')
# model = bn.independence_test(model, df, prune=True)
# bn.plot(model, title='Predictive Maintenance: Mixed Dataset DAG')
# model = bn.parameter_learning.fit(model, df, methodtype='bayes')

# print(model['independence_test'])

# Visualize
# bn.plot(model, title='Predictive Maintenance: Mixed Dataset DAG')
dotgraph = bn.plot_graphviz(model1)
dotgraph
dotgraph = bn.plot_graphviz(model2)
dotgraph
dotgraph = bn.plot_graphviz(model3)
dotgraph


# %%

# %%
import pandas as pd
import bnlearn as bn

# Load the predictive maintenance dataset
df = bn.import_example('predictive_maintenance')
print(df.head())

del df['UDI']
del df['Product ID']
del df['Type']

# Learn causal structure on mixed dataset
model = bn.structure_learning.fit(df)

model = bn.independence_test(model, df, prune=True)

# print(model['independence_test'])

# Visualize
# bn.plot(model, title='Predictive Maintenance: Mixed Dataset DAG')
bn.plot(model, interactive=True)
dotgraph = bn.plot_graphviz(model)
dotgraph
