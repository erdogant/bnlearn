import bnlearn
print(bnlearn.__version__)
dir(bnlearn)


# %%
dir(bnlearn.structure_learning)


# %% Load example dataframe from sprinkler
df = bnlearn.import_example()
# Structure learning
model = bnlearn.structure_learning.fit(df)
# Plot
G = bnlearn.plot(model)


# %% Try all methods vs score types
model_hc_bic = bnlearn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
model_hc_k2 = bnlearn.structure_learning.fit(df, methodtype='hc', scoretype='k2')
model_hc_bdeu = bnlearn.structure_learning.fit(df, methodtype='hc', scoretype='bdeu')
model_ex_bic = bnlearn.structure_learning.fit(df, methodtype='ex', scoretype='bic')
model_ex_k2 = bnlearn.structure_learning.fit(df, methodtype='ex', scoretype='k2')
model_ex_bdeu = bnlearn.structure_learning.fit(df, methodtype='ex', scoretype='bdeu')

bnlearn.compare_networks(model, model_hc_bic, pos=G['pos'])


# %% Example compare networks
# Load asia DAG
model = bnlearn.import_DAG('asia')
# plot ground truth
G = bnlearn.plot(model)
# Sampling
df = bnlearn.sampling(model, n=10000)
# Structure learning of sampled dataset
model_sl = bnlearn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
# Plot based on structure learning of sampled data
bnlearn.plot(model_sl, pos=G['pos'])
# Compare networks and make plot
bnlearn.compare_networks(model, model_sl, pos=G['pos'])


# %% PARAMETER LEARNING
dir(bnlearn.parameter_learning)

df = bnlearn.import_example()
model = bnlearn.import_DAG('sprinkler', CPD=False)
model_update = bnlearn.parameter_learning.fit(model, df)
bnlearn.plot(model_update)

# LOAD BIF FILE
model = bnlearn.import_DAG('alarm')
df = bnlearn.sampling(model, n=1000)
model_update = bnlearn.parameter_learning.fit(model, df)
G = bnlearn.plot(model_update)


# %% INFERENCE
model = bnlearn.import_DAG('sprinkler')
bnlearn.plot(model)
q1 = bnlearn.inference.fit(model, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})
q2 = bnlearn.inference.fit(model, variables=['Wet_Grass','Rain'], evidence={'Sprinkler':1})

print(q1)
print(q2)
