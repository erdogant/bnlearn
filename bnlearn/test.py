import bnlearn as bnlearn
print(bnlearn.__version__)
dir(bnlearn)
dir(bnlearn.structure_learning)

# %% Load example dataframe from sprinkler
df = bnlearn.import_example()
# Structure learning
model = bnlearn.structure_learning.fit(df)
# Plot
G = bnlearn.plot(model)

# %% Try all methods vs score types
model_hc_bic  = bnlearn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
model_hc_k2   = bnlearn.structure_learning.fit(df, methodtype='hc', scoretype='k2')
model_hc_bdeu = bnlearn.structure_learning.fit(df, methodtype='hc', scoretype='bdeu')
model_ex_bic  = bnlearn.structure_learning.fit(df, methodtype='ex', scoretype='bic')
model_ex_k2   = bnlearn.structure_learning.fit(df, methodtype='ex', scoretype='k2')
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
