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


# Structure learning with black list
model_bl = bnlearn.structure_learning.fit(df, methodtype='hc', white_list=['asia','tub','bronc','xray','smoke'])
bnlearn.plot(model_bl, pos=G['pos'])
bnlearn.compare_networks(model, model_bl, pos=G['pos'])

# %% PARAMETER LEARNING
dir(bnlearn.parameter_learning)

df = bnlearn.import_example()
model = bnlearn.import_DAG('sprinkler', CPD=False)
model_update = bnlearn.parameter_learning.fit(model, df)
bnlearn.plot(model_update)

model_true = bnlearn.import_DAG('sprinkler', CPD=True)

# %% LOAD BIF FILE
DAG = bnlearn.import_DAG('alarm')
df = bnlearn.sampling(DAG, n=1000)
model_update = bnlearn.parameter_learning.fit(DAG, df)
G = bnlearn.plot(model_update)


# %% INFERENCE
DAG = bnlearn.import_DAG('sprinkler')
bnlearn.plot(DAG)
q1 = bnlearn.inference.fit(DAG, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})
q2 = bnlearn.inference.fit(DAG, variables=['Wet_Grass','Rain'], evidence={'Sprinkler':1})

print(q1)
print(q2)


# %% INFERENCE 2
model = bnlearn.import_DAG('asia')
bnlearn.plot(model)
q1 = bnlearn.inference.fit(model, variables=['lung'], evidence={'bronc':1, 'smoke':1})
q2 = bnlearn.inference.fit(model, variables=['bronc','lung'], evidence={'smoke':1, 'xray':0, 'tub':1})
q3 = bnlearn.inference.fit(model, variables=['lung'], evidence={'bronc':1, 'smoke':1})

print(q1)
print(q2)


# %% Create a simple DAG:
DAG = BayesianModel([('Cloudy', 'Sprinkler'),
                       ('Cloudy', 'Rain'),
                       ('Sprinkler', 'Wet_Grass'),
                       ('Rain', 'Wet_Grass')])

# Cloudy
cpt_cloudy = TabularCPD(variable='Cloudy', variable_card=2, values=[[0.5], [0.5]])

# Sprinkler
cpt_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2,
                           values=[[0.5, 0.9], [0.5, 0.1]],
                           evidence=['Cloudy'], evidence_card=[2])
# Rain
cpt_rain = TabularCPD(variable='Rain', variable_card=2,
                      values=[[0.8, 0.2], [0.2, 0.8]],
                      evidence=['Cloudy'], evidence_card=[2])

# Wet Grass
cpt_wet_grass = TabularCPD(variable='Wet_Grass', variable_card=2,
                           values=[[1, 0.1, 0.1, 0.01],
                                  [0, 0.9, 0.9, 0.99]],
                           evidence=['Sprinkler', 'Rain'],
                           evidence_card=[2, 2])

DAG.add_cpds(cpt_cloudy, cpt_sprinkler, cpt_rain, cpt_wet_grass)

