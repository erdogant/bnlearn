# %%
import bnlearn as bn
print(bn.__version__)
print(dir(bn))

# %%
print(dir(bn.structure_learning))
print(dir(bn.parameter_learning))
print(dir(bn.inference))

# %%
# Example dataframe sprinkler_data.csv can be loaded with: 
df = bn.import_example()
# df = pd.read_csv('sprinkler_data.csv')
model = bn.structure_learning.fit(df)
G = bn.plot(model)


# %% Load example dataframe from sprinkler
DAG = bn.import_DAG('sprinkler', verbose=2)
df = bn.sampling(DAG, n=1000, verbose=2)

import bnlearn as bn
model = bn.import_DAG('D://PY/REPOSITORIES/bnlearn/bnlearn/data/hailfinder.bif')
bn.plot(model)

df = bn.import_example('water', n=1000)
# Structure learning
model = bn.structure_learning.fit(df)
# Plot
G = bn.plot(model)

model_hc_bic  = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')

# %% Load example dataframe from sprinkler
df = bn.import_example('sprinkler')
# Structure learning
model = bn.structure_learning.fit(df)
# Plot
G = bn.plot(model)

model_hc_bic  = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')

# %% Try all methods vs score types
model_hc_bic = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
model_hc_k2 = bn.structure_learning.fit(df, methodtype='hc', scoretype='k2')
model_hc_bdeu = bn.structure_learning.fit(df, methodtype='hc', scoretype='bdeu')
model_ex_bic = bn.structure_learning.fit(df, methodtype='ex', scoretype='bic')
model_ex_k2 = bn.structure_learning.fit(df, methodtype='ex', scoretype='k2')
model_ex_bdeu = bn.structure_learning.fit(df, methodtype='ex', scoretype='bdeu')

bn.compare_networks(model, model_hc_bic, pos=G['pos'])


# %% Example with dataset
import bnlearn as bn
DAG = bn.import_DAG('sprinkler')
# Print cpds
bn.print_CPD(DAG)
# plot ground truth
G = bn.plot(DAG)
df = bn.sampling(DAG, n=100)

# %% Inference using custom DAG
import bnlearn as bn
# Load asia DAG
df = bn.import_example('asia')
# from tabulate import tabulate
# print(tabulate(df.head(), tablefmt="grid", headers="keys"))
print(df)

edges = [('smoke', 'lung'),
         ('smoke', 'bronc'),
         ('lung', 'xray'),
         ('bronc', 'xray')]

# edges = [('smoke', 'xray'),
         # ('bronc', 'lung')]

# Make the actual Bayesian DAG
DAG = bn.make_DAG(edges)
# Plot the DAG
bn.plot(DAG)
# Print the CPDs
bn.print_CPD(DAG)

# Learn its parameters from data and perform the inference.
DAG = bn.parameter_learning.fit(DAG, df)
# Print the CPDs
bn.print_CPD(DAG)

# Make inference
q1 = bn.inference.fit(DAG, variables=['lung'], evidence={'smoke':1})
q2 = bn.inference.fit(DAG, variables=['bronc'], evidence={'smoke':1})
q3 = bn.inference.fit(DAG, variables=['lung'], evidence={'smoke':1, 'bronc':1})
q4 = bn.inference.fit(DAG, variables=['bronc','lung'], evidence={'smoke':1, 'xray':0})
q4 = bn.inference.fit(DAG, variables=['bronc','lung'], evidence={'smoke':0, 'xray':0})

# DAGmle = bn.parameter_learning.fit(DAG, df, methodtype='maximumlikelihood')
# bn.print_CPD(DAGmle)
# bn.print_CPD(DAGbay)

# # Make inference
# q1 = bn.inference.fit(DAGmle, variables=['lung'], evidence={'smoke':1})
# q2 = bn.inference.fit(DAGmle, variables=['bronc'], evidence={'smoke':1})
# q3 = bn.inference.fit(DAGmle, variables=['lung'], evidence={'smoke':1, 'bronc':1})
# q4 = bn.inference.fit(DAGmle, variables=['bronc','lung'], evidence={'smoke':1, 'xray':0})

# bn.compare_networks(DAGbay, DAGnew)
# bn.compare_networks(DAGnew, DAGbay)

# %% compute causalities
# Load asia DAG
import bnlearn as bn
df = bn.import_example('asia')
# print(tabulate(df.head(), tablefmt="grid", headers="keys"))
# print(df)

# Structure learning
model = bn.structure_learning.fit(df)
# Plot the DAG
bn.plot(model)
# Print the CPDs
bn.print_CPD(model)
# Comparison
bn.compare_networks(DAG, model)
bn.compare_networks(model, DAG)

# Learn its parameters from data and perform the inference.
DAGnew = bn.parameter_learning.fit(model, df, methodtype='bayes')
# Print the CPDs
bn.print_CPD(DAGnew)

# Make inference
q4 = bn.inference.fit(DAGnew, variables=['bronc','lung'], evidence={'smoke':1, 'xray':0})
# q4 = bn.inference.fit(DAGnew, variables=['bronc','lung'], evidence={'smoke':0, 'xray':0})


# %% Example compare networks
# Load asia DAG
import bnlearn as bn
DAG = bn.import_DAG('asia')
# plot ground truth
G = bn.plot(DAG)
# Sampling
df = bn.sampling(DAG, n=10000)
# Structure learning of sampled dataset
model_sl = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
# Plot based on structure learning of sampled data
bn.plot(model_sl, pos=G['pos'])
# Compare networks and make plot
bn.compare_networks(model, model_sl, pos=G['pos'])


# Structure learning with black list
model_wl = bn.structure_learning.fit(df, methodtype='hc', white_list=['asia','tub','bronc','xray','smoke'])
bn.plot(model_wl, pos=G['pos'])

model_bl = bn.structure_learning.fit(df, methodtype='hc', black_list=['asia','tub'])
bn.plot(model_bl, pos=G['pos'])

# Compare models
bn.compare_networks(model_bl, model_wl, pos=G['pos'])


# %% PARAMETER LEARNING
import bnlearn as bn
df = bn.import_example()
DAG = bn.import_DAG('sprinkler', CPD=False)
model_update = bn.parameter_learning.fit(DAG, df)
bn.plot(model_update)

model_true = bn.import_DAG('sprinkler', CPD=True)

# %% Example with one-hot RAW dataset: sprinkler.
# Load processed data
DAG = bn.import_DAG('sprinkler')

# Read raw data and process
df_raw = bn.import_example(data='sprinkler')
df = bn.df2onehot(df_raw, verbose=0)[1]
df.columns=df.columns.str.replace('_1.0','')

# Learn structure
DAG = bn.structure_learning.fit(df)
# Learn CPDs
model = bn.parameter_learning.fit(DAG, df)
q1 = bn.inference.fit(model, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})
q2 = bn.inference.fit(model, variables=['Wet_Grass','Rain'], evidence={'Sprinkler':1})


q2.values
q2.variables
q2.state_names
q2.name_to_no
q2.no_to_name,

# %% LOAD BIF FILE
DAG = bn.import_DAG('water', verbose=0)
df = bn.sampling(DAG, n=1000)
model_update = bn.parameter_learning.fit(DAG, df)
G = bn.plot(model_update)
bn.print_CPD(model_update)


# %% INFERENCE
DAG = bn.import_DAG('sprinkler')
bn.plot(DAG)
q1 = bn.inference.fit(DAG, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})
q2 = bn.inference.fit(DAG, variables=['Wet_Grass','Rain'], evidence={'Sprinkler': 1})

print(q1)
print(q2)


# %% INFERENCE 2
DAG = bn.import_DAG('asia')
# DAG = bn.import_DAG('sprinkler')
bn.plot(DAG)
q1 = bn.inference.fit(DAG, variables=['lung'], evidence={'bronc':1, 'smoke':1})
q2 = bn.inference.fit(DAG, variables=['bronc','lung'], evidence={'smoke':1, 'xray':0, 'tub':1})
q3 = bn.inference.fit(DAG, variables=['lung'], evidence={'bronc':1, 'smoke':1})

print(q1)
print(q2)


# %% Example with mixed dataset: titanic case
import bnlearn as bn
# Load example mixed dataset
df_raw = bn.import_example(data='titanic')
# Convert to onehot
dfhot, dfnum = bn.df2onehot(df_raw)
# Structure learning
DAG = bn.structure_learning.fit(dfnum, black_list=['Survived','Pclass','Sex','Embarked','Parch'])
# Plot
G = bn.plot(DAG)
# Parameter learning
model = bn.parameter_learning.fit(DAG, dfnum)
# Make inference
q1 = bn.inference.fit(model, variables=['Survived'], evidence={'Sex':1, 'Pclass':1})
q1 = bn.inference.fit(model, variables=['Survived'], evidence={'Sex':0})

bn.print_CPD(model)

# %%
DAG = bn.import_DAG('sprinkler', CPD=False)
DAG = bn.import_DAG('asia')
bn.plot(DAG)
bn.print_CPD(DAG)

df = bn.sampling(DAG, n=1000)
vector = bn.adjmat2vec(DAG['adjmat'])
adjmat = bn.vec2adjmat(vector['source'], vector['target'])

# %%
from pgmpy.factors.discrete import TabularCPD
edges = [('A', 'E'),
         ('S', 'E'),
         ('E', 'O'),
         ('E', 'R'),
         ('O', 'T'),
         ('R', 'T')]

DAG = bn.make_DAG(edges)
bn.plot(DAG)


cpd_A = TabularCPD(variable='A', variable_card=3, values=[[0.3], [0.5], [0.2]])
print(cpd_A)
cpd_S = TabularCPD(variable='S', variable_card=2, values=[[0.6],[ 0.4]])
print(cpd_S)
cpd_E = TabularCPD(variable='E', variable_card=2,
                           values=[
                               [0.75,0.72,0.88,0.64,0.70,0.90],
                               [0.25,0.28,0.12,0.36,0.30,0.10]
                               ],
                           evidence=['A', 'S'],
                           evidence_card=[3, 2])
print(cpd_E)


DAG = bn.make_DAG(DAG, CPD=cpd_A, checkmodel=False)
bn.print_CPD(DAG, checkmodel=False)

# %% Create a simple DAG:
# Building a causal DAG
import bnlearn as bn
from pgmpy.factors.discrete import TabularCPD

edges = [('Cloudy', 'Sprinkler'),
       ('Cloudy', 'Rain'),
       ('Sprinkler', 'Wet_Grass'),
       ('Rain', 'Wet_Grass')]

# DAG = bn.import_DAG('sprinkler')
DAG = bn.make_DAG(edges)
bn.plot(DAG)
bn.print_CPD(DAG)

# Cloudy
cpt_cloudy = TabularCPD(variable='Cloudy', variable_card=2, values=[[0.3], [0.7]])
print(cpt_cloudy)

# Sprinkler
cpt_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2,
                           values=[[0.4, 0.9], [0.6, 0.1]],
                           evidence=['Cloudy'], evidence_card=[2])
print(cpt_sprinkler)

# Rain
cpt_rain = TabularCPD(variable='Rain', variable_card=2,
                      values=[[0.8, 0.2], [0.2, 0.8]],
                      evidence=['Cloudy'], evidence_card=[2])
print(cpt_rain)

# Wet Grass
cpt_wet_grass = TabularCPD(variable='Wet_Grass', variable_card=2,
                           values=[[1, 0.1, 0.1, 0.01],
                                  [0, 0.9, 0.9, 0.99]],
                           evidence=['Sprinkler', 'Rain'],
                           evidence_card=[2, 2])
print(cpt_wet_grass)

# The make_DAG function will required a CPD for each node. If this is not the case, use the checkmodel=False
DAG = bn.make_DAG(DAG, CPD=cpt_cloudy, checkmodel=False)
DAG = bn.make_DAG(DAG, CPD=[cpt_cloudy, cpt_sprinkler], checkmodel=False)
DAG = bn.make_DAG(DAG, CPD=[cpt_cloudy, cpt_sprinkler, cpt_rain, cpt_wet_grass])
bn.print_CPD(DAG)

q1 = bn.inference.fit(DAG, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})

# %% Example from sphinx
# Import dataset
# Import the library
import bnlearn as bn

# Define the network structure
edges = [('Cloudy', 'Sprinkler'),
         ('Cloudy', 'Rain'),
         ('Sprinkler', 'Wet_Grass'),
         ('Rain', 'Wet_Grass')]

# Make the actual Bayesian DAG
DAG = bn.make_DAG(edges)

bn.plot(DAG)


bn.print_CPD(DAG)

# Import the library
from pgmpy.factors.discrete import TabularCPD

# Cloudy
cpt_cloudy = TabularCPD(variable='Cloudy', variable_card=2, values=[[0.5], [0.5]])
print(cpt_cloudy)

# Sprinkler
cpt_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2,
                           values=[[0.5, 0.9],
                                   [0.5, 0.1]],
                           evidence=['Cloudy'], evidence_card=[2])
print(cpt_sprinkler)

# Rain
cpt_rain = TabularCPD(variable='Rain', variable_card=2,
                      values=[[0.8, 0.2],
                              [0.2, 0.8]],
                      evidence=['Cloudy'], evidence_card=[2])
print(cpt_rain)

# Wet Grass
cpt_wet_grass = TabularCPD(variable='Wet_Grass', variable_card=2,
                           values=[[1, 0.1, 0.1, 0.01],
                                   [0, 0.9, 0.9, 0.99]],
                           evidence=['Sprinkler', 'Rain'],
                           evidence_card=[2, 2])
print(cpt_wet_grass)

DAG = bn.make_DAG(DAG, CPD=[cpt_cloudy, cpt_sprinkler, cpt_rain, cpt_wet_grass])


bn.print_CPD(DAG)


q1 = bn.inference.fit(DAG, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})


# %% Example to create a Bayesian Network, learn its parameters from data and perform the inference.

import bnlearn as bn

# Import example dataset
df = bn.import_example('sprinkler')

# Define the network structure
edges = [('Cloudy', 'Sprinkler'),
         ('Cloudy', 'Rain'),
         ('Sprinkler', 'Wet_Grass'),
         ('Rain', 'Wet_Grass')]

# Make the actual Bayesian DAG
DAG = bn.make_DAG(edges)
# [BNLEARN] Bayesian DAG created.

# Print the CPDs
bn.print_CPD(DAG)
# [bn.print_CPD] No CPDs to print. Use bn.plot(DAG) to make a plot.

# Plot the DAG
bn.plot(DAG)

# Parameter learning on the user-defined DAG and input data using maximumlikelihood
DAGmle = bn.parameter_learning.fit(DAG, df, methodtype='maximumlikelihood')
DAGbay = bn.parameter_learning.fit(DAG, df, methodtype='bayes')

# Print the learned CPDs
bn.print_CPD(DAGmle)
bn.print_CPD(DAGbay)

# Make inference
q1 = bn.inference.fit(DAGmle, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})
q1 = bn.inference.fit(DAGbay, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})

print(q1.values)

