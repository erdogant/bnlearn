# %%
import bnlearn as bn
print(bn.__version__)
print(dir(bn))

print(dir(bn.structure_learning))
print(dir(bn.parameter_learning))
print(dir(bn.inference))


# %% Coloring networks
import bnlearn as bn

# Load example dataset
df = bn.import_example(data='asia')

# Structure learning
model = bn.structure_learning.fit(df)

node_properties = bn.get_node_properties(model)
# model = bn.get_node_properties(model, node_size=100)

# Set color
node_properties['xray']['node_color']='#8A0707'
node_properties['xray']['node_size']=50
bn.plot(model, interactive=False)
bn.plot(model, interactive=False, node_properties=node_properties)
bn.plot(model, interactive=False, node_properties=node_properties, params_static = {'font_color':'#8A0707'})
bn.plot(model, interactive=False, params_static = {'width':15, 'height':8, 'font_size':14, 'font_family':'times new roman', 'alpha':0.8, 'node_shape':'o', 'facecolor':'white', 'font_color':'r'})
bn.plot(model, interactive=False, node_color='#8A0707', node_size=800, params_static = {'width':15, 'height':8, 'font_size':14, 'font_family':'times new roman', 'alpha':0.8, 'node_shape':'o', 'facecolor':'white', 'font_color':'#000000'})

# Add some parameters for the interactive plot
node_properties = bn.get_node_properties(model)
node_properties['xray']['node_color']='#8A0707'
node_properties['xray']['node_size']=50
bn.plot(model, interactive=True)
bn.plot(model, interactive=True, node_properties=node_properties)
bn.plot(model, interactive=True, node_color='#8A0707')
bn.plot(model, interactive=True, node_size=5)
bn.plot(model, interactive=True, node_properties=node_properties, node_size=20)
bn.plot(model, interactive=True, params_interactive = {'height':'800px', 'width':'70%', 'layout':None, 'bgcolor':'#0f0f0f0f'})


# %% Save and load trained models
import bnlearn as bn
# Import example
df = bn.import_example(data='asia')
# Learn structure
model = bn.structure_learning.fit(df, methodtype='tan', class_node='lung')
# Save model
bn.save(model, filepath='bnlearn_model', overwrite=True)
# Load model
model = bn.load(filepath='bnlearn_model')


# %% CHECK DIFFERENCES PGMPY vs. BNLEARN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pgmpy.estimators import TreeSearch
from pgmpy.models import BayesianModel
import networkx as nx
import bnlearn as bnlearn
from pgmpy.inference import VariableElimination
from pgmpy.estimators import BDeuScore, K2Score, BicScore
import bnlearn as bn


df=bnlearn.import_example(data='andes')

# PGMPY
est = TreeSearch(df)
dag = est.estimate(estimator_type="tan",class_node='DISPLACEM0')
bnq = BayesianModel(dag.edges())
bnq.fit(df, estimator=None) # None means maximum likelihood estimator
bn_infer = VariableElimination(bnq)
q = bn_infer.query(variables=['DISPLACEM0'], evidence={'RApp1':1})
print(q)

# BNLEARN
model = bnlearn.structure_learning.fit(df, methodtype='tan', class_node='DISPLACEM0', scoretype='bic')
model_bn = bnlearn.parameter_learning.fit(model, df, methodtype='ml') # maximum likelihood estimator
query=bnlearn.inference.fit(model_bn, variables=['DISPLACEM0'], evidence={'RApp1':1})

# DAG COMPARISON
assert np.all(model_bn['adjmat']==model['adjmat'])
assert dag.edges()==model['model'].edges()
assert dag.edges()==model['model_edges']

# COMPARE THE CPDs names
qbn_cpd = []
bn_cpd = []
for cpd in bnq.get_cpds(): qbn_cpd.append(cpd.variable)
for cpd in model_bn['model'].get_cpds(): bn_cpd.append(cpd.variable)

assert len(bn_cpd)==len(qbn_cpd)
assert np.all(np.isin(bn_cpd, qbn_cpd))

# COMPARE THE CPD VALUES
nr_diff = 0
for cpd_bnlearn in model_bn['model'].get_cpds():
    for cpd_pgmpy in bnq.get_cpds():
        if cpd_bnlearn.variable==cpd_pgmpy.variable:
            assert np.all(cpd_bnlearn.values==cpd_pgmpy.values)
            # if not np.all(cpd_bnlearn.values==cpd_pgmpy.values):
                # print('%s-%s'%(cpd_bnlearn.variable, cpd_pgmpy.variable))
                # print(cpd_bnlearn)
                # print(cpd_pgmpy)
                # nr_diff=nr_diff+1
                # input('press enter to see the next difference in CPD.')


#%% Example of interactive plotting
from pgmpy.estimators import TreeSearch
from pgmpy.models import BayesianModel
import bnlearn as bn

# Import example
df = bn.import_example(data='asia')

# Do the tan learning
est = TreeSearch(df)
dag = est.estimate(estimator_type="tan",class_node='lung')

# And now with bnlearn
model = bn.structure_learning.fit(df, methodtype='tan', class_node='lung')

# Compare results
assert dag.edges()==model['model'].edges()
assert dag.edges()==model['model_edges']

# from pgmpy.inference import VariableElimination
# bnp = BayesianModel(dag.edges())
# bn_infer = VariableElimination(bnp)
# bnp.fit(df)
# a = bn_infer.query(variables=['Target'], evidence={'X':'c'})
# print(a)

#%% Example of interactive plotting
import bnlearn as bn

# Load example dataset
df = bn.import_example(data='asia')

# Structure learning
model = bn.structure_learning.fit(df)

bn.plot(model, interactive=False, node_size=800)

# Add some parameters for the interactive plot
bn.plot(model, interactive=True, node_size=10, params_interactive = {'height':'600px'})

# Add more parameters for the interactive plot
bn.plot(model, interactive=True, node_size=10, params_interactive = {'height':'800px', 'width':'70%', 'notebook':False, 'heading':'bnlearn causal diagram', 'layout':None, 'font_color': False, 'bgcolor':'#ffffff'})

# %% TAN : Tree-augmented Naive Bayes (TAN)
# https://pgmpy.org/examples/Structure%20Learning%20with%20TAN.html
import bnlearn as bn

df = bn.import_example()
# Structure learning
model = bn.structure_learning.fit(df)
# bn.plot(model)
bn.plot(model, interactive=True)
# bn.plot(model, interactive=True, params = {'height':'800px'})


# %% Large dataset
import pandas as pd
df=pd.read_csv('c:/temp/features.csv')
model = bn.structure_learning.fit(df.iloc[:,0:1000], methodtype='cl', root_node='21')
model = bn.structure_learning.fit(df.iloc[:,0:100], methodtype='cs')
bn.plot(model)

# %% White_list edges
import bnlearn as bn
DAG = bn.import_DAG('asia')
# plot ground truth
df = bn.sampling(DAG, n=1000)

# Structure learning with black list
model = bn.structure_learning.fit(df, methodtype='hc', white_list=[('tub','lung'), ('smoke','bronc')], bw_list_method='edges')
bn.plot(model)
model = bn.structure_learning.fit(df, methodtype='hc', white_list=['tub','lung', 'smoke','bronc'], bw_list_method='nodes')
bn.plot(model)
model = bn.structure_learning.fit(df, methodtype='hc')
bn.plot(model, node_color='#000000')


# %% TAN : Tree-augmented Naive Bayes (TAN)
# https://pgmpy.org/examples/Structure%20Learning%20with%20TAN.html
import bnlearn as bn

df = bn.import_example()
# Structure learning
model = bn.structure_learning.fit(df, methodtype='tan', root_node='Cloudy', class_node='Rain', verbose=0)
bn.plot(model)
bn.plot(model, interactive=True, node_size=10)

# %% Download example
import bnlearn as bn
examples = ['titanic', 'sprinkler', 'alarm', 'andes', 'asia', 'sachs', 'water', 'miserables']
for example in examples:
    df = bn.import_example(data=example)
    # assert ~df.empty

# %%
import bnlearn as bn
df = bn.import_example()
model = bn.structure_learning.fit(df)
model = bn.structure_learning.fit(df, methodtype='hc')

# %% Predict
import bnlearn as bn

df = bn.import_example('asia')
edges = [('smoke', 'lung'),
         ('smoke', 'bronc'),
         ('lung', 'xray'),
         ('bronc', 'xray')]

# Make the actual Bayesian DAG
DAG = bn.make_DAG(edges, verbose=0)
model = bn.parameter_learning.fit(DAG, df, verbose=3)
# Generate some data based on DAG
df = bn.sampling(model, n=1000)
# Make predictions
Pout = bn.predict(model, df, variables=['bronc','xray'])
# query = bnlearn.inference.fit(model, variables=['bronc','xray'], evidence=evidence, to_df=False, verbose=0)
# print(query)


# %% topological sort example
edges = [('1', '2'),
         ('1', '3'),
         ('2', '4'),
         ('2', '3'),
         ('3', '4'),
         ('3', '5'),
         ]

# Make the actual Bayesian DAG
DAG = bn.make_DAG(edges, verbose=0)
# Plot
bn.plot(DAG, node_size=2000)
# Topological ordering
bn.topological_sort(DAG)

bn.topological_sort(DAG, '3')

# %%
import bnlearn as bn
DAG = bn.import_DAG('sprinkler', verbose=0)

bn.topological_sort(DAG, 'Rain')
bn.topological_sort(DAG)

# Different inputs
bn.topological_sort(DAG['adjmat'], 'Rain')
bn.topological_sort(bn.adjmat2vec(DAG['adjmat']), 'Rain')

# %%

import bnlearn as bn
DAG = bn.import_DAG('sprinkler')
df = bn.sampling(DAG, n=1000, verbose=0)
model = bn.structure_learning.fit(df, methodtype='chow-liu', root_node='Wet_Grass')
G = bn.plot(model)
bn.topological_sort(model, 'Rain')

# %%
# Example dataframe sprinkler_data.csv can be loaded with:
df = bn.import_example()
# df = pd.read_csv('sprinkler_data.csv')
model = bn.structure_learning.fit(df)
G = bn.plot(model)

# %% Load example dataframe from sprinkler
import bnlearn as bn
DAG = bn.import_DAG('sprinkler', verbose=0)
df = bn.sampling(DAG, n=1000, verbose=0)

# Structure learning
model = bn.structure_learning.fit(df, verbose=0)
# Plot
node_properties = bn.get_node_properties(model)
G = bn.plot(model)
model_hc_bic = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic', verbose=0)

node_properties['Cloudy']['node_size']=2000
node_properties['Cloudy']['node_color']='r'
G = bn.plot(model, node_properties=node_properties)

# %% Chow-Liu algorithm
DAG = bn.import_DAG('sprinkler', verbose=0)
df = bn.sampling(DAG, n=1000, verbose=0)

# Structure learning
model_hc_bic = bn.structure_learning.fit(df, methodtype='cl', root_node='Cloudy', verbose=0)
G = bn.plot(model)

# %% Load example dataframe from sprinkler
import bnlearn as bn
DAG = bn.import_DAG('alarm', verbose=0)
to_vector = bn.adjmat2vec(DAG['adjmat'])
to_adjmat = bn.vec2adjmat(to_vector['source'], to_vector['target'])

# %% Load example dataframe from sprinkler
import bnlearn as bn
df = bn.import_example('sprinkler')
# Structure learning
model = bn.structure_learning.fit(df, verbose=0)
# Plot
G = bn.plot(model, verbose=0)

model_hc_bic  = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic', verbose=0)

# %% Try all methods vs score types
import bnlearn as bn
df = bn.import_example()

model_hc_bic = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
model_hc_k2 = bn.structure_learning.fit(df, methodtype='hc', scoretype='k2')
model_hc_bdeu = bn.structure_learning.fit(df, methodtype='hc', scoretype='bdeu')
model_ex_bic = bn.structure_learning.fit(df, methodtype='ex', scoretype='bic')
model_ex_k2 = bn.structure_learning.fit(df, methodtype='ex', scoretype='k2')
model_ex_bdeu = bn.structure_learning.fit(df, methodtype='ex', scoretype='bdeu')
model_cs_k2 = bn.structure_learning.fit(df, methodtype='cs', scoretype='k2')
model_cs_bdeu = bn.structure_learning.fit(df, methodtype='cs', scoretype='bdeu')
model_cl = bn.structure_learning.fit(df, methodtype='cl', root_node='Cloudy')

G = bn.plot(model_hc_bic, verbose=0)

bn.compare_networks(model_hc_bic, model_cl, pos=G['pos'], verbose=0)

# %% Example with dataset
import bnlearn as bn
DAG = bn.import_DAG('sprinkler', verbose=3)
# Print cpds
bn.print_CPD(DAG)
# plot ground truth
G = bn.plot(DAG, verbose=0)
df = bn.sampling(DAG, n=100, verbose=3)

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
DAG = bn.make_DAG(edges, verbose=0)
bn.save(DAG, overwrite=True)
DAG1 = bn.load()

# Plot the DAG
bn.plot(DAG1, verbose=0)
# Print the CPDs
bn.print_CPD(DAG)

# Sampling
# df_sampling = bn.sampling(DAG, n=1000)

# Learn its parameters from data and perform the inference.
DAG = bn.parameter_learning.fit(DAG, df, verbose=3)
# Print the CPDs
bn.print_CPD(DAG)

# Sampling
df_sampling = bn.sampling(DAG, n=1000)

# Make inference
q1 = bn.inference.fit(DAG, variables=['lung'], evidence={'smoke':1}, verbose=3)
q2 = bn.inference.fit(DAG, variables=['bronc'], evidence={'smoke':1}, verbose=0)
q3 = bn.inference.fit(DAG, variables=['lung'], evidence={'smoke':1, 'bronc':1})
q4 = bn.inference.fit(DAG, variables=['bronc','lung'], evidence={'smoke':1, 'xray':0})
q4 = bn.inference.fit(DAG, variables=['bronc','lung'], evidence={'smoke':0, 'xray':0})

bn.topological_sort(DAG)

bn.query2df(q4)

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

# %% predict

df = bn.sampling(DAG, n=100)
out = bn.predict(DAG, df, variables='bronc')
out = bn.predict(DAG, df, variables=['bronc','xray'])
out = bn.predict(DAG, df, variables=['bronc','xray','smoke'])

print('done\n\n')
print(out)

# %% compute causalities
# Load asia DAG
import bnlearn as bn
df = bn.import_example('asia', verbose=0)
# print(tabulate(df.head(), tablefmt="grid", headers="keys"))
# print(df)

# Structure learning
model = bn.structure_learning.fit(df, verbose=0)
# Plot the DAG
bn.plot(model, verbose=0, interactive=True, node_color='#000000')
# Print the CPDs
bn.print_CPD(model)
# Comparison

# Learn its parameters from data and perform the inference.
DAG = bn.parameter_learning.fit(model, df, methodtype='bayes', verbose=0)
# Print the CPDs
bn.print_CPD(DAG)

# Nothing is changed for the DAG. Only the CPDs are estimated now.
bn.compare_networks(DAG, model, verbose=0)

# Make inference
q4 = bn.inference.fit(DAG, variables=['bronc','lung'], evidence={'smoke':1, 'xray':0}, verbose=3)
# q4 = bn.inference.fit(DAG, variables=['bronc','lung','xray'], evidence={'smoke':1}, verbose=3)
# q4 = bn.inference.fit(DAGnew, variables=['bronc','lung'], evidence={'smoke':0, 'xray':0})

# pd.DataFrame(index=q4.variables, data=q4.values, columns=q4.variables)

# %% Example compare networks
# Load asia DAG
import bnlearn as bn
DAG = bn.import_DAG('asia')
# plot ground truth
G = bn.plot(DAG)
# Sampling
df = bn.sampling(DAG, n=10000)

# Structure learning
model = bn.structure_learning.fit(df, verbose=0)
# Structure learning of sampled dataset
model_sl = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
# Plot based on structure learning of sampled data
bn.plot(model_sl, pos=G['pos'], interactive=True, params_interactive = {'height':'800px'})
# Compare networks and make plot
bn.compare_networks(model, model_sl, pos=G['pos'])


# Structure learning with black list
model_wl = bn.structure_learning.fit(df, methodtype='hc', white_list=['asia','tub','bronc','xray','smoke'], bw_list_method='edges')
bn.plot(model_wl, pos=G['pos'])

model_bl = bn.structure_learning.fit(df, methodtype='hc', black_list=['asia','tub'], bw_list_method='edges')
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
G = bn.plot(model_update, interactive=True)
bn.print_CPD(model_update)


# %% INFERENCE
DAG = bn.import_DAG('sprinkler')
bn.plot(DAG)
q1 = bn.inference.fit(DAG, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})
q2 = bn.inference.fit(DAG, variables=['Wet_Grass','Rain'], evidence={'Sprinkler': 1})

print(q1)
print(q1.df)
print(q2)
print(q2.df)


# %% INFERENCE 2
DAG = bn.import_DAG('asia')
# DAG = bn.import_DAG('sprinkler')
bn.plot(DAG)
q1 = bn.inference.fit(DAG, variables=['lung'], evidence={'bronc':1, 'smoke':1})
q2 = bn.inference.fit(DAG, variables=['bronc','lung'], evidence={'smoke':1, 'xray':0, 'tub':1})
q3 = bn.inference.fit(DAG, variables=['lung'], evidence={'bronc':1, 'smoke':1})

print(q1)
print(q1.df)
print(q2)
print(q2.df)


# %% Example with mixed dataset: titanic case
import bnlearn as bn
# Load example mixed dataset
df_raw = bn.import_example(data='titanic')
# Convert to onehot
dfhot, dfnum = bn.df2onehot(df_raw)

dfnum.loc[0:50,'Survived'] = 2
# Structure learning
# DAG = bn.structure_learning.fit(dfnum, methodtype='cl', black_list=['Embarked','Parch','Name'], root_node='Survived', bw_list_method='nodes')
DAG = bn.structure_learning.fit(dfnum, methodtype='hc', black_list=['Embarked','Parch','Name'], bw_list_method='edges')
# Plot
G = bn.plot(DAG)
# Parameter learning
model = bn.parameter_learning.fit(DAG, dfnum)
# Make inference
q1 = bn.inference.fit(model, variables=['Survived'], evidence={'Sex':True, 'Pclass':True}, verbose=0)
q2 = bn.inference.fit(model, variables=['Survived'], evidence={'Sex':0}, verbose=0)

print(q1)
print(q1.df)
# bn.print_CPD(model)

# Create test dataset
Xtest = bn.sampling(model, n=100)
# Predict the whole dataset
Pout = bn.predict(model, Xtest, variables=['Survived'])


# %%
import bnlearn as bn
DAG = bn.import_DAG('sprinkler', CPD=True)
# DAG = bn.import_DAG('asia')
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

# %%