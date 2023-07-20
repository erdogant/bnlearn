import pandas as pd
import numpy as np
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork, NaiveBayes
from pgmpy.estimators import ExhaustiveSearch, HillClimbSearch, TreeSearch
from pgmpy.factors.discrete import TabularCPD

# %% issue #84
# Load library
from pgmpy.factors.discrete import TabularCPD
import bnlearn as bn

# Create some edges, all starting from the same root node: A
edges = [('A', 'B'), ('A', 'C'), ('A', 'D')]
DAG = bn.make_DAG(edges, methodtype='naivebayes')

# Set CPDs
cpd_A = TabularCPD(variable='A', variable_card=3, values=[[0.3], [0.5], [0.2]])
print(cpd_A)
cpd_B = TabularCPD(variable='B', variable_card=2, values=[[0.4, 0.9], [0.6, 0.1]], evidence=['A'], evidence_card=[2])
print(cpd_B)
cpd_C = TabularCPD(variable='C', variable_card=2, values=[[0.4, 0.9], [0.6, 0.1]], evidence=['A'], evidence_card=[2])
print(cpd_C)
cpd_D = TabularCPD(variable='D', variable_card=2, values=[[0.4, 0.9], [0.6, 0.1]], evidence=['A'], evidence_card=[2])
print(cpd_D)

DAG = bn.make_DAG(DAG, CPD=[cpd_A, cpd_B, cpd_C, cpd_D], checkmodel=True)
# Plot the CPDs as a sanity check
bn.print_CPD(DAG, checkmodel=True)
# Plot the DAG
bn.plot(DAG)


# %%
import bnlearn as bn

# Load example dataset
Xy_train = bn.import_example('titanic')
Xy_train.drop(labels='Cabin', axis=1, inplace=True)
Xy_train = Xy_train.dropna(axis=0)

tarvar='Survived'
model = bn.structure_learning.fit(Xy_train, 
                                  methodtype='tan', 
                                  class_node = 'Survived')
model = bn.parameter_learning.fit(model, 
                                  Xy_train, 
                                  methodtype='bayes',
                                  scoretype='bdeu')
y_train_pred = bn.predict(model, Xy_train, variables = tarvar, verbose=4)



# %%
import bnlearn as bn

# Load example dataset
df = bn.import_example('sprinkler')

edges = [('Cloudy', 'Sprinkler'),
         ('Cloudy', 'Rain'),
         ('Sprinkler', 'Wet_Grass'),
         ('Rain', 'Wet_Grass')]

# Make the actual Bayesian DAG
DAG = bn.make_DAG(edges)
model = bn.parameter_learning.fit(DAG, df)
# Print CPDs
CPD = bn.print_CPD(model)

bn.check_model(CPD) # Input should be model
bn.check_model(model)

bn.plot(model, interactive=True, params_interactive={'filepath': r'c:/temp/bnlearn.html'})

# %%
import bnlearn as bn

# Example dataset
source=['Cloudy','Cloudy','Sprinkler','Rain']
target=['Sprinkler','Rain','Wet_Grass','Wet_Grass']
repeats=[1,2,1,3]

adjmat = bn.vec2adjmat(source, target, weights=repeats)

# Convert into sparse datamatrix
df = bn.vec2df(source, target, weights=repeats)
df = bn.adjmat2vec(adjmat)
# Make DAG
DAG = bn.make_DAG(list(zip(source, target)), verbose=0)
# Make plot
bn.plot(DAG, interactive=True, params_interactive={'filepath': r'c:/temp/bnlearn.html'})
# bn.plot(DAG, interactive=True)
# bn.plot(DAG, interactive=False)


# %% Import examples
import bnlearn as bn
df = bn.import_example(data='sprinkler', n=1000)

DAG = bn.import_DAG('Sprinkler')

# %% Working with continues data
import bnlearn as bn
df = bn.import_example(data='auto_mpg')

edges = [
    ("cylinders", "displacement"),
    ("displacement", "model_year"),
    ("displacement", "weight"),
    ("displacement", "horsepower"),
    ("weight", "model_year"),
    ("weight", "mpg"),
    ("horsepower", "acceleration"),
    ("mpg", "model_year"),
]

# Create DAG based on edges
DAG = bn.make_DAG(edges)

# df_num = df.select_dtypes(include='float64').columns.values
continuous_columns = ["mpg", "displacement", "horsepower", "weight", "acceleration"]
# Discretize the continous columns
df_disc = bn.discretize(df, edges, continuous_columns, max_iterations=1)

# Fit model based on DAG and discretize the continous columns
model_mle = bn.parameter_learning.fit(DAG, df_disc)
# model_mle = bn.parameter_learning.fit(DAG, data_disc, methodtype="maximumlikelihood")

# Make plot
bn.plot(model_mle)
bn.plot(model_mle, interactive=True)
# bn.independence_test(model, df)

print(model_mle["model"].get_cpds("mpg"))

print("Weight categories: ", df_disc["weight"].dtype.categories)
evidence = {"weight": bn.discretize_value(df_disc["weight"], 3000.0)}
print(evidence)
print(bn.inference.fit(model_mle, variables=["mpg"], evidence=evidence, verbose=0))



# %%
import bnlearn as bn
shapes = [(10000, 37), (10000, 223), (10000, 8), (10000, 11), (10000, 32), (352, 3)]
for i, data in enumerate(['alarm', 'andes', 'asia', 'sachs', 'water', 'stormofswords']):
    print(data)
    df = bn.import_example(data=data)
    assert df.shape==shapes[i]


# %% Notebook example
# Example dataframe sprinkler_data.csv can be loaded with: 
import bnlearn as bn
df = bn.import_example()
# df = pd.read_csv('sprinkler_data.csv')
model = bn.structure_learning.fit(df, verbose=0)

# Set some colors to the edges and nodes
node_properties = bn.get_node_properties(model)
node_properties['Sprinkler']['node_color']='#FF0000'

edge_properties = bn.get_edge_properties(model)
edge_properties[('Cloudy', 'Rain')]['color']='#FF0000'
edge_properties[('Cloudy', 'Rain')]['weight']=5

G = bn.plot(model,
            node_properties=node_properties,
            edge_properties=edge_properties,
            interactive=True,
            params_interactive={'notebook': False},
            )

# G = bn.plot(model, interactive=True, params_interactive={'notebook': True}, node_properties=node_properties, edge_properties=edge_properties)

# %%
import bnlearn as bn
model = bn.import_DAG('asia')

# Make single inference
query = bn.inference.fit(model, variables=['lung', 'bronc', 'xray'], evidence={'smoke':1})
print(query)
print(bn.query2df(query))

# Lets create an example dataset with 100 samples and make inferences on the entire dataset.
df = bn.sampling(model, n=10000)

# Each sample will be assesed and the states with highest probability are returned.
Pout = bn.predict(model, df, variables=['lung'])

print(Pout)
#     Cloudy  Rain         p
# 0        0     0  0.647249
# 1        0     0  0.604230
# ..     ...   ...       ...
# 998      0     0  0.604230
# 999      1     1  0.878049

# %%
import bnlearn as bn
model = bn.import_DAG('asia', CPD=True)
CPDs = bn.print_CPD(model)
CPDs.keys()
# dict_keys(['asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray'])
print(CPDs['smoke'])
#    smoke    p
# 0      0  0.5
# 1      1  0.5

print(model["model"].get_cpds('asia'))
print(model["model"].get_cpds(np.array(list(CPDs.keys()))[1]))

# %% Issue 65: return (fig, ax)
import bnlearn as bn
model = bn.import_DAG('sprinkler', CPD=True)
df = bn.sampling(model, n=1000, methodtype='bayes')
fig = bn.plot(model, params_static={'visible': True})
fig2 = bn.plot(model, interactive=True)


# %% Issue Mail:
# from pgmpy.models import BayesianNetwork

# # create instance
# hd_bn = BayesianNetwork()

# # create dag
# hd_bn.add_edges_from(hd_edges)

# # structure score
# print(f'K2 score: {structure_score(hd_bn,bn_df,scoring_method="k2")}')
# print(f'BDeu score: {metrics.structure_score(hd_bn,bn_df,scoring_method="bdeu")}')
# print(f'BDS score: {metrics.structure_score(hd_bn,bn_df,scoring_method="bds")}')
# print(f'BIC score: {metrics.structure_score(hd_bn,bn_df,scoring_method="bic")}')

# K2 score: -3469.9237414747818
# BDeu score: -3501.3510316331763
# BDS score: -3579.8380555853423
# BIC score: -3616.8520164722863

# %% Issue #60: Floating Point Errors.
import bnlearn as bn

# Load example dataset
df = bn.import_example('sprinkler')

edges = [('Cloudy', 'Sprinkler'),
         ('Cloudy', 'Rain'),
         ('Sprinkler', 'Wet_Grass'),
         ('Rain', 'Wet_Grass')]

# Make the actual Bayesian DAG
DAG = bn.make_DAG(edges)
model = bn.parameter_learning.fit(DAG, df)
# Print CPDs
CPD = bn.print_CPD(model)

bn.check_model(CPD) # Input should be model
bn.check_model(model)

bn.plot(model, interactive=True)

# %%
import bnlearn as bn
# print(bn.__version__)
# print(dir(bn.structure_learning))
# print(dir(bn.parameter_learning))
# print(dir(bn.inference))

# %%
# Load asia DAG
df = bn.import_example(data='asia')
# Structure learning of sampled dataset
model = bn.structure_learning.fit(df)
# Make plot
G = bn.plot(model)
G = bn.plot(model, interactive=True)

# %% Issue MAIL: Store CPDs after printing.
import bnlearn as bn

# Load example dataset
df = bn.import_example('sprinkler')

edges = [('Cloudy', 'Sprinkler'),
         ('Cloudy', 'Rain'),
         ('Sprinkler', 'Wet_Grass'),
         ('Rain', 'Wet_Grass')]

# Make the actual Bayesian DAG
DAG = bn.make_DAG(edges)
model = bn.parameter_learning.fit(DAG, df)

# Print CPDs
CPD = bn.print_CPD(model)

from tabulate import tabulate
print(tabulate(CPD['Cloudy'], tablefmt="grid", headers="keys"))


# %% Issue 37
import bnlearn as bn
# Load example DataFrame
df_as = bn.import_example('titanic')
dfhot, dfnum = bn.df2onehot(df_as)
# Train model
model_as = bn.structure_learning.fit(dfnum, methodtype='hc', scoretype='bic')
model_as_p = bn.parameter_learning.fit(model_as, dfnum, methodtype='bayes')
# Do the inference
query = bn.inference.fit(model_as_p, variables=['Sex', 'Parch'], evidence={'Survived':0, 'Pclass':1}, to_df=True)
print(query)
print(query.df)
bn.query2df(query)
bn.query2df(query, variables=['Sex'])


query = bn.inference.fit(model_as_p, variables=['Sex'], evidence={'Survived':0, 'Pclass':1})
query = bn.inference.fit(model_as_p, variables=['Parch'], evidence={'Survived':0, 'Pclass':1})


# %% Issue 57

import bnlearn as bn

# Load example dataset
df = bn.import_example('sprinkler')

edges = [('Cloudy', 'Sprinkler'),
         ('Cloudy', 'Rain'),
         ('Sprinkler', 'Wet_Grass'),
         ('Rain', 'Wet_Grass')]

# Make the actual Bayesian DAG
DAG = bn.make_DAG(edges, verbose=3, methodtype='markov')
model = bn.parameter_learning.fit(DAG, df, verbose=3, methodtype='bayes')

# Sampling
df = bn.sampling(model, n=100, methodtype='gibbs', verbose=3)

# Print CPDs
bn.print_CPD(model)

# Plot
bn.plot(model)


# %%
import bnlearn as bn
model = bn.import_DAG('water', verbose=0)
# Sampling
df = bn.sampling(model, n=1000, methodtype='bayes', verbose=3)


# %% Naive Bayesian model
import bnlearn as bn
df = bn.import_example('random_discrete')
# Structure learning
model = bn.structure_learning.fit(df, methodtype='nb', root_node="B", verbose=4, n_jobs=1)
model = bn.structure_learning.fit(df, methodtype='hc', verbose=4, n_jobs=1)
model = bn.structure_learning.fit(df, methodtype='cs', verbose=4, n_jobs=1)
model = bn.structure_learning.fit(df, methodtype='cl', verbose=4, n_jobs=1)
model = bn.structure_learning.fit(df, methodtype='tan', class_node="B", verbose=4, n_jobs=1)
model = bn.structure_learning.fit(df, methodtype='ex', verbose=4, n_jobs=1)
model = bn.independence_test(model, df, prune=True)
# Plot
bn.plot(model)


# %%
# from pgmpy.estimators import BayesianEstimator
# from pgmpy.utils import get_example_model
# from pgmpy.models import BayesianNetwork
# model = get_example_model('alarm')
# df = model.simulate(int(1e4))

# new_model = BayesianNetwork(model.edges())
# cpds = BayesianEstimator(new_model, df).get_parameters(prior_type='dirichlet', pseudo_counts=1)
# new_model.add_cpds(*cpds)
 
# %% LOAD BIF FILE
import bnlearn as bn
DAG = bn.import_DAG('water', verbose=0)
# Sampling
df = bn.sampling(DAG, n=1000)
# Parameter learning
model = bn.parameter_learning.fit(DAG, df, scoretype='bdeu', smooth=None)

# %%
import bnlearn as bn
df = bn.import_example('asia')
model = bn.structure_learning.fit(df)
bn.plot(model)
bn.plot(model, params_static={'layout':'spectral_layout'})
bn.plot(model, params_static={'layout':'planar_layout'})
bn.plot(model, params_static={'layout':'kamada_kawai_layout'})
bn.plot(model, params_static={'layout':'spring_layout'})
bn.plot(model, params_static={'layout':'circular_layout', "figsize": (15, 10)})


# %%

# Load example dataset
df = bn.import_example('sprinkler')

edges = [('Cloudy', 'Sprinkler'),
         ('Cloudy', 'Rain'),
         ('Sprinkler', 'Wet_Grass'),
         ('Rain', 'Wet_Grass')]

# Make the actual Bayesian DAG
DAG = bn.make_DAG(edges, verbose=0, methodtype='bayes')
model = bn.parameter_learning.fit(DAG, df, verbose=3)
bn.print_CPD(DAG)

model = bn.parameter_learning.fit(DAG, df, verbose=3)
bn.print_CPD(model)
bn.plot(model)

# %%
import bnlearn as bn

# Example dataset
source=['Cloudy','Cloudy','Sprinkler','Rain']
target=['Sprinkler','Rain','Wet_Grass','Wet_Grass']
weights=[1,2,1,3]

# Convert into sparse datamatrix
df = bn.vec2df(source, target, weights=weights)
# Make DAG
DAG = bn.make_DAG(list(zip(source, target)), verbose=0)
# Make plot
bn.plot(DAG, interactive=True)
bn.plot(DAG, interactive=False)


# %%
import bnlearn as bn

raw = bn.import_example('stormofswords')
# Convert raw data into sparse datamatrix
df = bn.vec2adjmat(raw['source'], raw['target'], raw['weight'])
# df = bn.vec2df(raw['source'], raw['target'], raw['weight'])
# Make the actual Bayesian DAG
DAG = bn.make_DAG(list(zip(raw['source'], raw['target'])), verbose=0)
# Make plot
bn.plot(DAG, interactive=True)
bn.plot(DAG, interactive=False)

# Parameter learning
model = bn.parameter_learning.fit(DAG, df, verbose=3)
# Structure learning
DAG_learned = bn.structure_learning.fit(df.iloc[:, 0:50])
# Keep only significant edges
DAG_learned = bn.independence_test(DAG_learned, df, prune=True)
# Plot
bn.plot(DAG_learned, interactive=True)

fig_properties = bn.plot(model, params_static={'layout':'spectral_layout'})
fig_properties = bn.plot(model, params_static={'layout':'planar_layout'})
fig_properties = bn.plot(model, params_static={'layout':'kamada_kawai_layout', 'figsize':(25,15)})
fig_properties = bn.plot(model, params_static={'layout':'spring_layout'})
fig_properties = bn.plot(model, params_static={'layout':'circular_layout'})


# Generate some data based on DAG
# df1 = bn.sampling(model, n=1000)
# Make predictions
# print(query)
query = bn.inference.fit(DAG, variables=['Grenn'], evidence={'Aemon': 1, 'Samwell': 1})
print(query)
query.df

# %%
import bnlearn as bn
df = bn.import_example()
# Structure learning
model = bn.structure_learning.fit(df, methodtype='tan', root_node='Cloudy', class_node='Rain', verbose=0)
bn.plot(model)
bn.plot(model, interactive=False, node_size=10)


# %% Check the stochastic component of bnlearn
DAG = bn.import_DAG('sprinkler', verbose=0)
df = bn.import_example('sprinkler')
adjmats = []

for i in range(0, 10):
    # print(i)
    df = bn.sampling(DAG, n=10000)
    model = bn.structure_learning.fit(df)
    # df = bn.import_example()
    # model = bn.structure_learning.fit(df)
    adjmats.append(model['adjmat'].values.ravel())

adjmats = np.array(adjmats)
adjmat = (adjmats[:, None, :] != adjmats).sum(2)

print(adjmat.sum(axis=0))
# print(adjmat.sum(axis=1))


# %% LOAD BIF FILE

# Example 1: Plot the TRUE DAG
DAG_1 = bn.import_DAG('sprinkler', verbose=0)
graph = bn.plot(DAG_1)

graph = bn.plot(DAG_1, params_static={'layout':'kamada_kawai_layout'})
graph = bn.plot(DAG_1, params_static={'layout':'spring_layout'})


# Example 2: Download small spinkler example with 1000 samples
df = bn.import_example('sprinkler')
DAG_2 = bn.structure_learning.fit(df)
bn.plot(DAG_2, pos=graph['pos'])

# Example 3: Generate Download small spinkler example with 5000 samples
df = bn.sampling(DAG_1, n=5000)
DAG_3 = bn.structure_learning.fit(df)
bn.plot(DAG_3, pos=graph['pos'])


# %% LOAD BIF FILE
DAG = bn.import_DAG('water', verbose=0)
# Sampling
df = bn.sampling(DAG, n=1000)
# Parameter learning
model = bn.parameter_learning.fit(DAG, df)
print(len(model['model_edges']))
# G = bn.plot(model)

# Test for independence
model1 = bn.independence_test(model, df, prune=False)
# bn.plot(model1, pos=G['pos']);
print(len(model1['model_edges']))
print(model1['independence_test'].shape)

# Test for independence
model2 = bn.independence_test(model, df, prune=True)
# bn.plot(model2, pos=G['pos']);
print(len(model2['model_edges']))
print(model2['independence_test'].shape)


assert model['model_edges']==model1['model_edges']
assert len(model1['model_edges'])==model1['independence_test'].shape[0]
assert len(model2['model_edges'])==model2['independence_test'].shape[0]
assert len(model2['model_edges'])<len(model1['model_edges'])
assert len(model2['model_edges'])<len(model['model_edges'])

# Plot
G = bn.plot(model1, interactive=True)
G = bn.plot(model2, interactive=False)


# %% Adjust some edge properties
# Load asia DAG
df = bn.import_example(data='sprinkler')
# Structure learning of sampled dataset
model = bn.structure_learning.fit(df)
# Compute edge strength with the chi_square test statistic
model = bn.independence_test(model, df, test='chi_square', prune=True)
# Make plot
bn.plot(model)
bn.plot(model, interactive=True, params_interactive={'filepath': r'c:/temp/output.html'})

# %% Adjust some edge properties
# Load asia DAG
df = bn.import_example(data='asia')
# Structure learning of sampled dataset
model = bn.structure_learning.fit(df)
# Make plot
G = bn.plot(model)

# Compute associations with the chi_square test statistic
model1 = bn.independence_test(model, df, test='chi_square', prune=False)
# Make plot
bn.plot(model1, pos=G['pos'])

# Compute associations with the chi_square test statistic
model2 = bn.independence_test(model, df, test='chi_square', prune=True)
# Make plot
bn.plot(model2, pos=G['pos'])


# %%
# edges=list(model['model_edges'])

# hn = hnet(multtest='holm', white_list=np.unique(edges))
# hn.association_learning(df)

# adjmat = hn.results['simmatP_cat'].copy()
# adjmat = hn.results['simmatLogP_cat'].copy()
# adjmat = adjmat.iloc[:, ismember(model['adjmat'].columns.values, adjmat.columns.values)[1]]
# adjmat = adjmat.iloc[ismember(model['adjmat'].index.values, adjmat.index.values)[1], :]
# adjmat = adjmat *model['adjmat']

# assert np.all(model['adjmat'].index.values==adjmat.index.values)
# assert np.all(model['adjmat'].columns.values==adjmat.columns.values)

# model['adjmat'] = adjmat
# # edge_properties = bn.get_edge_properties(model)
# G = bn.plot(model)


# scores = correlation_score(model['model'], df, test="chi_square", significance_level=0.05, return_summary=True)
# f1score = correlation_score(model['model'], df, test="log_likelihood", significance_level=0.05, return_summary=False)
# structure_score(model['model'], df, scoring_method='bic')


# edges=list(model['model_edges'])


# for edge in edges:
#     P=[]
#     print(edge)
#     uiyc = df[edge[1]].unique()
#     uiyc = uiyc[uiyc>0]

#     uicats = df[edge[0]].unique()
#     uicats = uicats[uicats>0]
#     for uicat in uicats:
#         for y in uiyc:
#             outtest = _prob_hypergeo(df[edge[0]]==uicat, df[edge[1]]==y)
#             print(outtest)
#             P.append(outtest['P'])


# y = df['Survived'].values
# out = hn.enrichment(df, y)


# # plot static
# G = bn.plot(model)


# %%
# %% Hypergeometric test


# def _prob_hypergeo(datac, yc):
#     """Compute hypergeometric Pvalue.

#     Description
#     -----------
#     Suppose you have a lot of 100 floppy disks (M), and you know that 20 of them are defective (n).
#     What is the prbability of drawing zero to 2 floppy disks (N=2), if you select 10 at random (N).
#     P=hypergeom.sf(2,100,20,10)

#     """
#     P = np.nan
#     logP = np.nan
#     M = len(yc)  # Population size: Total number of samples, eg total number of genes; 10000
#     n = np.sum(datac)  # Number of successes in population, known in pathway, eg 2000
#     N = np.sum(yc)  # sample size: Random variate, eg clustersize or groupsize, over expressed genes, eg 300
#     X = np.sum(np.logical_and(yc, datac.values)) - 1  # Let op, de -1 is belangrijk omdatje P<X wilt weten ipv P<=X. Als je P<=X doet dan kan je vele false positives krijgen als bijvoorbeeld X=1 en n=1 oid

#     # Do the hypergeo-test
#     if np.any(yc) and (X>0):
#         P = hypergeom.sf(X, M, n, N)
#         logP = hypergeom.logsf(X, M, n, N)

#     # Store
#     out = {}
#     out['category_label']=datac.name
#     out['P']=P
#     out['logP']=logP
#     out['overlap_X']=X
#     out['popsize_M']=M
#     out['nr_succes_pop_n']=n
#     out['samplesize_N']=N
#     out['dtype']='categorical'
#     return(out)


# %% Naive Bayesian model
df = bn.import_example('random_discrete')
# Structure learning
model = bn.structure_learning.fit(df, methodtype='naivebayes', root_node="B")
model = bn.independence_test(model, df, prune=True)
# Plot
bn.plot(model)

# %% Naive Bayesian model
from pgmpy.factors.discrete import TabularCPD
import bnlearn as bn

edges = [('A', 'B'), ('A', 'C'), ('A', 'D')]
DAG = bn.make_DAG(edges, methodtype='naivebayes')
bn.plot(DAG)

cpd_A = TabularCPD(variable='A', variable_card=3, values=[[0.3], [0.5], [0.2]])
print(cpd_A)
cpd_B = TabularCPD(variable='B', variable_card=2, values=[[0.4, 0.9], [0.6, 0.1]], evidence=['A'], evidence_card=[2])
print(cpd_B)
cpd_C = TabularCPD(variable='C', variable_card=2, values=[[0.4, 0.9], [0.6, 0.1]], evidence=['A'], evidence_card=[2])
print(cpd_C)
cpd_D = TabularCPD(variable='D', variable_card=2, values=[[0.4, 0.9], [0.6, 0.1]], evidence=['A'], evidence_card=[2])
print(cpd_D)

DAG = bn.make_DAG(DAG, CPD=[cpd_A, cpd_B, cpd_C, cpd_D], checkmodel=True)
bn.print_CPD(DAG, checkmodel=True)

# %% Adjust some edge properties

# Load asia DAG
df = bn.import_example(data='asia')
# Structure learning of sampled dataset
model = bn.structure_learning.fit(df)
# plot static
G = bn.plot(model)
# Compute associations with the chi_square test statistic
model = bn.independence_test(model, df)

# Set some edge properties
edge_properties = bn.get_edge_properties(model)
edge_properties['either', 'xray']['color']='#8A0707'
edge_properties['either', 'xray']['weight']=4
edge_properties['bronc', 'dysp']['weight']=10
edge_properties['bronc', 'dysp']['color']='#8A0707'

# Set some node properties
node_properties = bn.get_node_properties(model)
node_properties['xray']['node_color']='#8A0707'
node_properties['xray']['node_size']=20

# Plot
params_static={'edge_alpha': 0.6, 'arrowstyle': '->', 'arrowsize': 60}
bn.plot(model, interactive=False, node_properties=node_properties, edge_properties=edge_properties, params_static=params_static)
# bn.plot(model, interactive=True, node_properties=node_properties, edge_properties=edge_properties, params_static=params_static);

# %% TAN : Tree-augmented Naive Bayes (TAN)
# https://pgmpy.org/examples/Structure%20Learning%20with%20TAN.html
# https://pgmpy.org/models/naive.html

df = bn.import_example()
# Structure learning
model = bn.structure_learning.fit(df, methodtype='tan', root_node='Cloudy', class_node='Rain', verbose=0)
bn.plot(model)
bn.plot(model, interactive=True, node_size=10)


# %% Coloring networks

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
bn.plot(model, interactive=False, node_properties=node_properties, params_static={'font_color': '#8A0707'})
bn.plot(model, interactive=False, params_static={'width': 15, 'height': 8, 'font_size': 14, 'font_family': 'times new roman', 'alpha': 0.8, 'node_shape': 'o', 'facecolor': 'white', 'font_color': 'r'})
bn.plot(model, interactive=False, node_color='#8A0707', node_size=800, params_static={'width': 15, 'height': 8, 'font_size': 14, 'font_family': 'times new roman', 'alpha': 0.8, 'node_shape': 'o', 'facecolor': 'white', 'font_color': '#000000'})

# Add some parameters for the interactive plot
node_properties = bn.get_node_properties(model)
node_properties['xray']['node_color']='#8A0707'
node_properties['xray']['node_size']=50
bn.plot(model, interactive=True)
bn.plot(model, interactive=True, node_properties=node_properties)
bn.plot(model, interactive=True, node_color='#8A0707')
bn.plot(model, interactive=True, node_size=5)
bn.plot(model, interactive=True, node_properties=node_properties, node_size=20)
bn.plot(model, interactive=True, params_interactive={'height': '800px', 'width': '70%', 'layout': None, 'bgcolor': '#0f0f0f0f'})


# %% Save and load trained models
# Import example
df = bn.import_example(data='asia')
# Learn structure
model = bn.structure_learning.fit(df, methodtype='tan', class_node='lung')
# Save model
bn.save(model, filepath='bnlearn_model', overwrite=True)
# Load model
model = bn.load(filepath='bnlearn_model')


# %% CHECK DIFFERENCES PGMPY vs. BNLEARN


df=bn.import_example(data='andes')

# PGMPY
est = TreeSearch(df)
dag = est.estimate(estimator_type="tan", class_node='DISPLACEM0')
bnq = BayesianNetwork(dag.edges())
bnq.fit(df, estimator=None)  # None means maximum likelihood estimator
bn_infer = VariableElimination(bnq)
q = bn_infer.query(variables=['DISPLACEM0'], evidence={'RApp1': 1})
print(q)

# BNLEARN
model = bn.structure_learning.fit(df, methodtype='tan', class_node='DISPLACEM0', scoretype='bic')
model_bn = bn.parameter_learning.fit(model, df, methodtype='ml')  # maximum likelihood estimator
query=bn.inference.fit(model_bn, variables=['DISPLACEM0'], evidence={'RApp1': 1})

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


# %% Example of interactive plotting

# Import example
df = bn.import_example(data='asia')

# Do the tan learning
est = TreeSearch(df)
dag = est.estimate(estimator_type="tan", class_node='lung')

# And now with bnlearn
model = bn.structure_learning.fit(df, methodtype='tan', class_node='lung')

# Compare results
assert dag.edges()==model['model'].edges()
assert list(dag.edges())==model['model_edges']

# from pgmpy.inference import VariableElimination
# bnp = BayesianNetwork(dag.edges())
# bn_infer = VariableElimination(bnp)
# bnp.fit(df)
# a = bn_infer.query(variables=['Target'], evidence={'X':'c'})
# print(a)

# %% Example of interactive plotting

# Load example dataset
df = bn.import_example(data='asia')

# Structure learning
model = bn.structure_learning.fit(df)

model = bn.independence_test(model, df)

bn.plot(model, interactive=False, node_size=800)

# Add some parameters for the interactive plot
bn.plot(model, interactive=True, node_size=10, params_interactive={'height': '600px'})

# Add more parameters for the interactive plot
bn.plot(model, interactive=True, node_size=10, params_interactive={'height': '800px', 'width': '70%', 'notebook': False, 'heading': 'bnlearn causal diagram', 'layout': None, 'font_color': False, 'bgcolor': '#ffffff'})

# %% TAN : Tree-augmented Naive Bayes (TAN)
# https://pgmpy.org/examples/Structure%20Learning%20with%20TAN.html

df = bn.import_example()
# Structure learning
model = bn.structure_learning.fit(df)
# bn.plot(model)
bn.plot(model, interactive=True)
# bn.plot(model, interactive=True, params = {'height':'800px'})


# %% Large dataset
df=pd.read_csv('c:/temp/features.csv')
model = bn.structure_learning.fit(df.iloc[:, 0:1000], methodtype='cl', root_node='21')
model = bn.structure_learning.fit(df.iloc[:, 0:100], methodtype='cs')
bn.plot(model)

# %% White_list edges
DAG = bn.import_DAG('asia')
# plot ground truth
df = bn.sampling(DAG, n=1000)

# Structure learning with black list
model = bn.structure_learning.fit(df, methodtype='hc', white_list=[('tub', 'lung'), ('smoke', 'bronc')], bw_list_method='edges')
bn.plot(model)
model = bn.structure_learning.fit(df, methodtype='hc', white_list=['tub', 'lung', 'smoke', 'bronc'], bw_list_method='nodes')
bn.plot(model)
model = bn.structure_learning.fit(df, methodtype='hc')
bn.plot(model, node_color='#000000')


# %% TAN : Tree-augmented Naive Bayes (TAN)
# https://pgmpy.org/examples/Structure%20Learning%20with%20TAN.html

df = bn.import_example()
# Structure learning
model = bn.structure_learning.fit(df, methodtype='tan', root_node='Cloudy', class_node='Rain', verbose=0)
bn.plot(model)
bn.plot(model, interactive=True, node_size=10)

# %% Download example
examples = ['titanic', 'sprinkler', 'alarm', 'andes', 'asia', 'sachs', 'water', 'miserables']
for example in examples:
    df = bn.import_example(data=example)
    # assert ~df.empty

# %%
df = bn.import_example()
model = bn.structure_learning.fit(df)
model = bn.structure_learning.fit(df, methodtype='hc')

# %% Predict
df = bn.import_example('asia')
edges = [('smoke', 'lung'),
         ('smoke', 'bronc'),
         ('lung', 'xray'),
         ('bronc', 'xray')]


# Make the actual Bayesian DAG
DAG = bn.make_DAG(edges, verbose=0, methodtype='bayes')
model = bn.parameter_learning.fit(DAG, df, verbose=3)
# Generate some data based on DAG
df = bn.sampling(model, n=1000)
# Make predictions
Pout = bn.predict(model, df, variables=['bronc', 'xray'])
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
DAG = bn.import_DAG('sprinkler', verbose=0)

bn.topological_sort(DAG, 'Rain')
bn.topological_sort(DAG)

# Different inputs
bn.topological_sort(DAG['adjmat'], 'Rain')
bn.topological_sort(bn.adjmat2vec(DAG['adjmat']), 'Rain')

# %%

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
DAG = bn.import_DAG('alarm', verbose=0)
to_vector = bn.adjmat2vec(DAG['adjmat'])
to_adjmat = bn.vec2adjmat(to_vector['source'], to_vector['target'])

# %% Load example dataframe from sprinkler
df = bn.import_example('sprinkler')
# Structure learning
model = bn.structure_learning.fit(df, verbose=0)
# Plot
G = bn.plot(model, verbose=0)

model_hc_bic = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic', verbose=0)

# %% Try all methods vs score types
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
DAG = bn.import_DAG('sprinkler', verbose=3)
# Print cpds
bn.print_CPD(DAG)
# plot ground truth
G = bn.plot(DAG, verbose=0)
df = bn.sampling(DAG, n=100, verbose=3)

# %% Inference using custom DAG
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
q1 = bn.inference.fit(DAG, variables=['lung'], evidence={'smoke': 1}, verbose=3)
q2 = bn.inference.fit(DAG, variables=['bronc'], evidence={'smoke': 1}, verbose=0)
q3 = bn.inference.fit(DAG, variables=['lung'], evidence={'smoke': 1, 'bronc': 1})
q4 = bn.inference.fit(DAG, variables=['bronc', 'lung'], evidence={'smoke': 1, 'xray': 0})
q4 = bn.inference.fit(DAG, variables=['bronc', 'lung'], evidence={'smoke': 0, 'xray': 0})

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
out = bn.predict(DAG, df, variables=['bronc', 'xray'])
out = bn.predict(DAG, df, variables=['bronc', 'xray', 'smoke'])

print('done\n\n')
print(out)

# %% compute causalities
# Load asia DAG
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
q4 = bn.inference.fit(DAG, variables=['bronc', 'lung'], evidence={'smoke': 1, 'xray': 0}, verbose=3)
# q4 = bn.inference.fit(DAG, variables=['bronc','lung','xray'], evidence={'smoke':1}, verbose=3)
# q4 = bn.inference.fit(DAGnew, variables=['bronc','lung'], evidence={'smoke':0, 'xray':0})

# pd.DataFrame(index=q4.variables, data=q4.values, columns=q4.variables)

# %% Example compare networks
# Load asia DAG
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
bn.plot(model_sl, pos=G['pos'], interactive=True, params_interactive={'height': '800px'})
# Compare networks and make plot
bn.compare_networks(model, model_sl, pos=G['pos'])


# Structure learning with black list
model_wl = bn.structure_learning.fit(df, methodtype='hc', white_list=['asia', 'tub', 'bronc', 'xray', 'smoke'], bw_list_method='edges')
bn.plot(model_wl, pos=G['pos'])

model_bl = bn.structure_learning.fit(df, methodtype='hc', black_list=['asia', 'tub'], bw_list_method='edges')
bn.plot(model_bl, pos=G['pos'])

# Compare models
bn.compare_networks(model_bl, model_wl, pos=G['pos'])


# %% PARAMETER LEARNING
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
df.columns=df.columns.str.replace('_1.0', '')

# Learn structure
DAG = bn.structure_learning.fit(df)
# Learn CPDs
model = bn.parameter_learning.fit(DAG, df)
q1 = bn.inference.fit(model, variables=['Wet_Grass'], evidence={'Rain': 1, 'Sprinkler': 0, 'Cloudy': 1})
q2 = bn.inference.fit(model, variables=['Wet_Grass', 'Rain'], evidence={'Sprinkler': 1})


q2.values
q2.variables
q2.state_names
q2.name_to_no
q2.no_to_name,

# %% LOAD BIF FILE
DAG = bn.import_DAG('water', verbose=0)
# Sampling
df = bn.sampling(DAG, n=1000)
# Parameter learning
model = bn.parameter_learning.fit(DAG, df)
G = bn.plot(model)

# Test for independence
model1 = bn.independence_test(model, df, prune=False)
len(model1['model_edges'])
G = bn.plot(model1)

# Test for independence
model2 = bn.independence_test(model, df, prune=True)
len(model2['model_edges'])
G = bn.plot(model2)

# Plot
G = bn.plot(model)
G = bn.plot(model, interactive=True)
bn.print_CPD(model_update)


# %% INFERENCE
DAG = bn.import_DAG('sprinkler')
bn.plot(DAG)
q1 = bn.inference.fit(DAG, variables=['Wet_Grass'], evidence={'Rain': 1, 'Sprinkler': 0, 'Cloudy': 1})
q2 = bn.inference.fit(DAG, variables=['Wet_Grass', 'Rain'], evidence={'Sprinkler': 1})

print(q1)
print(q1.df)
print(q2)
print(q2.df)


# %% INFERENCE 2
DAG = bn.import_DAG('asia')
# DAG = bn.import_DAG('sprinkler')
bn.plot(DAG)
q1 = bn.inference.fit(DAG, variables=['lung'], evidence={'bronc': 1, 'smoke': 1})
q2 = bn.inference.fit(DAG, variables=['bronc', 'lung'], evidence={'smoke': 1, 'xray': 0, 'tub': 1})
q3 = bn.inference.fit(DAG, variables=['lung'], evidence={'bronc': 1, 'smoke': 1})

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

dfnum.loc[0:50, 'Survived'] = 2
# Structure learning
# DAG = bn.structure_learning.fit(dfnum, methodtype='cl', black_list=['Embarked','Parch','Name'], root_node='Survived', bw_list_method='nodes')
DAG = bn.structure_learning.fit(dfnum, methodtype='hc', black_list=['Embarked', 'Parch', 'Name'], bw_list_method='edges')
# Plot
G = bn.plot(DAG)
G = bn.plot(DAG, interactive=True)
# Parameter learning
model = bn.parameter_learning.fit(DAG, dfnum)
# Make inference
q1 = bn.inference.fit(model, variables=['Survived'], evidence={'Sex': True, 'Pclass': True}, verbose=0)
q2 = bn.inference.fit(model, variables=['Survived'], evidence={'Sex': 0}, verbose=0)

print(q1)
print(q1.df)
# bn.print_CPD(model)

# Create test dataset
Xtest = bn.sampling(model, n=100)
# Predict the whole dataset
Pout = bn.predict(model, Xtest, variables=['Survived'])


# %%
DAG = bn.import_DAG('sprinkler', CPD=True)
# DAG = bn.import_DAG('asia')
bn.plot(DAG)
bn.print_CPD(DAG)

df = bn.sampling(DAG, n=1000)
vector = bn.adjmat2vec(DAG['adjmat'])
adjmat = bn.vec2adjmat(vector['source'], vector['target'])

# %%
import bnlearn as bn
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
cpd_S = TabularCPD(variable='S', variable_card=2, values=[[0.6], [0.4]])
print(cpd_S)
cpd_E = TabularCPD(variable='E', variable_card=2,
                   values=[
                       [0.75, 0.72, 0.88, 0.64, 0.70, 0.90],
                       [0.25, 0.28, 0.12, 0.36, 0.30, 0.10]
                   ],
                   evidence=['A', 'S'],
                   evidence_card=[3, 2])
print(cpd_E)


DAG = bn.make_DAG(DAG, CPD=cpd_A, checkmodel=False)
bn.print_CPD(DAG, checkmodel=True)

# %% Create a simple DAG:
# Building a causal DAG

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
DAG = bn.make_DAG(DAG, CPD=[cpt_cloudy, cpt_sprinkler, cpt_rain, cpt_wet_grass], checkmodel=True)
bn.print_CPD(DAG)

q1 = bn.inference.fit(DAG, variables=['Wet_Grass'], evidence={'Rain': 1, 'Sprinkler': 0, 'Cloudy': 1})

# %% Example from sphinx
# Import dataset
# Import the library

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


q1 = bn.inference.fit(DAG, variables=['Wet_Grass'], evidence={'Rain': 1, 'Sprinkler': 0, 'Cloudy': 1})


# %% Example to create a Bayesian Network, learn its parameters from data and perform the inference.


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
q1 = bn.inference.fit(DAGmle, variables=['Wet_Grass', 'Cloudy', 'Sprinkler'], evidence={'Rain': 1})
q1 = bn.inference.fit(DAGmle, variables=['Cloudy', 'Wet_Grass', 'Sprinkler'], evidence={'Rain': 1})
q1 = bn.inference.fit(DAGbay, variables=['Wet_Grass'], evidence={'Rain': 1, 'Sprinkler': 0, 'Cloudy': 1})

print(q1.values)


# %%
    