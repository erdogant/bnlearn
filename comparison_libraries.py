# %% Comparison between methods

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from df2onehot import df2onehot

# %% Load data
import bnlearn as bn
# df = pd.DataFrame(np.random.randint(0, 5, size=(5000, 5)), columns=list('ABCDE'))
# # add 10th dependent variable
# df['F'] = df['A'] * df['B']
# df['E'] = df['A']==0

# %% Load data and drop continous and sensitive features
df = bn.import_example(data='census_income')
drop_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'race', 'sex']
df.drop(labels=drop_cols, axis=1, inplace=True)

# df = bn.import_example(data='student_train')
# drop_col = ['school','sex','age','Mjob', 'Fjob','reason','guardian']
# df = df.drop(columns=drop_col)
# dfhot = df2onehot(df, verbose=4)['onehot']

# %%
#################################
#           Bnlearn             #
#################################

# Load library
import bnlearn as bn
# Structure learning
model = bn.structure_learning.fit(df, methodtype='hillclimbsearch', scoretype='bic')
# Test for significance
model = bn.independence_test(model, df, prune=False)
# Make plot
G = bn.plot(model, interactive=False, params_static={'layout':'draw_circular'})

print(model['model_edges'])
# Customize the DAG or manually provide an entire DAG.
# model = bn.make_DAG(model['model_edges'])

model = bn.parameter_learning.fit(model, df)
CPD = bn.print_CPD(model)

# query = bn.inference.fit(model, variables=['education', 'workclass'], evidence={'salary':'>50K'})

# query = bn.inference.fit(model, variables=['salary'], evidence={'workclass':'Without-pay'})
# print(query)


query = bn.inference.fit(model, variables=['salary', 'marital-status'], evidence={'education':'HS-grad', 'workclass':'State-gov'})
print(query)

query = bn.inference.fit(model, variables=['salary'], evidence={'education':'Doctorate'})
print(query)

query = bn.inference.fit(model, variables=['salary'], evidence={'education':'Doctorate', 'workclass':'State-gov', 'marital-status':'Never-married'})
print(query)

query = bn.inference.fit(model, variables=['salary'], evidence={'education':'Doctorate', 'workclass':'State-gov', 'marital-status':'Never-married'})
print(query)

query = bn.inference.fit(model, variables=['workclass'], evidence={'education':'Doctorate', 'marital-status':'Never-married'})
print(query)


# %%
#################################
#           CausalML            #
#################################
df = bn.import_example(data='census_income')
drop_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'race', 'sex']
df.drop(labels=drop_cols, axis=1, inplace=True)

# Data processing
import pandas as pd
import numpy as np
# Create synthetic data
from causalml.dataset import synthetic_data
# Visualization
import seaborn as sns
# Machine learning model
from xgboost import XGBRegressor
from causalml.inference.meta import LRSRegressor, BaseSRegressor

# Set a seed for reproducibility
np.random.seed(42)
# Create a synthetic dataset
y, X, treatment, ite, _, _ = synthetic_data(mode=1, n=1000, p=5, sigma=1.0)
feature_names = ['X1', 'X2', 'X3', 'X4', 'X5']



# %%
# dfhot = df2onehot(df, verbose=4)['onehot']
# dfhot.drop(labels='salary_<=50K', axis=1, inplace=True)


# %%
#################################
#           Pgmpy               #
#################################
from pgmpy.estimators import HillClimbSearch, BicScore
est = HillClimbSearch(df)
scoring_method = BicScore(df)
best_model = est.estimate(scoring_method=scoring_method)
print(best_model.edges())


# %%
################################
#           CausalNex          #
################################
import bnlearn as bn
import warnings
warnings.filterwarnings("ignore")
from causalnex.structure.notears import from_pandas
import networkx as nx
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Load data
df = bn.import_example(data='census_income')
# Drop continous and sensitive features
drop_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'race', 'sex']
df.drop(labels=drop_cols, axis=1, inplace=True)

# Next, we want to make our data numeric, since this is what the NOTEARS expect
df_num = df.copy()
for col in df_num.columns:
    df_num[col] = le.fit_transform(df_num[col])

df_num.head(5)

# Bayesian Networks in CausalNex support only discrete distributions.
# Any continuous features, or features with a large number of categories, should be discretised prior to fitting the
# Bayesian Network. Models containing variables with many possible values will typically be badly fit, and exhibit
# poor performance.

# Structure learning
structure_model = from_pandas(df_num)

# The reason why we have a fully connected graph here is we haven’t applied thresholding to the weaker edges.
# Thresholding can be applied either by specifying the value for the parameter w_threshold in from_pandas,
# or we can remove the edges by calling the structure model function, remove_edges_below_threshold
structure_model.remove_edges_below_threshold(0.8)
print(structure_model.edges)

# pos = layout_func(sm)
# Use same position from bnlearn
pos=G['pos']

# for layout in layouts
plt.figure(figsize=(15,10));
edge_width = [ d['weight']*0.3 for (u,v,d) in structure_model.edges(data=True)]
nx.draw_networkx_labels(structure_model, pos, font_family="Yu Gothic", font_weight="bold")
nx.draw_networkx(structure_model,
                 # pos,
                 node_size=400,
                 arrowsize=20,
                 alpha=0.6,
                 edge_color='b',
                 width=edge_width)


from causalnex.network import BayesianNetwork

# Step 1: Create a new instance of BayesianNetwork
bn = BayesianNetwork(structure_model)
# Step 2: Reduce the cardinality of categorical features
# Use domain knowledge or other manners to remove redundant features.
# Step 3: Create Labels for Numeric Features
# Create a dictionary that describes the relation between numeric value and label.
# Step 4: Specify all of the states that each node can take
bn = bn.fit_node_states(df)
# Step 5: Fit Conditional Probability Distributions
bn = bn.fit_cpds(df, method="BayesianEstimator", bayes_prior="K2")
test = bn.cpds["education"]

 # %%
#################################
#           DoWhy               #
#################################
from dowhy import CausalModel
import dowhy.datasets
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# data = dowhy.datasets.linear_dataset(beta=10,
#         num_common_causes=5,
#         num_instruments = 2,
#         num_effect_modifiers=1,
#         num_samples=5000,
#         treatment_is_binary=True,
#         stddev_treatment_noise=10,
#         num_discrete_common_causes=1)
# df = data["df"]

df = bn.import_example(data='census_income')
# Drop continous and sensitive features
drop_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'race', 'sex']
df.drop(labels=drop_cols, axis=1, inplace=True)

# Treatment must by binary
df['education'] = df['education']=='Doctorate'
# df['education'] = df['education']=='HS-grad'

# Next, we want to make our data numeric, since this is what the NOTEARS expect
df_num = df.copy()
for col in df_num.columns:
    df_num[col] = le.fit_transform(df_num[col])


# Specify the treatment, outcome, and potential confounding variables
# Do not provide graph
treatment = "education"
outcome = "salary"

model= CausalModel(
        data=df_num,
        treatment=treatment,
        outcome=outcome,
        common_causes=list(df.columns[~np.isin(df.columns, [treatment, outcome])]),
        graph_builder='ges',
        alpha=0.05,
        )

model.view_model()

# Step 2: Identify causal effect and return target estimands
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)
# Step 3: Estimate the target estimand using a statistical method.
estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_stratification")
# Results
print(estimate)
print("Causal Estimate is " + str(estimate.value))
# Step 4: Refute the obtained estimate using multiple robustness checks.
refute_results = model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")
print(refute_results)




# %%
#################################
#           causalinference     #
#################################

from causalinference import CausalModel
from causalinference.utils import random_data
import bnlearn as bn

# Drop continous and sensitive features
df = bn.import_example(data='census_income')
drop_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'race', 'sex']
df.drop(labels=drop_cols, axis=1, inplace=True)
dfhot = df2onehot(df, verbose=4)['onehot']

# Next, we want to make our data numeric, since this is what the NOTEARS expect.
df_num = df.copy()
for col in df_num.columns:
    df_num[col] = le.fit_transform(df_num[col])
    
#Y is the outcome, D is treatment status, and X is the independent variable
# Y, D, X = random_data()
# X = np.c_[X, np.random.random(X.shape[0]), np.random.random(X.shape[0])]
Y = dfhot['salary_>50K'].values
D = dfhot['education_Doctorate'].values
X = df_num.copy()
causal = CausalModel(Y, D, X)

# General statistics
print(causal.summary_stats)

# The main part of causal analysis is acquiring the treatment effect information.
# The simplest one to do is by using the Ordinary Least Square method.
# ATE, ATC, and ATT stand for Average Treatment Effect, Average Treatment Effect for Control and Average Treatment Effect for Treated, respectively. Using this information, we could assess whether the treatment has an effect compared to the control.
causal.est_via_ols()
print(causal.estimates)


# Using the propensity score method, we could also get information regarding the probability of
# treatment conditional on the independent variables.
causal.est_propensity_s()
print(causal.propensity)

# %%
#################################
#           CausalImpact        #
#################################
# https://nbviewer.org/github/jamalsenouci/causalimpact/blob/master/GettingStarted.ipynb
# To illustrate how the package works, we create a simple toy dataset.
# It consists of a response variable y and a predictor x1.
# Note that in practice, we'd strive for including many more predictor variables and let the model choose an appropriate subset.
# The example data has 100 observations. We create an intervention effect by lifting the response variable by 10 units after timepoint 71.

# pip install causalimpact
from causalimpact import CausalImpact
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib
matplotlib.rcParams['figure.figsize'] = (15, 6)

df = bn.import_example(data='census_income')
drop_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'race', 'sex']
df.drop(labels=drop_cols, axis=1, inplace=True)
dfhot = df2onehot(df, verbose=4)['onehot']
y = dfhot['salary_>50K']
dfhot.drop(labels=['salary_>50K', 'salary_<=50K'], axis=1, inplace=True)

np.random.seed(1)
x1 = arma_generate_sample(ar=[0.999], ma=[0.9], nsample=100) + 100
y = 1.2 * x1 + np.random.randn(100)
y[71:100] = y[71:100] + 10
data = pd.DataFrame(np.array([y, x1]).T, columns=["y","x1"])


data.plot();

# To estimate the causal effect, we begin by specifying which period in the data should be used for training the model
# (pre-intervention period) and which period for computing a counterfactual prediction (post-intervention period).
pre_period = [0,69]
post_period = [71,99]
pre_period = [0,1000]
post_period = [1001,2000]

# This says that time points [0-69] will be used for training.
# Time points [71-99] will be used for computing predictions.
# Alternatively, we could specify the periods in terms of dates or time points; see Section 5 for an example.
cim = CausalImpact(dfhot, pre_period, post_period)


cim.run()
cim.plot()


import pandas as pd
from causalimpact import CausalImpact
# Load the data
df = pd.read_csv(r'https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv', sep=',')

n=1000
data = pd.DataFrame()
idx = df['Machine failure'].argsort()[::-1]
data['y'] = df['Machine failure'].iloc[idx[0:n]] + np.random.random(n)
# data['y'] = df['Air temperature [K]'].iloc[idx[0:1000]]
# data['x1'] = df['Process temperature [K]'].iloc[idx[0:n]]
# data['x2'] = df['Air temperature [K]'].iloc[idx[0:1000]]
# data['x3'] = df['Rotational speed [rpm]'].iloc[idx[0:1000]]
# data['x4'] = df['Torque [Nm]'].iloc[idx[0:1000]]
data['x5'] = df['Tool wear [min]'].iloc[idx[0:1000]]
data.reset_index(drop=True, inplace=True)

# Define pre- and post-intervention periods
pre_period = [0, 338]
post_period = [339, n-1]

# Run the analysis
impact = CausalImpact(data, pre_period, post_period)
impact.run()
impact.plot()
# Visualization
print(impact.summary())
# Causal impact report
print(impact.summary(output='report'))
impact.inferences


import pandas as pd
import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample
from causalimpact import CausalImpact
np.random.seed(1)
x1 = arma_generate_sample(ar=[0.999], ma=[0.9], nsample=100) + 100
y = 1.2 * x1 + np.random.randn(100)
y[71:100] = y[71:100] + 10
data = pd.DataFrame(np.array([y, x1]).T, columns=["y","x1"])
pre_period = [0,69]
post_period = [71,99]

# Run the analysis
impact = CausalImpact(data, pre_period, post_period)
impact.run()
impact.plot()
# Visualization
print(impact.summary())


