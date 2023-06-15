import bnlearn as bn
import datazets as dz
import numpy as np

# Get the data science salary data set
df = dz.get('ds_salaries')

# %% Pre processing
df['country'] = 'USA'
countries_europe = ['SM', 'DE', 'GB', 'ES', 'FR', 'RU', 'IT', 'NL', 'CH', 'CF', 'FI', 'UA', 'IE', 'GR', 'MK', 'RO', 'AL', 'LT', 'BA', 'LV', 'EE', 'AM', 'HR', 'SI', 'PT', 'HU', 'AT', 'SK', 'CZ', 'DK', 'BE', 'MD', 'MT']
df['country'][np.isin(df['company_location'], countries_europe)]='europe'

# Rename catagorical variables for better understanding
df['experience_level'] = df['experience_level'].replace({'EN': 'Entry-level', 'MI': 'Junior Mid-level', 'SE': 'Intermediate Senior-level', 'EX': 'Expert Executive-level / Director'}, regex=True)
df['employment_type'] = df['employment_type'].replace({'PT': 'Part-time', 'FT': 'Full-time', 'CT': 'Contract', 'FL': 'Freelance'}, regex=True)
df['company_size'] = df['company_size'].replace({'S': 'Small (less than 50)', 'M': 'Medium (50 to 250)', 'L': 'Large (>250)'}, regex=True)
df['remote_ratio'] = df['remote_ratio'].replace({0: 'No remote', 50: 'Partially remote', 100: '>80% remote'}, regex=True)

# Group similar job titles
titles = [['data scientist', 'data science', 'research', 'applied', 'specialist', 'ai', 'machine learning'],
          ['engineer', 'etl'],
          ['analyst', 'bi', 'business', 'product', 'modeler', 'analytics'],
          ['manager', 'head', 'director'],
          ['architect', 'cloud', 'aws'],
          ['lead/principal', 'lead', 'principal'],
          ]

job_title = df['job_title'].str.lower().copy()
df['job_title'] = 'Other'
for t in titles:
    for name in t:
        df['job_title'][list(map(lambda x: name in x, job_title))]=t[0]

# engineer          1654
# data scientist    1238
# analyst            902
# manager            158
# architect          118
# lead/principal      55
# Other                9
# Name: job_title, dtype: int64

# %% # Catagorize salaries

discretize_method='manual'

if discretize_method=='manual':
    salary_in_usd = df['salary_in_usd']
    # Remove redundant variables
    df.drop(labels=['salary_currency', 'salary', 'salary_in_usd'], inplace=True, axis=1)
    # Set salary
    df['salary_in_usd'] = None
    df['salary_in_usd'].loc[salary_in_usd<60000]='<60K'
    df['salary_in_usd'].loc[np.logical_and(salary_in_usd>=60000, salary_in_usd<100000)]='60-100K'
    df['salary_in_usd'].loc[np.logical_and(salary_in_usd>=100000, salary_in_usd<160000)]='100-160K'
    df['salary_in_usd'].loc[np.logical_and(salary_in_usd>=160000, salary_in_usd<250000)]='160-250K'
    df['salary_in_usd'].loc[salary_in_usd>=250000]='>250K'
else:
    # Discritize salary
    tmpdf = df[['experience_level', 'salary_in_usd', 'country']]
    # Create edges
    edges = [('experience_level', 'salary_in_usd'), ('country', 'salary_in_usd')]
    # Create DAG based on edges
    DAG = bn.make_DAG(edges)
    bn.plot(DAG)
    # Discretize the continous columns
    df_disc = bn.discretize(tmpdf, edges, ["salary_in_usd"], max_iterations=1)
    df['salary_in_usd'] = df_disc['salary_in_usd']
    df['salary_in_usd'].value_counts()
    # Remove redundant variables
    df.drop(labels=['salary_currency', 'salary'], inplace=True, axis=1)

# %% Learn the causal DAG from the data
# Structure learning
model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
# independence test
model = bn.independence_test(model, df, prune=False)
# Check best scoring type
best_scoretype = min(model['structure_scores'], key=lambda k: model['structure_scores'][k])
# Parameter learning
model = bn.parameter_learning.fit(model, df, methodtype="bayes")
# Plot
bn.plot(model, params_static={'layout': 'planar_layout'}, title='method=hc and score=bic')
bn.plot(model, title='method=hc and score=bic')
bn.plot(model, interactive=True, title='method=tan and score=bic')

# # Structure learning
# model_tan = bn.structure_learning.fit(df, methodtype='tan', class_node="salary_in_usd", scoretype='bic')
# # independence test
# model_tan = bn.independence_test(model_tan, df, prune=False)
# # Plot
# bn.plot(model_tan, params_static={'layout': 'planar_layout'}, title='method=tan and score=bic')
# bn.plot(model_tan, interactive=True, title='method=tan and score=bic')
# # Check best scoring type
# best_scoretype = min(model_tan['structure_scores'], key=lambda k: model_tan['structure_scores'][k])
# # Parameter learning
# model_tan = bn.parameter_learning.fit(model_tan, df, methodtype="bayes")

# bn.compare_networks(model, model_tan)

# %% Make inferences

query = bn.inference.fit(model, variables=['job_title'],
                         evidence={'company_size': 'Large (>250)'})
query = bn.inference.fit(model, variables=['job_title', 'salary_in_usd'], evidence={'company_size': 'Large (>250)'})
query = bn.inference.fit(model, variables=['job_title', 'salary_in_usd'], evidence={'company_size': 'Large (>250)'}, groupby=['job_title'])

query = bn.inference.fit(model, variables=['experience_level'], evidence={'company_size': 'Large (>250)'})
query = bn.inference.fit(model, variables=['experience_level'], evidence={'salary_in_usd': '100-160K'})
query = bn.inference.fit(model, variables=['experience_level'], evidence={'salary_in_usd': '80-100K'})
query = bn.inference.fit(model, variables=['experience_level'], evidence={'salary_in_usd': '<80K'})
# query = bn.inference.fit(model, variables=['experience_level', 'remote_ratio'], evidence={'salary_in_usd': '>250K'})

# query = bn.inference.fit(model_tan, variables=['salary_in_usd'], evidence={'company_size': 'Large (>250)'})
query = bn.inference.fit(model, variables=['salary_in_usd'], evidence={'company_size': 'Large (>250)'})

# query = bn.inference.fit(model, variables=['salary_in_usd'], evidence={'experience_level':'Entry-level', 'employment_type':'Part-time', 'company_size':'Small (less than 50)'})
# query = bn.inference.fit(model, variables=['salary_in_usd', 'experience_level'], evidence={'company_size': 'Large (>250)'})

query = bn.inference.fit(model, variables=['salary_in_usd'], evidence={'employment_type': 'Full-time',
                                                                                    'company_size': 'Large (>250)',
                                                                                    # 'remote_ratio': 'Partially remote',
                                                                                    'experience_level': 'Expert Executive-level / Director'})

query = bn.inference.fit(model,
                         variables=['job_title'],
                         evidence={'employment_type': 'Full-time',
                                   'company_size': 'Large (>250)',
                                   'remote_ratio': 'Partially remote',
                                   'country': 'europe'})

query = bn.inference.fit(model,
                         variables=['salary_in_usd'],
                         evidence={'employment_type': 'Full-time',
                                   'remote_ratio': 'Partially remote',
                                   'job_title': 'data scientist',
                                   'employee_residence': 'DE',
                                   # 'experience_level': 'Entry-level',
                                    'experience_level': 'Intermediate Senior-level',
                                   },
                         # groupby='experience_level',
                         )


# %%
