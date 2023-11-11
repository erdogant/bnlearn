# Import library
import datazets as dz
import numpy as np


# Get the data science salary data set
df = dz.get('ds_salaries')
# The features are as following
df.columns

titles = [['data scientist', 'data science', 'research', 'applied', 'specialist', 'ai', 'machine learning'],
          ['engineer', 'etl'],
          ['analyst', 'bi', 'business', 'product', 'modeler', 'analytics'],
          ['manager', 'head', 'director'],
          ['architect', 'cloud', 'aws'],
          ['lead/principal', 'lead', 'principal'],
          ]

# Aggregate job titles
job_title = df['job_title'].str.lower().copy()
df['job_title'] = 'Other'
# Store the new names
for t in titles:
    for name in t:
        df['job_title'][list(map(lambda x: name in x, job_title))]=t[0]
print(df['job_title'].value_counts())


# Rename catagorical variables for better understanding
df['experience_level'] = df['experience_level'].replace({'EN': 'Entry-level', 'MI': 'Junior Mid-level', 'SE': 'Intermediate Senior-level', 'EX': 'Expert Executive-level / Director'}, regex=True)
df['employment_type'] = df['employment_type'].replace({'PT': 'Part-time', 'FT': 'Full-time', 'CT': 'Contract', 'FL': 'Freelance'}, regex=True)
df['company_size'] = df['company_size'].replace({'S': 'Small (less than 50)', 'M': 'Medium (50 to 250)', 'L': 'Large (>250)'}, regex=True)
df['remote_ratio'] = df['remote_ratio'].replace({0: 'No remote', 50: 'Partially remote', 100: '>80% remote'}, regex=True)

# Add new feature
df['country'] = 'USA'
countries_europe = ['SM', 'DE', 'GB', 'ES', 'FR', 'RU', 'IT', 'NL', 'CH', 'CF', 'FI', 'UA', 'IE', 'GR', 'MK', 'RO', 'AL', 'LT', 'BA', 'LV', 'EE', 'AM', 'HR', 'SI', 'PT', 'HU', 'AT', 'SK', 'CZ', 'DK', 'BE', 'MD', 'MT']
df['country'][np.isin(df['company_location'], countries_europe)]='europe'
# Remove redundant variables
salary_in_usd = df['salary_in_usd']
df.drop(labels=['salary_currency', 'salary'], inplace=True, axis=1)

# %%
import bnlearn as bn
# Discretize the salary feature.
discretize_method='manual'

# Discretize Manually
if discretize_method=='manual':
    # Set salary
    df['salary_in_usd'] = None
    df['salary_in_usd'].loc[salary_in_usd<80000]='<80K'
    df['salary_in_usd'].loc[np.logical_and(salary_in_usd>=80000, salary_in_usd<100000)]='80-100K'
    df['salary_in_usd'].loc[np.logical_and(salary_in_usd>=100000, salary_in_usd<160000)]='100-160K'
    df['salary_in_usd'].loc[np.logical_and(salary_in_usd>=160000, salary_in_usd<250000)]='160-250K'
    df['salary_in_usd'].loc[salary_in_usd>=250000]='>250K'
else:
    # Discretize automatically but with prior knowledge.
    tmpdf = df[['experience_level', 'salary_in_usd', 'country']]
    # Create edges
    edges = [('experience_level', 'salary_in_usd'), ('country', 'salary_in_usd')]
    # Create DAG based on edges
    DAG = bn.make_DAG(edges)
    bn.plot(DAG)
    # Discretize the continous columns
    df_disc = bn.discretize(tmpdf, edges, ["salary_in_usd"], max_iterations=1)
    # Store
    df['salary_in_usd'] = df_disc['salary_in_usd']
    # Print
    print(df['salary_in_usd'].value_counts())
    
# %%
model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')

# %%

# independence test
model = bn.independence_test(model, df, prune=False)
# Parameter learning to learn the CPTs. This step is required to make inferences.
model = bn.parameter_learning.fit(model, df, methodtype="bayes")
# Plot
bn.plot(model, title='Salary data set')
bn.plot(model, interactive=True, title='method=tan and score=bic')

# %%

query = bn.inference.fit(model, variables=['job_title'],
                         evidence={'company_size': 'Large (>250)'})


query = bn.inference.fit(model,
                         variables=['salary_in_usd'],
                         evidence={'employment_type': 'Full-time',
                                   'remote_ratio': 'Partially remote',
                                   'job_title': 'data scientist',
                                   'employee_residence': 'DE',
                                   'experience_level': 'Entry-level'})
