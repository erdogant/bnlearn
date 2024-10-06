# %%
import pandas as pd
import bnlearn as bn

df = pd.read_csv(
    "bnlearn/tests/discretize/data/auto_mpg.csv",
    dtype={
        "mpg": "float64",
        "cylinders": "int64",
        "displacement": "float64",
        "horsepower": "float64",
        "weight": "float64",
        "acceleration": "float64",
        "model_year": "int64",
        "origin": "int64",
    },
)

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

continuous_columns = ["mpg", "displacement", "horsepower", "weight", "acceleration"]

# %%
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original', 
                 delim_whitespace=True, header=None,
                 names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name'])

df.dropna(inplace=True)
df.drop(['model year', 'origin', 'car name'], axis=1, inplace=True)
print(df.shape)
df.head()


edges = [
    ("cylinders", "displacement"),
    ("displacement", "weight"),
    ("displacement", "horsepower"),
    ("weight", "mpg"),
    ("horsepower", "acceleration"),
]

continuous_columns = ["mpg", "displacement", "horsepower", "weight", "acceleration"]


# %%


df_disc = bn.discretize(
    df,
    edges,
    continuous_columns,
    max_iterations=1,
)

DAG = bn.make_DAG(edges)
model_mle = bn.parameter_learning.fit(DAG, df_disc)
# model_mle = bn.parameter_learning.fit(DAG, data_disc, methodtype="maximumlikelihood")

bn.plot(model_mle)
print(model_mle["model"].get_cpds("mpg"))

# %%

print("Weight categories: ", df_disc["weight"].dtype.categories)
evidence = {"weight": bn.discretize_value(df_disc["weight"], 3000.0)}
print(evidence)
print(bn.inference.fit(model_mle, variables=["mpg"], evidence=evidence, verbose=0))
