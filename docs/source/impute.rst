Impute
========================

Impute missing values in a DataFrame using KNN imputation for numeric columns.
String columns are not included in the encoding.

Lets load a dataframe with missing values and perform the imputation.

.. code-block:: python

	# Initialize libraries
	import bnlearn as bn
	import pandas as pd

	# Load the dataset
	df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original', delim_whitespace=True, header=None, names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name'])

	imputed_df1 = bn.impute(df, n_neighbors=3, weights="distance", string_columns=['car name'])
	imputed_df2 = bn.impute(df, n_neighbors=3, weights="distance")



.. include:: add_bottom.add