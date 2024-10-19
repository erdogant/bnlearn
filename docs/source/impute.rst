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

	imputed_df1 = bn.knn_imputer(df, n_neighbors=3, weights="distance", string_columns=['car name'])
	imputed_df2 = bn.knn_imputer(df, n_neighbors=3, weights="distance")

	print(imputed_df1.head())
	print(imputed_df2.head())

	imputed_df3 = bn.mice_imputer(df, max_iter=10, string_columns=['car name'])
	print(imputed_df3.head())



.. include:: add_bottom.add