Impute
=========================================

The ``bnlearn`` library provides two different imputation methods for handling missing data. In both methods, categorical columns are excluded first, and missing numerical values are imputed using either the KNN or MICE approach. Once the numerical values are imputed, the resulting DataFrame is used to build a Nearest Neighbors (NN) model. Finally, missing categorical values are imputed using the 1-NN model based on the imputed numerical data.

KNN Imputer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Impute missing values in a DataFrame using K-Nearest Neighbors (KNN) imputation. This method finds the k-nearest neighbors for each missing value and imputes the value based on the neighbors' values.

Let's load a dataframe with missing values and perform the imputation:

.. code-block:: python

    # Initialize libraries
    import bnlearn as bn
    import pandas as pd

    # Load the dataset
    df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original', 
                     delim_whitespace=True, 
                     header=None, 
                     names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 
                            'acceleration', 'model year', 'origin', 'car name'])

    # Create some identical rows as test-case
    df.loc[1] = df.loc[0]
    df.loc[11] = df.loc[10]
    df.loc[50] = df.loc[20]

    # Set few rows to None
    index_nan = [0, 10, 20]
    carnames = df['car name'].loc[index_nan]
    df['car name'].loc[index_nan] = None
    df.isna().sum()

    # KNN imputer
    dfnew = bn.knn_imputer(df, n_neighbors=3, weights='distance', string_columns=['car name'])
    # Results
    np.all(dfnew['car name'].loc[index_nan].values == carnames.values)

MICE Imputer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Impute missing values in a DataFrame using Multiple Imputation by Chained Equations (MICE). This method performs multiple imputations by iteratively modeling each variable with missing values as a function of other variables.

Key features include:

* Supports MICE imputation for numeric columns
* String/categorical columns are encoded before imputation and restored post-imputation
* Includes options to specify the imputation estimator, number of iterations (max_iter), and verbosity level for logging
* Numeric columns are auto-identified and converted for imputation where necessary
* This enhancement improves missing data handling and supports mixed-type datasets

Let's load a dataframe with missing values and perform the imputation:

.. code-block:: python

    # Initialize libraries
    import bnlearn as bn
    import pandas as pd

    # Load the dataset
    df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original', 
                     delim_whitespace=True, 
                     header=None, 
                     names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 
                            'acceleration', 'model year', 'origin', 'car name'])

    # Create some identical rows as test-case
    df.loc[1] = df.loc[0]
    df.loc[11] = df.loc[10]
    df.loc[50] = df.loc[20]

    # Set few rows to None
    index_nan = [0, 10, 20]
    carnames = df['car name'].loc[index_nan]
    df['car name'].loc[index_nan] = None
    df.isna().sum()

    # MICE imputer
    dfnew = bn.mice_imputer(df, max_iter=5, string_columns='car name')
    # Results
    np.all(dfnew['car name'].loc[index_nan].values == carnames.values)

.. include:: add_bottom.add