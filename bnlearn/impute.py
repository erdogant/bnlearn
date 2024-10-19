import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# %% Impute
def knn_imputer(df, n_neighbors=2, weights="uniform", metric='nan_euclidean', string_columns=None, verbose=3):
    """Impute missing values.

    Impute missing values in a DataFrame using KNN imputation for numeric columns. String columns are not included in the encoding.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing both numeric and string columns.
    n_neighbors : int, optional
        Number of neighboring samples to use for imputation (default is 2).
    weights : str, optional
        Weight function used in prediction. Possible values:
        - 'uniform' : uniform weights. All points in each neighborhood are weighted equally.
        - 'distance' : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
        - callable : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.
    metric : default='nan_euclidean'
        Distance metric for searching neighbors.
    string_columns : list or str, optional
        A list of column names or a single column name (string) that contains string/categorical data.
        These columns will be removed using LabelEncoder before imputation (default is None).
        ['car name', 'origin']
    verbose : int, optional
        Level of verbosity to control printed messages during execution. Higher values give more detailed logs (default is 3).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with imputed values. Numeric columns will have missing values filled using KNN imputation,
        and original string columns (if any) will be retained.

    Notes
    -----
    - String columns are encoded to numerical values using LabelEncoder for imputation and decoded back after the imputation process.
    - The function automatically identifies numeric columns and handles conversion to appropriate data types if necessary.
    - The `verbose` parameter allows controlling how much detail is printed out for tracking the progress of imputation.

    Examples
    --------
    >>> from impute import knn_imputer, mice_imputer
    >>> df = pd.DataFrame({
    ...    'age': [25, np.nan, 27],
    ...    'income': [50000, 60000, np.nan],
    ...    'city': ['New York', np.nan, 'Los Angeles']
    ... })
    >>> knn_imputer(df, n_neighbors=3, weights='distance', string_columns='city')
         age   income          city
    0  25.0  50000.0      New York
    1  26.0  60000.0      Los Angeles
    2  27.0  55000.0  Los Angeles

    """
    # Convert string columns to categorical and then encode them
    if string_columns is not None:
        if isinstance(string_columns, str):
            string_columns = [string_columns]
        # Encode string columns if specified
        for col in string_columns:
            df[col] = df[col].astype('category')

    # Convert the remaining numeric columns to float (if not already)
    for col in df.columns:
        try:
            if (string_columns is None) or (not np.isin(col, string_columns)):
                df[col] = df[col].astype(float)
                if verbose>=4: print(f'[bnlearn] >float: {col}')
        except:
            if verbose>=4: print(f'[bnlearn] >Category forced: {col}')
            if string_columns is None: string_columns = []
            string_columns = string_columns + [col]
            df[col] = df[col].astype(str)
            df[col].fillna('None')

    # Initialize the KNN imputer
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, metric=metric)

    # Impute only the numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputed_values = imputer.fit_transform(df[numeric_cols])

    # Create a new DataFrame for imputed numeric values
    df_imputed = pd.DataFrame(imputed_values, columns=numeric_cols)

    # Add the original string columns back to the imputed DataFrame if any
    if string_columns is not None:
        for col in string_columns:
            df_imputed[col] = df[col]

    # Impute categorical columns with the most frequent value
    df_imputed = impute_catagorical_knn(df_imputed, string_columns, numeric_cols, n_neighbors)

    # Return
    return df_imputed


def mice_imputer(df, max_iter=10, estimator=None, string_columns=None, verbose=3):
    """Impute missing values using Multiple Imputation by Chained Equations (MICE).

    Impute missing values in a DataFrame using MICE imputation for numeric columns. String columns are not included in the encoding.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing both numeric and string columns.
    max_iter : int, optional
        Maximum number of imputation rounds to perform (default is 10).
    estimator : estimator object, optional
        The estimator to use at each step of the round-robin imputation. If None, BayesianRidge() is used.
    string_columns : list or str, optional
        A list of column names or a single column name (string) that contains string/categorical data.
        These columns will be removed using LabelEncoder before imputation (default is None).
        ['car name', 'origin']
    verbose : int, optional
        Level of verbosity to control printed messages during execution. Higher values give more detailed logs (default is 3).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with imputed values. Numeric columns will have missing values filled using MICE imputation,
        and original string columns (if any) will be retained.

    Notes
    -----
    - String columns are encoded to numerical values using LabelEncoder for imputation and decoded back after the imputation process.
    - The function automatically identifies numeric columns and handles conversion to appropriate data types if necessary.
    - The `verbose` parameter allows controlling how much detail is printed out for tracking the progress of imputation.
    - MICE is an iterative imputation method that models each feature with missing values as a function of other features.

    Examples
    --------
    >>> from impute import knn_imputer, mice_imputer
    >>> df = pd.DataFrame({
    ...    'age': [25, np.nan, 27],
    ...    'income': [50000, 60000, np.nan],
    ...    'city': ['New York', np.nan, 'Los Angeles']
    ... })
    >>> mice_imputer(df, max_iter=5, string_columns='city')
         age   income          city
    0  25.0  50000.0      New York
    1  26.2  60000.0      Los Angeles
    2  27.0  55123.7  Los Angeles

    """
    # Convert string columns to categorical and then encode them
    if string_columns is not None:
        if isinstance(string_columns, str):
            string_columns = [string_columns]
        # Encode string columns if specified
        for col in string_columns:
            df[col] = df[col].astype('category')

    # Convert the remaining numeric columns to float (if not already)
    for col in df.columns:
        try:
            if (string_columns is None) or (not np.isin(col, string_columns)):
                df[col] = df[col].astype(float)
                if verbose>=4: print(f'[bnlearn] >float: {col}')
        except:
            if verbose>=4: print(f'[bnlearn] >Category forced: {col}')
            if string_columns is None: string_columns = []
            string_columns = string_columns + [col]
            df[col] = df[col].astype(str)
            df[col].fillna('None')

    # Initialize the MICE imputer
    imputer = IterativeImputer(max_iter=max_iter, estimator=estimator, random_state=0)

    # Impute only the numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputed_values = imputer.fit_transform(df[numeric_cols])

    # Create a new DataFrame for imputed numeric values
    df_imputed = pd.DataFrame(imputed_values, columns=numeric_cols)

    # Impute categorical columns with the most frequent value
    df_imputed = impute_catagorical_knn(df_imputed, string_columns, numeric_cols, 3)

    # Return
    return df_imputed


def impute_catagorical_knn(df, string_columns, numeric_cols, n_neighbors):
    """
    Impute missing values in categorical columns using K-Nearest Neighbors (KNN) based on numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing both categorical and numeric columns.
    string_columns : list of str
        List of column names in `df` that are categorical and need imputation for missing values.
    numeric_cols : list of str
        List of column names in `df` that are numeric and will be used for distance calculation in KNN.
    n_neighbors : int
        The number of nearest neighbors to consider for imputation.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame `df` with missing values in the categorical columns imputed.

    Notes
    -----
    - The function uses K-Nearest Neighbors to find the nearest data points based on the numeric columns
      and imputes missing values in the categorical columns with the most frequent value (mode) among the neighbors.
    - The function ensures that missing categorical values are replaced by the mode of their nearest neighbors
      based on numeric data.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'age': [25, 25, 27, 29],
    ...     'income': [50000, 60000, 65000, 7000],
    ...     'city': ['New York', np.nan, 'Los Angeles', 'San Francisco']
    ... })
    >>> impute_catagorical_knn(df, string_columns=['city'], numeric_cols=['age', 'income'], n_neighbors=3)
          age  income           city
    0   25   50000        New York
    1   25   60000        New York
    2   27   65000     Los Angeles
    3   29    7000  San Francisco

    """
    # Impute categorical columns with the most frequent value
    if string_columns is not None:
        model = NearestNeighbors(n_neighbors=np.minimum(n_neighbors + 1, df.shape[0])).fit(df[numeric_cols])
        for col in string_columns:
            # Get all missing indexes
            missing_index = np.where(df[col].isna())[0]
            # For each missing catagory, find its nearest neighbors and impute the mode
            for row in missing_index:
                distances, indices = model.kneighbors([df[numeric_cols].loc[row]])
                most_frequent = df[col][indices[0]].mode()[0]
                # Impute
                df[col][missing_index] = most_frequent

    # Return
    return df
