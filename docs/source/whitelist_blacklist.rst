Black and white lists
========================

Input variablescan be black or white listed in the model.
When variables are black listed, they are excluded from the search and the resulting model will not contain any of those edges. If variables are white listed, the search is limited to only those edges. The resulting model will then only contain edges that are in white_list.

**White list example**

.. code-block:: python

    import bnlearn
    # Load example mixed dataset
    df_raw = bnlearn.import_example(data='titanic')

    # Convert to onehot
    dfhot, dfnum = bnlearn.df2onehot(df_raw)

    # Structure learning by including only 'Survived','Pclass','Sex','Embarked','Parch'.
    DAG_filtered = bnlearn.structure_learning.fit(dfnum, white_list=['Survived','Pclass','Sex','Embarked','Parch'], bw_list_method='filter')

    # Structure learning by enforcing variables 'Survived','Pclass','Sex','Embarked','Parch'.
    DAG_enforced = bnlearn.structure_learning.fit(dfnum, white_list=['Survived','Pclass','Sex','Embarked','Parch'], bw_list_method='enforce')

    # Plot
    Gf = bnlearn.plot(DAG_filtered)
    Ge = bnlearn.plot(DAG_enforced)



**Black list example**

.. code-block:: python

    import bnlearn
    # Load example mixed dataset
    df_raw = bnlearn.import_example(data='titanic')

    # Convert to onehot
    dfhot, dfnum = bnlearn.df2onehot(df_raw)

    # Structure learning after removing 'Survived','Pclass','Sex','Embarked','Parch'.
    DAG_filtered = bnlearn.structure_learning.fit(dfnum, black_list=['Survived','Pclass','Sex','Embarked','Parch'], bw_list_method='filter')

    # Structure learning by enforcing variables 'Survived','Pclass','Sex','Embarked','Parch'.
    DAG_enforced = bnlearn.structure_learning.fit(dfnum, black_list=['Survived','Pclass','Sex','Embarked','Parch'], bw_list_method='enforce')

    # Plot
    Gf = bnlearn.plot(DAG_filtered)
    Ge = bnlearn.plot(DAG_enforced)
