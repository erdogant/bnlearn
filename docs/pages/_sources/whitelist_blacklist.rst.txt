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
    DAG_nodes = bnlearn.structure_learning.fit(dfnum, methodtype='hc', bw_list_method='nodes', white_list=['Survived','Pclass','Sex','Embarked','Parch'])

    # Structure learning by enforcing variables 'Survived','Pclass','Sex','Embarked','Parch'.
    DAG_edges = bnlearn.structure_learning.fit(dfnum, methodtype='hc', bw_list_method='edges', white_list=['Survived','Pclass','Sex','Embarked','Parch'])

    # Plot
    Gf = bnlearn.plot(DAG_nodes)
    Ge = bnlearn.plot(DAG_edges)



**Black list example**

.. code-block:: python

    import bnlearn
    # Load example mixed dataset
    df_raw = bnlearn.import_example(data='titanic')

    # Convert to onehot
    dfhot, dfnum = bnlearn.df2onehot(df_raw)

    # Structure learning after removing 'Survived','Pclass','Sex','Embarked','Parch'.
    DAG_nodes = bnlearn.structure_learning.fit(dfnum, methodtype='hc', bw_list_method='nodes', black_list=['Survived','Pclass','Sex','Embarked','Parch'])

    # Structure learning by enforcing variables 'Survived','Pclass','Sex','Embarked','Parch'.
    DAG_edges = bnlearn.structure_learning.fit(dfnum, methodtype='hc', bw_list_method='edges', black_list=['Survived','Pclass','Sex','Embarked','Parch'])

    # Plot
    Gf = bnlearn.plot(DAG_nodes)
    Ge = bnlearn.plot(DAG_edges)





.. include:: add_bottom.add