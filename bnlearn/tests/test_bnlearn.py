# pip install pytest
# pytest -v

import pytest

import bnlearn as bn
from pgmpy.factors.discrete import TabularCPD
import numpy as np
import pandas as pd


# def test_load_examples():
#     shapes = [(10000, 37), (10000, 223), (10000, 8), (10000, 11), (10000, 32), (352, 3)]
#     for i, data in enumerate(['alarm', 'andes', 'asia', 'sachs', 'water', 'stormofswords']):
#         df = bn.import_example(data=data)
#         assert not df.empty


def test_QUERY():
    # Load example DataFrame
    df = bn.import_example('titanic')
    dfhot, dfnum = bn.df2onehot(df)
    # Train model
    model_as = bn.structure_learning.fit(dfnum, methodtype='hc', scoretype='bic')
    model_as_p = bn.parameter_learning.fit(model_as, dfnum, methodtype='bayes')
    # Do the inference
    variables_list = [['Sex', 'Parch', 'SibSp'], ['Sex', 'Parch'], ['Sex']]
    evidences_list = [{'Survived': 0, 'Pclass': 1, 'Embarked': 1}, {'Survived': 0, 'Pclass': 1}, {'Survived': 0}]
    sizes = [(48, 4), (48, 4), (48, 4), (8, 3), (8, 3), (8, 3), (2, 2), (2, 2), (2, 2)]
    i = 0
    for variables in variables_list:
        for evidences in evidences_list:
            query = bn.inference.fit(model_as_p, variables=variables, evidence=evidences, to_df=True, verbose=0)
            assert query.df.shape == sizes[i]
            assert list(query.df.columns) == variables + ['p']
            i = i + 1

    query = bn.inference.fit(model_as_p, variables=['Sex', 'Parch', 'SibSp'], evidence={'Survived': 0, 'Pclass': 1}, to_df=True, verbose=0)
    q = bn.query2df(query, variables=['SibSp', 'Sex'])
    assert q.shape == (48, 3)
    assert list(q.columns) == ['SibSp', 'Sex', 'p']


def test_import_DAG():
    import bnlearn as bn
    DAG = bn.import_DAG('sprinkler')
    # TEST 1: check output is unchanged
    assert DAG.keys() == {'model', 'adjmat'}
    # TEST 2: Check model output is unchanged
    assert DAG['adjmat'].sum().sum() == 4
    # TEST 3:
    assert 'pgmpy.models.BayesianNetwork' in str(type(DAG['model']))
    # TEST 4:
    # DAG = bn.import_DAG('alarm', verbose=0)
    # assert DAG.keys() == {'model', 'adjmat'}
    # DAG = bn.import_DAG('andes', verbose=0)
    # assert DAG.keys() == {'model', 'adjmat'}
    # DAG = bn.import_DAG('asia', verbose=0)
    # assert DAG.keys() == {'model', 'adjmat'}


def test_make_DAG():
    edges = [('Cloudy', 'Sprinkler')]
    methodtypes = ['bayes', 'naivebayes']
    for methodtype in methodtypes:
        DAG = bn.make_DAG(edges, methodtype=methodtype)
        # TEST 1
        if methodtype == 'bayes':
            assert 'pgmpy.models.BayesianNetwork' in str(type(DAG['model']))
        else:
            assert 'pgmpy.models.NaiveBayes.NaiveBayes' in str(type(DAG['model']))
    # TEST 2
    cpt_cloudy = TabularCPD(variable='Cloudy', variable_card=2, values=[[0.3], [0.7]])
    cpt_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2, values=[[0.4, 0.9], [0.6, 0.1]],
                               evidence=['Cloudy'], evidence_card=[2])
    assert bn.make_DAG(DAG, CPD=[cpt_cloudy, cpt_sprinkler], checkmodel=True)
    # TEST 3
    assert set(DAG.keys()) == {'adjmat', 'model', 'methodtype', 'model_edges'}


@pytest.fixture
def sprinkler_dag():
    return bn.import_DAG('Sprinkler')


def test_sampling_bayes(sprinkler_dag):
    n = np.random.randint(10, 1000)
    model = bn.import_DAG('sprinkler')
    df = bn.sampling(model, n=n, methodtype='bayes')
    assert df.shape == (n, 4)


@pytest.mark.skip('Will be fixed by https://github.com/pgmpy/pgmpy/issues/1582')
def test_sampling_gibbs(sprinkler_dag):
    n = np.random.randint(10, 1000)
    df = bn.sampling(sprinkler_dag, n=n, methodtype='gibbs')
    assert df.shape == (n, 4)


# def test_to_undirected():
#     # TEST 1:
#     randdata = ['sprinkler', 'alarm', 'andes', 'asia', 'sachs']
#     n = np.random.randint(0, len(randdata))
#     DAG = bn.import_DAG(randdata[n], CPD=False, verbose=0)
#     assert (DAG['adjmat'].sum().sum() * 2) == bn.to_undirected(DAG['adjmat']).sum().sum()


def test_compare_networks():
    DAG = bn.import_DAG('sprinkler', verbose=0)
    G = bn.compare_networks(DAG, DAG, showfig=False)
    assert np.all(G[0] == [[12, 0], [0, 4]])


def test_adjmat2vec():
    DAG = bn.import_DAG('sprinkler', verbose=0)
    out = bn.adjmat2vec(DAG['adjmat'])
    assert np.all(out['source'] == ['Cloudy', 'Cloudy', 'Sprinkler', 'Rain'])


def test_vec2adjmat():
    DAG = bn.import_DAG('sprinkler', verbose=0)
    out = bn.adjmat2vec(DAG['adjmat'])
    # TEST: conversion
    assert bn.vec2adjmat(out['source'], out['target']).shape == DAG['adjmat'].shape


def test_parameter_learning():
    df = bn.import_example()
    model = bn.import_DAG('sprinkler', CPD=False)
    model_update = bn.parameter_learning.fit(model, df)
    assert [*model_update.keys()] == ['model', 'adjmat', 'config', 'model_edges', 'structure_scores', 'independence_test']


def test_inference():
    DAG = bn.import_DAG('sprinkler')
    q1 = bn.inference.fit(DAG, variables=['Wet_Grass'], evidence={'Rain': 1, 'Sprinkler': 0, 'Cloudy': 1}, to_df=False, verbose=0)
    assert 'pgmpy.factors.discrete.DiscreteFactor.DiscreteFactor' in str(type(q1))
    assert q1.df is None
    q1 = bn.inference.fit(DAG, variables=['Wet_Grass'], evidence={'Rain': 1, 'Sprinkler': 0, 'Cloudy': 1}, to_df=True, verbose=0)
    assert q1.df is not None


def test_query2df():
    DAG = bn.import_DAG('sprinkler')
    query = bn.inference.fit(DAG, variables=['Wet_Grass'], evidence={'Rain': 1, 'Sprinkler': 0, 'Cloudy': 1}, to_df=False, verbose=0)
    df = bn.query2df(query)
    assert df.shape == (2, 2)
    assert np.all(df.columns == ['Wet_Grass', 'p'])
    query = bn.inference.fit(DAG, variables=['Wet_Grass', 'Sprinkler'], evidence={'Rain': 1, 'Cloudy': 1}, to_df=False, verbose=0)
    df = bn.query2df(query)
    assert np.all(np.isin(df.columns, ['Sprinkler', 'Wet_Grass', 'p']))
    assert df.shape == (4, 3)

    # Load example mixed dataset
    df_raw = bn.import_example(data='titanic')
    # Convert to onehot
    dfhot, dfnum = bn.df2onehot(df_raw)
    dfnum.loc[0:50, 'Survived'] = 2
    # Structure learning
    DAG = bn.structure_learning.fit(dfnum, methodtype='hc', black_list=['Embarked', 'Parch', 'Name'],
                                    bw_list_method='edges')
    # Parameter learning
    model = bn.parameter_learning.fit(DAG, dfnum)
    # Make inference
    q1 = bn.inference.fit(model, variables=['Survived'], evidence={'Sex': True, 'Pclass': True}, verbose=0)
    df = bn.query2df(q1)
    assert np.all(df == q1.df)
    assert df.shape == (3, 2)


# def test_predict():
#     df = bn.import_example('asia')
#     edges = [('smoke', 'lung'),
#              ('smoke', 'bronc'),
#              ('lung', 'xray'),
#              ('bronc', 'xray')]

#     # Make the actual Bayesian DAG
#     DAG = bn.make_DAG(edges, verbose=0)
#     model = bn.parameter_learning.fit(DAG, df, verbose=3)
#     # Generate some data based on DAG
#     Xtest = bn.sampling(model, n=100)
#     out = bn.predict(model, Xtest, variables=['bronc', 'xray'])
#     assert np.all(np.isin(out.columns, ['bronc', 'xray', 'p']))
#     assert out.shape == (100, 3)
#     out = bn.predict(model, Xtest, variables=['smoke', 'bronc', 'lung', 'xray'])
#     assert np.all(np.isin(out.columns, ['xray', 'bronc', 'lung', 'smoke', 'p']))
#     assert out.shape == (100, 5)
#     out = bn.predict(model, Xtest, variables='smoke')
#     assert np.all(out.columns == ['smoke', 'p'])
#     assert out.shape == (100, 2)


def test_topological_sort():
    DAG = bn.import_DAG('sprinkler')
    # Check DAG input
    assert bn.topological_sort(DAG, 'Rain') == ['Rain', 'Wet_Grass']
    assert bn.topological_sort(DAG) == ['Cloudy', 'Sprinkler', 'Rain', 'Wet_Grass']
    # Different inputs
    assert bn.topological_sort(DAG['adjmat'], 'Rain') == ['Rain', 'Wet_Grass']
    assert bn.topological_sort(bn.adjmat2vec(DAG['adjmat']), 'Rain')
    # Check model output
    df = bn.import_example('sprinkler')
    model = bn.structure_learning.fit(df, methodtype='chow-liu', root_node='Wet_Grass')
    assert bn.topological_sort(model, 'Rain') == ['Rain', 'Cloudy', 'Sprinkler']


# def test_save():
#     # Load asia DAG
#     df = bn.import_example('asia')
#     model = bn.structure_learning.fit(df, methodtype='tan', class_node='lung')
#     bn.save(model, overwrite=True)
#     # Load the DAG
#     model_load = bn.load()
#     assert model.keys() == model_load.keys()
#     for key in model.keys():
#         if not key == 'model':
#             assert np.all(model[key] == model_load[key])

#     edges = [('smoke', 'lung'),
#              ('smoke', 'bronc'),
#              ('lung', 'xray'),
#              ('bronc', 'xray')]

#     # Make the actual Bayesian DAG
#     DAG = bn.make_DAG(edges, verbose=0)
#     # Save the DAG
#     bn.save(DAG, overwrite=True)
#     # Load the DAG
#     DAGload = bn.load()
#     # Compare
#     assert DAG.keys() == DAGload.keys()
#     for key in DAG.keys():
#         if not key == 'model':
#             assert np.all(DAG[key] == DAGload[key])

#     # Learn its parameters from data and perform the inference.
#     model = bn.parameter_learning.fit(DAG, df, verbose=0)
#     # Save the DAG
#     bn.save(model, overwrite=True)
#     # Load the DAG
#     model_load = bn.load()
#     # Compare
#     assert model.keys() == model_load.keys()
#     for key in model.keys():
#         if not key == 'model':
#             assert np.all(model[key] == model_load[key])


def test_independence_test():
    df = bn.import_example(data='sprinkler')
    # Structure learning of sampled dataset
    model = bn.structure_learning.fit(df)
    # Compute edge weights based on chi_square test statistic
    tests = ['chi_square', 'g_sq', 'log_likelihood', 'freeman_tuckey', 'modified_log_likelihood', 'neyman', 'cressie_read']
    for test in tests:
        model = bn.independence_test(model, df, test=test)
        assert model.get('independence_test', None) is not None
        assert set(model['independence_test'].columns) == {test, 'dof', 'p_value', 'source', 'stat_test', 'target'}
        assert model['independence_test'].columns[-2] == test
        assert np.any(model['independence_test']['stat_test'])
        assert model['independence_test'].shape[0] > 1

    # Run 10 times with random data
    for i in np.arange(0, 10):
        df = bn.import_example(data='random_discrete')
        # Parameter learning
        model = bn.structure_learning.fit(df)
        # Test for independence
        assert bn.independence_test(model, df, prune=False)
        # Test for independence
        assert bn.independence_test(model, df, prune=True)


def test_edge_properties():
    # Example 1
    edges = [('A', 'B'), ('A', 'C'), ('A', 'D')]
    # Create DAG and store in model
    model = bn.make_DAG(edges)
    edge_properties = bn.get_edge_properties(model)
    # Check availability of properties
    assert edge_properties[('A', 'B')].get('color')
    assert edge_properties[('A', 'B')].get('weight')
    assert edge_properties[('A', 'C')].get('color')
    assert edge_properties[('A', 'C')].get('weight')
    assert edge_properties[('A', 'D')].get('color')
    assert edge_properties[('A', 'D')].get('weight')
    # Make plot
    # assert bn.plot(model, edge_properties=edge_properties, interactive=False)
    # assert bn.plot(model, interactive=False)

    edges = [('A', 'B'), ('A', 'C'), ('A', 'D')]
    # Create DAG and store in model
    methodtypes = ['bayes', 'naivebayes']
    for methodtype in methodtypes:
        model = bn.make_DAG(edges, methodtype=methodtype)
        # Remove methodtype
        model['methodtype'] = ''
        # Check if it is restored to the correct methodtype
        model = bn.make_DAG(model['model'])
        assert model['methodtype'] == methodtype

    # Load asia DAG
    df = bn.import_example(data='sprinkler')
    # Structure learning of sampled dataset
    model = bn.structure_learning.fit(df)
    edge_properties1 = bn.get_edge_properties(model)
    assert np.all(pd.DataFrame(edge_properties1).iloc[1, :] == 1)
    # Compute edge weights based on chi_square test statistic
    model = bn.independence_test(model, df, test='chi_square')
    # Get the edge properties
    edge_properties2 = bn.get_edge_properties(model)
    assert np.sum(pd.DataFrame(edge_properties2).iloc[1, :] > 1) > len(edge_properties2) - 2


def test_structure_scores():
    # Example 1
    # Load example dataset
    df = bn.import_example('sprinkler')
    edges = [('Cloudy', 'Sprinkler'),
             ('Cloudy', 'Rain'),
             ('Sprinkler', 'Wet_Grass'),
             ('Rain', 'Wet_Grass')]

    # Make the actual Bayesian DAG
    DAG = bn.make_DAG(edges)
    model = bn.parameter_learning.fit(DAG, df)
    assert set([*model['structure_scores'].keys()]) == {'bdeu', 'bds', 'bic', 'k2'}
    assert np.all(np.round([*model['structure_scores'].values()]) == np.array([-1953., -1953., -1954., -1961.]))
    # Print CPDs
    CPD = bn.print_CPD(model)
    bn.check_model(CPD)
    bn.check_model(model)

    df = bn.import_example('asia')
    model = bn.structure_learning.fit(df)
    assert set([*model['structure_scores'].keys()]) ==  {'bdeu', 'bds', 'bic', 'k2'}
