# pip install pytest
# pytest tests\test_bn.py

import bnlearn as bn
from pgmpy.factors.discrete import TabularCPD
import numpy as np


def test_import_DAG():
    DAG = bn.import_DAG('Sprinkler')
    # TEST 1: check output is unchanged
    assert [*DAG.keys()]==['model','adjmat']
    # TEST 2: Check model output is unchanged
    assert DAG['adjmat'].sum().sum()==4
    # TEST 3:
    assert 'pgmpy.models.BayesianModel.BayesianModel' in str(type(DAG['model']))
    # TEST 4:
    DAG = bn.import_DAG('alarm', verbose=0)
    assert [*DAG.keys()]==['model','adjmat']
    DAG = bn.import_DAG('andes', verbose=0)
    assert [*DAG.keys()]==['model','adjmat']
    DAG = bn.import_DAG('asia', verbose=0)
    assert [*DAG.keys()]==['model','adjmat']


def test_make_DAG():
    edges = [('Cloudy', 'Sprinkler')]
    DAG = bn.make_DAG(edges)
    # TEST 1
    assert 'pgmpy.models.BayesianModel.BayesianModel' in str(type(DAG['model']))
    # TEST 2
    cpt_cloudy = TabularCPD(variable='Cloudy', variable_card=2, values=[[0.3], [0.7]])
    cpt_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2, values=[[0.4, 0.9], [0.6, 0.1]], evidence=['Cloudy'], evidence_card=[2])
    assert bn.make_DAG(DAG, CPD=[cpt_cloudy, cpt_sprinkler], checkmodel=True)


def test_make_DAG():
    # TEST 1:
    df = bn.import_example()
    assert df.shape==(1000,4)


def test_sampling():
    # TEST 1:
    model = bn.import_DAG('Sprinkler')
    n = np.random.randint(10,1000)
    df = bn.sampling(model, n=n)
    assert df.shape==(n, 4)


def test_to_undirected():
    # TEST 1:
    randdata=['sprinkler','alarm','andes','asia','sachs']
    n = np.random.randint(0,len(randdata))
    DAG = bn.import_DAG(randdata[n], CPD=False, verbose=0)
    assert (DAG['adjmat'].sum().sum()*2)==bn.to_undirected(DAG['adjmat']).sum().sum()


def test_compare_networks():
    DAG = bn.import_DAG('Sprinkler', verbose=0)
    G = bn.compare_networks(DAG, DAG, showfig=False)
    assert np.all(G[0]==[[12,0],[0,4]])


def test_adjmat2vec():
    DAG = bn.import_DAG('Sprinkler', verbose=0)
    out = bn.adjmat2vec(DAG['adjmat'])
    assert np.all(out['source']==['Cloudy','Cloudy','Sprinkler','Rain'])


def test_vec2adjmat():
    DAG = bn.import_DAG('Sprinkler', verbose=0)
    out = bn.adjmat2vec(DAG['adjmat'])
    # TEST: conversion
    assert bn.vec2adjmat(out['source'], out['target']).shape==DAG['adjmat'].shape


def test_structure_learning():
    import bnlearn as bn
    df = bn.import_example()
    model = bn.structure_learning.fit(df)
    assert [*model.keys()]==['model', 'model_edges', 'adjmat', 'config']
    model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
    assert [*model.keys()]==['model', 'model_edges', 'adjmat', 'config']
    model = bn.structure_learning.fit(df, methodtype='hc', scoretype='k2')
    assert [*model.keys()]==['model', 'model_edges', 'adjmat', 'config']
    model = bn.structure_learning.fit(df, methodtype='cs', scoretype='bdeu')
    assert [*model.keys()]==['undirected', 'undirected_edges', 'pdag', 'pdag_edges', 'dag', 'dag_edges', 'model', 'model_edges', 'adjmat', 'config']
    model = bn.structure_learning.fit(df, methodtype='cs', scoretype='k2')
    assert [*model.keys()]==['undirected', 'undirected_edges', 'pdag', 'pdag_edges', 'dag', 'dag_edges', 'model', 'model_edges', 'adjmat', 'config']
    model = bn.structure_learning.fit(df, methodtype='ex', scoretype='bdeu')
    assert [*model.keys()]==['model', 'model_edges', 'adjmat', 'config']
    model = bn.structure_learning.fit(df, methodtype='ex', scoretype='k2')
    assert [*model.keys()]==['model', 'model_edges', 'adjmat', 'config']
    model = bn.structure_learning.fit(df, methodtype='cl', root_node='Cloudy')
    assert [*model.keys()]==['model', 'model_edges', 'adjmat', 'config']
    model = bn.structure_learning.fit(df, methodtype='tan', root_node='Cloudy', class_node='Rain')
    assert [*model.keys()]==['model', 'model_edges', 'adjmat', 'config']

    # Test the filtering
    DAG = bn.import_DAG('asia')
    # Sampling
    df = bn.sampling(DAG, n=1000)
    # Structure learning of sampled dataset
    model = bn.structure_learning.fit(df)
    assert np.all(np.isin(model['adjmat'].columns.values, ['smoke', 'bronc', 'lung', 'asia', 'tub', 'either', 'dysp', 'xray']))

    # hc filter on edges
    model = bn.structure_learning.fit(df, methodtype='hc', white_list=['smoke', 'either'], bw_list_method='nodes')
    assert np.all(model['adjmat'].columns.values==['smoke', 'either'])
    model = bn.structure_learning.fit(df, methodtype='hc', white_list=[('smoke', 'either')], bw_list_method='edges')
    assert np.all(np.isin(model['adjmat'].columns.values, ['smoke', 'bronc', 'lung', 'asia', 'tub', 'either', 'dysp', 'xray']))
    model = bn.structure_learning.fit(df, methodtype='hc', black_list=['smoke', 'either'], bw_list_method='nodes')
    assert np.all(np.isin(model['adjmat'].columns.values, ['bronc', 'lung', 'asia', 'tub', 'dysp', 'xray']))
    model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic', black_list=['smoke', 'either'], bw_list_method='edges')
    assert np.all(np.isin(model['adjmat'].columns.values, ['smoke', 'bronc', 'lung', 'asia', 'tub', 'either', 'dysp', 'xray']))
    # hc filter on node
    model = bn.structure_learning.fit(df, methodtype='ex', white_list=['smoke', 'either'], bw_list_method='nodes')
    assert np.all(model['adjmat'].columns.values==['either', 'smoke'])
    model = bn.structure_learning.fit(df, methodtype='ex', black_list=['asia', 'tub', 'either', 'dysp', 'xray'], bw_list_method='nodes')
    assert np.all(model['adjmat'].columns.values==['bronc', 'lung', 'smoke'])
    # cs filter
    model = bn.structure_learning.fit(df, methodtype='cs', white_list=['smoke', 'either'], bw_list_method='nodes')
    assert np.all(np.isin(model['adjmat'].columns.values, ['smoke', 'either']))
    model= bn.structure_learning.fit(df, methodtype='cs', black_list=['asia', 'tub', 'either', 'dysp', 'xray'], bw_list_method='nodes')
    assert np.all(np.isin(model['adjmat'].columns.values, ['smoke', 'lung', 'bronc']))
    # cl filter
    model = bn.structure_learning.fit(df, methodtype='cl', white_list=['smoke', 'either'], bw_list_method='nodes', root_node='smoke')
    assert np.all(model['adjmat'].columns.values==['smoke', 'either'])
    # tan
    model = bn.structure_learning.fit(df, methodtype='tan', white_list=['smoke', 'either'], bw_list_method='nodes', root_node='smoke', class_node='either')
    assert np.all(model['adjmat'].columns.values==['smoke', 'either'])


def test_parameter_learning():
    df = bn.import_example()
    model = bn.import_DAG('sprinkler', CPD=False)
    model_update = bn.parameter_learning.fit(model, df)
    assert [*model_update.keys()]==['model', 'adjmat', 'config']


def test_inference():
    DAG = bn.import_DAG('sprinkler')
    q1 = bn.inference.fit(DAG, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1}, to_df=False, verbose=0)
    assert 'pgmpy.factors.discrete.DiscreteFactor.DiscreteFactor' in str(type(q1))
    assert q1.df is None
    q1 = bn.inference.fit(DAG, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1}, to_df=True, verbose=0)
    assert q1.df is not None

def test_query2df():
    DAG = bn.import_DAG('sprinkler')
    query = bn.inference.fit(DAG, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1}, to_df=False, verbose=0)
    df = bn.query2df(query)
    assert df.shape==(2,2)
    assert np.all(df.columns==['Wet_Grass', 'p'])
    query = bn.inference.fit(DAG, variables=['Wet_Grass', 'Sprinkler'], evidence={'Rain':1, 'Cloudy':1}, to_df=False, verbose=0)
    df = bn.query2df(query)
    assert np.all(np.isin(df.columns, ['Sprinkler', 'Wet_Grass', 'p']))
    assert df.shape==(4,3)

def test_predict():
    df = bn.import_example('asia')
    edges = [('smoke', 'lung'),
             ('smoke', 'bronc'),
             ('lung', 'xray'),
             ('bronc', 'xray')]
    
    # Make the actual Bayesian DAG
    DAG = bn.make_DAG(edges, verbose=0)
    model = bn.parameter_learning.fit(DAG, df, verbose=3)
    # Generate some data based on DAG
    Xtest = bn.sampling(model, n=100)
    out = bn.predict(model, Xtest, variables=['bronc','xray'])
    assert np.all(np.isin(out.columns, ['bronc', 'xray', 'p']))
    assert out.shape==(100,3)
    out = bn.predict(model, Xtest, variables=['smoke','bronc','lung','xray'])
    assert np.all(np.isin(out.columns, ['xray', 'bronc', 'lung', 'smoke', 'p']))
    assert out.shape==(100,5)
    out = bn.predict(model, Xtest, variables='smoke')
    assert np.all(out.columns==['smoke', 'p'])
    assert out.shape==(100,2)

def test_topological_sort():
    DAG = bn.import_DAG('sprinkler')
    # Check DAG input
    assert bn.topological_sort(DAG, 'Rain')==['Rain', 'Wet_Grass']
    assert bn.topological_sort(DAG)==['Cloudy', 'Sprinkler', 'Rain', 'Wet_Grass']
    # Different inputs
    assert bn.topological_sort(DAG['adjmat'], 'Rain')==['Rain', 'Wet_Grass']
    assert bn.topological_sort(bn.adjmat2vec(DAG['adjmat']), 'Rain')==['Rain', 'Sprinkler']
    # Check model output
    df = bn.import_example('sprinkler')
    model = bn.structure_learning.fit(df, methodtype='chow-liu', root_node='Wet_Grass')
    assert bn.topological_sort(model, 'Rain')==['Rain', 'Cloudy', 'Sprinkler']
    
