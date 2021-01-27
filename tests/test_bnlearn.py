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
    randdata=['sprinkler','alarm','andes','asia','pathfinder','sachs']
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


def test_parameter_learning():
    df = bn.import_example()
    model = bn.import_DAG('sprinkler', CPD=False)
    model_update = bn.parameter_learning.fit(model, df)
    assert [*model_update.keys()]==['model', 'adjmat', 'config']


def test_inference():
    DAG = bn.import_DAG('sprinkler')
    q1 = bn.inference.fit(DAG, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})
    assert 'pgmpy.factors.discrete.DiscreteFactor.DiscreteFactor' in str(type(q1))
