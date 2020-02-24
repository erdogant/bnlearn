# pytest tests\test_bnlearn.py

import bnlearn
from pgmpy.factors.discrete import TabularCPD
import numpy as np


def test_import_DAG():
    DAG = bnlearn.import_DAG('Sprinkler')
    # TEST 1: check output is unchanged
    assert [*DAG.keys()]==['model','adjmat']
    # TEST 2: Check model output is unchanged
    assert DAG['adjmat'].sum().sum()==4
    # TEST 3:
    assert 'pgmpy.models.BayesianModel.BayesianModel' in str(type(DAG['model']))
    # TEST 4:
    DAG = bnlearn.import_DAG('alarm', verbose=0)
    assert [*DAG.keys()]==['model','adjmat']
    DAG = bnlearn.import_DAG('andes', verbose=0)
    assert [*DAG.keys()]==['model','adjmat']
    DAG = bnlearn.import_DAG('asia', verbose=0)
    assert [*DAG.keys()]==['model','adjmat']


def test_make_DAG():
    edges = [('Cloudy', 'Sprinkler')]
    DAG = bnlearn.make_DAG(edges)
    # TEST 1
    assert 'pgmpy.models.BayesianModel.BayesianModel' in str(type(DAG))
    # TEST 2
    cpt_cloudy = TabularCPD(variable='Cloudy', variable_card=2, values=[[0.3], [0.7]])
    cpt_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2, values=[[0.4, 0.9], [0.6, 0.1]], evidence=['Cloudy'], evidence_card=[2])
    assert bnlearn.make_DAG(DAG, CPD=[cpt_cloudy, cpt_sprinkler], checkmodel=True)


def test_make_DAG():
    # TEST 1:
    df = bnlearn.import_example()
    assert df.shape==(1000,4)


def test_sampling():
    # TEST 1:
    model = bnlearn.import_DAG('Sprinkler')
    n = np.random.randint(10,1000)
    df = bnlearn.sampling(model, n=n)
    assert df.shape==(n,4)


def test_to_undirected():
    # TEST 1:
    randdata=['sprinkler','alarm','andes','asia','pathfinder','sachs']
    n = np.random.randint(0,len(randdata))
    DAG = bnlearn.import_DAG(randdata[n], CPD=False, verbose=0)
    assert (DAG['adjmat'].sum().sum()*2)==bnlearn.to_undirected(DAG['adjmat']).sum().sum()
    
    
def test_compare_networks():
    DAG = bnlearn.import_DAG('Sprinkler', verbose=0)
    G = bnlearn.compare_networks(DAG, DAG, showfig=False)
    assert np.all(G[0]==[[12,0],[0,4]])


def test_adjmat2vec():
    DAG = bnlearn.import_DAG('Sprinkler', verbose=0)
    out = bnlearn.adjmat2vec(DAG['adjmat'])
    assert np.all(out['source']==['Cloudy','Cloudy','Sprinkler','Rain'])
    

def test_vec2adjmat():
    DAG = bnlearn.import_DAG('Sprinkler', verbose=0)
    out = bnlearn.adjmat2vec(DAG['adjmat'])
    # TEST: conversion
    assert bnlearn.vec2adjmat(out['source'], out['target']).shape==DAG['adjmat'].shape


def test_structure_learning():
    df = bnlearn.import_example()
    model = bnlearn.structure_learning.fit(df)
    assert [*model.keys()]==['model', 'model_edges', 'adjmat']

    
def test_parameter_learning():
    df = bnlearn.import_example()
    model = bnlearn.import_DAG('sprinkler', CPD=False)
    model_update = bnlearn.parameter_learning.fit(model, df)
    assert [*model_update.keys()]==['model', 'adjmat']


def test_inference():
    DAG = bnlearn.import_DAG('sprinkler')
    q1 = bnlearn.inference.fit(DAG, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})
    assert 'pgmpy.factors.discrete.DiscreteFactor.DiscreteFactor' in str(type(q1))
