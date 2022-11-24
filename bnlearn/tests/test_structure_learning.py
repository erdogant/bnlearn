import bnlearn as bn
import numpy as np
import pytest

from pgmpy.estimators import TreeSearch
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination


@pytest.fixture
def sprinkler():
    return bn.import_example()


@pytest.fixture
def asia_sampling():
    asia_dag = bn.import_DAG('asia')
    return bn.sampling(asia_dag, n=1000)


def test_default_method(sprinkler):
    model = bn.structure_learning.fit(sprinkler)
    assert model.keys() == {'model', 'model_edges', 'adjmat', 'config', 'structure_scores'}


def test_hc(sprinkler):
    model = bn.structure_learning.fit(sprinkler, methodtype='hc', scoretype='bic')
    assert model.keys() == {'model', 'model_edges', 'adjmat', 'config', 'structure_scores'}


def test_hc_bic(sprinkler):
    model = bn.structure_learning.fit(sprinkler, methodtype='hc', scoretype='bic')
    assert model.keys() == {'model', 'model_edges', 'adjmat', 'config', 'structure_scores'}


def test_hc_k2(sprinkler):
    model = bn.structure_learning.fit(sprinkler, methodtype='hc', scoretype='k2')
    assert model.keys() == {'model', 'model_edges', 'adjmat', 'config', 'structure_scores'}


def test_cs_bdeu(sprinkler):
    model = bn.structure_learning.fit(sprinkler, methodtype='cs', scoretype='bdeu')
    assert model.keys() == {'undirected', 'undirected_edges', 'pdag', 'pdag_edges', 'dag', 'dag_edges', 'model',
                            'model_edges', 'adjmat', 'config', 'structure_scores'}


def test_cs_k2(sprinkler):
    model = bn.structure_learning.fit(sprinkler, methodtype='cs', scoretype='k2')
    assert model.keys() == {'undirected', 'undirected_edges', 'pdag', 'pdag_edges', 'dag', 'dag_edges', 'model',
                            'model_edges', 'adjmat', 'config', 'structure_scores'}


def test_ex_bdeu(sprinkler):
    model = bn.structure_learning.fit(sprinkler, methodtype='ex', scoretype='bdeu')
    assert model.keys() == {'model', 'model_edges', 'adjmat', 'config', 'structure_scores'}


def test_ex_k2(sprinkler):
    model = bn.structure_learning.fit(sprinkler, methodtype='ex', scoretype='k2')
    assert model.keys() == {'model', 'model_edges', 'adjmat', 'config', 'structure_scores'}


def test_cl(sprinkler):
    model = bn.structure_learning.fit(sprinkler, methodtype='cl', root_node='Cloudy')
    assert model.keys() == {'model', 'model_edges', 'adjmat', 'config', 'structure_scores'}


def test_tan(sprinkler):
    model = bn.structure_learning.fit(sprinkler, methodtype='tan', root_node='Cloudy', class_node='Rain')
    assert model.keys() == {'model', 'model_edges', 'adjmat', 'config', 'structure_scores'}


def test_filtering_no_filter(asia_sampling):
    model = bn.structure_learning.fit(asia_sampling)
    assert set(model['adjmat'].columns.values) == {'smoke', 'bronc', 'lung', 'asia', 'tub', 'either', 'dysp', 'xray'}


def test_hc_white_list_nodes(asia_sampling):
    model = bn.structure_learning.fit(asia_sampling, methodtype='hc', white_list=['smoke', 'either'],
                                      bw_list_method='nodes')
    assert set(model['adjmat'].columns) == {'smoke', 'either'}


def test_hc_white_list_edges(asia_sampling):
    model = bn.structure_learning.fit(asia_sampling, methodtype='hc', white_list=['smoke', 'either'],
                                      bw_list_method='edges')
    assert set(model['adjmat'].columns) == {'smoke', 'bronc', 'lung', 'asia', 'tub', 'either', 'dysp', 'xray'}


def test_hc_black_list_nodes(asia_sampling):
    model = bn.structure_learning.fit(asia_sampling, methodtype='hc', black_list=['smoke', 'either'],
                                      bw_list_method='nodes')
    assert set(model['adjmat'].columns) == {'bronc', 'lung', 'asia', 'tub', 'dysp', 'xray'}


def test_hc_black_list_edges(asia_sampling):
    model = bn.structure_learning.fit(asia_sampling, methodtype='hc', scoretype='bic', black_list=['smoke', 'either'],
                                      bw_list_method='edges')
    assert set(model['adjmat'].columns) == {'smoke', 'bronc', 'lung', 'asia', 'tub', 'either', 'dysp', 'xray'}


def test_ex_white_list_nodes(asia_sampling):
    model = bn.structure_learning.fit(asia_sampling, methodtype='ex', white_list=['smoke', 'either'],
                                      bw_list_method='nodes')
    assert set(model['adjmat'].columns) == {'either', 'smoke'}


def test_ex_black_list_nodes(asia_sampling):
    model = bn.structure_learning.fit(asia_sampling, methodtype='ex',
                                      black_list=['asia', 'tub', 'either', 'dysp', 'xray'],
                                      bw_list_method='nodes')
    assert set(model['adjmat'].columns) == {'bronc', 'lung', 'smoke'}


def test_filtering(asia_sampling):
    # cs filter
    model = bn.structure_learning.fit(asia_sampling, methodtype='cs', white_list=['smoke', 'either'],
                                      bw_list_method='nodes')
    assert np.all(np.isin(model['adjmat'].columns.values, ['smoke', 'either']))
    model = bn.structure_learning.fit(asia_sampling, methodtype='cs',
                                      black_list=['asia', 'tub', 'either', 'dysp', 'xray'],
                                      bw_list_method='nodes')
    assert np.all(np.isin(model['adjmat'].columns.values, ['smoke', 'lung', 'bronc']))
    # cl filter
    model = bn.structure_learning.fit(asia_sampling, methodtype='cl', white_list=['smoke', 'either'],
                                      bw_list_method='nodes',
                                      root_node='smoke')
    assert np.all(model['adjmat'].columns.values == ['smoke', 'either'])
    # tan
    model = bn.structure_learning.fit(asia_sampling, methodtype='tan', white_list=['smoke', 'either'],
                                      bw_list_method='nodes',
                                      root_node='smoke', class_node='either')
    assert np.all(model['adjmat'].columns.values == ['smoke', 'either'])
    # naivebayes
    model = bn.structure_learning.fit(asia_sampling, methodtype='naivebayes', root_node="smoke")
    assert np.all(model['adjmat'].columns.values == ['smoke', 'asia', 'tub', 'lung', 'bronc', 'either', 'xray', 'dysp'])


@pytest.mark.skip(reason='find a faster way to test this')
def test_compare_pgmpy(sprinkler):
    andes = bn.import_example(data='andes')

    # PGMPY
    est = TreeSearch(andes)
    dag = est.estimate(estimator_type="tan", class_node='DISPLACEM0')
    bnq = BayesianNetwork(dag.edges())
    bnq.fit(andes, estimator=None)  # None means maximum likelihood estimator
    bn_infer = VariableElimination(bnq)
    q = bn_infer.query(variables=['DISPLACEM0'], evidence={'RApp1': 1})
    # print(q)

    # BNLEARN
    model = bn.structure_learning.fit(andes, methodtype='tan', class_node='DISPLACEM0', scoretype='bic')
    model_bn = bn.parameter_learning.fit(model, andes, methodtype='ml')  # maximum likelihood estimator
    query = bn.inference.fit(model_bn, variables=['DISPLACEM0'], evidence={'RApp1': 1})

    # DAG COMPARISON
    assert np.all(model_bn['adjmat'] == model['adjmat'])
    assert list(dag.edges()) == list(model['model'].edges())
    assert list(dag.edges()) == model['model_edges']

    # COMPARE THE CPDs names
    qbn_cpd = []
    bn_cpd = []
    for cpd in bnq.get_cpds(): qbn_cpd.append(cpd.variable)
    for cpd in model_bn['model'].get_cpds(): bn_cpd.append(cpd.variable)

    assert len(bn_cpd) == len(qbn_cpd)
    assert np.all(np.isin(bn_cpd, qbn_cpd))

    # COMPARE THE CPD VALUES
    nr_diff = 0
    for cpd_bnlearn in model_bn['model'].get_cpds():
        for cpd_pgmpy in bnq.get_cpds():
            if cpd_bnlearn.variable == cpd_pgmpy.variable:
                assert np.all(cpd_bnlearn.values == cpd_pgmpy.values)
                # if not np.all(cpd_bnlearn.values==cpd_pgmpy.values):
                # print('%s-%s'%(cpd_bnlearn.variable, cpd_pgmpy.variable))
                # print(cpd_bnlearn)
                # print(cpd_pgmpy)
                # nr_diff=nr_diff+1
                # input('press enter to see the next difference in CPD.')
