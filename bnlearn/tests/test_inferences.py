import pytest
import bnlearn as bn
from pgmpy.factors.discrete import TabularCPD
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests


def test_inference_sprinkler_example():
# Define the causal dependencies based on your expert/domain knowledge.

    # Define the network structure
    edges = [
        ('Cloudy', 'Sprinkler'),
        ('Cloudy', 'Rain'),
        ('Sprinkler', 'Wet_Grass'),
        ('Rain', 'Wet_Grass'),
    ]

    # Create the DAG
    model = bn.make_DAG(edges)
    DAG = model['model']
    CPDs = {}
    for cpd in DAG.get_cpds():
        CPDs[cpd.variable] = bn.query2df(cpd, verbose=0)['p']


    # Check that the CPDs match the expected output (all probabilities 0.5)
    expected_cpds = {
        'Sprinkler': [0.5, 0.5, 0.5, 0.5],
        'Rain': [0.5, 0.5, 0.5, 0.5],
        'Wet_Grass': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        'Cloudy': [0.5, 0.5],
    }
    for var, expected_probs in expected_cpds.items():
        assert var in CPDs
        actual = CPDs[var].reset_index(drop=True)
        np.testing.assert_allclose(actual.values, expected_probs, rtol=1e-6)


    # Define CPDs
    cpt_cloudy = TabularCPD(variable='Cloudy', variable_card=2, values=[[0.3], [0.7]])
    cpt_rain = TabularCPD(variable='Rain', variable_card=2,
                          values=[[0.8, 0.2],
                                  [0.2, 0.8]],
                          evidence=['Cloudy'], evidence_card=[2])
    cpt_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2,
                               values=[[0.5, 0.9],
                                       [0.5, 0.1]],
                               evidence=['Cloudy'], evidence_card=[2])
    cpt_wet_grass = TabularCPD(variable='Wet_Grass', variable_card=2,
                               values=[[1, 0.1, 0.1, 0.01],
                                       [0, 0.9, 0.9, 0.99]],
                               evidence=['Sprinkler', 'Rain'],
                               evidence_card=[2, 2])

    # Create the DAG with CPDs
    model = bn.make_DAG(edges, CPD=[cpt_cloudy, cpt_sprinkler, cpt_rain, cpt_wet_grass])


    DAG = model['model']
    CPDs = {}
    for cpd in DAG.get_cpds():
        CPDs[cpd.variable] = bn.query2df(cpd, verbose=0)['p']
        

    # Check that the CPDs match the expected output (provided values)
    expected_cpds = {
        'Cloudy': [0.3, 0.7],
        'Sprinkler': [0.5, 0.9, 0.5, 0.1],
        'Rain': [0.8, 0.2, 0.2, 0.8],
        'Wet_Grass': [1.00, 0.10, 0.10, 0.01, 0.00, 0.90, 0.90, 0.99],
    }
    # Perform inference
    q1 = bn.inference.fit(model, variables=['Wet_Grass'], evidence={'Sprinkler': 0}, to_df=True)

    # Assert the result is a DataFrame with correct columns and shape
    assert hasattr(q1, 'df')
    assert q1.df is not None
    assert 'Wet_Grass' in q1.df.columns
    assert 'p' in q1.df.columns
    assert q1.df.shape[0] == 2  # Wet_Grass is binary
    # Probabilities should sum to 1
    assert abs(q1.df['p'].sum() - 1.0) < 1e-6




def test_inference_do_operator():
    # Ground truth computed by hand from the sprinkler CPDs:
    # P(W=1|S=1) weights Cloudy by P(C|S=1) (observation), whereas
    # P(W=1|do(S=1)) cuts Cloudy->Sprinkler and weights by the prior P(C).
    model = bn.import_DAG('sprinkler', verbose=0)

    def p1(query):
        return float(query.df.loc[query.df['Wet_Grass'] == 1, 'p'].iloc[0])

    observational = bn.inference.fit(model, variables=['Wet_Grass'], evidence={'Sprinkler': 1}, verbose=0)
    assert p1(observational) == pytest.approx(0.927, abs=1e-3)

    interventional = bn.inference.fit(model, variables=['Wet_Grass'], do={'Sprinkler': 1}, verbose=0)
    assert p1(interventional) == pytest.approx(0.945, abs=1e-3)
    assert abs(interventional.df['p'].sum() - 1.0) < 1e-6

    # do and evidence combine: conditioning the intervened network on Cloudy=1
    combined = bn.inference.fit(model, variables=['Wet_Grass'], do={'Sprinkler': 1}, evidence={'Cloudy': 1}, verbose=0)
    assert p1(combined) == pytest.approx(0.972, abs=1e-3)

    # Evidence on a variable outside the intervention's adjustment set: in the
    # mutilated network Wet_Grass depends only on its parents, so this is
    # exactly P(W=1|S=1,R=1)=0.99 from the CPT. (pgmpy's adjustment-based
    # CausalInference.query returns 0.9612 here.)
    outside = bn.inference.fit(model, variables=['Wet_Grass'], do={'Sprinkler': 1}, evidence={'Rain': 1}, verbose=0)
    assert p1(outside) == pytest.approx(0.99, abs=1e-6)

    # The other query options keep working with do
    marginals = bn.inference.fit(model, variables=['Wet_Grass', 'Rain'], do={'Sprinkler': 1}, joint=False, to_df=False, verbose=0)
    assert set(marginals.keys()) == {'Wet_Grass', 'Rain'}

    # interventions are labeled as do(X) in the readable summary
    assert 'do(Sprinkler)=1' in interventional.text


def test_inference_do_operator_related_targets():
    # Intervening on causally related nodes (A is a parent of B): the mutilated
    # network holds both fixed, so P(C=1|do(A=1),do(B=1)) = P(C=1|A=1,B=1) = 1.
    # (pgmpy's adjustment-based CausalInference.query returns 0.7 here.)
    edges = [('A', 'B'), ('A', 'C'), ('B', 'C')]
    cpt_a = TabularCPD(variable='A', variable_card=2, values=[[0.6], [0.4]])
    cpt_b = TabularCPD(variable='B', variable_card=2,
                       values=[[0.7, 0.2],
                               [0.3, 0.8]],
                       evidence=['A'], evidence_card=[2])
    cpt_c = TabularCPD(variable='C', variable_card=2,
                       values=[[0.5, 0.5, 0.5, 0.0],
                               [0.5, 0.5, 0.5, 1.0]],
                       evidence=['A', 'B'], evidence_card=[2, 2])
    model = bn.make_DAG(edges, CPD=[cpt_a, cpt_b, cpt_c], verbose=0)

    query = bn.inference.fit(model, variables=['C'], do={'A': 1, 'B': 1}, verbose=0)
    assert float(query.df.loc[query.df['C'] == 1, 'p'].iloc[0]) == pytest.approx(1.0, abs=1e-6)


def test_inference_do_operator_validation():
    model = bn.import_DAG('sprinkler', verbose=0)
    with pytest.raises(Exception, match='do'):
        bn.inference.fit(model, variables=['Wet_Grass'], do={'NotANode': 1}, verbose=0)
    with pytest.raises(Exception, match='both'):
        bn.inference.fit(model, variables=['Wet_Grass'], do={'Rain': 1}, evidence={'Rain': 1}, verbose=0)


def test_make_DAG_naivebayes():
    edges = [('A', 'B'), ('A', 'C'), ('A', 'D')]
    DAG = bn.make_DAG(edges, methodtype='naivebayes')
    fig = bn.plot(DAG)
    assert isinstance(DAG, dict)
    assert 'model' in DAG and 'adjmat' in DAG
    assert set(fig).issuperset({'fig', 'ax', 'pos', 'G', 'node_properties', 'edge_properties'})


def test_make_DAG_naivebayes_with_cpd():
    edges = [('A', 'B'), ('A', 'C'), ('A', 'D')]
    CPD = bn.build_cpts_from_structure(edges, variable_card=3)
    DAG2 = bn.make_DAG(edges, CPD=CPD, methodtype='naivebayes')
    fig = bn.plot(DAG2)
    assert isinstance(DAG2, dict)
    assert 'model' in DAG2 and 'adjmat' in DAG2
    assert set(fig).issuperset({'fig', 'ax', 'pos', 'G', 'node_properties', 'edge_properties'})


def test_make_DAG_markov():
    edges = [('A', 'B'), ('A', 'C'), ('A', 'D')]
    DAG = bn.make_DAG(edges, methodtype='markov')
    fig = bn.plot(DAG)
    assert isinstance(DAG, dict)
    assert 'model' in DAG and 'adjmat' in DAG
    assert set(fig).issuperset({'fig', 'ax', 'pos', 'G', 'node_properties', 'edge_properties'})


def test_make_DAG_DBN():
    edges = [('A', 'B'), ('A', 'C'), ('A', 'D')]
    DAG = bn.make_DAG(edges, methodtype='DBN')
    fig = bn.plot(DAG)
    assert isinstance(DAG, dict)
    assert 'model' in DAG and 'adjmat' in DAG
    assert set(fig).issuperset({'fig', 'ax', 'pos', 'G', 'node_properties', 'edge_properties'})

    # Set custom timeslice
    edges_ts = [(('A', 0), ('B', 1)), (('A', 0), ('C', 0)), (('A', 0), ('D', 0))]
    DAG_ts = bn.make_DAG(edges_ts, methodtype='DBN')
    fig_ts = bn.plot(DAG_ts)
    assert isinstance(DAG_ts, dict)
    assert 'model' in DAG_ts and 'adjmat' in DAG_ts
    assert set(fig_ts).issuperset({'fig', 'ax', 'pos', 'G', 'node_properties', 'edge_properties'})

