"""Characterization tests for the bnlearn <-> pgmpy API surface.

Every public bnlearn flow that calls into pgmpy is exercised here with
assertions on observable behavior (edges, CPD normalization, probabilities,
output shapes). The suite runs fully offline (sprinkler is built in code) and
is the safety net for migrating off pgmpy==0.1.25: it must pass unchanged
before and after the migration.
"""
import os
import pickle
import shutil
import tempfile

import numpy as np
import pytest

import bnlearn as bn
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests

# Sprinkler ground truth: P(Wet_Grass=1 | Rain=1), computed exactly from the
# hand-coded CPDs via variable elimination (deterministic; locked as a
# regression value against the migration).
WETGRASS_GIVEN_RAIN = 0.9162


@pytest.fixture(scope="module")
def sprinkler_model():
    return bn.import_DAG('sprinkler')


@pytest.fixture(scope="module")
def sprinkler_df(sprinkler_model):
    # bn.sampling has no seed parameter; pgmpy 0.1.25 draws from numpy's
    # global RNG, so seed it here to keep the sampled frame deterministic
    np.random.seed(42)
    return bn.sampling(sprinkler_model, n=1000)


MINIMAL_BIF = """network unknown {
}
variable A {
  type discrete [ 2 ] { yes, no };
}
variable B {
  type discrete [ 2 ] { yes, no };
}
probability ( A ) {
  table 0.4, 0.6;
}
probability ( B | A ) {
  ( yes ) 0.9, 0.1;
  ( no ) 0.2, 0.8;
}
"""


def test_import_dag_from_bif_file(tmp_path):
    bif = tmp_path / "tiny.bif"
    bif.write_text(MINIMAL_BIF)
    model = bn.import_DAG(str(bif))
    assert set(model['model'].nodes()) == {'A', 'B'}
    assert list(model['model'].edges()) == [('A', 'B')]
    assert len(model['model'].get_cpds()) == 2
    assert model['adjmat'].loc['A', 'B']
    assert not model['adjmat'].loc['B', 'A']


def test_parameter_learning_maximum_likelihood(sprinkler_df):
    edges = [('Cloudy', 'Sprinkler'), ('Cloudy', 'Rain'),
             ('Sprinkler', 'Wet_Grass'), ('Rain', 'Wet_Grass')]
    DAG = bn.make_DAG(edges)
    model = bn.parameter_learning.fit(DAG, sprinkler_df, methodtype='ml')
    cpds = model['model'].get_cpds()
    assert len(cpds) == 4
    for cpd in cpds:
        # every conditional distribution must be normalized
        assert np.allclose(cpd.values.reshape(cpd.values.shape[0], -1).sum(axis=0), 1.0)


def test_structure_learning_naivebayes(sprinkler_df):
    model = bn.structure_learning.fit(sprinkler_df, methodtype='naivebayes', root_node='Wet_Grass')
    edges = list(model['model'].edges())
    assert len(edges) == 3
    assert all(source == 'Wet_Grass' for source, _ in edges)
    assert set(model['adjmat'].columns) == set(sprinkler_df.columns)


@pytest.mark.parametrize("scoretype", ['bds', 'aic'])
def test_structure_learning_hillclimb_scoretypes(sprinkler_df, scoretype):
    model = bn.structure_learning.fit(sprinkler_df, methodtype='hc',
                                      scoretype=scoretype)
    assert model['model'] is not None
    assert model['adjmat'].shape == (4, 4)


@pytest.fixture(scope="module")
def learned_model(sprinkler_df):
    # independence_test requires a structure-learned model (needs 'model_edges')
    return bn.structure_learning.fit(sprinkler_df, methodtype='hc',
                                     scoretype='bic')


@pytest.mark.parametrize("test", ['chi_square', 'g_sq', 'log_likelihood',
                                  'freeman_tuckey', 'modified_log_likelihood',
                                  'neyman', 'cressie_read'])
def test_independence_test_variants(learned_model, sprinkler_df, test):
    model = bn.independence_test(learned_model, sprinkler_df, test=test)
    results = model['independence_test']
    assert {'source', 'target', 'stat_test', 'p_value', test}.issubset(results.columns)
    assert len(results) == len(learned_model['model_edges'])
    assert results['p_value'].between(0, 1).all()


def test_predict(sprinkler_model, sprinkler_df):
    evidence_df = sprinkler_df[['Cloudy', 'Sprinkler', 'Rain']].head(50)
    Pout = bn.predict(sprinkler_model, evidence_df, variables=['Wet_Grass'])
    assert len(Pout) == 50
    assert {'Wet_Grass', 'p'}.issubset(Pout.columns)
    assert Pout['Wet_Grass'].isin([0, 1]).all()
    assert Pout['p'].between(0, 1).all()


@pytest.fixture
def save_dir():
    # pypickle >=2 silently refuses to write under "critical paths" (/var,
    # /private, ...), and on macOS pytest's tmp_path resolves to
    # /private/var/..., so bn.save() returns False there. Use a scratch
    # directory under the working tree instead.
    path = tempfile.mkdtemp(prefix="bnlearn_save_test_", dir=os.getcwd())
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_save_load_roundtrip(save_dir, sprinkler_model):
    filepath = os.path.join(save_dir, "model.pkl")
    assert bn.save(sprinkler_model, filepath=filepath, overwrite=True)
    loaded = bn.load(filepath)
    assert set(loaded['model'].edges()) == set(sprinkler_model['model'].edges())
    orig = {cpd.variable: cpd.values for cpd in sprinkler_model['model'].get_cpds()}
    for cpd in loaded['model'].get_cpds():
        assert np.allclose(cpd.values, orig[cpd.variable])


def test_load_rejects_pgmpy_0x_pickle(save_dir, sprinkler_model):
    # Models saved under pgmpy 0.x pickle the class path
    # pgmpy.models.BayesianNetwork.BayesianNetwork, which in pgmpy 1.x resolves
    # to a bare tombstone class: unpickling succeeds silently but yields a
    # half-broken object. bn.load() must refuse it rather than hand it back.
    from pgmpy.models.BayesianNetwork import BayesianNetwork
    zombie = BayesianNetwork.__new__(BayesianNetwork)
    zombie.__dict__.update(sprinkler_model['model'].__dict__)
    filepath = os.path.join(save_dir, "legacy_model.pkl")
    with open(filepath, "wb") as f:
        pickle.dump({'model': zombie, 'adjmat': sprinkler_model['adjmat']}, f)
    assert bn.load(filepath) is None


def test_structure_scores_all_discrete_methods(sprinkler_model, sprinkler_df):
    scores = bn.structure_scores(sprinkler_model, sprinkler_df,
                                 scoring_method=['k2', 'bic', 'bdeu', 'bds'])
    assert set(scores) == {'k2', 'bic', 'bdeu', 'bds'}
    for name, value in scores.items():
        assert np.isfinite(value), name


def test_inference_locked_value(sprinkler_model):
    query = bn.inference.fit(sprinkler_model, variables=['Wet_Grass'],
                             evidence={'Rain': 1}, to_df=True)
    p1 = float(query.df.loc[query.df['Wet_Grass'] == 1, 'p'].iloc[0])
    assert p1 == pytest.approx(WETGRASS_GIVEN_RAIN, abs=1e-4)


def test_print_cpd_returns_all_nodes(sprinkler_model):
    CPDs = bn.print_CPD(sprinkler_model)
    assert set(CPDs) == {'Cloudy', 'Sprinkler', 'Rain', 'Wet_Grass'}
    for df in CPDs.values():
        assert 'p' in df.columns
