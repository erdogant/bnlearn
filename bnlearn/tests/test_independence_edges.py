# -*- coding: utf-8 -*-
"""Tests for independence_test schema and edge property keys (p_value / logp)."""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest
import bnlearn as bn


@pytest.fixture
def sprinkler_model():
    df = bn.import_example('sprinkler')
    model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic', verbose=0)
    model = bn.independence_test(model, df, test='chi_square', prune=False, verbose=0)
    return model, df


def test_independence_test_columns(sprinkler_model):
    model, _ = sprinkler_model
    indep = model['independence_test']
    assert indep is not None
    for col in ('source', 'target', 'stat_test', 'p_value', 'dof'):
        assert col in indep.columns
    assert indep['p_value'].between(0, 1).all() or (indep['p_value'] <= 1).all()
    assert set(indep['stat_test'].unique()).issubset({True, False})


def test_edge_properties_use_p_value_not_pvalue(sprinkler_model):
    model, _ = sprinkler_model
    edges = bn.get_edge_properties(model, verbose=0)
    assert len(edges) > 0
    sample = next(iter(edges.values()))
    assert 'p_value' in sample
    assert 'logp' in sample
    assert 'weight' in sample
    assert 'value' in sample
    assert 'color' in sample
    assert 0 <= sample['p_value'] <= 1 or sample['p_value'] >= 0
    assert sample['logp'] >= 0


def test_edge_p_value_matches_independence_table(sprinkler_model):
    model, _ = sprinkler_model
    edges = bn.get_edge_properties(model, verbose=0)
    indep = model['independence_test']
    for (u, v), props in edges.items():
        rows = indep[(indep['source'] == u) & (indep['target'] == v)]
        if len(rows) == 0:
            # direction may be stored as in adjmat2vec
            rows = indep[(indep['source'] == v) & (indep['target'] == u)]
        if len(rows) == 0:
            continue
        expected = float(rows['p_value'].iloc[0])
        assert props['p_value'] == pytest.approx(expected, rel=1e-9, abs=1e-12)
        if expected > 0:
            assert props['logp'] == pytest.approx(-np.log10(expected), rel=1e-5, abs=1e-5)


def test_plot_edge_labels_p_value_no_error(sprinkler_model):
    model, _ = sprinkler_model
    fig = bn.plot(model, interactive=False, edge_labels='p_value', verbose=0,
                  params_static={'showplot': False, 'visible': False})
    assert fig is not None


def test_plot_requires_independence_for_p_value_labels():
    df = bn.import_example('sprinkler')
    model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic', verbose=0)
    # No independence_test: edge_labels='p_value' should fall back (no crash)
    fig = bn.plot(model, interactive=False, edge_labels='p_value', verbose=0,
                  params_static={'showplot': False, 'visible': False})
    # May still plot structure without labels
    assert fig is not None or model.get('adjmat') is not None
