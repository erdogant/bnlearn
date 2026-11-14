"""Tests for structure_learning.fit(methodtype='hc', skeleton='hnet')."""
import numpy as np
import pandas as pd
import pytest

import bnlearn as bn

pytest.importorskip('hnet')


def _mixed_df(n=2000, seed=1):
    """A -> B (categorical), A -> X1 -> X2 (numeric), C and X3 are independent."""
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 2, n)
    B = (A ^ (rng.random(n) < 0.2)).astype(int)
    C = rng.integers(0, 3, n)
    X1 = A * 2.0 + rng.normal(size=n)
    X2 = 0.8 * X1 + rng.normal(size=n)
    X3 = rng.normal(size=n)
    return pd.DataFrame(dict(A=A, B=B, C=C, X1=X1, X2=X2, X3=X3))


def _undirected(edges):
    return {frozenset(e) for e in edges}


def test_default_path_unchanged():
    df = _mixed_df()
    a = bn.structure_learning.fit(df.copy(), methodtype='hc')
    b = bn.structure_learning.fit(df.copy(), methodtype='hc', skeleton=None)
    assert sorted(a['model_edges']) == sorted(b['model_edges'])
    assert 'skeleton' not in a


def test_invalid_skeleton_and_methodtype():
    df = _mixed_df(n=200)
    with pytest.raises(Exception, match='skeleton'):
        bn.structure_learning.fit(df, methodtype='hc', skeleton='unknown')
    with pytest.raises(Exception, match='only works with'):
        bn.structure_learning.fit(df, methodtype='pc', skeleton='hnet')


def test_skeleton_mixed_data():
    df = _mixed_df()
    df_before = df.copy()
    out = bn.structure_learning.fit(df, methodtype='hc', skeleton='hnet')
    skel = out['skeleton']
    # Input is not modified.
    assert df.equals(df_before)
    # Skeleton covers every column, is symmetric and has no self loops.
    assert list(skel['adjmat'].columns) == list(df.columns)
    assert (skel['adjmat'].values == skel['adjmat'].values.T).all()
    assert not np.diag(skel['adjmat'].values).any()
    # Known dependencies are in the skeleton (categorical-categorical, categorical-numeric, numeric-numeric).
    found = _undirected(skel['edges'])
    assert {frozenset(('A', 'B')), frozenset(('A', 'X1')), frozenset(('X1', 'X2'))} <= found
    # Independent variables are isolated nodes and still part of the model.
    assert not any('X3' in e or 'C' in e for e in found)
    assert set(out['model'].nodes()) == set(df.columns)
    # The search never leaves the skeleton.
    assert _undirected(out['model_edges']) <= found
    assert out['config']['skeleton'] == 'hnet'


def test_strong_association_is_kept():
    # Very small p-values (underflow to 0) must still count as an edge.
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 5000)
    b = (a ^ (rng.random(5000) < 0.2)).astype(int)
    df = pd.DataFrame(dict(A=a, B=b, Z=rng.integers(0, 2, 5000)))
    out = bn.structure_learning.fit(df, methodtype='hc', skeleton='hnet')
    assert ('A', 'B') in out['skeleton']['edges']


def test_empty_skeleton_gives_disconnected_dag():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.integers(0, 3, size=(1000, 4)), columns=list('ABCD'))
    out = bn.structure_learning.fit(df, methodtype='hc', skeleton='hnet')
    assert out['skeleton']['edges'] == []
    assert out['model_edges'] == []
    assert set(out['model'].nodes()) == set(df.columns)


def test_continuous_only():
    df = _mixed_df()[['X1', 'X2', 'X3']]
    out = bn.structure_learning.fit(df, methodtype='hc', skeleton='hnet')
    assert out['skeleton']['edges'] == [('X1', 'X2')]
    assert _undirected(out['model_edges']) == {frozenset(('X1', 'X2'))}


def test_black_and_white_list_edges():
    df = _mixed_df()
    base = bn.structure_learning.fit(df.copy(), methodtype='hc', skeleton='hnet')
    # black_list: a learned edge can not come back.
    edge = base['model_edges'][0]
    out = bn.structure_learning.fit(df.copy(), methodtype='hc', skeleton='hnet', black_list=[edge], bw_list_method='edges')
    assert edge not in out['model_edges']
    # white_list is intersected with the skeleton: an edge outside of the skeleton is never added.
    out = bn.structure_learning.fit(df.copy(), methodtype='hc', skeleton='hnet', white_list=[('C', 'X3'), ('X3', 'C')], bw_list_method='edges')
    assert out['model_edges'] == []


def test_fixed_edges_are_kept():
    df = _mixed_df()
    out = bn.structure_learning.fit(df, methodtype='hc', skeleton='hnet', fixed_edges=[('C', 'X3')])
    assert ('C', 'X3') in out['model_edges']


def test_nodes_filter():
    df = _mixed_df()
    out = bn.structure_learning.fit(df, methodtype='hc', skeleton='hnet', black_list=['X3', 'C'], bw_list_method='nodes')
    assert set(out['skeleton']['adjmat'].columns) == {'A', 'B', 'X1', 'X2'}
    assert set(out['model'].nodes()) == {'A', 'B', 'X1', 'X2'}
