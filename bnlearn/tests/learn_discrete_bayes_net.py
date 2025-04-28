from itertools import zip_longest
from pathlib import Path
import datazets as ds

import pytest
import numpy as np
import pandas as pd

from bnlearn.learn_discrete_bayes_net import (
    bn_discretizer_free_number_rep,
    bn_discretizer_iteration_converge,
    bn_discretizer_p_data_model,
    continuous_to_discrete,
    discretize_all,
    equal_width_disc,
    graph_to_markov,
    graph_to_reverse_conti_order,
    graph_to_reverse_order,
    largest_class_value,
    log_prob_single_edge_last_term,
    log_prob_spouse_child_data,
    one_iteration,
    prior_of_intval,
    sort_disc_by_vorder,
    sortperm,
)

# TODO replace tuples and scalars with lists to be more consistent everywhere
graph = [1, (1, 2), (2, 4), (4, 0), (0, 4, 2, 6), (2, 3), (3, 5), 7]
discrete_index = [1, 6, 7]
continuous_index = [0, 2, 3, 4, 5]
sort_continuous = [5, 3, 0, 4, 2]

# TODO replace tuples and scalars with lists to be more consistent everywhere
markov = {
    0: ([4], [(6, 4, 2)]),
    1: ([], [2]),
    2: ([1], [4, (6, 0, 4), 3]),
    3: ([2], [5]),
    4: ([2], [0, (6, 0, 2)]),
    5: ([3], []),
    6: ([0, 4, 2], []),
    7: ([], []),
}

test_data = Path("./bnlearn/tests/discretize/data")


@pytest.fixture()
def data():
    return pd.read_csv(
        test_data / "auto_mpg.csv",
        dtype={
            "mpg": "float64",
            "cylinders": "int64",
            "displacement": "float64",
            "horsepower": "float64",
            "weight": "float64",
            "acceleration": "float64",
            "model_year": "int64",
            "origin": "int64",
        },
    )


@pytest.fixture()
def data_equal_width():
    return pd.read_csv(test_data / "equal_width.csv", dtype="int64")


@pytest.fixture()
def data_one_iteration():
    return pd.read_csv(test_data / "one_iteration.csv", dtype="int64")


@pytest.fixture()
def data_prior_of_intval():
    return pd.read_csv(test_data / "prior_of_intval.csv")


@pytest.fixture()
def continuous_sortperm():
    return pd.read_csv(test_data / "continuous_sortperm.csv", dtype="int64")


def test_fixtures(data, data_equal_width):
    assert data.shape == (392, 8)
    assert list(data.dtypes[discrete_index].unique()) == [np.dtype("int64")]
    assert list(data.dtypes[continuous_index].unique()) == [np.dtype("float64")]

    assert data_equal_width.shape == (392, 8)
    assert list(data_equal_width.dtypes.unique()) == [np.dtype("int64")]


@pytest.mark.parametrize(
    "target,parent,child_spouse", [(t, p, c) for (t, (p, c)) in markov.items()]
)
def test_graph_to_markov(target, parent, child_spouse):
    parent_set, child_spouse_set = graph_to_markov(graph, target)
    assert parent_set == parent
    assert child_spouse_set == child_spouse


def test_graph_to_reverse_order():
    assert graph_to_reverse_order(graph) == [7, 5, 3, 6, 0, 4, 2, 1]


def test_graph_to_reverse_conti_order():
    actual = graph_to_reverse_conti_order(graph, continuous_index)
    assert actual == sort_continuous


def test_log_prob_single_edge_last_term():
    expected = np.genfromtxt(
        test_data / "log_prob_single_edge_last_term.csv", delimiter=","
    )
    inv_p = log_prob_single_edge_last_term(pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1]))
    np.testing.assert_allclose(inv_p, expected)


def test_continuous_to_discrete(data, data_one_iteration):
    bin_edge = [9.0, 12.5, 17.55, 20.9, 28.9, 46.6]
    expected = data_one_iteration.iloc[:, 0]
    actual = continuous_to_discrete(data.iloc[:, 0], bin_edge)
    pd.testing.assert_series_equal(actual, expected)


def test_sort_disc_by_vorder():
    continuous_order = [6, 4, 1, 5, 3]
    disc_edge = [
        np.array([8.0, 12.35]),
        np.array([46.0]),
        np.array([9.0, 15.25]),
        np.array([1613.0]),
        np.array([68.0]),
    ]

    expected = [
        np.array([9.0, 15.25]),
        np.array([68.0]),
        np.array([46.0]),
        np.array([1613.0]),
        np.array([8.0, 12.35]),
    ]
    actual = sort_disc_by_vorder(continuous_order, disc_edge)

    for a, b in zip_longest(actual, expected):
        np.testing.assert_allclose(a, b)


def test_log_prob_spouse_child_data_one():
    data = pd.read_csv(test_data / "log_prob_spouse_child_data_one.csv", header=None)
    child_data = data.iloc[:, [0]]
    spouse_data = data.iloc[:, [1]]
    expected = data.iloc[:, 2:]
    actual = log_prob_spouse_child_data(child_data, spouse_data)
    np.testing.assert_allclose(actual[:, : expected.shape[1]], expected)


def test_log_prob_spouse_child_data_three():
    data = pd.read_csv(test_data / "log_prob_spouse_child_data_three.csv", header=None)
    child_data = data.iloc[:, [0]]
    spouse_data = data.iloc[:, 1:3]
    expected = data.iloc[:, 3:]
    actual = log_prob_spouse_child_data(child_data, spouse_data)
    np.testing.assert_allclose(actual[:, : expected.shape[1]], expected)


@pytest.mark.parametrize("target", continuous_index)
def test_equal_width_disc(target, data, data_equal_width):
    actual = equal_width_disc(data.iloc[:, target], 13)
    assert list(actual.values) == list(data_equal_width.iloc[:, target])
    assert actual.dtype == np.dtype("int64")


@pytest.mark.parametrize("i", range(len(continuous_index)))
def test_prior_of_intval(i: int, continuous_sortperm, data, data_prior_of_intval):
    target = continuous_index[i]
    increase_order = continuous_sortperm.iloc[:, i]
    conti = data.iloc[increase_order, target]

    actual = prior_of_intval(conti, 13)
    expected = data_prior_of_intval.iloc[:, i]
    np.testing.assert_allclose(actual, expected)


def test_largest_class_value(data):
    assert largest_class_value(data.iloc[:, discrete_index]) == 13


def test_bn_discretizer_p_data_model(data, data_equal_width, continuous_sortperm):
    i = 4
    target = continuous_index[i]
    increase_order = continuous_sortperm.iloc[:, i]

    data_integer_sort = data_equal_width.iloc[increase_order, :]
    parent_set, child_spouse_set = markov[target]

    expected = pd.read_csv(test_data / "bn_discretizer_p_data_model.csv", header=None)
    actual = bn_discretizer_p_data_model(
        data_integer_sort, parent_set, child_spouse_set, False
    )
    np.testing.assert_allclose(actual[0:10, :], expected.to_numpy())


def test_bn_discretizer_free_number_rep(data, data_equal_width, continuous_sortperm):
    i = 4
    target = continuous_index[i]
    increase_order = continuous_sortperm.iloc[:, i]

    conti = data.iloc[increase_order, target]
    data_integer_sort = data_equal_width.iloc[increase_order, :]
    parent_set, child_spouse_set = markov[target]

    actual = bn_discretizer_free_number_rep(
        conti, data_integer_sort, parent_set, child_spouse_set
    )
    np.testing.assert_allclose(actual, [8.0, 10.25, 12.35, 13.75, 16.05, 24.8])


@pytest.mark.parametrize("i,column", enumerate(continuous_index))
def test_sortperm(i, column, data, continuous_sortperm):
    pd.testing.assert_series_equal(
        sortperm(data.iloc[:, column]), continuous_sortperm.iloc[:, i]
    )


def test_one_iteration(data, data_equal_width, data_one_iteration):
    actual = one_iteration(
        data, data_equal_width, graph, discrete_index, sort_continuous, 13
    )

    disc_edge_collect = [
        np.array([8.0, 10.25, 12.35, 13.75, 16.05, 24.8]),
        np.array([46.0, 71.5, 99.0, 127.0, 151.0, 191.5, 195.5, 230.0]),
        np.array([9.0, 12.5, 17.55, 20.9, 28.9, 46.6]),
        np.array([1613.0, 2092.5, 2513.0, 2959.5, 3657.5, 4826.0, 5140.0]),
        np.array([68.0, 70.5, 93.5, 109.0, 159.5, 284.5, 414.5, 455.0]),
    ]

    for a, b in zip_longest(actual[1], disc_edge_collect):
        np.testing.assert_allclose(a, b)

    np.testing.assert_allclose(actual[0], data_one_iteration)


def test_bn_discretizer_iteration_converge_equal_width(data, data_equal_width):
    actual, _ = bn_discretizer_iteration_converge(
        data, graph, discrete_index, sort_continuous, 0
    )
    pd.testing.assert_frame_equal(actual, data_equal_width)


def test_bn_discretizer_iteration_converge_one(data):
    expected = pd.read_csv(
        test_data / "bn_discretizer_iteration_converge_cut_time_1.csv"
    )

    actual, _ = bn_discretizer_iteration_converge(
        data, graph, discrete_index, sort_continuous, 1
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_bn_discretizer_iteration_converge_eight(data):
    expected = pd.read_csv(
        test_data / "bn_discretizer_iteration_converge_cut_time_8.csv"
    )

    actual, _ = bn_discretizer_iteration_converge(
        data, graph, discrete_index, sort_continuous, 8
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_discretize_all(data):
    _, disc_edges = discretize_all(data, graph, continuous_index, 8)
    expected = [
        [9.0, 15.25, 17.65, 20.9, 25.65, 28.9, 46.6],
        [68.0, 70.5, 93.5, 109.0, 159.5, 259.0, 284.5, 455.0],
        [46.0, 71.5, 99.0, 127.0, 230.0],
        [1613.0, 2115.0, 2480.5, 2959.5, 3657.5, 5140.0],
        [8.0, 12.35, 13.75, 16.05, 22.85, 24.8],
    ]

    for a, b in zip_longest(disc_edges, expected):
        np.testing.assert_allclose(a, b)
