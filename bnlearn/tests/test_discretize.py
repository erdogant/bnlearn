import pytest
import pandas as pd
from bnlearn.discretize import _bayes_net_graph, discretize_value
import bnlearn as bn

def test_discritize():
    # Load data set
    df = bn.import_example(data='auto_mpg')
    # Define the edges
    edges = [
        ("cylinders", "displacement"),
        ("displacement", "model_year"),
        ("displacement", "weight"),
        ("displacement", "horsepower"),
        ("weight", "model_year"),
        ("weight", "mpg"),
        ("horsepower", "acceleration"),
        ("mpg", "model_year"),
    ]
    # Create DAG based on edges
    DAG = bn.make_DAG(edges)
    # Plot the DAG
    bn.plot(DAG)
    # Plot the DAG using graphviz
    bn.plot_graphviz(DAG)
    # A good habbit is to set the columns with continuous data as float
    continuous_columns = ["mpg", "displacement", "horsepower", "weight", "acceleration"]
    # Discretize the continous columns by specifying
    df_discrete = bn.discretize(df, edges, continuous_columns, max_iterations=1)
    # Check size
    assert df.shape== (392, 8)

def test_bayes_net_graph():
    nodes = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
    ]
    edges = [
        ("cylinders", "displacement"),
        ("displacement", "model_year"),
        ("displacement", "weight"),
        ("displacement", "horsepower"),
        ("weight", "model_year"),
        ("weight", "mpg"),
        ("horsepower", "acceleration"),
        ("mpg", "model_year"),
    ]

    graph = [(4, 0), 1, (1, 2), (2, 3), (2, 4), (3, 5), (0, 4, 2, 6), 7]
    assert _bayes_net_graph(nodes, edges) == graph


def test_bayes_net_graph_missing_source():
    nodes = ["a", "b"]
    edges = [("a", "b"), ("c", "a")]

    with pytest.raises(ValueError):
        _bayes_net_graph(nodes, edges)


def test_bayes_net_graph_missing_target():
    nodes = ["a", "b", "c"]
    edges = [("a", "b"), ("c", "d")]

    with pytest.raises(ValueError):
        _bayes_net_graph(nodes, edges)


cat = pd.CategoricalDtype(
    categories=pd.IntervalIndex.from_breaks([1613.0, 2217.0, 2959.5, 3657.5]),
    ordered=True,
)


def test_discretize_value():
    assert discretize_value(cat, 2000) == pd.Interval(1613.0, 2217.0)


def test_discretize_value_series():
    series = pd.Series([], dtype=cat)
    assert discretize_value(series, 2000) == pd.Interval(1613.0, 2217.0)


def test_discretize_value_edge():
    assert discretize_value(cat, 2959.5) == pd.Interval(2217.0, 2959.5)


def test_discretize_value_not_found():
    with pytest.raises(KeyError):
        discretize_value(cat, 42)


def test_discretize_value_not_category():
    with pytest.raises(AttributeError):
        discretize_value(pd.Series([], dtype=int), 42)


def test_discretize_value_not_interval_index():
    with pytest.raises(AttributeError):
        discretize_value(
            pd.CategoricalDtype(categories=["0", "1", "2"], ordered=True), 2000
        )


def test_discretize_value_no_categories():
    with pytest.raises(AttributeError):
        discretize_value(pd.CategoricalDtype(), 2000)
