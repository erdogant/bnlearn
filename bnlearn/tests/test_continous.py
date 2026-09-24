"""Tests for Gaussian structure scores in bnlearn.

Run from the project root with:

    pytest -q tests/test_gaussian_scores.py

The tests cover the local score formulas, AIC/BIC penalties, caching,
validation, score selection, and integration with hill-climb structure learning.
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests

import bnlearn as bn
from bnlearn.structure_learning import (
    AICGauss,
    BICGauss,
    LogLikelihoodGauss,
    SetScoringType,
)


@pytest.fixture
def gaussian_data():
    """Create reproducible linear-Gaussian test data."""
    rng = np.random.default_rng(42)
    n_samples = 500

    x = rng.normal(size=n_samples)
    z = rng.normal(size=n_samples)
    y = 2.5 * x + rng.normal(scale=0.35, size=n_samples)

    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def _manual_gaussian_log_likelihood(data, variable, parents, variance_floor=1e-12):
    """Reference implementation used only by the tests."""
    y = data[variable].to_numpy(dtype=float)
    n_samples = y.shape[0]

    if len(parents) == 0:
        residuals = y - y.mean()
    else:
        x = data.loc[:, list(parents)].to_numpy(dtype=float)
        design = np.column_stack((np.ones(n_samples), x))
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        residuals = y - design @ coefficients

    rss = float(residuals @ residuals)
    variance = max(rss / n_samples, variance_floor)

    return float(
        -0.5
        * n_samples
        * (np.log(2.0 * np.pi) + 1.0 + np.log(variance))
    )


def test_loglik_gauss_root_matches_manual_calculation(gaussian_data):
    """The intercept-only local score must match the Gaussian likelihood."""
    scorer = LogLikelihoodGauss(gaussian_data)

    observed = scorer.local_score("Y", [])
    expected = _manual_gaussian_log_likelihood(gaussian_data, "Y", [])

    assert observed == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_loglik_gauss_with_parent_matches_manual_calculation(gaussian_data):
    """The regression-based local score must match a direct NumPy calculation."""
    scorer = LogLikelihoodGauss(gaussian_data)

    observed = scorer.local_score("Y", ["X"])
    expected = _manual_gaussian_log_likelihood(gaussian_data, "Y", ["X"])

    assert observed == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_aic_and_bic_penalties(gaussian_data):
    """AIC-G and BIC-G must apply the documented higher-is-better penalties."""
    parents = ["X"]
    n_parameters = len(parents) + 2
    n_samples = gaussian_data.shape[0]

    loglik = LogLikelihoodGauss(gaussian_data).local_score("Y", parents)
    aic = AICGauss(gaussian_data).local_score("Y", parents)
    bic = BICGauss(gaussian_data).local_score("Y", parents)

    assert aic == pytest.approx(loglik - n_parameters)
    assert bic == pytest.approx(
        loglik - 0.5 * n_parameters * np.log(n_samples)
    )


@pytest.mark.parametrize(
    "score_class",
    [LogLikelihoodGauss, AICGauss, BICGauss],
)
def test_true_parent_improves_local_score(score_class, gaussian_data):
    """A strong true parent should improve all three local Gaussian scores."""
    scorer = score_class(gaussian_data)

    score_without_parent = scorer.local_score("Y", [])
    score_with_parent = scorer.local_score("Y", ["X"])

    assert score_with_parent > score_without_parent


def test_parent_order_uses_same_cached_log_likelihood(gaussian_data):
    """Equivalent parent sets should share one cached local likelihood."""
    scorer = LogLikelihoodGauss(gaussian_data)

    score_1 = scorer.local_score("Y", ["X", "Z"])
    score_2 = scorer.local_score("Y", ["Z", "X"])

    assert score_1 == pytest.approx(score_2)
    assert len(scorer._gaussian_score_cache) == 1


@pytest.mark.parametrize(
    "data, error_type, message",
    [
        (np.ones((5, 2)), TypeError, "pandas DataFrame"),
        (pd.DataFrame(), ValueError, "at least one sample"),
        (
            pd.DataFrame({"X": [1.0, 2.0], "group": ["a", "b"]}),
            ValueError,
            "numeric columns",
        ),
        (
            pd.DataFrame({"X": [1.0, np.nan]}),
            ValueError,
            "missing or infinite",
        ),
        (
            pd.DataFrame({"X": [1.0, np.inf]}),
            ValueError,
            "missing or infinite",
        ),
    ],
)
def test_gaussian_score_input_validation(data, error_type, message):
    """Invalid Gaussian score inputs should fail with informative errors."""
    with pytest.raises(error_type, match=message):
        LogLikelihoodGauss(data)


def test_variance_floor_validation(gaussian_data):
    """The variance floor must be strictly positive."""
    with pytest.raises(ValueError, match="variance_floor"):
        LogLikelihoodGauss(gaussian_data, variance_floor=0)


@pytest.mark.parametrize(
    "scoretype, expected_class",
    [
        ("loglik-g", LogLikelihoodGauss),
        ("aic-g", AICGauss),
        ("bic-g", BICGauss),
    ],
)
def test_set_scoring_type_returns_gaussian_scorer(
    gaussian_data,
    scoretype,
    expected_class,
):
    """The internal score factory must expose all three Gaussian scores."""
    scorer, _ = SetScoringType(gaussian_data, scoretype)

    assert isinstance(scorer, expected_class)


@pytest.mark.parametrize("scoretype", ["loglik-g", "aic-g", "bic-g"])
def test_hillclimb_supports_gaussian_scores(gaussian_data, scoretype):
    """Each Gaussian score must work end-to-end with hill-climb learning."""
    model = bn.structure_learning.fit(
        gaussian_data,
        methodtype="hc",
        scoretype=scoretype,
        max_indegree=2,
        max_iter=1000,
    )

    assert model is not None
    assert model["config"]["scoring"] == scoretype
    assert set(model["model"].nodes()) == set(gaussian_data.columns)
    assert scoretype in model["structure_scores"]
    assert np.isfinite(model["structure_scores"][scoretype])
    

# %% Parameter learning: linear-gaussian and CG
def test_parameter_learning_linear_gaussian(gaussian_data):
    """Pure continuous data: linear-gaussian parameter learning."""
    # Structure from HC with Gaussian score
    model = bn.structure_learning.fit(
        gaussian_data, methodtype='hc', scoretype='bic-g', max_iter=500
    )
    fitted = bn.parameter_learning.fit(model, gaussian_data, methodtype='linear-gaussian')
    assert fitted is not None
    assert 'model' in fitted
    assert fitted['config']['method'] in ('linear-gaussian', 'lg')
    assert fitted['config']['data_type'] == 'continuous'
    # For pure linear-gaussian, continuous_cpds key is present but set to None
    assert fitted.get('continuous_cpds') is None
    cpds = fitted['model'].get_cpds()
    assert len(cpds) == gaussian_data.shape[1]
    for cpd in cpds:
        assert getattr(cpd, 'std', 0) > 0


def test_parameter_learning_auto_continuous(gaussian_data):
    model = bn.structure_learning.fit(
        gaussian_data, methodtype='hc', scoretype='bic-g', max_iter=300
    )
    fitted = bn.parameter_learning.fit(model, gaussian_data, methodtype='auto')
    assert fitted['config']['method'] == 'linear-gaussian'


def test_parameter_learning_cg_mixed():
    """Mixed data: CG parameter learning yields discrete model + continuous_cpds."""
    import pandas as pd
    rng = np.random.default_rng(3)
    n = 400
    fail = rng.integers(0, 2, size=n)
    torque = rng.normal(loc=0, scale=1, size=n) + fail * 2.0
    wear = rng.normal(size=n)
    df = pd.DataFrame({'fail': fail, 'torque': torque, 'wear': wear})

    model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic-cg', max_iter=400)
    fitted = bn.parameter_learning.fit(model, df, methodtype='cg')
    assert fitted is not None
    assert fitted['config']['data_type'] == 'mixed'
    assert 'continuous_cpds' in fitted
    assert fitted['continuous_cpds'] is not None
    vars_cg = {c['variable'] for c in fitted['continuous_cpds']}
    assert 'torque' in vars_cg or 'wear' in vars_cg
    # Discrete result keys used by existing tests remain present
    for key in ('model', 'adjmat', 'config', 'model_edges', 'structure_scores', 'independence_test'):
        assert key in fitted


def test_parameter_learning_discrete_keys_stable(gaussian_data):
    """Discrete bayes path must not populate continuous_cpds (package test compatibility)."""
    df = bn.import_example()
    model = bn.import_DAG('sprinkler', CPD=False)
    fitted = bn.parameter_learning.fit(model, df, methodtype='bayes')
    # The key is present but must be None for the discrete path
    assert fitted.get('continuous_cpds') is None
    assert set(fitted.keys()) >= {
        'model', 'adjmat', 'config', 'model_edges', 'structure_scores', 'independence_test'
    }
