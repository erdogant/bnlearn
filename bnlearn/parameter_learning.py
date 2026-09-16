"""Parameter learning.

Overview
----------
Parameter learning is the task to estimate the values of the conditional probability distributions (CPDs).
To make sense of the given data, we can start by counting how often each state of the variable occurs.
If the variable is dependent on the parents, the counts are done conditionally on the parents states,
i.e. for seperately for each parent configuration

Supported model families:
    * Discrete nodes: Maximum Likelihood Estimation, Bayesian Estimation
    * Continuous nodes (all continuous): Linear Gaussian Bayesian Network (OLS)
    * Mixed / Conditional Gaussian: discrete CPTs + configuration-specific linear Gaussians
    * Dynamic Bayesian Networks (DBN)
"""
# ------------------------------------
# Name        : parameter_learning.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------


# %% Libraries
from pgmpy.parameter_estimator import DiscreteBayesianEstimator, LinearGaussianMLE
from pgmpy.models import LinearGaussianBayesianNetwork, DiscreteBayesianNetwork
# from pgmpy.factors.continuous import LinearGaussianCPD
import bnlearn
from bnlearn.utils import infer_data_type, edges_and_nodes_from_adjmat
import copy
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# %% Parameter learning
def fit(model, df, methodtype='bayes', scoretype='bdeu', smooth=None, n_jobs=-1, verbose=3):
    """Learn the parameters given the DAG and data.

    Fit overview
    -------------
    Maximum Likelihood Estimation
        A natural estimate for the CPDs is to simply use the *relative frequencies*
        with which the variable states have occurred. For example, if we observed 50
        occurrences of state 'cloudy' among 100 samples, we might estimate a 50%
        probability for that state. According to MLE, we should fill the CPDs to
        maximize P(data | model), which is achieved by using relative frequencies.

        While very straightforward, the ML estimator has the problem of *overfitting* to the data.
        If the observed data is not representative of the underlying distribution, ML estimations can be far off.
        When estimating parameters for Bayesian networks, lack of data is a frequent problem.
        Even if the total sample size is very large, the fact that state counts are done conditionally
        for each parent configuration causes fragmentation.
        If a variable has 3 parents that can each take 10 states, then state counts will
        be done separately for `10^3 = 1000` parent configurations.
        This makes MLE fragile and unstable for learning Bayesian Network parameters.
        A way to mitigate MLE's overfitting is *Bayesian Parameter Estimation*.

    Bayesian Parameter Estimation
        The Bayesian Parameter Estimator starts with already existing prior CPDs,
        that express our beliefs about the variables *before* the data was observed.
        Those "priors" are then updated, using the state counts from the observed data.

        One can think of the priors as consisting in *pseudo state counts*, that are added
        to the actual counts before normalization. Unless one wants to encode specific beliefs
        about the distributions of the variables, one commonly chooses uniform priors,
        i.e. ones that deem all states equiprobable.

        A very simple prior is the so-called *K2* prior, which simply adds `1` to the count of every single state.
        A somewhat more sensible choice of prior is *BDeu* (Bayesian Dirichlet equivalent uniform prior).
        For BDeu we need to specify an *equivalent sample size* `N` and then the pseudo-counts are
        the equivalent of having observed `N` uniform samples of each variable (and each parent configuration).

    Linear Gaussian (continuous)
        When all variables are continuous, parameters are estimated by ordinary least squares
        under a Linear Gaussian Bayesian Network (each node is a linear regression on its parents
        plus Gaussian noise).

    Conditional Gaussian (mixed)
        Discrete nodes keep tabular CPTs. Continuous nodes are linear Gaussians whose
        coefficients and residual variance may depend on the configuration of discrete parents.

    Parameters
    ----------
    model : dict
        Contains a model object with a key 'adjmat' (adjacency matrix).
    df : pd.DataFrame()
        Pandas DataFrame containing the data.
    methodtype : str, (default: 'bayes')
        Strategy for parameter learning.
            * 'ml', 'maximumlikelihood': Learning CPDs using Maximum Likelihood Estimators (discrete).
            * 'bayes': Bayesian Parameter Estimation (discrete).
            * 'DBN': DynamicBayesianNetwork
            * 'linear-gaussian', 'lg': Linear Gaussian BN (all continuous).
            * 'cg', 'conditional-gaussian': Conditional Gaussian for mixed data.
            * 'auto': choose discrete / linear-gaussian / cg from data types.
    scoretype : str, (default : 'bdeu')
        Scoring / prior for discrete Bayesian estimation.
            * 'bdeu'
            * 'dirichlet'
            * 'k2'
    smooth : float (default: None)
        The smoothing value (α) for Bayesian parameter estimation. Should be Nonnegative.
    n_jobs : int, (default: -1)
        Parallel jobs where supported.
    verbose : int, (default: 3)
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

    Returns
    -------
    dict with model.
        For linear-gaussian: model is a LinearGaussianBayesianNetwork.
        For cg: model is the discrete DiscreteBayesianNetwork (or None); continuous_cpds holds CG locals.
        Also includes data_type, discrete_cols, continuous_cols.

    Examples
    --------
    >>> import bnlearn as bn
    >>>
    >>> df = bn.import_example()
    >>> model = bn.import_DAG('sprinkler', CPD=False)
    >>>
    >>> # Parameter learning
    >>> model_update = bn.parameter_learning.fit(model, df)
    >>> bn.plot(model_update)
    >>>
    >>> # Continuous data
    >>> # model_lg = bn.parameter_learning.fit(model, df_cont, methodtype='linear-gaussian')
    >>>
    >>> # Mixed data
    >>> # model_cg = bn.parameter_learning.fit(model, df_mixed, methodtype='cg')

    """
    config = {}
    config['verbose'] = verbose
    config['method'] = methodtype
    config['n_jobs'] = n_jobs
    adjmat = model['adjmat']
    independence_test = model.get('independence_test', None)
    structure_config = model.get('config', {}) if isinstance(model, dict) else {}

    if (scoretype == 'dirichlet') and (smooth is None):
        raise Exception('[bnlearn] >dirichlet requires "smooth" to be not None')

    # Automatically set methodtype for DBN
    if structure_config.get('method') == 'DBN' or model.get('methodtype', {}) == 'DBN':
        config['method'] = 'DBN'
        if verbose >= 3:
            print('[bnlearn] >Methodtype is set to DynamicBayesianNetwork (DBN)')

    # Filter dataframe to adjacency variables (except DBN)
    if config['method'] == 'DBN':
        df = adjmat
    else:
        df = bnlearn._filter_df(adjmat, copy.deepcopy(df), verbose=config['verbose'])

    # Detect data types for auto / routing
    var_types = infer_data_type(df) if config['method'] != 'DBN' else {'dtype': 'discrete', 'discrete': [], 'continuous': []}
    config['data_type'] = var_types['dtype']
    config['discrete_cols'] = var_types['discrete']
    config['continuous_cols'] = var_types['continuous']

    # Resolve methodtype='auto'
    if config['method'] == 'auto':
        if config['data_type'] == 'continuous':
            config['method'] = 'linear-gaussian'
        elif config['data_type'] == 'mixed':
            config['method'] = 'cg'
        else:
            config['method'] = 'bayes'
        if verbose >= 3:
            print('[bnlearn] >methodtype="auto" -> [%s] for %s data' % (config['method'], config['data_type']))

    if config['verbose'] >= 3:
        print('[bnlearn] >Parameter learning> Computing parameters using [%s]' % (config['method']))

    # Extract underlying graph object when still a bnlearn dict
    model_obj = model['model'] if isinstance(model, dict) else model

    continuous_cpds = None
    out_model = None

    # --- Discrete MLE ---
    if config['method'] in ('ml', 'maximumlikelihood'):
        if 'BayesianNetwork' not in str(type(model_obj)):
            if config['verbose'] >= 3:
                print('[bnlearn] >Converting [%s] to BayesianNetwork model.' % (str(type(model_obj))))
            model_obj = bnlearn.to_bayesiannetwork(adjmat, verbose=config['verbose'])
        model_obj.fit(df, estimator=None)
        for cpd in model_obj.get_cpds():
            if config['verbose'] >= 2:
                print("[bnlearn] >CPD of {variable}:".format(variable=cpd.variable))
                print(cpd)
        out_model = model_obj

    # --- Discrete Bayesian ---
    elif config['method'] == 'bayes':
        if 'BayesianNetwork' not in str(type(model_obj)):
            if config['verbose'] >= 3:
                print('[bnlearn] >Converting [%s] to BayesianNetwork model.' % (str(type(model_obj))))
            model_obj = bnlearn.to_bayesiannetwork(adjmat, verbose=config['verbose'])
        estimator = DiscreteBayesianEstimator(
            prior_type=scoretype,
            equivalent_sample_size=1000,
            pseudo_counts=smooth,
            n_jobs=config['n_jobs'],
        )
        model_obj.fit(df, estimator=estimator)
        for cpd in model_obj.get_cpds():
            if config['verbose'] >= 2:
                print("[bnlearn] >CPD of {variable}:".format(variable=cpd.variable))
                print(cpd)
        out_model = model_obj

    # --- Dynamic BN ---
    elif config['method'] == 'DBN':
        model_obj.fit(df, estimator='MLE')
        for cpd in model_obj.get_cpds():
            if config['verbose'] >= 2:
                print("[bnlearn] >CPD of {variable}:".format(variable=cpd.variable))
                print(cpd)
        out_model = model_obj

    # --- Pure continuous: Linear Gaussian ---
    elif config['method'] in ('linear-gaussian', 'lg'):
        if config['data_type'] != 'continuous' and verbose >= 2:
            print('[bnlearn] >Warning: linear-gaussian expects all-continuous data; detected %s.' % config['data_type'])
        out_model = _fit_linear_gaussian(adjmat, df, verbose=verbose)
        for cpd in out_model.get_cpds():
            if config['verbose'] >= 2:
                print("[bnlearn] >CPD of {variable}:".format(variable=cpd.variable))
                print(cpd)

    # --- Mixed: Conditional Gaussian ---
    elif config['method'] in ('cg', 'conditional-gaussian'):
        if config['data_type'] == 'continuous':
            if verbose >= 2:
                print('[bnlearn] >Data are fully continuous; using linear-gaussian instead of cg.')
            config['method'] = 'linear-gaussian'
            out_model = _fit_linear_gaussian(adjmat, df, verbose=verbose)
        elif config['data_type'] == 'discrete':
            if verbose >= 2:
                print('[bnlearn] >Data are fully discrete; using bayes instead of cg.')
            config['method'] = 'bayes'
            if 'BayesianNetwork' not in str(type(model_obj)):
                model_obj = bnlearn.to_bayesiannetwork(adjmat, verbose=config['verbose'])
            estimator = DiscreteBayesianEstimator(
                prior_type=scoretype,
                equivalent_sample_size=1000,
                pseudo_counts=smooth,
                n_jobs=config['n_jobs'],
            )
            model_obj.fit(df, estimator=estimator)
            out_model = model_obj
        else:
            disc_method = 'bayes'
            out_model, continuous_cpds = _fit_conditional_gaussian(
                adjmat, df,
                discrete_cols=config['discrete_cols'],
                continuous_cols=config['continuous_cols'],
                method_discrete=disc_method,
                scoretype=scoretype,
                smooth=smooth,
                n_jobs=config['n_jobs'],
                verbose=verbose,
            )

    else:
        if config['verbose'] >= 2:
            print("[bnlearn] >Warning: methodtype [%s] is unknown. Returning None." % (config['method']))
        return None

    out = {}
    out['model'] = out_model
    out['adjmat'] = adjmat
    out['config'] = config
    out['independence_test'] = independence_test
    out['continuous_cpds'] = continuous_cpds  # CG local params when method is cg

    if out_model is not None and hasattr(out_model, 'edges'):
        out['model_edges'] = list(out_model.edges())
    else:
        edges, _ = edges_and_nodes_from_adjmat(adjmat)
        out['model_edges'] = edges

    # structure_scores expects a discrete-style model for some scorers; skip on pure LG/CG failures
    try:
        out['structure_scores'] = bnlearn.structure_scores(out, df, verbose=verbose)
    except Exception:
        out['structure_scores'] = None
        if verbose >= 4:
            print('[bnlearn] >structure_scores not available for this model type.')

    return out


# %% Linear-Gaussian parameter learning (pure continuous)
def _fit_linear_gaussian(adjmat, df, verbose=3):
    """Fit a LinearGaussianBayesianNetwork on continuous data."""
    edges, nodes = edges_and_nodes_from_adjmat(adjmat)
    if verbose >= 3:
        print('[bnlearn] >Fitting LinearGaussianBayesianNetwork (%d nodes, %d edges)' % (len(nodes), len(edges)))

    model = LinearGaussianBayesianNetwork(edges)
    model.add_nodes_from(nodes)
    # Ensure numeric continuous data
    df_num = df[nodes].apply(pd.to_numeric, errors='coerce')
    if not np.all(np.isfinite(df_num.to_numpy(dtype=float, copy=False))):
        raise ValueError('[bnlearn] >Linear-Gaussian parameter learning requires finite numeric values.')

    model.fit(df_num, estimator=LinearGaussianMLE())
    _validate_linear_gaussian(model, verbose=verbose)
    return model


def _validate_linear_gaussian(model, verbose=3):
    """Check LinearGaussianBayesianNetwork parameter consistency."""
    try:
        ok = model.check_model()
        if verbose >= 3 and ok:
            print('[bnlearn] >Linear-Gaussian model check: OK')
    except Exception as err:
        raise ValueError('[bnlearn] >Linear-Gaussian model validation failed: %s' % err)

    for cpd in model.get_cpds():
        std = getattr(cpd, 'std', None)
        if std is not None and (not np.isfinite(std) or std <= 0):
            raise ValueError('[bnlearn] >Invalid residual std for node %s: %s' % (cpd.variable, std))
        beta = getattr(cpd, 'beta', None)
        evidence = list(getattr(cpd, 'evidence', []) or [])
        if beta is not None and len(beta) != len(evidence) + 1:
            raise ValueError(
                '[bnlearn] >beta length mismatch for node %s: expected %d (intercept + parents), got %d'
                % (cpd.variable, len(evidence) + 1, len(beta))
            )
    return True


# %% Conditional-Gaussian style local fits (mixed data)
def _fit_cg_continuous_node(variable, parents, df, discrete_cols, continuous_cols, variance_floor=1e-12):
    """Fit CG local parameters for one continuous node.

    Continuous parents enter as linear regressors.
    Discrete parents define configurations; one regression (and residual std) per configuration.
    """
    parents = list(parents)
    disc_parents = [p for p in parents if p in discrete_cols]
    cont_parents = [p for p in parents if p in continuous_cols]

    y_all = df[variable].to_numpy(dtype=float, copy=False)
    n = len(y_all)

    if len(disc_parents) == 0:
        # Single linear Gaussian (same as pure continuous local CPD)
        if len(cont_parents) == 0:
            mu = float(np.mean(y_all))
            rss = float(np.sum((y_all - mu) ** 2))
            std = max(np.sqrt(rss / max(n, 1)), variance_floor)
            return {
                'variable': variable,
                'parents': parents,
                'disc_parents': [],
                'cont_parents': [],
                'configs': {(): {'beta': [mu], 'std': float(std), 'n': int(n)}},
            }

        X = df[cont_parents].to_numpy(dtype=float, copy=False)
        design = np.column_stack((np.ones(n, dtype=float), X))
        coef, _, _, _ = np.linalg.lstsq(design, y_all, rcond=None)
        resid = y_all - design.dot(coef)
        rss = float(np.dot(resid, resid))
        std = max(np.sqrt(rss / max(n, 1)), variance_floor)
        return {
            'variable': variable,
            'parents': parents,
            'disc_parents': [],
            'cont_parents': cont_parents,
            'configs': {(): {'beta': [float(c) for c in coef], 'std': float(std), 'n': int(n)}},
        }

    # One regression per discrete-parent configuration
    configs = {}
    grouped = df.groupby(disc_parents, dropna=False)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        y = group[variable].to_numpy(dtype=float, copy=False)
        n_g = len(y)
        if n_g == 0:
            continue
        if len(cont_parents) == 0:
            mu = float(np.mean(y))
            rss = float(np.sum((y - mu) ** 2))
            std = max(np.sqrt(rss / max(n_g, 1)), variance_floor)
            configs[key] = {'beta': [mu], 'std': float(std), 'n': int(n_g)}
        else:
            X = group[cont_parents].to_numpy(dtype=float, copy=False)
            design = np.column_stack((np.ones(n_g, dtype=float), X))
            coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            resid = y - design.dot(coef)
            rss = float(np.dot(resid, resid))
            std = max(np.sqrt(rss / max(n_g, 1)), variance_floor)
            configs[key] = {'beta': [float(c) for c in coef], 'std': float(std), 'n': int(n_g)}

    return {
        'variable': variable,
        'parents': parents,
        'disc_parents': disc_parents,
        'cont_parents': cont_parents,
        'configs': configs,
    }


def _fit_conditional_gaussian(adjmat, df, discrete_cols, continuous_cols, method_discrete='bayes', scoretype='bdeu', smooth=None, n_jobs=-1, verbose=3):
    """Fit mixed / Conditional Gaussian parameters.

    Discrete nodes: standard TabularCPDs on the discrete subgraph (parents must be discrete).
    Continuous nodes: configuration-specific linear Gaussians (CG local distributions).
    """
    edges, nodes = edges_and_nodes_from_adjmat(adjmat)
    parent_map = {n: [] for n in nodes}
    for u, v in edges:
        if v in parent_map:
            parent_map[v].append(u)

    # Warn on discrete nodes with continuous parents (not standard CG)
    for node in discrete_cols:
        cont_pars = [p for p in parent_map.get(node, []) if p in continuous_cols]
        if cont_pars and verbose >= 2:
            print('[bnlearn] >Warning: discrete node "%s" has continuous parents %s; '
                  'standard CG assumes discrete nodes have only discrete parents. '
                  'Those edges are ignored for the discrete CPT.' % (node, cont_pars))

    # Discrete sub-model
    disc_edges = [(u, v) for u, v in edges if u in discrete_cols and v in discrete_cols]
    discrete_model = None
    if len(discrete_cols) > 0:
        discrete_model = DiscreteBayesianNetwork(disc_edges)
        discrete_model.add_nodes_from(discrete_cols)
        df_disc = df[discrete_cols].copy()
        if method_discrete in ('ml', 'maximumlikelihood'):
            discrete_model.fit(df_disc, estimator=None)
        else:
            estimator = DiscreteBayesianEstimator(
                prior_type=scoretype,
                equivalent_sample_size=1000,
                pseudo_counts=smooth,
                n_jobs=n_jobs,
            )
            discrete_model.fit(df_disc, estimator=estimator)
        if verbose >= 3:
            print('[bnlearn] >Fitted discrete CPTs for %d nodes' % len(discrete_cols))
        for cpd in discrete_model.get_cpds():
            if verbose >= 2:
                print("[bnlearn] >CPD of {variable}:".format(variable=cpd.variable))
                print(cpd)

    # Continuous / CG local parameters
    continuous_cpds = []
    for node in continuous_cols:
        parents = parent_map.get(node, [])
        local = _fit_cg_continuous_node(node, parents, df, discrete_cols, continuous_cols)
        continuous_cpds.append(local)
        if verbose >= 2:
            n_cfg = len(local['configs'])
            print('[bnlearn] >CG CPD of %s: %d configuration(s), cont_parents=%s, disc_parents=%s'
                  % (node, n_cfg, local['cont_parents'], local['disc_parents']))

    _validate_cg_parameters(continuous_cpds, verbose=verbose)

    return discrete_model, continuous_cpds


def _validate_cg_parameters(continuous_cpds, verbose=3):
    """Validate CG local continuous parameters."""
    for local in continuous_cpds:
        var = local['variable']
        n_beta_expected = len(local['cont_parents']) + 1  # intercept + continuous parents
        for key, cfg in local['configs'].items():
            beta = cfg.get('beta', [])
            std = cfg.get('std', None)
            if std is None or not np.isfinite(std) or std <= 0:
                raise ValueError('[bnlearn] >Invalid CG std for %s config %s: %s' % (var, key, std))
            if len(beta) != n_beta_expected:
                raise ValueError(
                    '[bnlearn] >CG beta length mismatch for %s config %s: expected %d, got %d'
                    % (var, key, n_beta_expected, len(beta))
                )
    if verbose >= 3:
        print('[bnlearn] >Conditional-Gaussian parameter check: OK (%d continuous nodes)' % len(continuous_cpds))
    return True

