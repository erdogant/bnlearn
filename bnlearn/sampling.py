"""This is a module of bnlearn for the generation of synthetic data."""
# ------------------------------------
# Name        : sampling.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------
import logging
logger = logging.getLogger("bnlearn")
from pgmpy.sampling import BayesianModelSampling, GibbsSampling
from pgmpy.factors.discrete import State
from pgmpy.inference import VariableElimination
from pgmpy.models import LinearGaussianBayesianNetwork
import numpy as np
import pandas as pd
# import logging
# logging.getLogger("pgmpy").setLevel(logging.ERROR)

_original_from_records = pd.DataFrame.from_records.__func__


# %% Patch
#   Patch pd.DataFrame.from_records itself. This is the single choke point
#   that every code path goes through, regardless of how _return_samples was
#   imported. If the data passed in is already a DataFrame, return it directly.
#   Otherwise, call the original from_records as normal.

@classmethod
def _patched_from_records(cls, data, *args, **kwargs):
    if isinstance(data, pd.DataFrame):
        return data
    return _original_from_records(cls, data, *args, **kwargs)

pd.DataFrame.from_records = _patched_from_records

# %% Sampling from model
def sampling(DAG, n=1000, methodtype='bayes', evidence=None, do=None, seed=None):
    """Generate synthetic data using the joint distribution of the network.

    Parameters
    ----------
    DAG : dict
        Contains model and the adjmat of the DAG.
    methodtype : str (default: 'bayes')
        * 'bayes': Forward sampling using Bayesian. When ``evidence`` is
          provided, rejection sampling is used to draw samples that are
          consistent with the evidence.
        * 'gibbs' : Gibbs sampling (does not support ``evidence``).
        * 'auto': choose from model type (discrete / linear-gaussian / cg).
        * 'linear-gaussian', 'lg': sample from LinearGaussianBayesianNetwork.
        * 'cg', 'conditional-gaussian': sample mixed CG models.
    n : int, optional
        Number of samples to generate. The default is 1000.
    evidence : dict, optional
        Condition the samples on the given evidence, e.g. ``{'Rain': 1,
        'Cloudy': 0}``. Keys must be variable names in the model (case
        sensitive) and values the observed state. For continuous models,
        values may be numeric. Only supported for
        ``methodtype='bayes'`` (rejection) and continuous/CG paths. The default is
        None (unconditional sampling).
    do : dict, optional
        Interventions applied before sampling (continuous / CG / when the
        underlying model supports it). The default is None.
    seed : int, optional
        Random seed for continuous / CG sampling.

    Returns
    -------
    df : pd.DataFrame().
        Dataframe containing sampled data from the input DAG model.

    Example
    -------
    >>> # Example 1
    >>>
    >>> # Import library
    >>> import bnlearn as bn
    >>> # Load DAG with model
    >>> DAG = bn.import_DAG('sprinkler')
    >>> # Sampling
    >>> df = bn.sampling(DAG, n=1000, methodtype='bayes')
    >>>
    >>> # Example 2:
    >>>
    >>> # Load example dataset
    >>> df = bn.import_example('sprinkler')
    >>> edges = [('Cloudy', 'Sprinkler'),
    >>>         ('Cloudy', 'Rain'),
    >>>         ('Sprinkler', 'Wet_Grass'),
    >>>         ('Rain', 'Wet_Grass')]
    >>>
    >>> # Make the actual Bayesian DAG
    >>> DAG = bn.make_DAG(edges, methodtype='bayes')
    >>> # Fit model
    >>> model = bn.parameter_learning.fit(DAG, df, methodtype='bayes')
    >>> # Sampling using gibbs
    >>> df = bn.sampling(model, n=100, methodtype='gibbs')
    >>>
    >>> # Example 3: Conditional sampling
    >>>
    >>> # Draw samples in which it is raining and not cloudy
    >>> df = bn.sampling(model, n=100, evidence={'Rain': 1, 'Cloudy': 0})

    """
    if n<=0: raise ValueError('Number of samples (n) must be 1 or larger!')
    if DAG is None or not isinstance(DAG, dict) or DAG.get('model') is None:
        raise ValueError('The input model (DAG) must be a bnlearn model dict with a fitted model.')

    # Resolve methodtype='auto' from model contents
    if methodtype == 'auto':
        if DAG.get('continuous_cpds'):
            methodtype = 'cg'
        elif isinstance(DAG.get('model'), LinearGaussianBayesianNetwork) or 'LinearGaussian' in type(DAG['model']).__name__:
            methodtype = 'linear-gaussian'
        else:
            methodtype = 'bayes'

    # --- Continuous: Linear Gaussian ---
    if methodtype in ('linear-gaussian', 'lg'):
        return _sample_linear_gaussian(DAG, n=n, evidence=evidence, do=do, seed=seed)

    # --- Mixed: Conditional Gaussian ---
    if methodtype in ('cg', 'conditional-gaussian'):
        return _sample_cg(DAG, n=n, evidence=evidence, do=do, seed=seed)

    # --- Discrete (original path) ---
    if 'bayesiannetwork' not in str(type(DAG['model'])).lower():
        raise ValueError('The input model (DAG) must contain BayesianNetwork.')

    if len(DAG['model'].get_cpds())==0:
        raise Exception('[bnlearn] >Error! This is a Bayesian DAG containing only edges, and no CPDs. Tip: you need to specify or learn the CPDs. Try: DAG=bn.parameter_learning.fit(DAG, df). At this point you can make a plot with: bn.plot(DAG).')

    if methodtype=='bayes':
        infer_model = BayesianModelSampling(DAG['model'])
        if evidence is None:
            logger.info('Bayesian forward sampling for %.0d samples..' %(n))
            df = infer_model.forward_sample(size=n, seed=None, show_progress=logger.isEnabledFor(logging.INFO))
        else:
            states = _evidence_as_states(evidence, DAG['model'])
            if not _evidence_is_possible(evidence, DAG['model']):
                raise ValueError('[bnlearn] >evidence %s has zero probability under the model. Rejection sampling cannot produce matching samples.' %(evidence))
            logger.info('Bayesian rejection sampling for %.0d samples conditioned on %.0d evidence variable(s)..' %(n, len(states)))
            df = infer_model.rejection_sample(evidence=states, size=n, seed=None, show_progress=logger.isEnabledFor(logging.INFO))
    elif methodtype=='gibbs':
        if evidence is not None:
            raise ValueError("[bnlearn] >Gibbs sampling does not support conditioning on evidence. Use methodtype='bayes' together with evidence=... for conditional (rejection) sampling.")
        logger.info('Gibbs sampling for %.0d samples..' %(n))
        # Gibbs sampling
        gibbs = GibbsSampling(DAG['model'])
        df = gibbs.sample(size=n, seed=None)
    else:
        raise ValueError('[bnlearn] >Sampling methodtype [%s] is unknown. Use "bayes", "gibbs", "linear-gaussian", "cg", or "auto".' %(methodtype))
    return df


def _sample_linear_gaussian(DAG, n=1000, evidence=None, do=None, seed=None):
    """Sample from a fitted LinearGaussianBayesianNetwork."""
    lg = DAG['model']
    if not isinstance(lg, LinearGaussianBayesianNetwork) and 'LinearGaussian' not in type(lg).__name__:
        raise ValueError('[bnlearn] >linear-gaussian sampling requires a LinearGaussianBayesianNetwork.')
    logger.info('Linear-Gaussian sampling for %.0d samples..' % n)
    return lg.simulate(n_samples=n, do=do or None, evidence=evidence or None, seed=seed)


def _sample_cg(DAG, n=1000, evidence=None, do=None, seed=None):
    """Sample mixed CG: discrete forward/rejection, then continuous local Gaussians."""
    from bnlearn.inference import _cg_cpd_map, _cg_mean_std

    rng = np.random.default_rng(seed)
    evidence = dict(evidence or {})
    do = dict(do or {})
    continuous_cpds = DAG.get('continuous_cpds') or []
    cpd_map = _cg_cpd_map(continuous_cpds)
    disc_model = DAG.get('model')
    cfg = DAG.get('config') or {}
    discrete_cols = list(cfg.get('discrete_cols') or [])
    continuous_cols = list(cfg.get('continuous_cols') or list(cpd_map.keys()))

    logger.info('Conditional-Gaussian sampling for %.0d samples..' % n)
    # Discrete part
    if disc_model is not None and discrete_cols:
        disc_do = {k: v for k, v in do.items() if k in discrete_cols}
        disc_ev = {k: v for k, v in evidence.items() if k in discrete_cols}
        m = disc_model
        if disc_do:
            m = m.do(list(disc_do.keys()))
        sampler = BayesianModelSampling(m)
        samp_ev = {**disc_do, **disc_ev}
        if samp_ev:
            try:
                states = _evidence_as_states(samp_ev, m)
                df_disc = sampler.rejection_sample(evidence=states, size=n, seed=seed, show_progress=logger.isEnabledFor(logging.INFO))
            except Exception:
                df_disc = sampler.forward_sample(size=n, seed=seed, show_progress=logger.isEnabledFor(logging.INFO))
                for k, v in samp_ev.items():
                    df_disc[k] = v
        else:
            df_disc = sampler.forward_sample(size=n, seed=seed, show_progress=logger.isEnabledFor(logging.INFO))
    else:
        df_disc = pd.DataFrame(index=range(n))

    df = df_disc.copy()
    pending = [c for c in continuous_cols if c in cpd_map]
    resolved = set(df.columns)
    for k, v in {**do, **evidence}.items():
        if k in continuous_cols:
            df[k] = float(v)
            resolved.add(k)
            if k in pending:
                pending.remove(k)

    safety = 0
    while pending and safety < len(cpd_map) + 5:
        safety += 1
        progress = False
        for v in list(pending):
            local = cpd_map[v]
            need = list(local['disc_parents']) + list(local['cont_parents'])
            if not all((p in resolved) or (p in df.columns) for p in need):
                continue
            means, stds = [], []
            for i in range(n):
                row_ev = {}
                for p in need:
                    if p in df.columns:
                        row_ev[p] = df.iloc[i][p]
                    elif p in evidence:
                        row_ev[p] = evidence[p]
                    elif p in do:
                        row_ev[p] = do[p]
                try:
                    mu, std = _cg_mean_std(local, row_ev)
                except Exception:
                    mu, std = 0.0, 1.0
                means.append(mu)
                stds.append(std)
            df[v] = rng.normal(loc=np.asarray(means), scale=np.maximum(np.asarray(stds), 1e-12))
            resolved.add(v)
            pending.remove(v)
            progress = True
        if not progress:
            break

    for v in pending:
        logger.warning('Warning: could not sample CG node "%s".' % v)
        df[v] = np.nan
    return df


# %% Convert an evidence dict into pgmpy State tuples
def _evidence_as_states(evidence, model):
    """Convert an evidence dict {variable: state} into a list of pgmpy State tuples.

    Each variable is checked against the model, and each requested state against
    that variable's state space. Impossible evidence (an unknown variable or a
    state that the variable can never take) therefore fails fast with a clear
    message instead of hanging the rejection sampler, which would otherwise loop
    forever waiting to accept a sample that can never occur.
    """
    if not isinstance(evidence, dict):
        raise TypeError('[bnlearn] >evidence must be a dict of {variable: state}, e.g. {"Rain": 1}.')
    nodes = set(model.nodes())
    unknown = [var for var in evidence if var not in nodes]
    if len(unknown)>0:
        raise ValueError('[bnlearn] >evidence variable(s) %s are not in the model (case sensitive!). Available nodes: %s' %(unknown, sorted(nodes)))

    states = []
    for var, state in evidence.items():
        valid_states = model.get_cpds(var).state_names[var]
        if state not in valid_states:
            raise ValueError('[bnlearn] >evidence state [%s=%s] is not a valid state for variable [%s]. Valid states: %s' %(var, state, var, valid_states))
        states.append(State(var, state))
    return states


def _evidence_is_possible(evidence, model):
    """Return whether the joint evidence has non-zero probability."""
    if len(evidence) == 0:
        return True

    infer_model = VariableElimination(model)
    observed = {}
    for var, state in evidence.items():
        distribution = infer_model.query(variables=[var], evidence=observed, show_progress=False)
        state_number = distribution.get_state_no(var, state)
        # Check each conditional factor directly. Multiplying the factors can
        # underflow and incorrectly classify very unlikely evidence as impossible.
        if distribution.values[state_number] == 0:
            return False
        observed[var] = state
    return True
