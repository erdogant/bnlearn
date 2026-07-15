"""This is a module of bnlearn for the generation of synthetic data."""
# ------------------------------------
# Name        : sampling.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------

from pgmpy.sampling import BayesianModelSampling, GibbsSampling
from pgmpy.factors.discrete import State
from pgmpy.inference import VariableElimination
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
def sampling(DAG, n=1000, methodtype='bayes', evidence=None, verbose=0):
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
    n : int, optional
        Number of samples to generate. The default is 1000.
    evidence : dict, optional
        Condition the samples on the given evidence, e.g. ``{'Rain': 1,
        'Cloudy': 0}``. Keys must be variable names in the model (case
        sensitive) and values the observed state. Only supported for
        ``methodtype='bayes'``, where it switches to rejection sampling so
        every returned sample is consistent with the evidence. The default is
        None (unconditional sampling).
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

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
    >>> DAG = bn.make_DAG(edges, verbose=3, methodtype='bayes')
    >>> # Fit model
    >>> model = bn.parameter_learning.fit(DAG, df, verbose=3, methodtype='bayes')
    >>> # Sampling using gibbs
    >>> df = bn.sampling(model, n=100, methodtype='gibbs', verbose=0)
    >>>
    >>> # Example 3: Conditional sampling
    >>>
    >>> # Draw samples in which it is raining and not cloudy
    >>> df = bn.sampling(model, n=100, evidence={'Rain': 1, 'Cloudy': 0})

    """
    if n<=0: raise ValueError('Number of samples (n) must be 1 or larger!')
    if (DAG is None) or ('bayesiannetwork' not in str(type(DAG['model'])).lower()):
        raise ValueError('The input model (DAG) must contain BayesianNetwork.')

    if len(DAG['model'].get_cpds())==0:
        raise Exception('[bnlearn] >Error! This is a Bayesian DAG containing only edges, and no CPDs. Tip: you need to specify or learn the CPDs. Try: DAG=bn.parameter_learning.fit(DAG, df). At this point you can make a plot with: bn.plot(DAG).')

    if methodtype=='bayes':
        infer_model = BayesianModelSampling(DAG['model'])
        if evidence is None:
            if verbose>=3: print('[bnlearn] >Bayesian forward sampling for %.0d samples..' %(n))
            df = infer_model.forward_sample(size=n, seed=None, show_progress=(verbose>=3))
        else:
            states = _evidence_as_states(evidence, DAG['model'])
            if not _evidence_is_possible(evidence, DAG['model']):
                raise ValueError('[bnlearn] >evidence %s has zero probability under the model. Rejection sampling cannot produce matching samples.' %(evidence))
            if verbose>=3: print('[bnlearn] >Bayesian rejection sampling for %.0d samples conditioned on %.0d evidence variable(s)..' %(n, len(states)))
            df = infer_model.rejection_sample(evidence=states, size=n, seed=None, show_progress=(verbose>=3))
    elif methodtype=='gibbs':
        if evidence is not None:
            raise ValueError("[bnlearn] >Gibbs sampling does not support conditioning on evidence. Use methodtype='bayes' together with evidence=... for conditional (rejection) sampling.")
        if verbose>=3: print('[bnlearn] >Gibbs sampling for %.0d samples..' %(n))
        # Gibbs sampling
        gibbs = GibbsSampling(DAG['model'])
        df = gibbs.sample(size=n, seed=None)
    else:
        raise ValueError('[bnlearn] >Sampling methodtype [%s] is unknown. Use "bayes" or "gibbs".' %(methodtype))
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
