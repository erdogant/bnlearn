"""This is a module of bnlearn for the generation of synthetic data."""
# ------------------------------------
# Name        : sampling.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------

from pgmpy.sampling import BayesianModelSampling, GibbsSampling

# %% Sampling from model
def sampling(DAG, n=1000, methodtype='bayes', verbose=0):
    """Generate sample(s) using the joint distribution of the network.

    Parameters
    ----------
    DAG : dict
        Contains model and the adjmat of the DAG.
    methodtype : str (default: 'bayes')
        * 'bayes': Forward sampling using Bayesian.
        * 'gibbs' : Gibbs sampling.
    n : int, optional
        Number of samples to generate. The default is 1000.
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

    """
    if n<=0: raise ValueError('Number of samples (n) must be 1 or larger!')
    if (DAG is None) or ('BayesianNetwork' not in str(type(DAG['model']))):
        raise ValueError('The input model (DAG) must contain BayesianNetwork.')

    if len(DAG['model'].get_cpds())==0:
        raise Exception('[bnlearn] >Error! This is a Bayesian DAG containing only edges, and no CPDs. Tip: you need to specify or learn the CPDs. Try: DAG=bn.parameter_learning.fit(DAG, df). At this point you can make a plot with: bn.plot(DAG).')
        return

    if methodtype=='bayes':
        if verbose>=3: print('[bnlearn] >Bayesian forward sampling for %.0d samples..' %(n))
        # Bayesian Forward sampling and make dataframe
        infer_model = BayesianModelSampling(DAG['model'])
        df = infer_model.forward_sample(size=n, seed=None, show_progress=(True if verbose>=3 else False))
    elif methodtype=='gibbs':
        if verbose>=3: print('[bnlearn] >Gibbs sampling for %.0d samples..' %(n))
        # Gibbs sampling
        gibbs = GibbsSampling(DAG['model'])
        df = gibbs.sample(size=n, seed=None)
    else:
        if verbose>=3: print('[bnlearn] >Methodtype [%s] unknown' %(methodtype))
    return df
