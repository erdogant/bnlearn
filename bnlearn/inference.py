"""Inference is same as asking conditional probability questions to the models.
"""
# ------------------------------------
# Name        : inference.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------


# %% Libraries
from pgmpy.inference import VariableElimination
import bnlearn
import numpy as np


# %% Exact inference using Variable Elimination
def fit(model, variables=None, evidence=None, to_df=True, elimination_order='greedy', joint=True, groupby=None, verbose=3):
    """Inference using using Variable Elimination.

    The basic concept of variable elimination is same as doing marginalization over Joint Distribution.
    But variable elimination avoids computing the Joint Distribution by doing marginalization over much smaller factors.
    So basically if we want to eliminate X from our distribution, then we compute the product of all the factors
    involving X and marginalize over them, thus allowing us to work on much smaller factors.

    Parameters
    ----------
    model : dict
        Contains model.
    variables : List, optional
        For exact inference, P(variables | evidence). The default is None.
            * ['Name_of_node_1']
            * ['Name_of_node_1', 'Name_of_node_2']
    evidence : dict, optional
        For exact inference, P(variables | evidence). The default is None.
            * {'Rain':1}
            * {'Rain':1, 'Sprinkler':0, 'Cloudy':1}
    to_df : Bool, (default is True)
        The output is converted in the dataframe [query.df]. Enabling this function may heavily impact the processing speed.
    elimination_order: str or list (default='greedy')
        Order in which to eliminate the variables in the algorithm. If list is provided,
        should contain all variables in the model except the ones in `variables`. str options
        are: `greedy`, `WeightedMinFill`, `MinNeighbors`, `MinWeight`, `MinFill`. Please
        refer https://pgmpy.org/exact_infer/ve.html#module-pgmpy.inference.EliminationOrder
        for details.
    joint: boolean (default: True)
        If True, returns a Joint Distribution over `variables`.
        If False, returns a dict of distributions over each of the `variables`.
    groupby: list of strings (default: None)
        The query is grouped on the variable name by taking the maximum P value for each catagory.
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

    Returns
    -------
    query inference object.

    Examples
    --------
    >>> import bnlearn as bn
    >>>
    >>> # Load example data
    >>> model = bn.import_DAG('sprinkler')
    >>> bn.plot(model)
    >>>
    >>> # Do the inference
    >>> query = bn.inference.fit(model, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})
    >>> print(query)
    >>> query.df
    >>>
    >>> query = bn.inference.fit(model, variables=['Wet_Grass','Rain'], evidence={'Sprinkler':1})
    >>> print(query)
    >>> query.df
    >>>

    """
    if not isinstance(model, dict): raise Exception('[bnlearn] >Error: Input requires a object that contains the key: model.')
    adjmat = model['adjmat']
    if not np.all(np.isin(variables, adjmat.columns)):
        raise Exception('[bnlearn] >Error: [variables] should match names in the model (Case sensitive!)')
    if not np.all(np.isin([*evidence.keys()], adjmat.columns)):
        raise Exception('[bnlearn] >Error: [evidence] should match names in the model (Case sensitive!)')
    if verbose>=3: print('[bnlearn] >Variable Elimination.')

    # Extract model
    if isinstance(model, dict):
        model = model['model']

    # Check BayesianNetwork
    if 'BayesianNetwork' not in str(type(model)):
        if verbose>=1: print('[bnlearn] >Warning: Inference requires BayesianNetwork. hint: try: parameter_learning.fit(DAG, df, methodtype="bayes") <return>')
        return None

    # Convert to BayesianNetwork
    if 'BayesianNetwork' not in str(type(model)):
        model = bnlearn.to_bayesiannetwork(adjmat, verbose=verbose)

    try:
        model_infer = VariableElimination(model)
    except:
        raise Exception('[bnlearn] >Error: Input model does not contain learned CPDs. hint: did you run parameter_learning.fit()?')

    # Computing the probability P(class | evidence)
    query = model_infer.query(variables=variables, evidence=evidence, elimination_order=elimination_order, joint=joint, show_progress=(verbose>=3))
    # Store also in dataframe
    query.df = bnlearn.query2df(query, variables=variables, groupby=groupby, verbose=verbose) if to_df else None
    # Return
    return query
