"""Techniques for inference.

Description
-----------
Inference is same as asking conditional probability questions to the models.
i.e., What is the probability of a sprinkler is on given that it is raining which is basically equivalent of asking $ P(g^1 | i^1) $.
Inference algorithms deals with efficiently finding these conditional probability queries.
"""
# ------------------------------------
# Name        : inference.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------


# %% Libraries
from pgmpy.inference import VariableElimination
from bnlearn.bnlearn import to_BayesianModel
import numpy as np


# %% Exact inference using Variable Elimination
def fit(model, variables=None, evidence=None, verbose=3):
    """Inference using using Variable Elimination.

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
    verbose : int, optional
        Print progress to screen. The default is 3.
            * 0: NONE
            * 1: ERROR
            * 2: WARNING
            * 3: INFO (default)
            * 4: DEBUG
            * 5: TRACE

    Returns
    -------
    q.

    Examples
    --------
    >>> import bnlearn as bn
    >>>
    >>> # Load example data
    >>> model = bn.import_DAG('sprinkler')
    >>> bn.plot(model)
    >>>
    >>> # Do the inference
    >>> q1 = bn.inference.fit(model, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})
    >>> q2 = bn.inference.fit(model, variables=['Wet_Grass','Rain'], evidence={'Sprinkler':1})

    """
    if not isinstance(model, dict): raise Exception('[bnlearn] >Error: Input requires a DAG that contains the key: model.')
    adjmat = model['adjmat']
    if not np.all(np.isin(variables, adjmat.columns)):
        raise Exception('[bnlearn] >Error: [variables] should match names in the model (Case sensitive!)')
    if not np.all(np.isin([*evidence.keys()], adjmat.columns)):
        raise Exception('[bnlearn] >Error: [evidence] should match names in the model (Case sensitive!)')
    if verbose>=3: print('[bnlearn] >Variable Elimination..')

    # Extract model
    if isinstance(model, dict):
        model = model['model']

    # Check BayesianModel
    if 'BayesianModel' not in str(type(model)):
        if verbose>=1: print('[bnlearn] >Warning: Inference requires BayesianModel. hint: try: parameter_learning.fit(DAG, df, methodtype="bayes") <return>')
        return None

    # Convert to BayesianModel
    if 'BayesianModel' not in str(type(model)):
        model = to_BayesianModel(adjmat, verbose=verbose)

    try:
        model_infer = VariableElimination(model)
    except:
        raise Exception('[bnlearn] >Error: Input model does not contain learned CPDs. hint: did you run parameter_learning.fit?')

    # Computing the probability of Wet Grass given Rain.
    q = model_infer.query(variables=variables, evidence=evidence)
    print(q)
    # for varname in variables: print(q[varname])
    return(q)
