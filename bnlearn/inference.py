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


# %% Exact inference using Variable Elimination
def fit(model, variables=None, evidence=None, verbose=3):
    """Inference using using Variable Elimination.

    Parameters
    ----------
    model : dict
        Contains model.
    variables : List, optional
        For exact inference, P(variables | evidence). The default is None.
        ['Name_of_node_1']
        ['Name_of_node_1', 'Name_of_node_2']
    evidence : dict, optional
        For exact inference, P(variables | evidence). The default is None.
        {'Rain':1}
        {'Rain':1, 'Sprinkler':0, 'Cloudy':1}
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: NONE
        1: ERROR
        2: WARNING
        3: INFO (default)
        4: DEBUG
        5: TRACE

    Returns
    -------
    q.


    Examples
    --------
    >>> import bnlearn
    >>> model = bnlearn.import_DAG('sprinkler')
    >>> bnlearn.plot(model)
    >>> q1 = bnlearn.inference.fit(model, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})
    >>> q2 = bnlearn.inference.fit(model, variables=['Wet_Grass','Rain'], evidence={'Sprinkler':1})

    """
    if verbose>=3: print('[BNLEARN][inference] Variable Elimination..')
    if isinstance(model, dict):
        model = model['model']

    model_infer = VariableElimination(model)
    # Computing the probability of Wet Grass given Rain.
    q = model_infer.query(variables=variables, evidence=evidence)
    print(q)
    # for varname in variables: print(q[varname])
    return(q)
