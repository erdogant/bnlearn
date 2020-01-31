"""This function provides techniques for inference."""
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

    Description
    -----------
    Inference is same as asking conditional probability questions to the models.
    i.e., What is the probability of a sprinkler is on given that it is raining which is basically equivalent of asking $ P(g^1 | i^1) $.
    Inference algorithms deals with efficiently finding these conditional probability queries.


    Parameters
    ----------
    model : dict
        Contains model

    variables : list of strings, optional (default: None)
        For exact inference, P(variables | evidence).
        ['Name_of_node_1']
        ['Name_of_node_1', 'Name_of_node_2']

    evidence : dict, optional  (default: None)
        For exact inference, P(variables | evidence).
        {'Rain':1}
        {'Rain':1, 'Sprinkler':0, 'Cloudy':1}

    verbose : int [0-5] (default: 3)
        Print messages to screen.
        0: NONE
        1: ERROR
        2: WARNING
        3: INFO (default)
        4: DEBUG
        5: TRACE

    Returns
    -------
    None.


    Description Extensive
    ---------------------
    There are two main categories for inference algorithms:
        1. Exact Inference: These algorithms find the exact probability values for our queries.
        2. Approximate Inference: These algorithms try to find approximate values by saving on computation.

    Exact Inference
        There are multiple algorithms for doing exact inference.

    Two common Inference algorithms with variable Elimination
        1. Clique Tree Belief Propagation
        2. Variable Elimination

    The basic concept of variable elimination is same as doing marginalization over Joint Distribution.
    But variable elimination avoids computing the Joint Distribution by doing marginalization over much smaller factors.
    So basically if we want to eliminate $ X $ from our distribution, then we compute
    the product of all the factors involving $ X $ and marginalize over them,
    thus allowing us to work on much smaller factors.

    In the above equation we can see that we pushed the summation inside and operated
    the summation only factors that involved that variable and hence avoiding computing the
    complete joint distribution.


    Example
    -------
    import bnlearn
    model = bnlearn.import_DAG('sprinkler')
    bnlearn.plot(model)
    q1 = bnlearn.inference.fit(model, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})
    q2 = bnlearn.inference.fit(model, variables=['Wet_Grass','Rain'], evidence={'Sprinkler':1})

    """
    if verbose>=3: print('[BNLEARN][inference] Variable Elimination..')
    model_infer = VariableElimination(model['model'])
    # Computing the probability of Wet Grass given Rain.
    q = model_infer.query(variables=variables, evidence=evidence)
    print(q)
    # for varname in variables: print(q[varname])
    return(q)
