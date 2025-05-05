"""Inference is same as asking conditional probability questions to the models.

# ------------------------------------
# Name        : inference.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------

"""
# %% Libraries
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination
import numpy as np
import bnlearn


# %% Exact inference using Variable Elimination
def fit(model,
        variables=None,
        evidence=None,
        to_df=True,
        elimination_order='greedy',
        joint=True,
        groupby=None,
        plot=False,
        verbose=3,
        ):
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
        The output is converted in the dataframe [query.df]. Enabling this function may impact the processing speed.
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
    plot : bool, optional
        If True, display a bar plot.
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

    # Store dataframe in query
    if to_df or plot:
        # Convert to Dataframe
        query.df = bnlearn.query2df(query, variables=variables, groupby=groupby, verbose=verbose)
        # Make readable text
        query.text = summarize_inference(variables, evidence, query, plot=plot, verbose=verbose)
        if verbose>=3 and query.text is not None: print(query.text)
    else:
        query.df = None
        query.text = None

    # Return
    return query

#%%
def summarize_inference(variables, evidence, query, plot=False, verbose=3):
    """
    Summarize inference results based on a Bayesian Network inference output.

    Parameters
    ----------
    variables : list of str
        Variables being queried (e.g., ['Machine failure'] or multiple).
    evidence : dict
        Evidence variables and their fixed values (e.g., {'Torque [Nm]_category': 'high'}).
    query : Object from inference.fit()
        Inference output containing the queried variables and probability 'p' in a Dataframe (query.df)
    plot : bool, optional
        If True, display a bar plot.
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

    Returns
    -------
    str
        A textual summary.

    """
    df = query.df

    def is_binary(series):
        return sorted(series.dropna().unique()) in [[0, 1], [1, 0]]

    lines = []
    lines.append(f"\nSummary for variables: {variables}")
    evidence_txt = f"{', '.join([f'{k}={v}' for k, v in evidence.items()])}"
    lines.append(f"Given evidence: {evidence_txt}")

    for var in variables:
        lines.append(f"\n{var} outcomes:")
        grouped = df.groupby(var)['p'].sum()
        total = grouped.sum()
        for val, prob in grouped.items():
            description = f"{var}: {val}"
            lines.append(f"- {description} ({prob/total:.1%})")

    if plot:
        # Plot dominant probabilities
        for var in variables:
            grouped = df.groupby(var)['p'].sum()
            total = grouped.sum()
            percentages = (grouped / total) * 100

            plt.figure(figsize=(8, 4))
            bars = plt.barh(percentages.index.astype(str), percentages.values, color='#4a90e2', edgecolor='black')
            plt.xlabel('Percentage (%)', fontsize=12)
            plt.title(f'Inference Summary: {var}\n{evidence_txt}', fontsize=12)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.gca().invert_yaxis()

            # Add percentages at end of bars
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 1.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', fontsize=10)

            plt.xlim(0, max(percentages.values)*1.1)  # Make 10% larger
            plt.tight_layout()
            plt.show()

    return "\n".join(lines)
