"""Inference is same as asking conditional probability questions to the models."""
# ------------------------------------
# Name        : inference.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------

# %% Libraries
import logging
logger = logging.getLogger("bnlearn")
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination
from pgmpy.models import LinearGaussianBayesianNetwork
import numpy as np
import pandas as pd
import bnlearn
import warnings
from bnlearn.utils import model_kind

warnings.filterwarnings("ignore")


# %% Exact inference using Variable Elimination
def fit(model,
        variables=None,
        evidence=None,
        to_df=True,
        elimination_order='greedy',
        joint=True,
        groupby=None,
        plot=False,
        do=None,
        ):
    """Inference router: discrete Variable Elimination or continuous / CG.

    Chooses the backend from the fitted model:
        * discrete BayesianNetwork  -> :func:`fit_discrete`
        * LinearGaussianBayesianNetwork -> :func:`fit_continuous`
        * Conditional-Gaussian (mixed)  -> :func:`fit_continuous`

    Parameters match :func:`fit_discrete` and :func:`fit_continuous`.
    Evidence may contain discrete states and/or continuous numeric values when
    the model is linear-Gaussian or CG.

    """
    kind = model_kind(model)
    if kind in ('linear-gaussian', 'cg'):
        return fit_continuous(
            model,
            variables=variables,
            evidence=evidence,
            do=do,
            to_df=to_df,
            plot=plot,
            elimination_order=elimination_order,
            joint=joint,
            groupby=groupby,
        )
    return fit_discrete(
        model,
        variables=variables,
        evidence=evidence,
        to_df=to_df,
        elimination_order=elimination_order,
        joint=joint,
        groupby=groupby,
        plot=plot,
        do=do,
    )


# %% Exact inference using Variable Elimination (discrete — original behaviour)
def fit_discrete(model,
        variables=None,
        evidence=None,
        to_df=True,
        elimination_order='greedy',
        joint=True,
        groupby=None,
        plot=False,
        do=None,
        ):
    """Inference using Variable Elimination (discrete networks).

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
    do : dict, optional
        Interventions for causal inference, P(variables | do(X=x), evidence).
        Whereas evidence conditions on passively observed values, do simulates
        setting the variable by intervention: incoming edges of the intervened
        variables are cut (Pearl's do-operator, comparable to mutilated() in the
        R version of bnlearn). The query runs on the mutilated network, so it
        combines freely with evidence and the other query options. The default is None.
        Example: {'Sprinkler':1}
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
    >>> # Causal inference: P(Wet_Grass | do(Sprinkler=1)) differs from the
    >>> # observational P(Wet_Grass | Sprinkler=1) because the intervention
    >>> # cuts the Cloudy->Sprinkler edge.
    >>> query = bn.inference.fit(model, variables=['Wet_Grass'], do={'Sprinkler':1})
    >>> query.df
    >>>

    """
    if not isinstance(model, dict): raise Exception('[bnlearn] >Error: Input requires a object that contains the key: model.')
    adjmat = model['adjmat']
    if not np.all(np.isin(variables, adjmat.columns)):
        raise Exception('[bnlearn] >Error: [variables] should match names in the model (Case sensitive!)')
    if evidence is not None and not np.all(np.isin([*evidence.keys()], adjmat.columns)):
        raise Exception('[bnlearn] >Error: [evidence] should match names in the model (Case sensitive!)')
    if do is not None and not np.all(np.isin([*do.keys()], adjmat.columns)):
        raise Exception('[bnlearn] >Error: [do] should match names in the model (Case sensitive!)')
    if do is not None and evidence is not None and (set(do.keys()) & set(evidence.keys())):
        raise Exception('[bnlearn] >Error: A variable can not be in both [do] and [evidence]: %s' %(set(do.keys()) & set(evidence.keys())))
    logger.info('Causal inference with do-operator.' if do else '[bnlearn] >Variable Elimination.')
    # Extract model
    if isinstance(model, dict):
        model = model['model']

    # Check BayesianNetwork
    if 'BayesianNetwork' not in str(type(model)):
        logger.error('Warning: Inference requires BayesianNetwork. hint: try: parameter_learning.fit(DAG, df, methodtype="bayes") <return>')
        return None

    # Convert to BayesianNetwork
    if 'BayesianNetwork' not in str(type(model)):
        model = bnlearn.to_bayesiannetwork(adjmat)

    try:
        if do:
            # Query the mutilated network (incoming edges of the intervened nodes
            # are cut) with the interventions fixed as evidence. This is exact for
            # any mix of do and evidence, including interventions on causally
            # related nodes, where pgmpy's adjustment-based CausalInference.query
            # is not, and it keeps elimination_order/joint applicable.
            model = model.do(list(do.keys()))
        model_infer = VariableElimination(model)
    except ValueError as e:
        raise Exception(f'[bnlearn] >Error: {e}')
        # Input model does not contain learned CPDs. hint: did you run parameter_learning.fit()?

    # Computing the probability P(class | do, evidence): in the mutilated network,
    # fixing the intervened variables as evidence equals intervening on them.
    query_evidence = {**do, **(evidence or {})} if do else evidence
    query = model_infer.query(variables=variables, evidence=query_evidence, elimination_order=elimination_order, joint=joint, show_progress=logger.isEnabledFor(logging.INFO))

    # Store dataframe in query
    if isinstance(query, dict):
        # joint=False returns a dict of per-variable factors; there is no single
        # joint table to attach a dataframe or summary to. (Attaching attributes
        # to the dict raised AttributeError before, so this also unbreaks joint=False.)
        return query
    if to_df or plot:
        # Convert to Dataframe
        query.df = bnlearn.query2df(query, variables=variables, groupby=groupby)
        # Make readable text; label interventions as do(X) to keep them apart from observations
        summary_given = {**{f'do({k})': v for k, v in (do or {}).items()}, **(evidence or {})}
        query.text = summarize_inference(variables, summary_given, query, plot=plot)
        if query.text is not None: print(query.text)
    else:
        query.df = None
        query.text = None

    # Return
    return query


# %% Continuous / Conditional-Gaussian inference
class ContinuousQueryResult:
    """Container for continuous (and CG continuous) query results."""

    def __init__(self, means, variables, evidence=None, do=None, variances=None):
        self.means = means
        self.variances = variances
        self.variables = list(variables)
        self.evidence = evidence or {}
        self.do = do or {}
        self.df = None
        self.text = None

    def __repr__(self):
        parts = ['%s: mean=%.4f' % (k, v) for k, v in self.means.items()]
        return 'ContinuousQueryResult(%s)' % ', '.join(parts)


def fit_continuous(model,
                   variables=None,
                   evidence=None,
                   do=None,
                   to_df=True,
                   plot=False,
                   elimination_order='greedy',
                   joint=True,
                   groupby=None,
                   ):
    """Inference for Linear-Gaussian and Conditional-Gaussian (mixed) models.

    Parameters
    ----------
    model : dict
        bnlearn model after parameter learning (linear-gaussian or cg).
    variables : list of str
        Query variables.
    evidence : dict, optional
        Observed values (discrete states and/or continuous numbers).
    do : dict, optional
        Interventions; incoming edges are treated as cut. Continuous do fixes
        the value; discrete do uses the discrete sub-model mutilation when present.
    to_df, plot
        Summary options.
    elimination_order, joint, groupby
        Passed through to discrete sub-queries in CG models.

    Returns
    -------
    ContinuousQueryResult, discrete query object, or dict with keys
    'discrete' / 'continuous' for mixed CG queries.
    """
    if not isinstance(model, dict):
        raise Exception('[bnlearn] >Error: Input requires a object that contains the key: model.')
    if variables is None:
        raise Exception('[bnlearn] >Error: [variables] must be provided.')
    if isinstance(variables, str):
        variables = [variables]

    adjmat = model.get('adjmat')
    if adjmat is not None:
        cols = list(adjmat.columns.astype(str))
        if not np.all(np.isin(variables, cols)):
            raise Exception('[bnlearn] >Error: [variables] should match names in the model (Case sensitive!)')
        if evidence is not None and not np.all(np.isin([*evidence.keys()], cols)):
            raise Exception('[bnlearn] >Error: [evidence] should match names in the model (Case sensitive!)')
        if do is not None and not np.all(np.isin([*do.keys()], cols)):
            raise Exception('[bnlearn] >Error: [do] should match names in the model (Case sensitive!)')
    if do is not None and evidence is not None and (set(do.keys()) & set(evidence.keys())):
        raise Exception('[bnlearn] >Error: A variable can not be in both [do] and [evidence]: %s' % (set(do.keys()) & set(evidence.keys())))

    kind = model_kind(model)
    logger.info('Continuous/CG inference (%s)%s.' % (kind, ' with do-operator' if do else ''))
    if kind == 'linear-gaussian':
        return _query_linear_gaussian(model, variables=variables, evidence=evidence, do=do, to_df=to_df)

    if kind == 'cg':
        return _query_cg(
            model, variables=variables, evidence=evidence, do=do, to_df=to_df,
            elimination_order=elimination_order, joint=joint, groupby=groupby,
            plot=plot,
        )

    logger.error('Warning: fit_continuous expected a linear-gaussian or cg model; falling back to fit_discrete.')
    return fit_discrete(
        model, variables=variables, evidence=evidence, do=do, to_df=to_df,
        elimination_order=elimination_order, joint=joint, groupby=groupby, plot=plot,
    )


def _query_linear_gaussian(model_dict, variables, evidence=None, do=None, to_df=True):
    """Conditional means for a LinearGaussianBayesianNetwork (step 7 + 8)."""
    lg = model_dict['model']
    if lg is None or not isinstance(lg, LinearGaussianBayesianNetwork):
        logger.error('Warning: linear-gaussian inference requires a LinearGaussianBayesianNetwork.')
        return None

    evidence = dict(evidence or {})
    do = dict(do or {})
    known = {**do, **evidence}

    means = {}
    predict_vars = [v for v in variables if v not in known]
    for v in variables:
        if v in known:
            means[v] = float(known[v])

    if predict_vars:
        all_nodes = list(lg.nodes())
        row = {n: known[n] for n in all_nodes if n in known}
        if not row:
            try:
                samp = lg.simulate(n_samples=1, do=do or None, evidence=None, seed=0)
                for v in predict_vars:
                    if v in samp.columns:
                        means[v] = float(samp[v].iloc[0])
            except Exception:
                for v in predict_vars:
                    cpd = lg.get_cpds(v)
                    beta = getattr(cpd, 'beta', [0.0])
                    means[v] = float(beta[0]) if beta is not None else 0.0
        else:
            evidence_df = pd.DataFrame([row])
            try:
                pred = lg.predict(evidence_df)
                if isinstance(pred, pd.DataFrame):
                    for v in predict_vars:
                        if v in pred.columns:
                            means[v] = float(pred[v].iloc[0])
                elif isinstance(pred, tuple) and len(pred) >= 2:
                    order, mean_arr = pred[0], np.atleast_2d(pred[1])
                    for i, name in enumerate(order):
                        if name in predict_vars:
                            means[name] = float(mean_arr[0, i])
                else:
                    for v in predict_vars:
                        means[v] = np.nan
            except Exception as err:
                logger.warning('Warning: Linear-Gaussian predict failed (%s); using simulate.' % err)
                samp = lg.simulate(n_samples=200, do=do or None, evidence=evidence or None, seed=0)
                for v in predict_vars:
                    if v in samp.columns:
                        means[v] = float(samp[v].mean())

    result = ContinuousQueryResult(means=means, variables=variables, evidence=evidence, do=do)
    if to_df:
        result.df = pd.DataFrame([{'variable': k, 'mean': v} for k, v in means.items()])
        given = {**{f'do({k})': v for k, v in do.items()}, **evidence}
        lines = ['Continuous query: %s' % variables, 'Given: %s' % given]
        for k, v in means.items():
            lines.append('  %s -> mean = %.6f' % (k, v))
        result.text = '\n'.join(lines)
        logger.info(result.text)
    return result


def _cg_cpd_map(continuous_cpds):
    return {c['variable']: c for c in (continuous_cpds or [])}


def _cg_mean_std(local, evidence):
    """Evaluate CG local mean and std given evidence (discrete + continuous parents)."""
    disc_parents = local['disc_parents']
    cont_parents = local['cont_parents']
    key = tuple(evidence[p] for p in disc_parents) if disc_parents else ()
    configs = local['configs']
    cfg = configs.get(key)
    if cfg is None and len(disc_parents) == 1:
        cfg = configs.get((evidence[disc_parents[0]],))
    if cfg is None:
        if not configs:
            return np.nan, np.nan
        cfg = next(iter(configs.values()))
    beta = cfg['beta']
    std = cfg['std']
    mu = float(beta[0])
    for i, p in enumerate(cont_parents):
        if p not in evidence:
            raise ValueError('[bnlearn] >CG query for %s needs continuous parent "%s" in evidence or do.'
                             % (local['variable'], p))
        mu += float(beta[i + 1]) * float(evidence[p])
    return mu, float(std)


def _query_cg(model_dict, variables, evidence=None, do=None, to_df=True,
              elimination_order='greedy', joint=True, groupby=None, plot=False):
    """Hybrid CG query: discrete via fit_discrete, continuous via local Gaussians."""
    evidence = dict(evidence or {})
    do = dict(do or {})

    continuous_cpds = model_dict.get('continuous_cpds') or []
    cpd_map = _cg_cpd_map(continuous_cpds)
    cfg = model_dict.get('config') or {}
    discrete_cols = set(cfg.get('discrete_cols') or [])
    continuous_cols = set(cfg.get('continuous_cols') or [])
    if not discrete_cols and not continuous_cols:
        continuous_cols = set(cpd_map.keys())
        disc_model = model_dict.get('model')
        if disc_model is not None and hasattr(disc_model, 'nodes'):
            discrete_cols = set(str(n) for n in disc_model.nodes()) - continuous_cols

    disc_query = [v for v in variables if v in discrete_cols or (v not in continuous_cols and v not in cpd_map)]
    cont_query = [v for v in variables if v in continuous_cols or v in cpd_map]

    disc_result = None
    if disc_query and model_dict.get('model') is not None:
        disc_evidence = {k: v for k, v in evidence.items() if k in discrete_cols}
        disc_do = {k: v for k, v in do.items() if k in discrete_cols}
        disc_result = fit_discrete(
            model_dict,
            variables=disc_query,
            evidence=disc_evidence or None,
            do=disc_do or None,
            to_df=to_df,
            elimination_order=elimination_order,
            joint=joint,
            groupby=groupby,
            plot=plot,
        )

    cont_means = {}
    cont_stds = {}
    if cont_query:
        known = {**do, **evidence}
        pending = list(cont_query)
        resolved = dict(known)
        for v in list(pending):
            if v in do:
                cont_means[v] = float(do[v])
                cont_stds[v] = 0.0
                resolved[v] = float(do[v])
                pending.remove(v)

        safety = 0
        while pending and safety < len(pending) + 5:
            safety += 1
            progress = False
            for v in list(pending):
                local = cpd_map.get(v)
                if local is None:
                    logger.warning('Warning: no CG CPD for continuous node "%s".' % v)
                    pending.remove(v)
                    progress = True
                    continue
                need = list(local['disc_parents']) + list(local['cont_parents'])
                if all(p in resolved for p in need):
                    mu, std = _cg_mean_std(local, resolved)
                    cont_means[v] = mu
                    cont_stds[v] = std
                    resolved[v] = mu
                    pending.remove(v)
                    progress = True
            if not progress:
                break
        for v in pending:
            logger.warning('Warning: could not resolve CG node "%s" (missing parent evidence).' % v)
            cont_means[v] = np.nan
            cont_stds[v] = np.nan

    if cont_query and not disc_query:
        result = ContinuousQueryResult(means=cont_means, variables=variables, evidence=evidence, do=do, variances=cont_stds)
        if to_df:
            result.df = pd.DataFrame([
                {'variable': k, 'mean': cont_means[k], 'std': cont_stds.get(k)} for k in cont_query
            ])
            given = {**{f'do({k})': v for k, v in do.items()}, **evidence}
            lines = ['CG continuous query: %s' % cont_query, 'Given: %s' % given]
            for k in cont_query:
                lines.append('  %s -> mean = %.6f, std = %.6f' % (k, cont_means[k], cont_stds.get(k, np.nan)))
            result.text = '\n'.join(lines)
            logger.info(result.text)
        return result

    if disc_query and not cont_query:
        return disc_result

    combined = {
        'discrete': disc_result,
        'continuous': ContinuousQueryResult(means=cont_means, variables=cont_query, evidence=evidence, do=do, variances=cont_stds),
        'variables': variables,
        'evidence': evidence,
        'do': do,
    }
    if to_df and cont_means:
        combined['continuous'].df = pd.DataFrame([
            {'variable': k, 'mean': cont_means[k], 'std': cont_stds.get(k)} for k in cont_query
        ])
    logger.info('CG hybrid query: discrete=%s continuous=%s' % (disc_query, cont_query))
    for k, v in cont_means.items():
        print('  %s -> mean = %.6f' % (k, v))
    return combined

#%%
def summarize_inference(variables, evidence, query, plot=False):
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
            labels = [f'state_{x}' for x in percentages.index]
            bars = plt.barh(labels, percentages.values, color='#4a90e2', edgecolor='black')
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
