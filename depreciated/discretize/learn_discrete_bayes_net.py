"""
Learning Discrete Bayesian Networks from Continuous Data.

This paper introduces a principled Bayesian discretization method for continuous
variables in Bayesian networks with quadratic complexity instead of the cubic
complexity of other standard techniques. Empirical demonstrations show that the
proposed method is superior to the established minimum description length algorithm.

In addition, this paper shows how to incorporate existing methods into the structure
learning process to discretize all continuous variables and simultaneously learn
Bayesian network structures.

Functions
---------
discretize_all()
    discretize continuous variables in a Bayesian network
    for which the network structure is known in advance

References
----------
.. [1] Yi-Chun Chen, Tim Allan Wheeler, Mykel John Kochenderfer (2015),
       Learning Discrete Bayesian Networks from Continuous Data :arxiv:`1512.02406`

.. [2] Julia 0.4 implementation:
       https://github.com/sisl/LearnDiscreteBayesNets.jl
"""

import math
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

Graph = List[Union[int, Tuple[int, ...]]]


def combine_spouses_data(spouse_value_matrix):
    # maak van [[1,2],[2,3]] -> [(1,2), (2,3)]
    # gaan we niet doen?
    return spouse_value_matrix


def parent_class_combine():
    pass


def lfact(n):
    return math.lgamma(n + 1)


def log_prob_single_edge_last_term(cl: pd.Series):
    # assert isinstance(cl, np.ndarray), "cl argument should be ndarray"
    n = len(cl)
    uniq_cl = np.unique(cl, axis=0)
    val_cl = len(uniq_cl)

    # Survey distribution for each data point
    distr_table = np.zeros((n, val_cl), dtype="int64")

    for index_data in range(n):
        class_cl = np.where(uniq_cl == cl[index_data])[0][0]
        distr_table[index_data, class_cl] = 1

    # Survey class distribution between point i and point j
    distr_intval_table = np.zeros((n, n, val_cl), dtype="int64")

    for ind_init in range(n):
        for ind_end in range(ind_init, n):
            if ind_init == ind_end:
                current_distr = distr_table[ind_init]
            else:
                current_distr = (
                    distr_intval_table[ind_init, ind_end - 1] + distr_table[ind_end]
                )

            distr_intval_table[ind_init, ind_end] = current_distr

    # print(distr_intval_table[1, n])
    # Get 1/P(D|M) for single edge case and count only last term in objective func

    inv_p = np.zeros((n, n))

    for ind_init in range(n):
        for ind_end in range(n):
            if ind_end < ind_init:
                inv_p[ind_init, ind_end] = np.inf
            else:
                distr_in_intval = distr_intval_table[ind_init, ind_end]
                current = lfact(ind_end - ind_init + 1)
                for ind_cl in range(val_cl):
                    current -= lfact(distr_in_intval[ind_cl])

                inv_p[ind_init, ind_end] = current

    return inv_p


def log_prob_spouse_child_data(child: pd.DataFrame, spouse: pd.DataFrame):
    assert isinstance(child, pd.DataFrame), "child argument should be a DataFrame"
    assert isinstance(spouse, pd.DataFrame), "spouse argument should be a DataFrame"

    n = len(child)
    uniq_sp = spouse.drop_duplicates()
    uniq_ch = child.drop_duplicates()
    val_sp = len(uniq_sp)
    val_ch = len(uniq_ch)

    # Survey distribution for each data point
    distr_table = np.zeros((n, val_sp, val_ch), dtype="int64")
    for index_data in range(n):
        # Find index of the rows in the unique rows for child and spouse and set to one
        class_ch = np.where((uniq_ch == child.iloc[index_data, :]).all(axis=1))[0][0]
        class_sp = np.where((uniq_sp == spouse.iloc[index_data, :]).all(axis=1))[0][0]
        distr_table[index_data, class_sp, class_ch] = 1

    # Survey class distribution in between point i and point j
    distr_intval_table = np.zeros((n, n, val_sp, val_ch), dtype="int64")
    for ind_val_sp in range(val_sp):
        for ind_init in range(n):
            for ind_end in range(ind_init, n):
                if ind_init == ind_end:
                    current_distr = distr_table[ind_end, ind_val_sp]
                else:
                    current_distr = (
                        distr_intval_table[ind_init, ind_end - 1, ind_val_sp]
                        + distr_table[ind_end, ind_val_sp]
                    )

                distr_intval_table[ind_init, ind_end, ind_val_sp] = current_distr

    # Get the 1/P(D|M) for this child-spouse set

    inv_p = np.zeros((n, n))

    for ind_init in range(n):
        for ind_end in range(n):
            if ind_end < ind_init:
                inv_p[ind_init, ind_end] = np.inf
            else:
                current_val = 0.0
                for ind_val_sp in range(val_sp):
                    distr_in_intval = distr_intval_table[ind_init, ind_end, ind_val_sp]
                    total_num = distr_in_intval.sum()

                    if total_num > 0:
                        # The forth term in objective function
                        current = lfact(total_num)
                        for ind_val_ch in range(val_ch):
                            current -= lfact(distr_in_intval[ind_val_ch])

                        # The third term in objective function
                        current += (
                            lfact(total_num + val_ch - 1)
                            - lfact(val_ch - 1)
                            - lfact(total_num)
                        )

                        # The 3rd and 4th terms for this value of spouse
                        current_val += current

                inv_p[ind_init, ind_end] = current_val

    return inv_p


def prior_of_intval(continuous: pd.Series, lam: int) -> np.ndarray:
    N = len(continuous)
    d_1_N = continuous.iloc[N - 1] - continuous.iloc[0]
    prior = np.zeros(N, dtype="float64")
    # Small positive number to avoid zero 
    epsilon = 1e-10  
    for i in range(N - 1):
        d_i = continuous.iloc[i + 1] - continuous.iloc[i]
        prior[i] = 1 - math.exp(-lam * d_i / d_1_N) + epsilon
    prior[N - 1] = 1

    return prior


def largest_class_value(data_matrix):
    return data_matrix.nunique().max()


def bn_discretizer_p_data_model(
    data_matrix, parent_set, child_spouse_set, approx=False
):
    (N, n) = data_matrix.shape
    n_p = len(parent_set)
    n_c = len(child_spouse_set)
    n_r = n_p + n_c

    parent_class_number = 1
    if len(parent_set) > 0:
        for parent in parent_set:
            multi = data_matrix.iloc[:, parent].nunique()
            parent_class_number *= multi

        # parent_class_number_2 = parent_class_combine(data_matrix, parent_set)

    # -log(P(D|M)) part:

    log_P_data_model = np.zeros((N, N))
    # nearest_var_set = parent_set + child_spouse_set

    if approx:
        raise ValueError("approx true untested")
        # for parent in parent_set:
        #     single_variable_data = data_matrix.iloc[:, parent]
        #     table = log_prob_single_edge_last_term(single_variable_data)
        #     log_P_data_model += table
    else:
        if n_p > 0:
            parent_data = np.zeros((N, n_p), dtype="int64")
            for p in range(n_p):
                parent_data[:, p] = data_matrix.iloc[:, parent_set[p]]

            combined_parent = combine_spouses_data(parent_data)
            table = log_prob_single_edge_last_term(combined_parent)
            log_P_data_model += table

    for child_spouse in child_spouse_set:
        if type(child_spouse) == int:
            child_data = data_matrix.iloc[:, [child_spouse]]
            spouse_data = pd.DataFrame(np.zeros(N, dtype="int64"))
            table = log_prob_spouse_child_data(child_data, spouse_data)
        else:
            # A child has more parents
            child_data = data_matrix.iloc[:, [child_spouse[0]]]
            spouse_matrix = data_matrix.iloc[:, list(child_spouse[1:])]
            # spouse_data = combine_spouses_data(spouse_matrix)
            table = log_prob_spouse_child_data(child_data, spouse_matrix)

        log_P_data_model += table

    # Add -log(p_class_distribution_for_parent)

    for i in range(N):
        for j in range(i, N):
            log_P_data_model[i, j] += (
                lfact(j - i + parent_class_number)
                - lfact(j - i + 1)
                - lfact(parent_class_number - 1)
            )

    return log_P_data_model


def bn_discretizer_free_number_rep(
    continuous, data_matrix, parent_set, child_spouse_set, approx=False
) -> np.ndarray:
    p_data_model = bn_discretizer_p_data_model(
        data_matrix, parent_set, child_spouse_set, approx
    )

    lam = largest_class_value(data_matrix)

    split_on_intval = prior_of_intval(continuous, lam)

    N = len(continuous)
    not_split_on_intval = np.ones(N, dtype="float64") - split_on_intval

    conti_norep = continuous.unique()

    N_norep = len(conti_norep)
    conti_head = np.empty(N_norep, dtype="int64")
    conti_tail = np.empty(N_norep, dtype="int64")

    conti_head[0] = 0
    index_head = 0
    for i in range(1, N):
        if continuous.iloc[i] != continuous.iloc[i - 1]:
            index_head += 1
            conti_head[index_head] = i

    conti_tail[-1] = N - 1
    index_tail = N_norep - 1
    for i in reversed(range(N - 1)):
        if continuous.iloc[i] != continuous.iloc[i + 1]:
            index_tail -= 1
            conti_tail[index_tail] = i

    # print(conti_tail==conti_head)
    smallest_value = np.empty(N_norep, dtype="float64")
    optimal_disc = []

    length_conti_data = continuous.iloc[N - 1] - continuous.iloc[0]

    for a in range(N_norep):
        if a == 0:
            smallest_value[a] = (
                -math.log(split_on_intval[conti_tail[0]])
                + p_data_model[conti_head[0], conti_tail[0]]
            )
            optimal_disc.append([conti_tail[0]])
        else:
            current_value = np.inf
            current_disc = [0]
            for b in range(a + 1):
                if b == a:
                    value = (
                        (
                            (
                                (continuous.iloc[conti_tail[a]] - continuous.iloc[0])
                                / length_conti_data
                            )
                            * lam
                        )
                        - math.log(split_on_intval[conti_tail[a]])
                        + p_data_model[0, conti_tail[a]]
                    )
                else:
                    value = (
                        smallest_value[b]
                        + (
                            (
                                (
                                    continuous.iloc[conti_tail[a]]
                                    - continuous.iloc[conti_head[b + 1]]
                                )
                                / length_conti_data
                            )
                            * lam
                        )
                        - math.log(split_on_intval[conti_tail[a]])
                        + p_data_model[conti_head[b + 1], conti_tail[a]]
                    )

                if value < current_value:
                    current_value = value
                    if b == a:
                        current_disc = [conti_tail[a]]
                    else:
                        current_disc = optimal_disc[b] + [conti_tail[a]]

            smallest_value[a] = current_value
            optimal_disc.append(current_disc)

    # print(optimal_disc[-1])
    optimal_disc_value = np.empty(len(optimal_disc[-1]) + 1, dtype="float64")

    for i in range(len(optimal_disc[-1]) + 1):
        if i == 0:
            optimal_disc_value[i] = continuous.iloc[0]
        elif i == len(optimal_disc[-1]):
            optimal_disc_value[i] = continuous.iloc[-1]
        else:
            optimal_disc_value[i] = 0.5 * (
                continuous.iloc[optimal_disc[-1][i - 1]]
                + continuous.iloc[optimal_disc[-1][i - 1] + 1]
            )

    return optimal_disc_value


def equal_width_disc(continuous: pd.Series, m: int) -> pd.Series:
    """Equal width discretization"""
    min_value = continuous.min()
    max_value = continuous.max()
    span = (max_value - min_value) / m

    return continuous.sub(min_value).floordiv(span).astype("int64").add(1).clip(upper=m)


def equal_width_edge(continuous: pd.Series, m: int):
    raise ValueError("dead code? equal_width_edge never called")
    # min_value = continuous[0]
    # max_value = continuous[-1]
    # span = max_value - min_value
    # edge = [continuous[0]]
    # for i in range(1, m+1):
    #     edge.append(min_value + (span/m)*i)
    #
    # return edge


def cartesian_product():
    pass


def find_intval():
    pass


def continuous_to_discrete(data: pd.Series, bin_edge) -> pd.Series:
    return pd.cut(data, bin_edge, right=True, labels=False, include_lowest=True) + 1


def sort_disc_by_vorder(
    continuous_order, disc_edge: List[np.ndarray]
) -> List[np.ndarray]:
    return [disc_edge[i] for i in np.argsort(continuous_order, kind="stable")]


def rand_seq():
    pass


def disc_intval_seq():
    pass


def graph_to_reverse_order(graph: Graph) -> List[int]:
    return [n[-1] if isinstance(n, tuple) else n for n in reversed(graph)]


def graph_to_reverse_conti_order(
    graph: Graph, continuous_index: List[int]
) -> List[int]:
    return [n for n in graph_to_reverse_order(graph) if n in continuous_index]


def graph_to_markov(graph: Graph, target: int):
    parent_set = []
    child_spouse_set = []

    for i in range(len(graph)):
        condi_graph = graph[i]
        if isinstance(condi_graph, int):
            if target == condi_graph:
                parent_set = []

        elif target in condi_graph:
            index = condi_graph.index(target)
            if index == len(condi_graph) - 1:
                parent_set = list(condi_graph)[:-1]
            else:
                child = condi_graph[-1]
                spouse = []

                for j in range(len(condi_graph) - 1):
                    if j != index:
                        spouse.append(condi_graph[j])

                child_spouse_set.append(tuple([child] + spouse))

    child_spouse_set = [cs if len(cs) > 1 else cs[0] for cs in child_spouse_set]

    return parent_set, child_spouse_set


def sortperm(a):
    return np.argsort(a, kind="stable")


def one_iteration(
    data: pd.DataFrame,
    data_integer: pd.DataFrame,
    graph: Graph,
    discrete_index: List[int],
    continuous_index: List[int],
    l_card: int,
    approx=False,
) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    # Save discretization edge
    disc_edge_collect = []

    for i in range(len(continuous_index)):
        target = continuous_index[i]

        increase_order = sortperm(data.iloc[:, target])
        conti = data.iloc[increase_order, target]

        # sort data_integer properly
        data_integer_sort = data_integer.iloc[increase_order, :]

        parent_set, child_spouse_set = graph_to_markov(graph, target)

        if len(parent_set) + len(child_spouse_set) == 0:
            raise ValueError("Untested")
            # disc_edge = equal_width_edge(conti, l_card)
            # disc_edge_collect.append(disc_edge)
        else:
            disc_edge = bn_discretizer_free_number_rep(conti, data_integer_sort, parent_set, child_spouse_set, approx)
            disc_edge_collect.append(disc_edge)

        # Update by current discretization
        data_integer.iloc[:, target] = continuous_to_discrete(data.iloc[:, target], disc_edge)

    return data_integer, disc_edge_collect


def bn_discretizer_iteration_converge(
    data: pd.DataFrame,
    graph: Graph,
    discrete_index: List[int],
    continuous_index: List[int],
    cut_time: int,
    approx=False,
    verbose=3):

    # initial the first data_integer
    l_card: int = data.iloc[:, discrete_index].nunique().max()
    if np.isnan(l_card): l_card = 0

    # pre-equal-width-discretize on continuous variables
    data_integer = data.copy()
    for i in continuous_index:
        data_integer.isetitem(i, equal_width_disc(data.iloc[:, i], l_card))

    disc_edge_collect = [[] for _ in continuous_index]

    for times in range(cut_time):
        if verbose>=3: print('[bnlearn] >Discretizer for continuous values. Iteration [%d].' %(times))
        disc_edge_previous = disc_edge_collect

        data_integer, disc_edge_collect = one_iteration(data, data_integer, graph, discrete_index, continuous_index, l_card, approx)
        # print(disc_edge_collect)

        if all(np.array_equal(a, b) for a, b in zip(disc_edge_previous, disc_edge_collect)):
            break

    return data_integer, disc_edge_collect


def K2_f():
    pass


def parent_table_to_graph():
    pass


def K2_one_iteration_discretization():
    pass


def K2_w_discretization():
    pass


def discretize_all(data_matrix: pd.DataFrame, graph: Graph, continuous_index: List[int], cut_time: int, verbose=3):
    """
    discretize continuous variables in a Bayesian network for which
    the network structure is known in advance

    Parameters
    ----------
    data_matrix : pandas DataFrame
        The data to be discretized.
    graph : list of tuple or int
        A list of tuple or int representing the parents of each target node.
    continuous_index : list of int
        The indices of the columns in the DataFrame that should be discretized.
    cut_time : int
        The maximum number of iterations to use when optimizing.

    Returns
    -------
    tuple of DataFrame and list of array
        The discretized DataFrame where every continuous column is converted
        into categories as integer.
        A list of arrays corresponding to the continuous_index with all the
        discretization edges.
    """
    discrete_index = [
        i for i in range(data_matrix.shape[1]) if i not in continuous_index
    ]
    sort_continuous = graph_to_reverse_conti_order(graph, continuous_index)
    # discretizer
    data_integer, edge = bn_discretizer_iteration_converge(
        data_matrix,
        graph,
        discrete_index,
        sort_continuous,
        cut_time,
        False,
        verbose=verbose,
    )

    reorder_edge = sort_disc_by_vorder(sort_continuous, edge)
    return data_integer, reorder_edge


def Learn_DVBN():
    """
    learn a discrete Bayesian network given a topological ordering of the variables
    """
    pass


def Learn_Discrete_Bayesian_Net():
    """
    learn a discrete Bayesian network for which the network structure is not known
    in advance.
    """
    pass
