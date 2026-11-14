"""Benchmark: hillclimbsearch with and without an HNET skeleton on random discrete Bayesian networks.

For every random network (ground truth) data is sampled, and structure learning is performed twice:
    1. bn.structure_learning.fit(df, methodtype='hc')
    2. bn.structure_learning.fit(df, methodtype='hc', skeleton='hnet')
The learned DAGs are compared with the ground truth, and the results are averaged over the networks.

Usage
-----
    python benchmark_hnet_skeleton.py
    python benchmark_hnet_skeleton.py --n_networks 20 --n_samples 500 1000 10000 --seed 0
    python benchmark_hnet_skeleton.py --out_csv results.csv --out_rst results.rst
    python benchmark_hnet_skeleton.py --n_samples 1000 --multtest fdr_bh     # other multiple test correction in hnet

Reproducibility
---------------
The networks and the sampled data are fully determined by --seed, and so is the HNET skeleton.
The hill climbing of pgmpy breaks ties between equally scoring edges in the iteration order of sets, which depends on
Python's hash seed. This affects hc with and without the skeleton and mostly the edge directions: the directed
metrics can differ by about 0.02 between runs. The difference between hc and hc + hnet is much more stable.
For exactly reproducible numbers, run with the environment variable PYTHONHASHSEED=0.
Runtime is machine dependent.
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD

import bnlearn as bn


# %% Random ground-truth network
def random_network(n_nodes, rng, p_edge=0.25, max_parents=3, min_card=2, max_card=4, concentration=0.4):
    """Create a random discrete Bayesian network.

    Nodes are V0..V{n-1} in topological order. Each node can have an edge from every earlier node with probability
    ``p_edge`` (at most ``max_parents`` parents are kept). Each variable has between ``min_card`` and ``max_card`` states.
    The columns of every CPT are drawn from a Dirichlet distribution; a low ``concentration`` gives dependencies that are
    strong enough to be detected, but not deterministic.

    Returns
    -------
    edges : list of tuple
    cpds : list of TabularCPD
    nodes : list of str
    """
    nodes = [f'V{i}' for i in range(n_nodes)]
    card = {v: int(rng.integers(min_card, max_card + 1)) for v in nodes}
    edges = []
    for j in range(1, n_nodes):
        parents = [nodes[i] for i in range(j) if rng.random() < p_edge]
        rng.shuffle(parents)
        edges += [(p, nodes[j]) for p in parents[:max_parents]]
    parents = {v: [a for a, b in edges if b == v] for v in nodes}

    cpds = []
    for v in nodes:
        evidence_card = [card[p] for p in parents[v]]
        n_columns = int(np.prod(evidence_card)) if evidence_card else 1
        values = rng.dirichlet(np.ones(card[v]) * concentration, size=n_columns).T
        cpds.append(TabularCPD(v, card[v], values, evidence=parents[v] or None, evidence_card=evidence_card or None))
    return edges, cpds, nodes


# %% Scores
def edge_scores(true_edges, pred_edges):
    """Precision and recall of the undirected skeleton and of the directed edges."""
    t_und, p_und = {frozenset(e) for e in true_edges}, {frozenset(e) for e in pred_edges}
    t_dir, p_dir = set(true_edges), set(pred_edges)
    return {'und_precision': len(t_und & p_und) / max(len(p_und), 1),
            'und_recall': len(t_und & p_und) / max(len(t_und), 1),
            'dir_precision': len(t_dir & p_dir) / max(len(p_dir), 1),
            'dir_recall': len(t_dir & p_dir) / max(len(t_dir), 1)}


# %% One random network
def run_network(seed, n_samples, min_nodes, max_nodes, multtest='holm'):
    rng = np.random.default_rng(seed)
    edges = []
    while len(edges) == 0:
        edges, cpds, nodes = random_network(int(rng.integers(min_nodes, max_nodes + 1)), rng)
    isolated = [v for v in nodes if not any(v in e for e in edges)]
    truth = bn.make_DAG(edges, CPD=cpds, isolated_nodes=isolated, checkmodel=False)

    np.random.seed(seed)
    df = bn.sampling(truth, n=n_samples)

    t = time.time()
    model_hc = bn.structure_learning.fit(df.copy(), methodtype='hc')
    time_hc = time.time() - t
    t = time.time()
    model_hn = bn.structure_learning.fit(df.copy(), methodtype='hc', skeleton='hnet', params_hnet={'multtest': multtest})
    time_hn = time.time() - t

    skeleton = model_hn['skeleton']['edges']
    true_und = {frozenset(e) for e in edges}
    hit = len(true_und & {frozenset(e) for e in skeleton})

    row = {'seed': seed, 'n_samples': n_samples, 'n_nodes': len(nodes), 'n_edges': len(edges),
           'skeleton_edges': len(skeleton), 'skeleton_recall': hit / len(true_und), 'skeleton_precision': hit / max(len(skeleton), 1),
           'time_hc': time_hc, 'time_hc_hnet': time_hn}
    row.update({f'hc_{k}': v for k, v in edge_scores(edges, model_hc['model_edges']).items()})
    row.update({f'hc_hnet_{k}': v for k, v in edge_scores(edges, model_hn['model_edges']).items()})
    return row


# %% Summary table
METRICS = [('Undirected precision', 'und_precision'),
           ('Undirected recall', 'und_recall'),
           ('Directed precision', 'dir_precision'),
           ('Directed recall', 'dir_recall')]


def summarize(df):
    """Mean over networks for one sample size: rows are metrics; columns are hc, hc + hnet and the change."""
    rows = []
    for name, key in METRICS:
        a, b = df[f'hc_{key}'].mean(), df[f'hc_hnet_{key}'].mean()
        rows.append((name, a, b, b - a))
    a, b = df['time_hc'].mean(), df['time_hc_hnet'].mean()
    rows.append(('Runtime (seconds)', a, b, b / a))
    out = pd.DataFrame(rows, columns=['Metric', 'hc', 'hc + hnet', 'Change'])
    return out


def format_change(metric, value):
    return f'x{value:.1f}' if metric.startswith('Runtime') else f'{value:+.2f}'


def to_text(df_n, summary):
    lines = [f'{"Metric":<22}{"hc":>8}{"hc + hnet":>12}{"Change":>9}']
    for _, r in summary.iterrows():
        lines.append(f'{r["Metric"]:<22}{r["hc"]:>8.2f}{r["hc + hnet"]:>12.2f}{format_change(r["Metric"], r["Change"]):>9}')
    lines.append(f'Skeleton (hnet): recall={df_n["skeleton_recall"].mean():.2f}, precision={df_n["skeleton_precision"].mean():.2f}, '
                 f'edges={df_n["skeleton_edges"].mean():.1f} (true edges={df_n["n_edges"].mean():.1f})')
    return '\n'.join(lines)


def to_rst(df_n, summary, n_samples):
    """RST list-table that can be pasted in the documentation."""
    out = [f'.. list-table:: {n_samples:,} samples ({len(df_n)} random networks)',
           '   :header-rows: 1',
           '   :widths: 40 20 20 20',
           '']
    for header in ['Metric', 'hc', 'hc + hnet', 'Change']:
        out.append(f'   {"* -" if header == "Metric" else "  -"} {header}')
    for _, r in summary.iterrows():
        out.append(f'   * - {r["Metric"]}')
        out.append(f'     - {r["hc"]:.2f}')
        out.append(f'     - {r["hc + hnet"]:.2f}')
        out.append(f'     - {format_change(r["Metric"], r["Change"])}')
    return '\n'.join(out)


# %% Main
def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--n_networks', type=int, default=20, help='Number of random networks per sample size.')
    parser.add_argument('--n_samples', type=int, nargs='+', default=[500, 1000, 10000], help='Sample sizes.')
    parser.add_argument('--min_nodes', type=int, default=8)
    parser.add_argument('--max_nodes', type=int, default=12)
    parser.add_argument('--multtest', default='holm', help="Multiple test correction of hnet (params_hnet): 'holm', 'fdr_bh', ..., or 'none'.")
    parser.add_argument('--seed', type=int, default=0, help='Networks use the seeds seed, seed+1, ...')
    parser.add_argument('--out_csv', default=None, help='Write the results per network to this csv file.')
    parser.add_argument('--out_rst', default=None, help='Write the summary tables as RST to this file.')
    args = parser.parse_args()

    warnings.filterwarnings('ignore')
    bn.set_logger('error')
    multtest = None if args.multtest.lower() == 'none' else args.multtest

    rows = []
    for n_samples in args.n_samples:
        for i in range(args.n_networks):
            rows.append(run_network(args.seed + i, n_samples, args.min_nodes, args.max_nodes, multtest))
            print(f'\rn_samples={n_samples}: network {i + 1}/{args.n_networks}', end='', flush=True)
        print()
    results = pd.DataFrame(rows)

    rst = []
    for n_samples, df_n in results.groupby('n_samples', sort=False):
        summary = summarize(df_n)
        print(f'\n=== {n_samples:,} samples, {len(df_n)} random networks ({args.min_nodes}-{args.max_nodes} nodes), hnet multtest={args.multtest}')
        print(to_text(df_n, summary))
        rst.append(to_rst(df_n, summary, n_samples))

    if args.out_csv: results.to_csv(args.out_csv, index=False)
    if args.out_rst:
        with open(args.out_rst, 'w') as f:
            f.write('\n\n'.join(rst) + '\n')


if __name__ == '__main__':
    main()
