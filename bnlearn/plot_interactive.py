"""Interactive plotting for bnlearn (d3blocks based).

Overview
--------
This module holds everything that is needed for the interactive plots, so that
``plot.py`` stays focused on the static plot, properties and layouts.

Entry point
    * :func:`plot_interactive`: builds the tooltips and dispatches to the chart type.

Chart types (``params_interactive['type']``)
    * ``'flowgraph'`` : :func:`_plot_interactive_flowgraph`
        Layered process diagram. Hovering a node or an edge shows a formatted CPD table.
    * ``'d3graph'``   : :func:`_plot_interactive_d3graph`
        Force directed graph.

Tooltips / CPD tables
    * :func:`_node_cpd_tooltips`, :func:`_edge_cpd_tooltips`: ``{node|edge: html}``
    * :func:`_tabular_cpd_html`, :func:`_gaussian_cpd_html`, :func:`_linear_gaussian_cpd_html`
    * :func:`_inject_edge_tooltips`: d3blocks only supports plain-text edge hovers, so the
      generated HTML file is patched.

Notes
-----
This module must not import ``bnlearn.plot`` (``plot.py`` imports this module).
"""
# ------------------------------------
# Name        : plot_interactive.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------

# %% Libraries
import logging
logger = logging.getLogger("bnlearn")
import os
import re
import json
import html as _html
import itertools
import webbrowser
from packaging import version

import numpy as np
import pandas as pd
from ismember import ismember


# %% Entry point
def plot_interactive(model, params_interactive, nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights, title):
    """Create the interactive plot.

    Parameters
    ----------
    model : dict
        bnlearn model. Parameter-learned models (``bn.parameter_learning.fit``) give CPD tooltips.
    params_interactive : dict
        Interactive settings. ``params_interactive['type']`` is 'flowgraph' or 'd3graph'.
    nodelist, node_colors, node_sizes : list
        Node names and their properties (same order).
    edgelist, edge_colors, edge_weights : list
        Edges as (source, target) and their properties (same order).
    title : str
        Title of the plot.

    Returns
    -------
    str or None
        Absolute path of the HTML file, or None when nothing was created.

    """
    chart_type = params_interactive['type']

    if chart_type == 'd3graph':
        tooltip = _node_tooltips_d3graph(model, nodelist)
        return _plot_interactive_d3graph(params_interactive, nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights, title, tooltip)

    if chart_type == 'flowgraph':
        # Neat CPD tables for edges (CPD of the target) and nodes (own CPD)
        edge_tooltip = _edge_cpd_tooltips(model, edgelist)
        node_tooltip = _node_cpd_tooltips(model, nodelist)
        tooltip = [node_tooltip.get(n, None) for n in nodelist] if node_tooltip else nodelist
        return _plot_interactive_flowgraph(params_interactive, nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights, title, tooltip, edge_tooltip=edge_tooltip)

    logger.warning(f"D3Block for {chart_type} not implemented in bnlearn.")
    return None


# %% Shared helpers
def _import_d3blocks():
    """Import d3blocks, check the version and return the D3Blocks class."""
    try:
        from d3blocks import D3Blocks
        import d3blocks as d3
    except ModuleNotFoundError:
        raise Exception('[bnlearn] >"d3blocks" library is not installed. Pip install first: "pip install d3blocks"')

    if version.parse(d3.__version__) < version.parse("2.0.0"):
        raise ImportError('[bnlearn] >Error: d3blocks version should be >= 2.0.0. Pip install: "pip install -U d3blocks"')
    return D3Blocks


def _model_cpds(model):
    """Return ``({variable: cpd}, {variable: cg_local_params})`` from a bnlearn model dict."""
    cpds, cg = {}, {}
    if not isinstance(model, dict):
        return cpds, cg
    bnmodel = model.get('model', None)
    if hasattr(bnmodel, 'get_cpds'):
        try:
            cpds = {c.variable: c for c in bnmodel.get_cpds()}
        except Exception as err:
            logger.debug('Could not collect CPDs: %s' % err)
    cg = {c['variable']: c for c in (model.get('continuous_cpds') or []) if isinstance(c, dict) and 'variable' in c}
    return cpds, cg


def _cpd_html_for_node(node, cpds, cg, parent=None, max_rows=8, max_cols=6):
    """HTML CPD of ``node``; the hovered ``parent`` (edge) is highlighted. None if no CPD."""
    try:
        if node in cg:
            c = cg[node]
            configs = {k: {'beta': d['beta'], 'std': d['std'], 'n': d.get('n')} for k, d in c['configs'].items()}
            return _gaussian_cpd_html(node, c.get('disc_parents', []), c.get('cont_parents', []), configs, parent=parent, max_rows=max_rows)
        if node in cpds:
            cpd = cpds[node]
            if hasattr(cpd, 'state_names') and hasattr(cpd, 'get_values'):
                return _tabular_cpd_html(cpd, parent=parent, max_rows=max_rows, max_cols=max_cols)
            return _linear_gaussian_cpd_html(cpd, parent=parent)
    except Exception as err:
        logger.debug('No CPD tooltip for %s: %s' % (node, err))
    return None


# %% Interactive flowgraph
def _plot_interactive_flowgraph(params_interactive, nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights, title, tooltip, edge_tooltip=None):
    """Create an interactive flowgraph (layered process diagram) with d3blocks.

    ``edge_tooltip`` is an optional dict ``{(source, target): html}``. d3blocks only
    supports plain-text edge hovers, so the HTML is injected into the generated file.
    """
    D3Blocks = _import_d3blocks()

    defaults = {
        'figsize': [None, None],
        'notebook': False,
        'filepath': None,
        'direction': 'TB',
        'layout': 'dagre',
        'animate': False,
        'show_side_panel': True,
        'show_top_panel': True,
        'dark_mode': True,
        'bgcolor': None,
    }
    params_interactive = {**defaults, **params_interactive}

    if params_interactive['filepath'] is None:
        params_interactive['filepath'] = title.strip().replace(' ', '_') + '.html'

    nodelist = [
        x.encode('ascii', 'ignore').decode('utf-8').replace(' ', '_')
        for x in list(nodelist)
    ]

    X = pd.DataFrame(data=edgelist, columns=['source', 'target'])
    X['source'] = X['source'].astype(str).str.replace(' ', '_')
    X['target'] = X['target'].astype(str).str.replace(' ', '_')
    X['weight'] = edge_weights

    nodes = {}
    for i, name in enumerate(nodelist):
        nodes[name] = {
            'color': node_colors[i] if node_colors is not None else None,
            'tooltip': tooltip[i] if tooltip is not None else None,
        }

    # Edge tooltips need a post-processed HTML file, so the browser/notebook is
    # opened by us after patching instead of by d3blocks.
    edge_html = {}
    if edge_tooltip:
        for (u, v), h in edge_tooltip.items():
            edge_html[str(u).replace(' ', '_') + '->' + str(v).replace(' ', '_')] = h
    node_has_html = any(isinstance(t, str) and 'bn-cpd' in t for t in (tooltip or []))
    patch = len(edge_html) > 0 or node_has_html
    show_notebook = params_interactive['notebook']

    d3b = D3Blocks()
    d3b.flowgraph(
        X,
        nodes=nodes,
        direction=params_interactive['direction'],
        layout=params_interactive['layout'],
        # edge_color=edge_color,
        animate=params_interactive['animate'],
        title=title,
        filepath=params_interactive['filepath'],
        figsize=params_interactive['figsize'],
        notebook=False if patch else show_notebook,
        showfig=False if patch else True,
        show_side_panel=params_interactive['show_side_panel'],
        show_top_panel=params_interactive['show_top_panel'],
        dark_mode=params_interactive['dark_mode'],
        color_background=params_interactive['bgcolor'],
    )

    filepath = os.path.abspath(d3b.config['filepath'])
    if patch:
        # If patching fails (d3blocks changed?), the plain graph is still shown.
        _inject_edge_tooltips(filepath, edge_html)
        _open_html(filepath, notebook=show_notebook, figsize=params_interactive['figsize'])
    return filepath


# %% Edge CPD tooltips
_EDGE_TOOLTIP_CSS = """
<style>
#fg-tooltip { max-width: 520px !important; }
#fg-tooltip table.bn-cpd { border-collapse: collapse; margin-top: 5px; font-size: 11px; }
#fg-tooltip table.bn-cpd th, #fg-tooltip table.bn-cpd td { padding: 2px 7px; text-align: right; border: 1px solid rgba(128,128,128,.35); white-space: nowrap; }
#fg-tooltip table.bn-cpd th { font-weight: 600; background: rgba(128,128,128,.18); }
#fg-tooltip table.bn-cpd th.s, #fg-tooltip table.bn-cpd th.e { text-align: left; }
#fg-tooltip table.bn-cpd th.hl, #fg-tooltip table.bn-cpd td.hl { outline: 1.5px solid #f0b429; outline-offset: -1.5px; }
#fg-tooltip .bn-cpd-title { font-family: Georgia, serif; font-style: italic; margin-top: 4px; }
#fg-tooltip .bn-cpd-eq { font-family: Consolas, monospace; font-size: 11px; margin-top: 5px; }
#fg-tooltip .bn-cpd-warn { margin-top: 6px; padding: 4px 6px; border-left: 3px solid #f0b429; background: rgba(240,180,41,.14); font-size: 11px; white-space: normal; max-width: 330px; }
#fg-tooltip .bn-cpd-note { color: var(--text-muted); font-size: 10.5px; margin-top: 3px; }
</style>
"""

_EDGE_TOOLTIP_JS_OLD = re.compile(r"if \(d\.label\) html \+= '<div>' \+ escapeHtml\(d\.label\) \+ '</div>';")
_EDGE_TOOLTIP_JS_NEW = ("if (d.label) html += '<div>' + escapeHtml(d.label) + '</div>';"
                        " var __cpd = (window.__BN_EDGE_CPD || {})[d.source + '->' + d.target];"
                        " if (__cpd) html += __cpd;")


def _inject_edge_tooltips(filepath, edge_html):
    """Patch a d3blocks flowgraph HTML file so edge hovers also show ``edge_html``."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            page = f.read()
        block = _EDGE_TOOLTIP_CSS
        if edge_html:
            if _EDGE_TOOLTIP_JS_OLD.search(page):
                page = _EDGE_TOOLTIP_JS_OLD.sub(lambda m: _EDGE_TOOLTIP_JS_NEW, page, count=1)
                data = json.dumps(edge_html, ensure_ascii=False).replace('</', '<\\/')
                block += '<script>window.__BN_EDGE_CPD = ' + data + ';</script>\n'
            else:
                logger.warning('Edge CPD tooltips are not supported by this d3blocks version; skipped.')
        if '</head>' in page:
            page = page.replace('</head>', block + '</head>', 1)
        else:
            page = block + page
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(page)
        return True
    except Exception as err:
        logger.warning('Could not add edge CPD tooltips: %s' % err)
        return False


def _open_html(filepath, notebook=False, figsize=(None, None)):
    """Show an HTML file inline (notebook) or in the browser."""
    if notebook:
        try:
            from IPython.display import IFrame, display
            width = figsize[0] if figsize and figsize[0] else '100%'
            height = figsize[1] if figsize and figsize[1] else 650
            display(IFrame(src=filepath, width=width, height=height))
            return
        except Exception:
            pass
    webbrowser.open('file://' + filepath)


def _fmt_num(x, nd=3):
    """Compact number formatting for tooltips."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return _html.escape(str(x))
    if x != 0 and abs(x) < 10 ** (-nd):
        return '%.1e' % x
    if abs(x) >= 1e5:
        return '%.3g' % x
    return '%.*f' % (nd, x)


def _cpd_title(variable, evidence, parent):
    """P(variable | e1, e2, ...) with the hovered parent in bold."""
    if evidence:
        ev = ', '.join('<b>%s</b>' % _html.escape(str(e)) if e == parent else _html.escape(str(e)) for e in evidence)
        return '<div class="bn-cpd-title">P(%s | %s)</div>' % (_html.escape(str(variable)), ev)
    return '<div class="bn-cpd-title">P(%s)</div>' % _html.escape(str(variable))


def _tabular_cpd_html(cpd, parent=None, max_rows=8, max_cols=6):
    """HTML table for a (pgmpy) TabularCPD: states as rows, parent configurations as columns."""
    var = cpd.variable
    evidence = list(cpd.variables[1:])
    values = np.asarray(cpd.get_values(), dtype=float)
    names = cpd.state_names
    var_states = list(names[var])
    combos = list(itertools.product(*[list(names[e]) for e in evidence])) if evidence else [()]
    n_rows, n_cols = values.shape
    show_r, show_c = min(n_rows, max_rows), min(n_cols, max_cols)

    h = [_cpd_title(var, evidence, parent), '<table class="bn-cpd">']
    if evidence:
        for i, e in enumerate(evidence):
            cls = 'e hl' if e == parent else 'e'
            h.append('<tr><th class="%s">%s</th>' % (cls, _html.escape(str(e))))
            h.extend('<th>%s</th>' % _html.escape(str(combos[j][i])) for j in range(show_c))
            h.append('<th>…</th></tr>' if show_c < n_cols else '</tr>')
    else:
        h.append('<tr><th class="e"></th><th>P</th></tr>')
    for r in range(show_r):
        h.append('<tr><th class="s">%s=%s</th>' % (_html.escape(str(var)), _html.escape(str(var_states[r]))))
        for c in range(show_c):
            p = float(values[r, c])
            h.append('<td style="background:rgba(66,133,244,%.2f)">%s</td>' % (0.05 + 0.55 * min(max(p, 0.0), 1.0), _fmt_num(p)))
        h.append('<td>…</td></tr>' if show_c < n_cols else '</tr>')
    if show_r < n_rows:
        h.append('<tr><th class="s">…</th>' + '<td>…</td>' * (show_c + (1 if show_c < n_cols else 0)) + '</tr>')
    h.append('</table>')
    notes = []
    if show_c < n_cols:
        notes.append('showing %d of %d parent configurations' % (show_c, n_cols))
    if show_r < n_rows:
        notes.append('showing %d of %d states' % (show_r, n_rows))
    if notes:
        h.append('<div class="bn-cpd-note">%s</div>' % '; '.join(notes))
    if parent is not None and parent not in evidence:
        h.append('<div class="bn-cpd-warn">⚠ <b>%s</b> is not a parent in this CPD. Conditional Gaussian models '
                 'cannot condition a discrete node on a continuous one, so the table is the marginal '
                 'P(%s) and the edge %s → %s has no learned parameters.</div>'
                 % (_html.escape(str(parent)), _html.escape(str(var)), _html.escape(str(parent)), _html.escape(str(var))))
    return ''.join(h)


def _gaussian_cpd_html(variable, disc_parents, cont_parents, configs, parent=None, max_rows=8):
    """HTML for a (conditional) linear Gaussian: one regression row per discrete-parent configuration.

    configs : dict {tuple(discrete states): {'beta': [intercept, b1, ...], 'std': float, 'n': int | None}}
    """
    all_parents = list(disc_parents) + list(cont_parents)
    items = list(configs.items())
    show = items[:max_rows]
    has_n = any(c.get('n') is not None for _, c in items)

    h = [_cpd_title(variable, all_parents, parent)]
    # Equation for the common single-regression case
    if len(items) == 1 and len(disc_parents) == 0:
        c = items[0][1]
        terms = [_fmt_num(c['beta'][0])] + ['%s·%s' % (_fmt_num(b), _html.escape(str(p))) for b, p in zip(c['beta'][1:], cont_parents)]
        h.append('<div class="bn-cpd-eq">%s = %s + ε,&nbsp; ε ~ N(0, %s²)</div>' % (
            _html.escape(str(variable)), ' + '.join(terms), _fmt_num(c['std'])))

    h.append('<table class="bn-cpd"><tr>')
    for p in disc_parents:
        h.append('<th class="e%s">%s</th>' % (' hl' if p == parent else '', _html.escape(str(p))))
    h.append('<th>intercept</th>')
    for p in cont_parents:
        h.append('<th%s>β(%s)</th>' % (' class="hl"' if p == parent else '', _html.escape(str(p))))
    h.append('<th>σ</th>' + ('<th>n</th>' if has_n else '') + '</tr>')
    for key, c in show:
        h.append('<tr>')
        for k in key:
            h.append('<th class="s">%s</th>' % _html.escape(str(k)))
        beta = list(c['beta'])
        h.append('<td>%s</td>' % _fmt_num(beta[0]))
        for i, p in enumerate(cont_parents):
            h.append('<td%s>%s</td>' % (' class="hl"' if p == parent else '', _fmt_num(beta[i + 1]) if i + 1 < len(beta) else ''))
        h.append('<td>%s</td>' % _fmt_num(c['std']))
        if has_n:
            h.append('<td>%s</td>' % (int(c['n']) if c.get('n') is not None else ''))
        h.append('</tr>')
    h.append('</table>')
    if len(items) > len(show):
        h.append('<div class="bn-cpd-note">showing %d of %d configurations</div>' % (len(show), len(items)))
    return ''.join(h)


def _linear_gaussian_cpd_html(cpd, parent=None):
    """HTML for a pgmpy LinearGaussianCPD (version tolerant)."""
    beta = getattr(cpd, 'beta', None)
    if beta is None:
        beta = getattr(cpd, 'mean', None)
    std = getattr(cpd, 'std', None)
    if std is None and getattr(cpd, 'variance', None) is not None:
        std = float(np.sqrt(cpd.variance))
    if beta is None or std is None:
        return None
    evidence = list(getattr(cpd, 'evidence', []) or [])
    configs = {(): {'beta': [float(b) for b in np.ravel(beta)], 'std': float(std), 'n': None}}
    return _gaussian_cpd_html(cpd.variable, [], evidence, configs, parent=parent)


def _node_cpd_tooltips(model, nodelist, max_rows=8, max_cols=6):
    """Create a neat HTML CPD for every node that has learned parameters.

    Returns
    -------
    dict
        ``{node: html}``. Nodes without a CPD are left out.
    """
    cpds, cg = _model_cpds(model)
    out = {}
    for node in nodelist:
        h = _cpd_html_for_node(node, cpds, cg, max_rows=max_rows, max_cols=max_cols)
        if h:
            out[node] = h
    return out


def _edge_cpd_tooltips(model, edgelist, max_rows=8, max_cols=6):
    """Create a neat HTML CPD for every edge.

    For an edge ``A -> B`` the CPD of the target ``B`` is shown, because ``B`` is the node whose
    distribution is conditioned on ``A``. Supports discrete (tabular), linear-gaussian and
    conditional-gaussian (``model['continuous_cpds']``) models. Edges without learned parameters
    get no entry, so the default hover is kept.

    Returns
    -------
    dict
        ``{(source, target): html}``
    """
    cpds, cg = _model_cpds(model)
    if not cpds and not cg:
        logger.info('No learned parameters found: use bn.parameter_learning.fit() first to show CPDs on edge hover.')
        return {}

    out = {}
    for u, v in edgelist:
        h = _cpd_html_for_node(v, cpds, cg, parent=u, max_rows=max_rows, max_cols=max_cols)
        if h:
            out[(str(u), str(v))] = h
    return out


# %% Legacy node tooltips (d3graph)
def _node_tooltips_d3graph(model, nodelist):
    """Tooltips for d3graph: the pgmpy CPD of each node, or the node name when unavailable."""
    bnmodel = model.get('model', None) if isinstance(model, dict) else None
    if not hasattr(bnmodel, 'get_cpds'):
        return nodelist

    # Collect model nodes if available to avoid ValueError when node is missing
    try:
        model_nodes = set(bnmodel.nodes())
    except Exception:
        model_nodes = None

    tooltip = []
    for node in nodelist:
        tip = node
        # Only call get_cpds if the node is present in the underlying model
        if model_nodes is None or node in model_nodes:
            try:
                tip = bnmodel.get_cpds(node)
                if tip is None:
                    tip = node
            except Exception:
                # Fall back to the node label if anything goes wrong
                tip = node
        tooltip.append(tip)
    return tooltip


# %% Interactive d3graph
def _plot_interactive_d3graph(params_interactive, nodelist, node_colors, node_sizes, edgelist, edge_colors, edge_weights, title, tooltip):
    D3Blocks = _import_d3blocks()

    if params_interactive['filepath'] is None:
        params_interactive['filepath'] = title.strip().replace(' ', '_') + '.html'

    nodelist = list(map(lambda x: x.encode('ascii', 'ignore').decode("utf-8").replace(' ', '_'), list(nodelist)))

    # Initialize
    d3 = D3Blocks()

    X = pd.DataFrame(data=edgelist, columns=['source', 'target'])
    X['weight'] = edge_weights

    # If there are no edges or all weights are non-positive, d3graph/d3blocks
    # will raise an exception. Handle this case gracefully by logging and
    # returning None so callers can proceed (or fall back to a static plot).
    if X.shape[0] == 0 or (X['weight'] <= 0).all():
        logger.info('No edges with positive weight for interactive d3graph; skipping interactive plot.')
        return None

    d3.d3graph(X, showfig=False, title=title, notebook=params_interactive['notebook'])

    d3.D3graph.set_edge_properties(directed=True,
                                   minmax_distance=params_interactive['minmax_distance'],
                                   marker_color=edge_colors)

    _, IB = ismember([*d3.D3graph.node_properties.keys()], nodelist)

    d3.D3graph.set_node_properties(tooltip=np.array(tooltip)[IB],
                                   size=np.array(node_sizes)[IB],
                                   color=np.array(node_colors)[IB],
                                   fontcolor=params_interactive['font_color'])

    d3.D3graph.show(show_slider=params_interactive['show_slider'],
                    filepath=params_interactive['filepath'],
                    figsize=params_interactive['figsize'])

    return os.path.abspath(d3.D3graph.config['filepath'])
