"""
pyprism_v2.plotting
===================
Figures for the sweep.  Imports matplotlib lazily so the core package stays
NumPy-only.

Colour scheme
-------------
**Hue carries the substrate, lightness carries the method.**

    red     H1, the symmetry-blind hypergraph
    blue    H2, the symmetry- and irreducibility-aware hypergraph
    teal    PRISM and PRISM+, which exist only on H2 and are the proposal

A method keeps the *same lightness* in both ramps, so ``METIS/H1`` is dark red
and ``METIS/H2`` is dark blue -- the pair reads as one method seen through two
substrates, which is the comparison the figures exist to make.  Reserving a
third hue for PRISM keeps the proposal from being mistaken for just another
H2 baseline.
"""
from __future__ import annotations

import os

import numpy as np

__all__ = ['PALETTE', 'colour', 'substrate_of', 'base_name',
           'substrate_legend', 'plot_sweep', 'plot_fronts', 'assign_lanes',
           'plot_circuit', 'TEAL', 'RED', 'BLUE', 'C_H1', 'C_H2', 'C_PRISM',
           'C_PRISM_PLUS', 'C_NEUTRAL', 'INK', 'GRID']

INK = '#2b2f36'
GRID = dict(alpha=0.22, lw=0.6)

#: ramp anchors, light to dark.  Interpolated, so any number of methods can be
#: spread across a family without the ends becoming illegible.
TEAL = ('#a7e0da', '#5fc0b8', '#2aa39a', '#12817a', '#0a5a55', '#063e3a')
RED = ('#f7c9bb', '#f0a48d', '#e37a5f', '#d1553a', '#ae3a23', '#7f2716')
BLUE = ('#c2d8ee', '#98bae1', '#6f9ace', '#4a79b5', '#30588e', '#1d3a61')

#: canonical ordering; a method's position sets its lightness in *both* ramps.
#: METIS and KaHyPar sit at the dark end -- they are the strongest baselines and
#: the seeds PRISM+ multi-starts from, so they should read as the heavyweights.
_RANK = ('Naive', 'Random', 'Linear', 'Spectral', 'Greedy', 'KL', 'FM',
         'Louvain', 'GirvanNewman', 'WeightedSum', 'NSGA-II', 'SPEA2',
         'MOEA/D', 'EpsConstraint', 'pymoo/NSGA-II', 'pymoo/NSGA-III',
         'pymoo/SMS-EMOA', 'pymoo/AGE-MOEA', 'pymoo/MOEA-D',
         'METIS', 'KaHyPar')


def _lerp_hex(a, b, t):
    ca = tuple(int(a[i:i + 2], 16) for i in (1, 3, 5))
    cb = tuple(int(b[i:i + 2], 16) for i in (1, 3, 5))
    return '#%02x%02x%02x' % tuple(
        int(round(x + (y - x) * t)) for x, y in zip(ca, cb))


def _ramp(anchors, t):
    """Piecewise-linear interpolation along a hex ramp, ``t`` in [0, 1].

    Done by hand rather than through a matplotlib colormap so that ``colour``
    stays importable without matplotlib.
    """
    t = min(max(float(t), 0.0), 1.0)
    if t >= 1.0:
        return anchors[-1]
    u = t * (len(anchors) - 1)
    i = int(u)
    return _lerp_hex(anchors[i], anchors[i + 1], u - i)


def _shade(base):
    """Lightness position of a method, identical across substrates."""
    try:
        i = _RANK.index(base)
    except ValueError:
        return 0.55
    return 0.10 + 0.90 * i / (len(_RANK) - 1)


C_H1 = _ramp(RED, 0.62)          #: the H1 substrate itself, in figure text
C_H2 = _ramp(BLUE, 0.62)         #: the H2 substrate itself
C_PRISM = _ramp(TEAL, 0.45)
C_PRISM_PLUS = _ramp(TEAL, 0.82)
C_NEUTRAL = '#c9ced4'

_TRACK_CACHE = {}


def _registry_track(name):
    """Substrate of a method that carries no ``/H*`` suffix.

    Read from the registry rather than duplicated here, so adding a method in
    one place cannot silently give it the wrong colour.
    """
    if not _TRACK_CACHE:
        try:                                        # lazy: avoids a cycle
            from .partition import REGISTRY
            _TRACK_CACHE.update({m.name: m.track for m in REGISTRY.values()})
        except Exception:                           # pragma: no cover
            _TRACK_CACHE['__'] = 'H2'
    return _TRACK_CACHE.get(name, 'H2')


def substrate_of(label):
    """``'H1'``, ``'H2'``, or ``None`` for PRISM (which is H2 by definition but
    is coloured as the proposal, not as a baseline)."""
    if label.startswith('PRISM'):
        return None
    if label.endswith('/H1'):
        return 'H1'
    if label.endswith('/H2'):
        return 'H2'
    return _registry_track(label)


def base_name(label):
    return label[:-3] if label.endswith(('/H1', '/H2')) else label


def colour(label):
    if label.startswith('PRISM'):
        return C_PRISM_PLUS if label.rstrip().endswith('+') else C_PRISM
    ramp = RED if substrate_of(label) == 'H1' else BLUE
    return _ramp(ramp, _shade(base_name(label)))


#: materialised for anything that wants a plain dict; ``colour`` handles labels
#: outside it, so this is a convenience rather than the source of truth.
PALETTE = {'PRISM': C_PRISM, 'PRISM+': C_PRISM_PLUS}
PALETTE.update({f'{m}/H1': _ramp(RED, _shade(m)) for m in _RANK})
PALETTE.update({f'{m}/H2': _ramp(BLUE, _shade(m)) for m in _RANK})
PALETTE.update({m: _ramp(BLUE, _shade(m)) for m in _RANK})


def substrate_legend(ax, loc='best', fontsize=7.5, ncol=3, **kw):
    """A key for the hue convention, so a figure can be read on its own."""
    from matplotlib.patches import Patch
    h = [Patch(fc=C_H1, ec='none', label='$H_1$  symmetry-blind'),
         Patch(fc=C_H2, ec='none', label='$H_2$  symmetry-aware'),
         Patch(fc=C_PRISM_PLUS, ec='none', label='PRISM  (proposed)')]
    return ax.legend(handles=h, loc=loc, fontsize=fontsize, ncol=ncol,
                     frameon=False, **kw)


def assign_lanes(layer):
    """Spread the gates of one layer across non-overlapping lanes.

    Gates in the same layer share an x coordinate, so if their qubit spans
    overlap the vertical connectors are drawn on top of one another and the
    layer becomes unreadable.  Treating each gate as the closed interval
    ``[min(q_g), max(q_g)]`` -- a single-qubit gate being the degenerate
    interval ``[v, v]`` -- this is interval partitioning: sort by lower
    endpoint and put each gate in the first lane whose last interval ended
    strictly below it.  The greedy pass is optimal, so the lane count equals
    the largest number of mutually overlapping gates in the layer and nothing
    is spread further than it has to be.

    Returns ``(lane_of_gid, n_lanes)``.
    """
    items = sorted(((min(g.qubits), max(g.qubits), g.gid) for g in layer),
                   key=lambda z: (z[0], z[1]))
    lane_hi, lane_of = [], {}
    for lo, hi, gid in items:
        for li, prev_hi in enumerate(lane_hi):
            if lo > prev_hi:                 # strict: touching endpoints clash
                lane_hi[li] = hi
                lane_of[gid] = li
                break
        else:
            lane_hi.append(hi)
            lane_of[gid] = len(lane_hi) - 1
    return lane_of, max(len(lane_hi), 1)


def plot_circuit(layout, gates, n, path, packets=None, title='',
                 lane_span=0.78):
    """Draw the circuit with one lane per overlapping gate group."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
    from matplotlib.lines import Line2D
    from .gates import spec

    # Diagonal entanglers are the ones that pack, so they take the PRISM teal
    # and a packet span is a wash of the same hue; everything that cannot pack
    # is red.  Same convention as the method colours: teal is what the symmetry
    # buys, red is what it does not.
    C_DIAG, C_NDIAG, C_MULTI, C_1Q, C_PKT = (_ramp(TEAL, 0.55), C_H1,
                                             '#7a4f9c', '#c3cad2',
                                             _ramp(TEAL, 0.40))
    depth = len(layout)
    lanes = [assign_lanes(layer) for layer in layout]
    max_lanes = max((nl for _, nl in lanes), default=1)

    def xpos(g):
        lane_of, nl = lanes[g.layer]
        if nl <= 1:
            return float(g.layer)
        return g.layer + (lane_of[g.gid] - (nl - 1) / 2.0) * (lane_span / nl)

    w = min(0.40 * depth * max(1.0, 0.55 + 0.15 * max_lanes) + 3, 26)
    fig, ax = plt.subplots(figsize=(w, 0.30 * n + 2.2))
    for v in range(n):
        ax.plot([-0.6, depth - 0.4], [v, v], color='#dfe4ea', lw=1.0, zorder=0)
    for t in range(depth):
        if t % 2:
            ax.axvspan(t - 0.5, t + 0.5, color='#f2f4f6', lw=0, zorder=-1)

    if packets:
        by_id = {g.gid: g for g in gates}
        for root, gids in packets:
            if len(gids) < 2:
                continue
            xs = [xpos(by_id[i]) for i in gids]
            ax.add_patch(FancyBboxPatch(
                (min(xs) - 0.22, root - 0.30), max(xs) - min(xs) + 0.44, 0.60,
                boxstyle='round,pad=0.02,rounding_size=0.12',
                fc=C_PKT, ec='none', alpha=0.18, zorder=1))

    for g in gates:
        sp = spec(g.name)
        col = (C_1Q if g.m == 1 else C_MULTI if g.m >= 3
               else (C_DIAG if sp.diagonal else C_NDIAG))
        x = xpos(g)
        if g.m == 1:
            ax.add_patch(Rectangle((x - 0.16, g.qubits[0] - 0.16), 0.32, 0.32,
                                   fc=col, ec='white', lw=0.5, zorder=3))
        else:
            qs = sorted(g.qubits)
            ax.plot([x, x], [qs[0], qs[-1]], color=col, lw=1.4, zorder=2,
                    solid_capstyle='round')
            for j, q in enumerate(g.qubits):
                ax.add_patch(Circle((x, q), 0.125 if j == 0 else 0.09, fc=col,
                                    ec='white', lw=0.55, zorder=4))
    ax.set_xlim(-1, depth)
    ax.set_ylim(-1, n)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f'$q_{{{v}}}$' for v in range(n)], fontsize=7)
    ax.set_xlabel('layer $t$')
    ax.set_title(title, fontsize=10)
    ax.invert_yaxis()
    ax.grid(False)
    for s_ in ('top', 'right', 'left'):
        ax.spines[s_].set_visible(False)
    handles = [Line2D([], [], color=C_DIAG, lw=3, label='2q diagonal (packs)'),
               Line2D([], [], color=C_NDIAG, lw=3,
                      label='2q non-diagonal (cannot pack)'),
               Line2D([], [], color=C_MULTI, lw=3, label='$m_g\\geq3$'),
               Line2D([], [], color=C_1Q, lw=3, label='1q')]
    if packets:
        handles.append(Line2D([], [], color=C_PKT, lw=6, alpha=0.35,
                              label='packet span'))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.13),
              ncol=5, frameon=False, fontsize=8)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def _order(df, metric='hv_norm'):
    g = df.groupby('method')[metric].mean().sort_values(ascending=False)
    return list(g.index)


def plot_sweep(df, meta, path, title=''):
    """Four panels: front quality, ebits, runtime, and scaling with n."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = _order(df)
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.22)

    # (a) hypervolume, normalised per instance
    ax = fig.add_subplot(gs[0, 0])
    data = [df[df.method == m]['hv_norm'].dropna().values for m in methods]
    bp = ax.boxplot(data, patch_artist=True, widths=0.62, showfliers=False,
                    medianprops=dict(color=INK, lw=1.3))
    for patch, m in zip(bp['boxes'], methods):
        patch.set_facecolor(colour(m))
        patch.set_alpha(0.9)
        patch.set_edgecolor('#41474f')
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods, rotation=55, ha='right', fontsize=6.5)
    ax.set_ylabel('hypervolume / best on that instance')
    ax.set_title('(a)  Front quality, normalised per instance',
                 fontsize=10, loc='left')
    ax.grid(axis='y', **GRID)

    # (b) ebits relative to Naive
    ax = fig.add_subplot(gs[0, 1])
    piv = df.pivot_table(index=['n', 'k', 'seed'], columns='method',
                         values='E_CD')
    if 'Naive' in piv.columns:
        rel = piv.div(piv['Naive'], axis=0)
        data = [rel[m].dropna().values for m in methods if m in rel.columns]
        lbl = [m for m in methods if m in rel.columns]
        bp = ax.boxplot(data, patch_artist=True, widths=0.62, showfliers=False,
                        medianprops=dict(color=INK, lw=1.3))
        for patch, m in zip(bp['boxes'], lbl):
            patch.set_facecolor(colour(m))
            patch.set_alpha(0.9)
            patch.set_edgecolor('#41474f')
        ax.axhline(1.0, ls='--', color=INK, lw=1.0)
        ax.set_xticks(range(1, len(lbl) + 1))
        ax.set_xticklabels(lbl, rotation=55, ha='right', fontsize=6.5)
    ax.set_ylabel('$E$ / $E_{\\mathrm{Naive}}$')
    ax.set_title('(b)  Ebit cost relative to the trivial partition',
                 fontsize=10, loc='left')
    ax.grid(axis='y', **GRID)

    # (c) runtime
    ax = fig.add_subplot(gs[1, 0])
    med = [max(df[df.method == m]['seconds'].median(), 1e-6) for m in methods]
    ax.bar(np.arange(len(methods)), med,
           color=[colour(m) for m in methods], edgecolor='#41474f', lw=0.5)
    ax.set_yscale('log')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=55, ha='right', fontsize=6.5)
    ax.set_ylabel('median wall-clock  [s, log]')
    ax.set_title('(c)  Cost of each method', fontsize=10, loc='left')
    ax.grid(axis='y', which='both', **GRID)

    # (d) scaling with n
    ax = fig.add_subplot(gs[1, 1])
    show = [m for m in methods
            if m in ('PRISM', 'PRISM+') or m.split('/')[0] in
            ('KaHyPar', 'METIS', 'FM', 'NSGA-II', 'Naive', 'Spectral')][:9]
    for m in show:
        sub = df[df.method == m].groupby('n')['E_CD'].mean()
        if len(sub) < 2:
            continue
        ax.plot(sub.index, sub.values, '-o', ms=4, color=colour(m),
                lw=2.0 if m.startswith('PRISM') else 1.1, label=m)
    ax.set_xlabel('$n$   [qubits]')
    ax.set_ylabel('mean $E$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('(d)  Scaling of the ebit cost', fontsize=10, loc='left')
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    ax.grid(which='both', **GRID)

    if title:
        fig.suptitle(title, y=0.985, fontsize=11)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_fronts(fronts, path, title='', top=10):
    """The CD/CK fronts of one instance."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ranked = sorted(fronts.items(),
                    key=lambda kv: -(len(kv[1]) if kv[1] else 0))
    picks = [k for k, _ in ranked if k.startswith('PRISM')]
    picks += [k for k, _ in ranked if not k.startswith('PRISM')][:top]
    for lab in picks:
        f = fronts.get(lab) or []
        if not f:
            continue
        E = np.array([p[0] for p in f], float)
        K = np.array([p[1] for p in f], float)
        o = np.argsort(E)
        ax.plot(E[o], K[o], '-', lw=2.0 if lab.startswith('PRISM') else 1.0,
                color=colour(lab),
                alpha=1.0 if lab.startswith('PRISM') else 0.75, label=lab)
    ax.set_xlabel('$E(A,B)$   [ebits, crossings teleported]')
    ax.set_ylabel('$K(A,B)=\\sum\\log\\gamma$   [crossings knitted]')
    ax.set_title(title or 'CD/CK Pareto fronts', fontsize=10, loc='left')
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(**GRID)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
