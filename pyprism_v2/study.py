"""
pyprism_v2.study
================
Study writer: full results plus a directory of individual PDF figures.

The design is a *blocked* one --- every method partitions the same instances ---
so the comparisons are paired and the right tools are distribution-free paired
tests, not t-tests on independent samples.  All statistics are implemented here
in NumPy; SciPy is not a dependency.

  Omnibus      Friedman over all methods blocked by instance, with the Nemenyi
               post-hoc critical difference.
  Pairwise     Wilcoxon signed-rank, PRISM+ against each baseline, with
               Holm-Bonferroni control of the family-wise error rate.
  Effect size  Cliff's delta with the conventional magnitude thresholds.
  Robustness   Dolan-More performance profiles and bootstrap intervals.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict

import numpy as np

from .plotting import (PALETTE, colour, substrate_of, substrate_legend,
                       INK, GRID, C_H1, C_H2, C_PRISM, C_PRISM_PLUS,
                       C_NEUTRAL)

__all__ = ['write_study']


# ===========================================================================
#  statistics, NumPy only
# ===========================================================================
def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _gser(a, x, itmax=500, eps=3e-14):
    ap, s, d = a, 1.0 / a, 1.0 / a
    for _ in range(itmax):
        ap += 1.0
        d *= x / ap
        s += d
        if abs(d) < abs(s) * eps:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a, x, itmax=500, eps=3e-14):
    tiny = 1e-300
    b, c = x + 1.0 - a, 1.0 / tiny
    d = 1.0 / b if b != 0 else 1.0 / tiny
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x, df):
    """P(chi2_df > x) via the regularised incomplete gamma function."""
    if x <= 0:
        return 1.0
    a, xx = df / 2.0, x / 2.0
    return 1.0 - _gser(a, xx) if xx < a + 1.0 else _gcf(a, xx)


def rankdata(a):
    a = np.asarray(a, float)
    o = np.argsort(a, kind='mergesort')
    r = np.empty(len(a), float)
    r[o] = np.arange(1, len(a) + 1, dtype=float)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt))
    np.add.at(s, inv, r)
    return (s / cnt)[inv]


def wilcoxon(x, y):
    """Paired Wilcoxon with tie and continuity corrections."""
    d = np.asarray(x, float) - np.asarray(y, float)
    n_tie = int((d == 0).sum())
    nz = d[d != 0]
    n = len(nz)
    if n == 0:
        return dict(p=1.0, W=np.nan, z=0.0, n_pos=0, n_neg=0, n_tie=n_tie)
    r = rankdata(np.abs(nz))
    wp, wm = float(r[nz > 0].sum()), float(r[nz < 0].sum())
    W = min(wp, wm)
    mu = n * (n + 1) / 4.0
    _, cnt = np.unique(np.abs(nz), return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24.0 - float((cnt ** 3 - cnt).sum()) / 48.0
    if var <= 0:
        return dict(p=1.0, W=W, z=0.0, n_pos=int((nz > 0).sum()),
                    n_neg=int((nz < 0).sum()), n_tie=n_tie)
    z = (W - mu + 0.5) / math.sqrt(var)
    return dict(p=min(2.0 * norm_sf(abs(z)), 1.0), W=W, z=z,
                n_pos=int((nz > 0).sum()), n_neg=int((nz < 0).sum()),
                n_tie=n_tie)


def sign_test(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n, 1.0)


def friedman(mat):
    mat = np.asarray(mat, float)
    N, k = mat.shape
    R = np.vstack([rankdata(row) for row in mat])
    Rj = R.mean(0)
    chi = (12.0 * N / (k * (k + 1))) * (float((Rj ** 2).sum())
                                        - k * (k + 1) ** 2 / 4.0)
    return dict(chi2=chi, df=k - 1, p=chi2_sf(chi, k - 1), mean_ranks=Rj,
                N=N, k=k)


_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
        9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
        15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}


def nemenyi_cd(k, N):
    return _Q05.get(k, 3.6) * math.sqrt(k * (k + 1) / (6.0 * N))


def cliffs_delta(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = int((x[:, None] > y[None, :]).sum())
    lt = int((x[:, None] < y[None, :]).sum())
    d = (gt - lt) / float(len(x) * len(y))
    a = abs(d)
    return d, ('negligible' if a < 0.147 else 'small' if a < 0.330 else
               'medium' if a < 0.474 else 'large')


def bootstrap_ci(a, n_boot=4000, alpha=0.05, seed=0):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return (np.nan,) * 3
    rng = np.random.default_rng(seed)
    bs = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(1)
    return (float(a.mean()), float(np.quantile(bs, alpha / 2)),
            float(np.quantile(bs, 1 - alpha / 2)))


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj, run = np.empty(m), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def stars(p):
    return ('***' if p < 1e-3 else '**' if p < 1e-2 else
            '*' if p < 0.05 else 'n.s.')


# ===========================================================================
#  the study writer
# ===========================================================================
def write_study(df, meta, outdir, fronts_by_instance=None, champion='PRISM+',
                title='', verbose=True, latex=True):
    """Write every table and figure of the study into ``outdir``."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    figdir = os.path.join(outdir, 'figures')
    os.makedirs(figdir, exist_ok=True)
    made = []

    def save(fig, name):
        p = os.path.join(figdir, name)
        fig.savefig(p, bbox_inches='tight', dpi=150)
        plt.close(fig)
        made.append(p)
        if verbose:
            print(f'    {p}')

    df = df.copy()
    df['inst'] = (df['n'].astype(str) + '_' + df['k'].astype(str) + '_'
                  + df['seed'].astype(str))

    # A method that failed on *every* instance has all-NaN E_CD, so
    # `pivot_table` drops its column entirely -- while it survives in the
    # hypervolume pivot, because a failure still records hv = 0.  Analysing it
    # would compare against nothing, so it is reported and then excluded.
    n_runs = df.groupby('method').size()
    n_fail = df.groupby('method')['E_CD'].apply(lambda x: int(x.isna().sum()))
    dead = sorted(m for m in n_runs.index if n_fail[m] == n_runs[m])
    partial = sorted(m for m in n_runs.index
                     if 0 < n_fail[m] < n_runs[m])
    if dead and verbose:
        print(f'    excluding {len(dead)} method(s) with no successful run: '
              f'{", ".join(dead)}')
    df = df[~df.method.isin(dead)]
    if df.empty:
        raise RuntimeError('every method failed; nothing to plot')

    order = (df.groupby('method')['hv_norm'].mean()
             .sort_values(ascending=False).index.tolist())
    piv_hv = df.pivot_table(index='inst', columns='method', values='hv_norm')
    piv_E = df.pivot_table(index='inst', columns='method', values='E_CD')
    # only methods with data in *both* pivots can be compared
    order = [m for m in order if m in piv_hv.columns and m in piv_E.columns]
    complete = [m for m in order if piv_hv[m].notna().all()
                and piv_E[m].notna().all()]
    P = piv_hv[complete].dropna() if complete else piv_hv.iloc[:, :0]

    # ---- 01 front quality -------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.2))
    data = [df[df.method == m]['hv_norm'].dropna().values for m in order]
    parts = ax.violinplot(data, showextrema=False, widths=0.9)
    for b, m in zip(parts['bodies'], order):
        b.set_facecolor(colour(m))
        b.set_alpha(0.35)
    bp = ax.boxplot(data, widths=0.3, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.3))
    for patch, m in zip(bp['boxes'], order):
        patch.set_facecolor(colour(m))
        patch.set_edgecolor('#41474f')
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=55, ha='right', fontsize=7.5)
    ax.set_ylabel('hypervolume / best on that instance')
    ax.set_title('Front quality, normalised per instance', fontsize=11,
                 loc='left')
    ax.grid(axis='y', **GRID)
    substrate_legend(ax, loc='lower left')
    save(fig, 'fig_01_front_quality.pdf')

    # ---- 02 ebits relative to Naive --------------------------------------
    if 'Naive' in piv_E.columns:
        rel = piv_E.div(piv_E['Naive'], axis=0)
        lbl = [m for m in order if m in rel.columns and m != 'Naive']
        fig, ax = plt.subplots(figsize=(11, 5.2))
        bp = ax.boxplot([rel[m].dropna().values for m in lbl],
                        patch_artist=True, widths=0.62, showfliers=False,
                        medianprops=dict(color=INK, lw=1.3))
        for patch, m in zip(bp['boxes'], lbl):
            patch.set_facecolor(colour(m))
            patch.set_edgecolor('#41474f')
        ax.axhline(1.0, ls='--', color=INK, lw=1.0)
        ax.set_xticks(range(1, len(lbl) + 1))
        ax.set_xticklabels(lbl, rotation=55, ha='right', fontsize=7.5)
        ax.set_ylabel('$E$ / $E_{\\mathrm{Naive}}$')
        ax.set_title('Ebit cost relative to the trivial partition '
                     '(below 1 is better)', fontsize=11, loc='left')
        ax.grid(axis='y', **GRID)
        substrate_legend(ax, loc='upper left')
        save(fig, 'fig_02_ebits_relative.pdf')

    # ---- 03 runtime -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.0))
    med = [max(df[df.method == m]['seconds'].median(), 1e-6) for m in order]
    ax.bar(np.arange(len(order)), med, color=[colour(m) for m in order],
           edgecolor='#41474f', lw=0.5)
    ax.set_yscale('log')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=55, ha='right', fontsize=7.5)
    ax.set_ylabel('median wall-clock  [s, log]')
    ax.set_title('Cost of each method', fontsize=11, loc='left')
    ax.grid(axis='y', which='both', **GRID)
    save(fig, 'fig_03_runtime.pdf')

    # ---- 04 scaling with n ------------------------------------------------
    ns = sorted(df['n'].unique())
    if len(ns) > 1:
        fig, ax = plt.subplots(figsize=(8, 5.4))
        for m in order:
            sub = df[df.method == m].groupby('n')['E_CD'].mean()
            if sub.notna().sum() < 2:
                continue
            ax.plot(sub.index, sub.values, '-o', ms=4, color=colour(m),
                    lw=2.2 if m.startswith('PRISM') else 1.0,
                    alpha=1.0 if m.startswith('PRISM') else 0.7, label=m)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('$n$   [qubits]')
        ax.set_ylabel('mean $E$   [ebits]')
        ax.set_title('Scaling of the ebit cost with register size',
                     fontsize=11, loc='left')
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax.grid(which='both', **GRID)
        save(fig, 'fig_04_scaling_n.pdf')

    # ---- 05 scaling with depth -------------------------------------------
    ks = sorted(df['k'].unique())
    if len(ks) > 1:
        fig, ax = plt.subplots(figsize=(8, 5.4))
        for m in order:
            sub = df[df.method == m].groupby('k')['E_CD'].mean()
            if sub.notna().sum() < 2:
                continue
            ax.plot(sub.index, sub.values, '-o', ms=4, color=colour(m),
                    lw=2.2 if m.startswith('PRISM') else 1.0,
                    alpha=1.0 if m.startswith('PRISM') else 0.7, label=m)
        ax.set_xticks(ks)
        ax.set_xlabel('$k$   (depth $= k\\cdot n$)')
        ax.set_ylabel('mean $E$   [ebits]')
        ax.set_title('Scaling with circuit depth', fontsize=11, loc='left')
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax.grid(**GRID)
        save(fig, 'fig_05_scaling_depth.pdf')

    # ---- 06 win rate and mean rank ---------------------------------------
    wins, n_inst = defaultdict(float), 0
    for _, sub in df.groupby('inst'):
        n_inst += 1
        best = sub['hv'].max()
        w = sub[sub['hv'] >= best - 1e-9]['method'].tolist()
        for m in w:
            wins[m] += 1.0 / len(w)
    ranks = df.copy()
    ranks['rank'] = ranks.groupby('inst')['hv'].rank(ascending=False,
                                                     method='average')
    mean_rank = ranks.groupby('method')['rank'].mean()
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.bar(np.arange(len(order)), [wins.get(m, 0) / n_inst * 100 for m in order],
           color=[colour(m) for m in order], edgecolor='#41474f', lw=0.5)
    for i, m in enumerate(order):
        ax.text(i, wins.get(m, 0) / n_inst * 100 + 1,
                f'{mean_rank.get(m, np.nan):.1f}', ha='center', fontsize=7,
                color='#41474f')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=55, ha='right', fontsize=7.5)
    ax.set_ylabel('instances won on hypervolume  [%]')
    ax.set_title('Win rate   (label = mean rank, lower is better)',
                 fontsize=11, loc='left')
    ax.grid(axis='y', **GRID)
    substrate_legend(ax, loc='upper right')
    save(fig, 'fig_06_win_rank.pdf')

    # ---- 07 Friedman + Nemenyi critical difference ------------------------
    fr = cd = None
    if len(P) >= 3 and P.shape[1] >= 3:
        fr = friedman(-P.values)          # higher hv is better -> negate
        cd = nemenyi_cd(fr['k'], fr['N'])
        names = list(P.columns)
        idx = np.argsort(fr['mean_ranks'])
        rk = fr['mean_ranks'][idx]
        nm = [names[i] for i in idx]
        fig, ax = plt.subplots(figsize=(10, 0.42 * len(nm) + 3.0))
        lo, hi = math.floor(rk.min() - 0.5), math.ceil(rk.max() + 0.5)
        ax.set_xlim(hi, lo)
        ax.set_ylim(-0.55 * len(nm) - 1.6, 2.1)
        ax.hlines(0, lo, hi, color=INK, lw=1.2)
        for t in range(lo, hi + 1):
            ax.vlines(t, 0, 0.16, color=INK, lw=1.0)
            ax.text(t, 0.32, str(t), ha='center', fontsize=8, color=INK)
        for i, (name, r) in enumerate(zip(nm, rk)):
            y = -0.55 * (i + 1)
            right = i < len(nm) / 2
            xt = hi if right else lo
            ax.plot([r, r], [0, y], color=colour(name), lw=1.4)
            ax.plot([r, xt], [y, y], color=colour(name), lw=1.4)
            ax.text(xt, y, f'  {name} ({r:.2f})  ',
                    ha='right' if right else 'left', va='center', fontsize=8,
                    color=INK)
        y0, used = -0.55 * (len(nm) + 0.7), []
        for i in range(len(nm)):
            j = i
            while j + 1 < len(nm) and rk[j + 1] - rk[i] <= cd:
                j += 1
            if j > i and not any(a <= i and j <= b for a, b in used):
                used.append((i, j))
                ax.plot([rk[i] - 0.03, rk[j] + 0.03], [y0, y0], color=INK,
                        lw=3.2, solid_capstyle='round')
                y0 -= 0.20
        ax.plot([hi - 0.05, hi - 0.05 - cd], [1.5, 1.5], color=INK, lw=2.2)
        ax.text(hi - 0.05 - cd / 2, 1.7, f'CD = {cd:.2f}', ha='center',
                fontsize=8, color=INK)
        ax.axis('off')
        ax.set_title(f"Friedman $\\chi^2$={fr['chi2']:.1f}, df={fr['df']}, "
                     f"$p$={fr['p']:.2e}  ·  Nemenyi CD, $\\alpha$=0.05, "
                     f"$N$={fr['N']}", fontsize=10, loc='left')
        save(fig, 'fig_07_critical_difference.pdf')

    # ---- 08 paired tests, champion vs each baseline -----------------------
    tests = []
    if champion in piv_hv.columns and champion in piv_E.columns:
        raw = []
        others = [m for m in order if m != champion and m in piv_hv.columns
                  and m in piv_E.columns]
        for m in others:
            both = piv_hv[[champion, m]].dropna()
            if len(both) < 3:
                continue
            w = wilcoxon(both[champion].values, both[m].values)
            d, mag = cliffs_delta(both[champion].values, both[m].values)
            bE = piv_E[[champion, m]].dropna()
            imp = 100 * (bE[m] - bE[champion]) / bE[m].replace(0, np.nan)
            mu, lo, hi = bootstrap_ci(imp.values)
            tests.append({'baseline': m, 'n_pairs': len(both),
                          'wins': w['n_pos'], 'ties': w['n_tie'],
                          'losses': w['n_neg'], 'wilcoxon_p': w['p'],
                          'sign_p': sign_test(both[champion].values,
                                              both[m].values),
                          'cliffs_delta': d, 'cliffs_mag': mag,
                          'mean_E_reduction_pct': mu, 'ci_lo': lo, 'ci_hi': hi})
            raw.append(w['p'])
        if tests:
            for t, a in zip(tests, holm(raw)):
                t['wilcoxon_p_holm'] = a
                t['significant_005'] = bool(a < 0.05)
            td = pd.DataFrame(tests)
            fig, ax = plt.subplots(figsize=(9, 0.42 * len(td) + 2.2))
            y = np.arange(len(td))
            ax.barh(y, td['cliffs_delta'],
                    color=[colour(m) for m in td['baseline']],
                    edgecolor='#41474f', lw=0.5, height=0.62)
            for thr, lab in ((0.147, 'small'), (0.330, 'medium'),
                             (0.474, 'large')):
                ax.axvline(thr, ls=':', lw=1.0, color='#7d8790')
                ax.axvline(-thr, ls=':', lw=1.0, color='#7d8790')
            for i, r in td.iterrows():
                ax.text(r['cliffs_delta'] + (0.02 if r['cliffs_delta'] >= 0
                                             else -0.02), i,
                        f"  {r['cliffs_delta']:+.2f} ({r['cliffs_mag']}) "
                        f"{stars(r['wilcoxon_p_holm'])}",
                        va='center', ha='left' if r['cliffs_delta'] >= 0
                        else 'right', fontsize=7.5, color=INK)
            ax.axvline(0, color=INK, lw=1.0)
            ax.set_yticks(y)
            ax.set_yticklabels(td['baseline'], fontsize=8)
            ax.set_xlabel(f"Cliff's $\\delta$   (positive = {champion} better)")
            ax.set_xlim(-1.15, 1.15)
            ax.set_title(f'Effect size of {champion} against each baseline\n'
                         'stars: Holm-adjusted Wilcoxon signed-rank',
                         fontsize=10, loc='left')
            ax.grid(axis='x', **GRID)
            save(fig, 'fig_08_effect_size.pdf')

            fig, ax = plt.subplots(figsize=(9, 0.42 * len(td) + 2.2))
            ax.barh(y, td['wins'], color=C_PRISM_PLUS, edgecolor='white',
                    label=f'{champion} wins')
            ax.barh(y, td['ties'], left=td['wins'], color=C_NEUTRAL,
                    edgecolor='white', label='tie')
            ax.barh(y, td['losses'], left=td['wins'] + td['ties'],
                    color=C_H1, edgecolor='white', label=f'{champion} loses')
            ax.set_yticks(y)
            ax.set_yticklabels(td['baseline'], fontsize=8)
            ax.set_xlabel('instances')
            ax.set_title(f'Win / tie / loss on hypervolume, {champion} vs each '
                         'baseline', fontsize=10, loc='left')
            ax.legend(fontsize=8, frameon=False, ncol=3,
                      loc='upper center', bbox_to_anchor=(0.5, -0.12))
            ax.grid(axis='x', **GRID)
            save(fig, 'fig_09_win_tie_loss.pdf')

    # ---- 10 performance profile ------------------------------------------
    if len(P) >= 2 and complete:
        C = piv_E[complete].dropna().values.astype(float)
        if C.size and C.shape[1] >= 2:
            best = C.min(axis=1, keepdims=True)
            ratio = C / np.maximum(best, 1e-9)
            taus = np.linspace(1.0, float(np.nanmax(ratio)) * 1.02, 300)
            fig, ax = plt.subplots(figsize=(8, 5.4))
            for j, m in enumerate(complete):
                ax.step(taus, [(ratio[:, j] <= t).mean() for t in taus],
                        where='post', color=colour(m),
                        lw=2.2 if m.startswith('PRISM') else 1.0,
                        alpha=1.0 if m.startswith('PRISM') else 0.7, label=m)
            ax.set_xlabel('$\\tau$   (factor of the best method on that instance)')
            ax.set_ylabel('$\\rho(\\tau)$   fraction of instances')
            ax.set_title('Dolan–Moré performance profile on $E$   '
                         '(higher and further left is better)',
                         fontsize=11, loc='left')
            ax.legend(fontsize=7, frameon=False, ncol=2, loc='lower right')
            ax.grid(**GRID)
            save(fig, 'fig_10_performance_profile.pdf')

    # ---- 11 quality vs cost ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.4))
    for m in order:
        sub = df[df.method == m]
        x = max(sub['seconds'].median(), 1e-6)
        y = sub['hv_norm'].mean()
        ax.scatter(x, y, s=150, color=colour(m), edgecolors='white', lw=1.2,
                   zorder=3)
        ax.annotate(m, (x, y), textcoords='offset points', xytext=(7, 4),
                    fontsize=7.5, color=INK)
    ax.set_xscale('log')
    ax.set_xlabel('median wall-clock  [s, log]')
    ax.set_ylabel('mean normalised hypervolume')
    ax.set_title('Quality against cost   (upper-left is the frontier)',
                 fontsize=11, loc='left')
    ax.grid(which='both', **GRID)
    save(fig, 'fig_11_quality_vs_cost.pdf')

    # ---- 12 heat map, method x instance -----------------------------------
    if len(P) >= 2 and complete:
        fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(P) + 3),
                                        0.32 * len(complete) + 2.4))
        im = ax.imshow(P[complete].T.values, aspect='auto', cmap='RdYlGn',
                       vmin=0, vmax=1)
        ax.set_yticks(range(len(complete)))
        ax.set_yticklabels(complete, fontsize=7.5)
        ax.set_xticks(range(len(P)))
        ax.set_xticklabels(P.index, rotation=90, fontsize=6.5)
        ax.set_xlabel('instance  (n_k_seed)')
        plt.colorbar(im, ax=ax, label='normalised hypervolume')
        ax.set_title('Per-instance front quality', fontsize=11, loc='left')
        ax.grid(False)
        save(fig, 'fig_12_heatmap.pdf')

    # ---- 13 substrate: compression ----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].hist(meta['compression'], bins=max(8, len(meta) // 2),
                 color=C_PRISM, edgecolor='#41474f', lw=0.5)
    mu = meta['compression'].mean()
    axes[0].axvline(mu, color=C_H1, lw=1.6)
    axes[0].set_xlabel('$|E_1| / |E_2|$')
    axes[0].set_ylabel('instances')
    axes[0].set_title(f'(a)  Net compression from packing (mean {mu:.2f}×)',
                      fontsize=10, loc='left')
    axes[0].grid(axis='y', **GRID)
    for kk in sorted(meta['k'].unique()):
        sub = meta[meta.k == kk].sort_values('n')
        axes[1].plot(sub['n'], sub['H1_nets'], '--o', ms=4, color=C_H1,
                     alpha=0.75, label=f'$H_1$, k={kk}')
        axes[1].plot(sub['n'], sub['H2_nets'], '-o', ms=4, color=C_H2,
                     label=f'$H_2$, k={kk}')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('$n$')
    axes[1].set_ylabel('nets')
    axes[1].set_title('(b)  Substrate size  (dashed $H_1$, solid $H_2$)',
                      fontsize=10, loc='left')
    axes[1].legend(fontsize=7.5, frameon=False)
    axes[1].grid(which='both', **GRID)
    fig.tight_layout()
    save(fig, 'fig_13_substrate.pdf')

    # ---- 14 light cone ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    sc = axes[0].scatter(meta['n'], meta['lc_scrambling_depth'],
                         c=meta['k'], cmap='viridis', s=50)
    plt.colorbar(sc, ax=axes[0], label='$k$')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('$n$')
    axes[0].set_ylabel('scrambling depth')
    axes[0].set_title('(a)  Depth at which the causal cone covers $Q$',
                      fontsize=10, loc='left')
    axes[0].grid(which='both', **GRID)
    axes[1].scatter(meta['lc_mean_coupling'], meta['compression'],
                    c=meta['n'], cmap='plasma', s=50)
    axes[1].set_xlabel('mean light-cone coupling')
    axes[1].set_ylabel('net compression')
    axes[1].set_title('(b)  Causal coupling vs packing', fontsize=10,
                      loc='left')
    axes[1].grid(**GRID)
    fig.tight_layout()
    save(fig, 'fig_14_lightcone.pdf')

    # ---- 15 PRISM vs PRISM+ ----------------------------------------------
    if 'PRISM' in piv_E.columns and 'PRISM+' in piv_E.columns:
        both = piv_E[['PRISM', 'PRISM+']].dropna()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
        lo = float(min(both.min())) * 0.9
        hi = float(max(both.max())) * 1.1
        axes[0].plot([lo, hi], [lo, hi], '--', color=INK, lw=1.0)
        axes[0].scatter(both['PRISM'], both['PRISM+'], s=45,
                        color=PALETTE['PRISM+'], alpha=0.8, edgecolors='none')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_xlabel('PRISM  $E$   (random seeding)')
        axes[0].set_ylabel('PRISM+  $E$   (METIS/KaHyPar multi-start)')
        w = wilcoxon(both['PRISM+'].values, both['PRISM'].values)
        axes[0].set_title('(a)  Does multi-start seeding help?\n'
                          f"W/T/L = {w['n_neg']}/{w['n_tie']}/{w['n_pos']}, "
                          f"Wilcoxon $p$={w['p']:.3f} {stars(w['p'])}",
                          fontsize=10, loc='left')
        axes[0].grid(which='both', **GRID)
        gain = 100 * (both['PRISM'] - both['PRISM+']) / both['PRISM']
        idx = piv_E.loc[both.index]
        nn = [int(i.split('_')[0]) for i in both.index]
        axes[1].scatter(nn, gain, s=50, color=PALETTE['PRISM+'], alpha=0.85,
                        edgecolors='none')
        axes[1].axhline(0, color=INK, lw=1.0)
        axes[1].set_xscale('log')
        axes[1].set_xlabel('$n$')
        axes[1].set_ylabel('PRISM+ improvement over PRISM  [%]')
        axes[1].set_title('(b)  Seeding pays off as the register grows',
                          fontsize=10, loc='left')
        axes[1].grid(which='both', **GRID)
        fig.tight_layout()
        save(fig, 'fig_15_prism_vs_prismplus.pdf')

    # ---- 16 front richness and objective correlation ----------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fs = [df[df.method == m]['front_size'].dropna().values for m in order]
    bp = axes[0].boxplot(fs, patch_artist=True, widths=0.6, showfliers=False,
                         medianprops=dict(color=INK, lw=1.2))
    for patch, m in zip(bp['boxes'], order):
        patch.set_facecolor(colour(m))
        patch.set_edgecolor('#41474f')
    axes[0].set_xticks(range(1, len(order) + 1))
    axes[0].set_xticklabels(order, rotation=60, ha='right', fontsize=6.5)
    axes[0].set_ylabel('non-dominated points')
    axes[0].set_title('(a)  Front richness', fontsize=10, loc='left')
    axes[0].grid(axis='y', **GRID)
    axes[1].scatter(meta['n'], meta['spearman_H2'], s=50, color=C_H2,
                    label='$H_2$')
    axes[1].scatter(meta['n'], meta['spearman_H1'], s=30, color=C_H1,
                    alpha=0.75, label='$H_1$')
    axes[1].set_xscale('log')
    axes[1].set_ylim(0.9, 1.005)
    axes[1].set_xlabel('$n$')
    axes[1].set_ylabel('Spearman $\\rho(E_{CD}, K_{CK})$')
    axes[1].set_title('(b)  The currencies are collinear as totals —\n'
                      'the front comes from the mode choice', fontsize=10,
                      loc='left')
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].grid(which='both', **GRID)
    fig.tight_layout()
    save(fig, 'fig_16_front_and_correlation.pdf')

    # ---- 17 example CD/CK fronts -----------------------------------------
    if fronts_by_instance:
        for key, fronts in list(fronts_by_instance.items())[:3]:
            fig, ax = plt.subplots(figsize=(8, 5.6))
            for lab, f in fronts.items():
                if not f:
                    continue
                E = np.array([p[0] for p in f], float)
                K = np.array([p[1] for p in f], float)
                o = np.argsort(E)
                ax.plot(E[o], K[o], '-', color=colour(lab),
                        lw=2.2 if lab.startswith('PRISM') else 0.9,
                        alpha=1.0 if lab.startswith('PRISM') else 0.6,
                        label=lab)
            ax.set_xlabel('$E(A,B)$   [ebits, crossings teleported]')
            ax.set_ylabel('$K(A,B)=\\sum\\log\\gamma$   [crossings knitted]')
            ax.set_title(f'CD/CK Pareto fronts — instance {key}', fontsize=11,
                         loc='left')
            ax.legend(fontsize=6.5, frameon=False, ncol=2)
            ax.grid(**GRID)
            save(fig, f'fig_17_fronts_{key}.pdf')

    # ---- 18 the price of being symmetry-blind, H1 against H2 --------------
    # Every method that runs on both substrates is the same algorithm twice,
    # so the H1 -> H2 difference isolates the substrate and nothing else.
    pairs = [m[:-3] for m in piv_E.columns if m.endswith('/H1')
             and m[:-3] + '/H2' in piv_E.columns]
    sub_tests = []
    if pairs:
        for b in pairs:
            both = piv_E[[b + '/H1', b + '/H2']].dropna()
            if len(both) < 3:
                continue
            w = wilcoxon(both[b + '/H2'].values, both[b + '/H1'].values)
            red = 100 * (both[b + '/H1'] - both[b + '/H2']) / both[b + '/H1']
            mu, lo, hi = bootstrap_ci(red.values)
            sub_tests.append({'method': b, 'n_pairs': len(both),
                              'E_H1': float(both[b + '/H1'].mean()),
                              'E_H2': float(both[b + '/H2'].mean()),
                              'reduction_pct': mu, 'ci_lo': lo, 'ci_hi': hi,
                              'wilcoxon_p': w['p'],
                              'n_better': w['n_neg'], 'n_tie': w['n_tie'],
                              'n_worse': w['n_pos']})
        if sub_tests:
            sub_tests.sort(key=lambda r: -r['reduction_pct'])
            st = pd.DataFrame(sub_tests)
            fig, axes = plt.subplots(
                1, 2, figsize=(13, 0.44 * len(st) + 2.6),
                gridspec_kw=dict(width_ratios=[1.25, 1.0]))
            y = np.arange(len(st))
            axes[0].hlines(y, st['E_H2'], st['E_H1'], color='#b6bdc6', lw=2.0,
                           zorder=1)
            axes[0].scatter(st['E_H1'], y, s=90, color=C_H1, zorder=3,
                            edgecolors='white', lw=1.0, label='$H_1$')
            axes[0].scatter(st['E_H2'], y, s=90, color=C_H2, zorder=3,
                            edgecolors='white', lw=1.0, label='$H_2$')
            axes[0].set_yticks(y)
            axes[0].set_yticklabels(st['method'], fontsize=8.5)
            axes[0].invert_yaxis()
            axes[0].set_xlabel('mean $E$   [ebits]')
            axes[0].set_title('(a)  The same algorithm on each substrate',
                              fontsize=10, loc='left')
            axes[0].legend(fontsize=8, frameon=False, ncol=2)
            axes[0].grid(axis='x', **GRID)
            axes[1].barh(y, st['reduction_pct'],
                         xerr=[st['reduction_pct'] - st['ci_lo'],
                               st['ci_hi'] - st['reduction_pct']],
                         color=C_H2, edgecolor='#41474f', lw=0.5, height=0.6,
                         error_kw=dict(ecolor=INK, lw=1.0, capsize=2.5))
            for i, r in st.iterrows():
                axes[1].text(max(r['ci_hi'], r['reduction_pct']) + 0.6, i,
                             f"{r['reduction_pct']:+.1f}\\%  "
                             f"{stars(r['wilcoxon_p'])}",
                             va='center', fontsize=7.5, color=INK)
            axes[1].axvline(0, color=INK, lw=1.0)
            axes[1].set_yticks(y)
            axes[1].set_yticklabels([])
            axes[1].invert_yaxis()
            axes[1].set_xlabel('ebits saved by $H_2$   [%, 95\\% bootstrap CI]')
            axes[1].set_title('(b)  What the symmetry is worth', fontsize=10,
                              loc='left')
            axes[1].grid(axis='x', **GRID)
            fig.tight_layout()
            save(fig, 'fig_18_substrate_ablation.pdf')
            st.to_csv(os.path.join(outdir, 'substrate_ablation.csv'),
                      index=False)

    # ===== tables ==========================================================
    summ = df.groupby('method').agg(
        hv_norm=('hv_norm', 'mean'), E_mean=('E_CD', 'mean'),
        E_median=('E_CD', 'median'), K_mean=('K_CK', 'mean'),
        front=('front_size', 'mean'), sec=('seconds', 'mean'),
        fails=('E_CD', lambda s: int(s.isna().sum()))).reindex(order)
    summ['win_rate'] = [wins.get(m, 0.0) / n_inst for m in summ.index]
    summ['mean_rank'] = mean_rank.reindex(summ.index)
    summ.to_csv(os.path.join(outdir, 'summary.csv'))
    df.drop(columns=['inst']).to_csv(os.path.join(outdir, 'raw.csv'),
                                     index=False)
    meta.to_csv(os.path.join(outdir, 'instances.csv'), index=False)
    piv_E.to_csv(os.path.join(outdir, 'pivot_E.csv'))
    piv_hv.to_csv(os.path.join(outdir, 'pivot_hv.csv'))
    if tests:
        pd.DataFrame(tests).to_csv(os.path.join(outdir, 'stats_tests.csv'),
                                   index=False)

    # ---- LaTeX ------------------------------------------------------------
    tex = {}
    if latex:
        from .latex import write_latex
        if verbose:
            print(f'  tables into {outdir}/tables/')
        tex = write_latex(outdir, df, meta, order, wins, mean_rank, n_inst,
                          tests=tests, sub_tests=sub_tests, fr=fr, cd=cd,
                          champion=champion, verbose=verbose)

    with open(os.path.join(outdir, 'REPORT.md'), 'w') as f:
        f.write(f'# {title or "pyprism_v2 study"}\n\n')
        f.write(f'{n_inst} instances, {len(order)} methods, blocked paired '
                f'design.\n\n')
        if dead:
            f.write(f'**Excluded** (no successful run): '
                    f'{", ".join("`" + m + "`" for m in dead)}\n\n')
        if partial:
            f.write(f'**Partial failures** (some instances only): '
                    f'{", ".join("`" + m + "`" for m in partial)}\n\n')
        f.write('## Summary\n\n')
        f.write(summ.to_markdown(floatfmt='.3f') + '\n')
        if fr:
            f.write(f"\n## Omnibus\n\nFriedman chi2 = {fr['chi2']:.2f}, "
                    f"df = {fr['df']}, p = {fr['p']:.3e}; "
                    f"Nemenyi CD = {cd:.3f} at alpha = 0.05, N = {fr['N']}.\n")
        if tests:
            f.write(f'\n## {champion} against each baseline\n\n')
            f.write(pd.DataFrame(tests).to_markdown(index=False,
                                                    floatfmt='.4g') + '\n')
        f.write('\n## Figures\n\n')
        for p in made:
            f.write(f'- `figures/{os.path.basename(p)}`\n')

    with open(os.path.join(outdir, 'study.json'), 'w') as f:
        json.dump({'n_instances': int(n_inst), 'methods': order,
                   'excluded_no_successful_run': dead,
                   'partial_failures': partial,
                   'friedman': ({k: (float(v) if not isinstance(v, np.ndarray)
                                     else v.tolist())
                                 for k, v in fr.items()} if fr else None),
                   'nemenyi_cd': float(cd) if cd else None,
                   'tests': tests,
                   'figures': [os.path.basename(p) for p in made]},
                  f, indent=1, default=float)
    return summ, made
