"""
pyprism_v2.latex
================
Paper-grade LaTeX tables for the study.

Every table is a self-contained ``table`` float using ``booktabs`` rules only,
so it can be dropped straight into a manuscript.  Colour is optional and
opt-in: each file declares

    \\providecommand{\\prismsub}[2]{#2}

as a no-op, and ``prism_colours.tex`` renews it to tint the method name with
the same hue the figures use --- red for the symmetry-blind substrate
:math:`H_1`, blue for the symmetry-aware :math:`H_2`, teal for PRISM.  Input it
or don't; the tables compile either way, and in either order.

A note on aggregation
---------------------
``E`` and ``K`` are extensive: they grow with the register, so a mean taken
across a sweep of ``n`` is dominated by the spread of ``n`` rather than by any
difference between methods.  The performance table therefore reports the raw
means -- which is what was asked for and what a reader expects to see -- beside
``E / E_naive``, which is scale-free and is the column the comparison should
actually be read from.  ``table_scaling`` breaks the raw numbers out per ``n``
so nothing is hidden behind the aggregate.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .plotting import C_H1, C_H2, C_PRISM_PLUS, substrate_of, base_name

__all__ = ['write_latex', 'table_performance', 'table_tests', 'table_scaling',
           'table_substrate', 'table_ablation', 'pval_stars']


# ---------------------------------------------------------------------------
#  formatting
# ---------------------------------------------------------------------------
_SPECIAL = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
            '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}'}


def esc(s):
    """Escape TeX specials.  Backslash first, or the escapes get re-escaped."""
    s = str(s).replace('\\', r'\textbackslash{}')
    for a, b in _SPECIAL.items():
        s = s.replace(a, b)
    return s


def _prec(x):
    """Decimals to show, from the magnitude -- three significant figures."""
    a = abs(float(x))
    if a >= 1000:
        return 0
    if a >= 100:
        return 0
    if a >= 10:
        return 1
    if a >= 1:
        return 2
    return 3


def num(x, dec=None):
    if x is None or (isinstance(x, float) and x != x):
        return '--'
    x = float(x)
    d = _prec(x) if dec is None else dec
    s = f'{x:,.{d}f}'.replace(',', r'\,')
    return f'${s}$'


def pm(m, s, dec=None):
    """``mean ± std`` at a shared precision, so the pair reads as one number."""
    if m is None or (isinstance(m, float) and m != m):
        return '--'
    d = _prec(m) if dec is None else dec
    a = f'{float(m):,.{d}f}'.replace(',', r'\,')
    if s is None or (isinstance(s, float) and s != s):
        return f'${a}$'
    b = f'{float(s):,.{d}f}'.replace(',', r'\,')
    return f'${a} \\pm {b}$'


def pval(p):
    """A p-value the way a journal wants it: three decimals until it stops
    being informative, then a power of ten."""
    if p is None or (isinstance(p, float) and p != p):
        return '--'
    p = float(p)
    if p >= 1e-3:
        return f'${p:.3f}$'
    if p < 1e-300:
        return r'$<10^{-300}$'
    e = math.floor(math.log10(p))
    m = p / 10.0 ** e
    if round(m, 1) >= 10.0:
        m, e = m / 10.0, e + 1
    return f'${m:.1f}\\times 10^{{{e}}}$'


def stars(p):
    """Significance marks, as a superscript *inside* a math group."""
    if p is None or (isinstance(p, float) and p != p):
        return ''
    return ('{}^{***}' if p < 1e-3 else '{}^{**}' if p < 1e-2 else
            '{}^{*}' if p < 0.05 else '')


def pval_stars(p):
    """``pval`` with the stars folded in, rather than set as a second math
    group butted against the first."""
    s, m = pval(p), stars(p)
    return s[:-1] + m + '$' if (m and s.endswith('$')) else s


def bold(s, on=True):
    return f'\\textbf{{{s}}}' if on else s


def _sub_tag(label):
    s = substrate_of(label)
    return 'PRISM' if s is None else s


def tag(label, show=None):
    """The method name, wrapped so the optional colour file can tint it."""
    return f'\\prismsub{{{_sub_tag(label)}}}{{{esc(show or label)}}}'


_PREAMBLE = (r'% requires \usepackage{booktabs}.  Optional: \input the '
             r'companion prism_colours.tex for the figure hues.' '\n'
             r'\providecommand{\prismsub}[2]{#2}' '\n')


def _float(body, caption, label, spec, header, placement='t',
           size=r'\small', note=None, star=False):
    env = 'table*' if star else 'table'
    out = [_PREAMBLE,
           f'\\begin{{{env}}}[{placement}]', r'  \centering', f'  {size}',
           f'  \\caption{{{caption}}}', f'  \\label{{{label}}}',
           f'  \\begin{{tabular}}{{{spec}}}', r'    \toprule',
           header, r'    \midrule']
    out += body
    out += [r'    \bottomrule', r'  \end{tabular}']
    if note:
        out += [r'  \begin{minipage}{\linewidth}\vspace{2pt}\footnotesize',
                f'  {note}', r'  \end{minipage}']
    out += [f'\\end{{{env}}}', '']
    return '\n'.join(out)


# ---------------------------------------------------------------------------
#  tables
# ---------------------------------------------------------------------------
def _grouped(order):
    """Split the method labels into H1, H2 and PRISM, preserving rank order."""
    g = {'H1': [], 'H2': [], 'PRISM': []}
    for m in order:
        g[_sub_tag(m)].append(m)
    return g


def table_performance(df, order, wins, mean_rank, n_inst,
                      label='tab:prism-performance', caption=None):
    """Per-method E, K and hypervolume as mean ± std, grouped by substrate."""
    import pandas as pd

    g = df.groupby('method')
    piv_E = df.pivot_table(index='inst', columns='method', values='E_CD')
    rel = (piv_E.div(piv_E['Naive'], axis=0) if 'Naive' in piv_E.columns
           else piv_E * np.nan)

    rows = {}
    for m in order:
        s = g.get_group(m) if m in g.groups else None
        if s is None:
            continue
        r = rel[m] if m in rel.columns else pd.Series(dtype=float)
        rows[m] = dict(
            E=s['E_CD'].mean(), E_sd=s['E_CD'].std(),
            R=r.mean(), R_sd=r.std(),
            K=s['K_CK'].mean(), K_sd=s['K_CK'].std(),
            HV=s['hv_norm'].mean(), HV_sd=s['hv_norm'].std(),
            F=s['front_size'].mean(),
            rank=float(mean_rank.get(m, np.nan)),
            win=100.0 * wins.get(m, 0.0) / max(n_inst, 1),
            sec=s['seconds'].median())

    def best(key, lower=True):
        v = {m: rows[m][key] for m in rows if rows[m][key] == rows[m][key]}
        if not v:
            return None
        return (min if lower else max)(v, key=v.get)

    bE, bR, bK = best('E'), best('R'), best('K')
    bHV, bRk, bW = best('HV', False), best('rank'), best('win', False)

    body, grp = [], _grouped(list(rows))
    titles = {'H1': r'\emph{Symmetry-blind substrate} $H_1$',
              'H2': r'\emph{Symmetry-aware substrate} $H_2$',
              'PRISM': r'\emph{Proposed}'}
    first = True
    for key in ('H1', 'H2', 'PRISM'):
        if not grp[key]:
            continue
        if not first:
            body.append(r'    \addlinespace[3pt]')
        first = False
        body.append(f'    \\multicolumn{{8}}{{l}}{{{titles[key]}}} \\\\')
        for m in grp[key]:
            d = rows[m]
            show = base_name(m) if key in ('H1', 'H2') else m
            body.append(
                '    ' + ' & '.join([
                    tag(m, show),
                    bold(pm(d['E'], d['E_sd']), m == bE),
                    bold(pm(d['R'], d['R_sd'], 3), m == bR),
                    bold(pm(d['K'], d['K_sd']), m == bK),
                    bold(pm(d['HV'], d['HV_sd'], 3), m == bHV),
                    bold(num(d['rank'], 2), m == bRk),
                    bold(num(d['win'], 1), m == bW),
                    num(d['sec'], 2),
                ]) + r' \\')

    header = ('    Method & $E$ [ebits] & $E/E_{\\mathrm{naive}}$ & '
              '$K=\\sum\\log\\gamma$ & HV & rank & win [\\%] & $t$ [s] \\\\')
    cap = caption or (
        f'Partitioning performance over {n_inst} random circuit instances, '
        'mean $\\pm$ standard deviation. Every method is \\emph{scored} on the '
        'exact Track~2 cost whatever substrate it optimised on, so the columns '
        'are directly comparable. Best in each column in bold.')
    note = (r'$E$ and $K$ are extensive, so their means aggregate across '
            r'register sizes; $E/E_{\mathrm{naive}}$ and the per-instance '
            r'normalised hypervolume~HV are the scale-free columns, and '
            r'Table~\ref{tab:prism-scaling} breaks $E$ out by $n$. '
            r'Rank is the mean rank on HV (lower is better), win the share of '
            r'instances on which the method attains the best hypervolume, '
            r'$t$ the median wall-clock time.')
    return _float(body, cap, label, 'l ccccrrr', header, note=note, star=True)


def table_tests(tests, champion, fr=None, cd=None,
                label='tab:prism-tests', caption=None):
    """Paired tests of the champion against every baseline."""
    if not tests:
        return ''
    body = []
    best_d = max(t['cliffs_delta'] for t in tests)
    for t in tests:
        wtl = f"{t['wins']}/{t['ties']}/{t['losses']}"
        ci = (f"$[{t['ci_lo']:.1f},\\,{t['ci_hi']:.1f}]$"
              if t['ci_lo'] == t['ci_lo'] else '--')
        body.append('    ' + ' & '.join([
            tag(t['baseline']),
            f"${t['n_pairs']}$",
            esc(wtl),
            bold(f"${t['cliffs_delta']:+.2f}$",
                 t['cliffs_delta'] == best_d) + f" ({t['cliffs_mag']})",
            pval(t['wilcoxon_p']),
            pval_stars(t.get('wilcoxon_p_holm')),
            pval(t['sign_p']),
            num(t['mean_E_reduction_pct'], 1),
            ci,
        ]) + r' \\')
    header = ('    Baseline & $N$ & W/T/L & Cliff\'s $\\delta$ & '
              '$p_{\\mathrm{Wilcoxon}}$ & $p_{\\mathrm{Holm}}$ & '
              '$p_{\\mathrm{sign}}$ & $\\Delta E$ [\\%] & 95\\% CI \\\\')
    om = ''
    if fr:
        om = (f" Friedman $\\chi^2={fr['chi2']:.1f}$, "
              f"$\\mathrm{{df}}={fr['df']}$, $p={pval(fr['p']).strip('$')}$ "
              f"over {fr['k']} methods and $N={fr['N']}$ blocks rejects the "
              f"hypothesis of equal performance"
              + (f"; the Nemenyi critical difference is ${cd:.2f}$ at "
                 f"$\\alpha=0.05$." if cd else '.'))
    cap = caption or (
        f'{esc(champion)} against each baseline on normalised hypervolume: '
        'paired, distribution-free tests over the same instances.' + om)
    note = (r'W/T/L counts instances on which ' + esc(champion) +
            r' wins, ties and loses. $p_{\mathrm{Holm}}$ is the '
            r'Holm--Bonferroni adjustment controlling the family-wise error '
            r'rate across the ' + str(len(tests)) + r' comparisons; '
            r'$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$. '
            r"Cliff's $\delta$ is reported with the conventional magnitude "
            r'thresholds ($0.147$, $0.330$, $0.474$). $\Delta E$ is the mean '
            r'reduction in ebits with a $4000$-resample bootstrap interval.')
    return _float(body, cap, label, 'l r c l rrr rc', header,
                  size=r'\footnotesize', note=note, star=True)


def table_scaling(df, order, label='tab:prism-scaling', caption=None):
    """Mean ± std ebit cost per register size, so nothing hides in the
    aggregate of Table~\\ref{tab:prism-performance}."""
    ns = sorted(df['n'].unique())
    if len(ns) < 2:
        return ''
    piv = df.pivot_table(index='method', columns='n', values='E_CD',
                         aggfunc='mean')
    sd = df.pivot_table(index='method', columns='n', values='E_CD',
                        aggfunc='std')
    best = {n: piv[n].idxmin() for n in ns if piv[n].notna().any()}
    body, grp = [], _grouped([m for m in order if m in piv.index])
    titles = {'H1': r'\emph{Symmetry-blind substrate} $H_1$',
              'H2': r'\emph{Symmetry-aware substrate} $H_2$',
              'PRISM': r'\emph{Proposed}'}
    first = True
    for key in ('H1', 'H2', 'PRISM'):
        if not grp[key]:
            continue
        if not first:
            body.append(r'    \addlinespace[3pt]')
        first = False
        body.append(f'    \\multicolumn{{{len(ns) + 1}}}{{l}}'
                    f'{{{titles[key]}}} \\\\')
        for m in grp[key]:
            show = base_name(m) if key in ('H1', 'H2') else m
            cells = [bold(pm(piv.loc[m, n], sd.loc[m, n] if m in sd.index
                             else np.nan), best.get(n) == m) for n in ns]
            body.append('    ' + ' & '.join([tag(m, show)] + cells) + r' \\')
    header = ('    Method & ' + ' & '.join(f'$n={n}$' for n in ns) + r' \\')
    cap = caption or (
        'Ebit cost $E$ by register size, mean $\\pm$ standard deviation over '
        'the depth multipliers $k$ (depth $=k\\cdot n$). Best per column in '
        'bold.')
    return _float(body, cap, label, 'l' + 'c' * len(ns), header,
                  size=r'\footnotesize', star=True)


def table_substrate(meta, label='tab:prism-substrate', caption=None):
    """What symmetry and irreducibility do to the hypergraph itself."""
    m = meta.sort_values(['n', 'k'])
    g = m.groupby(['n', 'k']).agg(
        gates=('gates', 'mean'), h1=('H1_nets', 'mean'),
        h2=('H2_nets', 'mean'), pk=('packets', 'mean'),
        pg=('packed_gates', 'mean'), cx=('compression', 'mean'),
        sd=('lc_scrambling_depth', 'mean'), rho=('spearman_H2', 'mean'))
    body = []
    for (n, k), r in g.iterrows():
        body.append('    ' + ' & '.join([
            f'${n}$', f'${k}$', num(r['gates'], 0),
            f"\\prismsub{{H1}}{{{num(r['h1'], 0)}}}",
            f"\\prismsub{{H2}}{{{num(r['h2'], 0)}}}",
            f"${r['cx']:.2f}\\times$",
            num(r['pk'], 0), num(r['pg'], 0),
            num(r['sd'], 1), num(r['rho'], 3),
        ]) + r' \\')
    header = ('    $n$ & $k$ & gates & $|\\mathcal{E}_1|$ & '
              '$|\\mathcal{E}_2|$ & ratio & packets & packed & $d_{\\ast}$ & '
              '$\\rho$ \\\\')
    cap = caption or (
        'The substrate itself. $|\\mathcal{E}_1|$ counts one net per '
        'multi-qubit gate; $|\\mathcal{E}_2|$ counts nets after gates sharing '
        'an on-site symmetric root are absorbed into distributable packets.')
    note = (r'$d_{\ast}$ is the scrambling depth, the layer at which the '
            r'causal cone of a qubit covers $Q$. $\rho$ is the Spearman '
            r'correlation between $E$ and $K$ over random bipartitions of '
            r'$H_2$: near unity, which is why the Pareto front comes from the '
            r'per-crossing mode choice and not from the two totals.')
    return _float(body, cap, label, 'rr r rr r rr rr', header,
                  size=r'\small', note=note)


def table_ablation(sub_tests, label='tab:prism-ablation', caption=None):
    """The same algorithm on each substrate: what the symmetry is worth."""
    if not sub_tests:
        return ''
    body = []
    for t in sub_tests:
        ci = (f"$[{t['ci_lo']:.1f},\\,{t['ci_hi']:.1f}]$"
              if t['ci_lo'] == t['ci_lo'] else '--')
        body.append('    ' + ' & '.join([
            esc(t['method']),
            f"${t['n_pairs']}$",
            f"\\prismsub{{H1}}{{{num(t['E_H1'])}}}",
            f"\\prismsub{{H2}}{{{num(t['E_H2'])}}}",
            f"${t['reduction_pct']:+.1f}$", ci,
            f"{t['n_better']}/{t['n_tie']}/{t['n_worse']}",
            pval_stars(t['wilcoxon_p']),
        ]) + r' \\')
    header = ('    Method & $N$ & $E$ on $H_1$ & $E$ on $H_2$ & '
              '$\\Delta$ [\\%] & 95\\% CI & B/T/W & $p$ \\\\')
    cap = caption or (
        'The price of being symmetry-blind. Each row is one algorithm run '
        'twice, unchanged, on the two substrates, so the difference isolates '
        'the hypergraph and nothing else. Positive $\\Delta$ means $H_2$ costs '
        'fewer ebits.')
    note = (r'B/T/W counts instances on which $H_2$ is better, ties, and is '
            r'worse. $p$ is a paired Wilcoxon signed-rank test; '
            r'$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.')
    return _float(body, cap, label, 'l r cc rc c r', header, note=note)


# ---------------------------------------------------------------------------
#  colour companion and the bundle
# ---------------------------------------------------------------------------
def colour_file():
    return '\n'.join([
        r'% Optional: \input this next to the tables to tint the method names',
        r'% with the same hues the figures use.  Needs \usepackage{xcolor}.',
        f'\\definecolor{{prismH1}}{{HTML}}{{{C_H1.lstrip("#").upper()}}}',
        f'\\definecolor{{prismH2}}{{HTML}}{{{C_H2.lstrip("#").upper()}}}',
        f'\\definecolor{{prismPRISM}}{{HTML}}'
        f'{{{C_PRISM_PLUS.lstrip("#").upper()}}}',
        r'\providecommand{\prismsub}[2]{#2}',
        r'\renewcommand{\prismsub}[2]{\textcolor{prism#1}{#2}}',
        '',
    ])


_PREVIEW = r"""\documentclass[10pt]{article}
\usepackage[a3paper,landscape,margin=12mm]{geometry}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{amsmath}
\input{prism_colours}
\pagestyle{empty}
\begin{document}
%s
\end{document}
"""


def write_latex(outdir, df, meta, order, wins, mean_rank, n_inst,
                tests=None, sub_tests=None, fr=None, cd=None,
                champion='PRISM+', compile_preview=True, verbose=True):
    """Write every table into ``outdir/tables`` and try to build a preview."""
    d = os.path.join(outdir, 'tables')
    os.makedirs(d, exist_ok=True)
    files = {}

    def put(name, text):
        if not text:
            return
        p = os.path.join(d, name)
        with open(p, 'w', encoding='utf-8') as f:
            f.write(text)
        files[name] = p
        if verbose:
            print(f'    {p}')

    put('prism_colours.tex', colour_file())
    put('table_performance.tex',
        table_performance(df, order, wins, mean_rank, n_inst))
    put('table_tests.tex', table_tests(tests, champion, fr, cd))
    put('table_scaling.tex', table_scaling(df, order))
    put('table_substrate.tex', table_substrate(meta))
    put('table_ablation.tex', table_ablation(sub_tests))

    bundle = [t for t in ('table_performance.tex', 'table_tests.tex',
                          'table_ablation.tex', 'table_scaling.tex',
                          'table_substrate.tex') if t in files]
    put('all_tables.tex',
        '% \\input this after \\usepackage{booktabs} (and xcolor, for the\n'
        '% optional \\input{prism_colours}).\n'
        + '\n'.join(f'\\input{{{t[:-4]}}}' for t in bundle) + '\n')
    put('preview.tex', _PREVIEW % '\n'.join(f'\\input{{{t[:-4]}}}\\clearpage'
                                            for t in bundle))

    if compile_preview:
        pdf = _compile(d, verbose=verbose)
        if pdf:
            files['preview.pdf'] = pdf
    return files


def _compile(d, verbose=True):
    """Best-effort pdflatex.  A missing TeX install is not an error."""
    import shutil
    import subprocess
    exe = shutil.which('pdflatex')
    if not exe:
        if verbose:
            print('    (no pdflatex; skipping the table preview)')
        return None
    try:
        for _ in range(2):
            r = subprocess.run(
                [exe, '-interaction=nonstopmode', '-halt-on-error',
                 'preview.tex'], cwd=d, capture_output=True, text=True,
                timeout=180)
        pdf = os.path.join(d, 'preview.pdf')
        if r.returncode == 0 and os.path.exists(pdf):
            for ext in ('.aux', '.log', '.out'):
                try:
                    os.remove(os.path.join(d, 'preview' + ext))
                except OSError:
                    pass
            if verbose:
                print(f'    {pdf}')
            return pdf
        if verbose:
            tail = [l for l in r.stdout.splitlines() if l.startswith('!')][:3]
            print('    (table preview did not compile: '
                  + ('; '.join(tail) or 'see tables/preview.log') + ')')
    except Exception as exc:                                # pragma: no cover
        if verbose:
            print(f'    (table preview skipped: {exc})')
    return None
