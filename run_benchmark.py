#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
run_benchmark.py
================
Benchmark every registered method against PRISM and PRISM+ on random quantum
circuits, over a sweep of register sizes with depth scaled to the register.

    n     in {20, 50, 100, 200}
    depth = k * n   for k in {2, 3, 4}

Nothing is simulated: both objectives are static functions of the gate list and
the bipartition, so the sweep reaches ``n = 200`` on a laptop.

    E(A,B) = sum over crossings of e_i         [ebits, exact distribution]
    K(A,B) = sum over crossings of log gamma_i [knitting, quasiprobability]

Every method optimises on the substrate it is given -- ``H1`` symmetry-blind,
``H2`` symmetry- and irreducibility-aware -- and every method is **scored** on
the exact Track 2 cost, because that is what a distributed compiler pays.
Fronts are compared by hypervolume against a per-instance reference point, so
the numbers are comparable across methods within an instance.

Usage
-----
    python3 run_benchmark.py                      # the full sweep
    python3 run_benchmark.py --quick              # a fast smoke run
    python3 run_benchmark.py --n 20,50 --k 2,3 --seeds 3
    python3 run_benchmark.py --skip GirvanNewman,SPEA2

Optional extras, used automatically when installed::

    pip install PyMetis kahypar pymoo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyprism_v2 as pp
from pyprism_v2.benchmark import sweep, run_instance, default_moo_budget
from pyprism_v2.partition import REGISTRY, available
from pyprism_v2.partition.prism import scaled_sweeps


def _summary(df):
    import pandas as pd
    g = df.groupby('method')
    out = g.agg(hv_norm=('hv_norm', 'mean'),
                E_mean=('E_CD', 'mean'),
                K_mean=('K_CK', 'mean'),
                front=('front_size', 'mean'),
                sec=('seconds', 'mean'),
                fails=('E_CD', lambda s: int(s.isna().sum())))
    # win rate on hypervolume, ties shared
    wins = {}
    n_inst = 0
    for _, sub in df.groupby(['n', 'k', 'seed']):
        n_inst += 1
        best = sub['hv'].max()
        w = sub[sub['hv'] >= best - 1e-9]['method'].tolist()
        for m in w:
            wins[m] = wins.get(m, 0.0) + 1.0 / len(w)
    out['win_rate'] = [wins.get(m, 0.0) / max(n_inst, 1) for m in out.index]
    ranks = df.copy()
    ranks['rank'] = ranks.groupby(['n', 'k', 'seed'])['hv'].rank(
        ascending=False, method='average')
    out['mean_rank'] = ranks.groupby('method')['rank'].mean()
    return out.sort_values('hv_norm', ascending=False)


def main():
    ap = argparse.ArgumentParser(
        description='PRISM v2 benchmark over random quantum circuits.')
    ap.add_argument('--n', default='20,50,100,200',
                    help='comma-separated register sizes')
    ap.add_argument('--k', default='1,2,3',
                    help='comma-separated depth multipliers, depth = k * n')
    ap.add_argument('--seeds', type=int, default=100,
                    help='circuits per (n, k) cell')
    ap.add_argument('--family', default='rqc',
                    choices=sorted(pp.FAMILIES))
    ap.add_argument('--topology', default='grid',
                    choices=['grid', 'ring', 'line', 'full', 'heavy-hex'])
    ap.add_argument('--eps', type=float, default=0.10)
    ap.add_argument('--sweeps', type=int, default=None,
                    help='PRISM sweeps per replica (default scales with n)')
    ap.add_argument('--replicas', type=int, default=8)
    ap.add_argument('--temps', type=int, default=4)
    ap.add_argument('--workers', type=int, default=-1)
    ap.add_argument('--skip', default='', help='comma-separated method names')
    ap.add_argument('--kahypar-ini', default=None)
    ap.add_argument('--outdir', default='benchmark_results+v2')
    ap.add_argument('--quick', action='store_true',
                    help='n=20,50 k=2 one seed, small budgets')
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--no-circuits', action='store_true',
                    help='do not save the circuit instances')
    ap.add_argument('--no-qasm', action='store_true',
                    help='save circuits as JSON only, skip the QASM export')
    args = ap.parse_args()

    if args.quick:
        args.n, args.k, args.seeds = '20,50', '2', 1
        args.sweeps = args.sweeps or 120

    ns = tuple(int(x) for x in args.n.split(','))
    ks = tuple(int(x) for x in args.k.split(','))
    seeds = tuple(range(args.seeds))
    skip = tuple(x.strip() for x in args.skip.split(',') if x.strip())
    os.makedirs(args.outdir, exist_ok=True)

    print('=' * 78)
    print(f'pyprism_v2 {pp.__version__} benchmark')
    print(f'  n      = {list(ns)}')
    print(f'  k      = {list(ks)}   (depth = k * n)')
    print(f'  seeds  = {args.seeds} per cell   family={args.family} '
          f'topology={args.topology}')
    # KaHyPar can be importable yet unusable: the pip wheel ships no .ini.
    # Resolve it once here so the run does not fail on every instance.
    from pyprism_v2.partition.baselines import kahypar_ready, HAVE_KAHYPAR
    if HAVE_KAHYPAR and 'KaHyPar' not in skip:
        ready, info = kahypar_ready(args.kahypar_ini)
        if ready:
            args.kahypar_ini = info
        else:
            skip = skip + ('KaHyPar',)
            print('  KaHyPar skipped -- ' + info.splitlines()[0])
            for line in info.splitlines()[1:]:
                print('    ' + line)

    reg = available(max(ns), extra_skip=skip)
    print(f'  methods available at n={max(ns)}: {len(reg)} '
          f'+ PRISM and PRISM+ (both on the Track 2 substrate)')
    import importlib
    absent = []
    for m in REGISTRY.values():
        if not m.needs or m.name in [r.name for r in reg]:
            continue
        try:
            importlib.import_module(m.needs)
            absent.append(f'{m.name} (installed, unusable)')
        except Exception:
            absent.append(f'{m.name} (no {m.needs})')
    if absent:
        print(f'  skipped: {", ".join(sorted(set(absent)))}')
    print(f'  PRISM sweeps per replica: '
          + ', '.join(f'n={n}:{args.sweeps or scaled_sweeps(n)}' for n in ns))
    print(f'  MOO budget: '
          + ', '.join(f'n={n}:{default_moo_budget(n)["pop_size"]}x'
                      f'{default_moo_budget(n)["generations"]}' for n in ns))
    print('=' * 78)

    cfg = None
    if args.sweeps or args.replicas != 5 or args.temps != 4 or args.workers != 1:
        cfg = pp.PrismConfig(n_pref=args.replicas, n_temp=args.temps,
                             sweeps=args.sweeps or 400, eps=args.eps,
                             workers=args.workers)

    t0 = time.time()
    circ_dir = (None if args.no_circuits
                else os.path.join(args.outdir, 'circuits'))
    rows, metas, kept_fronts, saved = sweep(
        ns=ns, ks=ks, seeds=seeds, family=args.family,
        topology=args.topology, eps=args.eps, cfg=cfg, skip=skip,
        kahypar_ini=args.kahypar_ini, save_dir=circ_dir,
        save_qasm=not args.no_qasm)
    wall = time.time() - t0

    # a manifest pins the whole study: environment, arguments, and a content
    # hash per instance, so a result can always be traced to its input
    from pyprism_v2.io import write_manifest, environment
    man = write_manifest(args.outdir, saved, config=vars(args),
                         extra={'wall_seconds': round(wall, 2),
                                'skipped_methods': list(skip)})
    if saved:
        mb = sum(e.get('bytes_json', 0) + e.get('bytes_qasm', 0)
                 for e in saved) / 1e6
        print(f'\nsaved {len(saved)} circuit instances to {circ_dir}/ '
              f'({mb:.1f} MB)')
    print(f'manifest: {man}')

    try:
        import pandas as pd
    except ImportError:
        with open(os.path.join(args.outdir, 'rows.json'), 'w') as f:
            json.dump(rows, f, indent=1, default=float)
        print('\npandas not installed; wrote rows.json only')
        return 0

    df = pd.DataFrame(rows)
    meta = pd.DataFrame(metas)
    summ = _summary(df)
    if args.no_plots:
        # without the study writer, still leave the tables behind
        df.to_csv(os.path.join(args.outdir, 'raw.csv'), index=False)
        meta.to_csv(os.path.join(args.outdir, 'instances.csv'), index=False)
        summ.to_csv(os.path.join(args.outdir, 'summary.csv'))

    print(f'\nfinished {len(metas)} instances in {wall:.1f}s\n')
    with pd.option_context('display.width', 170,
                           'display.float_format', lambda v: f'{v:9.3f}'):
        print(summ.to_string())

    print('\ncircuit and substrate sizes')
    cols = ['n', 'k', 'gates', 'H1_nets', 'H2_nets', 'packets', 'compression',
            'lc_scrambling_depth', 'spearman_H2', 'wall']
    with pd.option_context('display.width', 170,
                           'display.float_format', lambda v: f'{v:8.2f}'):
        print(meta[cols].to_string(index=False))

    base = summ.loc['Naive', 'E_mean'] if 'Naive' in summ.index else np.nan
    for m in ('PRISM', 'PRISM+'):
        if m in summ.index and base == base:
            d = 100 * (base - summ.loc[m, 'E_mean']) / max(base, 1e-9)
            print(f'\n  {m:10s} mean E is {d:+.1f}% vs Naive, '
                  f'win rate {summ.loc[m, "win_rate"]*100:.0f}%, '
                  f'mean rank {summ.loc[m, "mean_rank"]:.2f}')
    fails = df[df['E_CD'].isna()]
    if len(fails):
        print(f'\n  {len(fails)} method runs failed:')
        for nm, sub in fails.groupby('method'):
            print(f'    {nm}: {sub.iloc[0]["note"][:90]}')

    if not args.no_plots:
        try:
            from pyprism_v2.study import write_study
            print(f'\nwriting the study into {args.outdir}/')
            write_study(df, meta, args.outdir, fronts_by_instance=kept_fronts,
                        champion='PRISM+',
                        title=f'pyprism_v2 {pp.__version__} — {len(metas)} '
                              f'instances, n in {list(ns)}, '
                              f'depth = k·n for k in {list(ks)}')
        except ImportError as e:
            print(f'\nplotting unavailable ({e}); skipping figures')

    print(f'\nstudy written to {args.outdir}/  '
          f'(raw.csv, summary.csv, instances.csv, pivot_*.csv, '
          f'stats_tests.csv, REPORT.md, study.json, figures/*.pdf)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
