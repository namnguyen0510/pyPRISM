"""
pyprism_v2.partition.prism
==========================
PRISM: a two-dimensional replica ensemble indexed by ``(w, T)``.

Grid
----
Columns are preference vectors ``w_i``, rows temperatures ``T_j``.  Within a
column, exchange along ``T`` preserves the escape behaviour of ordinary
parallel tempering.  Between columns, exchange along ``w`` moves solutions
between regions of the front.  Setting ``n_pref = 1`` recovers single-objective
PRISM-LCT exactly.

Exchange criteria
-----------------
``T`` axis (same objective, different temperature)::

    accept with min(1, exp[(beta_i - beta_j)(f_i - f_j)])

``w`` axis (same temperature, different objective) -- a Hamiltonian exchange::

    accept with min(1, exp[ beta ( f_i(x_i) + f_j(x_j)
                                   - f_i(x_j) - f_j(x_i) ) ])

which is the rigorous form of "Metropolis on the neighbour's scalarisation".

Seeding
-------
``PRISM`` starts every replica at random.  ``PRISM+`` multi-starts from the
METIS and KaHyPar solutions, spread over the grid, with the remainder random.

Temperature
-----------
Temperatures are **calibrated to the objective**, per column, and this is not
optional.  The scalarisation is normalised by the trivial split, so ``S`` lives
on a scale that depends on both the instance and the preference vector: on a
64-qubit instance it spans about ``[0.003, 0.05]`` at ``w0 = 0.05`` but
``[0.02, 0.48]`` at ``w0 = 0.5`` -- an order of magnitude between columns of
the same grid.  A fixed absolute ladder therefore cannot be right for more than
one column at a time, and the earlier default of ``T in [0.03, 1.2]`` was hot
enough that the *coldest* rung accepted a typical worsening move nine times in
ten.  Every rung was then an unbiased random walk, replica exchange accepted
around 85% of swaps instead of the 20-40% that indicates a working ladder, and
the ensemble drifted away from good solutions rather than towards them.  On a
64-qubit instance that cost a factor of thirteen in ebits.

:func:`calibrate_temperatures` fixes the scale by measuring the median
``|dS|`` of a single feasible move at that column's preference vector and
placing the rungs so the coldest accepts such a move with probability
``t_accept_cold`` and the hottest with ``t_accept_hot``:

    T = |dS|_median / ln(1 / p)

Set ``auto_temp=False`` to recover the old fixed ladder, which is retained only
so the difference can be reported.

Scaling
-------
Moves are evaluated with :class:`~pyprism_v2.objectives.CostState`, so a
single-qubit flip costs ``O(deg(v))`` rather than ``O(|nets|)``.  Light-cone
coupling biases which qubit is proposed: the one whose causal coupling points
mostly across the current cut is the one worth moving.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from multiprocessing import Pool

import numpy as np

from .refine import endpoint_polish
from ..objectives import (CostState, mode_front, balance, feasible, pareto,
                          knee)

__all__ = ['PrismConfig', 'part_prism', 'scaled_sweeps',
           'calibrate_temperatures']


def scaled_sweeps(n, base=400):
    """Sweeps per replica at register size n.

    A move costs ``O(deg(v))`` and the mean net degree grows with n, so a flat
    sweep count would make the large instances dominate the sweep.  The budget
    is reported alongside the results.
    """
    if n <= 32:
        return base
    if n <= 64:
        return int(base * 0.8)
    if n <= 128:
        return int(base * 0.5)
    return int(base * 0.3)


@dataclass
class PrismConfig:
    n_pref: int = 5             # preference-vector rungs (the w axis)
    n_temp: int = 4             # temperature rungs (the T axis)
    sweeps: int = 400          # see `scaled_sweeps` for the size-aware default
    swap_every: int = 25        # sweeps per round == the w-axis swap interval
    tabu: int = 6
    eps: float = 0.10
    seed: int = 0
    workers: int = 1
    # --- temperature ------------------------------------------------------
    # Calibrated per column by default; see `calibrate_temperatures`.  The
    # acceptance targets, not the temperatures, are the tunable quantities,
    # because they are the things that mean the same on every instance.
    auto_temp: bool = True
    t_accept_cold: float = 0.01   # coldest rung: accept a typical worsening
    t_accept_hot: float = 0.50    # hottest rung:   ... move this often
    calib_samples: int = 256
    t_min: float = 0.03           # used only when auto_temp is False
    t_max: float = 1.2
    # --- endpoint refinement ----------------------------------------------
    # Applied to the returned front, not inside the ladder, and identical to
    # what PRISM++ applies -- same function in `refine.py`, same parameters.
    # Only one of the two having it would make the PRISM / PRISM+ rows of the
    # benchmark a comparison of post-processing rather than of searches.
    endpoint_polish: bool = True
    ep_starts: int = 4            # distinct basins to descend from
    ep_cap: int = 24              # moves per FM pass
    boundary_hops: int = 1        # dilate P(A), P(B) by this many hops
    boundary_min: int = 8         # below this, fall back to the whole side


def calibrate_temperatures(hg, mask, w, norm, n, eps, cfg, seed=0):
    """Temperature ladder for one column, scaled to that column's objective.

    Samples moves from ``mask`` and takes the median ``|dS|``.  A rung that
    should accept such a move with probability ``p`` sits at
    ``T = |dS| / ln(1/p)``, so the ladder is fixed by the two acceptance
    targets rather than by absolute numbers that mean nothing across instances.

    The sample uses the *same move mix as the search* --- half single-qubit
    flips, half ``A``/``B`` pair swaps, matching ``_advance`` --- because those
    two move types have quite different ``|dS|`` and calibrating on the wrong
    one mis-sets the ladder.  It also matters at small ``n``: with ``n = 6`` and
    ``eps = 0.10`` no single flip is balance-feasible at all, so a flip-only
    sample would measure nothing and silently fall back.

    Falls back to the fixed ladder only if no feasible move of either kind can
    be sampled, which means the instance admits no moves at all.
    """
    rng = np.random.default_rng(seed)
    st = CostState(hg, mask).set_pref(w[0] / norm[0], w[1] / norm[1])
    base = st.S + 1e-4 * balance(mask, n)
    inA = [q for q in range(n) if (mask >> q) & 1]
    inB = [q for q in range(n) if not (mask >> q) & 1]
    deltas = []
    for _ in range(cfg.calib_samples):
        if rng.random() < 0.5:
            v = int(rng.integers(n))
            cand = mask ^ (1 << v)
            if not feasible(cand, n, eps):
                continue
            s_c = st.peek(v)[2]
        else:
            if not inA or not inB:
                continue
            a_, b_ = int(rng.choice(inA)), int(rng.choice(inB))
            cand = mask ^ (1 << a_) ^ (1 << b_)
            if not feasible(cand, n, eps):
                continue
            st.flip(a_)
            st.flip(b_)
            s_c = st.S
            st.flip(b_)             # O(deg) each way, unlike reset()
            st.flip(a_)
        d = abs(s_c + 1e-4 * balance(cand, n) - base)
        if d > 0.0:
            deltas.append(d)
    if not deltas:
        return list(np.geomspace(cfg.t_min, cfg.t_max, cfg.n_temp)), float('nan')
    med = float(np.median(deltas))
    t_cold = med / math.log(1.0 / max(cfg.t_accept_cold, 1e-12))
    t_hot = med / math.log(1.0 / max(cfg.t_accept_hot, 1e-12))
    if cfg.n_temp == 1:
        return [t_cold], med
    return list(np.geomspace(t_cold, max(t_hot, t_cold * 1.001),
                             cfg.n_temp)), med


# --- worker state ----------------------------------------------------------
_WS: dict = {}


def _init_worker(hg, lc, norm, cfg):
    _WS['hg'] = hg
    _WS['lc'] = lc
    _WS['norm'] = norm
    _WS['cfg'] = cfg


def _scal_from(E, K, w, norm, mask, n):
    return (w[0] * E / norm[0] + w[1] * K / norm[1] + 1e-4 * balance(mask, n))


def _lc_weights(lc, mask, n):
    """Light-cone-biased proposal distribution.  Uniform when ``lc is None``,
    which recovers plain single-vertex tempering."""
    if lc is None:
        return None
    inA = np.array([(mask >> v) & 1 for v in range(n)], dtype=bool)
    cross = np.where(inA[:, None] != inA[None, :], lc, 0.0).sum(1)
    same = np.where(inA[:, None] == inA[None, :], lc, 0.0).sum(1)
    s = cross - same
    s = s - s.min() + 1e-3
    return s / s.sum()


def _advance(args):
    """Advance one column by ``rounds_sweeps`` sweeps with T-axis exchange."""
    (w, masks, temps, rounds_sweeps, rng_seed) = args
    hg, lc, norm, cfg = _WS['hg'], _WS['lc'], _WS['norm'], _WS['cfg']
    n = hg.n
    rng = np.random.default_rng(rng_seed)

    a, b = w[0] / norm[0], w[1] / norm[1]
    reps = []
    for m in masks:
        st = CostState(hg, m).set_pref(a, b)
        reps.append({'st': st, 's': st.S + 1e-4 * balance(m, n), 'tabu': {}})
    visited = [(r['st'].E, r['st'].K, r['st'].mask) for r in reps]
    acc = att = 0
    # a round is the w-axis swap interval, so the T stride must be shorter or
    # no temperature swap would ever be attempted
    t_stride = max(1, rounds_sweeps // 5)

    def swap_T():
        nonlocal acc, att
        for ri in range(len(reps) - 1):
            cold, hot = reps[ri], reps[ri + 1]
            d = (1.0 / temps[ri] - 1.0 / temps[ri + 1]) * (cold['s'] - hot['s'])
            att += 1
            if d >= 0 or rng.random() < math.exp(d):
                reps[ri], reps[ri + 1] = hot, cold
                acc += 1

    # the proposal distribution is O(n^2); it changes slowly, so it is
    # refreshed on a stride rather than rebuilt for every single move
    pw_stride = max(4, rounds_sweeps // 8)
    pw_cache = [None] * len(reps)
    for it in range(rounds_sweeps):
        for ri, rep in enumerate(reps):
            T, st = temps[ri], rep['st']
            if it % pw_stride == 0 or pw_cache[ri] is None:
                pw_cache[ri] = _lc_weights(lc, st.mask, n)
            pw = pw_cache[ri]
            if rng.random() < 0.5:
                v = int(rng.choice(n, p=pw) if pw is not None
                        else rng.integers(n))
                if rep['tabu'].get(v, -1) > it:
                    continue
                cand = st.mask ^ (1 << v)
                if not feasible(cand, n, cfg.eps):
                    continue
                E, K, S = st.peek(v)
                s2 = S + 1e-4 * balance(cand, n)
                if s2 <= rep['s'] or rng.random() < math.exp(-(s2 - rep['s']) / T):
                    st.flip(v)
                    rep['s'] = s2
                    rep['tabu'][v] = it + cfg.tabu
                    visited.append((E, K, cand))
            else:
                inA = [q for q in range(n) if (st.mask >> q) & 1]
                inB = [q for q in range(n) if not (st.mask >> q) & 1]
                if not inA or not inB:
                    continue
                a_, b_ = int(rng.choice(inA)), int(rng.choice(inB))
                cand = st.mask ^ (1 << a_) ^ (1 << b_)
                if not feasible(cand, n, cfg.eps):
                    continue
                saved = st.mask
                st.flip(a_)
                E, K = st.flip(b_)
                s2 = st.S + 1e-4 * balance(cand, n)
                if s2 <= rep['s'] or rng.random() < math.exp(-(s2 - rep['s']) / T):
                    rep['s'] = s2
                    rep['tabu'][a_] = rep['tabu'][b_] = it + cfg.tabu
                    visited.append((E, K, cand))
                else:
                    st.reset(saved)
        if it and it % t_stride == 0:
            swap_T()
    swap_T()
    return ([r['st'].mask for r in reps], visited, acc, att,
            [r['s'] for r in reps])


def _polish(hg, mask, w, norm, n, eps, max_steps=None):
    """Steepest-descent single-vertex moves until no scalarised gain.

    Each step scans all n qubits, so the number of improving steps is capped:
    the ladder has already done the exploring and polish is only meant to clean
    up the last few moves.
    """
    max_steps = max_steps if max_steps is not None else max(8, n // 8)
    st = CostState(hg, mask).set_pref(w[0] / norm[0], w[1] / norm[1])
    s = st.S + 1e-4 * balance(mask, n)
    improved, steps = True, 0
    while improved and steps < max_steps:
        steps += 1
        improved = False
        best_v, best_s = None, s
        for v in range(n):
            cand = st.mask ^ (1 << v)
            if not feasible(cand, n, eps):
                continue
            _E, _K, S = st.peek(v)
            s2 = S + 1e-4 * balance(cand, n)
            if s2 < best_s - 1e-12:
                best_v, best_s = v, s2
        if best_v is not None:
            st.flip(best_v)
            s, improved = best_s, True
    return st.mask


def _polish_worker(args):
    """Polish the coldest rung of the column only.

    Steepest descent scans all n qubits per improving step, so polishing every
    rung costs n_temp times as much for solutions that the ladder has already
    driven together.  The coldest rung is the one carrying the column's best
    solution, so that is the one worth refining.
    """
    w, masks = args
    hg, norm, cfg = _WS['hg'], _WS['norm'], _WS['cfg']
    out = []
    for m in masks[:1]:
        pm = _polish(hg, m, w, norm, hg.n, cfg.eps)
        st = CostState(hg, pm)
        out.append((st.E, st.K, pm))
    for m in masks[1:]:
        st = CostState(hg, m)
        out.append((st.E, st.K, m))
    return out


def part_prism(n, hg, cfg: PrismConfig, seeds=None, lc=None, label='PRISM',
               max_expand=48, trace=None):
    """Run the ensemble.  Returns ``(front, diagnostics)``.

    ``max_expand`` bounds how many non-dominated partitions have their full
    CD/CK mode front expanded; each expansion is one ``O(|nets|)`` rebuild, and
    an evenly spaced subset keeps the reported front representative without
    paying for thousands of them.

    ``trace``, if given, is called once per round as ``trace(round, state)``
    with the live grid and the running counters.  It is observation only --- it
    is never consulted, so a traced run and an untraced run with the same seed
    take the same path.  This is how the diagnostics in the method note are
    collected without a second, drifting implementation of the ladder.
    """
    from ..objectives import evaluate
    from .baselines import part_naive
    ref = evaluate(hg, part_naive(n))
    norm = (max(ref[0], 1.0), max(ref[1], 1e-9))
    ws = [(a, 1.0 - a) for a in np.linspace(0.05, 0.95, cfg.n_pref)]
    rng = np.random.default_rng(cfg.seed)

    grid, pool_seeds, si = [], list(seeds or []), 0
    for i in range(cfg.n_pref):
        col = []
        for j in range(cfg.n_temp):
            if si < len(pool_seeds):
                col.append(pool_seeds[si])
                si += 1
            else:
                order = rng.permutation(n)
                col.append(sum(1 << int(q) for q in order[:n // 2]))
        grid.append(col)

    # One ladder per column.  S is normalised by the trivial split, so its
    # scale depends on w as well as on the instance -- an order of magnitude
    # between the middle column and the outer ones.  A single absolute ladder
    # is therefore wrong for all but one column, whatever its values.
    if cfg.auto_temp:
        cal = [calibrate_temperatures(hg, grid[i][0], ws[i], norm, n, cfg.eps,
                                      cfg, seed=cfg.seed + 31 * i)
               for i in range(cfg.n_pref)]
        col_temps = [c[0] for c in cal]
        col_dS = [c[1] for c in cal]
    else:
        fixed = list(np.geomspace(cfg.t_min, cfg.t_max, cfg.n_temp))
        col_temps = [fixed for _ in range(cfg.n_pref)]
        col_dS = [float('nan')] * cfg.n_pref
    # the w-axis couples two columns, so its criterion needs a temperature
    # meaningful to both: the geometric mean of the two rungs being exchanged
    w_temps = [[math.sqrt(col_temps[i][j] * col_temps[i + 1][j])
                for j in range(cfg.n_temp)]
               for i in range(max(cfg.n_pref - 1, 1))]

    n_rounds = max(1, cfg.sweeps // max(cfg.swap_every, 1))
    per_round = max(1, cfg.sweeps // n_rounds)
    lc_c = None
    if lc is not None:
        from ..lightcone import centre_lightcone
        lc_c = centre_lightcone(lc, n)

    visited_all, acc_t, att_t, acc_w, att_w = [], 0, 0, 0, 0
    use_pool = cfg.workers > 1 and cfg.n_pref > 1

    def scal_of(mask, w):
        st = CostState(hg, mask).set_pref(w[0] / norm[0], w[1] / norm[1])
        return st.S + 1e-4 * balance(mask, n)

    def run(pool):
        nonlocal grid, visited_all, acc_t, att_t, acc_w, att_w
        for rd in range(n_rounds):
            payload = [(ws[i], grid[i], col_temps[i], per_round,
                        cfg.seed + 1009 * rd + 7 * i)
                       for i in range(cfg.n_pref)]
            outs = (pool.map(_advance, payload) if pool
                    else [_advance(pl) for pl in payload])
            col_t = []
            for i, (masks, vis, a, t_, _ss) in enumerate(outs):
                grid[i] = masks
                visited_all.extend(vis)
                acc_t += a
                att_t += t_
                col_t.append((a, t_))
            for i in range(cfg.n_pref - 1):
                for j in range(cfg.n_temp):
                    xi, xj = grid[i][j], grid[i + 1][j]
                    d = ((scal_of(xi, ws[i]) + scal_of(xj, ws[i + 1])
                          - scal_of(xj, ws[i]) - scal_of(xi, ws[i + 1]))
                         / w_temps[i][j])
                    att_w += 1
                    if d >= 0 or rng.random() < math.exp(min(d, 0.0)):
                        grid[i][j], grid[i + 1][j] = xj, xi
                        acc_w += 1
            if trace is not None:
                trace(rd, {'grid': [list(col) for col in grid],
                           'ws': ws, 'temps': col_temps, 'col_T': col_t,
                           'acc_T': acc_t, 'att_T': att_t,
                           'acc_w': acc_w, 'att_w': att_w,
                           'visited': len(visited_all)})

    # Provenance.  `_advance` records each replica's starting state, so a seed
    # handed in by PRISM+ lands on the front whether or not the ladder ever
    # improved on it.  That is correct -- it is a real partition -- but it
    # makes "PRISM+ reached E" ambiguous between the search and the seed.
    # Keeping the initial masks lets the diagnostics say which.
    init_masks = {m for col in grid for m in col}

    init = (hg, lc_c, norm, cfg)
    if use_pool:
        with Pool(processes=min(cfg.workers, cfg.n_pref),
                  initializer=_init_worker, initargs=init) as pool:
            run(pool)
            polished = pool.map(_polish_worker,
                                [(ws[i], grid[i]) for i in range(cfg.n_pref)])
    else:
        _init_worker(*init)
        run(None)
        polished = [_polish_worker((ws[i], grid[i]))
                    for i in range(cfg.n_pref)]
    for chunk in polished:
        visited_all.extend(chunk)

    # --- endpoint refinement ----------------------------------------------
    # The ladder minimises S_w; the front is read at its E and K corners, and a
    # net that takes the K branch of `min(a e_i, b k_i)` drops out of the S
    # gradient entirely, so S can be at a local minimum while E is far from
    # one.  Descend on the reported quantity, from the cheapest few distinct
    # basins.  Identical to what PRISM++ does -- see `refine.endpoint_polish`.
    ep_steps = {}
    if getattr(cfg, 'endpoint_polish', False) and visited_all:
        for which, tag in ((0, 'E'), (1, 'K')):
            seen, starts = set(), []
            for v in sorted(visited_all, key=lambda z: z[which]):
                if v[2] in seen:
                    continue
                seen.add(v[2])
                starts.append(v[2])
                if len(starts) >= cfg.ep_starts:
                    break
            tot = 0
            for start in starts:
                m2, steps = endpoint_polish(n, hg, start, cfg.eps,
                                            which=which,
                                            hops=cfg.boundary_hops,
                                            min_size=cfg.boundary_min,
                                            cap=cfg.ep_cap)
                tot += steps
                if m2 != start and feasible(m2, n, cfg.eps):
                    visited_all.append((*evaluate(hg, m2), m2))
            ep_steps[tag] = tot

    # Expanding the mode front of every visited partition would mean one full
    # O(|nets|) rebuild each, which dominates at scale.  The endpoint
    # currencies are already known from the search, so the non-dominated
    # partitions are selected first and only those are expanded.
    seeds_front = pareto([(E, K, m) for E, K, m in visited_all])
    if len(seeds_front) > max_expand:
        # keep an evenly spaced subset so the front stays representative
        idx = np.linspace(0, len(seeds_front) - 1, max_expand).round().astype(int)
        seeds_front = [seeds_front[i] for i in sorted(set(idx.tolist()))]
    out = []
    for _, _, m in seeds_front:
        st = CostState(hg, m)
        for E, K in mode_front(st.costs()):
            out.append((E, K, m))
    front = pareto(out)

    e_init = min((E for E, _K, m in visited_all if m in init_masks),
                 default=float('inf'))
    e_search = min((E for E, _K, m in visited_all if m not in init_masks),
                   default=float('inf'))
    diag = {'label': label, 'rounds': n_rounds, 'sweeps_per_round': per_round,
            'grid': f'{cfg.n_pref}w x {cfg.n_temp}T',
            'swap_accept_T': acc_t / max(att_t, 1),
            'swap_accept_w': acc_w / max(att_w, 1),
            'visited': len(visited_all), 'front': len(front),
            'seeded': len(pool_seeds),
            'auto_temp': bool(cfg.auto_temp),
            'temps': [[float(t) for t in col] for col in col_temps],
            'median_dS': [float(x) for x in col_dS],
            'endpoint_steps': ep_steps,
            # what the ladder contributed over the states it started from
            'best_E_initial': float(e_init),
            'best_E_search': float(e_search),
            'search_improved': bool(e_search < e_init)}
    return front, diag
