"""
pyprism_v2.partition.baselines
==============================
Single-objective partitioners.

Native (always available, NumPy only)
    naive, random, linear, spectral, kernighan_lin, fiduccia_mattheyses,
    greedy_growth, louvain, girvan_newman

External (used when the official package is installed)
    metis     -- PyMetis, on the light-cone-augmented clique expansion
    kahypar   -- KaHyPar, on the hypergraph directly, connectivity-1 objective

Every partitioner returns a bitmask over qubits: bit ``v`` set means ``v`` is
in block A.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict

import numpy as np

from ..objectives import CostState, balance, feasible

# --- optional external partitioners --------------------------------------
# These are prebuilt C extensions and some wheels are compiled for AVX2 or
# AVX-512.  On an older CPU such a wheel does not raise ImportError --- it
# executes an illegal instruction and takes the whole interpreter down with
# SIGILL, which no `except` can catch.  PYPRISM_DISABLE is the escape hatch:
# it skips the import entirely, so a machine with a bad wheel can still run
# everything else without uninstalling anything.
#
#     PYPRISM_DISABLE=kahypar python3 generate_dataset.py ...
#     PYPRISM_DISABLE=kahypar,pymetis python3 benchmark_sweep.py ...
#
# Diagnosing one of these is otherwise unpleasant, because the traceback
# points at the import line and says nothing about why.
_DISABLED = {x.strip().lower()
             for x in os.environ.get('PYPRISM_DISABLE', '').split(',')
             if x.strip()}

if 'pymetis' in _DISABLED:
    pymetis = None
else:
    try:
        import pymetis
    except ImportError:                                      # pragma: no cover
        pymetis = None

if 'kahypar' in _DISABLED:
    kahypar = None
else:
    try:
        import kahypar
    except ImportError:                                      # pragma: no cover
        kahypar = None

__all__ = ['clique_weights', 'repair', 'part_naive', 'part_random',
           'part_linear', 'part_spectral', 'part_kernighan_lin',
           'part_fiduccia_mattheyses', 'part_greedy', 'part_louvain',
           'part_girvan_newman', 'part_metis', 'part_kahypar',
           'find_kahypar_ini', 'kahypar_ready', 'HAVE_METIS',
           'HAVE_KAHYPAR']

HAVE_METIS = pymetis is not None
HAVE_KAHYPAR = kahypar is not None


# ---------------------------------------------------------------------------
def clique_weights(n, hg, lc=None, lc_alpha=0.35):
    """Clique expansion of the hypergraph, optionally light-cone augmented.

    A hyperedge of size ``|e|`` contributes ``w(e)/(|e|-1)`` to each of its
    pairs -- the standard expansion, exact for ``|e| = 2``.
    """
    W = np.zeros((n, n))
    for e in hg.nets:
        p = e.pins
        if len(p) < 2:
            continue
        w = (e.w_ebit + e.w_logk) / (len(p) - 1)
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                W[p[i], p[j]] += w
                W[p[j], p[i]] += w
    if lc is not None and lc_alpha:
        from ..lightcone import centre_lightcone
        scale = W.max() if W.max() > 0 else 1.0
        W = W + lc_alpha * scale * centre_lightcone(lc, n)
    return W


def repair(mask, n, eps):
    """Nudge a partition back inside the balance tolerance."""
    guard = 0
    while not feasible(mask, n, eps) and guard < 4 * n:
        guard += 1
        a = bin(mask).count('1')
        if a * 2 > n:
            v = next(q for q in range(n) if (mask >> q) & 1)
            mask &= ~(1 << v)
        else:
            v = next(q for q in range(n) if not (mask >> q) & 1)
            mask |= (1 << v)
    return mask


# ---------------------------------------------------------------------------
#  trivial
# ---------------------------------------------------------------------------
def part_naive(n, hg=None, **kw):
    """First ``n/2`` qubit indices."""
    return sum(1 << q for q in range(n // 2))


def part_linear(n, hg=None, **kw):
    """Contiguous blocks -- identical to naive for a linear index order, kept
    separate because it is the natural cut on a line topology."""
    return sum(1 << q for q in range(n // 2))


def part_random(n, hg, seed=0, eps=0.10, restarts=32, **kw):
    """Best of ``restarts`` random balanced bipartitions."""
    rng = np.random.default_rng(seed)
    best, best_c = None, None
    for _ in range(restarts):
        order = rng.permutation(n)
        m = sum(1 << int(q) for q in order[:n // 2])
        st = CostState(hg, m)
        c = st.E + st.K
        if best_c is None or c < best_c:
            best, best_c = m, c
    return best


# ---------------------------------------------------------------------------
#  spectral
# ---------------------------------------------------------------------------
def part_spectral(n, hg, lc=None, **kw):
    """Fiedler bisection of the (light-cone-augmented) clique expansion."""
    W = clique_weights(n, hg, lc)
    L = np.diag(W.sum(1)) - W
    vals, vecs = np.linalg.eigh(L)
    f = vecs[:, 1] if n > 1 else np.zeros(n)
    order = np.argsort(f)
    return sum(1 << int(q) for q in order[:n // 2])


# ---------------------------------------------------------------------------
#  local search
# ---------------------------------------------------------------------------
def part_kernighan_lin(n, hg, seed=0, eps=0.10, start=None, passes=6,
                       candidates=24, **kw):
    """Kernighan--Lin style pair-swap descent on the true objective sum.

    Swaps are evaluated by applying the two flips to the incremental state and
    reverting on rejection, so a candidate costs ``O(deg)`` rather than a full
    rescan.  Each pass considers the ``candidates`` most promising qubits per
    side --- those whose own flip helps most --- which keeps a pass linear in
    ``n`` instead of quadratic.
    """
    rng = np.random.default_rng(seed)
    mask = part_spectral(n, hg) if start is None else start
    st = CostState(hg, mask)
    for _ in range(passes):
        base = st.E + st.K
        gains = []
        for v in range(n):
            E, K, _ = st.peek(v)
            gains.append((base - (E + K), v))
        gains.sort(reverse=True)
        A = [v for _, v in gains if (st.mask >> v) & 1][:candidates]
        B = [v for _, v in gains if not (st.mask >> v) & 1][:candidates]
        improved = False
        for a in A:
            for b in B:
                cand = st.mask ^ (1 << a) ^ (1 << b)
                if not feasible(cand, n, eps):
                    continue
                saved = st.mask
                st.flip(a)
                st.flip(b)
                if st.E + st.K < base - 1e-12:
                    base, improved = st.E + st.K, True
                    break
                st.reset(saved)
            if improved:
                break
        if not improved:
            break
    return st.mask


def part_fiduccia_mattheyses(n, hg, seed=0, eps=0.10, start=None,
                             passes=8, **kw):
    """FM-style single-vertex descent with the incremental evaluator.

    This is the workhorse local search: each pass scans every qubit, takes the
    best feasible single flip, and stops when no flip improves.  Uses
    :class:`CostState`, so a pass is ``O(n * deg)`` rather than ``O(n * |nets|)``.
    """
    mask = part_spectral(n, hg) if start is None else start
    st = CostState(hg, mask)
    for _ in range(passes):
        best_v, best_c = None, st.E + st.K
        for v in range(n):
            cand = mask ^ (1 << v)
            if not feasible(cand, n, eps):
                continue
            E, K, _ = st.peek(v)
            if E + K < best_c - 1e-12:
                best_v, best_c = v, E + K
        if best_v is None:
            break
        st.flip(best_v)
        mask = st.mask
    return mask


def part_greedy(n, hg, seed=0, eps=0.10, lc=None, **kw):
    """Grow block A from the highest-degree seed, always adding the qubit that
    costs least, until the balance target is met."""
    W = clique_weights(n, hg, lc)
    deg = W.sum(1)
    start = int(np.argmax(deg))
    A = {start}
    target = n // 2
    while len(A) < target:
        cand = [v for v in range(n) if v not in A]
        gains = [W[v, list(A)].sum() - W[v, [u for u in range(n)
                                             if u not in A and u != v]].sum()
                 for v in cand]
        A.add(cand[int(np.argmax(gains))])
    return sum(1 << v for v in A)


# ---------------------------------------------------------------------------
#  community detection
# ---------------------------------------------------------------------------
def part_louvain(n, hg, seed=0, eps=0.10, lc=None, **kw):
    """Louvain modularity on the clique expansion, then merge the communities
    into two balanced blocks (largest-first bin packing)."""
    W = clique_weights(n, hg, lc)
    rng = np.random.default_rng(seed)
    comm = list(range(n))
    m2 = W.sum()
    if m2 <= 0:
        return part_naive(n)
    k = W.sum(1)
    improved = True
    rounds = 0
    while improved and rounds < 12:
        improved = False
        rounds += 1
        for v in rng.permutation(n):
            v = int(v)
            best, best_gain = comm[v], 0.0
            cur = comm[v]
            for u in range(n):
                if W[v, u] == 0 or comm[u] == cur:
                    continue
                c = comm[u]
                members = [x for x in range(n) if comm[x] == c]
                gain = W[v, members].sum() - k[v] * k[members].sum() / m2
                old = [x for x in range(n) if comm[x] == cur and x != v]
                gain -= W[v, old].sum() - k[v] * k[old].sum() / m2
                if gain > best_gain + 1e-12:
                    best, best_gain = c, gain
            if best != cur:
                comm[v] = best
                improved = True
    groups = defaultdict(list)
    for v, c in enumerate(comm):
        groups[c].append(v)
    A, sizeA = set(), 0
    for g in sorted(groups.values(), key=len, reverse=True):
        if sizeA + len(g) <= n // 2 or sizeA < n // 4:
            A.update(g)
            sizeA += len(g)
    return repair(sum(1 << v for v in A) if A else part_naive(n), n, eps)


def part_girvan_newman(n, hg, seed=0, eps=0.10, lc=None, max_n=64, **kw):
    """Girvan--Newman edge-betweenness bisection.

    ``O(n^3)`` and only meaningful on small instances; the driver skips it
    beyond ``max_n``.
    """
    if n > max_n:
        raise RuntimeError(f'girvan_newman skipped: n={n} > max_n={max_n}')
    W = clique_weights(n, hg, lc)
    adj = (W > 0).astype(float)
    for _ in range(n * 2):
        # edge betweenness by BFS from every source
        eb = defaultdict(float)
        for s in range(n):
            dist = np.full(n, -1)
            dist[s] = 0
            order, queue, parents = [], [s], defaultdict(list)
            sigma = np.zeros(n)
            sigma[s] = 1
            while queue:
                v = queue.pop(0)
                order.append(v)
                for w in np.nonzero(adj[v])[0]:
                    if dist[w] < 0:
                        dist[w] = dist[v] + 1
                        queue.append(int(w))
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        parents[w].append(v)
            delta = np.zeros(n)
            for w in reversed(order):
                for v in parents[w]:
                    c = sigma[v] / max(sigma[w], 1e-12) * (1 + delta[w])
                    eb[tuple(sorted((v, w)))] += c
                    delta[v] += c
        if not eb:
            break
        u, v = max(eb, key=eb.get)
        adj[u, v] = adj[v, u] = 0
        # connected components
        seen, comps = set(), []
        for s in range(n):
            if s in seen:
                continue
            stack, comp = [s], []
            seen.add(s)
            while stack:
                x = stack.pop()
                comp.append(x)
                for y in np.nonzero(adj[x])[0]:
                    if int(y) not in seen:
                        seen.add(int(y))
                        stack.append(int(y))
            comps.append(comp)
        if len(comps) >= 2:
            comps.sort(key=len, reverse=True)
            A = set(comps[0])
            for c in comps[1:]:
                if len(A) + len(c) <= n // 2:
                    A.update(c)
            return repair(sum(1 << v for v in A), n, eps)
    return part_naive(n)


# ---------------------------------------------------------------------------
#  external, official packages
# ---------------------------------------------------------------------------
def part_metis(n, hg, seed=0, eps=0.10, lc=None, **kw):
    """METIS via the official PyMetis binding, on the light-cone-augmented
    clique expansion."""
    if pymetis is None:
        raise RuntimeError('PyMetis is not installed')
    M = clique_weights(n, hg, lc)
    adj = defaultdict(list)
    mx = 0.0
    for a in range(n):
        for b in range(a + 1, n):
            if M[a, b] > 0:
                adj[a].append((b, float(M[a, b])))
                adj[b].append((a, float(M[a, b])))
                mx = max(mx, float(M[a, b]))
    scale = 1000.0 / max(mx, 1e-9)
    xadj, adjncy, eweights = [0], [], []
    for v in range(n):
        for (u, w) in sorted(adj[v]):
            adjncy.append(u)
            eweights.append(max(1, int(round(w * scale))))
        xadj.append(len(adjncy))
    _, membership = pymetis.part_graph(2, xadj=xadj, adjncy=adjncy,
                                       eweights=eweights)
    return repair(sum(1 << v for v in range(n) if membership[v] == 0), n, eps)


def find_kahypar_ini(explicit=None):
    """Locate a KaHyPar ``.ini`` configuration.

    The ``kahypar`` pip wheel ships the extension but **not** the tuned
    configuration files, so an otherwise working install still has nothing to
    load.  This searches the obvious places and, failing that, raises with the
    one command that fixes it.
    """
    import glob
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get('KAHYPAR_CONFIG'):
        cands.append(os.environ['KAHYPAR_CONFIG'])
    roots = []
    if kahypar is not None and getattr(kahypar, '__file__', None):
        pkg = os.path.dirname(os.path.abspath(kahypar.__file__))
        roots += [pkg, os.path.dirname(pkg)]
    roots += [os.getcwd(),
              os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
              os.path.expanduser('~/.kahypar'),
              os.path.expanduser('~/kahypar'),
              '/usr/local/share/kahypar', '/usr/share/kahypar',
              '/opt/homebrew/share/kahypar']
    for r in roots:
        if not r or not os.path.isdir(r):
            continue
        for pat in ('*.ini', 'config/*.ini', 'share/kahypar/config/*.ini',
                    '**/*.ini'):
            try:
                cands += sorted(glob.glob(os.path.join(r, pat),
                                          recursive=('**' in pat)))
            except OSError:
                pass
    # prefer the connectivity-1 configs, which match the ebit objective
    cands.sort(key=lambda c: (0 if 'km1' in os.path.basename(c) else 1,))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise RuntimeError(
        'kahypar is installed but no .ini configuration was found.\n'
        '  The pip wheel does not ship the config files.  Fetch one with\n\n'
        '    curl -LO https://raw.githubusercontent.com/kahypar/kahypar/'
        'master/config/km1_kKaHyPar_sea20.ini\n\n'
        '  then pass --kahypar-ini km1_kKaHyPar_sea20.ini, set '
        '$KAHYPAR_CONFIG,\n'
        '  or leave it in the working directory.')


def kahypar_ready(explicit=None):
    """True when KaHyPar can actually run: importable *and* configured."""
    if kahypar is None:
        return False, 'kahypar not installed'
    try:
        return True, find_kahypar_ini(explicit)
    except RuntimeError as exc:
        return False, str(exc)


def part_kahypar(n, hg, seed=0, eps=0.10, ini=None, **kw):
    """KaHyPar on the hypergraph directly, connectivity-1 objective.

    connectivity-1 is exactly the ebit metric lifted to k blocks: a net
    spanning ``lambda`` blocks costs ``lambda - 1`` cat-state broadcasts.
    """
    if kahypar is None:
        raise RuntimeError('kahypar is not installed')
    ini_path = ini or find_kahypar_ini()
    # A fresh Context per call, deliberately.  `kahypar.partition` mutates the
    # Context it is given: `setupContext` fills `max_part_weights` from the
    # hypergraph's total weight.  Reusing that Context for a second call leaves
    # those weights populated while `use_individual_part_weights` is still
    # false, which trips KaHyPar's sanity check and drops into an interactive
    # prompt --
    #     Individual block weights specified, but
    #     --use-individual-part-weights=false. Use them (Y/N)?
    # -- which would hang an unattended sweep.  Constructing the Context is
    # cheap next to partitioning, so it is rebuilt every time.
    ctx = kahypar.Context()
    ctx.loadINIconfiguration(ini_path)
    ctx.setK(2)
    ctx.setEpsilon(eps)
    ctx.setSeed(seed)
    try:
        ctx.suppressOutput(True)
    except AttributeError:                                   # pragma: no cover
        pass
    mx = max([e.w_ebit + e.w_logk for e in hg.nets] + [1.0])
    scale = 1000.0 / mx
    index_vector, edge_vector, edge_weights = [0], [], []
    for e in hg.nets:
        if len(e.pins) < 2:
            continue
        edge_vector.extend(int(p) for p in e.pins)
        index_vector.append(len(edge_vector))
        edge_weights.append(max(1, int(round((e.w_ebit + e.w_logk) * scale))))
    if not edge_weights:
        return part_naive(n)
    hyper = kahypar.Hypergraph(n, len(edge_weights), index_vector, edge_vector,
                               2, edge_weights, [1] * n)
    kahypar.partition(hyper, ctx)
    return repair(sum(1 << v for v in range(n) if hyper.blockID(v) == 0),
                  n, eps)
