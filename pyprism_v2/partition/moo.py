"""
pyprism_v2.partition.moo
========================
Multi-objective optimisers, for comparison against PRISM's replica ensemble.

The decision variable is a bitmask over qubits; the objectives are the two
currencies ``(E, K)`` of :mod:`pyprism_v2.objectives`, and each candidate
contributes its whole CD/CK mode front, so every optimiser is compared on the
same footing as PRISM.

Native (always available, NumPy only)
    ``nsga2``       fast non-dominated sort + crowding distance
    ``spea2``       strength Pareto with k-th nearest-neighbour density
    ``moead``       Tchebycheff decomposition over a weight simplex
    ``weighted_sum`` scalarisation sweep, the classical baseline
    ``epsilon_constraint`` minimise K subject to E <= eps, swept over eps

pymoo (used when installed)
    ``pymoo_nsga2``, ``pymoo_nsga3``, ``pymoo_sms``, ``pymoo_age``,
    ``pymoo_moead``

A note on the classical scalarisation baselines: a weighted sum can only reach
points on the convex hull of the front, so on a front with concave stretches it
misses solutions by construction.  The epsilon-constraint sweep does not have
that limitation, which is why both are included.
"""
from __future__ import annotations

import math

import numpy as np

from ..objectives import CostState, mode_front, feasible, balance, pareto

try:
    import pymoo                                              # noqa: F401
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize as pymoo_minimize
    HAVE_PYMOO = True
except Exception:                                             # pragma: no cover
    HAVE_PYMOO = False

__all__ = ['HAVE_PYMOO', 'nsga2', 'spea2', 'moead', 'weighted_sum',
           'epsilon_constraint', 'pymoo_run', 'PYMOO_ALGOS']


# ---------------------------------------------------------------------------
#  shared machinery
# ---------------------------------------------------------------------------
def _bits_to_mask(bits):
    m = 0
    for v, b in enumerate(bits):
        if b:
            m |= 1 << v
    return m


def _repair_bits(bits, n, eps, rng):
    """Make a binary genome feasible by moving the excess side's members."""
    bits = bits.astype(bool).copy()
    guard = 0
    while not feasible(_bits_to_mask(bits), n, eps) and guard < 4 * n:
        guard += 1
        a = int(bits.sum())
        if 2 * a > n:
            idx = np.flatnonzero(bits)
        else:
            idx = np.flatnonzero(~bits)
        if len(idx) == 0:
            break
        bits[int(rng.choice(idx))] ^= True
    return bits


def _eval(hg, bits, n):
    st = CostState(hg, _bits_to_mask(bits))
    return st.E, st.K


def _front_of(hg, pop, n):
    """All mode-front points contributed by a population."""
    out = []
    for bits in pop:
        m = _bits_to_mask(bits)
        st = CostState(hg, m)
        for E, K in mode_front(st.costs()):
            out.append((E, K, m))
    return pareto(out)


def _init_pop(n, size, eps, rng, seeds=None):
    pop = []
    for s in (seeds or []):
        bits = np.array([(s >> v) & 1 for v in range(n)], dtype=bool)
        pop.append(bits)
        if len(pop) >= size:
            break
    while len(pop) < size:
        bits = np.zeros(n, dtype=bool)
        bits[rng.permutation(n)[:n // 2]] = True
        pop.append(_repair_bits(bits, n, eps, rng))
    return pop


# ---------------------------------------------------------------------------
#  hypergraph-aware variation operators
# ---------------------------------------------------------------------------
#  Generic binary crossover and bitflip mutation treat the genome as an
#  unstructured bit string: two-point crossover cuts at an arbitrary qubit
#  index, and every qubit mutates with the same probability 1/n.  Neither
#  consults the hypergraph, so an evolutionary run explores a neighbourhood
#  that PRISM -- whose moves are drawn from the incidence structure -- does
#  not.  Comparing them then measures operator design as much as search
#  strategy.
#
#  These two operators close that gap.  Both read `hg` and nothing else.
# ---------------------------------------------------------------------------
def boundary_weights(hg, mask, n):
    """Per-qubit count of severed nets incident to that qubit.

    A qubit with a high count sits on the cut and moving it changes the cost;
    a qubit with zero is interior and flipping it can only make things worse.
    This is the hypergraph analogue of the light-cone proposal bias in
    ``prism._lc_weights``.
    """
    full = (1 << n) - 1
    other = full ^ mask
    w = np.zeros(n)
    for e in hg.nets:
        pm = e.pin_mask
        if (pm & mask) and (pm & other):
            for q in e.pins:
                w[q] += 1.0
    return w


def hg_mutate(hg, bits, n, eps, rng, expected=2.0, floor=0.05):
    """Flip qubits in proportion to how much of the cut they carry.

    ``expected`` sets the mean number of flips, so the mutation rate is
    comparable to ``BitflipMutation(prob=1/n)`` in aggregate; what changes is
    *where* the flips land.  ``floor`` keeps a small uniform component so an
    interior qubit is never unreachable and the chain stays irreducible.
    """
    w = boundary_weights(hg, _bits_to_mask(bits), n)
    w = w + floor * max(w.max(), 1.0)
    p = np.clip(w / w.sum() * expected, 0.0, 1.0)
    child = bits ^ (rng.random(n) < p)
    return _repair_bits(child, n, eps, rng)


def hg_crossover(hg, a, b, n, eps, rng):
    """Inherit one whole net from ``a``, the rest of the register from ``b``.

    Two-point crossover on an arbitrary qubit ordering splits nets at
    meaningless boundaries.  Taking a net's pins as a unit preserves the thing
    the objective actually charges for --- a packet whose members all sit on
    one side costs nothing, and that structure survives recombination here
    instead of being cut in half.
    """
    if not hg.nets:
        return _crossover_mutate(a, b, n, eps, rng, hg=hg)
    child = b.copy()
    e = hg.nets[int(rng.integers(len(hg.nets)))]
    for q in e.pins:
        child[q] = a[q]
    return _repair_bits(child, n, eps, rng)


def _crossover_mutate(a, b, n, eps, rng, p_mut=None, hg=None):
    """Recombine and mutate one child.

    With ``hg`` supplied this uses the hypergraph-aware operators, so the
    in-tree optimisers explore the same neighbourhood as the pymoo ones and as
    PRISM; the two MOO families then differ only by library and selection
    scheme.  Without it, the original generic binary behaviour.
    """
    if hg is not None:
        return hg_mutate(hg, hg_crossover(hg, a, b, n, eps, rng), n, eps, rng)
    p_mut = p_mut if p_mut is not None else 1.0 / n
    cut = int(rng.integers(1, n))
    child = np.concatenate([a[:cut], b[cut:]])
    flip = rng.random(n) < p_mut
    child = child ^ flip
    return _repair_bits(child, n, eps, rng)


def _nds(F):
    """Fast non-dominated sort; returns a list of fronts (index lists)."""
    N = len(F)
    S = [[] for _ in range(N)]
    nd = np.zeros(N, dtype=int)
    fronts = [[]]
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if (F[p][0] <= F[q][0] and F[p][1] <= F[q][1]
                    and (F[p][0] < F[q][0] or F[p][1] < F[q][1])):
                S[p].append(q)
            elif (F[q][0] <= F[p][0] and F[q][1] <= F[p][1]
                  and (F[q][0] < F[p][0] or F[q][1] < F[p][1])):
                nd[p] += 1
        if nd[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                nd[q] -= 1
                if nd[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def _crowding(F, idx):
    d = np.zeros(len(idx))
    Fi = np.array([F[i] for i in idx], float)
    for k in range(2):
        o = np.argsort(Fi[:, k])
        d[o[0]] = d[o[-1]] = np.inf
        rng_k = Fi[o[-1], k] - Fi[o[0], k]
        if rng_k <= 0:
            continue
        for j in range(1, len(idx) - 1):
            d[o[j]] += (Fi[o[j + 1], k] - Fi[o[j - 1], k]) / rng_k
    return d


# ---------------------------------------------------------------------------
#  native algorithms
# ---------------------------------------------------------------------------
def nsga2(n, hg, pop_size=40, generations=30, eps=0.10, seed=0, seeds=None,
          **kw):
    """NSGA-II: fast non-dominated sort with crowding-distance selection."""
    rng = np.random.default_rng(seed)
    pop = _init_pop(n, pop_size, eps, rng, seeds)
    F = [_eval(hg, b, n) for b in pop]
    for _ in range(generations):
        kids = []
        for _ in range(pop_size):
            i, j = rng.integers(0, len(pop), 2)
            kids.append(_crossover_mutate(pop[i], pop[j], n, eps, rng, hg=hg))
        allp = pop + kids
        allF = F + [_eval(hg, b, n) for b in kids]
        fronts, newp, newF = _nds(allF), [], []
        for fr in fronts:
            if len(newp) + len(fr) <= pop_size:
                newp += [allp[i] for i in fr]
                newF += [allF[i] for i in fr]
            else:
                cd = _crowding(allF, fr)
                order = np.argsort(-cd)
                for t in order[:pop_size - len(newp)]:
                    newp.append(allp[fr[t]])
                    newF.append(allF[fr[t]])
                break
        pop, F = newp, newF
    return _front_of(hg, pop, n)


def spea2(n, hg, pop_size=40, generations=30, archive=20, eps=0.10, seed=0,
          seeds=None, **kw):
    """SPEA2: strength-based fitness with k-th nearest-neighbour density."""
    rng = np.random.default_rng(seed)
    pop = _init_pop(n, pop_size, eps, rng, seeds)
    arch = []
    for _ in range(generations):
        union = pop + arch
        F = [_eval(hg, b, n) for b in union]
        N = len(union)
        strength = np.zeros(N)
        for i in range(N):
            for j in range(N):
                if i != j and F[i][0] <= F[j][0] and F[i][1] <= F[j][1] \
                        and (F[i][0] < F[j][0] or F[i][1] < F[j][1]):
                    strength[i] += 1
        raw = np.zeros(N)
        for i in range(N):
            for j in range(N):
                if i != j and F[j][0] <= F[i][0] and F[j][1] <= F[i][1] \
                        and (F[j][0] < F[i][0] or F[j][1] < F[i][1]):
                    raw[i] += strength[j]
        A = np.array(F, float)
        rngs = np.maximum(A.max(0) - A.min(0), 1e-12)
        D = np.sqrt((((A[:, None, :] - A[None, :, :]) / rngs) ** 2).sum(-1))
        k = max(1, int(math.sqrt(N)))
        dens = 1.0 / (np.sort(D, axis=1)[:, min(k, N - 1)] + 2.0)
        fit = raw + dens
        keep = np.argsort(fit)[:archive]
        arch = [union[i] for i in keep]
        kids = []
        for _ in range(pop_size):
            i, j = rng.integers(0, len(arch), 2)
            kids.append(_crossover_mutate(arch[i], arch[j], n, eps, rng, hg=hg))
        pop = kids
    return _front_of(hg, pop + arch, n)


def moead(n, hg, pop_size=40, generations=30, neighbours=8, eps=0.10, seed=0,
          seeds=None, **kw):
    """MOEA/D with Tchebycheff decomposition over a uniform weight simplex."""
    rng = np.random.default_rng(seed)
    W = np.stack([np.linspace(0.02, 0.98, pop_size),
                  1 - np.linspace(0.02, 0.98, pop_size)], 1)
    D = np.abs(W[:, None, 0] - W[None, :, 0])
    B = np.argsort(D, axis=1)[:, :neighbours]
    pop = _init_pop(n, pop_size, eps, rng, seeds)
    F = np.array([_eval(hg, b, n) for b in pop], float)
    z = F.min(0)
    scale = np.maximum(F.max(0) - F.min(0), 1e-9)

    def tcheby(f, w):
        return float(np.max(w * np.abs((f - z) / scale)))

    for _ in range(generations):
        for i in range(pop_size):
            a, b = rng.choice(B[i], 2, replace=False)
            child = _crossover_mutate(pop[a], pop[b], n, eps, rng, hg=hg)
            fc = np.array(_eval(hg, child, n), float)
            z = np.minimum(z, fc)
            for j in B[i]:
                if tcheby(fc, W[j]) <= tcheby(F[j], W[j]):
                    pop[j], F[j] = child, fc
    return _front_of(hg, pop, n)


def weighted_sum(n, hg, points=15, restarts=3, eps=0.10, seed=0, seeds=None,
                 **kw):
    """Classical scalarisation sweep: minimise ``w.E + (1-w).K`` by descent.

    Only reaches the convex hull of the front by construction, which is
    precisely the classical limitation a decomposition ensemble is meant to
    avoid; included so that limitation is visible rather than assumed.
    """
    rng = np.random.default_rng(seed)
    from .baselines import part_spectral
    ref = CostState(hg, part_spectral(n, hg))
    norm = (max(ref.E, 1.0), max(ref.K, 1e-9))
    out = []
    for w0 in np.linspace(0.02, 0.98, points):
        w = (w0, 1 - w0)
        best = None
        for r in range(restarts):
            m = (part_spectral(n, hg) if r == 0 else
                 _bits_to_mask(_repair_bits(rng.random(n) < 0.5, n, eps, rng)))
            st = CostState(hg, m)
            s = w[0] * st.E / norm[0] + w[1] * st.K / norm[1]
            improved = True
            while improved:
                improved = False
                for v in range(n):
                    cand = st.mask ^ (1 << v)
                    if not feasible(cand, n, eps):
                        continue
                    E, K, _ = st.peek(v)
                    s2 = w[0] * E / norm[0] + w[1] * K / norm[1]
                    if s2 < s - 1e-12:
                        st.flip(v)
                        s, improved = s2, True
                        break
            if best is None or s < best[0]:
                best = (s, st.mask)
        st = CostState(hg, best[1])
        for E, K in mode_front(st.costs()):
            out.append((E, K, best[1]))
    return pareto(out)


def epsilon_constraint(n, hg, points=12, eps=0.10, seed=0, seeds=None, **kw):
    """Minimise K subject to ``E <= bound``, swept over the bound.

    Unlike a weighted sum this can reach concave parts of the front.
    """
    from .baselines import part_spectral
    rng = np.random.default_rng(seed)
    base = CostState(hg, part_spectral(n, hg))
    hi = max(base.E, 1.0)
    out = []
    for bound in np.linspace(0.15 * hi, 1.6 * hi, points):
        st = CostState(hg, part_spectral(n, hg))
        improved = True
        while improved:
            improved = False
            best_v, best = None, None
            for v in range(n):
                cand = st.mask ^ (1 << v)
                if not feasible(cand, n, eps):
                    continue
                E, K, _ = st.peek(v)
                pen = K + 10.0 * max(0.0, E - bound)
                cur = st.K + 10.0 * max(0.0, st.E - bound)
                if best is None or pen < best:
                    if pen < cur - 1e-12:
                        best_v, best = v, pen
            if best_v is not None:
                st.flip(best_v)
                improved = True
        for E, K in mode_front(st.costs()):
            out.append((E, K, st.mask))
    return pareto(out)


# ---------------------------------------------------------------------------
#  pymoo adapters
# ---------------------------------------------------------------------------
PYMOO_ALGOS = ('pymoo_nsga2', 'pymoo_nsga3', 'pymoo_sms', 'pymoo_age',
               'pymoo_moead')


def pymoo_run(algo, n, hg, pop_size=40, generations=30, eps=0.10, seed=0,
              seeds=None, hypergraph_ops=True, **kw):
    """Run a pymoo algorithm on the binary partition problem.

    Feasibility is handled by repairing the genome inside ``_evaluate``, so the
    algorithm always sees balanced solutions and no constraint machinery is
    needed.

    ``seeds`` is honoured.  It used to be accepted and silently dropped, which
    made every pymoo row an unseeded run competing against warm-started PRISM
    rows -- a difference in starting point reported as a difference in method.
    The seeds now occupy the first rows of the initial population, exactly as
    :func:`_init_pop` places them for the in-tree optimisers, so both families
    begin from the same partitions.

    ``hypergraph_ops`` (default ``True``) replaces pymoo's generic binary
    operators with :func:`hg_crossover` and :func:`hg_mutate`, so the search
    moves through the hypergraph rather than through an unstructured bit
    string.  Without it a pymoo row differs from PRISM in *two* ways at once
    --- algorithm and neighbourhood --- and the table cannot attribute the
    difference to either.  Set it ``False`` for the ablation.
    """
    if not HAVE_PYMOO:
        raise RuntimeError('pymoo is not installed')
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.crossover import Crossover
    from pymoo.core.mutation import Mutation
    from pymoo.optimize import minimize
    from pymoo.operators.sampling.rnd import BinaryRandomSampling
    from pymoo.operators.crossover.pntx import TwoPointCrossover
    from pymoo.operators.mutation.bitflip import BitflipMutation

    rng = np.random.default_rng(seed)

    class HGMutation(Mutation):
        """Bitflip biased towards qubits carrying the cut (`hg_mutate`)."""

        def _do(self, problem, X, **kw):
            X = np.asarray(X, dtype=bool).copy()
            for i in range(X.shape[0]):
                X[i] = hg_mutate(hg, X[i], n, eps, rng)
            return X

    class HGCrossover(Crossover):
        """Net-preserving recombination (`hg_crossover`): two parents in,
        two children out, each inheriting one whole net from the other."""

        def __init__(self):
            super().__init__(2, 2)

        def _do(self, problem, X, **kw):
            _, n_mate, n_var = X.shape
            Y = np.empty((2, n_mate, n_var), dtype=bool)
            for k in range(n_mate):
                a = np.asarray(X[0, k], dtype=bool)
                b = np.asarray(X[1, k], dtype=bool)
                Y[0, k] = hg_crossover(hg, a, b, n, eps, rng)
                Y[1, k] = hg_crossover(hg, b, a, n, eps, rng)
            return Y

    class PartitionProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=n, n_obj=2, n_constr=0, xl=0, xu=1,
                             vtype=bool)

        def _evaluate(self, x, out, *a, **k):
            bits = _repair_bits(np.asarray(x, dtype=bool), n, eps, rng)
            st = CostState(hg, _bits_to_mask(bits))
            out['F'] = [st.E, st.K]

    prob = PartitionProblem()

    # Initial population: seeds first, random balanced cuts after.  pymoo
    # accepts a raw (pop_size, n) array in place of a Sampling object and uses
    # it verbatim as generation zero.
    if seeds:
        X0 = np.array(_init_pop(n, pop_size, eps, rng, seeds=seeds), dtype=bool)
        sampling = X0
    else:
        sampling = BinaryRandomSampling()

    # `hypergraph_ops=False` restores the generic binary operators, which is
    # how the ablation "does structure-awareness matter?" is run.
    if hypergraph_ops:
        xover, mut = HGCrossover(), HGMutation()
    else:
        xover, mut = TwoPointCrossover(), BitflipMutation(prob=1.0 / n)

    common = dict(sampling=sampling, crossover=xover, mutation=mut,
                  eliminate_duplicates=True)
    if algo == 'pymoo_nsga2':
        from pymoo.algorithms.moo.nsga2 import NSGA2
        alg = NSGA2(pop_size=pop_size, **common)
    elif algo == 'pymoo_nsga3':
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.util.ref_dirs import get_reference_directions
        ref = get_reference_directions('das-dennis', 2, n_partitions=pop_size - 1)
        alg = NSGA3(ref_dirs=ref, pop_size=pop_size, **common)
    elif algo == 'pymoo_sms':
        from pymoo.algorithms.moo.sms import SMSEMOA
        alg = SMSEMOA(pop_size=pop_size, **common)
    elif algo == 'pymoo_age':
        from pymoo.algorithms.moo.age import AGEMOEA
        alg = AGEMOEA(pop_size=pop_size, **common)
    elif algo == 'pymoo_moead':
        from pymoo.algorithms.moo.moead import MOEAD
        from pymoo.util.ref_dirs import get_reference_directions
        ref = get_reference_directions('das-dennis', 2, n_partitions=pop_size - 1)
        # MOEA/D takes ref_dirs positionally and rejects eliminate_duplicates,
        # so it is built separately -- but it gets the same seeded sampling.
        alg = MOEAD(ref, n_neighbors=8, sampling=sampling,
                    crossover=xover, mutation=mut)
    else:
        raise ValueError(f'unknown pymoo algorithm {algo!r}')

    res = minimize(prob, alg, ('n_gen', generations), seed=seed, verbose=False)
    X = np.atleast_2d(res.X)
    pop = [_repair_bits(np.asarray(x, dtype=bool), n, eps, rng) for x in X]
    return _front_of(hg, pop, n)
