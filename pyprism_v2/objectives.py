"""
pyprism_v2.objectives
=====================
The two objectives, the CD/CK mode front, and an incremental evaluator.

Both objectives are **static**: no statevector, no shots, no backend.  Each is
a function of the gate list and the bipartition alone,

    E(A,B) = sum over crossings of e_i        [ebits, exact distribution]
    K(A,B) = sum over crossings of log g_i    [knitting, quasiprobability]

Why there is a front
--------------------
As *totals* the two are near-collinear: both grow with the number of
crossings, so a naive ``(E, K)`` front degenerates to a point.  They are the
same boundary read in two currencies, not two costs paid together.

The front comes from the **per-crossing mode choice**.  Each crossing is
executed one way or the other -- distributed exactly by gate teleportation at
``e_i`` ebits, or simulated by quasiprobability at ``log gamma_i``.  Choosing a
subset ``S`` to distribute gives ``E = sum_{i in S} e_i`` and
``K = sum_{i not in S} log g_i``.  Sorting the crossings by
``log(gamma_i)/e_i`` descending -- "dearest to knit per ebit spent" -- and
distributing a prefix generates every non-dominated point.  The endpoints are
pure knitting (``E = 0``, no quantum link at all) and pure distribution
(``K = 0``), so sweeping the preference vector sweeps the entanglement budget.

Incremental evaluation
----------------------
A full sweep is ``O(|nets|)``.  For ``n = 200`` a circuit has tens of thousands
of nets and a local search makes thousands of moves, which is intractable if
every move rescans.  :class:`CostState` keeps the running cost and, on a
single-qubit flip, rescans only the nets incident to that qubit -- typically
two orders of magnitude fewer.
"""
from __future__ import annotations

import math

import numpy as np

from .gates import spec as _spec, gamma_of_gate as _gamma

__all__ = ['net_costs', 'evaluate', 'mode_front', 'scalarised', 'knee',
           'balance', 'feasible', 'CostState', 'objective_correlation',
           'hypervolume', 'pareto']

_PACKET_CACHE: dict = {}


def _gamma_from_phases(phases):
    p = np.asarray(phases, dtype=float)
    if p.size < 2:
        return 1.0
    d = p[:, None] - p[None, :]
    return float(1.0 + 2.0 * np.abs(np.sin(d / 2.0)).max())


def packet_cost(net, mask):
    """``(ebits, log gamma)`` of a v-rooted packet under ``mask``.

    One cat-entangler broadcast places a copy of the root's computational-basis
    value on the far block, and every member whose remote operands all sit on
    that far block is then executed locally against the copy.  Those members
    share a single ebit -- the connectivity-1 metric, one ebit for k = 2.

    A member whose remote operands straddle the boundary is **not** served by
    that copy.  Its branch unitary still acts on both blocks, so it must be
    distributed in its own right, and is charged its exact cost at the realised
    cut.  This only arises for members of arity three or more: a two-qubit
    member has a single remote operand, which cannot straddle anything.
    Omitting this term under-charges the cut, which is the unsafe direction.

    Knitting: the served members commute with Z at the root and so compose into
    a single controlled-W on the far block, whose eigenphases are the subset
    sums of the per-gate branch phases; hence ``gamma <= 3`` for all of them
    together.  Straddling members multiply their own gamma on top.
    """
    root_in_A = (mask >> net.root) & 1
    sig, opaque, bit = 0, False, 1
    e_own, k_own = 0, 0.0
    for remote, phases, name, params, qubits in net.members:
        sides = {(mask >> q) & 1 for q in remote}
        if len(sides) > 1:                      # straddles: copy insufficient
            s = _spec(name)
            A_local = frozenset(j for j, q in enumerate(qubits)
                                if (mask >> q) & 1)
            if A_local and len(A_local) != s.m:
                e_own += s.ebits(A_local)
                k_own += math.log(max(_gamma(name, params, A_local), 1.0))
            bit <<= 1
            continue
        if sides and sides.pop() != root_in_A:  # served by the shared copy
            if phases is None:
                opaque = True
            else:
                sig |= bit
        bit <<= 1
    shared = bool(sig or opaque)
    if not shared and not e_own:
        return 0, 0.0
    if not shared:
        return e_own, k_own
    if opaque:                      # a non-diagonal member crosses
        return 1 + e_own, math.log(3.0) + k_own
    # Content-addressed, deliberately.  `gamma` is a function of the phases of
    # the crossing members and of nothing else, so the phases *are* the key.
    # Keying on `(net.nid, sig)` is wrong: `nid` is an index within one
    # hypergraph and restarts at zero for every circuit, so in a sweep the
    # first instance to populate a slot would silently hand its gamma to every
    # later instance whose net index and crossing pattern happened to match.
    flat = tuple(p for i, m in enumerate(net.members)
                 if (sig >> i) & 1 for p in m[1])
    hit = _PACKET_CACHE.get(flat)
    if hit is None:
        if len(flat) <= 14:
            sums = np.zeros(1)
            for p in flat:
                sums = np.concatenate([sums, sums + p])
            sums = np.unique(np.round(np.mod(sums, 2 * math.pi), 9))
            gam = _gamma_from_phases(sums)
        else:
            gam = 3.0               # the universal controlled-gate bound
        hit = (1, math.log(max(gam, 1.0)))
        _PACKET_CACHE[flat] = hit
    return hit[0] + e_own, hit[1] + k_own


def gate_cost_exact(net, mask):
    """``(ebits, log gamma)`` of a single-gate net at the realised local cut."""
    s = _spec(net.name)
    A_local = frozenset(j for j, q in enumerate(net.qubits) if (mask >> q) & 1)
    if not A_local or len(A_local) == s.m:
        return 0, 0.0
    return (s.ebits(A_local),
            math.log(max(_gamma(net.name, net.params, A_local), 1.0)))


def _one_net_cost(e, mask, exact):
    if exact:
        return (packet_cost(e, mask) if e.kind == 'packet'
                else gate_cost_exact(e, mask))
    return (e.w_ebit, e.w_logk)


def net_costs(hg, mask, exact=None):
    """Per-cut-net costs ``[(e_i, log gamma_i), ...]``."""
    if exact is None:
        exact = (hg.track == 2)
    full = (1 << hg.n) - 1
    out = []
    for e in hg.nets:
        pm = e.pin_mask
        if not (pm & mask) or not (pm & (full ^ mask)):
            continue
        out.append(_one_net_cost(e, mask, exact))
    return out


def evaluate(hg, mask, exact=None):
    """The two endpoint currencies of the boundary induced by ``mask``."""
    cs = net_costs(hg, mask, exact)
    return sum(c[0] for c in cs), sum(c[1] for c in cs)


def mode_front(costs):
    """Exact Pareto front over CD/CK mode assignments for one partition."""
    if not costs:
        return [(0.0, 0.0)]
    order = sorted(costs, key=lambda c: -(c[1] / max(c[0], 1e-9)))
    E, K = 0.0, float(sum(c[1] for c in order))
    pts, seen = [(0.0, K)], {(0.0, round(K, 9))}
    for de, dk in order:
        E += de
        K = max(K - dk, 0.0)
        key = (round(E, 9), round(K, 9))
        if key not in seen:
            seen.add(key)
            pts.append((E, K))
    return pts


def scalarised(hg, mask, w, norm, exact=None):
    """``min`` over mode assignments of the normalised weighted sum."""
    best = None
    for E, K in mode_front(net_costs(hg, mask, exact)):
        s = w[0] * E / norm[0] + w[1] * K / norm[1]
        if best is None or s < best[0]:
            best = (s, E, K)
    return best


def knee(front):
    """Closest point to the utopia corner after min-max normalisation."""
    if not front:
        return None
    E = np.array([p[0] for p in front], float)
    K = np.array([p[1] for p in front], float)
    en = (E - E.min()) / max(float(np.ptp(E)), 1e-12)
    kn = (K - K.min()) / max(float(np.ptp(K)), 1e-12)
    return front[int(np.argmin(en ** 2 + kn ** 2))]


def balance(mask, n):
    return abs(2 * bin(mask).count('1') - n) / n


def feasible(mask, n, eps=0.10):
    return 0 < bin(mask).count('1') < n and balance(mask, n) <= eps + 1e-12


# ---------------------------------------------------------------------------
#  incremental evaluator
# ---------------------------------------------------------------------------
class CostState:
    """Running ``(E, K)`` with ``O(deg(v))`` single-qubit flips.

    ``hg.incidence[v]`` lists the nets touching qubit ``v``; flipping ``v``
    can only change those, so the update rescans them instead of the whole
    hypergraph.  Everything else is unchanged, which is what makes local search
    feasible at ``n = 200``.
    """

    __slots__ = ('hg', 'exact', 'mask', 'full', 'per', 'E', 'K',
                 'a', 'b', 'S')

    def __init__(self, hg, mask, exact=None, pref=None):
        self.hg = hg
        self.exact = (hg.track == 2) if exact is None else exact
        self.full = (1 << hg.n) - 1
        # `pref = (a, b)` turns on the mode-aware scalarisation, see `set_pref`
        self.a, self.b = pref if pref else (0.0, 0.0)
        if not hg.incidence:
            hg.build_incidence()
        self.reset(mask)

    def set_pref(self, a, b):
        """Track the mode-aware scalarised cost as well as the currencies.

        For weights ``w`` and normalisers, the scalarisation that PRISM
        minimises is ``min`` over CD/CK mode assignments of
        ``w0 E/normE + w1 K/normK``.  Because each crossing is assigned
        independently, that minimum *decomposes per net*::

            S = sum_i min(a e_i, b k_i),   a = w0/normE,  b = w1/normK

        so it can be maintained incrementally exactly like E and K.  Without
        this the scalarisation collapses onto ``E + K``, the preference axis
        stops discriminating, and the ladder loses its spread.
        """
        self.a, self.b = a, b
        self.S = sum(min(a * c[0], b * c[1])
                     for c in self.per if c is not None)
        return self

    def reset(self, mask):
        self.mask = mask
        self.per = [None] * len(self.hg.nets)
        E = K = S = 0.0
        a, b = self.a, self.b
        for i, e in enumerate(self.hg.nets):
            pm = e.pin_mask
            if (pm & mask) and (pm & (self.full ^ mask)):
                c = _one_net_cost(e, mask, self.exact)
                self.per[i] = c
                E += c[0]
                K += c[1]
                S += min(a * c[0], b * c[1])
        self.E, self.K, self.S = E, K, S
        return self

    def flip(self, v):
        """Toggle qubit ``v`` and return the new ``(E, K)``."""
        mask = self.mask ^ (1 << v)
        E, K, S = self.E, self.K, self.S
        a, b = self.a, self.b
        for i in self.hg.incidence[v]:
            old = self.per[i]
            if old is not None:
                E -= old[0]
                K -= old[1]
                S -= min(a * old[0], b * old[1])
            e = self.hg.nets[i]
            pm = e.pin_mask
            if (pm & mask) and (pm & (self.full ^ mask)):
                c = _one_net_cost(e, mask, self.exact)
                self.per[i] = c
                E += c[0]
                K += c[1]
                S += min(a * c[0], b * c[1])
            else:
                self.per[i] = None
        self.mask, self.E, self.K, self.S = mask, E, K, S
        return E, K

    def peek(self, v):
        """``(E, K, S)`` after flipping ``v``, without committing."""
        mask = self.mask ^ (1 << v)
        E, K, S = self.E, self.K, self.S
        a, b = self.a, self.b
        for i in self.hg.incidence[v]:
            old = self.per[i]
            if old is not None:
                E -= old[0]
                K -= old[1]
                S -= min(a * old[0], b * old[1])
            e = self.hg.nets[i]
            pm = e.pin_mask
            if (pm & mask) and (pm & (self.full ^ mask)):
                c = _one_net_cost(e, mask, self.exact)
                E += c[0]
                K += c[1]
                S += min(a * c[0], b * c[1])
        return E, K, S

    def costs(self):
        return [c for c in self.per if c is not None]


# ---------------------------------------------------------------------------
#  front metrics and diagnostics
# ---------------------------------------------------------------------------
def pareto(points):
    """Non-dominated subset of ``[(E, K, payload), ...]``, minimising both."""
    out = []
    for p in sorted(points, key=lambda z: (z[0], z[1])):
        if not out or p[1] < out[-1][1] - 1e-12:
            out.append(p)
    ded, seen = [], set()
    for p in out:
        key = (round(p[0], 9), round(p[1], 9))
        if key not in seen:
            seen.add(key)
            ded.append(p)
    return ded


def hypervolume(front, ref):
    """2-D hypervolume of a minimisation front against reference ``ref``."""
    pts = sorted([(p[0], p[1]) for p in front
                  if p[0] <= ref[0] and p[1] <= ref[1]])
    if not pts:
        return 0.0
    hv, prev_k = 0.0, ref[1]
    for E, K in pts:
        if K < prev_k:
            hv += (ref[0] - E) * (prev_k - K)
            prev_k = K
    return float(hv)


def igd(front, reference_front):
    """Inverted generational distance to a reference front."""
    if not front or not reference_front:
        return float('nan')
    F = np.array([[p[0], p[1]] for p in front], float)
    R = np.array([[p[0], p[1]] for p in reference_front], float)
    d = np.sqrt(((R[:, None, :] - F[None, :, :]) ** 2).sum(-1)).min(1)
    return float(d.mean())


def objective_correlation(hg, n, samples=300, eps=0.10, seed=0):
    """Spearman rho between the two currencies over random balanced cuts."""
    rng = np.random.default_rng(seed)
    Es, Ks = [], []
    for _ in range(samples):
        order = rng.permutation(n)
        m = sum(1 << int(q) for q in order[:n // 2])
        if not feasible(m, n, eps):
            continue
        E, K = evaluate(hg, m)
        Es.append(E)
        Ks.append(K)
    if len(Es) < 3:
        return 0.0, len(Es)

    def rank(x):
        x = np.asarray(x, float)
        o = np.argsort(x, kind='mergesort')
        r = np.empty(len(x), float)
        r[o] = np.arange(len(x), dtype=float)
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        s = np.zeros(len(cnt))
        np.add.at(s, inv, r)
        return (s / cnt)[inv]

    ra, rb = rank(Es), rank(Ks)
    ra -= ra.mean()
    rb -= rb.mean()
    d = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return (float(ra @ rb / d) if d > 0 else 0.0), len(Es)
