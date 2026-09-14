"""
pyprism_v2.circuits
===================
Random circuit generation against a coupling map.

A circuit **C** is a depth-`D` layout `L = (L_1, ..., L_D)` on qubits
`Q = {0, ..., n-1}`; each gate is `g = (U_g, q_g)` with support `q_g` a subset
of `Q`.

Two modelling choices matter for a partitioning benchmark.

**The coupling map.**  On an all-to-all random circuit every bipartition cuts
about the same number of gates, so there is nothing to optimise and the
benchmark is degenerate.  Real random-circuit benchmarks are written against a
device topology, and it is that locality which makes a partition meaningful.

**Parametrised entanglers.**  A fixed-angle entangler has ``log(gamma)/e``
pinned near 1.0 --- CX and CZ give ``(1, log 3)``, SWAP and iSWAP give
``(2, log 7)`` --- so a circuit built only from those makes the two objectives
collinear.  Hardware-native random circuits use parametrised entanglers
(Sycamore-style fSim, IBM-style RZZ/CPHASE), whose ``gamma = 1 + 2|sin(t/2)|``
ranges over ``[1, 3]``, and that is what lets the objectives conflict.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .gates import GATE_LIB, spec

__all__ = ['Gate', 'FAMILIES', 'coupling_map', 'random_circuit',
           'circuit_symmetry', 'circuit_stats']


@dataclass(frozen=True)
class Gate:
    gid: int
    name: str
    qubits: tuple            # q_g, in the gate's own operand order
    params: tuple
    layer: int

    @property
    def m(self):
        return len(self.qubits)


#: family -> (single-qubit pool, entangler pool); entries are
#: ``(name, n_params, sampling weight)`` and parameters are drawn from [0, 2pi).
FAMILIES: dict = {
    # generic hardware-native random circuit
    'rqc':  ([('RZ', 1, 2), ('SX', 0, 2), ('RX', 1, 1), ('RY', 1, 1),
              ('U3', 3, 1), ('H', 0, 1)],
             [('CPHASE', 1, 3), ('CRZ', 1, 2), ('CZ', 0, 1), ('CX', 0, 1),
              ('ISWAP', 0, 1), ('SWAP', 0, 1), ('CCX', 0, 0.4)]),
    # U(1)_N-covariant throughout: particle number is conserved
    'u1':   ([('RZ', 1, 2), ('PHASE', 1, 2), ('S', 0, 1), ('T', 0, 1)],
             [('CPHASE', 1, 3), ('CRZ', 1, 2), ('ISWAP', 0, 1),
              ('SWAP', 0, 1), ('CZ', 0, 1)]),
    # fully diagonal entanglers: maximal packing headroom
    'diag': ([('RZ', 1, 2), ('SX', 0, 1), ('RX', 1, 1)],
             [('CPHASE', 1, 3), ('CRZ', 1, 2), ('CZ', 0, 1)]),
    # QAOA-like: diagonal cost layer, RX mixer (which breaks U(1))
    'qaoa': ([('RX', 1, 1)], [('CPHASE', 1, 3), ('CZ', 0, 1)]),
    # Clifford+T style, all fixed-angle: the objectives collapse here
    'clifford': ([('H', 0, 1), ('S', 0, 1), ('T', 0, 1), ('X', 0, 1)],
                 [('CX', 0, 3), ('CZ', 0, 2), ('SWAP', 0, 1)]),
}


def coupling_map(n, topology='grid'):
    """Adjacency of the device the circuit is written for."""
    adj = defaultdict(set)

    def link(i, j):
        adj[i].add(j)
        adj[j].add(i)

    if topology == 'full':
        for i in range(n):
            for j in range(i + 1, n):
                link(i, j)
    elif topology == 'line':
        for i in range(n - 1):
            link(i, i + 1)
    elif topology == 'ring':
        for i in range(n):
            link(i, (i + 1) % n)
    elif topology == 'grid':
        w = int(math.ceil(math.sqrt(n)))
        for i in range(n):
            r, c = divmod(i, w)
            if c + 1 < w and i + 1 < n:
                link(i, i + 1)
            if (r + 1) * w + c < n:
                link(i, (r + 1) * w + c)
    elif topology == 'heavy-hex':
        # a linear spine with every third qubit carrying a pendant, which is
        # the connectivity pattern IBM's heavy-hex lattices are built from
        for i in range(n - 1):
            link(i, i + 1)
        for i in range(0, n - 3, 6):
            link(i, min(i + 3, n - 1))
    else:
        raise ValueError(f'unknown topology {topology!r}')
    return {v: sorted(adj[v]) for v in range(n)}


def _cum(pool):
    w = np.array([p[2] for p in pool], float)
    return np.cumsum(w / w.sum())


def _pick(rng, pool, cum):
    """Weighted choice by inverse-CDF; `rng.choice(..., p=...)` rebuilds an
    alias table on every call, which dominates generation at n = 200."""
    return pool[int(np.searchsorted(cum, rng.random()))]


def random_circuit(n, depth, seed=0, family='rqc', topology='grid',
                   two_qubit_frac=0.62, multi_frac=0.06, relabel=True):
    """Return ``(layout, gates, adj)``.

    ``layout`` is the depth-D list of layers, each a list of :class:`Gate`;
    ``gates`` is the flat list in time order; ``adj`` is the coupling map.

    ``relabel`` applies a random permutation to the qubit indices once the
    circuit has been laid out.  Without it the index order coincides with the
    device geometry, so "the first n/2 indices" *is* the optimal bisection of a
    grid and the trivial baseline ties with everything else --- a benchmark in
    which the trivial method wins measures nothing.  Relabelling leaves the
    structure intact but stops it being handed to the partitioner for free.
    """
    rng = np.random.default_rng(seed)
    pool1, pool2 = FAMILIES[family]
    pool2_small = [p for p in pool2 if GATE_LIB[p[0]][0] == 2]
    pool2_multi = [p for p in pool2 if GATE_LIB[p[0]][0] >= 3]
    cum1, cum2s = _cum(pool1), _cum(pool2_small)
    cum2m = _cum(pool2_multi) if pool2_multi else None
    adj = coupling_map(n, topology)
    layout, gates, gid = [], [], 0
    for t in range(depth):
        layer, busy = [], set()
        order = rng.permutation(n)
        for v in order:
            v = int(v)
            if v in busy:
                continue
            if rng.random() < two_qubit_frac:
                want_multi = pool2_multi and rng.random() < multi_frac
                nm, npar, _ = (_pick(rng, pool2_multi, cum2m) if want_multi
                               else _pick(rng, pool2_small, cum2s))
                arity = GATE_LIB[nm][0]
                free = [u for u in adj[v] if u not in busy]
                if len(free) < arity - 1:
                    continue
                pick = rng.choice(free, size=arity - 1, replace=False)
                qs = tuple([v] + [int(u) for u in pick])
                busy.update(qs)
            else:
                nm, npar, _ = _pick(rng, pool1, cum1)
                qs = (v,)
                busy.add(v)
            ps = tuple(float(x) for x in rng.uniform(0, 2 * math.pi, size=npar))
            g = Gate(gid, nm, qs, ps, t)
            layer.append(g)
            gates.append(g)
            gid += 1
        layout.append(layer)

    if relabel:
        perm = rng.permutation(n)
        layout = [[Gate(g.gid, g.name, tuple(int(perm[q]) for q in g.qubits),
                        g.params, g.layer) for g in layer] for layer in layout]
        gates = [g for layer in layout for g in layer]
        gates.sort(key=lambda g: g.gid)
        adj = {int(perm[v]): sorted(int(perm[u]) for u in us)
               for v, us in adj.items()}
    return layout, gates, adj


def circuit_symmetry(gates):
    """``Sym(C)`` over **all** gates --- single-qubit gates included.

    Testing only the two-qubit types is unsound: a circuit of RX mixers and CZ
    couplers would be reported U(1)-conserving although ``[R_x(t), N] != 0``.
    """
    diag = u1 = True
    for g in gates:
        s = spec(g.name)
        diag &= s.diagonal
        u1 &= s.u1
        if not (diag or u1):
            break
    return {'sigma_L_diagonal': bool(diag), 'sigma_N_u1': bool(u1)}


def circuit_stats(gates, n, depth):
    ent = [g for g in gates if g.m >= 2]
    return {'n': n, 'depth': depth, 'gates': len(gates),
            'single': sum(1 for g in gates if g.m == 1),
            'two': sum(1 for g in gates if g.m == 2),
            'multi': sum(1 for g in gates if g.m >= 3),
            'entanglers': len(ent),
            'diag_entanglers': sum(1 for g in ent if spec(g.name).diagonal),
            **circuit_symmetry(gates)}
