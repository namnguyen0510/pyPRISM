"""
pyprism_v2.lightcone
====================
Causal-cone propagation and the light-cone-augmented interaction graph.

Propagating causal support layer by layer: a multi-qubit gate merges the cones
of its operands, so ``u in cone[v]`` means an operation on ``u`` at an earlier
layer can influence ``v``.

``W[u, v]`` is the fraction of the depth for which ``u`` and ``v`` share causal
support.  Two qubits that became causally linked early and stayed linked are
expensive to separate whether or not a gate acts on them directly, because
every later gate on one propagates into the other's cone.

Scaling
-------
The dense form is ``O(D n^2)`` in time and ``O(n^2)`` in memory, which is fine
to ``n`` of a few hundred.  Cone sets are stored as Python ints used as
bitsets, so the per-layer merge is a handful of machine words rather than a
set union.
"""
from __future__ import annotations

import numpy as np

__all__ = ['light_cone_graph', 'lightcone_stats', 'centre_lightcone']


def light_cone_graph(n, layout):
    """Return ``(W, first, depth_to_full)``.

    ``W[u, v]``     fraction of depth for which u and v share causal support
    ``first[u, v]`` first layer at which u entered v's cone, or -1
    ``depth_to_full`` layer at which every cone first covers all of Q

    Implementation
    --------------
    The cone sets are held as a boolean matrix ``C``, ``C[v, u]`` meaning
    ``u in cone[v]``.  A gate merges rows by OR; the per-layer accumulation is
    then one vectorised ``W += C`` instead of an ``O(n^2)`` Python loop.

    Cones grow monotonically, and in a scrambling circuit they saturate after
    a depth of order ``log n`` --- typically layer 10 of several hundred.  Once
    ``C`` is all-ones nothing can change again, so the remaining ``D - t``
    layers are added in a single step.  That shortcut is what keeps the cost
    near-linear in depth rather than quadratic.
    """
    D = max(len(layout), 1)
    C = np.eye(n, dtype=bool)
    W = np.zeros((n, n))
    first = np.full((n, n), -1, dtype=np.int32)
    depth_to_full = -1
    for t, layer in enumerate(layout):
        for g in layer:
            if g.m < 2:
                continue
            idx = list(g.qubits)
            merged = C[idx].any(0)
            C[idx] = merged
        newly = C & (first < 0)
        if newly.any():
            first[newly] = t
        W += C
        if depth_to_full < 0 and C.all():
            depth_to_full = t
            W += C * float(D - 1 - t)      # nothing can change from here
            break
    np.fill_diagonal(W, 0.0)
    np.fill_diagonal(first, -1)
    W = (W + W.T) / (2.0 * D)
    return W, first, depth_to_full


def lightcone_stats(W, first, depth_to_full, n, D):
    off = ~np.eye(n, dtype=bool)
    linked = first[off] >= 0
    return {'mean_coupling': float(W[off].mean()),
            'linked_fraction': float(linked.mean()),
            'median_first_link': (float(np.median(first[off][linked]))
                                  if linked.any() else -1.0),
            'scrambling_depth': int(depth_to_full),
            'depth': int(D)}


def centre_lightcone(W, n):
    """Keep only above-median causal coupling, rescaled to ``[0, 1]``.

    The raw cone matrix is dense --- in a scrambling circuit almost every pair
    is causally linked --- so adding it directly to a graph substrate just
    contributes a near-uniform background and washes out the real weights.
    Centring on the median keeps the couplings that are stronger than typical,
    which is where the structure is.
    """
    off = ~np.eye(n, dtype=bool)
    lcn = np.clip(W - np.median(W[off]), 0.0, None)
    mx = lcn.max()
    return lcn / mx if mx > 0 else lcn
