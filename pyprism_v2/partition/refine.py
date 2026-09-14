"""
pyprism_v2.partition.refine
===========================
Refinement primitives shared by both PRISM searches.

Nothing here is part of either search.  These are operations on a partition
that has already been produced --- which is exactly why they live in their own
module rather than inside one of the two optimisers.

Why that placement matters
--------------------------
:func:`endpoint_polish` is applied to the *returned front*, after the ensembles
have finished.  Give it to one search and not the other and any comparison
between them measures the polish, not the search: PRISM would appear to beat
PRISM+ by a margin that is entirely post-processing.  Both call the same
function here with the same parameters, so a difference between the two rows in
the benchmark is a difference between a seven-stage cooled ensemble and a
two-dimensional ``(w, T)`` ladder, which is what those rows are meant to
compare.

The three pieces:

``bal_ok``          the balance predicate, expressed on a block size so an
                    operator can ask "may I move a qubit this way?" without
                    materialising the candidate mask.
``boundary``        ``(P(A), P(B))``, the qubits on the cut, optionally dilated.
``endpoint_polish`` a Fiduccia--Mattheyses descent on one currency, for one
                    endpoint of the front.
"""
from __future__ import annotations

from ..objectives import CostState, feasible

__all__ = ['bal_ok', 'boundary', 'endpoint_polish', 'teleport_point']


def teleport_point(front, label='', tol=1e-12):
    """The all-teleport endpoint of a front: cheapest ``E`` at ``K = 0``.

    Every caller that reports a single ``E`` per method wants this point, and
    every one of them used to spell it as::

        min((p for p in front if p[1] <= tol), key=lambda p: p[0])

    which raises a bare ``ValueError: min() arg is an empty sequence`` when the
    front has no ``K = 0`` point --- naming neither the method nor the reason.
    A front normally has one, because ``mode_front`` walks every crossing into
    the distributed mode and ends at ``K = 0``; it can lack one if the front
    was truncated or came back empty.

    So: prefer a genuine ``K <= tol`` point, fall back to the smallest ``K``
    available and say nothing (the caller re-prices on ``H2`` anyway), and
    raise something that names ``label`` if the front is empty.
    """
    if not front:
        raise ValueError(f'{label or "optimiser"} returned an empty front: '
                         f'no partition to report')
    exact = [p for p in front if p[1] <= tol]
    return min(exact or front, key=lambda p: (p[1] > tol, p[0]))


def bal_ok(p, n, eps):
    """Is a block of ``p`` qubits within the balance tolerance?

    The same predicate as :func:`~pyprism_v2.objectives.feasible`, expressed on
    the block size alone.
    """
    return 0 < p < n and abs(2 * p - n) <= eps * n + 1e-12


def boundary(n, hg, mask, hops=0, min_size=0):
    """``(P(A), P(B))``: qubits on the cut, optionally dilated by ``hops``.

    On a graph the boundary is wide, because edges are pairwise and a cut of
    ``c`` edges touches up to ``2c`` vertices.  On this substrate the cut
    concentrates on hub qubits: measured on a 100-qubit instance, a
    Kernighan--Lin partition severing 69 nets put every one of them on just 7
    distinct qubits, so ``|P(A)| x |P(B)| = 12`` candidate swaps --- a
    neighbourhood three orders of magnitude smaller than the one an
    unrestricted move sees.

    ``hops=1`` adds every qubit sharing a net with a boundary qubit, so the cut
    can migrate through the hubs instead of being pinned by them.  ``min_size``
    falls back to the whole side if the result is still degenerate.
    """
    full = (1 << n) - 1
    other = full ^ mask
    pa, pb = [], []
    for q in range(n):
        in_a = (mask >> q) & 1
        far = other if in_a else mask
        for ei in hg.incidence[q]:
            if hg.nets[ei].pin_mask & far:
                (pa if in_a else pb).append(q)
                break

    for _ in range(hops):
        seen_a, seen_b = set(pa), set(pb)
        for q in list(seen_a | seen_b):
            for ei in hg.incidence[q]:
                for r in hg.nets[ei].pins:
                    (seen_a if (mask >> r) & 1 else seen_b).add(r)
        pa, pb = sorted(seen_a), sorted(seen_b)

    if min_size and (len(pa) < min_size or len(pb) < min_size):
        pa = [q for q in range(n) if (mask >> q) & 1]
        pb = [q for q in range(n) if not (mask >> q) & 1]
    return pa, pb


def endpoint_polish(n, hg, mask, eps, which=0, hops=1, min_size=8, cap=24,
                    max_steps=64):
    """Fiduccia--Mattheyses descent on ONE currency, for one endpoint.

    Why an endpoint needs its own descent
    -------------------------------------
    Both searches minimise ``F = S_w``, the mode-aware scalarisation
    ``sum_i min(a e_i, b k_i)``.  That is the right cost for a partition
    executed with each crossing in its cheaper mode --- but it is not the cost
    at either *endpoint* of the front.  A net whose ``b k_i`` beats its
    ``a e_i`` contributes only ``k_i`` to ``S`` and drops out of the gradient
    entirely, so ``S`` can sit at a local minimum while ``E = sum_i e_i`` is far
    from one.  Measured on n064_k2_s0: the ensemble's champion had ``E = 56``;
    descending on ``E`` reaches ``E = 19``, the value KaHyPar and the
    evolutionary methods both report.  The search was not short of budget or of
    neighbours --- it was descending on a different function from the one the
    front is read at.

    A front whose endpoint is not locally optimal in the quantity that endpoint
    reports is dominated at its own corner, which is a defect of the front and
    not a property of the instance.

    Why FM and not steepest descent
    -------------------------------
    Steepest descent stops at the first local optimum, and that optimum is
    often not the good one: on n064_k1_s1 it halts at ``E = 17`` while the field
    reaches 13.  A pass here repeatedly takes the BEST available move ---
    *including an uphill one* --- locks the qubit it moved so it cannot come
    straight back, and records the running value; at the end it rewinds to the
    cheapest point it passed through.  Uphill moves are therefore free: they
    are kept only if something downhill of them was reached later.  A pass that
    finds no improvement is discarded whole and the polish stops.

    The lock is what makes an uphill move productive rather than a random walk.
    Without it the next step would simply undo the previous one and the pass
    would oscillate on the same pair forever.

    ``which`` is 0 for ``E`` and 1 for ``K``.  Moves are single flips over the
    dilated boundary; a swap is two of them, and with ``eps`` admitting a
    block-size change of at least one in each direction the intermediate state
    is feasible, so pairs compose out of singles without being enumerated.
    """
    st = CostState(hg, mask)
    best_mask = mask
    best_val = (st.E, st.K)[which]
    steps = 0

    while steps < max_steps:
        steps += 1
        pa, pb = boundary(n, hg, mask, hops, min_size)
        pool = sorted(set(pa) | set(pb))
        if not pool:
            break
        locked = set()
        cur = (st.E, st.K)[which]
        seq, run_best, run_best_k = [], cur, 0
        for _ in range(min(len(pool), max(4, cap))):
            pick, pick_val = None, None
            for v in pool:
                if v in locked or not feasible(mask ^ (1 << v), n, eps):
                    continue
                val = st.peek(v)[which]
                if pick_val is None or val < pick_val:
                    pick, pick_val = v, val
            if pick is None:
                break
            st.flip(pick)
            mask ^= 1 << pick
            locked.add(pick)
            seq.append(pick)
            if pick_val < run_best - 1e-12:
                run_best, run_best_k = pick_val, len(seq)

        for v in reversed(seq[run_best_k:]):        # rewind to the best prefix
            st.flip(v)
            mask ^= 1 << v
        if run_best < best_val - 1e-12 and feasible(mask, n, eps):
            best_mask, best_val = mask, run_best
        else:
            mask = best_mask                        # pass found nothing
            st.reset(mask)
            break
    return best_mask, steps
