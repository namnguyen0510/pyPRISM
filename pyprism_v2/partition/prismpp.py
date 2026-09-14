"""
pyprism_v2.partition.prismpp
============================
PRISM++ : the full seven-stage optimiser of the manuscript.

``part_prism`` in :mod:`~pyprism_v2.partition.prism` implements a
two-dimensional ``(w, T)`` replica ensemble with two move types and a
single-vertex polish.  That is a subset of the specified method.  This module
implements the whole of it:

===== ============================================ =======================
stage description                                   section
===== ============================================ =======================
  3   parallel-tempered ensemble, cooled            3.2.3  (`_ensemble`)
  4   four move operators with Tabu memory          3.2.4  (`_propose`)
  5   adaptive operator-weight rebalancing          3.2.5  (`_rebalance`)
  6   inverse-score consensus aggregation           3.2.6  (`consensus`)
  7   greedy boundary polish                        3.2.7  (`boundary_polish`)
===== ============================================ =======================

**Scope: stages 3-7 only.**  The substrate (3.2.1) and the seeding (3.2.2) are
deliberately *not* re-implemented here: PRISM++ consumes the same hypergraph
and the same seed partitions as PRISM+, so a three-way comparison isolates the
search itself.  Anything PRISM++ wins or loses is attributable to the move
operators, the cooled ladder, the rebalancing, the consensus step or the
boundary polish --- not to a different starting point.  ``augmented_graph``
and ``seed_set`` remain available for anyone who wants the full pipeline, but
the driver does not call them by default.

Relation to ``part_prism``
--------------------------
The manuscript's cost ``F`` is a single scalar.  PRISM carries two currencies
and a preference vector, so PRISM++ runs the specified ensemble **once per
preference rung** with ``F = S_w``, the mode-aware scalarisation of
:meth:`~pyprism_v2.objectives.CostState.set_pref`, and unions the resulting
fronts.  With ``n_pref = 1`` this reduces exactly to the manuscript's
algorithm.

A note on ``T_max``
-------------------
The manuscript sets ``T_max = max(2, |F_0| / 2)``.  That floor of 2 presumes
``F`` is an unnormalised cut count, of order tens to hundreds.  PRISM's ``S``
is normalised by the trivial split and lives on ``[0, 0.5]``, so
``max(2, |F_0|/2)`` would put every replica at a temperature four times the
entire range of the objective --- a pure random walk, which is precisely the
defect that the calibrated ladder in ``prism.py`` was introduced to fix.
``t_max_rule='paper'`` reproduces the literal formula for comparison;
``'scaled'`` (the default) drops the floor and uses ``|F_0| / 2``, which is
what the formula means when ``F`` is normalised.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..objectives import (CostState, balance, evaluate, feasible, knee,
                          mode_front, pareto)
# Shared with `prism.py`, deliberately.  The endpoint polish acts on the
# returned front, not inside either search; giving it to one optimiser and not
# the other would make the PRISM / PRISM+ comparison a comparison of
# post-processing.  See `refine.py`.
from .refine import bal_ok as _bal_ok, boundary as _boundary, endpoint_polish

__all__ = ['PrismPPConfig', 'part_prismpp', 'augmented_graph', 'seed_set',
           'consensus', 'boundary_polish', 'endpoint_polish',
           'cone_bfs_partition', 'MOVE_NAMES']

MOVE_NAMES = ('boundary_swap', 'single_move', 'double_swap', 'cluster_move')


def _lc_pick(cands, mask, lcw, rng):
    """Choose from ``cands`` in proportion to causal coupling across the cut.

    The specified operators sample uniformly.  PRISM instead draws the qubit
    whose light-cone coupling points mostly *across* the current boundary,
    which is the one worth moving; that bias is most of what separates the two
    searches.  Falls back to uniform when no light cone is supplied.
    """
    if not cands:
        return None
    if lcw is None:
        return int(rng.choice(cands))
    w = lcw[np.asarray(cands, dtype=int)]
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 0:
        return int(rng.choice(cands))
    return int(rng.choice(cands, p=w / tot))


def _lc_scores(lc, mask, n):
    """Per-qubit cross-minus-same causal coupling, shifted positive."""
    if lc is None:
        return None
    inA = np.array([(mask >> v) & 1 for v in range(n)], dtype=bool)
    cross = np.where(inA[:, None] != inA[None, :], lc, 0.0).sum(1)
    same = np.where(inA[:, None] == inA[None, :], lc, 0.0).sum(1)
    sc = cross - same
    return sc - sc.min() + 1e-3


@dataclass
class PrismPPConfig:
    # --- budget and ensemble (Table B3) ----------------------------------
    B: int = 150                       # outer iterations per replica at n=64
    # 0 = flat B.  Off by default, and deliberately: the ensemble budget looked
    # like the scaling defect and was not.  What PRISM was short of at large n
    # was a descent on the currency the front is reported in (`endpoint_polish`
    # below), not proposals -- at n=128 the budget makes no measurable
    # difference to E in either direction, and it is not free.  The mechanism
    # is kept because it is the right knob if the seeds are ever diverse enough
    # for the ensemble to have somewhere to go.
    b_scale: float = 0.0               # see `budget`
    n_replicas: int = 4                # K = min(4, max(2, |S|))
    alpha_cool: float = 0.92
    # C5, the hard balance constraint ||A|-|B|| <= beta.  The specification
    # fixes beta = 2 as an *absolute qubit count*, which does not scale: it is
    # 12.5% imbalance at n=16 and 0.8% at n=256.  Against the eps=0.10 that
    # every other method here uses it is four times tighter at n=100, and
    # measurably so -- on n100_k2_s0 the best partition found by PRISM+ sits
    # at |A|-|B| = 8, which beta=2 forbids outright.  `beta=None` derives it
    # from eps so PRISM++ searches the same feasible set as its comparators;
    # set an int to recover the literal constraint.
    #
    # The derivation must be *exactly* eps, not approximately.  An earlier
    # `max(2, int(eps*n))` put a floor of 2 on it to stop beta collapsing at
    # small n -- but at n=16, int(0.10*16) = 1, so the floor took over and C5
    # admitted |A|-|B| = 2, an imbalance of 0.125 against eps = 0.10.  PRISM
    # then returned a 9/7 split on 221 of the 300 n=16 instances while every
    # comparator was held to 8/8, and won that column on a partition the
    # constraint forbids.  `beta_of` now returns floor(eps*n) and the ensemble
    # tests `feasible()` directly, so the two cannot drift apart again.
    beta: int | None = None
    # --- move operators ---------------------------------------------------
    pi0: tuple = (0.45, 0.30, 0.15, 0.10)
    eta_ema: float = 0.7
    cluster_max: int = 3
    # --- polish -----------------------------------------------------------
    max_passes: int = 25               # P, stage 7
    # --- graph ------------------------------------------------------------
    w_floor: float = 0.1
    lc_half: float = 0.5               # the 1/2 on the light-cone term, eq. (6)
    oe_w: tuple = (0.05, 0.2)          # (w1, w2), eq. (5)
    # --- PRISM coupling ---------------------------------------------------
    n_pref: int = 5                    # preference rungs; 1 == the manuscript
    eps: float = 0.10                  # balance tolerance for feasibility
    seed: int = 0
    # 'calibrated' (default) | 'paper' (max(2,|F0|/2)) | 'scaled' (|F0|/2)
    t_max_rule: str = 'calibrated'
    t_accept_hot: float = 0.50         # target acceptance at the hottest rung
    max_expand: int = 48
    own_seeding: bool = False          # True re-enables stage 3.2.2
    # --- improvements over the literal specification ----------------------
    # Both default ON; set to 0 / False to recover the specified behaviour.
    boundary_hops: int = 1             # dilate P(A), P(B) by this many hops
    boundary_min: int = 8              # below this, fall back to the whole side
    lc_bias: bool = True               # light-cone-weighted operand choice
    use_consensus: bool = True         # stage 6; False for the C ablation
    endpoint_polish: bool = True       # stage 7b, see `endpoint_polish`
    ep_starts: int = 4                 # distinct basins to descend from
    ep_cap: int = 24                   # swap candidates per side, by gain

    # The three strides are fractions of the budget, so they must be taken
    # against the *scaled* budget: keyed off the flat `B` they would fire the
    # same number of times at n=256 as at n=16 while the run is four times
    # longer, which cools the ladder to nothing a quarter of the way in.
    def tau_cool(self, n):
        return max(15, self.budget(n) // 15)

    def tau_ex(self, n):
        return max(8, self.budget(n) // 25)

    def tau_rb(self, n):
        return max(20, self.budget(n) // 12)

    def tabu_cap(self, n):
        return max(6, n // 3)

    def passes_of(self, n):
        """Polish passes.  See :func:`boundary_polish` for why this is now a
        sweep count rather than a swap count."""
        return self.max_passes

    def beta_of(self, n):
        """C5 in qubits.  Derived from ``eps`` unless pinned to an int.

        ``floor(eps*n)`` is precisely the largest ``|A|-|B|`` that
        :func:`~pyprism_v2.objectives.feasible` admits, so with ``beta=None``
        C5 and the benchmark's feasibility test are the same constraint.  No
        floor: a floor is how the two came apart at n=16.
        """
        return int(self.beta) if self.beta is not None else int(self.eps * n)

    def budget(self, n):
        """Outer iterations per replica at register size ``n``.

        ``B`` alone is flat in ``n``, which is a budget that shrinks in real
        terms: the number of qubits the search must place grows, the number of
        proposals it gets does not.  Measured over 1500 instances, the PRISM
        search improved on its own seeds on 34/40 instances at n=16 but only
        2/40 at n=128 -- at that size the ensemble was a perturbation on the
        seed, not a search.  The budget therefore grows linearly in the number
        of qubits it has to move, normalised so ``budget(64) == B`` and the
        historical setting is unchanged at the size it was tuned on.

        ``b_scale = 0`` recovers the flat ``B``.  The floor at ``B`` keeps the
        small end unchanged: n=16 and n=32 already reach the same optimum as
        every other method, so there is nothing there to buy.

        The growth is ``sqrt(n/64)``, not linear.  A proposal costs ``O(deg)``
        and the mean net degree itself grows with ``n``, so a budget linear in
        ``n`` makes the total work quadratic --- at n=256 that is a four-fold
        budget bought with a sixteen-fold runtime, and the largest cell would
        dominate a sweep on its own.
        """
        if self.b_scale <= 0:
            return self.B
        return int(self.B * max(1.0, self.b_scale * math.sqrt(n / 64.0)))


# ===========================================================================
#  stage 1 -- the augmented graph
# ===========================================================================
def augmented_graph(n, hg, lc=None, cfg: PrismPPConfig | None = None):
    """``G_aug``: interaction weights, light-cone increment, operator entanglement.

    Section 3.2.1, eq. (6):

        w_aug(u,v) = max( w_int(u,v) + (1/2) dw_LC(u,v) + w_OE(u,v), w_floor )

    ``w_int`` counts co-incident nets, ``dw_LC`` is the causal coupling from
    :func:`~pyprism_v2.lightcone.light_cone_graph`, and ``w_OE`` grades a pair
    by the operator Schmidt rank of the nets joining them --- a SWAP and a CNOT
    act on two qubits each but are not equally hard to cut.  Only the seeders
    of stage 2 consume this; the objective is untouched by it.
    """
    cfg = cfg or PrismPPConfig()
    w_int = np.zeros((n, n))
    w_oe = np.zeros((n, n))
    w1, w2 = cfg.oe_w
    for e in hg.nets:
        pins = e.pins
        if len(pins) < 2:
            continue
        # operator-entanglement grade of this net, eq. (5)
        chi = max(2.0, math.exp(e.w_logk))
        g = w1 * e.w_ebit + w2 * math.log2(chi)
        for i, u in enumerate(pins):
            for v in pins[i + 1:]:
                w_int[u, v] += 1.0
                w_int[v, u] += 1.0
                w_oe[u, v] += g
                w_oe[v, u] += g
    w = w_int + w_oe
    if lc is not None:
        w = w + cfg.lc_half * np.asarray(lc, dtype=float)
    w = np.maximum(w, cfg.w_floor)
    np.fill_diagonal(w, 0.0)
    return w


# ===========================================================================
#  stage 2 -- multi-start seeding
# ===========================================================================
def cone_bfs_partition(n, lc, first=None):
    """Cone-guided BFS seed, section 3.2.2 strategy (5).

    Start from the qubit with the *smallest* final causal cone --- the one
    least entangled with the rest --- and grow along its cone until half the
    register is claimed.  This is the only seeder that uses the circuit's
    causal structure rather than its connectivity.
    """
    W = np.asarray(lc, dtype=float)
    cone = W.sum(1)
    q0 = int(np.argmin(cone))
    order = np.argsort(-W[q0])              # nearest in cone first
    A = [q0] + [int(q) for q in order if int(q) != q0]
    return sum(1 << q for q in A[:max(1, (n + 1) // 2)])


def seed_set(n, hg, lc=None, cfg: PrismPPConfig | None = None, rng=None):
    """The six seeding strategies of section 3.2.2.

    Returns ``[(name, mask), ...]``, silently dropping strategies whose
    optional dependency is absent.  Diversity is the point: deterministic
    seeders concentrate in one basin, so two random balanced cuts are always
    included.
    """
    cfg = cfg or PrismPPConfig()
    rng = rng or np.random.default_rng(cfg.seed)
    from .baselines import (part_kernighan_lin, part_spectral, part_louvain,
                            part_metis, HAVE_METIS)
    out = []
    kw = dict(seed=cfg.seed, eps=cfg.eps, lc=lc)
    for name, fn in (('KL', part_kernighan_lin),
                     ('spectral', part_spectral),
                     ('louvain', part_louvain)):
        try:
            out.append((name, fn(n, hg, **kw)))
        except Exception:
            pass
    if HAVE_METIS:
        try:
            out.append(('metis', part_metis(n, hg, **kw)))
        except Exception:
            pass
    if lc is not None:
        try:
            out.append(('cone-bfs', cone_bfs_partition(n, lc)))
        except Exception:
            pass
    for i in range(2):
        order = rng.permutation(n)
        out.append((f'random{i}', sum(1 << int(q) for q in order[:n // 2])))
    return out


# ===========================================================================
#  boundary sets, shared by the move operators and the polish
# ===========================================================================
# ===========================================================================
#  stage 4 -- the four move operators
# ===========================================================================
def _propose(n, hg, mask, pi, tabu, rng, cfg, adj=None, bcache=None,
             lcw=None):
    """Sample a move type from ``pi`` and realise it (section 3.2.4).

    Returns ``(candidate_mask, touched_qubits, move_index)`` or ``None`` if the
    move aborts.  The Tabu queue is consulted here but only *appended to* by
    the caller, and only on acceptance --- section 3.2.4 is explicit that a
    rejected move leaves ``T`` unchanged.
    """
    o = int(rng.choice(4, p=pi))
    inA = [q for q in range(n) if (mask >> q) & 1]
    inB = [q for q in range(n) if not (mask >> q) & 1]

    if o == 0:                                    # M1 boundary swap
        pa, pb = (bcache if bcache is not None else
                  _boundary(n, hg, mask, cfg.boundary_hops,
                            cfg.boundary_min))
        if not pa or not pb:
            return None
        for _ in range(8):                        # "retry until" one is free
            qa = _lc_pick(pa, mask, lcw, rng)
            qb = _lc_pick(pb, mask, lcw, rng)
            if qa is None or qb is None:
                return None
            if qa not in tabu or qb not in tabu:
                return mask ^ (1 << qa) ^ (1 << qb), (qa, qb), o
        return None

    if o == 1:                                    # M2 single move
        # The specified rule moves a qubit from the LARGER side to the smaller,
        # which makes M2 a restoring force towards |A| = n/2 rather than a
        # move.  M1 and M3 are balance-preserving by construction, so with M2
        # pulling inwards the only operator that can leave the balance point is
        # M4, at a 10% prior --- and the seeds and the consensus partition are
        # all exactly balanced.  The ensemble therefore sat at |A| = n/2 while
        # eps allowed |A| in [n/2 - eps*n/2, n/2 + eps*n/2], and that band is
        # where the good partitions live: on n064_k2_s0 the optimum is
        # |A| = 30 with E = 19, and every method that reached it used the
        # allowance, while PRISM held 32 and paid E = 56.
        #
        # Here the direction is chosen uniformly among those that stay
        # feasible, so |A| diffuses over the whole feasible band instead of
        # being pinned to its centre.
        if not inA or not inB:
            return None
        p = len(inA)
        ok = [S for S, p2 in ((inA, p - 1), (inB, p + 1))
              if _bal_ok(p2, n, cfg.eps)]
        if not ok:
            return None
        S = ok[0] if len(ok) == 1 else (inA if rng.random() < 0.5 else inB)
        for _ in range(8):
            q = _lc_pick(S, mask, lcw, rng)
            if q is None:
                return None
            if q not in tabu:
                return mask ^ (1 << q), (q,), o
        return None

    if o == 2:                                    # M3 double swap
        if min(len(inA), len(inB)) < 2:
            return None
        a2 = rng.choice(inA, size=2, replace=False)
        b2 = rng.choice(inB, size=2, replace=False)
        cand = mask
        for q in list(a2) + list(b2):
            cand ^= 1 << int(q)
        # M3 deliberately ignores the Tabu queue: cooperative displacements
        # are the ensemble's basin-escape mechanism and would be strangled by
        # it (section 3.2.4).
        return cand, tuple(int(q) for q in list(a2) + list(b2)), o

    # M4 cluster move
    side = inA if rng.random() < 0.5 else inB
    if len(side) <= 3:
        return None
    s = int(rng.choice(side))
    Z = {s}
    if adj is not None:
        same = set(side)
        nbr = np.argsort(-adj[s])
        for r in nbr:
            r = int(r)
            if len(Z) >= cfg.cluster_max:
                break
            if r in same and r not in Z and adj[s, r] > 0:
                Z.add(r)
    if len(Z) >= len(side) - 1:
        return None
    cand = mask
    for q in Z:
        cand ^= 1 << q
    return cand, tuple(sorted(Z)), 3


# ===========================================================================
#  stage 5 -- adaptive operator-weight rebalancing
# ===========================================================================
def _rebalance(pi, acc, att, cfg):
    """Eq. (10): Laplace-smoothed acceptance rates, EMA-blended into ``pi``.

    The Laplace ``+1`` is what stops an operator that had a bad run from being
    zeroed permanently --- with no smoothing, one unlucky rebalance window
    removes a move type for the rest of the search.
    """
    r = np.array([(acc[o] + 1.0) / (att[o] + 1.0) for o in range(4)])
    r = r / r.sum()
    pi = cfg.eta_ema * np.asarray(pi) + (1.0 - cfg.eta_ema) * r
    return pi / pi.sum()


# ===========================================================================
#  stage 6 -- inverse-score consensus aggregation
# ===========================================================================
def consensus(n, states, eps=1e-9, score=None, bal_eps=None, hg=None,
              pref=None):
    """Section 3.2.6.  ``states`` is ``[(mask, F), ...]``.

    Weight ``w_k = 1 / ((F_k - F_min) + eps)`` concentrates voting mass on the
    cheapest replicas while staying bounded when several tie.  Qubits are
    ranked by their weighted side-A vote and a prefix of that ranking forms
    ``A``.

    **Where the prefix is cut.**  The specification takes the top ``ceil(n/2)``,
    which makes the consensus partition an exact half every time.  That is one
    point of a whole feasible interval: with tolerance ``bal_eps`` any prefix
    length in ``[n/2 - bal_eps*n/2, n/2 + bal_eps*n/2]`` is admissible, and on
    these instances the good partitions are not at the centre of it.  Given a
    way to score, every feasible prefix is evaluated and the cheapest is
    returned; the ranking is unchanged, only the cut point is chosen rather
    than assumed.  Without one the exact half is kept.

    The prefixes are nested --- prefix ``p`` is prefix ``p-1`` plus one qubit
    --- so with ``hg`` and ``pref`` the whole scan is one ``CostState`` walked
    forward by single flips, ``O(n deg)`` in total.  Scoring each prefix from
    scratch is ``O(n |E|)``, which is tolerable while ``bal_eps`` restricts the
    scan to a narrow band around ``n/2`` and is not once the constraint is
    lifted and every ``p`` in ``(0, n)`` is admissible.
    """
    if not states:
        return None
    fmin = min(f for _m, f in states)
    sigma = np.zeros(n)
    for m, f in states:
        wk = 1.0 / ((f - fmin) + eps)
        for q in range(n):
            if (m >> q) & 1:
                sigma[q] += wk
    order = [int(q) for q in np.argsort(-sigma)]
    half = (n + 1) // 2
    if bal_eps is None or (score is None and hg is None):
        return sum(1 << q for q in order[:half])

    best, best_p = None, half
    if hg is not None and pref is not None:
        st = CostState(hg, 0).set_pref(pref[0], pref[1])
        m = 0
        for p in range(1, n):
            st.flip(order[p - 1])
            m |= 1 << order[p - 1]
            if not _bal_ok(p, n, bal_eps):
                continue
            s = st.S + 1e-4 * balance(m, n)
            if best is None or s < best:
                best, best_p = s, p
    else:
        for p in range(1, n):
            if not _bal_ok(p, n, bal_eps):
                continue
            s = score(sum(1 << q for q in order[:p]))
            if best is None or s < best:
                best, best_p = s, p
    return sum(1 << q for q in order[:best_p])


# ===========================================================================
#  stage 7 -- greedy boundary polish
# ===========================================================================
def boundary_polish(n, hg, mask, score, cfg, max_passes=None, pref=None):
    """Section 3.2.7: exhaustive improving boundary swaps.

    A pass sweeps ``P(A) x P(B)`` and applies **every** strictly improving
    swap it finds, locking each qubit it moves for the rest of that pass; the
    boundary is then rebuilt and the next pass begins.  It stops on a pass that
    finds nothing or after ``P`` passes.

    Why not first-improvement
    -------------------------
    The literal reading of 3.2.7 -- take the first improving swap and restart
    -- makes a pass worth exactly one swap, so ``P`` passes buy ``P`` swaps
    however large the instance is.  Measured over 1500 instances, the pass
    counter hit its cap on *every* instance at n >= 128, meaning the polish was
    still finding improvements when it was cut off and had made at most eight
    swaps on a 256-qubit partition.  Sweeping the whole product before
    rebuilding the boundary costs the same scan and buys up to
    ``min(|P(A)|, |P(B)|)`` swaps, which is what makes the polish scale.

    The lock is what makes that safe.  Two improving swaps found in the same
    scan are only independently improving if they are disjoint: once ``qa``
    has crossed to B it is no longer a member of ``P(A)``, and pairing it again
    would be scoring a move from a partition that no longer exists.  Locking
    both endpoints on use restricts a pass to a set of disjoint swaps, each
    evaluated against the mask in force when it was applied.

    This differs from ``prism._polish`` in kind, not degree.  That one moves a
    single vertex, which changes the balance and so is repeatedly blocked by
    the feasibility test; this one exchanges a pair and is balance-preserving
    by construction, so it can always act.  It is also restricted to the
    boundary, where the cost actually lives.
    """
    passes, swaps, moves = 0, 0, 0
    best = score(mask)
    limit = cfg.max_passes if max_passes is None else max_passes
    # A pass enumerates |P(A)| x |P(B)| candidates, which is hundreds on a
    # 64-qubit instance.  Rebuilding a CostState for each would make the
    # polish cost more than the whole ensemble, so when the preference is
    # supplied the scan flips in place and flips back -- O(deg) either way
    # against O(|E|) for a rebuild.
    st = None
    if pref is not None:
        st = CostState(hg, mask).set_pref(pref[0], pref[1])

    def _try(cand, touched):
        """Score ``cand``, commit if it improves, restore if not."""
        nonlocal mask, best
        if st is None:
            s = score(cand)
            if s < best - 1e-12:
                mask, best = cand, s
                return True
            return False
        for q in touched:
            st.flip(q)
        s = st.S + 1e-4 * balance(cand, n)
        if s < best - 1e-12:
            mask, best = cand, s
            return True
        for q in reversed(touched):            # reject: restore
            st.flip(q)
        return False

    while passes < limit:
        passes += 1
        pa, pb = _boundary(n, hg, mask, cfg.boundary_hops, cfg.boundary_min)
        locked = set()
        improved = False

        # --- balance-preserving swaps -------------------------------------
        for qa in sorted(pa):
            if qa in locked:
                continue
            for qb in sorted(pb):
                if qb in locked:
                    continue
                cand = mask ^ (1 << qa) ^ (1 << qb)
                if not feasible(cand, n, cfg.eps):
                    continue
                if _try(cand, (qa, qb)):
                    improved = True
                    swaps += 1
                    locked.update((qa, qb))
                    break              # qa has moved; on to the next qa

        # --- single moves, which are what reach the rest of the band -------
        # A swap holds |A| fixed, so a polish made only of swaps can never
        # leave the block size it started at.  Every seed here is an exact
        # half and eps admits |A| anywhere in [n/2 - eps*n/2, n/2 + eps*n/2],
        # so without this sweep the polish explores one slice of the feasible
        # set.  On n064_k2_s0 the optimum sits at |A| = 30 against a seed at
        # 32, and it is unreachable by any number of swaps.
        for q in sorted(set(pa) | set(pb)):
            if q in locked:
                continue
            cand = mask ^ (1 << q)
            if not feasible(cand, n, cfg.eps):
                continue
            if _try(cand, (q,)):
                improved = True
                moves += 1
                locked.add(q)

        if not improved:
            break
    return mask, passes, swaps, moves


# ===========================================================================
#  stage 3 -- the parallel-tempered ensemble
# ===========================================================================
def _calibrate_tmax(n, hg, mask, pref, cfg, rng, adj, samples=192):
    """``T_max`` from the median ``|dF|`` of the ensemble's own move mix.

    Sampled through :func:`_propose` itself, so all four operators contribute
    in their prior proportions and the estimate reflects the moves the search
    will actually make --- a double swap displaces four qubits and has a very
    different ``|dF|`` from a single move.

    ``T_max`` is set so the hottest replica accepts a typical worsening move
    with probability ``t_accept_hot``; the ladder ``T_k = T_max 2^{-k}`` then
    cools from there, and the coldest of ``K = 4`` rungs sits at ``T_max / 8``.

    Evaluated **incrementally**: flip the touched qubits, read ``S``, flip them
    back.  Scoring each sample by constructing a fresh ``CostState`` costs
    ``O(|E|)``, which at n=256 is tens of thousands of net evaluations per
    sample and made calibration alone a large fraction of the run -- budget
    spent measuring the objective rather than minimising it.  Flip and restore
    is ``O(deg)`` and gives the identical number.
    """
    a, b = pref
    st = CostState(hg, mask).set_pref(a, b)
    base = st.S + 1e-4 * balance(mask, n)
    pi = np.array(cfg.pi0, dtype=float)
    deltas = []
    for _ in range(samples):
        prop = _propose(n, hg, mask, pi, set(), rng, cfg, adj)
        if prop is None:
            continue
        cand, touched, _o = prop
        if not feasible(cand, n, cfg.eps):
            continue
        for q in touched:
            st.flip(q)
        d = abs(st.S + 1e-4 * balance(cand, n) - base)
        for q in reversed(touched):        # restore, O(deg) not O(|E|)
            st.flip(q)
        if d > 0.0:
            deltas.append(d)
    if not deltas:
        return None
    return float(np.median(deltas)) / math.log(1.0 / cfg.t_accept_hot)


def _ensemble(n, hg, w, norm, seeds, lc, cfg, rng, adj, trace=None):
    """One preference rung: K cooled replicas with exchange, stages 3-6."""
    a, b = w[0] / norm[0], w[1] / norm[1]

    def score_of(mask):
        return (CostState(hg, mask).set_pref(a, b).S
                + 1e-4 * balance(mask, n))

    beta_n = cfg.beta_of(n)
    B = cfg.budget(n)
    tau_cool, tau_ex, tau_rb = cfg.tau_cool(n), cfg.tau_ex(n), cfg.tau_rb(n)
    K = min(cfg.n_replicas, max(2, len(seeds)))
    F0 = abs(score_of(seeds[0][1]))
    if cfg.t_max_rule == 'paper':
        t_max = max(2.0, F0 / 2.0)
    elif cfg.t_max_rule == 'scaled':
        t_max = max(F0 / 2.0, 1e-12)
    else:                                       # 'calibrated', the default
        t_max = _calibrate_tmax(n, hg, seeds[0][1], (a, b), cfg, rng, adj)
        if t_max is None:
            t_max = max(F0 / 2.0, 1e-12)
    temps = [t_max * (2.0 ** -k) for k in range(K)]

    # Seed-to-rung assignment.  temps[k] = T_max 2^-k descends, so rung 0 is
    # the HOTTEST.  Assigning seeds in their given order puts the best seed on
    # the hottest rung, where it is immediately random-walked away, and leaves
    # the coldest rung frozen on the worst seed -- the ensemble then cannot
    # improve on its own input.
    #
    # Take the K CHEAPEST seeds and reverse them, so the best is paired with
    # the coldest rung and the K-th best with the hottest.  Selecting the
    # cheapest K matters as soon as more than K seeds exist: sorting all of
    # them worst-first and taking the head hands every replica to the *worst*
    # K and discards the best ones outright, which is what happened the moment
    # the seed set was topped up for diversity.
    ranked = sorted(seeds, key=lambda nm_m: score_of(nm_m[1]))[:K][::-1]

    reps = []
    for k in range(K):
        m = ranked[k % len(ranked)][1]
        st = CostState(hg, m).set_pref(a, b)
        reps.append({'st': st, 'F': st.S + 1e-4 * balance(m, n),
                     'T': temps[k], 'pi': np.array(cfg.pi0, dtype=float),
                     'tabu': deque(maxlen=cfg.tabu_cap(n)),
                     'acc': np.zeros(4), 'att': np.zeros(4), 'bc': None,
                     'lcw': None})

    visited = [(r['st'].E, r['st'].K, r['st'].mask) for r in reps]
    ex_acc = ex_att = 0
    move_acc = np.zeros(4)
    move_att = np.zeros(4)

    for it in range(B):
        for rep in reps:
            st = rep['st']
            # the boundary set is O(n * deg) to build and changes by at most a
            # couple of qubits per accepted move, so it is refreshed on a
            # stride rather than rebuilt for every proposal
            if rep['bc'] is None or it % 8 == 0:
                rep['bc'] = _boundary(n, hg, st.mask, cfg.boundary_hops,
                                      cfg.boundary_min)
            if rep['lcw'] is None or it % 8 == 0:
                rep['lcw'] = (_lc_scores(lc, st.mask, n)
                              if cfg.lc_bias else None)
            prop = _propose(n, hg, st.mask, rep['pi'], set(rep['tabu']),
                            rng, cfg, adj, bcache=rep['bc'], lcw=rep['lcw'])
            if prop is None:
                continue
            cand, touched, o = prop
            rep['att'][o] += 1
            move_att[o] += 1
            # C5, the hard balance constraint, checked before any evaluation.
            # Tested through `feasible` rather than against `beta_n` directly:
            # the two are the same constraint by construction of `beta_of`, and
            # writing it once means a candidate the benchmark would reject can
            # never enter the ensemble.  It used to be spelled out here against
            # a beta with a floor of 2, and at n=16 that admitted a 9/7 split
            # the benchmark calls infeasible.
            if not feasible(cand, n, cfg.eps):
                continue
            saved = st.mask
            for q in touched:
                E2, K2 = st.flip(q)
            F2 = st.S + 1e-4 * balance(cand, n)
            d = F2 - rep['F']
            if d <= 0 or rng.random() < math.exp(-d / max(rep['T'], 1e-12)):
                rep['F'] = F2
                rep['acc'][o] += 1
                move_acc[o] += 1
                rep['tabu'].extend(touched)
                rep['bc'] = None            # boundary is stale after a move
                rep['lcw'] = None
                visited.append((E2, K2, cand))
            else:
                st.reset(saved)

        if it and it % tau_ex == 0:                     # exchange, eq. (8)
            for k in range(len(reps) - 1):
                hot, cold = reps[k], reps[k + 1]        # temps descend with k
                M = -((hot['F'] - cold['F'])
                      * (1.0 / hot['T'] - 1.0 / cold['T']))
                ex_att += 1
                if M >= 0 or rng.random() < math.exp(M):
                    reps[k], reps[k + 1] = cold, hot
                    reps[k]['T'], reps[k + 1]['T'] = hot['T'], cold['T']
                    ex_acc += 1

        if it and it % tau_cool == 0:                   # cooling
            for rep in reps:
                rep['T'] *= cfg.alpha_cool

        if it and it % tau_rb == 0:                     # stage 5
            for rep in reps:
                rep['pi'] = _rebalance(rep['pi'], rep['acc'], rep['att'], cfg)
                rep['acc'][:] = 0
                rep['att'][:] = 0

    # --- stage 6: consensus ------------------------------------------------
    # `use_consensus=False` is the stage-6 ablation: the ensemble's champion is
    # then simply its cheapest replica, with no aggregation across them.
    cs = (consensus(n, [(r['st'].mask, r['F']) for r in reps],
                    bal_eps=cfg.eps, hg=hg, pref=(a, b))
          if cfg.use_consensus else None)
    best_F = min(r['F'] for r in reps)
    if cs is not None and feasible(cs, n, cfg.eps):
        f_cs = score_of(cs)
        if f_cs < best_F:
            visited.append((*evaluate(hg, cs), cs))
            best_F = f_cs

    # --- stage 7: greedy boundary polish ----------------------------------
    # The champion is taken over FEASIBLE replicas.  Every replica is feasible
    # by the C5 test above, but seeds are supplied from outside and a seed that
    # violates eps would otherwise be handed straight to the polish -- which
    # only tests its candidates, never its starting point -- and could be
    # returned unchanged.
    ok = [r for r in reps if feasible(r['st'].mask, n, cfg.eps)]
    champion = min(ok or reps, key=lambda r: r['F'])['st'].mask
    if cs is not None and feasible(cs, n, cfg.eps) and score_of(cs) < \
            score_of(champion):
        champion = cs
    polished, passes, swaps, moves = boundary_polish(n, hg, champion, score_of,
                                                     cfg, pref=(a, b))
    visited.append((*evaluate(hg, polished), polished))

    # One filter, at the exit: nothing infeasible reaches the front.  The
    # ensemble cannot produce an infeasible mask any more, but the seeds are
    # someone else's and this is the last place to catch one.
    visited = [v for v in visited if feasible(v[2], n, cfg.eps)]

    diag = {'K': K, 't_max': t_max, 'temps': temps, 'B': B,
            'exchange_accept': ex_acc / max(ex_att, 1),
            'move_attempts': move_att.tolist(),
            'move_accepts': move_acc.tolist(),
            'move_rate': (move_acc / np.maximum(move_att, 1)).tolist(),
            'pi_final': [r['pi'].tolist() for r in reps],
            'polish_passes': passes, 'polish_swaps': swaps,
            'polish_moves': moves, 'beta': beta_n,
            'nA': bin(polished).count('1'),
            'consensus_used': bool(cs is not None and cs == champion)}
    return visited, diag


# ===========================================================================
#  driver
# ===========================================================================
def part_prismpp(n, hg, cfg: PrismPPConfig | None = None, seeds=None, lc=None,
                 label='PRISM++', trace=None):
    """PRISM++: all seven stages.  Returns ``(front, diagnostics)``.

    ``seeds`` overrides stage 2 if given; otherwise the six strategies of
    section 3.2.2 are run.  With ``cfg.n_pref = 1`` this is exactly the
    manuscript's algorithm; with more, the ensemble is run once per preference
    rung and the fronts are unioned.
    """
    from .baselines import part_naive
    cfg = cfg or PrismPPConfig()
    rng = np.random.default_rng(cfg.seed)
    ref = evaluate(hg, part_naive(n))
    norm = (max(ref[0], 1.0), max(ref[1], 1e-9))

    if seeds is not None:
        S = [(f'seed{i}', m) for i, m in enumerate(seeds)]
        # Top up on collapse.  The supplied seeds are KL/H2, Greedy/H2,
        # Spectral/H2 and KL/H1, four different algorithms -- but on the large
        # instances they agree: at n=128 all four return the same partition,
        # E = 39 for every one.  A parallel-tempered ensemble seeded with four
        # copies of one point is four copies of one search, and the exchange
        # step has nothing to exchange.  Cone-BFS and random balanced cuts are
        # added until there are `n_replicas` distinct starts, so the ladder
        # always has real diversity to work with.  Nothing is removed, so a
        # non-degenerate seed set is untouched.
        have = {m for _nm, m in S}
        if len(have) < cfg.n_replicas:
            extra = []
            if lc is not None:
                try:
                    extra.append(('cone-bfs', cone_bfs_partition(n, lc)))
                except Exception:
                    pass
            i = 0
            while len(have) + len(
                    {m for _nm, m in extra}) < cfg.n_replicas and i < 32:
                order = rng.permutation(n)
                extra.append((f'div{i}',
                              sum(1 << int(q) for q in order[:n // 2])))
                i += 1
            for nm, m in extra:
                if m not in have:
                    have.add(m)
                    S.append((nm, m))
    elif cfg.own_seeding:
        S = seed_set(n, hg, lc=lc, cfg=cfg, rng=rng)
    else:
        # stages 3-7 only: start where PRISM starts, at random balanced cuts,
        # so nothing is attributable to a better seed
        S = []
        for i in range(max(cfg.n_replicas, 2)):
            order = rng.permutation(n)
            S.append((f'random{i}', sum(1 << int(q) for q in order[:n // 2])))
    # the cluster move needs a same-side neighbour relation; the light cone is
    # already that, and using it avoids pulling in stage 3.2.1
    adj = (np.asarray(lc, dtype=float) if lc is not None
           else np.zeros((n, n)))

    ws = [(a, 1.0 - a) for a in np.linspace(0.05, 0.95, cfg.n_pref)]
    visited_all, per_rung = [], []
    for i, w in enumerate(ws):
        vis, dg = _ensemble(n, hg, w, norm, S, lc, cfg,
                            np.random.default_rng(cfg.seed + 101 * i), adj)
        visited_all.extend(vis)
        per_rung.append(dg)
        if trace is not None:
            trace(i, {'w': w, 'diag': dg, 'visited': len(visited_all)})

    # --- stage 7b: polish each endpoint in its own currency ----------------
    # The rungs above minimise S_w; the front is read at its E and K corners.
    # Descending on the reported quantity is what makes those corners locally
    # optimal -- see `endpoint_polish` for the measurement that motivated it.
    # Multi-start, because a single descent lands in a single basin.  The
    # starts are the `ep_starts` cheapest *distinct* masks in the currency
    # being polished: they cost nothing to find, they are already good, and
    # they are spread across whatever basins the rungs reached.  On n128 the
    # best-E mask is itself a local optimum of E, so with one start the polish
    # returns immediately and the second-best basin is never examined.
    ep_steps = {}
    if cfg.endpoint_polish and visited_all:
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

    init_masks = {m for _nm, m in S}
    seeds_front = pareto([(E, K, m) for E, K, m in visited_all])
    if len(seeds_front) > cfg.max_expand:
        idx = np.linspace(0, len(seeds_front) - 1,
                          cfg.max_expand).round().astype(int)
        seeds_front = [seeds_front[j] for j in sorted(set(idx.tolist()))]
    out = []
    for _E, _K, m in seeds_front:
        st = CostState(hg, m)
        for E, K in mode_front(st.costs()):
            out.append((E, K, m))
    front = pareto(out)

    # An optimiser must not return nothing.  Two ways the front could come back
    # empty: every visited mask failed the feasibility filter at the exit of
    # `_ensemble` (possible when a caller supplies seeds that violate its own
    # eps), or the mode-front expansion produced no non-dominated point.  Both
    # are pathological rather than impossible, and a caller that then does
    # `min(p for p in front if ...)` gets a bare ValueError naming nothing.
    # Fall back to a balanced cut, which is always feasible and always
    # priceable, and say so in the diagnostics so the row is not silently
    # mistaken for a search result.
    fallback = False
    if not front:
        fallback = True
        half = sum(1 << q for q in range((n + 1) // 2))
        cand = [half] + [m for _nm, m in S if feasible(m, n, cfg.eps)]
        m0 = min(cand, key=lambda m: evaluate(hg, m)[0])
        st = CostState(hg, m0)
        front = pareto([(E, K, m0) for E, K in mode_front(st.costs())])
        if not front:                       # nothing crosses: E = K = 0
            front = [(0.0, 0.0, m0)]

    e_init = min((E for E, _K, m in visited_all if m in init_masks),
                 default=float('inf'))
    e_search = min((E for E, _K, m in visited_all if m not in init_masks),
                   default=float('inf'))
    mv_att = np.sum([d['move_attempts'] for d in per_rung], axis=0)
    mv_acc = np.sum([d['move_accepts'] for d in per_rung], axis=0)
    diag = {'label': label, 'stages': 7, 'seeds': [nm for nm, _m in S],
            'n_seeds': len(S), 'rungs': cfg.n_pref,
            'visited': len(visited_all), 'front': len(front),
            'exchange_accept': float(np.mean([d['exchange_accept']
                                              for d in per_rung])),
            'move_names': MOVE_NAMES,
            'move_attempts': mv_att.tolist(),
            'move_rate': (mv_acc / np.maximum(mv_att, 1)).tolist(),
            'pi_final': per_rung[-1]['pi_final'][0],
            'polish_passes': [d['polish_passes'] for d in per_rung],
            'polish_swaps': [d['polish_swaps'] for d in per_rung],
            'polish_moves': [d['polish_moves'] for d in per_rung],
            'nA': [d['nA'] for d in per_rung],
            'beta': per_rung[0]['beta'], 'B': per_rung[0]['B'],
            'endpoint_steps': ep_steps, 'fallback_front': fallback,
            't_max': [d['t_max'] for d in per_rung],
            'best_E_initial': float(e_init),
            'best_E_search': float(e_search),
            'search_improved': bool(e_search < e_init)}
    return front, diag
