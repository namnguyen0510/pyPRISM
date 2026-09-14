"""
pyprism_v2.hypergraph
=====================
The two substrates.

**Track 1** is what a symmetry-blind partitioner must use: one net per
multi-qubit gate, weighted by the *worst case* over the bipartitions of
``q_g``, because it cannot know in advance which side the cut will fall on.

**Track 2** adds the two mechanisms of the method.

*Irreducibility* --- cut-resolved weights.  ``chi = sum_delta chi_delta`` is
evaluated at the realised local cut, so a cut separating a control from its
targets costs ``ceil(log2 2) = 1`` ebit whatever the arity, instead of the
worst case over cuts.

*Symmetry* --- distributable packets.  A maximal run of gates rooted at a qubit
``v``, all of which have ``v`` in their ``zsupp``, is executable with one
cat-entangler broadcast, hence one ebit total however many gates it contains.
The run breaks exactly at the first gate with ``v`` not in ``zsupp(g)`` --- that
is, exactly where the on-site symmetry at ``v`` breaks.

A packet net is charged by the connectivity-1 metric, which is one ebit per
remote block --- and exactly KaHyPar's ``connectivityMinusOne`` objective.
That correspondence is why nets, not edges, are the right substrate.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .gates import spec, gamma_of_gate, diag_phases

__all__ = ['Net', 'Hypergraph', 'build_track1', 'build_track2',
           'extract_packets']


@dataclass
class Net:
    """A hyperedge.

    ``kind`` is ``'gate'`` (a single gate, charged per cut) or ``'packet'``
    (a v-rooted distributable packet, charged once).
    """
    nid: int
    kind: str
    pins: tuple
    pin_mask: int
    gates: tuple = ()
    root: int | None = None
    # Track 1 constants: worst case over the bipartitions of q_g
    w_ebit: int = 0
    w_logk: float = 0.0
    # Track 2 evaluation data
    name: str = ''
    qubits: tuple = ()
    params: tuple = ()
    # packet members: ((remote_qubits, phases|None, name, params, qubits), ...)
    # the gate identity is kept so that a member whose remote operands straddle
    # the boundary can be charged its own exact cost -- see objectives.packet_cost
    members: tuple = ()


@dataclass
class Hypergraph:
    n: int
    nets: list
    track: int
    label: str = ''
    incidence: list = field(default_factory=list)   # qubit -> [net index]

    def build_incidence(self):
        inc = [[] for _ in range(self.n)]
        for i, e in enumerate(self.nets):
            for q in e.pins:
                inc[q].append(i)
        self.incidence = inc
        return inc

    @property
    def n_nets(self):
        return len(self.nets)

    @property
    def n_pins(self):
        return sum(len(e.pins) for e in self.nets)

    def stats(self):
        sizes = [len(e.pins) for e in self.nets] or [0]
        return {'nets': self.n_nets, 'pins': self.n_pins,
                'mean_net_size': float(np.mean(sizes)),
                'max_net_size': int(np.max(sizes)),
                'packets': sum(1 for e in self.nets if e.kind == 'packet')}


def _gate_net(nid, g):
    s = spec(g.name)
    gam = gamma_of_gate(g.name, g.params, None)
    return Net(nid=nid, kind='gate', pins=tuple(sorted(g.qubits)),
               pin_mask=sum(1 << q for q in g.qubits), gates=(g.gid,),
               w_ebit=s.ebits_worst, w_logk=math.log(max(gam, 1.0)),
               name=g.name, qubits=g.qubits, params=g.params)


def build_track1(n, gates):
    """H_1: one net per multi-qubit gate, worst-case weights."""
    nets = [_gate_net(i, g)
            for i, g in enumerate(g_ for g_ in gates if g_.m >= 2)]
    hg = Hypergraph(n=n, nets=nets, track=1, label='Track 1 (symmetry-blind)')
    hg.build_incidence()
    return hg


def extract_packets(n, gates):
    """Maximal v-rooted distributable packets.

    Walking the gates on qubit ``v`` in time order, ``v`` stays inside an open
    packet as long as every gate touching it has ``v`` in its ``zsupp``.  A
    single-qubit gate with ``v`` in ``zsupp`` (RZ, S, T, PHASE) is transparent:
    it neither joins the run nor breaks it.

    Returns ``(packets, covered)``; each multi-qubit gate is assigned to at
    most one packet --- the largest containing it, ties to the smaller root.
    """
    by_qubit = defaultdict(list)
    for g in gates:
        for v in g.qubits:
            by_qubit[v].append(g)

    candidates = []
    for v in range(n):
        run = []
        for g in by_qubit[v]:
            s = spec(g.name)
            keeps_v = g.qubits.index(v) in s.zsupp
            if keeps_v and g.m >= 2:
                run.append(g.gid)
            elif keeps_v:
                continue                       # 1q diagonal: transparent
            else:
                if run:
                    candidates.append((v, run))
                run = []
        if run:
            candidates.append((v, run))

    candidates.sort(key=lambda c: (-len(c[1]), c[0]))
    covered, packets = set(), []
    for root, gids in candidates:
        keep = [i for i in gids if i not in covered]
        if not keep:
            continue
        covered.update(keep)
        packets.append((root, keep))
    return packets, covered


def build_track2(n, gates):
    """H_2: distributable packets (symmetry) + cut-resolved gate nets
    (irreducibility)."""
    by_id = {g.gid: g for g in gates}
    packets, covered = extract_packets(n, gates)
    nets = []

    for root, gids in packets:
        pins = {root}
        members = []
        for gid in gids:
            g = by_id[gid]
            s = spec(g.name)
            remote = tuple(q for q in g.qubits if q != root)
            pins.update(remote)
            if s.diagonal:
                # phases of the branch unitary conditioned on root = |1>,
                # used for the joint gamma of the packet
                loc, mm = g.qubits.index(root), s.m
                ph = diag_phases(g.name, g.params)
                sel = [i for i in range(2 ** mm) if (i >> (mm - 1 - loc)) & 1]
                base = [i & ~(1 << (mm - 1 - loc)) for i in sel]
                members.append((remote, tuple(float(x) for x in
                                              ph[sel] - ph[base]),
                                g.name, g.params, g.qubits))
            else:
                members.append((remote, None, g.name, g.params, g.qubits))
        nets.append(Net(nid=len(nets), kind='packet', pins=tuple(sorted(pins)),
                        pin_mask=sum(1 << q for q in pins), gates=tuple(gids),
                        root=root, members=tuple(members), w_ebit=1,
                        w_logk=math.log(3.0)))

    for g in gates:
        if g.m < 2 or g.gid in covered:
            continue
        nets.append(_gate_net(len(nets), g))

    hg = Hypergraph(n=n, nets=nets, track=2,
                    label='Track 2 (symmetry + irreducibility)')
    hg.build_incidence()
    return hg
