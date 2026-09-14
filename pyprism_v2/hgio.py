"""
pyprism_v2.hgio
===============
Persistence of the two hypergraph substrates.

Both substrates are *derivable* from the gate list --- ``build_track1`` and
``build_track2`` are deterministic --- so saving them is redundant in the
strict sense.  They are saved anyway for three reasons.  A reader can inspect
the substrate without running the package; another tool can consume the
hypergraph directly, in a form close to the hMETIS/KaHyPar text convention;
and a stored substrate pins down what the numbers in a study were computed
from, even if the builder changes later.

Format
------
``{'format': 1, 'track': 1|2, 'n': int, 'nets': [...]}``, one record per net::

    {'nid', 'kind', 'pins', 'w_ebit', 'w_logk',
     'gates',                       # gate ids contained
     'root',                        # packet only
     'name', 'qubits', 'params',    # gate net only
     'members'}                     # packet only, for the exact evaluator

``members`` carries each constituent's remote operands, its diagonal branch
phases (or ``null`` when non-diagonal), and its gate identity.  The identity is
what lets :func:`pyprism_v2.objectives.packet_cost` charge a member whose
remote operands straddle the boundary its own exact cost, so it must survive
the round trip.

A companion ``.hgr`` writer emits the plain hMETIS format used by most
partitioners, for interoperability; it carries pins and integer weights only,
so it is lossy and is not read back.
"""
from __future__ import annotations

import gzip
import json
import math

from .hypergraph import Net, Hypergraph

__all__ = ['HG_FORMAT', 'hypergraph_to_dict', 'dict_to_hypergraph',
           'save_hypergraph', 'load_hypergraph', 'write_hmetis',
           'hypergraph_stats', 'load_dataset_instance']

HG_FORMAT = 1


def _f(x):
    return float(f'{float(x):.17g}')


def hypergraph_to_dict(hg):
    nets = []
    for e in hg.nets:
        d = {'nid': e.nid, 'kind': e.kind, 'pins': list(map(int, e.pins)),
             'gates': list(map(int, e.gates)),
             'w_ebit': int(e.w_ebit), 'w_logk': _f(e.w_logk)}
        if e.kind == 'packet':
            d['root'] = int(e.root)
            d['members'] = [
                {'remote': list(map(int, remote)),
                 'phases': (None if phases is None
                            else [_f(p) for p in phases]),
                 'name': name, 'params': [_f(p) for p in params],
                 'qubits': list(map(int, qubits))}
                for remote, phases, name, params, qubits in e.members]
        else:
            d['name'] = e.name
            d['qubits'] = list(map(int, e.qubits))
            d['params'] = [_f(p) for p in e.params]
        nets.append(d)
    return {'format': HG_FORMAT, 'track': hg.track, 'n': hg.n,
            'label': hg.label, 'nets': nets}


def dict_to_hypergraph(d):
    if d.get('format') != HG_FORMAT:
        raise ValueError(f'unsupported hypergraph format {d.get("format")!r}; '
                         f'this build reads {HG_FORMAT}')
    n = int(d['n'])
    nets = []
    for r in d['nets']:
        pins = tuple(int(q) for q in r['pins'])
        common = dict(nid=int(r['nid']), kind=r['kind'], pins=pins,
                      pin_mask=sum(1 << q for q in pins),
                      gates=tuple(int(g) for g in r['gates']),
                      w_ebit=int(r['w_ebit']), w_logk=float(r['w_logk']))
        if r['kind'] == 'packet':
            members = tuple(
                (tuple(int(q) for q in m['remote']),
                 (None if m['phases'] is None
                  else tuple(float(p) for p in m['phases'])),
                 m['name'], tuple(float(p) for p in m['params']),
                 tuple(int(q) for q in m['qubits']))
                for m in r['members'])
            nets.append(Net(root=int(r['root']), members=members, **common))
        else:
            nets.append(Net(name=r['name'],
                            qubits=tuple(int(q) for q in r['qubits']),
                            params=tuple(float(p) for p in r['params']),
                            **common))
    hg = Hypergraph(n=n, nets=nets, track=int(d['track']),
                    label=d.get('label', ''))
    hg.build_incidence()
    return hg


def save_hypergraph(path, hg):
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'wt', encoding='utf-8') as f:
        json.dump(hypergraph_to_dict(hg), f, separators=(',', ':'))
    return path


def load_hypergraph(path):
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt', encoding='utf-8') as f:
        return dict_to_hypergraph(json.load(f))


def write_hmetis(path, hg, scale=1000):
    """Plain hMETIS hypergraph, for external partitioners.

    Line 1 is ``<nets> <vertices> 1`` (the trailing 1 means weighted nets).
    Each following line is ``<weight> <pin> <pin> ...`` with 1-based vertices.
    Weights must be integers, so ``w_ebit`` is scaled; this is lossy and the
    file is write-only.
    """
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'{len(hg.nets)} {hg.n} 1\n')
        for e in hg.nets:
            w = max(1, int(round(e.w_ebit * scale)))
            f.write(str(w) + ' '
                    + ' '.join(str(q + 1) for q in e.pins) + '\n')
    return path


def load_dataset_instance(instdir, rebuild=False, check=True):
    """Load one directory written by ``generate_dataset.py``.

    Returns ``(layout, gates, n, meta, h1, h2)``.  With ``rebuild`` the two
    substrates are rebuilt from the gate list instead of read from disk, which
    is how the stored files are checked against the current builder; ``check``
    verifies the circuit against the SHA-256 recorded at generation time, so a
    silently edited or truncated instance is caught here rather than showing up
    as an inexplicable number three cells later.
    """
    import os

    from .io import load_instance, circuit_hash
    from .hypergraph import build_track1, build_track2

    layout, gates, n, meta = load_instance(instdir, 'circuit')
    if check and meta.get('sha256_16'):
        got = circuit_hash(gates)
        if got != meta['sha256_16']:
            raise ValueError(f'{instdir}: circuit hash {got} does not match '
                             f'the recorded {meta["sha256_16"]}')
    if rebuild:
        return layout, gates, n, meta, build_track1(n, gates), \
            build_track2(n, gates)
    return (layout, gates, n, meta,
            load_hypergraph(os.path.join(instdir, 'H1.json.gz')),
            load_hypergraph(os.path.join(instdir, 'H2.json.gz')))


def hypergraph_stats(hg):
    s = hg.stats()
    sizes = [len(e.pins) for e in hg.nets] or [0]
    return {**s,
            'net_size_min': int(min(sizes)), 'net_size_max': int(max(sizes)),
            'gate_nets': sum(1 for e in hg.nets if e.kind == 'gate'),
            'packet_nets': sum(1 for e in hg.nets if e.kind == 'packet'),
            'packed_gates': sum(len(e.gates) for e in hg.nets
                                if e.kind == 'packet'),
            'total_w_ebit': int(sum(e.w_ebit for e in hg.nets)),
            'total_w_logk': float(sum(e.w_logk for e in hg.nets))}
