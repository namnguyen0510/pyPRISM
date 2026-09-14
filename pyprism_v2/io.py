"""
pyprism_v2.io
=============
Persistence of circuit instances, for reproducibility and for reuse elsewhere.

Three representations, in decreasing fidelity.

``circuit.json.gz``
    The native, lossless form: every gate with its exact parameters, plus the
    generator arguments that produced it.  ``load_circuit`` reconstructs the
    identical gate list, so a study can be re-scored years later even if the
    generator has changed.  Floats are written with 17 significant digits,
    which round-trips IEEE-754 doubles exactly.

``circuit.qasm``
    OpenQASM 2.0, for Qiskit and anything else that reads it.  Two gates in
    the library have no ``qelib1.inc`` equivalent and are emitted as their
    standard decompositions -- ``iSWAP`` as ``s,s,h,cx,cx,h`` and ``C3Z`` as
    ``h,c3x,h`` -- so the QASM is equivalent but not gate-for-gate identical.
    Use the JSON if you need the original gate identities.

``manifest.json``
    Provenance: package version, interpreter, library versions, the exact
    sweep arguments, and a content hash of the gate list.

Round trip
----------
    >>> from pyprism_v2.io import save_instance, load_instance, to_qiskit
    >>> save_instance('out/circuits', 'n20_k2_s0', layout, gates, 20, meta)
    >>> layout, gates, n, meta = load_instance('out/circuits', 'n20_k2_s0')
    >>> qc = to_qiskit(gates, n)          # needs qiskit
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

from .circuits import Gate
from .gates import GATE_LIB

__all__ = ['CIRCUIT_FORMAT', 'circuit_to_dict', 'dict_to_circuit',
           'save_circuit', 'load_circuit', 'to_qasm', 'save_qasm',
           'to_qiskit', 'save_instance', 'load_instance', 'write_manifest',
           'environment', 'circuit_hash']

CIRCUIT_FORMAT = 1


def _f(x):
    """17 significant digits round-trips an IEEE-754 double exactly."""
    return float(f'{float(x):.17g}')


# ---------------------------------------------------------------------------
#  native format
# ---------------------------------------------------------------------------
def circuit_to_dict(layout, gates, n, meta=None):
    # `meta` is an explicit dict rather than **kwargs: study metadata legitimately
    # contains keys like 'n' and 'depth', which would collide with the
    # positional parameters.
    return {
        'format': CIRCUIT_FORMAT,
        'n_qubits': int(n),
        'depth': len(layout),
        'n_gates': len(gates),
        'meta': dict(meta or {}),
        # gid is implicit in the ordering, but stored so a subset can be
        # round-tripped without renumbering
        'gates': [[g.gid, g.name, list(map(int, g.qubits)),
                   [_f(p) for p in g.params], int(g.layer)] for g in gates],
    }


def dict_to_circuit(d):
    """Return ``(layout, gates, n, meta)``."""
    if d.get('format') != CIRCUIT_FORMAT:
        raise ValueError(f'unsupported circuit format {d.get("format")!r}; '
                         f'this build reads {CIRCUIT_FORMAT}')
    n, depth = int(d['n_qubits']), int(d['depth'])
    gates = [Gate(int(gid), str(name), tuple(int(q) for q in qs),
                  tuple(float(p) for p in ps), int(t))
             for gid, name, qs, ps, t in d['gates']]
    layout = [[] for _ in range(depth)]
    for g in gates:
        layout[g.layer].append(g)
    return layout, gates, n, d.get('meta', {})


def circuit_hash(gates):
    """Content hash of the gate list -- two circuits with the same hash are
    identical down to the last parameter bit."""
    h = hashlib.sha256()
    for g in gates:
        h.update(f'{g.gid}|{g.name}|{tuple(g.qubits)}|'
                 f'{tuple(f"{p:.17g}" for p in g.params)}|{g.layer}\n'
                 .encode())
    return h.hexdigest()[:16]


def save_circuit(path, layout, gates, n, meta=None):
    """Write the native format; ``.gz`` suffix turns on compression."""
    d = circuit_to_dict(layout, gates, n, meta)
    d['sha256_16'] = circuit_hash(gates)
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'wt', encoding='utf-8') as f:
        json.dump(d, f, separators=(',', ':'))
    return path


def load_circuit(path):
    """Read the native format.  Returns ``(layout, gates, n, meta)``."""
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt', encoding='utf-8') as f:
        d = json.load(f)
    layout, gates, n, meta = dict_to_circuit(d)
    stored = d.get('sha256_16')
    if stored and stored != circuit_hash(gates):
        raise ValueError(f'{path}: content hash mismatch, file is corrupt')
    return layout, gates, n, meta


# ---------------------------------------------------------------------------
#  OpenQASM 2.0
# ---------------------------------------------------------------------------
#: name -> (qasm mnemonic, n_params) for gates that map one-to-one onto
#: `qelib1.inc`.  Everything else is decomposed in `_qasm_lines`.
_QASM_DIRECT = {
    'I': 'id', 'X': 'x', 'Y': 'y', 'Z': 'z', 'H': 'h', 'S': 's', 'T': 't',
    'SX': 'sx', 'RX': 'rx', 'RY': 'ry', 'RZ': 'rz', 'PHASE': 'p', 'U3': 'u3',
    'CX': 'cx', 'CY': 'cy', 'CZ': 'cz', 'CH': 'ch', 'CRX': 'crx',
    'CRY': 'cry', 'CRZ': 'crz', 'CPHASE': 'cp', 'CU3': 'cu3', 'SWAP': 'swap',
    'CCX': 'ccx', 'CSWAP': 'cswap', 'C3X': 'c3x',
}


def _qasm_lines(g, reg='q'):
    """QASM statements for one gate."""
    def q(i):
        return f'{reg}[{g.qubits[i]}]'

    args = ('(' + ','.join(f'{p:.17g}' for p in g.params) + ')'
            if g.params else '')
    name = _QASM_DIRECT.get(g.name)
    if name is not None:
        return [f'{name}{args} ' + ','.join(q(i) for i in range(g.m)) + ';']
    if g.name == 'ISWAP':
        # qelib1.inc has no iswap; this is Qiskit's own iSwapGate definition
        a, b = q(0), q(1)
        return [f's {a};', f's {b};', f'h {a};',
                f'cx {a},{b};', f'cx {b},{a};', f'h {b};']
    if g.name == 'C3Z':
        # no c3z in qelib1.inc; conjugating the target of c3x by H gives it
        t = q(3)
        return [f'h {t};',
                'c3x ' + ','.join(q(i) for i in range(4)) + ';',
                f'h {t};']
    raise ValueError(f'no QASM mapping for gate {g.name!r}')


#: every mnemonic emitted must exist in Qiskit's qelib1.inc
QELIB1_GATES = frozenset({
    'id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'sx', 'sxdg',
    'rx', 'ry', 'rz', 'p', 'u', 'u1', 'u2', 'u3',
    'cx', 'cy', 'cz', 'ch', 'csx', 'cp', 'cu1', 'cu3', 'cu',
    'crx', 'cry', 'crz', 'swap', 'ccx', 'cswap', 'rxx', 'rzz',
    'c3x', 'c3sqrtx', 'c4x', 'rccx', 'rc3x',
})


def to_qasm(gates, n, header=''):
    """OpenQASM 2.0 source for the circuit."""
    out = [f'// {line}' for line in (header or '').splitlines()]
    out += ['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{n}];']
    for g in sorted(gates, key=lambda z: (z.layer, z.gid)):
        out.extend(_qasm_lines(g))
    return '\n'.join(out) + '\n'


def save_qasm(path, gates, n, header=''):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(to_qasm(gates, n, header))
    return path


# ---------------------------------------------------------------------------
#  Qiskit
# ---------------------------------------------------------------------------
def to_qiskit(gates, n):
    """Build a ``qiskit.QuantumCircuit`` directly, with no QASM round trip.

    Preferred over the QASM path: every gate keeps its original identity
    (``iSWAP`` stays an ``iSwapGate``) and the parameters are the exact
    doubles, not reparsed decimal text.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import (iSwapGate, PhaseGate, CPhaseGate,
                                        CU3Gate, C3XGate)
    qc = QuantumCircuit(n)
    simple = {
        'I': 'id', 'X': 'x', 'Y': 'y', 'Z': 'z', 'H': 'h', 'S': 's', 'T': 't',
        'SX': 'sx', 'RX': 'rx', 'RY': 'ry', 'RZ': 'rz', 'U3': 'u',
        'CX': 'cx', 'CY': 'cy', 'CZ': 'cz', 'CH': 'ch', 'CRX': 'crx',
        'CRY': 'cry', 'CRZ': 'crz', 'SWAP': 'swap', 'CCX': 'ccx',
        'CSWAP': 'cswap',
    }
    for g in sorted(gates, key=lambda z: (z.layer, z.gid)):
        qs = list(g.qubits)
        if g.name in simple:
            getattr(qc, simple[g.name])(*g.params, *qs)
        elif g.name == 'PHASE':
            qc.append(PhaseGate(g.params[0]), qs)
        elif g.name == 'CPHASE':
            qc.append(CPhaseGate(g.params[0]), qs)
        elif g.name == 'CU3':
            qc.append(CU3Gate(*g.params), qs)
        elif g.name == 'ISWAP':
            qc.append(iSwapGate(), qs)
        elif g.name == 'C3X':
            qc.append(C3XGate(), qs)
        elif g.name == 'C3Z':
            qc.h(qs[3])
            qc.append(C3XGate(), qs)
            qc.h(qs[3])
        else:
            raise ValueError(f'no Qiskit mapping for gate {g.name!r}')
    return qc


# ---------------------------------------------------------------------------
#  provenance and instance bundles
# ---------------------------------------------------------------------------
def environment():
    """Everything needed to explain why a number came out as it did."""
    from . import __version__
    env = {'pyprism_v2': __version__,
           'python': sys.version.split()[0],
           'platform': platform.platform(),
           'machine': platform.machine(),
           'utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    for mod in ('numpy', 'pandas', 'matplotlib', 'pymetis', 'kahypar',
                'pymoo', 'qiskit'):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, '__version__', 'unknown')
        except Exception:
            env[mod] = None
    return env


def save_instance(outdir, tag, layout, gates, n, meta=None, qasm=True):
    """Write ``<tag>.json.gz``, optionally ``<tag>.qasm``, and return the
    per-instance manifest entry."""
    os.makedirs(outdir, exist_ok=True)
    meta = dict(meta or {})
    jpath = os.path.join(outdir, f'{tag}.json.gz')
    save_circuit(jpath, layout, gates, n, meta)
    entry = {'tag': tag, 'n_qubits': int(n), 'depth': len(layout),
             'n_gates': len(gates), 'sha256_16': circuit_hash(gates),
             'json': os.path.basename(jpath), 'qasm': None,
             'bytes_json': os.path.getsize(jpath)}
    if qasm:
        qpath = os.path.join(outdir, f'{tag}.qasm')
        hdr = (f'pyprism_v2 instance {tag}\n'
               f'n={n} depth={len(layout)} gates={len(gates)}\n'
               + '\n'.join(f'{k}={v}' for k, v in sorted(meta.items())))
        save_qasm(qpath, gates, n, hdr)
        entry['qasm'] = os.path.basename(qpath)
        entry['bytes_qasm'] = os.path.getsize(qpath)
    return entry


def load_instance(indir, tag):
    """Inverse of :func:`save_instance`.  Returns ``(layout, gates, n, meta)``."""
    p = os.path.join(indir, f'{tag}.json.gz')
    if not os.path.exists(p):
        p = os.path.join(indir, f'{tag}.json')
    return load_circuit(p)


def write_manifest(outdir, entries, config=None, extra=None):
    """One file that pins the whole study: environment, sweep arguments and
    every instance with its content hash."""
    man = {'format': CIRCUIT_FORMAT, 'environment': environment(),
           'config': config or {}, 'n_instances': len(entries),
           'instances': entries}
    if extra:
        man.update(extra)
    path = os.path.join(outdir, 'manifest.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(man, f, indent=1, default=str)
    return path
