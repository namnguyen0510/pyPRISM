#!/usr/bin/env python3
"""exp05_suites.py -- PRISM against every comparator on the standard suites.

Answers the reviewer objection that the evaluation uses only random circuits.
Three community suites are ingested into the ``pyprism_v2`` instance format and
then swept with exactly the roster of :mod:`benchmark_instance`, so the numbers
are directly comparable with the random-circuit study.

  QASMBench   Li et al., https://github.com/pnnl/QASMBench   (OpenQASM 2 files)
  MQT Bench   Quetschlich et al., https://github.com/munich-quantum-toolkit/bench
  SupermarQ   Tomesh et al., https://github.com/Infleqtion/client-superstaq

Three stages, each resumable:

    python3 exp05_suites.py ingest --suites qasmbench,mqtbench,supermarq
    python3 exp05_suites.py sweep
    python3 exp05_suites.py report

Ingestion notes that matter for the result
------------------------------------------
* QASMBench ships an algorithmic ``x.qasm`` and a hardware-compiled
  ``x_transpiled.qasm`` for most circuits.  We take the **algorithmic** one.
  The transpiled file has already been rewritten into a device basis, which
  destroys exactly the diagonal structure the packet construction exploits;
  benchmarking on it would measure the compiler, not the partitioner.
* Circuits are rebased onto the PRISM gate library with Qiskit's
  ``BasisTranslator``.  Every rewrite is exact.  A gate that can only be reached
  through ``cx`` (``rxx``, ``ryy``, ``ecr``, ...) loses its diagonal form; the
  count of such gates is recorded per instance as ``rebase_via_cx`` so the
  effect on packing can be audited rather than assumed.
* Qiskit 2.x refuses ``c3x``/``c3z``/``mcx`` as basis gates, so a gate with
  three or more controls is decomposed to CCX/CX and the C3X/C3Z rows of
  Table 1 are not exercised by this corpus.  The count is recorded as
  ``rebase_lost_multicontrol``.
* ``measure``, ``barrier``, ``reset`` and classical control are dropped.  A
  circuit containing mid-circuit measurement followed by conditional operations
  is skipped, since its partition cost is not defined by the model of Sec. 3.1.
* ``cu(t,p,l,g)`` is emitted as ``PHASE(g)`` on the control followed by
  ``CU3(t,p,l)``; dropping the control phase would make the rewrite inexact.
* Circuits of extreme depth are excluded by ``--maxdepth`` (absolute) and
  ``--max-depth-ratio`` (depth relative to register size).  Both stages honour
  them: ``ingest`` screens before and after the rebase, ``sweep`` re-applies
  them to an already-ingested corpus so the ceiling can be tightened without
  re-ingesting.  Cost is roughly linear in depth for the causal-cone pass and
  for every incremental re-pricing, so a handful of very deep circuits
  otherwise dominates the wall clock while contributing no more partitioning
  structure than a shallow one.

Dependencies: qiskit for all three suites; ``mqt.bench`` and ``supermarq`` only
for their own suites.  Each suite is skipped with a warning if unavailable.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pyprism_v2.circuits import Gate, circuit_stats                # noqa: E402
from pyprism_v2.hypergraph import build_track1, build_track2, \
    extract_packets                                                # noqa: E402
from pyprism_v2.hgio import save_hypergraph, write_hmetis, \
    hypergraph_stats                                               # noqa: E402
from pyprism_v2.io import save_instance, circuit_hash              # noqa: E402
from pyprism_v2.lightcone import light_cone_graph, lightcone_stats  # noqa: E402

# ---------------------------------------------------------------------------
# progress reporting
# ---------------------------------------------------------------------------

def _hr(ch='-', w=96):
    print(ch * w, flush=True)


def _banner(title, w=96):
    _hr('=', w)
    print(title, flush=True)
    _hr('=', w)


def _hms(sec):
    sec = int(max(sec, 0))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f'{h:d}:{m:02d}:{s:02d}' if h else f'{m:d}:{s:02d}'


def depth_ok(meta, maxdepth, ratio):
    """``(True, '')`` or ``(False, reason)`` for the shared depth limits.

    Single source of truth for ``exp05 sweep`` and ``exp06``: an absolute
    ceiling and a ceiling relative to the register size.  Either may be 0 to
    disable it.
    """
    dep = meta.get('depth', 0)
    nq = max(meta.get('n', 1), 1)
    if maxdepth and dep > maxdepth:
        return False, f'depth {dep} > {maxdepth}'
    if ratio and dep > ratio * nq:
        return False, f'depth/n = {dep / nq:.0f} > {ratio}'
    return True, ''


def _eta(done, total, t0):
    """``elapsed / mean / ETA`` for a loop that has finished ``done`` of
    ``total`` items since ``t0``."""
    el = time.time() - t0
    mean = el / max(done, 1)
    return f'{_hms(el)} elapsed | {mean:5.1f}s/item | ETA {_hms(mean * (total - done))}'


# ---------------------------------------------------------------------------
# gate mapping
# ---------------------------------------------------------------------------

#: Qiskit basis we rebase onto.  Every name here has an exact counterpart in
#: ``pyprism_v2.gates.GATE_LIB``; nothing else is allowed to survive.
NATIVE_BASIS = ['id', 'h', 'x', 'y', 'z', 's', 't', 'sx',
                'rx', 'ry', 'rz', 'p', 'u3', 'u',
                'cx', 'cy', 'cz', 'ch', 'crx', 'cry', 'crz', 'cp', 'cu3',
                'swap', 'iswap', 'ccx', 'cswap']

#: qiskit name -> (PRISM name, arity).  Parameters carry through unchanged
#: except where noted in :func:`_map_op`.
QK2PRISM = {
    'id': ('I', 1), 'h': ('H', 1), 'x': ('X', 1), 'y': ('Y', 1),
    'z': ('Z', 1), 's': ('S', 1), 't': ('T', 1), 'sx': ('SX', 1),
    'rx': ('RX', 1), 'ry': ('RY', 1), 'rz': ('RZ', 1), 'p': ('PHASE', 1),
    'u1': ('PHASE', 1), 'u3': ('U3', 1), 'u': ('U3', 1),
    'cx': ('CX', 2), 'cy': ('CY', 2), 'cz': ('CZ', 2), 'ch': ('CH', 2),
    'crx': ('CRX', 2), 'cry': ('CRY', 2), 'crz': ('CRZ', 2),
    'u2': ('U3', 1),
    'cp': ('CPHASE', 2), 'cu1': ('CPHASE', 2), 'cu3': ('CU3', 2),
    'cu': ('CU3', 2),
    'swap': ('SWAP', 2), 'iswap': ('ISWAP', 2),
    'ccx': ('CCX', 3), 'cswap': ('CSWAP', 3),
    'c3x': ('C3X', 4), 'c3z': ('C3Z', 4),
}

#: control-flow operations the cost model of Sec. 3.1 does not cover
CONTROL_FLOW = {'if_else', 'while_loop', 'for_loop', 'switch_case', 'c_if'}

#: dropped without comment
IGNORED = {'barrier', 'measure', 'reset', 'delay', 'snapshot', 'save_statevector'}

#: gates that only reach the native basis through ``cx``, so their diagonal
#: structure (and hence packability) is not preserved by the rebase
VIA_CX = {'rxx', 'ryy', 'rzx', 'ecr', 'dcx', 'xx_plus_yy', 'xx_minus_yy'}

#: Qiskit 2.x refuses ``c3x``/``c3z``/``mcx`` as basis gates, so a gate with
#: three or more controls is decomposed to CCX/CX by the rebase and its C3X /
#: C3Z entry in Table 1 is not exercised.  Counted per instance so the loss is
#: visible rather than silent.
MULTICONTROL = {'mcx', 'mcx_gray', 'mcx_recursive', 'mcx_vchain', 'c3x',
                'c4x', 'c3sx', 'mcphase', 'mcrz', 'mcry', 'mcrx'}


def _map_op(name, params, qubits):
    """One Qiskit instruction -> ``[(prism_name, params, qubits), ...]``.

    A list rather than a single gate because ``cu(theta,phi,lam,gamma)`` is
    ``cu3(theta,phi,lam)`` preceded by a phase ``gamma`` on the control, and
    dropping that phase would make the rewrite inexact.  Returns ``None`` when
    the operation has no exact expression in the PRISM library.
    """
    name = name.lower()
    if name not in QK2PRISM:
        return None
    pname, arity = QK2PRISM[name]
    if len(qubits) != arity:
        return None
    p = tuple(float(x) for x in params)
    out = []
    if name == 'u2':                      # u2(phi,lam) == u3(pi/2,phi,lam)
        p = (math.pi / 2,) + p
    if name == 'cu':                      # cu(t,p,l,g) == p(g) on ctrl . cu3
        if len(p) == 4:
            out.append(('PHASE', (p[3],), (qubits[0],)))
        p = p[:3]
    if pname == 'U3' and len(p) < 3:
        p = p + (0.0,) * (3 - len(p))
    if pname == 'CU3' and len(p) < 3:
        p = p + (0.0,) * (3 - len(p))
    out.append((pname, p, tuple(qubits)))
    return out


# ---------------------------------------------------------------------------
# qiskit circuit -> pyprism_v2 layout
# ---------------------------------------------------------------------------

def qiskit_to_layout(qc, optimisation_level=0):
    """Rebase, schedule ASAP, and return ``(layout, gates, n, info)``.

    ``layout`` is a list of layers with pairwise-disjoint supports, as required
    by Sec. 3.1.  Raises ``ValueError`` if the circuit cannot be expressed in
    the model (classical control, mid-circuit conditionals).
    """
    from qiskit import transpile

    raw_names = collections.Counter(i.operation.name.lower() for i in qc.data)
    for inst in qc.data:
        op = inst.operation
        nm = op.name.lower()
        if nm in CONTROL_FLOW or getattr(op, 'condition', None) is not None:
            raise ValueError(f'control flow / classical condition ({nm})')

    # NB. ``remove_final_measurements`` returns a *new* circuit; do not fall
    # back with ``or qc``, because an all-measurement circuit becomes empty and
    # QuantumCircuit is falsy when it has no instructions.
    stripped = qc.remove_final_measurements(inplace=False)
    if stripped is not None:
        qc = stripped
    t = transpile(qc, basis_gates=NATIVE_BASIS,
                  optimization_level=optimisation_level, seed_transpiler=0)

    n = t.num_qubits
    index = {q: i for i, q in enumerate(t.qubits)}
    free = [0] * n                      # next free layer per qubit
    layers = collections.defaultdict(list)
    gates, gid, dropped = [], 0, collections.Counter()

    for inst in t.data:
        nm = inst.operation.name.lower()
        if nm in IGNORED:
            continue
        qs = tuple(index[q] for q in inst.qubits)
        emitted = _map_op(nm, getattr(inst.operation, 'params', ()), qs)
        if emitted is None:
            dropped[nm] += 1
            continue
        for pname, params, pq in emitted:
            lay = max(free[q] for q in pq)
            g = Gate(gid, pname, pq, params, lay)
            layers[lay].append(g)
            gates.append(g)
            gid += 1
            for q in pq:
                free[q] = lay + 1

    if dropped:
        raise ValueError(f'unmapped operations after rebase: {dict(dropped)}')
    if not gates:
        raise ValueError('no gates after rebase')

    depth = max(layers) + 1
    layout = [layers.get(t_, []) for t_ in range(depth)]
    info = {
        'rebase_via_cx': sum(v for k, v in raw_names.items() if k in VIA_CX),
        'rebase_lost_multicontrol':
            sum(v for k, v in raw_names.items() if k in MULTICONTROL),
        'raw_gate_census': dict(raw_names),
        'raw_size': int(sum(raw_names.values())),
        'mapped_size': len(gates),
    }
    return layout, gates, n, info


def cirq_to_qiskit(circuit):
    """SupermarQ returns cirq circuits; go through QASM 2."""
    import cirq
    import qiskit.qasm2 as q2
    txt = cirq.qasm(circuit)
    return q2.loads(txt, custom_instructions=q2.LEGACY_CUSTOM_INSTRUCTIONS)


# ---------------------------------------------------------------------------
# suite readers -- each yields (tag, suite, family, qiskit circuit)
# ---------------------------------------------------------------------------

def iter_qasmbench(root, include_transpiled=False):
    import qiskit.qasm2 as q2
    pats = [os.path.join(root, sz, '*', '*.qasm')
            for sz in ('small', 'medium', 'large')]
    files = sorted(f for p in pats for f in glob.glob(p))
    print(f'  QASMBench: {len(files)} .qasm files under {root}', flush=True)
    for f in files:
        base = os.path.basename(f)[:-5]
        if base.endswith('_transpiled') and not include_transpiled:
            continue
        try:
            qc = q2.load(f, include_path=[root, os.path.dirname(f)],
                         custom_instructions=q2.LEGACY_CUSTOM_INSTRUCTIONS)
        except Exception as e:                                   # noqa: BLE001
            print(f'    x {base:34s} unreadable: '
                  f'{type(e).__name__}: {e}'[:120], flush=True)
            continue
        family = base.rsplit('_n', 1)[0]
        yield f'qb_{base}', 'QASMBench', family, qc


def iter_mqtbench(sizes):
    try:
        from mqt.bench import get_benchmark, BenchmarkLevel
        from mqt.bench.benchmarks import get_available_benchmark_names
    except Exception as e:                                       # noqa: BLE001
        print(f'  ! mqt.bench unavailable ({e}); suite skipped')
        return
    names = sorted(get_available_benchmark_names())
    print(f'  MQT Bench: {len(names)} families x {len(sizes)} sizes '
          f'= {len(names) * len(sizes)} candidates', flush=True)
    for name in names:
        for n in sizes:
            try:
                qc = get_benchmark(name, BenchmarkLevel.ALG, n)
            except Exception as e:                               # noqa: BLE001
                print(f'    x {f"mqt_{name}_n{n}":<32s} not generated: '
                      f'{type(e).__name__}: {e}'[:120], flush=True)
                continue
            yield f'mqt_{name}_n{qc.num_qubits}', 'MQTBench', name, qc


def iter_supermarq(sizes):
    try:
        import supermarq  # noqa: F401
        from supermarq.benchmarks import (ghz, mermin_bell, bit_code,
                                          phase_code, hamiltonian_simulation,
                                          qaoa_vanilla_proxy,
                                          qaoa_fermionic_swap_proxy, vqe_proxy)
    except Exception as e:                                       # noqa: BLE001
        print(f'  ! supermarq unavailable ({e}); suite skipped')
        return

    builders = [
        ('ghz', lambda n: ghz.GHZ(n)),
        ('mermin_bell', lambda n: mermin_bell.MerminBell(n)),
        ('bit_code', lambda n: bit_code.BitCode(n, 1, [0] * n)),
        ('phase_code', lambda n: phase_code.PhaseCode(n, 1, [0] * n)),
        ('hamiltonian_sim', lambda n: hamiltonian_simulation.HamiltonianSimulation(n)),
        ('qaoa_vanilla', lambda n: qaoa_vanilla_proxy.QAOAVanillaProxy(n)),
        ('qaoa_fswap', lambda n: qaoa_fermionic_swap_proxy.QAOAFermionicSwapProxy(n)),
        ('vqe', lambda n: vqe_proxy.VQEProxy(n, 1)),
    ]
    print(f'  SupermarQ: {len(builders)} families x {len(sizes)} sizes '
          f'= {len(builders) * len(sizes)} candidates', flush=True)
    for name, make in builders:
        for n in sizes:
            try:
                b = make(n)
                c = b.circuit()
                if isinstance(c, (list, tuple)):
                    c = c[0]
                qc = cirq_to_qiskit(c)
            except Exception as e:                               # noqa: BLE001
                print(f'    x {f"sm_{name}_n{n}":<32s} not generated: '
                      f'{type(e).__name__}: {e}'[:120], flush=True)
                continue
            yield f'sm_{name}_n{qc.num_qubits}', 'SupermarQ', name, qc


# ---------------------------------------------------------------------------
# stage 1: ingest
# ---------------------------------------------------------------------------

def write_instance(outdir, tag, suite, family, layout, gates, n, info, hgr=True):
    d = os.path.join(outdir, tag)
    os.makedirs(d, exist_ok=True)
    h1 = build_track1(n, gates)
    h2 = build_track2(n, gates)
    packets, covered = extract_packets(n, gates)
    W, first, dtf = light_cone_graph(n, layout)
    s1, s2 = hypergraph_stats(h1), hypergraph_stats(h2)
    depth = len(layout)

    meta = {
        'tag': tag, 'n': n, 'depth': depth, 'suite': suite, 'family': family,
        'generator': 'exp05_suites.qiskit_to_layout', 'relabel': False,
        'sha256_16': circuit_hash(gates),
        **circuit_stats(gates, n, depth),
        'gate_census': dict(collections.Counter(g.name for g in gates)),
        'H1': s1, 'H2': s2,
        'compression': s1['nets'] / max(s2['nets'], 1),
        'packets': len(packets), 'packed_gates': len(covered),
        'lightcone': lightcone_stats(W, first, dtf, n, depth),
        **info,
    }
    save_instance(d, 'circuit', layout, gates, n, meta=meta, qasm=True)
    save_hypergraph(os.path.join(d, 'H1.json.gz'), h1)
    save_hypergraph(os.path.join(d, 'H2.json.gz'), h2)
    if hgr:
        write_hmetis(os.path.join(d, 'H1.hgr'), h1)
        write_hmetis(os.path.join(d, 'H2.hgr'), h2)
    with open(os.path.join(d, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    return meta


def cmd_ingest(a):
    sizes = [int(x) for x in a.sizes.split(',')]
    suites = [s.strip().lower() for s in a.suites.split(',')]
    os.makedirs(a.out, exist_ok=True)

    sources = []
    if 'qasmbench' in suites:
        sources.append(iter_qasmbench(a.qasmbench, a.include_transpiled))
    if 'mqtbench' in suites:
        sources.append(iter_mqtbench(sizes))
    if 'supermarq' in suites:
        sources.append(iter_supermarq(sizes))

    _banner(f'INGEST  suites={",".join(suites)}  n in [{a.nmin},{a.nmax}]  '
            f'gates <= {a.maxgates}')
    print(f'  destination : {a.out}')
    print(f'  sizes asked : {sizes}')
    print(f'  depth limit : {a.maxdepth or "none"}   '
          f'depth/n limit: {a.max_depth_ratio or "none"}')
    print(f'  hMETIS files: {"no" if a.no_hgr else "yes"}   '
          f'force rebuild: {a.force}')
    _hr()

    index, skipped = [], []
    reasons = collections.Counter()
    t_start = time.time()
    seen = cached = 0
    hdr = (f'  {"tag":34s} {"n":>4s} {"depth":>6s} {"gates":>7s} '
           f'{"|E1|":>6s} {"|E2|":>6s} {"pack":>6s} {"cx":>4s} {"s":>6s}')

    for src in sources:
        for tag, suite, family, qc in src:
            seen += 1
            d = os.path.join(a.out, tag)
            if os.path.exists(os.path.join(d, 'meta.json')) and not a.force:
                with open(os.path.join(d, 'meta.json')) as f:
                    m = json.load(f)
                # A cached instance was ingested under whatever ceiling was in
                # force then, so re-apply the current limits; otherwise the
                # index silently disagrees with what `sweep` will run.
                dep, nq = m.get('depth', 0), max(m.get('n', 1), 1)
                if a.maxdepth and dep > a.maxdepth:
                    why = f'depth {dep} > {a.maxdepth} (cached)'
                    skipped.append((tag, why)); reasons['too deep'] += 1
                    print(f'    - {tag:34s} {why}', flush=True)
                    continue
                if a.max_depth_ratio and dep > a.max_depth_ratio * nq:
                    why = f'depth/n = {dep / nq:.0f} > {a.max_depth_ratio} (cached)'
                    skipped.append((tag, why))
                    reasons['extreme aspect ratio'] += 1
                    print(f'    - {tag:34s} {why}', flush=True)
                    continue
                index.append(m)
                cached += 1
                print(f'    = {tag:34s} cached', flush=True)
                continue
            if qc.num_qubits < a.nmin or qc.num_qubits > a.nmax:
                why = f'n={qc.num_qubits} outside [{a.nmin},{a.nmax}]'
                skipped.append((tag, why)); reasons['register size'] += 1
                if a.verbose:
                    print(f'    - {tag:34s} {why}', flush=True)
                continue
            if qc.size() > a.maxgates:
                why = f'{qc.size()} gates > {a.maxgates}'
                skipped.append((tag, why)); reasons['too many gates'] += 1
                if a.verbose:
                    print(f'    - {tag:34s} {why}', flush=True)
                continue
            # Cheap pre-rebase depth screen.  The rebase can only make a
            # circuit deeper, so a circuit already over the ceiling cannot come
            # back under it, and we skip before paying for the transpile.
            raw_depth = qc.depth()
            if a.maxdepth and raw_depth > a.maxdepth:
                why = f'depth {raw_depth} > {a.maxdepth} (pre-rebase)'
                skipped.append((tag, why)); reasons['too deep'] += 1
                print(f'    - {tag:34s} {why}', flush=True)
                continue
            t0 = time.time()
            try:
                layout, gates, n, info = qiskit_to_layout(qc)
            except Exception as e:                               # noqa: BLE001
                why = f'{type(e).__name__}: {e}'[:110]
                skipped.append((tag, why)); reasons['not expressible'] += 1
                print(f'    x {tag:34s} {why}', flush=True)
                continue
            if not any(g.m >= 2 for g in gates):
                skipped.append((tag, 'no multi-qubit gates'))
                reasons['no multi-qubit gates'] += 1
                print(f'    - {tag:34s} no multi-qubit gates', flush=True)
                continue
            # Authoritative depth check: the ASAP layout after rebasing is what
            # the partitioners and the causal-cone pass actually walk, and the
            # rebase can inflate depth several-fold.
            depth = len(layout)
            if a.maxdepth and depth > a.maxdepth:
                why = f'depth {depth} > {a.maxdepth} (post-rebase, raw {raw_depth})'
                skipped.append((tag, why)); reasons['too deep'] += 1
                print(f'    - {tag:34s} {why}', flush=True)
                continue
            if a.max_depth_ratio and depth > a.max_depth_ratio * n:
                why = (f'depth/n = {depth / n:.0f} > {a.max_depth_ratio} '
                       f'(depth {depth}, n {n})')
                skipped.append((tag, why)); reasons['extreme aspect ratio'] += 1
                print(f'    - {tag:34s} {why}', flush=True)
                continue
            if len(index) % 20 == 0:
                print(hdr, flush=True)
            meta = write_instance(a.out, tag, suite, family,
                                  layout, gates, n, info, hgr=not a.no_hgr)
            meta['ingest_seconds'] = round(time.time() - t0, 2)
            index.append(meta)
            print(f'    + {tag:34s} {n:4d} {len(layout):6d} {len(gates):7d} '
                  f'{meta["H1"]["nets"]:6d} {meta["H2"]["nets"]:6d} '
                  f'{meta["compression"]:5.2f}x {info["rebase_via_cx"]:4d} '
                  f'{meta["ingest_seconds"]:6.2f}', flush=True)

    with open(os.path.join(a.out, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump({'instances': index, 'skipped': skipped}, f, indent=1)

    _hr()
    by = collections.Counter(m['suite'] for m in index)
    print(f'  candidates seen : {seen}')
    print(f'  ingested        : {len(index)}  ({cached} reused from cache)')
    for k in sorted(by):
        sub_i = [m for m in index if m['suite'] == k]
        ns = [m['n'] for m in sub_i]
        cmp_ = [m['compression'] for m in sub_i]
        print(f'      {k:12s} {len(sub_i):4d} instances, '
              f'n {min(ns)}-{max(ns)}, compression '
              f'{min(cmp_):.2f}-{max(cmp_):.2f}x '
              f'(median {statistics.median(cmp_):.2f}x)')
    print(f'  skipped         : {len(skipped)}')
    for k, v in reasons.most_common():
        print(f'      {k:24s} {v:4d}')
    viacx = [m for m in index if m.get('rebase_via_cx', 0)]
    lost = [m for m in index if m.get('rebase_lost_multicontrol', 0)]
    if viacx:
        print(f'  ! {len(viacx)} instances contain gates rebased through cx '
              f'(diagonal form not preserved); see meta["rebase_via_cx"]')
    if lost:
        print(f'  ! {len(lost)} instances contain >=3-control gates decomposed '
              f'to CCX/CX; see meta["rebase_lost_multicontrol"]')
    print(f'  wall clock      : {_hms(time.time() - t_start)}')
    print(f'  index           : {os.path.join(a.out, "index.json")}')
    _hr()


# ---------------------------------------------------------------------------
# stage 2: sweep -- identical roster to the random-circuit study
# ---------------------------------------------------------------------------

def cmd_sweep(a):
    from benchmark_instance import run as run_instance

    dirs = sorted(d for d in glob.glob(os.path.join(a.instances, '*'))
                  if os.path.isdir(d)
                  and os.path.exists(os.path.join(d, 'meta.json')))

    # A corpus ingested with a looser ceiling can still be swept selectively,
    # so the same limits are available here without re-ingesting.
    deep = []
    if a.maxdepth or a.max_depth_ratio:
        keep = []
        for d in dirs:
            try:
                with open(os.path.join(d, 'meta.json')) as f:
                    m = json.load(f)
            except Exception:                                    # noqa: BLE001
                keep.append(d)
                continue
            ok, why = depth_ok(m, a.maxdepth, a.max_depth_ratio)
            if ok:
                keep.append(d)
            else:
                deep.append((os.path.basename(d), why))
        dirs = keep
    if a.limit:
        dirs = dirs[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    if not dirs:
        print(f'no ingested instances under {a.instances}.\n'
              f'Run:  python3 exp05_suites.py ingest --out {a.instances}')
        return

    _banner(f'SWEEP  {len(dirs)} instances  ->  {a.out}')
    print(f'  balance tolerance eps = {a.eps}   PRISM eps = {a.prism_eps}')
    print(f'  PRISM budget B = {a.prism_B}, preference rungs = {a.n_pref}, '
          f'seed = {a.seed}, skip_slow = {a.skip_slow}')
    print(f'  depth limit = {a.maxdepth or "none"}, '
          f'depth/n limit = {a.max_depth_ratio or "none"}'
          + (f'  ({len(deep)} instances excluded)' if deep else ''))
    for t, why in deep[:20]:
        print(f'      - {t:32s} {why}')
    if len(deep) > 20:
        print(f'      - ... and {len(deep) - 20} more')
    _hr()

    t_start = time.time()
    ok = fail = cached = 0
    times = []
    for i, d in enumerate(dirs, 1):
        tag = os.path.basename(d)
        try:
            with open(os.path.join(d, 'meta.json')) as f:
                m = json.load(f)
        except Exception:                                        # noqa: BLE001
            m = {}
        shape = (f'n={m.get("n", "?"):>4} d={m.get("depth", "?"):>6} '
                 f'g={m.get("gates", "?"):>7} '
                 f'|E2|={m.get("H2", {}).get("nets", "?"):>6}')
        done = os.path.join(a.out, tag, 'results.json')
        if os.path.exists(done) and not a.force:
            cached += 1
            print(f'  [{i:4d}/{len(dirs)}] = {tag:32s} {shape}  cached',
                  flush=True)
            continue
        print(f'  [{i:4d}/{len(dirs)}] . {tag:32s} {shape}  running...',
              flush=True)
        t0 = time.time()
        try:
            run_instance(d, a.out, eps=a.eps, seed=a.seed, prism_B=a.prism_B,
                         n_pref=a.n_pref, skip_slow=a.skip_slow,
                         verbose=False, save_parts=False,
                         prism_eps=a.prism_eps)
            dt = time.time() - t0
            times.append(dt)
            ok += 1
            best = ''
            try:
                with open(done) as f:
                    R = [r for r in json.load(f).get('results', [])
                         if r.get('E') is not None]
                if R:
                    b = min(R, key=lambda r: r['E'])
                    pr = [r for r in R if r.get('method') == 'PRISM']
                    best = (f'  {len(R):2d} methods | best E={b["E"]:.0f} '
                            f'({b["method"]})')
                    if pr:
                        best += f' | PRISM E={pr[0]["E"]:.0f}'
            except Exception:                                    # noqa: BLE001
                pass
            print(f'  [{i:4d}/{len(dirs)}] + {tag:32s} {dt:7.1f}s{best}')
            print(f'                       {_eta(ok, len(dirs) - cached, t_start)}',
                  flush=True)
        except Exception:                                        # noqa: BLE001
            fail += 1
            print(f'  [{i:4d}/{len(dirs)}] x {tag:32s} FAILED')
            traceback.print_exc(limit=3)

    _hr()
    print(f'  completed {ok}, cached {cached}, failed {fail}')
    if times:
        print(f'  per-instance  min {min(times):.1f}s  '
              f'median {statistics.median(times):.1f}s  max {max(times):.1f}s')
    print(f'  wall clock    {_hms(time.time() - t_start)}')
    print(f'  next          python3 exp05_suites.py report '
          f'--results {a.out} --instances {a.instances}')
    _hr()


# ---------------------------------------------------------------------------
# stage 3: report
# ---------------------------------------------------------------------------

def _friedman_ranks(per_instance):
    """Mean rank of each method over the instances where all are present."""
    methods = set.intersection(*(set(r) for r in per_instance.values())) \
        if per_instance else set()
    acc = collections.defaultdict(list)
    for tag, res in per_instance.items():
        order = sorted(methods, key=lambda m: res[m]['E'])
        # average ranks over ties
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and res[order[j + 1]]['E'] == res[order[i]]['E']:
                j += 1
            r = (i + j) / 2 + 1
            for k in range(i, j + 1):
                acc[order[k]].append(r)
            i = j + 1
    return {m: statistics.mean(v) for m, v in acc.items()}, sorted(methods)


def _load(resdir, instdir=None):
    per = {}
    meta = {}
    for p in sorted(glob.glob(os.path.join(resdir, '*', 'results.json'))):
        with open(p) as f:
            J = json.load(f)
        tag = os.path.basename(os.path.dirname(p))
        inst = dict(J.get('instance', {}))
        # `benchmark_instance.run` rebuilds its own meta and drops the suite
        # label, so recover it from the instance directory when available.
        if instdir:
            mp = os.path.join(instdir, tag, 'meta.json')
            if os.path.exists(mp):
                with open(mp) as f:
                    m = json.load(f)
                inst.setdefault('suite', m.get('suite'))
                inst.setdefault('family', m.get('family'))
                inst['compression'] = m.get('compression', inst.get('compression'))
        if not inst.get('suite'):
            inst['suite'] = {'qb': 'QASMBench', 'mqt': 'MQTBench',
                             'sm': 'SupermarQ'}.get(tag.split('_')[0], '?')
        J['instance'] = inst
        rows = {}
        for r in J.get('results', []):
            m = r.get('method')
            if m and r.get('E') is not None and r.get('log10_shots') is not None:
                rows[m] = r
        if rows:
            per[tag] = rows
            meta[tag] = J['instance']
    return per, meta


def cmd_report(a):
    per, meta = _load(a.results, getattr(a, 'instances', None))
    if not per:
        print(f'no results under {a.results}')
        return
    _banner(f'REPORT  {len(per)} instances  <-  {a.results}')
    suites = collections.defaultdict(dict)
    for tag, rows in per.items():
        suites[meta[tag].get('suite', '?')][tag] = rows
    suites['ALL'] = per

    out_rows = {}
    for suite in sorted(suites):
        sub = suites[suite]
        ranks, methods = _friedman_ranks(sub)
        recs = []
        for m in methods:
            E = [sub[t][m]['E'] for t in sub]
            K = [sub[t][m]['log10_shots'] for t in sub]
            rho = [k / e for e, k in zip(E, K) if e]
            sec = [sub[t][m].get('seconds', float('nan')) for t in sub]
            recs.append({
                'method': m, 'n': len(E),
                'E_mean': statistics.mean(E),
                'E_sd': statistics.pstdev(E),
                'K_mean': statistics.mean(K),
                'K_sd': statistics.pstdev(K),
                'rho_mean': statistics.mean(rho) if rho else float('nan'),
                'rho_sd': statistics.pstdev(rho) if rho else float('nan'),
                'sec_mean': statistics.mean([s for s in sec if s == s] or [0]),
                'rank': ranks.get(m, float('nan')),
            })
        recs.sort(key=lambda r: r['rank'])
        out_rows[suite] = recs
        fams = collections.Counter(meta[t].get('family', '?') for t in sub)
        print(f'\n=== {suite}  ({len(sub)} instances, {len(fams)} families, '
              f'{len(methods)} methods common to all) ===')
        print(f'{"method":22s} {"E":>16s} {"log10 g^2":>16s} '
              f'{"rho":>14s} {"rank":>6s} {"s":>7s}')
        for r in recs:
            print(f'{r["method"]:22s} {r["E_mean"]:8.1f}+-{r["E_sd"]:6.1f} '
                  f'{r["K_mean"]:8.1f}+-{r["K_sd"]:6.1f} '
                  f'{r["rho_mean"]:7.4f}+-{r["rho_sd"]:5.4f} '
                  f'{r["rank"]:6.2f} {r["sec_mean"]:7.2f}')

    # rho by family: Clifford-only families sit at 2 log10 3 = 0.954 by
    # construction, so a per-suite mean hides where the objectives decouple.
    fam = collections.defaultdict(list)
    for tag, rows in per.items():
        r = rows.get('PRISM') or next(iter(rows.values()))
        if r['E']:
            fam[(meta[tag].get('suite', '?'),
                 meta[tag].get('family', '?'))].append(r['log10_shots'] / r['E'])
    print('\n=== rho = log10(gamma^2)/E by family (PRISM), lowest first ===')
    print(f'{"suite":12s} {"family":26s} {"n":>4s} {"rho":>8s}')
    for (su, fa), v in sorted(fam.items(), key=lambda kv: statistics.mean(kv[1])):
        flag = '  <- Clifford-only (rho = 2 log10 3)' \
            if abs(statistics.mean(v) - 0.9542425094) < 1e-6 else ''
        print(f'{su:12s} {fa:26s} {len(v):4d} {statistics.mean(v):8.4f}{flag}')

    with open(os.path.join(a.results, 'summary.json'), 'w') as f:
        json.dump(out_rows, f, indent=1)
    print(f'\nwrote {os.path.join(a.results, "summary.json")}')
    if a.tex:
        _write_tex(out_rows, a.tex)
        print(f'\nwrote {a.tex}')


def _esc(s):
    return s.replace('_', r'\_')


def _write_tex(out_rows, path):
    L = [r'% generated by exp05_suites.py report',
         r'\begin{table}[t]', r'  \centering', r'  \small',
         r'  \caption{\textbf{Partitioning cost on the standard benchmark '
         r'suites.} $\rho=\log_{10}\gamma^2/E$. Ranks are mean Friedman ranks '
         r'on $E$ over the instances on which every method completed.}',
         r'  \label{tab:suites}',
         r'  \begin{tabular}{llrrrrr}', r'    \toprule',
         r'    Suite & Method & $E$ & $\log_{10}\gamma^2$ & $\rho$ & '
         r'Rank & t [s] \\', r'    \midrule']
    for suite in sorted(out_rows):
        if suite == 'ALL':
            continue
        for i, r in enumerate(out_rows[suite]):
            head = _esc(suite) if i == 0 else ''
            L.append(f'    {head} & {_esc(r["method"])} & '
                     f'{r["E_mean"]:.1f} $\\pm$ {r["E_sd"]:.1f} & '
                     f'{r["K_mean"]:.1f} $\\pm$ {r["K_sd"]:.1f} & '
                     f'{r["rho_mean"]:.3f} & {r["rank"]:.2f} & '
                     f'{r["sec_mean"]:.2f} \\\\')
        L.append(r'    \midrule')
    if L[-1] == r'    \midrule':          # no rule immediately before bottomrule
        L.pop()
    L += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


# ---------------------------------------------------------------------------

def main():
    P = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = P.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('ingest')
    p.add_argument('--suites', default='qasmbench,mqtbench,supermarq')
    p.add_argument('--qasmbench', default=os.path.join(HERE, 'benchmarks', 'QASMBench'))
    p.add_argument('--out', default=os.path.join(HERE, 'benchmarks', 'instances'))
    p.add_argument('--sizes', default='8,12,16,20,24,32,40,48,64',
                   help='register sizes requested from the generated suites')
    p.add_argument('--nmin', type=int, default=6)
    p.add_argument('--nmax', type=int, default=64)
    p.add_argument('--maxgates', type=int, default=60000)
    p.add_argument('--maxdepth', type=int, default=4000,
                   help='skip circuits deeper than this; 0 disables. Checked '
                        'both before the rebase (cheap) and on the realised '
                        'ASAP layout (authoritative).')
    p.add_argument('--max-depth-ratio', type=float, default=200.0,
                   dest='max_depth_ratio',
                   help='skip circuits whose depth exceeds this multiple of '
                        'the register size; 0 disables. Catches the shallow-'
                        'register / enormous-depth shapes that dominate '
                        'sweep time without adding partitioning structure.')
    p.add_argument('--include-transpiled', action='store_true')
    p.add_argument('--no-hgr', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--verbose', action='store_true')
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser('sweep')
    p.add_argument('--instances', default=os.path.join(HERE, 'benchmarks', 'instances'))
    p.add_argument('--out', default=os.path.join(HERE, 'exp05_suites'))
    p.add_argument('--eps', type=float, default=0.10)
    p.add_argument('--prism-eps', type=float, default=1.0, dest='prism_eps')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--prism-B', type=int, default=150, dest='prism_B')
    p.add_argument('--n-pref', type=int, default=3, dest='n_pref')
    p.add_argument('--skip-slow', action='store_true')
    p.add_argument('--maxdepth', type=int, default=4000,
                   help='skip ingested instances deeper than this; 0 disables')
    p.add_argument('--max-depth-ratio', type=float, default=200.0,
                   dest='max_depth_ratio',
                   help='skip instances with depth > ratio * n; 0 disables')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--force', action='store_true')
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser('report')
    p.add_argument('--results', default=os.path.join(HERE, 'exp05_suites'))
    p.add_argument('--instances', default=os.path.join(HERE, 'benchmarks', 'instances'),
                   help='read suite labels from the ingested meta.json files')
    p.add_argument('--tex', default=os.path.join(HERE, 'exp05_suites.tex'))
    p.set_defaults(func=cmd_report)

    a = P.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
