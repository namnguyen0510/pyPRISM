#!/usr/bin/env python3
"""Self-contained checks for pyprism_v2.  Run: python3 tests.py"""
from __future__ import annotations
import math, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyprism_v2 as pp
from pyprism_v2.gates import (gamma_two_qubit, unitary_of, gamma_numeric,
                              spec, gamma_of_gate, bipartitions, GATE_LIB,
                              schmidt_rank, charge_grading, PROBE)
from pyprism_v2.objectives import CostState, mode_front, net_costs, evaluate

FAILS = []


def check(name, ok, detail=''):
    print(f'  {"PASS" if ok else "FAIL"}  {name}' + (f'   {detail}' if detail else ''))
    if not ok:
        FAILS.append(name)


def main():
    print('gate invariants')
    ref = {'CX': 3.0, 'CY': 3.0, 'CZ': 3.0, 'CH': 3.0, 'SWAP': 7.0, 'ISWAP': 7.0}
    bad = {k: gamma_two_qubit(unitary_of(k)) for k in ref
           if abs(gamma_two_qubit(unitary_of(k)) - ref[k]) > 1e-6}
    check('gamma matches the published values', not bad, str(bad) or
          'CX=CY=CZ=CH=3, SWAP=iSWAP=7')

    from pyprism_v2.gates import ctrl, phase_gate
    check('gamma(CS) = 1+sqrt(2)',
          abs(gamma_two_qubit(ctrl(phase_gate(math.pi / 2))) - (1 + 2 ** .5)) < 1e-6)

    bad = []
    for nm, (m, npar, _) in GATE_LIB.items():
        if m < 2:
            continue
        for ps in ((0.3,) * npar, (1.7,) * npar, PROBE[:npar], (2.9,) * npar):
            U = unitary_of(nm, ps)
            want = max(gamma_numeric(U, A, m)[0] for A, _ in bipartitions(m))
            got = gamma_of_gate(nm, ps, None)
            if abs(want - got) > 1e-6:
                bad.append(f'{nm}{ps}: closed {got:.5f} != numeric {want:.5f}')
    check('closed-form gamma agrees with the numeric reference', not bad,
          '; '.join(bad[:3]))

    bad = []
    for nm, (m, npar, _) in GATE_LIB.items():
        if m < 2:
            continue
        s = spec(nm)
        if not s.u1:
            continue
        U = unitary_of(nm, PROBE[:npar])
        for A, _ in bipartitions(m):
            if sum(charge_grading(U, A, m).values()) != schmidt_rank(U, A, m):
                bad.append(f'{nm} {A}')
    check('charge grading is exact: chi = sum_delta chi_delta', not bad,
          str(bad[:3]))

    chis = {spec(nm).chi_worst for nm, (m, _, _) in GATE_LIB.items() if m == 2}
    check('two-qubit chi in {1,2,4}', chis <= {1, 2, 4}, str(sorted(chis)))

    zs = {'CZ': (0, 1), 'CX': (0,), 'SWAP': (), 'CCX': (0, 1), 'ISWAP': (),
          'H': (), 'RZ': (0,), 'CSWAP': (0,), 'C3Z': (0, 1, 2, 3)}
    bad = {k: spec(k).zsupp for k, v in zs.items() if spec(k).zsupp != v}
    check('zsupp matches the hand-derived values', not bad, str(bad))

    print('\npipeline')
    n, D = 24, 48
    lay, g, adj = pp.random_circuit(n, D, seed=3)
    hg1, hg2 = pp.build_track1(n, g), pp.build_track2(n, g)
    packets, covered = pp.extract_packets(n, g)
    check('Track 2 has fewer nets than Track 1',
          hg2.n_nets < hg1.n_nets,
          f'{hg1.n_nets} -> {hg2.n_nets} ({hg1.n_nets/hg2.n_nets:.2f}x)')

    allg = [i for _, gg in packets for i in gg]
    check('packets are disjoint', len(allg) == len(set(allg)))
    by_id = {x.gid: x for x in g}
    bad = sum(1 for root, gids in packets for gid in gids
              if by_id[gid].qubits.index(root) not in spec(by_id[gid].name).zsupp)
    check('every packed gate has its root in zsupp', bad == 0, f'{bad} bad')

    rng = np.random.default_rng(0)
    worse = 0
    for _ in range(80):
        o = rng.permutation(n)
        m = sum(1 << int(q) for q in o[:n // 2])
        if pp.evaluate(hg2, m)[0] > pp.evaluate(hg1, m, exact=False)[0]:
            worse += 1
    check('Track 2 ebits <= Track 1 ebits on the same cut', worse == 0,
          f'{worse}/80')

    print('\nincremental evaluator')
    m0 = sum(1 << q for q in range(n // 2))
    st = CostState(hg2, m0)
    bad = 0
    for _ in range(40):
        v = int(rng.integers(n))
        st.flip(v)
        E, K = pp.evaluate(hg2, st.mask)
        if abs(E - st.E) > 1e-9 or abs(K - st.K) > 1e-9:
            bad += 1
    check('CostState.flip agrees with a full rescan', bad == 0, f'{bad}/40')

    st = CostState(hg2, m0).set_pref(0.3, 0.7)
    S_ref = sum(min(0.3 * c[0], 0.7 * c[1]) for c in net_costs(hg2, m0))
    check('mode-aware scalarisation decomposes per net',
          abs(st.S - S_ref) < 1e-9, f'{st.S:.6f} vs {S_ref:.6f}')

    E, K = pp.evaluate(hg2, m0)
    mf = mode_front(net_costs(hg2, m0))
    check('mode-front endpoints reproduce the currencies',
          abs(mf[0][1] - K) < 1e-9 and abs(mf[-1][0] - E) < 1e-9,
          f'({mf[-1][0]:.0f}, {mf[0][1]:.3f}) vs ({E}, {K:.3f})')
    check('mode front is monotone',
          all(mf[i][0] <= mf[i+1][0] and mf[i][1] >= mf[i+1][1]
              for i in range(len(mf) - 1)))

    print('\npartitioners')
    lc, _, _ = pp.light_cone_graph(n, lay)
    ok = 0
    for meth in pp.available(n):
        try:
            kw = dict(seed=0, eps=0.10, lc=lc)
            if meth.kind == 'multi':
                kw.pop('lc')
                kw.update(pop_size=8, generations=3, points=3, restarts=1)
                r = meth.fn(n, hg2, **kw)
                assert r and all(len(p) == 3 for p in r)
            else:
                mk = meth.fn(n, hg2, **kw)
                assert pp.feasible(mk, n, 0.10 + 1e-9), f'{meth.name} unbalanced'
            ok += 1
        except Exception as e:
            check(f'{meth.name} runs', False, f'{type(e).__name__}: {e}')
    check(f'all {ok} available methods run and respect balance', True)

    fr, dg = pp.part_prism(n, hg2, pp.PrismConfig(sweeps=80, seed=0, workers=1),
                           lc=lc)
    check('PRISM returns a front', len(fr) > 1, f'|F|={len(fr)}')
    check('PRISM exchanges on both ladder axes',
          dg['swap_accept_T'] > 0 and dg['swap_accept_w'] > 0,
          f"T={dg['swap_accept_T']:.2f} w={dg['swap_accept_w']:.2f}")
    check('PRISM front is non-dominated',
          all(fr[i][1] > fr[i+1][1] for i in range(len(fr) - 1)))

    # --- the temperature ladder must stay tethered to the objective --------
    # These four exist because the ladder was once fixed at an absolute
    # T in [0.03, 1.2] while S is normalised to a scale that depends on the
    # instance and on w.  Every rung then ran at effectively infinite
    # temperature: swap acceptance sat near 0.85, the search was an unbiased
    # random walk, and on a 64-qubit instance it cost a factor of thirteen in
    # ebits.  Nothing in the old suite could see it, because the ladder still
    # returned a valid, non-dominated front -- it was simply a bad one.
    check('replica exchange accepts in a sane band, not always',
          0.02 < dg['swap_accept_T'] < 0.75 and 0.02 < dg['swap_accept_w'] < 0.75,
          f"T={dg['swap_accept_T']:.2f} w={dg['swap_accept_w']:.2f}"
          ' (near 1.0 means the ladder is far too hot)')
    check('the ladder improves on the states it starts from',
          dg['search_improved'],
          f"initial E={dg['best_E_initial']:.0f} -> "
          f"search E={dg['best_E_search']:.0f}")
    ts, ds = dg['temps'], dg['median_dS']
    spread = max(ds) / max(min(ds), 1e-12)
    t_spread = max(t[0] for t in ts) / max(min(t[0] for t in ts), 1e-12)
    check('temperatures track the per-column objective scale',
          dg['auto_temp'] and (spread < 1.5 or t_spread > 1.5),
          f'median|dS| spread {spread:.1f}x across columns, '
          f'cold-rung T spread {t_spread:.1f}x')
    cold = [t[0] for t in ts]
    check('the coldest rung is cold relative to a typical move',
          all(c < d for c, d in zip(cold, ds)),
          'T_cold < median|dS| for every column')

    print('\npersistence')
    import tempfile, re, json
    from pyprism_v2.io import (save_instance, load_instance, circuit_hash,
                               to_qasm, QELIB1_GATES, write_manifest,
                               environment)
    with tempfile.TemporaryDirectory() as td:
        meta = {'n_qubits': n, 'depth': D, 'seed': 3, 'family': 'rqc'}
        e = save_instance(td, 'inst', lay, g, n, meta=meta)
        lay2, g2, n2, m2 = load_instance(td, 'inst')
        check('circuit round trip is bit-exact',
              n2 == n and m2 == meta and circuit_hash(g) == circuit_hash(g2)
              and all(a.name == b.name and a.qubits == b.qubits
                      and a.layer == b.layer
                      and all(x == y for x, y in zip(a.params, b.params))
                      for a, b in zip(g, g2)),
              f'{len(g2)} gates, sha {e["sha256_16"]}')
        h1b, h2b = pp.build_track1(n, g2), pp.build_track2(n, g2)
        check('reloaded circuit rebuilds the same hypergraphs',
              hg1.stats() == h1b.stats() and hg2.stats() == h2b.stats())
        check('reloaded circuit gives the same objectives',
              pp.evaluate(hg2, m0) == pp.evaluate(h2b, m0))

        # every emitted mnemonic must exist in qelib1.inc
        bad, ops = set(), 0
        txt = to_qasm(g, n)
        nq = int(re.search(r'qreg q\[(\d+)\];', txt).group(1))
        for line in txt.splitlines():
            line = line.strip()
            if not line or line.startswith(('//', 'OPENQASM', 'include',
                                            'qreg')):
                continue
            mn = re.match(r'([a-z0-9]+)', line).group(1)
            if mn not in QELIB1_GATES:
                bad.add(mn)
            if any(int(i) >= nq for i in re.findall(r'q\[(\d+)\]', line)):
                bad.add('qubit-out-of-range')
            ops += 1
        check('QASM uses only qelib1.inc gates, qubits in range', not bad,
              f'{ops} statements' + (f', bad: {sorted(bad)}' if bad else ''))

        # every gate type in the library must have a QASM mapping
        from pyprism_v2.circuits import Gate
        miss = []
        for nm, (m_, npar, _) in pp.GATE_LIB.items():
            try:
                to_qasm([Gate(0, nm, tuple(range(m_)),
                              (0.3,) * npar, 0)], m_)
            except Exception as exc:
                miss.append(f'{nm}: {exc}')
        check('every gate in the library has a QASM mapping', not miss,
              '; '.join(miss[:3]))

        mp = write_manifest(td, [e], config={'test': True})
        man = json.load(open(mp))
        check('manifest records the environment and instance hashes',
              man['environment']['pyprism_v2'] == pp.__version__
              and man['instances'][0]['sha256_16'] == circuit_hash(g))

    print()
    print('ALL CHECKS PASSED' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}')
    return 0 if not FAILS else 1


if __name__ == '__main__':
    sys.exit(main())
