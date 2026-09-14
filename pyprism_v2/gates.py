"""
pyprism_v2.gates
================
Gate library and the static per-gate invariants PRISM reasons about.

Two invariant families, matching the two mechanisms of the method.

**Symmetry** --- ``zsupp(g) = {j : [U_g, Z_j] = 0}``, the local qubit indices
whose computational basis the gate preserves.  A gate can sit inside a
distributable packet rooted at any ``v in zsupp(g)``.

**Irreducibility** --- ``chi`` is the operator Schmidt rank across a local cut;
where U(1)_N holds it is graded by charge transfer, ``chi = sum_delta
chi_delta``.  From it ``e = ceil(log2 chi)`` and the quasiprobability 1-norm
``gamma``.

Scaling note
------------
The structural invariants (arity, ``zsupp``, diagonality, U(1) covariance,
``chi`` and its grading) are *parameter independent* at generic parameters, so
they are computed once per gate **name** and cached.  Only ``gamma`` depends on
the angles, and it is evaluated from a closed form.  This is what makes a
100k-gate circuit tractable: a naive cache keyed on ``(name, params)`` misses
on every gate because the angles are continuous floats.

At degenerate parameters (``theta = 0``, where a controlled rotation becomes
the identity) the generic ``chi`` is retained.  That never under-charges a
cut, and ``gamma -> 1`` correctly reports the gate as free to knit.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ['GATE_LIB', 'GateSpec', 'spec', 'gamma_of_gate', 'unitary_of',
           'bipartitions', 'gamma_two_qubit', 'z_support', 'schmidt_rank',
           'charge_grading', 'eigenphase_diameter', 'PROBE']

# ---------------------------------------------------------------------------
#  single-qubit building blocks
# ---------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


def rot(axis, t):
    return math.cos(t / 2) * I2 - 1j * math.sin(t / 2) * axis


def phase_gate(lam):
    return np.diag([1.0, np.exp(1j * lam)]).astype(complex)


def u3(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
                    dtype=complex)


def ctrl(W, n_ctrl=1):
    d = W.shape[0]
    dim = (2 ** n_ctrl) * d
    M = np.eye(dim, dtype=complex)
    M[dim - d:, dim - d:] = W
    return M


SWAP4 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                 dtype=complex)
ISWAP4 = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
                  dtype=complex)

#: name -> (arity m_g, n_params, builder)
GATE_LIB: dict = {
    # m_g = 1
    'I':      (1, 0, lambda: I2),
    'X':      (1, 0, lambda: X),
    'Y':      (1, 0, lambda: Y),
    'Z':      (1, 0, lambda: Z),
    'H':      (1, 0, lambda: H),
    'S':      (1, 0, lambda: phase_gate(math.pi / 2)),
    'T':      (1, 0, lambda: phase_gate(math.pi / 4)),
    'SX':     (1, 0, lambda: rot(X, math.pi / 2) * np.exp(1j * math.pi / 4)),
    'RX':     (1, 1, lambda t: rot(X, t)),
    'RY':     (1, 1, lambda t: rot(Y, t)),
    'RZ':     (1, 1, lambda t: rot(Z, t)),
    'PHASE':  (1, 1, phase_gate),
    'U3':     (1, 3, u3),
    # m_g = 2
    'CX':     (2, 0, lambda: ctrl(X)),
    'CY':     (2, 0, lambda: ctrl(Y)),
    'CZ':     (2, 0, lambda: ctrl(Z)),
    'CH':     (2, 0, lambda: ctrl(H)),
    'CRX':    (2, 1, lambda t: ctrl(rot(X, t))),
    'CRY':    (2, 1, lambda t: ctrl(rot(Y, t))),
    'CRZ':    (2, 1, lambda t: ctrl(rot(Z, t))),
    'CPHASE': (2, 1, lambda t: ctrl(phase_gate(t))),
    'CU3':    (2, 3, lambda a, b, c: ctrl(u3(a, b, c))),
    'SWAP':   (2, 0, lambda: SWAP4),
    'ISWAP':  (2, 0, lambda: ISWAP4),
    # m_g >= 3
    'CCX':    (3, 0, lambda: ctrl(X, 2)),
    'CSWAP':  (3, 0, lambda: ctrl(SWAP4, 1)),
    'C3X':    (4, 0, lambda: ctrl(X, 3)),
    'C3Z':    (4, 0, lambda: ctrl(Z, 3)),
}

#: Generic parameter point at which the structural invariants are evaluated.
#:
#: The cached invariants are constant off a measure-zero variety, so the probe
#: only has to miss it --- but it should miss it by as wide a margin as
#: possible, for two independent reasons.
#:
#: *Conditioning.*  ``schmidt_rank`` counts singular values above a relative
#: tolerance and ``z_support`` tests a commutator against an absolute one, so
#: the probe should maximise the smallest quantity that must stay non-zero.
#: Maximising that margin over all 28 gates puts ``theta`` at ``pi``.
#:
#: *Genericity.*  But ``theta = pi`` is exactly where the parametrised gates
#: acquire finite order: ``CPHASE(pi)`` is ``CZ``, ``CRX(pi)`` is ``CX`` up to
#: phase.  The five invariants cached here are indifferent to that, but an
#: invariant added later need not be, so the probe should also stay clear of
#: low-order roots of unity.
#:
#: Maximising the margin subject to ``theta/pi``, ``varphi/pi`` and
#: ``lambda/pi`` being quadratic irrationals drawn from independent fields
#: gives the point below: 95.5% of the attainable margin, and no closer than
#: 5.9e-3 to any rational ``p/q`` with ``q <= 12``.  The previous choice,
#: ``(0.7, 1.1, 0.4)``, reached only 34.3% of the margin and had
#: ``theta/pi = 0.2228`` sitting 5.9e-4 from ``2/9`` --- sound, but a hair from
#: a ninth root of unity.  Both probes yield identical invariants for every
#: gate in the library; this one is simply better conditioned.
PROBE = (math.pi * (1.0 + math.sqrt(5.0)) / 4.0,   # pi * golden ratio / 2
         math.pi / math.sqrt(2.0),
         math.pi / math.sqrt(3.0))


def unitary_of(name, params=()):
    m, npar, builder = GATE_LIB[name]
    return builder(*params[:npar]) if npar else builder()


# ---------------------------------------------------------------------------
#  symmetry
# ---------------------------------------------------------------------------
def kron_all(ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def commutes(A, B, tol=1e-8):
    return np.allclose(A @ B - B @ A, 0, atol=tol)


def z_support(U, m):
    """``zsupp(g) = {j : [U_g, Z_j] = 0}`` --- the per-qubit refinement of
    on-site charge covariance, and the exact condition for packing."""
    out = []
    for j in range(m):
        if commutes(U, kron_all([Z if k == j else I2 for k in range(m)])):
            out.append(j)
    return tuple(out)


def is_diagonal(U):
    return np.allclose(U - np.diag(np.diag(U)), 0, atol=1e-8)


def number_op(m):
    tot = np.zeros((2 ** m, 2 ** m), dtype=complex)
    for q in range(m):
        tot += kron_all([(I2 - Z) / 2 if k == q else I2 for k in range(m)])
    return tot


def is_u1(U, m):
    return commutes(U, number_op(m))


# ---------------------------------------------------------------------------
#  irreducibility
# ---------------------------------------------------------------------------
def realign(U, A, m):
    B = tuple(q for q in range(m) if q not in A)
    a, b = len(A), len(B)
    T = U.reshape([2] * (2 * m))
    perm = list(A) + [m + q for q in A] + list(B) + [m + q for q in B]
    return np.transpose(T, perm).reshape(4 ** a, 4 ** b)


def schmidt_rank(U, A, m, tol=1e-9):
    s = np.linalg.svd(realign(U, A, m), compute_uv=False)
    return int(np.sum(s > tol * max(1.0, s[0])))


def charge_grading(U, A, m):
    """``{delta: chi_delta}`` for a U(1)-covariant U across the cut A|B."""
    dim, qa = 2 ** m, list(A)
    ch = np.array([sum((i >> (m - 1 - q)) & 1 for q in qa) for i in range(dim)])
    out = {}
    for delta in range(-len(A), len(A) + 1):
        Ud = np.where((ch[:, None] - ch[None, :]) == delta, U, 0)
        if np.allclose(Ud, 0, atol=1e-10):
            continue
        out[delta] = schmidt_rank(Ud, A, m)
    return out


def bipartitions(m):
    seen, out = set(), []
    for r in range(1, m):
        for A in itertools.combinations(range(m), r):
            B = tuple(q for q in range(m) if q not in A)
            key = frozenset([A, B])
            if key not in seen:
                seen.add(key)
                out.append((A, B))
    return out


# ---------------------------------------------------------------------------
#  gamma  (quasiprobability 1-norm)
# ---------------------------------------------------------------------------
MAGIC = np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
                 dtype=complex) / math.sqrt(2)
_KLEIN = np.array([[1, 1, -1, 1], [1, -1, 1, 1], [1, -1, -1, -1], [1, 1, 1, -1]],
                  dtype=complex)


def gamma_two_qubit(U):
    """``gamma = 2 (sum_i |u_i|)^2 - 1``, the optimal LO = LOCC quasiprobability
    1-norm (Schmitt, Piveteau & Sutter, Quantum 9, 1634 (2025))."""
    Ut = U / (np.linalg.det(U) ** 0.25)
    Um = MAGIC.conj().T @ Ut @ MAGIC
    lam = np.linalg.eigvals(Um.T @ Um)
    d0 = np.sqrt(lam.astype(complex))
    best = None
    for eps in itertools.product([1, -1], repeat=4):
        d = d0 * np.array(eps, dtype=complex)
        if abs(np.prod(d) - 1.0) > 1e-6:
            continue
        for perm in itertools.permutations(range(4)):
            u = np.linalg.solve(_KLEIN, d[list(perm)])
            tot = float(np.sum(np.abs(u)))
            if best is None or tot < best - 1e-12:
                best = tot
    return float(2.0 * best ** 2 - 1.0) if best else 1.0


def eigenphase_diameter(W):
    """Smallest arc of the unit circle containing ``spec(W)``."""
    ang = np.sort(np.mod(np.angle(np.linalg.eigvals(W)), 2 * math.pi))
    if len(ang) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * math.pi]]))
    return float(2 * math.pi - gaps.max())


def _controlled_branches(U, A, m):
    a, b = len(A), m - len(A)
    Bq = tuple(q for q in range(m) if q not in A)
    perm = list(A) + list(Bq)
    T = np.transpose(U.reshape([2] * (2 * m)),
                     perm + [m + p for p in perm]).reshape(2 ** a, 2 ** b,
                                                           2 ** a, 2 ** b)
    Ws = []
    for i in range(2 ** a):
        for j in range(2 ** a):
            blk = T[i, :, j, :]
            if np.allclose(blk, 0, atol=1e-9):
                continue
            if i != j:
                return None
            Ws.append(blk)
    return Ws if all(np.allclose(W @ W.conj().T, np.eye(2 ** b), atol=1e-7)
                     for W in Ws) else None


def gamma_numeric(U, A, m):
    """gamma for cutting U across A|B, with a provenance tag.  This is the
    reference implementation; :func:`gamma_of_gate` uses closed forms that are
    checked against it in the test suite."""
    if m == 1:
        return 1.0, 'trivial'
    if m == 2:
        return gamma_two_qubit(U), 'optimal'
    Bq = tuple(q for q in range(m) if q not in A)
    for side in (A, Bq):
        Ws = _controlled_branches(U, side, m)
        if Ws is not None and len(Ws) >= 2:
            g = 1.0
            for i in range(len(Ws)):
                for j in range(i + 1, len(Ws)):
                    W = Ws[i].conj().T @ Ws[j]
                    g = max(g, 1.0 + 2.0 * abs(math.sin(
                        eigenphase_diameter(W) / 2)))
            return g, 'upper bound'
    b = min(len(A), m - len(A))
    return float(4 ** (2 * b)), 'wire-cut bound'      # teleport across and back


def _g_rot(t):
    return 1.0 + 2.0 * abs(math.sin(t / 2.0))


def _g_cu3(a, b, c):
    return 1.0 + 2.0 * math.sqrt(max(0.0, 1.0 - math.cos((b + c) / 2.0) ** 2
                                     * math.cos(a / 2.0) ** 2))


#: closed-form gamma per gate name.  A float is parameter independent; a
#: callable takes the gate parameters.  Verified against `gamma_numeric`.
GAMMA_CLOSED: dict = {
    'CX': 3.0, 'CY': 3.0, 'CZ': 3.0, 'CH': 3.0,
    'SWAP': 7.0, 'ISWAP': 7.0,
    'CRX': _g_rot, 'CRY': _g_rot, 'CRZ': _g_rot, 'CPHASE': _g_rot,
    'CU3': _g_cu3,
    'CCX': 3.0, 'C3X': 3.0, 'C3Z': 3.0,
}


# ---------------------------------------------------------------------------
#  the cached structural record
# ---------------------------------------------------------------------------
@dataclass
class GateSpec:
    """Structural invariants of a gate *type*, independent of its angles."""
    name: str
    m: int
    n_params: int
    zsupp: tuple
    diagonal: bool
    u1: bool
    chi: dict = field(default_factory=dict)        # frozenset(A_local) -> chi
    grading: dict = field(default_factory=dict)    # frozenset(A_local) -> {d: chi_d}
    gamma_cut: dict = field(default_factory=dict)  # frozenset(A_local) -> float|None
    chi_worst: int = 1
    gamma_worst: float = 1.0
    ebits_worst: int = 0

    def ebits(self, A_local):
        c = self.chi[frozenset(A_local)]
        return int(math.ceil(math.log2(c))) if c > 1 else 0


_SPEC_CACHE: dict = {}


def spec(name) -> GateSpec:
    """Structural invariants of a gate type, computed once and cached."""
    if name in _SPEC_CACHE:
        return _SPEC_CACHE[name]
    m, npar, _ = GATE_LIB[name]
    U = unitary_of(name, PROBE[:npar])
    s = GateSpec(name=name, m=m, n_params=npar, zsupp=z_support(U, m),
                 diagonal=is_diagonal(U), u1=is_u1(U, m))
    if m >= 2:
        for A, B in bipartitions(m):
            k, kb = frozenset(A), frozenset(B)
            s.chi[k] = s.chi[kb] = schmidt_rank(U, A, m)
            g, _tag = gamma_numeric(U, A, m)
            s.gamma_cut[k] = s.gamma_cut[kb] = g
            if s.u1:
                s.grading[k] = charge_grading(U, A, m)
        s.chi_worst = max(s.chi.values())
        s.gamma_worst = max(s.gamma_cut.values())
        s.ebits_worst = int(math.ceil(math.log2(s.chi_worst)))
    _SPEC_CACHE[name] = s
    return s


def gamma_of_gate(name, params=(), A_local=None):
    """gamma for one gate at its actual angles.

    Uses the closed form where the gate has one, so no linear algebra runs per
    gate instance.  ``A_local`` selects the local cut; ``None`` means the worst
    case over the bipartitions of ``q_g``.
    """
    s = spec(name)
    if s.m < 2:
        return 1.0
    closed = GAMMA_CLOSED.get(name)
    if closed is not None and (A_local is None or s.m == 2
                               or len(set(s.gamma_cut.values())) == 1):
        return float(closed(*params[:s.n_params]) if callable(closed)
                     else closed)
    if A_local is None:
        return s.gamma_worst
    return float(s.gamma_cut[frozenset(A_local)])


def diag_phases(name, params=()):
    """Diagonal phases of a diagonal gate, for the packet joint-gamma."""
    return np.angle(np.diag(unitary_of(name, params)))
