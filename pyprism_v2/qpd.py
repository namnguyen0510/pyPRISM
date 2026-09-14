"""
pyprism_v2.qpd
==============
Quasiprobability decomposition of a severed **diagonal** two-qubit gate.

This is the reconstruction that circuit knitting actually performs, as opposed
to simulating each block with the crossing gates dropped.  It is exact: the
identity below reproduces the two-qubit channel ``rho -> U rho U^dagger``
term by term, and :func:`verify` checks that numerically rather than asserting
it.

The decomposition
-----------------
Write the non-local part of a controlled phase as a ``ZZ`` rotation.  With
``CPHASE(lam) = diag(1, 1, 1, e^{i lam})``,

    lam |11><11| = (lam/4)(I@I - Z@I - I@Z + Z@Z)

so ``CPHASE(lam)`` is a product of two **local** phases and
``exp(i lam/4 Z@Z)``, i.e. ``U = exp(-i theta/2 Z@Z)`` at ``theta = -lam/2``.
Only that ``ZZ`` factor has to be cut; the local phases go into whichever block
owns the qubit.

For ``U = exp(-i theta/2 A@B)`` with ``A = B = Z``, write ``c = cos(theta/2)``,
``s = sin(theta/2)``:

    U rho U^d = c^2 rho + s^2 (A@B) rho (A@B) - i c s [A@B, rho]

The first two terms are already local products.  The commutator is handled by
the identity

    [AB, rho] = (1/2)( [A, {B, rho}] + {A, [B, rho]} )

together with two single-qubit facts:

    {A, sigma}/2   = M_+^A(sigma) - M_-^A(sigma)      measure A, keep the sign
    -i [A, sigma]  = R_A(+pi/2)(sigma) - R_A(-pi/2)(sigma)

Both are differences of *local* CP maps.  Substituting gives six sampling
branches, listed in :data:`BRANCH_NAMES`.

Why the cost is ``1 + 2|sin theta|``
------------------------------------
The 1-norm of the coefficients is what inflates the shot budget.  Counting
naively gives eight signed terms and ``1 + 4|sin theta|``, but a measurement
branch is **one** circuit execution whose result is multiplied by the outcome
``m = +-1`` --- ``M_+ - M_-`` is not two runs.  Counting the two measurement
pairs once each gives

    gamma = c^2 + s^2 + 4|cs| = 1 + 2|sin theta|

which is the optimal value for a two-qubit rotation and agrees with
``gates.GAMMA_CLOSED``: at ``lam = pi`` (a CZ) ``theta = -pi/2`` and
``gamma = 3``.

What this costs in practice
---------------------------
``gamma`` multiplies **per severed net**, so a partition severing ``k`` of them
needs ``gamma^{2k}`` times the shots for fixed accuracy.  At ``k = 10`` and
``gamma = 3`` that is ``10^9``.  This module is therefore the correct
reconstruction, not a cheap one, and the honest way to use it at scale is to
report ``gamma^2`` as the cost rather than to pretend the sampling is
affordable.  :func:`exact_expectation` sums over all ``6^k`` branches instead
of sampling them, which is exact and feasible only for small ``k``.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

__all__ = ['BRANCH_NAMES', 'zz_branches', 'cphase_branches', 'gamma_zz',
           'verify']

I2 = np.eye(2, dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
P0 = np.array([[1, 0], [0, 0]], dtype=complex)
P1 = np.array([[0, 0], [0, 1]], dtype=complex)

#: what each branch does, in order
BRANCH_NAMES = ('II', 'ZZ', 'RZ+ @ measZ', 'RZ- @ measZ',
                'measZ @ RZ+', 'measZ @ RZ-')


def _rz(phi):
    """``exp(-i phi Z / 2)`` --- ``R_Z(phi)``, so ``R_A(+-pi/2)`` above."""
    return np.diag([np.exp(-0.5j * phi), np.exp(0.5j * phi)]).astype(complex)


def _kraus_unitary(u):
    """A CP map given by a single unitary/operator, as a Kraus list."""
    return [u]


def _kraus_meas_signed():
    """``M_+ - M_-``: measure Z, multiply the estimator by the outcome.

    Returned as ``[(+1, P0), (-1, P1)]`` --- a *signed* Kraus list, because the
    map is a difference of two CP maps rather than a channel.  One circuit run
    realises it: measure, read ``m``, weight by ``m``.
    """
    return [(1.0, P0), (-1.0, P1)]


def gamma_zz(theta):
    """``gamma = 1 + 2|sin theta|`` for cutting ``exp(-i theta/2 Z@Z)``."""
    return 1.0 + 2.0 * abs(math.sin(theta))


def zz_branches(theta):
    """The six branches of ``exp(-i theta/2 Z@Z)``.

    Returns ``[(coeff, kraus_u, kraus_v), ...]`` where each ``kraus_*`` is a
    list of ``(sign, op)`` acting on that side alone.  ``sum |coeff|`` is
    :func:`gamma_zz`.
    """
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    cs = c * s                                    # = sin(theta)/2
    U = [(1.0, I2)]
    Zk = [(1.0, Z2)]
    Rp = [(1.0, _rz(math.pi / 2))]
    Rm = [(1.0, _rz(-math.pi / 2))]
    M = _kraus_meas_signed()
    return [
        (c * c, U, U),
        (s * s, Zk, Zk),
        (cs, Rp, M),
        (-cs, Rm, M),
        (cs, M, Rp),
        (-cs, M, Rm),
    ]


def cphase_branches(lam):
    """Branches for a severed ``CPHASE(lam)``, plus the local phase factors.

    Returns ``(branches, local_u, local_v, phase)`` where ``local_*`` are the
    single-qubit diagonal unitaries that stay on their own side and ``phase``
    is the scalar.  Reassembling::

        CPHASE(lam) = phase * (local_u @ local_v) * exp(i lam/4 Z@Z)

    and only the last factor is cut, at ``theta = -lam/2``.
    """
    theta = -lam / 2.0
    local = np.diag([np.exp(-0.25j * lam), np.exp(0.25j * lam)]).astype(complex)
    return zz_branches(theta), local, local, np.exp(0.25j * lam)


# ---------------------------------------------------------------------------
#  verification
# ---------------------------------------------------------------------------
def _apply(kraus_u, kraus_v, rho):
    """Apply a signed local Kraus pair to a 2-qubit density matrix."""
    out = np.zeros_like(rho)
    for su, ku in kraus_u:
        for sv, kv in kraus_v:
            K = np.kron(ku, kv)
            out = out + su * sv * (K @ rho @ K.conj().T)
    return out


def verify(angles=(0.3, 1.0, math.pi / 2, math.pi, 2.4, -0.7), trials=3,
           tol=1e-10, seed=0):
    """Check the decomposition against the exact channel, and the 1-norm.

    For random mixed states ``rho`` the branch sum must reproduce
    ``U rho U^dagger`` to machine precision, and ``sum |coeff|`` must equal
    ``1 + 2|sin theta|``.  Returns ``(max_error, max_gamma_error)``.
    """
    rng = np.random.default_rng(seed)
    worst, worst_g = 0.0, 0.0
    for theta in angles:
        br = zz_branches(theta)
        g = sum(abs(cf) for cf, _u, _v in br)
        worst_g = max(worst_g, abs(g - gamma_zz(theta)))
        ZZ = np.kron(Z2, Z2)
        U = (math.cos(theta / 2) * np.eye(4)
             - 1j * math.sin(theta / 2) * ZZ)
        for _ in range(trials):
            M = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
            rho = M @ M.conj().T
            rho = rho / np.trace(rho)
            exact = U @ rho @ U.conj().T
            got = np.zeros_like(rho)
            for cf, ku, kv in br:
                got = got + cf * _apply(ku, kv, rho)
            worst = max(worst, float(np.abs(got - exact).max()))
    return worst, worst_g


def exact_expectation(block_eval, k):
    """Sum over all ``6^k`` branch combinations instead of sampling them.

    ``block_eval(choices)`` returns the observable's value for one assignment
    of branch indices to the ``k`` severed nets, already multiplied by the two
    local Kraus signs.  Exact, and tractable only while ``6^k`` is small ---
    which is the point: ``gamma^{2k}`` is not a modelling artefact, it is the
    real cost of reconstruction, and summing it exactly makes that visible
    rather than hiding it behind a sampler that would need the same budget.
    """
    if 6 ** k > 2_000_000:
        raise ValueError(f'{k} severed nets is 6^{k} = {6 ** k:,} branches; '
                         f'sample instead, or partition more cheaply')
    total = 0.0
    for choices in itertools.product(range(6), repeat=k):
        total += block_eval(choices)
    return total


if __name__ == '__main__':
    err, gerr = verify()
    print(f'max |branch sum - U rho U^dag| = {err:.3e}')
    print(f'max |sum|coeff| - (1+2|sin t|)| = {gerr:.3e}')
    print('OK' if err < 1e-10 and gerr < 1e-12 else 'FAILED')
