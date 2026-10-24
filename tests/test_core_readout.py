"""
Tests for the phase-readout (Rayleigh) solver.

The important ones are the STRUCTURAL claims -- that w really is recoverable
from two O(1) Hermitian expectation values, that the energy inner product is a
similarity transform (so it cannot move a mode), and that the readout signal is
orders of magnitude larger than the residual's.  Those are what the method
rests on, and none of them need a noisy simulation to check.

The slow shot-noise measurement lives in run_rayleigh_demo.py.
"""

import numpy as np
import pytest

from qnmlab._core.pencil import equilibrate, leaver_matrix_std, residual, sigma_min
from qnmlab._core.quantum import ansatz_state
from qnmlab._core.rayleigh import (calibrate, clenshaw_curtis_weights, eigenvalue_condition,
                          energy_gram, energy_normed_operator, observables,
                          omega_from, qnm_filter, residual_shot_cost,
                          variance_cost)

WQ = 0.483643872211 - 0.096758775978j


def _exact(A):
    w = 1j * np.linalg.eigvals(A)
    return w[np.argmin(np.abs(w - WQ))]


# ----------------------------------------------------------------------
# the energy inner product
# ----------------------------------------------------------------------
def test_quadrature_weights_sum_to_interval():
    for n in (7, 15, 31):
        assert clenshaw_curtis_weights(n).sum() == pytest.approx(1.0, abs=1e-12)


def test_energy_gram_is_positive_definite():
    for n in (7, 15):
        assert np.linalg.eigvalsh(energy_gram(n, 2)).min() > 0


def test_energy_norm_is_a_similarity_so_it_cannot_move_a_mode():
    """The whole point: better conditioning, identical physics."""
    from qnmlab._core.hyperboloidal import operator
    A, _ = operator(7, 2)
    Ahat = energy_normed_operator(7, 2)
    assert np.allclose(np.sort_complex(np.linalg.eigvals(A)),
                       np.sort_complex(np.linalg.eigvals(Ahat)), atol=1e-8)


def test_energy_norm_improves_conditioning_and_gain_grows():
    from qnmlab._core.hyperboloidal import operator
    gains = []
    for n in (7, 15, 31):
        A, _ = operator(n, 2)
        gains.append(eigenvalue_condition(A) / eigenvalue_condition(
            energy_normed_operator(n, 2)))
    assert all(g > 1.0 for g in gains), "energy norm must not make it worse"
    assert gains[-1] > gains[0], "the gain must grow with resolution"


# ----------------------------------------------------------------------
# the readout identity -- the core of the method
# ----------------------------------------------------------------------
def test_omega_is_recoverable_from_two_hermitian_expectations():
    """Re w = -<H2>, Im w = <H1>, on the exact eigenvector."""
    A = energy_normed_operator(7, 2)
    ev, V = np.linalg.eig(A)
    k = int(np.argmin(np.abs(1j * ev - WQ)))
    z = V[:, k] / np.linalg.norm(V[:, k])
    H1, H2 = observables(A)
    for H in (H1, H2):
        assert np.allclose(H, H.conj().T), "measured operators must be Hermitian"
    e1 = float(np.real(np.vdot(z, H1 @ z)))
    e2 = float(np.real(np.vdot(z, H2 @ z)))
    assert abs(omega_from(e1, e2) - 1j * ev[k]) < 1e-10


def test_the_measured_values_are_order_one_not_order_1e14():
    """This is the entire fix: what the device reads is O(1)."""
    A = energy_normed_operator(7, 2)
    ev, V = np.linalg.eig(A)
    k = int(np.argmin(np.abs(1j * ev - WQ)))
    z = V[:, k] / np.linalg.norm(V[:, k])
    H1, H2 = observables(A)
    vals = [abs(float(np.real(np.vdot(z, H @ z)))) for H in (H1, H2)]
    assert min(vals) > 1e-2, f"readout must be O(1), got {vals}"
    # the residual encoding's target, for contrast
    P = equilibrate(leaver_matrix_std(WQ, l=2, N=8))
    assert sigma_min(P) ** 2 < 1e-6
    assert min(vals) / sigma_min(P) ** 2 > 1e5


def test_variance_vanishes_on_an_eigenvector():
    A = energy_normed_operator(7, 2)
    ev, V = np.linalg.eig(A)
    k = int(np.argmin(np.abs(1j * ev - WQ)))
    z = V[:, k] / np.linalg.norm(V[:, k])
    AdA = A.conj().T @ A
    var = float(np.real(np.vdot(z, AdA @ z))) - abs(complex(np.vdot(z, A @ z))) ** 2
    # Was `< 1e-16`, which is BELOW machine epsilon (2.22e-16): it passed under
    # numpy 1.26 by luck of rounding and failed under numpy 2.5 at exactly one
    # unit of roundoff.  1e-12 is safely above roundoff and still ~10 orders of
    # magnitude below the variance of any non-eigenvector state.
    assert abs(var) < 1e-12


# ----------------------------------------------------------------------
# mode selection -- the relocated problem
# ----------------------------------------------------------------------
def test_steering_beats_cold_start():
    """
    Cold multistart converges to a spurious minimum that is not an eigenvalue;
    steering from a basin-level lead recovers the fundamental's basin.
    """
    A = energy_normed_operator(7, 2)
    true = _exact(A)
    cold = calibrate(A, layers=3, restarts=6, seed=0, w_filter=qnm_filter)
    warm = calibrate(A, layers=3, restarts=4, seed=2, w_filter=qnm_filter,
                     w_lead=0.48 - 0.10j, mu=1.0)
    assert abs(warm["w"] - true) < abs(cold["w"] - true) / 10


def test_qnm_filter_rejects_the_mirror_branch():
    assert qnm_filter(0.4836 - 0.0968j)
    assert not qnm_filter(-0.4836 - 0.0968j)     # mirror branch
    assert not qnm_filter(0.4836 + 0.0968j)      # growing, unphysical


# ----------------------------------------------------------------------
# the shot-cost conversion
# ----------------------------------------------------------------------
def test_residual_cost_uses_the_quadratic_not_the_derivative():
    """
    sigma_min has a V-shaped minimum, so sigma^2 is quadratic in the offset.
    A derivative-based estimate understates the residual arm badly; guard
    against silently reintroducing it.
    """
    tol = 1e-3
    shots, s0, resolvable = residual_shot_cost(tol, std_at_ref=1.579e-2)
    naive = 2.0 * s0 * 2.694 * tol            # the wrong, linearized version
    assert resolvable > 5 * naive, "quadratic term must dominate near the mode"
    assert shots > 1e12
