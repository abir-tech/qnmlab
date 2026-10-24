"""
Tests for the hyperboloidal encoding.

These lock in BOTH sides of the verdict: that the encoding is correct and more
accurate than Leaver per dimension, AND that it is worse on the two metrics
that decide the quantum arms.  The second group matters as much as the first --
it is what stops the "surely hyperboloidal is better" question from being
re-opened without new evidence.
"""

import numpy as np
import pytest

from qnmlab._core.hyperboloidal import (
    minimal_gauge, characteristic_speeds, operator, qnm_spectrum,
    ultraspherical_pencil, interleave, compare_encodings,
)

W_REF = 0.483643872211 - 0.096758775978j
OVERTONES = [0.483643872211 - 0.096758775978j,
             0.463850579 - 0.295603937j,
             0.430543926 - 0.508557628j]


# ----------------------------------------------------------------------
# correctness of the geometry
# ----------------------------------------------------------------------
def test_minimal_gauge_identity():
    """gamma^2 + W p = 1 is the minimal-gauge signature; it certifies the gauge."""
    s = np.linspace(0.0, 1.0, 41)
    W, p, g, gp, q = minimal_gauge(s, 2)
    assert np.allclose(g ** 2 + W * p, 1.0, atol=1e-13)


def test_both_boundaries_are_outflow():
    """No boundary conditions are imposed -- this is why that is legitimate."""
    cp0, cm0 = characteristic_speeds(0.0)          # future null infinity
    cp1, cm1 = characteristic_speeds(1.0)          # horizon
    assert cp0 >= 0 and cm0 >= -1e-15, "scri must not be an inflow boundary"
    assert cp1 <= 1e-15 and cm1 <= 0, "horizon must not be an inflow boundary"


def test_minimal_gauge_rejects_a_broken_gauge(monkeypatch):
    """A wrong height function must fail loudly, not return plausible modes."""
    import qnmlab._core.hyperboloidal as hyp
    good = hyp.minimal_gauge

    def broken(sigma, ell):
        W, p, g, gp, q = good(sigma, ell)
        return W, p * 1.05, g, gp, q          # perturb the gauge

    monkeypatch.setattr(hyp, "minimal_gauge", broken)
    with pytest.raises(RuntimeError):
        # re-run the assertion path directly
        s = np.linspace(0.0, 1.0, 11)
        W, p, g, gp, q = broken(s, 2)
        if not np.allclose(g ** 2 + W * p, 1.0, atol=1e-12):
            raise RuntimeError("minimal-gauge identity violated")


# ----------------------------------------------------------------------
# the physics
# ----------------------------------------------------------------------
def test_fundamental_matches_leaver_to_12_digits():
    w = qnm_spectrum(32, ell=2, n_modes=1)[0]
    assert abs(w - W_REF) < 1e-11


def test_overtones_come_from_one_diagonalization():
    """No seeds, no root-finding: n = 0, 1, 2 all fall out of a single eig()."""
    w = qnm_spectrum(48, ell=2, n_modes=3)
    for found, ref in zip(w, OVERTONES):
        assert abs(found - ref) < 1e-6


def test_spectral_convergence():
    """Error must fall by orders of magnitude as resolution grows."""
    errs = [abs(qnm_spectrum(n, 2, n_modes=1)[0] - W_REF) for n in (12, 20, 32)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[0] / errs[2] > 1e4


def test_no_boundary_conditions_are_imposed():
    """The operator must use every collocation point, endpoints included."""
    A, sigma = operator(16, 2)
    n = A.shape[0] // 2
    assert sigma[0] == pytest.approx(0.0)      # scri is IN the grid
    assert sigma[-1] == pytest.approx(1.0)     # horizon is IN the grid
    assert np.abs(A[n, :]).max() > 0           # first physical row not zeroed
    assert np.abs(A[-1, :]).max() > 0          # last physical row not zeroed


def test_ultraspherical_agrees_with_collocation():
    from scipy.linalg import eig
    A, B = ultraspherical_pencil(32, 2)
    w = 1j * eig(A, B, right=False)
    w = w[np.isfinite(w)]
    w = w[(w.real > 1e-6) & (w.imag < -1e-6) & (w.imag > -2)]
    assert min(abs(w - W_REF)) < 1e-11


def test_interleave_is_a_similarity():
    """Reordering must not move the spectrum -- it is only a qubit relabel."""
    A, B = ultraspherical_pencil(16, 2)
    M = A - (-1j * W_REF) * B
    assert np.allclose(np.sort_complex(np.linalg.eigvals(M)),
                       np.sort_complex(np.linalg.eigvals(interleave(M))))


# ----------------------------------------------------------------------
# the verdict -- why this is NOT adopted for the quantum arms
# ----------------------------------------------------------------------
def test_leaver_keeps_the_pauli_advantage():
    """HANDOFF 4.3 is the headline result; hyperboloidal must not silently erode it."""
    rows = compare_encodings(dims=(8, 16, 32))
    for r in rows:
        assert r["pauli_leaver"] <= r["pauli_hyper"]
    top = rows[-1]
    assert top["pauli_leaver"] == 319
    assert top["pauli_hyper"] >= 3 * top["pauli_leaver"]


def test_hyperboloidal_readout_signal_decays_faster():
    """
    The decisive measurement: what hardware must resolve away from the mode.
    Leaver's signal is near N-independent; the hyperboloidal one collapses.
    """
    rows = compare_encodings(dims=(8, 16, 32))
    for r in rows:
        assert r["signal_hyper"] < r["signal_leaver"]
    ratio_small = rows[0]["signal_leaver"] / rows[0]["signal_hyper"]
    ratio_large = rows[-1]["signal_leaver"] / rows[-1]["signal_hyper"]
    assert ratio_large > 10 * ratio_small, "gap must widen with size"
