"""
Tests for the October 2026 extensions:
  * the phase-readout operator for any metric (Kazakov-Solodukhin),
  * general-commuting measurement grouping,
  * the adjoint-gradient VQE used inside the 2D search.
"""
import itertools

import numpy as np
import pytest

from qnmlab._core import hyperboloidal as H
from qnmlab._core import hyperboloidal_metric as HM
from qnmlab._core.grouping import diagonalising_circuit, gc_partition, to_sim_gates, unitary, z_image
from qnmlab._core.quantum import ansatz_state
from qnmlab._core.rayleigh import energy_normed_operator
from qnmlab._core.search2d import VQE, Pencil


# ---------------------------------------------------------------- readout for any metric
def test_general_operator_reproduces_schwarzschild_exactly():
    s, _ = H.cheb_lobatto(12)
    for old, new in zip(H.minimal_gauge(s, 2), HM.coefficients(s, "schwarzschild", 0.0, 2)):
        assert np.max(np.abs(old - new)) < 1e-12
    assert np.max(np.abs(energy_normed_operator(7, 2) - HM.energy_normed_operator(7, 2))) < 1e-10


@pytest.mark.parametrize("a", [0.2, 0.5, 1.0])
def test_kazakov_solodukhin_matches_published_fundamental(a):
    """Published values carry 6 decimals, so agreement to <1e-6 is the ceiling."""
    ref = HM.REFERENCE[("kazakov_solodukhin", a)]
    assert abs(HM.fundamental(31, 2, "kazakov_solodukhin", a, near=ref) - ref) < 1e-6


def test_ks_with_zero_deformation_is_schwarzschild():
    w = HM.fundamental(31, 2, "kazakov_solodukhin", 0.0)
    assert abs(w - HM.REFERENCE[("schwarzschild", 0.0)]) < 1e-10


def test_ks_energy_norm_is_a_similarity():
    A, _ = HM.operator(7, 2, "kazakov_solodukhin", 1.0)
    Ah = HM.energy_normed_operator(7, 2, "kazakov_solodukhin", 1.0)
    assert np.allclose(np.sort_complex(np.linalg.eigvals(A)),
                       np.sort_complex(np.linalg.eigvals(Ah)), atol=1e-8)


# ---------------------------------------------------------------- grouping
def test_gc_grouping_covers_all_paulis_in_nine_commuting_families():
    labels = ["".join(t) for t in itertools.product("IXYZ", repeat=3)][1:]
    fams = gc_partition(labels)
    assert len(fams) == 9
    assert sorted(l for f in fams for l in f) == sorted(labels)


def test_diagonalising_circuits_map_every_member_to_a_z_string():
    labels = ["".join(t) for t in itertools.product("IXYZ", repeat=3)][1:]
    for fam in gc_partition(labels):
        U = unitary(to_sim_gates(diagonalising_circuit(fam, 3)), 3)
        for lab in fam:
            sign, zl = z_image(lab, U, 3)
            assert abs(sign) == 1 and set(zl) <= {"I", "Z"}


def test_gc_plan_reads_the_exact_omega_without_noise():
    from qnmlab._core.hardware_run import calibrate_theta
    from qnmlab._core import mitigation as MI
    A = HM.energy_normed_operator(3, 2, "kazakov_solodukhin", 1.0)
    w = HM.fundamental(3, 2, "kazakov_solodukhin", 1.0)
    th, fid, _ = calibrate_theta(A, layers=3, n=3, target=w, restarts=4)
    z = ansatz_state(th, 3, 3)
    w_state = 1j * np.vdot(z, A @ z)
    for kind in ("gc", "qwc"):
        P = MI.plan(A, 3, kind)
        assert abs(MI.run(P, th, p2=0.0, readout=(0.0, 0.0)) - w_state) < 1e-9


# ---------------------------------------------------------------- VQE inside the search
def test_vqe_circuit_is_the_project_ansatz():
    v = VQE(4, 3)
    th = np.random.default_rng(0).uniform(0, 2 * np.pi, v.n_params)
    assert np.allclose(v.state(th), ansatz_state(th, 4, 3))


def test_adjoint_gradient_matches_finite_differences():
    v = VQE(3, 3)
    th = np.random.default_rng(1).uniform(0, 2 * np.pi, v.n_params)
    Hm = Pencil("schwarzschild", qubits=3).R(0.47 - 0.12j)
    _, g = v.energy_grad(th, Hm)
    fd = [(v.energy_grad(th + 1e-6 * e, Hm)[0] - v.energy_grad(th - 1e-6 * e, Hm)[0]) / 2e-6
          for e in np.eye(len(th))]
    assert np.max(np.abs(g - np.array(fd))) < 1e-7
