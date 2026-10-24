"""
Pure-NumPy variational quantum solver for QNMs -- zero exotic dependencies.

Everything here runs on a tiny exact statevector simulator so you can test the
whole hybrid idea without installing Qiskit.  A Qiskit mirror for real hardware
lives in quantum_qiskit.py.

The physics loop
----------------
For a fixed trial frequency w, VQE finds

        E(w) = min_theta  <psi(theta)| R(w) |psi(theta)>   ~   sigma_min(P(w))^2 ,

the ground-state energy of the Hermitian residual R(w) = P(w)^dagger P(w).
Because E(w) -> 0 only at a QNM, the outer classical search

        w_QNM = argmin_w  E(w)

locates the quasinormal frequency.  This is the "variational + Hermitian
residual" route: VQE only ever sees a Hermitian, positive-semidefinite
operator, even though the underlying QNM problem is non-Hermitian.

Ansatz note
-----------
The ground eigenvector of R(w) is genuinely complex, so the ansatz needs both
RY and RZ rotations -- an RY-only ("real") ansatz cannot represent it and VQE
stalls.  Each layer is  (RY, RZ) on every qubit + a linear CNOT chain.
"""

import numpy as np
from scipy.optimize import minimize

from .pencil import leaver_matrix_std, equilibrate, residual


# ----------------------------------------------------------------------
# minimal exact statevector simulator
# ----------------------------------------------------------------------
def _apply_1q(state, U, q, n):
    st = state.reshape([2] * n)
    st = np.moveaxis(st, q, 0)
    st = np.tensordot(U, st, axes=(1, 0))
    st = np.moveaxis(st, 0, q)
    return st.reshape(-1)


def _cnot(state, c, t, n):
    st = state.reshape([2] * n)
    st = np.moveaxis(st, [c, t], [0, 1])
    out = st.copy()
    out[1, 0, ...] = st[1, 1, ...]
    out[1, 1, ...] = st[1, 0, ...]
    out = np.moveaxis(out, [0, 1], [c, t])
    return out.reshape(-1)


def _RY(t):
    return np.array([[np.cos(t / 2), -np.sin(t / 2)],
                     [np.sin(t / 2), np.cos(t / 2)]], dtype=complex)


def _RZ(t):
    return np.array([[np.exp(-1j * t / 2), 0],
                     [0, np.exp(1j * t / 2)]], dtype=complex)


def ansatz_state(theta, n_qubits, layers):
    """Hardware-efficient ansatz: (RY,RZ) per qubit + CNOT chain, `layers` deep."""
    theta = np.asarray(theta).reshape(layers + 1, n_qubits, 2)
    st = np.zeros(2 ** n_qubits, dtype=complex)
    st[0] = 1.0
    for l in range(layers):
        for q in range(n_qubits):
            st = _apply_1q(st, _RY(theta[l, q, 0]), q, n_qubits)
            st = _apply_1q(st, _RZ(theta[l, q, 1]), q, n_qubits)
        for q in range(n_qubits - 1):
            st = _cnot(st, q, q + 1, n_qubits)
    for q in range(n_qubits):
        st = _apply_1q(st, _RY(theta[layers, q, 0]), q, n_qubits)
        st = _apply_1q(st, _RZ(theta[layers, q, 1]), q, n_qubits)
    return st


# ----------------------------------------------------------------------
# VQE: ground-state energy of a Hermitian operator
# ----------------------------------------------------------------------
def vqe_ground_energy(H, n_qubits, layers=5, restarts=8, seed=0,
                      theta0=None, shots=None, return_theta=False):
    """
    Minimise <psi(theta)|H|psi(theta)> over the ansatz parameters.

    H         : (2^n_qubits square) Hermitian matrix (will be norm-rescaled).
    theta0    : optional warm-start parameters (skips random restarts if given).
    shots     : if set, add sampling noise ~ 1/sqrt(shots) to each energy eval.
    """
    nrm = np.linalg.norm(H, 2)
    Hn = H / nrm
    n_params = (layers + 1) * n_qubits * 2
    rng = np.random.default_rng(seed)

    def energy(theta):
        st = ansatz_state(theta, n_qubits, layers)
        e = float(np.real(st.conj() @ (Hn @ st)))
        if shots:
            e += rng.normal(0.0, 1.0 / np.sqrt(shots))  # crude shot-noise proxy
        return e

    best_f, best_x = np.inf, None
    starts = [theta0] if theta0 is not None else [rng.uniform(0, 2 * np.pi, n_params)
                                                  for _ in range(restarts)]
    for x0 in starts:
        res = minimize(energy, x0, method="L-BFGS-B", options={"maxiter": 1000})
        if res.fun < best_f:
            best_f, best_x = res.fun, res.x
    best_f *= nrm
    return (best_f, best_x) if return_theta else best_f


# ----------------------------------------------------------------------
# hybrid outer loop: find the QNM by minimising E(w)
# ----------------------------------------------------------------------
def hybrid_qnm(seed_w, l=2, N=8, layers=5, restarts=8, equil=True,
               shots=None, verbose=True):
    """
    Locate a QNM near `seed_w` (standard M w units) with the variational loop.

    Returns the found frequency, the final VQE energy, and the number of VQE
    calls used.  Warm-starts the ansatz across frequencies for speed.
    """
    n_qubits = int(np.log2(N))
    assert 2 ** n_qubits == N, "N must be a power of two"

    state = {"theta": None, "calls": 0}

    def objective(v):
        w = v[0] + 1j * v[1]
        P = leaver_matrix_std(w, l=l, N=N)
        if equil:
            P = equilibrate(P)
        R = residual(P)
        e, theta = vqe_ground_energy(
            R, n_qubits, layers=layers,
            restarts=(restarts if state["theta"] is None else 1),
            theta0=state["theta"], shots=shots, return_theta=True, seed=state["calls"])
        state["theta"] = theta          # warm start next frequency
        state["calls"] += 1
        if verbose and state["calls"] % 10 == 0:
            print(f"    [{state['calls']:3d} VQE calls]  w = {w:.5f}  E = {e:.3e}")
        return e

    res = minimize(objective, [seed_w.real, seed_w.imag],
                   method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-12, "maxiter": 400})
    w_found = res.x[0] + 1j * res.x[1]
    return w_found, res.fun, state["calls"]
