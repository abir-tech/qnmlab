"""
hyperboloidal.py -- Schwarzschild on hyperboloidal (horizon-penetrating,
infinity-reaching) slices, discretised on Chebyshev-Lobatto points.

On these slices the quasinormal boundary conditions become mere regularity, both
boundaries are outflow, and the problem is a LINEAR eigenvalue problem

        d_tau u = A u ,        w = i * eig(A) .

Every mode is an eigenvalue of one fixed matrix -- which is what makes a phase
readout possible (qnmlab.readout).  qnmlab._core.hyperboloidal_metric derives the
same operator for any static, spherically symmetric lapse f(r).

Minimal gauge, Schwarzschild, M = 1, sigma = 2M/r in [0, 1] (0 = future null
infinity, 1 = horizon):

    W psi_tautau + 2 gamma psi_tausigma + gamma' psi_tau = (p psi_sigma)_sigma - q psi
    p = sigma^2 (1 - sigma)/2,  gamma = 2 sigma^2 - 1,  W = 8 (1 + sigma),
    q = (l(l+1) + sigma)/2,     with the identity gamma^2 + W p = 1 asserted.

Measured (l = 2 scalar): the fundamental converges to 4.4e-13 at 32 collocation
points, and one diagonalisation returns the first overtones with no seed.

Also provides the banded ultraspherical (Olver-Townsend) pencil used by the
generalised Rayleigh readout.

Caveat: Chebyshev discretisation of a non-normal operator also produces spurious
eigenvalues; physical modes are the ones that do not move when the resolution
changes.
"""

import math

import numpy as np


# ----------------------------------------------------------------------
# 1. minimal-gauge coefficient functions
# ----------------------------------------------------------------------
def minimal_gauge(sigma, ell):
    """
    (W, p, gamma, gamma', q) on sigma in [0,1] for Schwarzschild scalar, M = 1.

    Asserts the minimal-gauge identity gamma^2 + W p = 1, which certifies the
    height function and the sign conventions together.
    """
    sigma = np.asarray(sigma, dtype=float)
    W = 8.0 * (1.0 + sigma)
    p = sigma ** 2 * (1.0 - sigma) / 2.0
    g = 2.0 * sigma ** 2 - 1.0
    gp = 4.0 * sigma
    q = (ell * (ell + 1) + sigma) / 2.0
    if not np.allclose(g ** 2 + W * p, 1.0, atol=1e-12):
        raise RuntimeError("minimal-gauge identity gamma^2 + W p = 1 violated; "
                           "the height function or a sign convention is wrong")
    return W, p, g, gp, q


def characteristic_speeds(sigma):
    """(c+, c-).  Both >= 0 at scri, both <= 0 at the horizon: outflow both ends."""
    sigma = np.asarray(sigma, dtype=float)
    return (1.0 - sigma) / 4.0, -sigma ** 2 / (4.0 * (1.0 + sigma))


# ----------------------------------------------------------------------
# 2. Chebyshev collocation
# ----------------------------------------------------------------------
def cheb_lobatto(n_pts):
    """
    Chebyshev-Lobatto nodes on sigma in [0,1] and the differentiation matrix.

    sigma = 0 is future null infinity, sigma = 1 is the horizon.  `n_pts` is the
    polynomial degree N; the grid has N+1 points.
    """
    j = np.arange(n_pts + 1)
    x = np.cos(np.pi * j / n_pts)
    c = np.ones(n_pts + 1)
    c[0] = c[n_pts] = 2.0
    c = c * (-1.0) ** j
    X = np.tile(x, (n_pts + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n_pts + 1))
    D = D - np.diag(D.sum(axis=1))
    # sigma = (1 - x)/2   =>   d/dsigma = -2 d/dx
    return (1.0 - x) / 2.0, -2.0 * D


def operator(n_pts=32, ell=2):
    """
    The evolution generator A with  d_tau (psi, psi_dot) = A (psi, psi_dot).

    Returns (A, sigma).  A is 2(n_pts+1) square.  NO boundary conditions are
    imposed -- both boundaries are outflow, which is the entire point.
    """
    sigma, D = cheb_lobatto(n_pts)
    W, p, g, gp, q = minimal_gauge(sigma, ell)
    n = n_pts + 1
    Winv = np.diag(1.0 / W)
    L1 = Winv @ (D @ np.diag(p) @ D - np.diag(q))
    L2 = -Winv @ (2.0 * np.diag(g) @ D + np.diag(gp))
    return np.block([[np.zeros((n, n)), np.eye(n)], [L1, L2]]), sigma


def qnm_spectrum(n_pts=32, ell=2, n_modes=None, damped_only=True):
    """
    Every QNM from ONE diagonalization -- no seed, no root-find.

    w = i * eig(A).  Returns frequencies sorted by decreasing Im(w), i.e. the
    fundamental first, then overtones.
    """
    A, _ = operator(n_pts, ell)
    w = 1j * np.linalg.eigvals(A)
    w = w[np.isfinite(w)]
    if damped_only:
        w = w[(w.real > 1e-6) & (w.imag < -1e-6)]
    w = w[np.argsort(-w.imag)]
    return w if n_modes is None else w[:n_modes]


# ----------------------------------------------------------------------
# 3. banded ultraspherical (Olver-Townsend) variant
# ----------------------------------------------------------------------
def _X(lam, n):
    """Multiplication by x in the C^(lam) basis -- tridiagonal."""
    M = np.zeros((n, n))
    for k in range(n):
        if k + 1 < n:
            M[k + 1, k] = (k + 1) / (2 * (k + lam))
        if k - 1 >= 0:
            M[k - 1, k] = (k + 2 * lam - 1) / (2 * (k + lam))
    return M


def _mul(coef, lam, n):
    """a(x) as a polynomial in X_lam -- banded, bandwidth = deg(a)."""
    Xl = _X(lam, n)
    out = np.zeros((n, n))
    P = np.eye(n)
    for c in coef:
        out = out + c * P
        P = P @ Xl
    return out


def _D(k, n):
    M = np.zeros((n, n))
    f = 2.0 ** (k - 1) * math.factorial(k - 1)
    for m in range(n - k):
        M[m, m + k] = f * (m + k)
    return M


def _S(k, n):
    M = np.zeros((n, n))
    if k == 0:
        M[0, 0] = 1.0
        for m in range(1, n):
            M[m, m] = 0.5
        for m in range(n - 2):
            M[m, m + 2] = -0.5
    else:
        for m in range(n):
            M[m, m] = k / (m + k)
        for m in range(n - 2):
            M[m, m + 2] = -k / (m + 2 + k)
    return M


def ultraspherical_pencil(n_coef=16, ell=2):
    """
    Banded coefficient-space form:  A z = lambda B z,  w = i lambda.

    Exploits the fact that the minimal-gauge coefficients are LOW-DEGREE
    POLYNOMIALS in sigma, so every operator is banded (B has bandwidth 5).
    Kept because it is the natural "can we get linearity AND sparsity" attempt;
    the measured answer is no -- see item 4 of the module docstring.  Retained
    for the record, and because the pencil form is useful classically.
    """
    n = n_coef
    S0, S1, D1, D2 = _S(0, n), _S(1, n), _D(1, n), _D(2, n)
    S10 = S1 @ S0
    p4 = [4 * c for c in (1 / 16, 1 / 16, -1 / 16, -1 / 16)]     # 4p
    pp2 = [2 * c for c in (1 / 8, -2 / 8, -3 / 8)]               # 2p'
    q = [ell * (ell + 1) / 2 + 0.25, 0.25]
    g4 = [4 * c for c in (-0.5, 1.0, 0.5)]                       # 4 gamma
    gp = [2.0, 2.0]
    W = [12.0, 4.0]
    A21 = _mul(p4, 2, n) @ D2 + _mul(pp2, 2, n) @ S1 @ D1 - _mul(q, 2, n) @ S10
    A22 = -_mul(g4, 2, n) @ S1 @ D1 - _mul(gp, 2, n) @ S10
    Z = np.zeros((n, n))
    A = np.block([[Z, S10], [A21, A22]])
    B = np.block([[S10, Z], [Z, _mul(W, 2, n) @ S10]])
    return A, B


def interleave(M):
    """
    Reorder (psi block, v block) -> alternating.  This is a relabelling of the
    component index from the most- to the least-significant qubit, so it is free
    on hardware; it turns the 2x2 block layout into a genuinely banded matrix.
    """
    d = M.shape[0]
    n = d // 2
    perm = np.empty(d, dtype=int)
    perm[0::2] = np.arange(n)
    perm[1::2] = np.arange(n, d)
    return M[np.ix_(perm, perm)]


# ----------------------------------------------------------------------
# 4. the measurement that produced the verdict
# ----------------------------------------------------------------------
def compare_encodings(dims=(8, 16, 32), ell=2,
                      w_ref=0.483643872211 - 0.096758775978j, offset=0.05):
    """
    Leaver vs hyperboloidal at EQUAL qubit count, on the metrics that decide it.

    Returns a list of dicts with the Pauli support of the VQE residual and the
    normalized off-mode sigma_min (the quantity hardware must resolve).
    Reproduces the tables in the module docstring.
    """
    from .pencil import leaver_matrix_std, equilibrate, sigma_min
    from .hardware import pauli_decompose

    def n_pauli(H):
        return sum(1 for c, lab in pauli_decompose(H, tol=1e-10)
                   if set(lab) != {"I"})

    rows = []
    for dim in dims:
        qb = int(np.log2(dim))
        P = equilibrate(leaver_matrix_std(w_ref, l=ell, N=dim))
        Pn = P / np.linalg.norm(P, 2)
        Poff = equilibrate(leaver_matrix_std(w_ref + offset, l=ell, N=dim))
        A, _ = operator(dim // 2 - 1, ell)
        Aq = A - (-1j * w_ref) * np.eye(dim)
        An = Aq / np.linalg.norm(Aq, 2)
        Aoff = A - (-1j * (w_ref + offset)) * np.eye(dim)
        rows.append({
            "dim": dim, "qubits": qb, "pauli_max": 4 ** qb - 1,
            "pauli_leaver": n_pauli(Pn.conj().T @ Pn),
            "pauli_hyper": n_pauli(An.conj().T @ An),
            "signal_leaver": sigma_min(Poff) / np.linalg.norm(Poff, 2),
            "signal_hyper": sigma_min(Aoff) / np.linalg.norm(Aoff, 2),
        })
    return rows
