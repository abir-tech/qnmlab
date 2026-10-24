"""
Construction of the frequency-dependent operator P(w) and the Hermitian
objects the quantum layer minimises.

Key idea
--------
A QNM is a frequency w where P(w) is singular, i.e. where its smallest
singular value sigma_min(P(w)) -> 0.  We build P(w) from Leaver's three-term
recurrence truncated to size N: the tridiagonal matrix

        [ b0  a0             ]
        [ g1  b1  a1         ]
  T  =  [     g2  b2  a2     ]                          (T = P(w) here)
        [         ..  ..  .. ]
        [             gN bN  ]

whose (near-)null space appears exactly at QNM frequencies.  Tridiagonal =
sparse = few Pauli terms = ideal for a quantum encoding.

Two Hermitian objects are exposed for the quantum solver:

  * residual   R(w) = P(w)^dagger P(w)          (N x N,  ceil(log2 N) qubits)
        ground eigenvalue  =  sigma_min(P)^2
  * dilation   H(w) = [[0, P],[P^dagger, 0]]    (2N x 2N, one extra qubit)
        eigenvalue nearest 0  =  +/- sigma_min(P)

Conditioning
------------
The Leaver matrix entries grow like n^2, so raw P(w) is very badly scaled
(cond ~ 1e8 already at N = 16).  `equilibrate` applies a fixed, w-independent
diagonal similarity that leaves the singular *locations* untouched but brings
the operator norm from ~1e5 down to ~10, which is what makes VQE tractable.
This scaling IS the practical answer to the manuscript's kappa ~ 1e21 warning.
"""

import numpy as np
from scipy import linalg
from .leaver import _abg


def leaver_matrix(w, l=2, N=16, s=0, units="2M=1"):
    """
    Tridiagonal Leaver matrix P(w) = T_N(w).

    `w` is expected in 2M = 1 units (the native units of the recurrence).
    Pass a standard M w value times two, or use `w_std` below.
    """
    T = np.zeros((N, N), dtype=complex)
    for n in range(N):
        a, b, g = _abg(n, l, w, s)
        T[n, n] = b
        if n + 1 < N:
            T[n, n + 1] = a
        if n - 1 >= 0:
            T[n, n - 1] = g
    return T


def leaver_matrix_std(w_std, l=2, N=16, s=0):
    """Same as leaver_matrix but `w_std` is in standard M w units (M = 1)."""
    return leaver_matrix(2.0 * w_std, l=l, N=N, s=s)


def equilibrate(P):
    """
    Fixed diagonal similarity P -> D P D with D = diag(1/(1+n)).

    D is nonsingular and w-independent, so P and DPD are singular at exactly
    the same frequencies -- the QNM condition is preserved -- but the operator
    norm and condition number drop by orders of magnitude.
    """
    N = P.shape[0]
    d = 1.0 / (1.0 + np.arange(N))
    D = np.diag(d)
    return D @ P @ D


def residual(P):
    """R(w) = P^dagger P.  Hermitian, PSD, ground eigenvalue = sigma_min(P)^2."""
    return P.conj().T @ P


def dilation(P):
    """H(w) = [[0, P],[P^dagger, 0]].  Hermitian, eigval nearest 0 = +/-sigma_min."""
    N = P.shape[0]
    Z = np.zeros((N, N), dtype=complex)
    return np.block([[Z, P], [P.conj().T, Z]])


def sigma_min(P):
    """Smallest singular value of P (the residual that vanishes at a QNM)."""
    return linalg.svd(P, compute_uv=False)[-1]


def sigma_min_std(w_std, l=2, N=16, s=0, equil=True):
    """sigma_min of the (optionally equilibrated) Leaver matrix at M w = w_std."""
    P = leaver_matrix_std(w_std, l=l, N=N, s=s)
    if equil:
        P = equilibrate(P)
    return sigma_min(P)
