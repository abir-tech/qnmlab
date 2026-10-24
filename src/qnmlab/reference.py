"""
Classical reference frequencies -- the ground truth every quantum result is
scored against.

For any model, the hyperboloidal operator is diagonalised at high resolution and
only eigenvalues that do not move between two resolutions are kept (spurious
eigenvalues of a non-normal discretisation move; physical modes do not).
"""
import numpy as np

from ._core import hyperboloidal_metric as _HM


def reference_modes(bh, ell=2, n_modes=3, resolution=47, stability=1e-6):
    """
    The first `n_modes` quasinormal frequencies (M omega) of black hole `bh`, ordered
    by decreasing damping rate (fundamental first).

    Schwarzschild l=2 reproduces the continued-fraction value to ~1e-12.
    """
    bh.warn_if_experimental()
    hi = _modes(bh, ell, resolution)
    lo = _modes(bh, ell, resolution - 16)
    stable = [w for w in hi if np.min(np.abs(lo - w)) < stability]
    return np.array(stable[:n_modes])


def _modes(bh, ell, n_pts):
    A, _ = _HM.operator(n_pts, ell, bh.key, bh.parameter)
    w = 1j * np.linalg.eigvals(A)
    w = w[np.isfinite(w) & (w.real > 1e-6) & (w.imag < -1e-6)]
    return w[np.argsort(-w.imag)]
