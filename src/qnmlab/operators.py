"""
Operators -- the matrices a quantum processor is asked about.

Two jobs, two representations each:

  readout_operator  the mode frequency is an EIGENVALUE (phase readout)
      "hyperboloidal"   energy-normed hyperboloidal collocation operator, any model
      "ultraspherical"  banded generalised pencil (A, B), Schwarzschild only

  search_pencil     the mode is where P(omega) becomes SINGULAR (search stage)
      "leaver"          tridiagonal Leaver pencil, Schwarzschild only
      "frobenius"       general Frobenius pencil, any model

Sizes are given in qubits: the operator dimension is 2**qubits.
"""
from dataclasses import dataclass

import numpy as np

from ._core import hyperboloidal_metric as _HM
from ._core.hyperboloidal import ultraspherical_pencil
from ._core.pencil import leaver_matrix_std
from ._core.search2d import Pencil as _Pencil


@dataclass
class ReadoutOperator:
    """matrix: the operator (or A of the pencil); B: None, or the pencil's B."""
    bh: object
    ell: int
    qubits: int
    representation: str
    matrix: np.ndarray
    B: np.ndarray = None

    @property
    def eigenvalues(self):
        """Mode frequencies of this finite operator, physical branch, fundamental first."""
        if self.B is None:
            lam = np.linalg.eigvals(self.matrix)
        else:
            from scipy.linalg import eig
            lam = eig(self.matrix, self.B, right=False)
        w = 1j * lam
        w = w[np.isfinite(w) & (w.real > 0.05) & (w.imag < 0)]
        return w[np.argsort(-w.imag)]


def readout_operator(bh, ell=2, qubits=3, representation="hyperboloidal"):
    """Operator whose eigenvalues are the modes, sized for `qubits` qubits."""
    bh.warn_if_experimental()
    if qubits < 2:
        raise ValueError("need at least 2 qubits")
    if representation == "hyperboloidal":
        A = _HM.energy_normed_operator(_HM.qubits_to_npts(qubits), ell, bh.key, bh.parameter)
        return ReadoutOperator(bh, ell, qubits, representation, A)
    if representation == "ultraspherical":
        if bh.key != "schwarzschild":
            raise NotImplementedError("the ultraspherical pencil is implemented for Schwarzschild only")
        A, B = ultraspherical_pencil(2 ** (qubits - 1), ell)
        return ReadoutOperator(bh, ell, qubits, representation, A, B)
    raise ValueError(f"unknown representation {representation!r}")


class SearchPencil(_Pencil):
    """Equilibrated P(omega) for the search; .sig2(w) is the normalised sigma_min^2."""

    def __init__(self, bh, ell=2, qubits=4, representation="leaver"):
        bh.warn_if_experimental()
        self.bh, self.representation = bh, representation
        self.model, self.a, self.ell, self.N, self.qubits = bh.key, bh.parameter, ell, 2 ** qubits, qubits
        self._cache = {}
        if representation == "leaver":
            if bh.key != "schwarzschild":
                raise ValueError("the Leaver pencil is Schwarzschild-only; use representation='frobenius'")
            self._P = lambda w: leaver_matrix_std(w, l=ell, N=self.N)
        elif representation == "frobenius":
            from ._core.recurrence import RecurrencePencil
            m = bh.metric()
            rp = RecurrencePencil(m["f"], m["r_h"], 1, ell)
            self._P = lambda w: rp.matrix(w, self.N)
        else:
            raise ValueError(f"unknown representation {representation!r}")


def search_pencil(bh, ell=2, qubits=4, representation="leaver"):
    return SearchPencil(bh, ell, qubits, representation)
