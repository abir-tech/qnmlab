"""
scaling.py -- what changes as the qubit count grows, for the phase readout.

For each qubit count q (operator dimension 2^q) and each black hole:
  * discretisation error of the fundamental (what more qubits BUY),
  * Pauli terms and measurement circuits, QWC vs general-commuting (what they COST),
  * shots per circuit needed for a target accuracy, computed from the EXACT
    single-shot variance of every measurement circuit on the exact eigenvector.

Shot cost per circuit:  spread(S) = sqrt( sum_g Var_g / S ),  solved for S.
"""
import itertools

import numpy as np

from . import hyperboloidal_metric as HM
from .grouping import gc_partition, pmat
from .hardware import pauli_decompose
from .hardware_run import observables, qwc_groups


def _eigvec(A, w):
    lam, V = np.linalg.eig(A)
    k = int(np.argmin(np.abs(1j * lam - w)))
    return V[:, k] / np.linalg.norm(V[:, k])


def _group_variance(z, items, n):
    """Single-shot variance of sum_P c_P P for commuting P, evaluated exactly."""
    Hg = sum(c * pmat(l) for c, l in items)
    m = float(np.real(np.vdot(z, Hg @ z)))
    m2 = float(np.real(np.vdot(z, Hg @ (Hg @ z))))
    return max(m2 - m * m, 0.0)


def readout_row(qubits, model="schwarzschild", a=0.0, ell=2, target=1e-3, gc=True):
    n_pts = HM.qubits_to_npts(qubits)
    A = HM.energy_normed_operator(n_pts, ell, model, a)
    w = HM.fundamental(n_pts, ell, model, a)
    ref = HM.fundamental(47, ell, model, a)
    z = _eigvec(A, w)
    H1, H2 = observables(A)
    terms = {"H1": [(c, l) for c, l in pauli_decompose(H1, tol=1e-10) if set(l) != {"I"}],
             "H2": [(c, l) for c, l in pauli_decompose(H2, tol=1e-10) if set(l) != {"I"}]}
    labels = sorted({l for v in terms.values() for _, l in v})
    qwc = qwc_groups([(1.0, l) for l in labels], qubits)
    row = {"qubits": qubits, "dim": 2 ** qubits, "model": model, "a": a,
           "disc_error": abs(w - ref), "omega": w, "pauli_terms": len(labels),
           "qwc_circuits": len(qwc), "norm_H1": float(np.linalg.norm(H1, 2)),
           "norm_H2": float(np.linalg.norm(H2, 2))}
    # variance with QWC grouping (separately per observable, as on hardware)
    var = 0.0
    for name in ("H1", "H2"):
        coeff = dict((l, c) for c, l in terms[name])
        for g in qwc_groups([(coeff[l], l) for l in coeff], qubits):
            var += _group_variance(z, [(c, l) for c, l in g["terms"]], qubits)
    row["shots_per_circuit_qwc"] = var / target ** 2
    if gc:
        fams = gc_partition(labels)
        row["gc_circuits"] = len(fams)
        var = 0.0
        for fam in fams:
            for name in ("H1", "H2"):
                coeff = dict((l, c) for c, l in terms[name])
                items = [(coeff[l], l) for l in fam if l in coeff]
                if items:
                    var += _group_variance(z, items, qubits)
        row["shots_per_circuit_gc"] = var / target ** 2
    return row
