"""
mitigation.py -- measurement plans (QWC vs general-commuting grouping) and a
noise model with BOTH stochastic and coherent two-qubit error, plus optional
Pauli twirling and CNOT-folding ZNE.

Why two noise models: the hardware run shows a bias (~0.04) but not its origin.
Depolarizing (stochastic) and coherent ZZ over-rotation both reproduce it, and
the two mitigation techniques behave OPPOSITELY under them -- so any claim that
"twirling" or "ZNE" helps must be stated per model, and settled on hardware.
"""
import math

import numpy as np

from .grouping import diagonalising_circuit, gc_partition, to_sim_gates, unitary, z_image
from .hardware import pauli_decompose
from .hardware_run import (_cx_matrix, _depolarize, _embed, _mat1, ansatz_gates,
                           apply_readout_noise, basis_change_gates, counts_to_probs,
                           mitigate_readout, observables, pauli_from_probs, qwc_groups,
                           sample_counts)

_PAULI1 = {"X": ("x",), "Y": ("ry", math.pi), "Z": ("rz", math.pi)}


def plan(A, n, kind="gc"):
    """Measurement plan for omega = -<H2> + i<H1>.
    Returns dict(circuits=[(meas_gates, [(label, sign, zlabel)])], terms, identity)."""
    H1, H2 = observables(A)
    terms = {}
    for name, H in (("H1", H1), ("H2", H2)):
        for c, lab in pauli_decompose(H, tol=1e-10):
            terms.setdefault(lab, {})[name] = float(np.real(c))
    nonid = [l for l in terms if set(l) != {"I"}]
    ident = {k: terms.get("I" * n, {}).get(k, 0.0) for k in ("H1", "H2")}
    circuits = []
    if kind == "qwc":
        for g in qwc_groups([(1.0, l) for l in nonid], n):
            circuits.append((basis_change_gates(g["basis"]),
                             [(l, 1.0, "".join("I" if c == "I" else "Z" for c in l))
                              for _, l in g["terms"]]))
    elif kind == "gc":
        for fam in gc_partition(nonid):
            gts = to_sim_gates(diagonalising_circuit(fam, n))
            U = unitary(gts, n)
            circuits.append((gts, [(l,) + z_image(l, U, n) for l in fam]))
    else:
        raise ValueError(kind)
    return {"circuits": circuits, "terms": terms, "identity": ident, "n": n, "kind": kind}


def _conj_cx(pc, pt):
    xc, zc = pc in "XY", pc in "ZY"
    xt, zt = pt in "XY", pt in "ZY"
    xt, zc = xt ^ xc, zc ^ zt
    lab = lambda x, z: "I" if not (x or z) else "X" if x and not z else "Z" if z and not x else "Y"
    return lab(xc, zc), lab(xt, zt)


def simulate(gates, n, p2=0.0, eps=0.0, rng=None):
    """Density-matrix run.  After every CX: coherent ZZ(eps) then depolarizing p2.
    Single-qubit gates get p2/10.  rng given -> one random Pauli-twirl instance."""
    d = 2 ** n
    rho = np.zeros((d, d), complex)
    rho[0, 0] = 1.0

    def app(U, r):
        return U @ r @ U.conj().T

    for g in gates:
        if g[0] == "cx":
            c, t = g[1], g[2]
            if rng is not None:
                pc, pt = rng.choice(list("IXYZ")), rng.choice(list("IXYZ"))
                for q, p in ((c, pc), (t, pt)):
                    if p != "I":
                        rho = app(_embed(_mat1(*_PAULI1[p]), q, n), rho)
            rho = app(_cx_matrix(c, t, n), rho)
            if eps:
                par = [((i >> (n - 1 - c)) & 1) ^ ((i >> (n - 1 - t)) & 1) for i in range(d)]
                rho = app(np.diag([np.exp(-1j * eps / 2 * (1 - 2 * b)) for b in par]), rho)
            rho = _depolarize(rho, [c, t], p2, n)
            if rng is not None:
                for q, p in zip((c, t), _conj_cx(pc, pt)):
                    if p != "I":
                        rho = app(_embed(_mat1(*_PAULI1[p]), q, n), rho)
        else:
            rho = app(_embed(_mat1(g[0], g[2] if len(g) > 2 else None), g[1], n), rho)
            rho = _depolarize(rho, [g[1]], p2 / 10, n)
    return np.clip(np.real(np.diag(rho)), 0, None)


def fold(gates, k):
    return [x for g in gates for x in ([g] * k if g[0] == "cx" else [g])]


def omega_from_probs(P, probs_list, readout=None):
    e = dict(P["identity"])
    for (mg, items), pr in zip(P["circuits"], probs_list):
        if readout is not None:
            pr = mitigate_readout(pr, P["n"], *readout)
        for lab, sgn, zl in items:
            v = sgn * pauli_from_probs(pr, zl, P["n"])
            for k, c in P["terms"][lab].items():
                e[k] += c * v
    return -e["H2"] + 1j * e["H1"]


def run(P, theta, layers=3, p2=0.0, eps=0.0, readout=(0.002, 0.012), twirls=0,
        zne=False, shots=None, seed=0):
    """omega from a full simulated hardware run.  shots=None -> infinite shots."""
    n = P["n"]
    ro = (np.full(n, readout[0]), np.full(n, readout[1]))
    base = ansatz_gates(theta, n, layers)
    rng = np.random.default_rng(seed)

    def at_scale(k):
        probs = []
        for mg, _ in P["circuits"]:
            gl = fold(base + mg, k)
            pr = (np.mean([simulate(gl, n, p2, eps, rng) for _ in range(twirls)], axis=0)
                  if twirls else simulate(gl, n, p2, eps))
            pr = apply_readout_noise(pr, n, *ro)
            if shots:
                pr = counts_to_probs(sample_counts(pr, shots, rng), n)
            probs.append(pr)
        return omega_from_probs(P, probs, ro)

    w1 = at_scale(1)
    return (3 * w1 - at_scale(3)) / 2 if zne else w1


def n_circuits(P, zne=False, twirls=0):
    """Hardware circuits submitted (each twirl instance is a separate circuit)."""
    return len(P["circuits"]) * (2 if zne else 1) * max(twirls, 1) + 2
