"""
search2d.py -- the cold-start search with a REAL VQE at every grid point.

WHAT "THE SEARCH WITH THE VQE INSIDE" MEANS
------------------------------------------
The search scans trial frequencies w over a 2D region of the complex plane and
asks, at each one, a yes/no question:  "is sigma_min(P(w))^2 below a threshold?"
sigma_min^2 is the GROUND-STATE ENERGY of the Hermitian operator R(w) = P^dag P,
so answering the question needs a VQE: minimise <psi(theta)|R(w)|psi(theta)>.

Earlier versions answered it with the EXACT ground energy plus simulated shot
noise ("oracle" mode) -- i.e. it ASSUMED every VQE succeeded.  Here every grid
point runs an actual VQE ("vqe" mode), so a VQE that gets stuck above threshold
prunes the mode's cell and the search fails visibly.  Running both modes on the
same noise draws gives the number that matters: how many verdicts the VQE flips.

THE ALGORITHM (three rules; the third was learned from a bug)
---------------------------------------------------------------------
1. THRESHOLD, do not minimise: near a mode sigma^2 -> 0, so the argmin of a noisy
   field is noise.  A threshold only needs noise below itself.
2. Threshold and shots track the grid spacing h:  thresh = (slope*h)^2,
   shots = (5*l1/thresh)^2, so noise = thresh/5 at every level.
3. Keep EVERY cell with a point under threshold (no cap by position or value --
   both caps were bugs, see 4.12b), subdivide it 2x2, repeat.

SIMULATION SPEED vs HARDWARE COST -- two different counts, both reported
-----------------------------------------------------------------------
The simulator differentiates the energy by the ADJOINT method (cost ~3
statevector passes per gradient), which hardware cannot do.  Hardware would use
the parameter-shift rule: 2 circuit evaluations per parameter per gradient.  So
`hw_evals` counts what a device would have to execute, (1 + 2*n_params) energy
evaluations per optimizer call, independent of how fast the laptop is.

Shot noise is added to the FINAL energy of each point at the per-cell budget.
The optimizer itself runs on exact energies; on hardware it would run at a lower
shot count, which is not modelled here -- stated, not hidden.
"""
import time

import numpy as np
from scipy.optimize import minimize

from .hardware import pauli_decompose
from .pencil import equilibrate, leaver_matrix_std

MODES = {
    "schwarzschild": {"fundamental": 0.483643872211 - 0.096758775978j,
                      "overtone n=1": 0.463850579 - 0.295603937j},
}
KS_FUND = {0.2: 0.482034 - 0.096654j, 0.5: 0.473887 - 0.096109j, 1.0: 0.448362 - 0.094248j}


# ---------------------------------------------------------------- the operator
class Pencil:
    """Equilibrated P(w) for a model.  Schwarzschild: Leaver tridiagonal.
    Kazakov-Solodukhin: general Frobenius pencil (qnm.recurrence)."""

    def __init__(self, model="schwarzschild", a=0.0, ell=2, qubits=4):
        self.model, self.a, self.ell, self.N = model, a, ell, 2 ** qubits
        self.qubits = qubits
        if model == "schwarzschild":
            self._P = lambda w: leaver_matrix_std(w, l=ell, N=self.N)
        elif model == "kazakov_solodukhin":
            import sympy as sp
            from . import metrics
            from .recurrence import RecurrencePencil
            m = metrics.kazakov_solodukhin(1, sp.nsimplify(a))
            rp = RecurrencePencil(m["f"], m["r_h"], 1, ell)
            self._P = lambda w: rp.matrix(w, self.N)
        else:
            raise ValueError(model)
        self._cache = {}

    def R(self, w):
        """R(w)/||R||, the operator whose ground energy the VQE finds."""
        P = equilibrate(self._P(w))
        R = P.conj().T @ P
        return R / np.linalg.norm(R, 2)

    def sig2(self, w):
        """Exact ground energy of R(w)/||R|| = (sigma_min/||P||)^2."""
        k = (round(w.real, 10), round(w.imag, 10))
        if k not in self._cache:
            self._cache[k] = float(np.linalg.eigvalsh(self.R(w))[0])
        return self._cache[k]

    def l1(self, w):
        """Pauli 1-norm of R(w)/||R||: sets the shot noise of one energy estimate."""
        return sum(abs(c) for c, lab in pauli_decompose(self.R(w), tol=1e-10)
                   if set(lab) != {"I"})

    def slope(self, w_mode, r=(0.005, 0.01, 0.02, 0.04), n_ang=8):
        """d sigma / d|w| near a mode, averaged on rings (uses the mode location:
        a calibration, reported as such)."""
        vals = []
        for rad in r:
            ring = [np.sqrt(max(self.sig2(w_mode + rad * np.exp(2j * np.pi * k / n_ang)), 0)) / rad
                    for k in range(n_ang)]
            vals.append(np.mean(ring))
        return float(np.mean(vals))


# ---------------------------------------------------------------- VQE with adjoint gradient
def _ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(t):
    return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]])


_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.diag([1.0 + 0j, -1.0])


def _gate_list(n, layers):
    """Same ansatz as qnm.quantum.ansatz_state: (RY,RZ) per qubit + CX chain."""
    g, k = [], 0
    for l in range(layers + 1):
        for q in range(n):
            g.append(("ry", q, k)); g.append(("rz", q, k + 1)); k += 2
        if l < layers:
            for q in range(n - 1):
                g.append(("cx", q, q + 1))
    return g


def _apply1(st, U, q, n):
    v = np.moveaxis(st.reshape([2] * n), q, 0)
    return np.moveaxis(np.tensordot(U, v, axes=(1, 0)), 0, q).reshape(-1)


def _cx(st, c, t, n):
    v = np.moveaxis(st.reshape([2] * n), [c, t], [0, 1]).copy()
    v[1] = v[1][::-1].copy()
    return np.moveaxis(v, [0, 1], [c, t]).reshape(-1)


class VQE:
    def __init__(self, qubits, layers=3):
        self.n, self.layers = qubits, layers
        self.gates = _gate_list(qubits, layers)
        self.n_params = (layers + 1) * qubits * 2

    def state(self, th):
        st = np.zeros(2 ** self.n, complex); st[0] = 1
        for g in self.gates:
            if g[0] == "cx":
                st = _cx(st, g[1], g[2], self.n)
            else:
                st = _apply1(st, (_ry if g[0] == "ry" else _rz)(th[g[2]]), g[1], self.n)
        return st

    def energy_grad(self, th, H):
        n = self.n
        psi = self.state(th)
        lam = H @ psi
        E = float(np.real(np.vdot(psi, lam)))
        grad = np.zeros_like(th)
        for g in reversed(self.gates):
            if g[0] == "cx":
                psi = _cx(psi, g[1], g[2], n); lam = _cx(lam, g[1], g[2], n)
                continue
            U = (_ry if g[0] == "ry" else _rz)(th[g[2]])
            psi = _apply1(psi, U.conj().T, g[1], n)          # state before the gate
            G = (_Y if g[0] == "ry" else _Z)
            dpsi = _apply1(psi, -0.5j * G @ U, g[1], n)       # dU/dtheta |psi>
            grad[g[2]] = 2 * np.real(np.vdot(lam, dpsi))
            lam = _apply1(lam, U.conj().T, g[1], n)
        return E, grad

    def ground(self, H, theta0, maxiter=400):
        res = minimize(self.energy_grad, theta0, args=(H,), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-12})
        return float(res.fun), res.x, int(res.nfev), int(res.nit)


# ---------------------------------------------------------------- the search
def search(pencil, region=((0.30, 0.70), (-0.32, -0.02)), levels=4, start=9, sub=5,
           slope=2.64, mode="vqe", layers=3, seed=11, l1_ref=None, cold_restarts=4,
           verbose=True, max_cells=4000, recheck_band=0.0, recheck_restarts=2):
    """
    Adaptive thresholded scan.  mode = "vqe" (real VQE per point) or "oracle"
    (exact ground energy).  Both draw IDENTICAL shot noise for a given seed, so
    their verdicts can be compared point by point.

    recheck_band > 0 turns on the SECOND-OPINION rule: a VQE result that lands
    above the threshold but within recheck_band x threshold is re-run from
    `recheck_restarts` fresh random starts and the lowest energy kept.  A stuck
    warm start can only make a point look WORSE (higher energy), so false
    negatives -- pruning a mode's cell -- are the failure to guard against, and
    they can only come from points near the threshold.  Measured need: 4 qubits
    without it flipped 3 verdicts and lost the fundamental.

    Returns a dict with per-level records, every evaluated point, and costs.
    """
    rng = np.random.default_rng(seed)          # measurement noise ONLY
    rng_vqe = np.random.default_rng(seed + 10_000)   # VQE random starts: separate stream,
    # so the oracle and VQE runs see IDENTICAL noise draws point by point (an earlier
    # version shared one stream, so their noise drifted apart after the first VQE).
    w_ref = 0.48 - 0.10j
    l1 = l1_ref if l1_ref is not None else pencil.l1(w_ref)
    vqe = VQE(pencil.qubits, layers) if mode == "vqe" else None
    solved = []                     # (w, theta) for warm starts
    regions = [region]
    n_side = start
    out = {"levels": [], "points": [], "mode": mode, "l1": l1, "slope": slope,
           "hw_evals": 0, "sim_seconds": 0.0, "shots": 0.0, "n_params": vqe.n_params if vqe else None}
    t0 = time.time()
    for lv in range(levels):
        (a0, b0), _ = regions[0]
        h = (b0 - a0) / (n_side - 1)
        thr = (slope * h) ** 2
        shots = (5 * l1 / thr) ** 2
        noise_sd = l1 / np.sqrt(shots)
        keep, n_pts, flips, vqe_err = [], 0, 0, []
        for (a, b), (c, d) in regions:
            hit = False
            for x in np.linspace(a, b, n_side):
                for y in np.linspace(c, d, n_side):
                    w = x + 1j * y
                    exact = pencil.sig2(w)
                    z = rng.normal(0.0, noise_sd)
                    if vqe is not None:
                        H = pencil.R(w)
                        if solved:
                            th0 = min(solved, key=lambda s: abs(s[0] - w))[1]
                            E, th, nfev, _ = vqe.ground(H, th0)
                        else:                                   # first point: cold
                            E, th = np.inf, None
                            nfev = 0
                            for _ in range(cold_restarts):
                                e, t_, nf, _ = vqe.ground(H, rng_vqe.uniform(0, 2 * np.pi, vqe.n_params))
                                nfev += nf
                                if e < E:
                                    E, th = e, t_
                        if recheck_band > 0 and thr < E + z <= recheck_band * thr:
                            out.setdefault("rechecks", 0)
                            out["rechecks"] += 1
                            for _ in range(recheck_restarts):
                                e, t_, nf, _ = vqe.ground(H, rng_vqe.uniform(0, 2 * np.pi, vqe.n_params))
                                nfev += nf
                                if e < E:
                                    E, th = e, t_
                        solved.append((w, th))
                        out["hw_evals"] += nfev * (1 + 2 * vqe.n_params)
                        vqe_err.append(E - exact)
                        val = E + z
                    else:
                        val = exact + z
                    flag = val <= thr
                    if flag != (exact + z <= thr):
                        flips += 1
                        out.setdefault("flipped", []).append(
                            {"level": lv, "w": w, "vqe": val - z, "exact": exact, "threshold": thr})
                    hit |= flag
                    out["points"].append((lv, w, val, exact, flag))
                    n_pts += 1
                    out["shots"] += shots
            if hit:
                keep.append(((a, b), (c, d)))
        rec = {"level": lv, "h": h, "threshold": thr, "shots_per_point": shots,
               "points": n_pts, "kept_cells": len(keep), "verdicts_flipped_by_vqe": flips,
               "median_vqe_error": float(np.median(vqe_err)) if vqe_err else None,
               "max_vqe_error": float(np.max(vqe_err)) if vqe_err else None,
               "elapsed_s": time.time() - t0}
        out["levels"].append(rec)
        if verbose:
            extra = (f"  VQE: flipped {flips}, median err {rec['median_vqe_error']:.1e}"
                     if vqe is not None else "")
            print(f"level {lv}: h={h:.4f} thr={thr:.2e} shots/pt={shots:.1e} pts={n_pts:5d} "
                  f"kept={len(keep):4d}  [{rec['elapsed_s']:6.1f}s]{extra}")
        if not keep:
            break
        if 4 * len(keep) > max_cells:
            if verbose:
                print(f"  stopping: {4*len(keep)} cells would exceed max_cells={max_cells}")
            break
        regions = []
        for (a, b), (c, d) in keep:
            mx, my = 0.5 * (a + b), 0.5 * (c + d)
            regions += [((a, mx), (c, my)), ((mx, b), (c, my)), ((a, mx), (my, d)), ((mx, b), (my, d))]
        n_side = sub
    out["sim_seconds"] = time.time() - t0
    return out


def score(result, modes):
    """Best distance from any flagged point at the deepest level reached to each true mode."""
    last = max(p[0] for p in result["points"])
    flagged = [p[1] for p in result["points"] if p[0] == last and p[4]]
    return {k: (min(abs(w - m) for w in flagged) if flagged else np.inf) for k, m in modes.items()}
