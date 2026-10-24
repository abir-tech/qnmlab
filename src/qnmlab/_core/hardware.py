"""
hardware.py -- the path from statevector demo to real quantum hardware.

This module adds everything the pure-statevector solver (quantum.py) lacks
for a realistic hardware run:

  1. PAULI DECOMPOSITION      H = sum_k c_k P_k  (P_k = tensor products of
                              I,X,Y,Z).  Real devices can only measure Pauli
                              strings, so this is how any Hermitian observable
                              is actually estimated on hardware.

  2. NOISY SIMULATION         a density-matrix simulator with a depolarizing
                              error channel after every gate -- the standard
                              first-order model of real-device noise.

  3. SHOT SAMPLING            expectation values estimated from a finite
                              measurement budget, allocated across Pauli terms.

  4. SPSA OPTIMIZER           the optimizer of choice on noisy hardware:
                              it tolerates noisy energies (unlike L-BFGS,
                              which chokes on them) and needs only 2 energy
                              evaluations per step regardless of the number
                              of parameters.

  5. ZERO-NOISE EXTRAPOLATION (ZNE) -- the flagship error-MITIGATION method:
                              run the same circuit at amplified noise levels
                              (1x, 2x, 3x), fit the trend, extrapolate to the
                              zero-noise limit.  No extra qubits needed.

  6. DILATION ENCODING        VQE on H(w) = [[0,P],[P^dag,0]] instead of
                              R = P^dag P.  The ground eigenvalue is
                              -sigma_min(P) instead of sigma_min(P)^2, so the
                              signal is SQRT-larger -- crucial when noise
                              floors are the binding constraint.

Terminology check (honesty): these are error *mitigation* techniques.  True
error *correction* (surface codes, logical qubits) needs thousands of physical
qubits per logical qubit and is not available on today's devices.
"""

import numpy as np
from scipy import linalg

from .quantum import ansatz_state  # reuse the exact-statevector ansatz

# ----------------------------------------------------------------------
# 1. Pauli decomposition
# ----------------------------------------------------------------------
_PAULIS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_matrix(label):
    M = _PAULIS[label[0]]
    for ch in label[1:]:
        M = np.kron(M, _PAULIS[ch])
    return M


def pauli_decompose(H, tol=1e-12):
    """
    Decompose a 2^n x 2^n Hermitian H into Pauli strings:

        H = sum_k  c_k  P_k ,     c_k = Tr(P_k H) / 2^n   (real for Hermitian H)

    Returns a list of (coeff, label) with |coeff| > tol.  This is what gets
    measured on a real device: each P_k needs its own measurement basis.
    """
    n = int(np.log2(H.shape[0]))
    assert 2 ** n == H.shape[0]
    labels = ["".join(s) for s in _product("IXYZ", n)]
    terms = []
    for lab in labels:
        c = np.trace(_pauli_matrix(lab) @ H).real / 2 ** n
        if abs(c) > tol:
            terms.append((c, lab))
    return terms


def _product(alphabet, repeat):
    if repeat == 1:
        for ch in alphabet:
            yield (ch,)
    else:
        for ch in alphabet:
            for rest in _product(alphabet, repeat - 1):
                yield (ch,) + rest


def pauli_reconstruct(terms, n):
    """Rebuild the matrix from its Pauli terms (for validation)."""
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for c, lab in terms:
        H += c * _pauli_matrix(lab)
    return H


# ----------------------------------------------------------------------
# 2. Noisy density-matrix simulator
# ----------------------------------------------------------------------
def _RY(t):
    return np.array([[np.cos(t / 2), -np.sin(t / 2)],
                     [np.sin(t / 2), np.cos(t / 2)]], dtype=complex)


def _RZ(t):
    return np.array([[np.exp(-1j * t / 2), 0],
                     [0, np.exp(1j * t / 2)]], dtype=complex)


def _embed_1q(U, q, n):
    ops = [np.eye(2, dtype=complex)] * n
    ops[q] = U
    M = ops[0]
    for o in ops[1:]:
        M = np.kron(M, o)
    return M


def _cnot_matrix(c, t, n):
    dim = 2 ** n
    M = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
        if bits[c] == 1:
            bits[t] ^= 1
        j = 0
        for b in bits:
            j = (j << 1) | b
        M[j, i] = 1.0
    return M


def _depolarize(rho, qubits, p, n):
    """Depolarizing channel with probability p on the given qubits."""
    if p <= 0:
        return rho
    out = (1 - p) * rho
    for q in qubits:
        for lab in ("X", "Y", "Z"):
            K = _embed_1q(_PAULIS[lab], q, n)
            out += (p / (3 * len(qubits))) * (K @ rho @ K.conj().T)
    return out


class NoisyAnsatzSim:
    """
    Density-matrix simulation of the (RY,RZ)+CNOT ansatz with depolarizing
    noise after every gate.

    p1 : error probability per single-qubit gate  (typical hardware: ~1e-4..1e-3)
    p2 : error probability per CNOT               (typical hardware: ~1e-3..1e-2)

    `noise_scale` multiplies both -- that is the knob ZNE turns.
    """

    def __init__(self, n_qubits, layers, p1=1e-3, p2=1e-2):
        self.n = n_qubits
        self.L = layers
        self.p1 = p1
        self.p2 = p2
        # pre-build the CNOT matrices once
        self._cnots = [_cnot_matrix(q, q + 1, n_qubits)
                       for q in range(n_qubits - 1)]

    def density_matrix(self, theta, noise_scale=1.0):
        n, L = self.n, self.L
        theta = np.asarray(theta).reshape(L + 1, n, 2)
        p1 = min(self.p1 * noise_scale, 1.0)
        p2 = min(self.p2 * noise_scale, 1.0)
        dim = 2 ** n
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 0] = 1.0
        for l in range(L):
            for q in range(n):
                U = _embed_1q(_RY(theta[l, q, 0]), q, n)
                rho = U @ rho @ U.conj().T
                rho = _depolarize(rho, [q], p1, n)
                U = _embed_1q(_RZ(theta[l, q, 1]), q, n)
                rho = U @ rho @ U.conj().T
                rho = _depolarize(rho, [q], p1, n)
            for q in range(n - 1):
                U = self._cnots[q]
                rho = U @ rho @ U.conj().T
                rho = _depolarize(rho, [q, q + 1], p2, n)
        for q in range(n):
            U = _embed_1q(_RY(theta[L, q, 0]), q, n)
            rho = U @ rho @ U.conj().T
            rho = _depolarize(rho, [q], p1, n)
            U = _embed_1q(_RZ(theta[L, q, 1]), q, n)
            rho = U @ rho @ U.conj().T
            rho = _depolarize(rho, [q], p1, n)
        return rho

    def energy(self, theta, H, noise_scale=1.0, shots=None, rng=None):
        """
        <H> = Tr(rho H), optionally with shot noise.

        With `shots`, the observable is measured term-by-term in its Pauli
        decomposition with the budget split evenly; each term's estimate gets
        binomial sampling error  sqrt((1 - <P>^2)/shots_per_term).
        """
        rho = self.density_matrix(theta, noise_scale)
        if shots is None:
            return float(np.real(np.trace(rho @ H)))
        terms = pauli_decompose(H)
        # identity term costs no measurements
        meas_terms = [(c, l) for c, l in terms if set(l) != {"I"}]
        ident = sum(c for c, l in terms if set(l) == {"I"})
        shots_per = max(1, shots // max(1, len(meas_terms)))
        rng = rng or np.random.default_rng()
        total = ident
        for c, lab in meas_terms:
            P = _pauli_matrix(lab)
            mean = float(np.real(np.trace(rho @ P)))
            mean = np.clip(mean, -1.0, 1.0)
            sampled = mean + rng.normal(0.0, np.sqrt(max(0.0, 1 - mean ** 2) / shots_per))
            total += c * sampled
        return float(total)


# ----------------------------------------------------------------------
# 3. SPSA -- the noise-tolerant optimizer used on real hardware
# ----------------------------------------------------------------------
def spsa_minimize(fun, x0, n_iter=300, a=0.2, c=0.15, seed=0,
                  A=None, alpha=0.602, gamma=0.101):
    """
    Simultaneous Perturbation Stochastic Approximation.

    Each iteration estimates the FULL gradient from just TWO function
    evaluations by perturbing all parameters simultaneously with a random
    +/-1 vector.  Gain sequences follow Spall's standard schedule.  Because
    it never trusts any single (noisy) evaluation too much, it converges
    where line-search methods (L-BFGS) fail on sampled energies.
    """
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float)
    A = A or n_iter // 10
    best_x, best_f = x.copy(), np.inf
    for k in range(n_iter):
        ak = a / (k + 1 + A) ** alpha
        ck = c / (k + 1) ** gamma
        delta = rng.choice([-1.0, 1.0], size=x.shape)
        f_plus = fun(x + ck * delta)
        f_minus = fun(x - ck * delta)
        ghat = (f_plus - f_minus) / (2 * ck) * delta
        x = x - ak * ghat
        f_now = min(f_plus, f_minus)
        if f_now < best_f:
            best_f, best_x = f_now, x.copy()
    # final unbiased evaluation at the best point
    return best_x, fun(best_x)


# ----------------------------------------------------------------------
# 4. Zero-noise extrapolation (ZNE)
# ----------------------------------------------------------------------
def zne_energy(sim, theta, H, scales=(1.0, 2.0, 3.0), shots=None, rng=None,
               order=2):
    """
    Error-mitigated energy by Richardson extrapolation.

    Evaluate the SAME circuit at artificially amplified noise levels (on
    hardware this is done by gate folding G -> G G^dag G; in our simulator we
    simply scale the depolarizing probability), then fit a polynomial in the
    noise scale and read off the value at scale 0.

    Returns (mitigated_energy, raw_energies_at_each_scale).
    """
    es = [sim.energy(theta, H, noise_scale=s, shots=shots, rng=rng)
          for s in scales]
    coeffs = np.polyfit(scales, es, deg=min(order, len(scales) - 1))
    e0 = float(np.polyval(coeffs, 0.0))
    return e0, es


# ----------------------------------------------------------------------
# 5. VQE runners: noisy, mitigated, and the dilation encoding
# ----------------------------------------------------------------------
def vqe_noisy(H, n_qubits, layers=3, p1=1e-3, p2=1e-2, shots=None,
              n_iter=300, restarts=3, seed=0, mitigate=False,
              scales=(1.0, 2.0, 3.0), theta0=None):
    """
    Full noisy VQE:  SPSA optimises the (noisy, sampled) energy; the final
    answer is either the raw noisy energy or the ZNE-mitigated one.

    theta0 : optional warm-start parameters.  Standard hardware practice is
             to pre-optimise on a classical simulator and only evaluate (or
             briefly refine) on the device; pass those parameters here.  With
             theta0 given and n_iter=0, the function performs pure noisy
             EVALUATION at theta0 (no optimisation) -- the cleanest way to
             isolate "what does noise do to the answer" from "can the
             optimiser survive noise".

    Returns dict with raw / mitigated energies and the optimal parameters.
    """
    sim = NoisyAnsatzSim(n_qubits, layers, p1=p1, p2=p2)
    nrm = np.linalg.norm(H, 2)
    Hn = H / nrm
    rng = np.random.default_rng(seed)

    if theta0 is not None and n_iter == 0:
        theta = np.asarray(theta0, dtype=float)
    else:
        best = {"f": np.inf, "x": None}
        starts = ([np.asarray(theta0, dtype=float)] if theta0 is not None else
                  [rng.uniform(0, 2 * np.pi, (layers + 1) * n_qubits * 2)
                   for _ in range(restarts)])
        for r, x0 in enumerate(starts):
            obj = lambda th: sim.energy(th, Hn, noise_scale=1.0,
                                        shots=shots, rng=rng)
            x, f = spsa_minimize(obj, x0, n_iter=n_iter, seed=seed + 17 * r)
            if f < best["f"]:
                best = {"f": f, "x": x}
        theta = best["x"]
    raw = sim.energy(theta, Hn, noise_scale=1.0, shots=shots, rng=rng) * nrm
    out = {"raw": raw, "theta": theta, "norm": nrm}
    if mitigate:
        e0, es = zne_energy(sim, theta, Hn, scales=scales, shots=shots, rng=rng)
        out["mitigated"] = e0 * nrm
        out["scale_energies"] = [e * nrm for e in es]
    # noiseless reference with the same parameters (what mitigation aims for)
    st = ansatz_state(theta, n_qubits, layers)
    out["noiseless_same_theta"] = float(np.real(st.conj() @ (H @ st)))
    return out


def vqe_dilation(P, layers=3, beta=0.05, p1=1e-3, p2=1e-2, shots=None,
                 n_iter=300, restarts=3, seed=0, mitigate=False,
                 scales=(1.0, 2.0, 3.0), theta0=None):
    """
    Estimate sigma_min(P) via VQE on the Hermitian dilation

        H = [[0, P], [P^dag, 0]]     (2N x 2N -> one extra qubit),

    whose spectrum is {+/- sigma_i(P)}.  The pair nearest zero is
    +/- sigma_min.

    How we target it variationally
    ------------------------------
    Minimise the "folded + pinned" objective

        F(theta) = <H^2>  +  beta * <H> ,        beta small > 0.

    <H^2> alone is minimised by BOTH +/- sigma_min eigenstates (folded
    spectrum at 0); the small linear pin term breaks the tie in favour of the
    -sigma_min eigenstate.  At the optimum, therefore,

        sigma_min  =  - <H>          (a LINEAR readout).

    Why this beats the residual encoding under noise
    ------------------------------------------------
    With R = P^dag P the reported number is <R> = sigma_min^2: to resolve a
    sigma_min of 1e-4 you must measure an observable at the 1e-8 level.  With
    the dilation the reported number is <H> = -sigma_min itself: you only
    need 1e-4 resolution.  The shot cost improves from O(1/sigma^4) to
    O(1/sigma^2) -- exactly the regime where Block C of run_hybrid_demo.py
    showed the residual encoding drowning in noise.

    Returns dict with:  sigma_est (from -<H>), raw/mitigated versions, theta.
    """
    N = P.shape[0]
    Z = np.zeros_like(P)
    H = np.block([[Z, P], [P.conj().T, Z]])
    n_qubits = int(np.log2(H.shape[0]))
    assert 2 ** n_qubits == H.shape[0], "dilation size must be a power of two"

    nrm = np.linalg.norm(H, 2)
    Hn = H / nrm
    H2n = Hn @ Hn
    sim = NoisyAnsatzSim(n_qubits, layers, p1=p1, p2=p2)
    rng = np.random.default_rng(seed)

    def objective(th):
        e2 = sim.energy(th, H2n, noise_scale=1.0, shots=shots, rng=rng)
        e1 = sim.energy(th, Hn, noise_scale=1.0, shots=shots, rng=rng)
        return e2 + beta * e1

    if theta0 is not None and n_iter == 0:
        theta = np.asarray(theta0, dtype=float)
    else:
        best = {"f": np.inf, "x": None}
        starts = ([np.asarray(theta0, dtype=float)] if theta0 is not None else
                  [rng.uniform(0, 2 * np.pi, (layers + 1) * n_qubits * 2)
                   for _ in range(restarts)])
        for r, x0 in enumerate(starts):
            x, f = spsa_minimize(objective, x0, n_iter=n_iter, seed=seed + 17 * r)
            if f < best["f"]:
                best = {"f": f, "x": x}
        theta = best["x"]
    e1_raw = sim.energy(theta, Hn, noise_scale=1.0, shots=shots, rng=rng)
    out = {"sigma_est": -e1_raw * nrm, "raw_H": e1_raw * nrm,
           "theta": theta, "norm": nrm}
    if mitigate:
        e0, es = zne_energy(sim, theta, Hn, scales=scales, shots=shots, rng=rng)
        out["sigma_mitigated"] = -e0 * nrm
        out["scale_energies"] = [e * nrm for e in es]
    # noiseless reference with the same parameters
    st = ansatz_state(theta, n_qubits, layers)
    out["noiseless_same_theta"] = -float(np.real(st.conj() @ (H @ st)))
    return out
