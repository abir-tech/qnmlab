"""
hardware_run.py -- the hardware experiment, end to end, with one source of truth.

WHAT RUNS ON HARDWARE, AND WHAT DOES NOT
----------------------------------------
The full 2D search costs ~5.5e12 shots -- decades of QPU time.  It cannot run on
hardware and this module does not try.  What runs is the READOUT VALIDATION:

    given a quantum state prepared close to the target eigenvector, does a real
    device read the correct quasinormal frequency off two Hermitian observables?

That converts "simulated study" into "the readout was demonstrated on a device",
which is the single caveat every referee will raise.

Be precise about the claim.  The state is CALIBRATED CLASSICALLY, using the known
eigenvector, so this experiment validates the READOUT, not the search.  It does
not show that the device can find the mode -- that part of the method
(qnmlab.search) remains simulation-only.

ONE SOURCE OF TRUTH
-------------------
Every circuit is built once, as a backend-agnostic gate list.  The local dry run
executes that list on a density-matrix simulator; the hardware export translates
that same list to Qiskit.  What is tested is exactly what is submitted.

QUBIT ORDERING -- the classic silent bug
---------------------------------------
Internally everything is BIG-ENDIAN: qubit 0 is the leftmost Pauli label
character and the most significant bit of a probability index, matching
qnm.quantum and qnm.hardware.  Qiskit is LITTLE-ENDIAN: qubit 0 is the rightmost
character of a counts key.  `verify_qiskit_equivalence` checks the circuits
agree numerically with local Qiskit before anything is submitted.

A REMOTE PROVIDER NEED NOT FOLLOW QISKIT'S ORDER.  qBraid's qir-sv simulator
returned qubit-0-LEFTMOST keys; assuming Qiskit order turned a noiseless run into
omega = 0.157-0.448j (true 0.482-0.097j).  So the order is never assumed: the two
readout-calibration circuits are asymmetric ("100", "011") and
`detect_bit_order` reads the provider's convention off them on every run.
Getting this wrong produces plausible, wrong frequencies.

ERROR MITIGATION (not correction)
---------------------------------
True error CORRECTION needs many physical qubits per logical qubit and is not
available at this scale.  Two MITIGATION layers are implemented:

  * READOUT mitigation (tensored).  Two calibration circuits ("100", "011": each
    qubit sees both 0 and 1) give per-qubit bit-flip rates; their inverse is applied to every measured
    distribution.  Cheap, and for shallow circuits readout error is often the
    dominant error.
  * ZERO-NOISE EXTRAPOLATION by CNOT folding.  Each CX becomes CX CX CX (scale 3)
    or CX^5 (scale 5): logically unchanged, three or five times the two-qubit
    noise.  Extrapolate to zero.  Optional -- it multiplies the circuit count.

HARDWARE PITFALL THAT SILENTLY DESTROYS ZNE: a transpiler will cancel CX CX = I.
The Qiskit export inserts barriers between folded gates, and `zne_sanity` checks
that noisier scales really are noisier.  If scale 3 and scale 1 give the same
result, the folding was optimised away and the extrapolation is meaningless.
"""

import json
import math

import numpy as np
from scipy.optimize import minimize

from .hardware import pauli_decompose
from .quantum import ansatz_state

WQ = 0.483643872211 - 0.096758775978j

# ----------------------------------------------------------------------
# qBraid on-demand pricing, verified against docs.qbraid.com/v2/home/pricing
# 1 credit = $0.01.  Submitting a list of circuits creates ONE JOB PER CIRCUIT,
# so the per-task fee is paid per circuit.
# ----------------------------------------------------------------------
QBRAID_DEVICES = {
    "qbraid:qbraid:sim:qir-sv":        dict(kind="sim", task=0,  shot=0.0,    note="free statevector simulator"),
    "ionq:ionq:sim:simulator":         dict(kind="sim", task=0,  shot=0.0,    note="free IonQ simulator"),
    "aws:iqm:qpu:garnet":              dict(kind="sc",  task=30, shot=0.145,  note="IQM Garnet, 20q superconducting"),
    "aws:iqm:qpu:emerald":             dict(kind="sc",  task=30, shot=0.16,   note="IQM Emerald, 54q superconducting"),
    "aws:rigetti:qpu:cepheus-1-108q":  dict(kind="sc",  task=30, shot=0.0425, note="Rigetti Cepheus, 108q superconducting"),
    "aws:ionq:qpu:forte-1":            dict(kind="ion", task=30, shot=8.0,    note="IonQ Forte, trapped ion"),
    "ionq:ionq:qpu:forte-1":           dict(kind="ion", task=30, shot=8.0,    note="IonQ Forte, trapped ion"),
    "aws:aqt:qpu:ibex-q1":             dict(kind="ion", task=30, shot=2.35,   note="AQT IBEX Q1, trapped ion"),
}
# Devices routed through Open Quantum cost 0 qBraid credits and are billed to the
# Open Quantum account instead.  PUBLIC-compute prices from openquantum.com/pricing
# (read 2026-09-17), in the same units as above (1 credit = $0.01):
#   $0.18/task; per shot Emerald $0.00096, Garnet $0.00087, Cepheus $0.000255.
# Free "Spark" credits: 25 Spark = $50, so 1 Spark = $2; they expire after 90 days.
# `rigetti:rigetti:qpu:cepheus-1-108q` (direct) is deliberately absent: it bills
# per MINUTE (12,000 credits = $120/min), so submission to it stays refused.
for _vendor, _base, _shot in (("iqm:qpu:garnet", "aws:iqm:qpu:garnet", 0.087),
                              ("iqm:qpu:emerald", "aws:iqm:qpu:emerald", 0.096),
                              ("rigetti:qpu:cepheus-1-108q", "aws:rigetti:qpu:cepheus-1-108q", 0.0255)):
    QBRAID_DEVICES["openquantum:" + _vendor] = dict(
        QBRAID_DEVICES[_base], task=18, shot=_shot, billing="openquantum",
        note=QBRAID_DEVICES[_base]["note"] + " via Open Quantum (public compute)")
# Execution-time ASSUMPTIONS (queue time excluded -- it is unpredictable and can
# dwarf everything below).  Label these as estimates wherever quoted.
THROUGHPUT = {"sim": (50000.0, 1.0), "sc": (1000.0, 5.0), "ion": (20.0, 15.0)}
#            kind: (shots per second, overhead seconds per task)

# ======================================================================
# 1. circuits as gate lists
# ======================================================================
def ansatz_gates(theta, n, layers):
    """(RY, RZ) per qubit + linear CX chain, `layers` deep, plus a final layer.
    Must match qnm.quantum.ansatz_state exactly -- checked by a test."""
    th = np.asarray(theta, dtype=float).reshape(layers + 1, n, 2)
    g = []
    for l in range(layers):
        for q in range(n):
            g.append(("ry", q, float(th[l, q, 0])))
            g.append(("rz", q, float(th[l, q, 1])))
        for q in range(n - 1):
            g.append(("cx", q, q + 1))
    for q in range(n):
        g.append(("ry", q, float(th[layers, q, 0])))
        g.append(("rz", q, float(th[layers, q, 1])))
    return g


def basis_change_gates(basis):
    """Rotate so that measuring Z reads the requested Pauli on each qubit.
    X: H.   Y: Sdg then H  (H Sdg maps Y -> Z).   Z or I: nothing."""
    g = []
    for q, p in enumerate(basis):
        if p == "X":
            g.append(("h", q))
        elif p == "Y":
            g.append(("sdg", q))
            g.append(("h", q))
    return g


def fold_cnots(gates, scale):
    """ZNE folding: CX -> CX^scale for odd scale.  Logically identical."""
    if scale == 1:
        return list(gates)
    if scale % 2 == 0:
        raise ValueError("fold scale must be an odd integer (1, 3, 5, ...)")
    out = []
    for gt in gates:
        if gt[0] == "cx":
            out.extend([gt] * scale)
        else:
            out.append(gt)
    return out


# ======================================================================
# 2. qubit-wise commuting grouping
# ======================================================================
def qwc_groups(terms, n):
    """
    Greedy qubit-wise-commuting grouping.  Strings that agree on every qubit
    (or have I there) share one measurement circuit.  Each group records its
    measurement basis and its (coeff, label) members.
    """
    def compatible(a, b):
        return all(x == y or x == "I" or y == "I" for x, y in zip(a, b))

    groups = []
    for c, lab in sorted(terms, key=lambda t: -abs(t[0])):
        for grp in groups:
            if all(compatible(lab, m) for _, m in grp["terms"]):
                grp["terms"].append((c, lab))
                grp["basis"] = "".join(y if x == "I" else x
                                       for x, y in zip(grp["basis"], lab))
                break
        else:
            groups.append({"basis": lab, "terms": [(c, lab)]})
    for grp in groups:                       # fill remaining identity slots
        grp["basis"] = grp["basis"].replace("I", "Z")
    return groups


# ======================================================================
# 3. density-matrix simulation of a gate list (the dry run)
# ======================================================================
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _mat1(name, arg=None):
    if name == "ry":
        c, s = math.cos(arg / 2), math.sin(arg / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    if name == "rz":
        return np.array([[np.exp(-1j * arg / 2), 0], [0, np.exp(1j * arg / 2)]], dtype=complex)
    if name == "h":
        return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
    if name == "sdg":
        return np.array([[1, 0], [0, -1j]], dtype=complex)
    if name == "x":
        return _X.copy()
    raise ValueError(name)


def _embed(U, q, n):
    M = np.array([[1.0 + 0j]])
    for k in range(n):
        M = np.kron(M, U if k == q else _I2)
    return M


def _cx_matrix(c, t, n):
    d = 2 ** n
    M = np.zeros((d, d), dtype=complex)
    for i in range(d):
        bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
        if bits[c]:
            bits[t] ^= 1
        j = 0
        for b in bits:
            j = (j << 1) | b
        M[j, i] = 1.0
    return M


def _depolarize(rho, qubits, p, n):
    if p <= 0:
        return rho
    out = (1.0 - p) * rho
    for q in qubits:
        for P in (_X, _Y, _Z):
            K = _embed(P, q, n)
            out = out + (p / (3 * len(qubits))) * (K @ rho @ K.conj().T)
    return out


def simulate_probs(gates, n, p1=0.0, p2=0.0):
    """Ideal-measurement probabilities (big-endian) after the gate list."""
    d = 2 ** n
    rho = np.zeros((d, d), dtype=complex)
    rho[0, 0] = 1.0
    cache = {}
    for gt in gates:
        if gt[0] == "cx":
            key = ("cx", gt[1], gt[2])
            if key not in cache:
                cache[key] = _cx_matrix(gt[1], gt[2], n)
            U = cache[key]
            rho = U @ rho @ U.conj().T
            rho = _depolarize(rho, [gt[1], gt[2]], p2, n)
        else:
            arg = gt[2] if len(gt) > 2 else None
            U = _embed(_mat1(gt[0], arg), gt[1], n)
            rho = U @ rho @ U.conj().T
            rho = _depolarize(rho, [gt[1]], p1, n)
    return np.clip(np.real(np.diag(rho)), 0.0, None)


# ----- readout noise and its mitigation (tensored) --------------------
def _per_qubit_apply(probs, n, mats):
    """Apply a 2x2 matrix to each qubit's bit of a big-endian probability vector."""
    v = probs.reshape([2] * n)
    for q in range(n):
        v = np.moveaxis(np.tensordot(mats[q], np.moveaxis(v, q, 0), axes=(1, 0)), 0, q)
    return v.reshape(-1)


def readout_confusion(p01, p10):
    """C[measured, true].  p01 = P(read 1 | true 0), p10 = P(read 0 | true 1)."""
    return np.array([[1.0 - p01, p10], [p01, 1.0 - p10]])


def apply_readout_noise(probs, n, p01, p10):
    p01 = np.broadcast_to(p01, (n,)); p10 = np.broadcast_to(p10, (n,))
    return _per_qubit_apply(probs, n, [readout_confusion(a, b) for a, b in zip(p01, p10)])


def sample_counts(probs, shots, rng):
    p = np.clip(probs, 0, None)
    p = p / p.sum()
    k = rng.multinomial(int(shots), p)
    return {i: int(c) for i, c in enumerate(k) if c}


def counts_to_probs(counts, n):
    v = np.zeros(2 ** n)
    tot = sum(counts.values())
    for i, c in counts.items():
        v[int(i)] += c
    return v / max(tot, 1)


def calibration_preps(n):
    """
    The two tensored-readout calibration states, as big-endian bitstrings
    (qubit 0 leftmost).  Each qubit is prepared in 0 in one circuit and 1 in the
    other, which is all per-qubit (p01, p10) needs.

    DELIBERATELY ASYMMETRIC ("100" / "011", not "000" / "111").  All-zeros and
    all-ones read the same in either bit order, so they cannot reveal it.  The
    first real run (qBraid qir-sv, 2026-09-17) returned keys qubit-0-LEFTMOST,
    while local Qiskit returns them qubit-0-RIGHTMOST; assuming Qiskit order gave
    omega = 0.157-0.448j instead of 0.482-0.097j.  The bit order is a property of
    the provider, so it is now MEASURED from these two circuits on every run.
    """
    return {"calA": "1" + "0" * (n - 1), "calB": "0" + "1" * (n - 1)}


def calibration_gates(n):
    """
    X where the prep bit is 1; RZ(pi/2) where it is 0.  The RZ does nothing to |0>
    (a phase) and is a virtual, error-free frame change on superconducting devices,
    but it means NO QUBIT IS IDLE.  Without it IQM Garnet via qBraid rejected
    "100"/"011" with "QASM precompilation failed: Virtual qubit indices must start
    at 0" -- qubit 0 received no gate in the second circuit (2026-09-17).
    """
    return {name: [("x", q) if b == "1" else ("rz", q, math.pi / 2)
                   for q, b in enumerate(prep)]
            for name, prep in calibration_preps(n).items()}


def fit_readout(cal_probs, preps, n):
    """Per-qubit (p01, p10).  cal_probs: {name: big-endian probs};
    preps: {name: prepared bitstring}."""
    p01, p10 = np.full(n, np.nan), np.full(n, np.nan)
    for name, prep in preps.items():
        v = cal_probs[name].reshape([2] * n)
        for q in range(n):
            m = np.moveaxis(v, q, 0).reshape(2, -1).sum(axis=1)
            if prep[q] == "0":
                p01[q] = float(m[1])       # true 0, read 1
            else:
                p10[q] = float(m[0])       # true 1, read 0
    if np.isnan(p01).any() or np.isnan(p10).any():
        raise ValueError("calibration preps do not put every qubit in both 0 and 1")
    return p01, p10


def normalise_key(key, n):
    """Provider count keys -> an n-character bitstring (register spaces removed)."""
    bits = str(key).replace(" ", "")
    if bits.startswith("0x"):
        bits = format(int(bits, 16), f"0{n}b")
    return bits[-n:].zfill(n)


def detect_bit_order(raw_cal_counts, preps, n):
    """
    Decide from the calibration circuits how a provider orders its count keys.
    Returns "qiskit" (qubit 0 rightmost) or "big" (qubit 0 leftmost).  Raises if
    the two circuits disagree or neither order explains the data -- that means
    readout is too bad or the circuits were altered, and nothing should be analysed.
    """
    # A result bit that never changes across BOTH circuits cannot be a working qubit:
    # every position is prepared as 1 in one of them.  First seen on Rigetti
    # Cepheus via qBraid (2026-09-17): position 1 read 0 in 1,600 of 1,600 shots.
    stuck = []
    for pos in range(n):
        ones = sum(c for name in preps for k, c in raw_cal_counts[name].items()
                   if normalise_key(k, n)[pos] == "1")
        total = sum(sum(raw_cal_counts[name].values()) for name in preps)
        if ones == 0 or ones == total:
            stuck.append(pos)
    if stuck:
        raise ValueError(f"result bit position(s) {stuck} never change across both calibration "
                         "circuits, although every qubit is prepared as 1 in one of them -- "
                         "that qubit's measurement is missing or dead.  Do NOT analyse or "
                         "submit a full run on this device until this is explained.")
    votes = set()
    for name, prep in preps.items():
        raw = raw_cal_counts[name]
        top = normalise_key(max(raw, key=lambda k: raw[k]), n)
        if top == prep:
            votes.add("big")
        elif top == prep[::-1]:
            votes.add("qiskit")
        else:
            raise ValueError(f"{name}: prepared {prep}, most common outcome {top} -- "
                             "matches neither bit order")
    if len(votes) != 1:
        raise ValueError(f"calibration circuits disagree on bit order: {votes}")
    return votes.pop()


def counts_from_device(raw_counts, n, order):
    """Raw provider counts -> {big-endian int index: count}, given the detected order."""
    out = {}
    for key, c in raw_counts.items():
        bits = normalise_key(key, n)
        if order == "qiskit":
            bits = bits[::-1]
        elif order != "big":
            raise ValueError(order)
        idx = int(bits, 2)
        out[idx] = out.get(idx, 0) + int(c)
    return out


def mitigate_readout(probs, n, p01, p10):
    """Apply the inverse per-qubit confusion.  Result may dip slightly negative;
    that keeps expectation values unbiased, so it is NOT clipped."""
    inv = [np.linalg.inv(readout_confusion(a, b)) for a, b in zip(p01, p10)]
    return _per_qubit_apply(probs, n, inv)


# ======================================================================
# 4. expectation values from measured distributions
# ======================================================================
def pauli_from_probs(probs, label, n):
    """<P> for a Pauli string whose basis change has already been applied."""
    idx = np.arange(2 ** n)
    sign = np.ones(2 ** n)
    for q, p in enumerate(label):
        if p != "I":
            sign *= 1 - 2 * ((idx >> (n - 1 - q)) & 1)
    return float(np.dot(probs, sign))


def observables(A):
    """(H1, H2), Hermitian, with omega = -<H2> + i<H1>."""
    return (A + A.conj().T) / 2.0, (A - A.conj().T) / 2.0j


def omega_from(e1, e2):
    return -float(e2) + 1j * float(e1)


# ======================================================================
# 5. the job specification
# ======================================================================
def calibrate_theta(A, layers=3, n=4, target=WQ, restarts=20, seed=0):
    """
    Classical state preparation: maximise fidelity to the target eigenvector.

    HONESTY NOTE: this uses the known eigenvector.  The hardware experiment
    therefore validates the READOUT given a good state, not the search.
    """
    lam, V = np.linalg.eig(A)
    k = int(np.argmin(np.abs(1j * lam - target)))
    u = V[:, k] / np.linalg.norm(V[:, k])
    rng = np.random.default_rng(seed)
    npar = (layers + 1) * n * 2
    best = (-1.0, None)
    for _ in range(restarts):
        f = lambda t: 1.0 - abs(np.vdot(ansatz_state(t, n, layers), u)) ** 2
        r = minimize(f, rng.uniform(0, 2 * np.pi, npar), method="L-BFGS-B",
                     options={"maxiter": 5000})
        if 1.0 - r.fun > best[0]:
            best = (1.0 - r.fun, r.x)
    return best[1], best[0], complex(1j * lam[k])


def build_job(theta, A, n=4, layers=3, zne_scales=(1,), with_readout_cal=True,
              grouping="qwc"):
    """
    Everything that will be submitted, as data.  Returns a dict with:
      circuits : list of {name, observable, basis, scale, gates}
      groups   : per-observable QWC groups with their Pauli terms and norms

    grouping = "qwc"  qubit-wise commuting, one circuit set per observable (the
                      September hardware run: 28 circuits at 3 qubits)
    grouping = "gc"   general commuting, H1 and H2 measured from the SAME circuits:
                      9 circuits at 3 qubits, each ending in a short Clifford that
                      rotates its commuting family onto Z-strings (qnm.grouping).
    """
    H1, H2 = observables(A)
    obs = {}
    for name, H in (("H1", H1), ("H2", H2)):
        nrm = float(np.linalg.norm(H, 2))
        terms = [(c, l) for c, l in pauli_decompose(H / nrm, tol=1e-10)]
        ident = sum(c for c, l in terms if set(l) == {"I"})
        nonid = [(c, l) for c, l in terms if set(l) != {"I"}]
        obs[name] = {"norm": nrm, "identity": float(ident),
                     "groups": qwc_groups(nonid, n) if grouping == "qwc" else [],
                     "terms": [(float(np.real(c)), l) for c, l in nonid]}
    base = ansatz_gates(theta, n, layers)
    circuits = []
    if grouping == "gc":
        from .grouping import diagonalising_circuit, gc_partition, to_sim_gates, unitary, z_image
        labels = sorted({l for o in obs.values() for _, l in o["terms"]})
        families = []
        for fam in gc_partition(labels):
            diag = to_sim_gates(diagonalising_circuit(fam, n))
            U = unitary(diag, n)
            families.append((diag, [(l,) + z_image(l, U, n) for l in fam]))
        for scale in zne_scales:
            for k, (diag, items) in enumerate(families):
                circuits.append({"name": f"F{k}_s{scale}", "observable": "both", "group": k,
                                 "scale": scale, "basis": None, "measure": items,
                                 "gates": fold_cnots(base + diag, scale)})
    elif grouping != "qwc":
        raise ValueError(grouping)
    for scale in (zne_scales if grouping == "qwc" else ()):
        body = fold_cnots(base, scale)
        for oname, o in obs.items():
            for gi, grp in enumerate(o["groups"]):
                circuits.append({"name": f"{oname}_g{gi}_s{scale}", "observable": oname,
                                 "group": gi, "scale": scale, "basis": grp["basis"],
                                 "gates": body + basis_change_gates(grp["basis"])})
    if with_readout_cal:
        preps = calibration_preps(n)
        for cname, g in calibration_gates(n).items():
            circuits.append({"name": cname, "observable": None, "group": None,
                             "scale": 1, "basis": "Z" * n, "gates": g,
                             "prep": preps[cname]})
    return {"n": n, "layers": layers, "zne_scales": list(zne_scales), "grouping": grouping,
            "observables": obs, "circuits": circuits, "theta": list(map(float, theta))}


def job_stats(job):
    n_cal = sum(1 for c in job["circuits"] if c["observable"] is None)
    n_meas = len(job["circuits"]) - n_cal
    n_cx = sum(1 for g in ansatz_gates(job["theta"], job["n"], job["layers"]) if g[0] == "cx")
    return {"circuits": len(job["circuits"]), "measurement": n_meas,
            "calibration": n_cal, "cx_per_circuit_scale1": n_cx}


# ======================================================================
# 6. post-processing (identical for dry run and real hardware)
# ======================================================================
def analyse(job, counts_by_name, mitigate=True):
    """
    counts_by_name : {circuit name: {big-endian int index: count}}
    Returns omega, per-scale expectations, and the fitted readout error.
    """
    n = job["n"]
    p01 = p10 = None
    preps = {c["name"]: c["prep"] for c in job["circuits"] if c["observable"] is None}
    if mitigate and preps and all(k in counts_by_name for k in preps):
        p01, p10 = fit_readout({k: counts_to_probs(counts_by_name[k], n) for k in preps},
                               preps, n)
    per_scale = {}
    for scale in job["zne_scales"]:
        if job.get("grouping") == "gc":
            coeff = {k: dict((l, c) for c, l in o["terms"]) for k, o in job["observables"].items()}
            e = {k: o["identity"] for k, o in job["observables"].items()}
            for c in job["circuits"]:
                if c["observable"] != "both" or c["scale"] != scale:
                    continue
                probs = counts_to_probs(counts_by_name[c["name"]], n)
                if p01 is not None:
                    probs = mitigate_readout(probs, n, p01, p10)
                for lab, sign, zl in c["measure"]:
                    v = sign * pauli_from_probs(probs, zl, n)
                    for k in e:
                        e[k] += coeff[k].get(lab, 0.0) * v
            per_scale[scale] = {k: e[k] * job["observables"][k]["norm"] for k in e}
            continue
        vals = {}
        for oname, o in job["observables"].items():
            e = o["identity"]
            for gi, grp in enumerate(o["groups"]):
                cname = f"{oname}_g{gi}_s{scale}"
                probs = counts_to_probs(counts_by_name[cname], n)
                if p01 is not None:
                    probs = mitigate_readout(probs, n, p01, p10)
                for c, lab in grp["terms"]:
                    e += c * pauli_from_probs(probs, lab, n)
            vals[oname] = e * o["norm"]
        per_scale[scale] = vals
    scales = job["zne_scales"]
    if len(scales) == 1:
        e1, e2 = per_scale[scales[0]]["H1"], per_scale[scales[0]]["H2"]
    else:
        deg = min(len(scales) - 1, 2)
        e1 = float(np.polyval(np.polyfit(scales, [per_scale[s]["H1"] for s in scales], deg), 0.0))
        e2 = float(np.polyval(np.polyfit(scales, [per_scale[s]["H2"] for s in scales], deg), 0.0))
    return {"omega": omega_from(e1, e2), "per_scale": per_scale,
            "readout_p01": None if p01 is None else p01.tolist(),
            "readout_p10": None if p10 is None else p10.tolist()}


def noise_type_diagnostic(result, reference):
    """
    Is the device's error RANDOM-like (ZNE valid) or COHERENT (ZNE invalid)?
    Needs fold scales (1, 3).  Folding triples the two-qubit gates, so for random
    noise the error at scale 3 is ~3x the error at scale 1 and points the SAME way
    (complex ratio ~ 3 + 0i); coherent error does not scale like that and rotates.
    reference = the NOISELESS readout of the prepared state, i<z|A|z> (computable
    classically from the angles) -- not the exact mode: an imperfect state starts off
    the mode before any noise, and only the noise part should be tested.
    Thresholds set from simulation.
    """
    exact = reference
    ps = result["per_scale"]
    if not {1, 3} <= set(ps):
        return None
    w = {k: omega_from(ps[k]["H1"], ps[k]["H2"]) for k in (1, 3)}
    ratio = (w[3] - exact) / (w[1] - exact)
    mag, ang = abs(ratio), float(np.degrees(np.angle(ratio)))
    random_like = 2.0 <= mag <= 4.0 and abs(ang) <= 20.0
    return {"omega_s1": w[1], "omega_s3": w[3], "ratio": ratio, "magnitude": mag,
            "angle_deg": ang, "random_like": random_like,
            "verdict": ("random-like: ZNE and the two-black-hole difference are expected to work"
                        if random_like else
                        "NOT random-like: do not trust ZNE; the difference may not cancel")}


def zne_sanity(result, job):
    """
    Folding should make the state NOISIER, i.e. move <H> toward the maximally
    mixed value Tr(H)/d.  If scale 3 sits no closer to it than scale 1, the
    transpiler probably cancelled CX CX = I and the extrapolation is meaningless.

    NOT "toward zero": an earlier version assumed |<H>| shrinks under noise and
    raised a false alarm on a local simulator, where folding cannot be cancelled.
    H1 and H2 carry identity components, so the mixed-state value is non-zero --
    measured, noise GREW Im(omega) from -0.097 to -0.215.  Distance from the
    mixed value is the quantity noise reliably shrinks.
    """
    ps = result["per_scale"]
    if len(ps) < 2:
        return "ZNE not used"
    mixed = {k: o["identity"] * o["norm"] for k, o in job["observables"].items()}
    s = sorted(ps)
    m = [abs(ps[k]["H1"] - mixed["H1"]) + abs(ps[k]["H2"] - mixed["H2"]) for k in s]
    if m[-1] >= m[0] * 0.999:
        return ("WARNING: noisier scales are not closer to the mixed state -- CX folding "
                "was likely optimised away by the transpiler.  Do not trust the "
                "extrapolation.")
    return f"ok: folding drives the state toward mixed ({m[0]:.4f} -> {m[-1]:.4f})"


# ======================================================================
# 7. local dry run -- the rehearsal before spending credits
# ======================================================================
def dry_run(job, shots, p1=1e-3, p2=5e-3, p01=0.015, p10=0.03, seed=0,
            mitigate=True):
    """Run every circuit of the job on a noisy simulator and analyse it with the
    SAME `analyse` used for hardware results."""
    rng = np.random.default_rng(seed)
    n = job["n"]
    counts = {}
    for c in job["circuits"]:
        probs = simulate_probs(c["gates"], n, p1=p1, p2=p2)
        probs = apply_readout_noise(probs, n, p01, p10)
        counts[c["name"]] = sample_counts(probs, shots, rng)
    return analyse(job, counts, mitigate=mitigate)


# ======================================================================
# 8. cost and time
# ======================================================================
def estimate(job, shots, device_id):
    d = QBRAID_DEVICES[device_id]
    n_tasks = len(job["circuits"])
    total_shots = n_tasks * shots
    credits = n_tasks * d["task"] + total_shots * d["shot"]
    rate, overhead = THROUGHPUT[d["kind"]]
    seconds = total_shots / rate + n_tasks * overhead
    return {"device": device_id, "tasks": n_tasks, "shots_total": total_shots,
            "credits": credits, "usd": credits * 0.01,
            "exec_seconds_est": seconds, "note": d["note"]}


# ======================================================================
# 9. export to Qiskit (what qBraid accepts)
# ======================================================================
def to_qiskit(circuit, n):
    """Translate a gate list to a Qiskit circuit.  Barriers after every CX keep a
    transpiler from cancelling ZNE folds.  Measurement is appended here.

    Qubit mapping: internal big-endian qubit q -> Qiskit qubit q.  Qiskit writes
    counts keys LITTLE-endian; `counts_from_qiskit` undoes that."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n, n, name=circuit["name"])
    for gt in circuit["gates"]:
        op = gt[0]
        if op == "ry":
            qc.ry(gt[2], gt[1])
        elif op == "rz":
            qc.rz(gt[2], gt[1])
        elif op == "h":
            qc.h(gt[1])
        elif op == "sdg":
            qc.sdg(gt[1])
        elif op == "x":
            qc.x(gt[1])
        elif op == "cx":
            qc.cx(gt[1], gt[2])
            if circuit.get("scale", 1) > 1:
                qc.barrier()
        else:
            raise ValueError(op)
    qc.measure(range(n), range(n))
    return qc


def counts_from_qiskit(counts, n):
    """LOCAL Qiskit keys are little-endian bitstrings (qubit 0 rightmost).  Return
    {big-endian int index: count}.  Do NOT assume this for a remote provider --
    use detect_bit_order + counts_from_device (see calibration_preps)."""
    return counts_from_device(counts, n, "qiskit")


def verify_qiskit_equivalence(job, atol=1e-9):
    """
    Pre-flight check.  For every circuit, compare Qiskit's exact probabilities
    (after endianness conversion) with the internal simulator's.  Any mismatch
    means a gate, sign or ordering error, and submitting would waste credits.
    """
    from qiskit.quantum_info import Statevector
    n = job["n"]
    worst = 0.0
    for c in job["circuits"]:
        qc = to_qiskit(c, n).remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc)
        pq = sv.probabilities_dict()
        p_ext = np.zeros(2 ** n)
        for key, v in pq.items():
            p_ext[int(key[::-1], 2)] += v
        p_int = simulate_probs(c["gates"], n)
        worst = max(worst, float(np.max(np.abs(p_ext - p_int))))
    return worst, worst <= atol


def save_job(job, path):
    blob = dict(job)
    blob["circuits"] = [{k: (v if k != "gates" else [list(g) for g in v])
                         for k, v in c.items()} for c in job["circuits"]]
    for o in blob["observables"].values():
        for grp in o["groups"]:
            grp["terms"] = [[float(c), l] for c, l in grp["terms"]]
    with open(path, "w") as f:
        json.dump(blob, f, indent=1)
    return path
