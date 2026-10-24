"""
Running the phase readout on real hardware.

    job = qnmlab.hardware.build_job(op, state)           # every circuit, as data
    circuits = [qnmlab.hardware.to_qiskit(c, op.qubits) for c in job["circuits"]]
    ... submit with your provider (Qiskit, qBraid, ...), collect raw counts ...
    omega = qnmlab.hardware.analyse_counts(job, raw_counts_by_name)

Two calibration circuits with asymmetric bit patterns are always included:
`analyse_counts` measures the provider's bit order from them (providers
differ), refuses to continue if a qubit is stuck, and removes readout error.
Qiskit is needed only for `to_qiskit` / `verify` (pip install qnmlab[hardware]).
"""
import numpy as np

from ._core import hardware_run as _hr


def build_job(op, state, grouping="auto", zne=False):
    """All measurement and calibration circuits for reading `state` on `op`."""
    if op.representation != "hyperboloidal":
        raise NotImplementedError("hardware jobs are built for the hyperboloidal readout")
    if grouping == "auto":
        grouping = "gc" if op.qubits <= 3 else "qwc"
    return _hr.build_job(state.theta, op.matrix, n=op.qubits, layers=state.layers,
                         zne_scales=(1, 3) if zne else (1,), grouping=grouping)


def to_qiskit(circuit, qubits):
    """One circuit of a job as a qiskit.QuantumCircuit (measurements included)."""
    return _hr.to_qiskit(circuit, qubits)


def verify(job, atol=1e-9):
    """Check, before spending anything, that the Qiskit circuits reproduce the
    package's own simulation.  Returns (worst probability difference, passed)."""
    return _hr.verify_qiskit_equivalence(job, atol)


def analyse_counts(job, raw_counts, mitigate=True):
    """
    raw_counts: {circuit name: {bitstring: count}} exactly as the provider returned.
    Returns a dict with omega, the detected bit order and the fitted readout errors.
    Raises ValueError if the calibration circuits show a stuck qubit or match
    neither bit order.
    """
    n = job["n"]
    preps = {c["name"]: c["prep"] for c in job["circuits"] if c["observable"] is None}
    order = _hr.detect_bit_order(raw_counts, preps, n)
    counts = {k: _hr.counts_from_device(v, n, order) for k, v in raw_counts.items()}
    out = _hr.analyse(job, counts, mitigate=mitigate)
    out["bit_order"] = order
    if len(job["zne_scales"]) > 1:
        out["zne_check"] = _hr.zne_sanity(out, job)
    return out


def simulate_counts(job, noise, shots, seed=0):
    """Raw counts (qubit 0 leftmost) for a job on a simulated noisy device -- a
    rehearsal of the hardware analysis path."""
    from ._core.hardware_run import apply_readout_noise, sample_counts
    from ._core.mitigation import simulate
    rng = np.random.default_rng(seed)
    n = job["n"]
    out = {}
    for c in job["circuits"]:
        p = simulate(c["gates"], n, noise.p2, noise.eps)
        p = apply_readout_noise(p, n, *[np.full(n, x) for x in noise.readout])
        out[c["name"]] = {format(i, f"0{n}b"): int(v) for i, v in sample_counts(p, shots, rng).items()}
    return out


def circuit_count(job):
    return len(job["circuits"])
