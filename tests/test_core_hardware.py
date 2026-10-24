"""
Tests for the hardware run.  Each one encodes a check that was verified by hand
while building the module, so none of it can silently regress before credits are
spent.  None of these need Qiskit or qBraid installed.
"""
import itertools

import numpy as np
import pytest

from qnmlab._core.hardware_run import (analyse, ansatz_gates, apply_readout_noise,
                              basis_change_gates, build_job, calibration_preps,
                              counts_from_device, counts_from_qiskit, detect_bit_order,
                              estimate, fit_readout, fold_cnots, mitigate_readout,
                              omega_from, observables, pauli_from_probs,
                              qwc_groups, simulate_probs, zne_sanity)
from qnmlab._core.quantum import ansatz_state
from qnmlab._core.rayleigh import energy_normed_operator

N, L = 4, 3
PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
         "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}


def _pmat(label):
    M = PAULI[label[0]]
    for ch in label[1:]:
        M = np.kron(M, PAULI[ch])
    return M


def _exact_counts(job, noise=None):
    out = {}
    for c in job["circuits"]:
        if noise:
            p = simulate_probs(c["gates"], job["n"], noise["p1"], noise["p2"])
            p = apply_readout_noise(p, job["n"], noise["p01"], noise["p10"])
        else:
            p = simulate_probs(c["gates"], job["n"])
        out[c["name"]] = {i: int(round(v * 1e12)) for i, v in enumerate(p) if v > 0}
    return out


@pytest.fixture(scope="module")
def setup():
    A = energy_normed_operator(7, 2)
    theta = np.random.default_rng(3).uniform(0, 2 * np.pi, (L + 1) * N * 2)
    return A, theta


# ---------------------------------------------------------------- circuits
def test_gate_list_reproduces_reference_ansatz(setup):
    _, th = setup
    p_ref = np.abs(ansatz_state(th, N, L)) ** 2
    assert np.max(np.abs(p_ref - simulate_probs(ansatz_gates(th, N, L), N))) < 1e-12


def test_basis_changes_measure_every_pauli(setup):
    """All 255 non-identity strings, on a random state."""
    _, th = setup
    psi = ansatz_state(th, N, L)
    for lab in ["".join(t) for t in itertools.product("IXYZ", repeat=N)][1:]:
        exact = float(np.real(np.vdot(psi, _pmat(lab) @ psi)))
        pr = simulate_probs(ansatz_gates(th, N, L) + basis_change_gates(lab.replace("I", "Z")), N)
        assert abs(exact - pauli_from_probs(pr, lab, N)) < 1e-12, lab


def test_qwc_groups_are_genuinely_compatible(setup):
    A, _ = setup
    job = build_job(np.zeros((L + 1) * N * 2), A, n=N, layers=L)
    for o in job["observables"].values():
        for grp in o["groups"]:
            for _, lab in grp["terms"]:
                assert all(p == "I" or p == b for p, b in zip(lab, grp["basis"]))


def test_folding_is_logically_identical_but_noisier(setup):
    _, th = setup
    g1 = ansatz_gates(th, N, L)
    g3 = fold_cnots(g1, 3)
    assert sum(g[0] == "cx" for g in g3) == 3 * sum(g[0] == "cx" for g in g1)
    assert np.allclose(simulate_probs(g1, N), simulate_probs(g3, N), atol=1e-12)
    with pytest.raises(ValueError):
        fold_cnots(g1, 2)


# ---------------------------------------------------------------- the readout
def test_pipeline_reproduces_the_state_exactly(setup):
    """87 circuits, exact probabilities -> must equal <z|A|z> to machine precision."""
    A, th = setup
    job = build_job(th, A, n=N, layers=L)
    z = ansatz_state(th, N, L)
    H1, H2 = observables(A)
    w_state = omega_from(np.real(np.vdot(z, H1 @ z)), np.real(np.vdot(z, H2 @ z)))
    r = analyse(job, _exact_counts(job), mitigate=False)
    assert abs(r["omega"] - w_state) < 1e-9


def test_readout_mitigation_removes_readout_bias(setup):
    A, th = setup
    job = build_job(th, A, n=N, layers=L)
    truth = analyse(job, _exact_counts(job), mitigate=False)["omega"]
    noisy = _exact_counts(job, dict(p1=0, p2=0, p01=0.02, p10=0.04))
    raw = analyse(job, noisy, mitigate=False)["omega"]
    fixed = analyse(job, noisy, mitigate=True)
    assert abs(raw - truth) > 1e-3, "readout noise should visibly bias the raw result"
    assert abs(fixed["omega"] - truth) < 1e-9, "tensored mitigation should remove it exactly"
    assert np.allclose(fixed["readout_p01"], 0.02) and np.allclose(fixed["readout_p10"], 0.04)


# ---------------------------------------------------------------- ZNE check
def test_zne_sanity_ok_on_real_folding(setup):
    A, th = setup
    job = build_job(th, A, n=N, layers=L, zne_scales=(1, 3))
    r = analyse(job, _exact_counts(job, dict(p1=1e-3, p2=5e-3, p01=0, p10=0)))
    assert zne_sanity(r, job).startswith("ok")


def test_zne_sanity_warns_when_folding_is_cancelled(setup):
    """The hardware failure the check exists for: a transpiler cancels CX CX = I."""
    A, th = setup
    job = build_job(th, A, n=N, layers=L, zne_scales=(1, 3))
    base = ansatz_gates(th, N, L)
    noise = dict(p1=1e-3, p2=5e-3)
    counts = {}
    for c in job["circuits"]:
        gates = c["gates"]
        if c["observable"] is not None and c["scale"] > 1:
            gates = base + basis_change_gates(c["basis"])        # folds removed
        p = simulate_probs(gates, N, noise["p1"], noise["p2"])
        counts[c["name"]] = {i: int(round(v * 1e12)) for i, v in enumerate(p) if v > 0}
    assert zne_sanity(analyse(job, counts), job).startswith("WARNING")


# ---------------------------------------------------------------- ordering
def test_qiskit_counts_endianness_on_asymmetric_state():
    """
    Flip ONLY qubit 0.  Internally (big-endian) that is index 8 = '1000'.
    Qiskit writes clbit 0 rightmost, so it reports '0001'.  An ordering bug
    would land the flip on qubit 3 and silently corrupt every Pauli.
    """
    converted = counts_from_qiskit({"0001": 1000}, 4)
    assert converted == {8: 1000}
    probs = np.zeros(16)
    probs[8] = 1.0
    assert pauli_from_probs(probs, "ZIII", 4) == -1.0     # qubit 0 flipped
    assert pauli_from_probs(probs, "IIIZ", 4) == +1.0     # qubit 3 untouched


def test_qiskit_counts_round_trip_every_index():
    for idx in range(16):
        internal = format(idx, "04b")                   # q0 q1 q2 q3
        qiskit_key = internal[::-1]                     # q3 q2 q1 q0
        assert counts_from_qiskit({qiskit_key: 7}, 4) == {idx: 7}


def test_calibration_is_asymmetric_and_covers_both_states():
    """All-zeros / all-ones cannot reveal bit order; these must."""
    for n in (3, 4):
        preps = calibration_preps(n)
        for q in range(n):
            assert {p[q] for p in preps.values()} == {"0", "1"}
        assert all(p != p[::-1] for p in preps.values())


def test_no_circuit_leaves_a_qubit_idle(setup):
    """IQM's precompiler rejects circuits whose gate-bearing qubits do not start at 0."""
    A, th = setup
    for zne in ((1,), (1, 3)):
        for c in build_job(th, A, n=N, layers=L, zne_scales=zne)["circuits"]:
            touched = {g[1] for g in c["gates"]} | {g[2] for g in c["gates"] if g[0] == "cx"}
            assert touched == set(range(N)), (c["name"], touched)


def test_calibration_circuits_prepare_exactly_their_bitstrings():
    for name, gates in __import__("qnmlab._core.hardware_run", fromlist=["x"]).calibration_gates(N).items():
        p = simulate_probs(gates, N)
        assert p[int(calibration_preps(N)[name], 2)] == pytest.approx(1.0, abs=1e-12)


def _as_provider_keys(counts, n, order):
    """Big-endian int counts -> the string keys a provider would return."""
    out = {}
    for i, c in counts.items():
        b = format(i, f"0{n}b")
        out[b[::-1] if order == "qiskit" else b] = c
    return out


@pytest.mark.parametrize("order", ["big", "qiskit"])
def test_bit_order_is_detected_and_the_frequency_survives(setup, order):
    """
    Regression for the first qBraid run: qir-sv returned qubit-0-LEFTMOST keys,
    the script assumed Qiskit order, and a noiseless run read 0.157-0.448j.
    Whichever order the provider uses, detection must recover the right omega.
    """
    A, th = setup
    job = build_job(th, A, n=N, layers=L)
    truth = analyse(job, _exact_counts(job), mitigate=False)["omega"]
    noisy = _exact_counts(job, dict(p1=0, p2=0, p01=0.03, p10=0.06))
    raw = {k: _as_provider_keys(v, N, order) for k, v in noisy.items()}
    preps = {c["name"]: c["prep"] for c in job["circuits"] if c["observable"] is None}
    assert detect_bit_order(raw, preps, N) == order
    counts = {k: counts_from_device(v, N, order) for k, v in raw.items()}
    assert abs(analyse(job, counts)["omega"] - truth) < 1e-9
    wrong = {k: counts_from_device(v, N, "big" if order == "qiskit" else "qiskit")
             for k, v in raw.items()}
    assert abs(analyse(job, wrong, mitigate=False)["omega"] - truth) > 1e-2


def test_bit_order_detection_refuses_garbage():
    preps = calibration_preps(N)
    with pytest.raises(ValueError):
        detect_bit_order({k: {"0110": 900} for k in preps}, preps, N)


def test_stuck_qubit_is_named_not_misread_as_bit_order():
    """The real Rigetti Cepheus smoke-test counts, 2026-09-17: the middle bit never moves."""
    raw = {"calA": {"000": 93, "001": 695, "100": 2, "101": 10},
           "calB": {"000": 41, "001": 1, "100": 722, "101": 36}}
    with pytest.raises(ValueError, match=r"position\(s\) \[1\] never change"):
        detect_bit_order(raw, calibration_preps(3), 3)


# ---------------------------------------------------------------- cost
def test_cost_counts_one_task_per_circuit(setup):
    A, th = setup
    job = build_job(th, A, n=N, layers=L)
    e = estimate(job, 1000, "aws:iqm:qpu:garnet")
    nt = len(job["circuits"])
    assert e["tasks"] == nt
    assert e["credits"] == pytest.approx(nt * 30 + nt * 1000 * 0.145)
    assert e["usd"] == pytest.approx(e["credits"] * 0.01)


def test_free_simulators_cost_nothing(setup):
    A, th = setup
    job = build_job(th, A, n=N, layers=L)
    assert estimate(job, 100000, "qbraid:qbraid:sim:qir-sv")["usd"] == 0.0
