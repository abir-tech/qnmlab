"""Tests of the public interface: every number a user would quote."""
import warnings

import numpy as np
import pytest

import qnmlab as ql
from qnmlab._core.pencil import equilibrate
from qnmlab._core.recurrence import RecurrencePencil


# ---------------------------------------------------------------- models and reference
def test_schwarzschild_reference_matches_published_values():
    bh = ql.Schwarzschild()
    w = ql.reference_modes(bh, 2, 2)
    assert abs(w[0] - ql.benchmarks.PUBLISHED[("Schwarzschild", 2, 0)]) < 1e-10
    assert abs(w[1] - ql.benchmarks.PUBLISHED[("Schwarzschild", 2, 1)]) < 2e-6
    for ell in (3, 4):
        assert abs(ql.reference_modes(bh, ell, 1)[0] - ql.benchmarks.PUBLISHED[("Schwarzschild", ell, 0)]) < 2e-6


@pytest.mark.parametrize("a", [0.2, 0.5, 1.0])
def test_kazakov_solodukhin_reference_matches_published_values(a):
    bh = ql.KazakovSolodukhin(a)
    assert abs(ql.reference_modes(bh, 2, 1)[0] - ql.benchmarks.published(bh)) < 1e-6


@pytest.mark.parametrize("bh", [ql.ReissnerNordstrom(0.3), ql.ReissnerNordstrom(0.6),
                                ql.Hayward(0.3), ql.Hayward(0.6)])
def test_experimental_models_agree_with_the_independent_frobenius_method(bh):
    """No published values are bundled for these models, so the readout operator's
    mode is checked against a different method: the Frobenius pencil must be
    near-singular there and clearly not 0.01 away."""
    with pytest.warns(UserWarning, match="experimental"):
        w = ql.reference_modes(bh, 2, 1)[0]
    m = bh.metric()
    rp = RecurrencePencil(m["f"], m["r_h"], 1, 2)

    def rel_sigma(z):
        s = np.linalg.svd(equilibrate(rp.matrix(z, 32)), compute_uv=False)
        return s[-1] / s[0]
    assert rel_sigma(w) < 1e-7
    assert rel_sigma(w + 0.01) > 100 * rel_sigma(w)


def test_invalid_parameters_are_rejected():
    for make, bad in ((ql.ReissnerNordstrom, 1.0), (ql.Hayward, 0.8), (ql.KazakovSolodukhin, -0.1)):
        with pytest.raises(ValueError):
            make(bad)


# ---------------------------------------------------------------- operators
def test_readout_operator_eigenvalues_are_the_modes():
    op = ql.readout_operator(ql.KazakovSolodukhin(1.0), qubits=4)
    assert op.matrix.shape == (16, 16)
    assert abs(op.eigenvalues[0] - ql.benchmarks.published(ql.KazakovSolodukhin(1.0))) < 1e-4


def test_ultraspherical_pencil_contains_the_fundamental():
    op = ql.readout_operator(ql.Schwarzschild(), qubits=5, representation="ultraspherical")
    assert np.min(np.abs(op.eigenvalues - ql.benchmarks.PUBLISHED[("Schwarzschild", 2, 0)])) < 1e-5


def test_search_pencil_is_singular_at_the_mode():
    pen = ql.search_pencil(ql.Schwarzschild(), qubits=4)
    w = ql.benchmarks.PUBLISHED[("Schwarzschild", 2, 0)]
    assert pen.sig2(w) < 1e-10 < pen.sig2(w + 0.05)
    with pytest.raises(ValueError):
        ql.search_pencil(ql.KazakovSolodukhin(0.5), representation="leaver")


# ---------------------------------------------------------------- readout
def test_measure_mode_ideal_hits_the_operator_eigenvalue():
    r = ql.measure_mode(ql.KazakovSolodukhin(0.5), qubits=3, layers=3)
    assert r.meta["error_vs_operator"] < 1e-4
    assert r.meta["lead_source"] == "classical reference"


def test_noisy_readout_is_biased_and_mitigation_runs():
    bh = ql.Schwarzschild()
    op = ql.readout_operator(bh, qubits=3)
    state = ql.prepare_state(op, lead=0.48 - 0.097j, layers=2, restarts=2)
    ideal = ql.phase_readout(op, state)
    noisy = ql.phase_readout(op, state, noise=ql.NoiseModel.emerald_random(), shots=None)
    assert ideal.circuits == 0 and noisy.circuits == 11          # 9 groups + 2 calibration
    assert 0.01 < abs(noisy.omega - ideal.omega) < 0.06


def test_cold_start_warns():
    """Without a lead the solver may land on any eigenvector -- or, as here, on the
    unphysical mirror branch, which prepare_state refuses."""
    op = ql.readout_operator(ql.Schwarzschild(), qubits=3)
    with pytest.warns(UserWarning, match="cold start"):
        try:
            ql.prepare_state(op, lead=None, layers=1, restarts=1)
        except RuntimeError:
            pass


# ---------------------------------------------------------------- search
def test_search_shortlist_contains_both_modes_without_verification():
    res = ql.find_modes(ql.Schwarzschild(), qubits=4, levels=5, verify=False)
    leads = np.array([c.lead for c in res.candidates])
    for key in (("Schwarzschild", 2, 0), ("Schwarzschild", 2, 1)):
        assert np.min(np.abs(leads - ql.benchmarks.PUBLISHED[key])) < 5e-3
    assert res.meta["slope_source"].startswith("measured")


@pytest.mark.slow
def test_search_with_verification_accepts_exactly_the_two_modes():
    res = ql.find_modes(ql.Schwarzschild(), qubits=4, levels=5)
    assert len(res.modes) == 2
    for key in (("Schwarzschild", 2, 0), ("Schwarzschild", 2, 1)):
        assert np.min(np.abs(np.array(res.modes) - ql.benchmarks.PUBLISHED[key])) < 5e-3


# ---------------------------------------------------------------- hardware path
@pytest.mark.parametrize("order", ["big", "qiskit"])
def test_hardware_analysis_measures_bit_order_and_recovers_omega(order):
    op = ql.readout_operator(ql.Schwarzschild(), qubits=3)
    state = ql.prepare_state(op, lead=0.48 - 0.097j, layers=2, restarts=2)
    job = ql.hardware.build_job(op, state)
    raw = ql.hardware.simulate_counts(job, ql.NoiseModel(readout=(0.02, 0.04)), shots=200000)
    if order == "qiskit":
        raw = {k: {b[::-1]: c for b, c in v.items()} for k, v in raw.items()}
    out = ql.hardware.analyse_counts(job, raw)
    assert out["bit_order"] == order
    assert abs(out["omega"] - state.omega_state) < 0.01


def test_stuck_qubit_is_refused():
    op = ql.readout_operator(ql.Schwarzschild(), qubits=3)
    state = ql.prepare_state(op, lead=0.48 - 0.097j, layers=1, restarts=1)
    job = ql.hardware.build_job(op, state)
    raw = {"calA": {"000": 93, "001": 695, "100": 2, "101": 10},
           "calB": {"000": 41, "001": 1, "100": 722, "101": 36}}
    with pytest.raises(ValueError, match="never change"):
        ql.hardware.analyse_counts(job, raw)


# ---------------------------------------------------------------- benchmarks and CLI
def test_packaged_emerald_counts_reproduce_the_published_result():
    assert abs(ql.benchmarks.reanalyse_emerald_2026() - (0.43919 - 0.10430j)) < 1e-5
    assert abs(ql.benchmarks.reanalyse_emerald_2026(mitigate=False) - (0.43622 - 0.11552j)) < 1e-5


def test_compare_identifies_the_mode():
    c = ql.benchmarks.compare(0.4392 - 0.1043j, ql.Schwarzschild())
    assert c["nearest_overtone"] == 0 and c["error"] < 0.05


def test_cli_prints_modes(capsys):
    from qnmlab.cli import main
    main(["modes", "--model", "kazakov-solodukhin", "--parameter", "0.5", "--n", "1"])
    assert "0.4738" in capsys.readouterr().out
