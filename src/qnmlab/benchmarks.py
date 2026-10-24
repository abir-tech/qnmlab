"""
Benchmarks: published reference frequencies, the 2026 IQM Emerald run (raw
counts included, so the hardware result can be re-derived), and a comparison
helper.
"""
import json
from importlib import resources

import numpy as np

from .models import KazakovSolodukhin, Schwarzschild
from .reference import reference_modes

# Published scalar quasinormal frequencies, M omega, M = 1.
PUBLISHED = {
    ("Schwarzschild", 2, 0): 0.483643872211 - 0.096758775978j,
    ("Schwarzschild", 2, 1): 0.463851 - 0.295604j,
    ("Schwarzschild", 3, 0): 0.675366 - 0.096500j,
    ("Schwarzschild", 4, 0): 0.867416 - 0.096392j,
    ("Kazakov-Solodukhin (a/M=0.2)", 2, 0): 0.482034 - 0.096654j,
    ("Kazakov-Solodukhin (a/M=0.5)", 2, 0): 0.473887 - 0.096109j,
    ("Kazakov-Solodukhin (a/M=1.0)", 2, 0): 0.448362 - 0.094248j,
}


def published(bh, ell=2, overtone=0):
    """Published value for (bh, ell, overtone), or None."""
    return PUBLISHED.get((str(bh), ell, overtone))


def compare(omega, bh, ell=2, overtone=0):
    """Error of `omega` against the reference mode, and which mode it is nearest."""
    ref = reference_modes(bh, ell, n_modes=max(overtone + 2, 2))
    k = int(np.argmin(np.abs(ref - omega)))
    return {"reference": complex(ref[overtone]), "error": float(abs(omega - ref[overtone])),
            "nearest_overtone": k, "distances": [float(abs(omega - w)) for w in ref],
            "published": published(bh, ell, overtone)}


def emerald_2026():
    """The packaged IQM Emerald dataset (dict with raw counts, angles and settings)."""
    with resources.files("qnmlab.data").joinpath("emerald_2026.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def reanalyse_emerald_2026(mitigate=True):
    """
    Re-derive the hardware frequency from the raw counts, with the same analysis
    as the original run: qubit-wise groups, tensored readout mitigation.
    Returns the measured omega (0.43919 - 0.10430i with mitigation).
    """
    from ._core.hardware_run import analyse, build_job, counts_from_device
    from ._core.hyperboloidal_metric import energy_normed_operator
    d = emerald_2026()
    A = energy_normed_operator(3, d["ell"])
    job = build_job(np.array(d["theta"]), A, n=d["qubits"], layers=d["layers"], grouping=d["grouping"])
    counts = {k: counts_from_device(v, d["qubits"], "qiskit") for k, v in d["raw_counts"].items()}
    return complex(analyse(job, counts, mitigate=mitigate)["omega"])


VALIDATED_MODELS = (Schwarzschild, KazakovSolodukhin)
