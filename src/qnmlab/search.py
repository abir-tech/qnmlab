"""
Seedless mode search: a thresholded scan of the complex frequency plane, then
verification of every candidate by the phase readout.

  1. SCAN     at each grid point, a variational solver (or the exact value)
              estimates sigma_min^2 of the search pencil; points below a
              threshold (slope * h)^2 are kept and their cells refined.
  2. SHORTLIST  cluster centroids of the points that pass at the finest level.
  3. VERIFY   a variance-minimising state is steered to each candidate; the
              candidate is accepted if the state's frequency stays within
              `accept_tol` of it (a false candidate drifts to another eigenvalue).

The scan alone returns false candidates too (region edges, truncation
artefacts); verification is what turns the shortlist into modes.
"""
from dataclasses import dataclass, field

import numpy as np

from ._core.pipeline import calibrate as _calibrate
from ._core.pipeline import leads_from_search as _leads
from ._core.search2d import search as _search
from .operators import readout_operator, search_pencil
from .reference import reference_modes

# d sigma_min / d|omega| near the fundamental, Schwarzschild l = 2, Leaver pencil
# (measured on rings around the mode; the threshold rule needs it).
_SCHWARZSCHILD_SLOPE = {3: 2.79, 4: 2.37, 5: 1.69}


@dataclass
class Candidate:
    lead: complex
    state_omega: complex
    accepted: bool


@dataclass
class SearchResult:
    modes: list                       # accepted frequencies (readout-state values)
    candidates: list
    shots: float                      # measurement shots the scan would need
    circuit_evaluations: float        # hardware circuit evaluations of the solver (0 if exact)
    levels: list
    meta: dict = field(default_factory=dict)


def find_modes(bh, ell=2, qubits=4, region=((0.30, 0.70), (-0.32, -0.02)), levels=5,
               solver="exact", representation=None, slope=None, verify=True, readout_qubits=None,
               accept_tol=1e-2, readout_layers=5, readout_restarts=4, recheck_band=10.0, seed=11,
               verbose=False):
    """
    Find quasinormal modes inside `region` = ((Re min, Re max), (Im min, Im max))
    without a starting guess.

    solver          "exact" (fast; exact sigma_min^2 plus simulated shot noise) or
                    "variational" (a real variational solver at every grid point;
                    hours at 4 qubits)
    representation  "leaver" (Schwarzschild default) or "frobenius" (any model)
    slope           threshold slope; default: measured value for Schwarzschild l=2,
                    otherwise calibrated on the classical reference mode (recorded in meta)
    """
    representation = representation or ("leaver" if bh.key == "schwarzschild" else "frobenius")
    pen = search_pencil(bh, ell, qubits, representation)
    slope_source = "given"
    if slope is None:
        if bh.key == "schwarzschild" and ell == 2 and representation == "leaver" and qubits in _SCHWARZSCHILD_SLOPE:
            slope, slope_source = _SCHWARZSCHILD_SLOPE[qubits], "measured table (Schwarzschild, l=2)"
        else:
            slope = pen.slope(reference_modes(bh, ell, 1)[0])
            slope_source = "calibrated on the classical reference mode"
    mode = {"exact": "oracle", "variational": "vqe"}[solver]
    res = _search(pen, region, levels=levels, slope=slope, mode=mode, seed=seed,
                  recheck_band=recheck_band if mode == "vqe" else 0.0, verbose=verbose, max_cells=20000)
    leads = _leads(res)
    candidates = []
    if verify:
        op = readout_operator(bh, ell, readout_qubits or qubits)
        for k, lead in enumerate(leads):
            best = _calibrate(op.matrix, layers=readout_layers, restarts=readout_restarts,
                              seed=seed + k, lead=lead)
            w = best["w"]
            candidates.append(Candidate(lead, w, w is not None and abs(w - lead) <= accept_tol))
            if verbose:
                print(f"  candidate {lead:.4f} -> {w if w is None else f'{w:.5f}'}"
                      f"  {'accepted' if candidates[-1].accepted else 'rejected'}")
    else:
        candidates = [Candidate(lead, None, False) for lead in leads]
    modes = [c.state_omega for c in candidates if c.accepted]
    return SearchResult(modes, candidates, res["shots"], res["hw_evals"], res["levels"],
                        {"black_hole": str(bh), "ell": ell, "qubits": qubits, "solver": solver,
                         "representation": representation, "slope": slope, "slope_source": slope_source,
                         "region": region})
