"""Command line:  qnmlab modes | readout | search  (qnmlab <command> -h for options)."""
import argparse

from . import models

MODELS = {
    "schwarzschild": lambda p: models.Schwarzschild(),
    "kazakov-solodukhin": models.KazakovSolodukhin,
    "reissner-nordstrom": models.ReissnerNordstrom,
    "hayward": models.Hayward,
}


def _bh(a):
    return MODELS[a.model](a.parameter)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qnmlab", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("modes", "readout", "search"):
        p = sub.add_parser(name)
        p.add_argument("--model", choices=sorted(MODELS), default="schwarzschild")
        p.add_argument("--parameter", type=float, default=0.0, help="a, Q or l (units of M)")
        p.add_argument("--ell", type=int, default=2)
    sub.choices["modes"].add_argument("--n", type=int, default=3, help="number of modes")
    r = sub.choices["readout"]
    r.add_argument("--qubits", type=int, default=3)
    r.add_argument("--overtone", type=int, default=0)
    r.add_argument("--noise", choices=("ideal", "emerald-random", "emerald-coherent"), default="ideal")
    r.add_argument("--shots", type=int, default=None, help="shots per circuit (default: infinite)")
    s = sub.choices["search"]
    s.add_argument("--qubits", type=int, default=4)
    s.add_argument("--levels", type=int, default=5)
    s.add_argument("--solver", choices=("exact", "variational"), default="exact")
    a = ap.parse_args(argv)
    bh = _bh(a)

    if a.cmd == "modes":
        from .reference import reference_modes
        for k, w in enumerate(reference_modes(bh, a.ell, a.n)):
            print(f"n={k}:  M omega = {w.real:.10f} {w.imag:+.10f} i")
    elif a.cmd == "readout":
        from .readout import NoiseModel, measure_mode
        noise = {"ideal": NoiseModel.ideal(), "emerald-random": NoiseModel.emerald_random(),
                 "emerald-coherent": NoiseModel.emerald_coherent()}[a.noise]
        r = measure_mode(bh, a.ell, a.qubits, a.overtone, noise=noise, shots=a.shots)
        print(f"{bh}, l={a.ell}, {a.qubits} qubits, noise={a.noise}, shots/circuit={a.shots or 'infinite'}")
        print(f"  measured   M omega = {r.omega:.6f}")
        print(f"  reference  M omega = {r.meta['reference']:.6f}  (error {r.meta['error_vs_reference']:.2e})")
        print(f"  circuits {r.circuits}; lead from {r.meta['lead_source']}")
    else:
        from .search import find_modes
        res = find_modes(bh, a.ell, a.qubits, levels=a.levels, solver=a.solver, verbose=True)
        print(f"{len(res.candidates)} candidates, {len(res.modes)} accepted, scan shots {res.shots:.2e}")
        for w in res.modes:
            print(f"  mode  M omega = {w:.6f}")


if __name__ == "__main__":
    main()
