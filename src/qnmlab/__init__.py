"""
qnmlab -- black-hole quasinormal modes with quantum algorithms.

    import qnmlab as ql
    bh = ql.KazakovSolodukhin(a=0.5)
    ql.reference_modes(bh)                         # classical ground truth
    op = ql.readout_operator(bh, qubits=3)         # modes are its eigenvalues
    r = ql.measure_mode(bh, qubits=3, noise=ql.NoiseModel.emerald_random(), shots=800)
    res = ql.find_modes(ql.Schwarzschild(), qubits=4)   # search with no starting guess
"""
from . import benchmarks, hardware
from .models import BlackHole, Hayward, KazakovSolodukhin, ReissnerNordstrom, Schwarzschild, available_models
from .operators import readout_operator, search_pencil
from .readout import NoiseModel, measure_mode, phase_readout, prepare_state
from .reference import reference_modes
from .search import find_modes

__version__ = "0.1.1"

__all__ = ["BlackHole", "Schwarzschild", "KazakovSolodukhin", "ReissnerNordstrom", "Hayward",
           "available_models", "reference_modes", "readout_operator", "search_pencil", "NoiseModel",
           "prepare_state", "phase_readout", "measure_mode", "find_modes", "benchmarks", "hardware",
           "__version__"]
