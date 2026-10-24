"""
Phase readout: prepare an eigenstate, then read the mode frequency from two
order-one averages,

    Re omega = -<H2>,   Im omega = <H1>,   H1 = (A + A^dag)/2,  H2 = (A - A^dag)/(2i).

The state is prepared by minimising the variance <A^dag A> - |<A>|^2, which is
zero exactly on eigenvectors.  A random start lands on an arbitrary eigenvector,
so preparation takes a LEAD -- a rough frequency to steer towards -- supplied by
qnmlab.search (no prior knowledge) or, as a convenience, by the classical
reference.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np

from ._core import mitigation as _MI
from ._core.pipeline import calibrate as _calibrate
from ._core.search2d import VQE as _VQE
from .reference import reference_modes


@dataclass(frozen=True)
class NoiseModel:
    """Simulated device noise.

    p2        depolarising probability after each two-qubit gate (single-qubit: p2/10)
    eps       coherent ZZ over-rotation (radians) after each two-qubit gate
    readout   (P(read 1 | 0), P(read 0 | 1)) per qubit, corrected by mitigation
    """
    p2: float = 0.0
    eps: float = 0.0
    readout: tuple = (0.0, 0.0)

    @classmethod
    def ideal(cls):
        return cls()

    @classmethod
    def emerald_random(cls):
        """Random two-qubit error tuned to the bias of the 2026 IQM Emerald run."""
        return cls(p2=0.012, readout=(0.002, 0.012))

    @classmethod
    def emerald_coherent(cls):
        """Coherent alternative that reproduces the same bias."""
        return cls(p2=0.006, eps=0.3, readout=(0.002, 0.012))

    @property
    def is_ideal(self):
        return self.p2 == 0 and self.eps == 0 and self.readout == (0.0, 0.0)


@dataclass
class PreparedState:
    theta: np.ndarray
    layers: int
    qubits: int
    omega_state: complex          # i <z|A|z>, the noiseless readout of this state
    variance: float
    lead: complex = None


@dataclass
class Readout:
    omega: complex
    omega_state: complex
    noise: NoiseModel
    shots: int
    circuits: int
    meta: dict = field(default_factory=dict)


def default_layers(qubits):
    """Ansatz depth: 2 layers already prepare the 3-qubit eigenvectors exactly (fewer
    CNOTs, less noise); 4 qubits needed 5 layers for a reliable overtone."""
    return 2 if qubits <= 3 else 5


def prepare_state(op, lead=None, layers=None, restarts=4, seed=0):
    """Variance-minimising state preparation, steered towards `lead` if given."""
    layers = layers or default_layers(op.qubits)
    if op.representation != "hyperboloidal":
        raise NotImplementedError("state preparation is implemented for the hyperboloidal operator")
    if lead is None:
        warnings.warn("no lead given: a cold start may settle on any eigenvector, not the mode you want",
                      stacklevel=2)
    best = _calibrate(op.matrix, layers=layers, restarts=restarts, seed=seed, lead=lead)
    if best["theta"] is None:
        raise RuntimeError("no physically admissible state found; try more restarts or a lead")
    return PreparedState(best["theta"], layers, op.qubits, best["w"], best["var"], lead)


def phase_readout(op, state, noise=NoiseModel(), shots=None, grouping="auto", zne=False, seed=0):
    """
    Read the frequency of a prepared state.  Ideal noise and infinite shots return
    the exact value i<z|A|z>; otherwise every measurement circuit is simulated with
    `noise`, sampled with `shots` per circuit, and readout-mitigated.
    """
    if grouping == "auto":
        grouping = "gc" if op.qubits <= 3 else "qwc"
    plan = None
    if noise.is_ideal and shots is None and not zne:
        omega = state.omega_state
    else:
        plan = _MI.plan(op.matrix, op.qubits, grouping)
        omega = _MI.run(plan, state.theta, layers=state.layers, p2=noise.p2, eps=noise.eps,
                        readout=noise.readout, zne=zne, shots=shots, seed=seed)
    circuits = 0 if plan is None else _MI.n_circuits(plan, zne=zne)
    return Readout(complex(omega), state.omega_state, noise, shots, circuits,
                   {"grouping": grouping, "zne": zne})


def measure_mode(bh, ell=2, qubits=3, overtone=0, lead=None, noise=NoiseModel(), shots=None,
                 layers=None, restarts=4, seed=0):
    """
    One call: build the operator, prepare the state, read the frequency, compare.
    With lead=None the classical reference supplies the lead (recorded in meta);
    use qnmlab.find_modes for leads found with no prior knowledge.
    """
    from .operators import readout_operator
    op = readout_operator(bh, ell, qubits)
    ref = reference_modes(bh, ell, n_modes=overtone + 1)[overtone]
    lead_source = "given"
    if lead is None:
        lead, lead_source = ref, "classical reference"
    state = prepare_state(op, lead, layers, restarts, seed)
    r = phase_readout(op, state, noise, shots, seed=seed)
    target = op.eigenvalues[np.argmin(np.abs(op.eigenvalues - ref))]
    r.meta.update({"black_hole": str(bh), "ell": ell, "qubits": qubits, "lead": lead,
                   "lead_source": lead_source, "reference": ref,
                   "operator_eigenvalue": target, "discretisation_error": abs(target - ref),
                   "error_vs_operator": abs(r.omega - target), "error_vs_reference": abs(r.omega - ref)})
    return r


def ansatz_parameters(qubits, layers):
    return _VQE(qubits, layers).n_params
