"""
Black-hole models.

Every model is a static, spherically symmetric metric

    ds^2 = -f(r) dt^2 + dr^2 / f(r) + r^2 dOmega^2      (units M = 1)

described by its lapse f(r) and one deformation parameter.  Two models are
VALIDATED against published quasinormal frequencies; the other two are
EXPERIMENTAL: their phase-readout operator is cross-checked against the
independent Frobenius pencil, but not against published values.
"""
import warnings
from dataclasses import dataclass

from ._core import hyperboloidal_metric as _HM


@dataclass(frozen=True)
class BlackHole:
    """A black hole the package can compute modes for.

    key        internal registry name
    parameter  the deformation (a for Kazakov-Solodukhin, Q for Reissner-Nordstrom,
               l for Hayward; 0 for Schwarzschild)
    validated  True if checked against published quasinormal frequencies
    """
    key: str
    parameter: float
    name: str
    validated: bool

    def metric(self):
        """dict with the sympy lapse 'f', horizon 'r_h', mass 'M'."""
        return _HM.MODELS[self.key](self.parameter)

    def warn_if_experimental(self):
        if not self.validated:
            warnings.warn(f"{self.name} is experimental in qnmlab: its operators are cross-checked "
                          "against an independent method but not against published frequencies.",
                          stacklevel=3)

    def __str__(self):
        return self.name


def Schwarzschild():
    return BlackHole("schwarzschild", 0.0, "Schwarzschild", True)


def KazakovSolodukhin(a):
    """Quantum-corrected Schwarzschild, f = sqrt(r^2 - a^2)/r - 2/r (Kazakov & Solodukhin 1994)."""
    if not 0 <= a:
        raise ValueError("deformation a must be >= 0")
    return BlackHole("kazakov_solodukhin", float(a), f"Kazakov-Solodukhin (a/M={a})", True)


def ReissnerNordstrom(Q):
    """Charged black hole, f = 1 - 2/r + Q^2/r^2.  Experimental."""
    if not 0 <= Q < 1:
        raise ValueError("charge must satisfy 0 <= Q/M < 1 for a black hole with a horizon")
    return BlackHole("reissner_nordstrom", float(Q), f"Reissner-Nordstrom (Q/M={Q})", False)


def Hayward(l):
    """Regular (Hayward) black hole, f = 1 - 2 r^2 / (r^3 + 2 l^2).  Experimental."""
    if not 0 <= l < 4 / 3 ** 1.5:
        raise ValueError("Hayward parameter must satisfy 0 <= l/M < 4/(3 sqrt 3) ~ 0.770 for a horizon")
    return BlackHole("hayward", float(l), f"Hayward (l/M={l})", False)


def available_models():
    """Constructors and their validation status."""
    return {"Schwarzschild": "validated", "KazakovSolodukhin(a)": "validated",
            "ReissnerNordstrom(Q)": "experimental", "Hayward(l)": "experimental"}
