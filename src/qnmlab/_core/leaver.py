"""
Leaver continued-fraction solver: the trusted CLASSICAL ground truth.

This is the reference every quantum result is checked against.  It solves
the Schwarzschild scalar QNM problem to machine precision using Leaver's
(1985) three-term recurrence and its continued fraction.

Units
-----
Leaver's recurrence is written in units 2M = 1 (horizon at r = 1).  The raw
frequencies it returns are therefore  w_raw = 2 * (M w).  All public helpers
here return the *standard* dimensionless combination  M w  (with M = 1),
i.e. they already divide by two.  So `solve()` returns 0.483644 - 0.096759 i
for the l = 2 fundamental, matching the literature directly.
"""

import numpy as np
from scipy.optimize import fsolve


def _abg(n, l, w, s=0):
    """Three-term recurrence coefficients (2M = 1 units, spin s)."""
    a = n * n + (2 - 2j * w) * n + 1 - 2j * w
    b = -(2 * n * n + (2 - 8j * w) * n + 1 - 4j * w - 8 * w * w + l * (l + 1) - s * s)
    g = n * n - 4j * w * n - 4 * w * w - s * s
    return a, b, g


def continued_fraction(l, w, s=0, n_terms=1000):
    """
    Evaluate Leaver's continued fraction  b0 - a0 g1/(b1 - a1 g2/(...)).
    A QNM is a root (in the 2M = 1 frequency `w`) of this function.
    """
    fr = 0.0
    for n in range(n_terms, 0, -1):
        Ap, _, _ = _abg(n - 1, l, w, s)
        _, B, _ = _abg(n, l, w, s)
        _, _, G = _abg(n, l, w, s)
        fr = Ap * G / (B - fr)
    _, b0, _ = _abg(0, l, w, s)
    return b0 - fr


def solve(l=2, s=0, seed=0.4837 - 0.0968j, n_terms=1000):
    """
    Find one QNM near `seed` (given in standard M w units, M = 1).
    Returns the frequency M w and the residual |CF|.
    """
    seed_raw = 2.0 * seed  # convert to 2M = 1 units for the recurrence

    def eqs(v):
        val = continued_fraction(l, v[0] + 1j * v[1], s, n_terms)
        return [val.real, val.imag]

    sol = fsolve(eqs, [seed_raw.real, seed_raw.imag], full_output=False)
    w_raw = sol[0] + 1j * sol[1]
    res = abs(continued_fraction(l, w_raw, s, n_terms))
    return 0.5 * w_raw, res  # divide by two -> standard M w


# reference table (standard M w units, M = 1) for quick validation
REFERENCE = {
    ("scalar", 2, 0): 0.483644 - 0.096759j,
    ("scalar", 2, 1): 0.463851 - 0.295604j,
    ("scalar", 3, 0): 0.675366 - 0.096500j,
    ("scalar", 4, 0): 0.867416 - 0.096392j,
}
