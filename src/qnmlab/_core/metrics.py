"""
Black-hole metric definitions for the QNM solver.

Every metric is given as a *lapse function* f(r) in the line element

    ds^2 = -f(r) dt^2 + f(r)^-1 dr^2 + r^2 dOmega^2 .

For the spectral method we only ever need three things per metric:

    * f(r)          -- the lapse, as a sympy expression in the symbol r
    * r_h           -- the (outer) horizon radius, where f(r_h) = 0
    * mass M        -- the ADM mass (fixes the asymptotic factor r^{2 i M w})

Horizons are supplied in closed form to avoid root-branch ambiguity.

Adding a new black hole = adding one entry to METRICS. That is the whole
"many models" axis of the project.
"""

import sympy as sp

r, M, a = sp.symbols("r M a", positive=True)


def schwarzschild(M_val):
    """Classical Schwarzschild:  f = 1 - 2M/r,  r_h = 2M."""
    f = 1 - 2 * M / r
    return {
        "name": "Schwarzschild",
        "f": f.subs(M, M_val),
        "r_h": 2 * M_val,
        "M": M_val,
        "params": {},
    }


def kazakov_solodukhin(M_val, a_val):
    """
    Kazakov-Solodukhin quantum-corrected Schwarzschild:

        f = sqrt(r^2 - a^2)/r - 2M/r ,   r_h = sqrt(4 M^2 + a^2).

    a -> 0 recovers Schwarzschild.  a is the quantum-deformation scale.
    """
    f = sp.sqrt(r**2 - a**2) / r - 2 * M / r
    return {
        "name": f"Kazakov-Solodukhin (a={a_val})",
        "f": f.subs({M: M_val, a: a_val}),
        "r_h": sp.sqrt(4 * M_val**2 + a_val**2),
        "M": M_val,
        "params": {"a": a_val},
    }


def reissner_nordstrom(M_val, Q_val):
    """
    Reissner-Nordstrom charged BH:  f = 1 - 2M/r + Q^2/r^2.
    Outer horizon r_h = M + sqrt(M^2 - Q^2).  Included as a second test model.
    """
    Q = sp.Symbol("Q", nonnegative=True)
    f = 1 - 2 * M / r + Q**2 / r**2
    r_h = M_val + sp.sqrt(M_val**2 - Q_val**2)
    return {
        "name": f"Reissner-Nordstrom (Q={Q_val})",
        "f": f.subs({M: M_val, Q: Q_val}),
        "r_h": r_h,
        "M": M_val,
        "params": {"Q": Q_val},
    }


def hayward(M_val, l_val):
    """
    Hayward regular BH:  f = 1 - 2 M r^2 / (r^3 + 2 M l^2).
    Horizon supplied numerically by the caller-independent cubic; here we
    solve it symbolically and keep the largest real root.
    """
    l = sp.Symbol("l", positive=True)
    f = 1 - 2 * M * r**2 / (r**3 + 2 * M * l**2)
    fs = f.subs({M: M_val, l: l_val})
    roots = sp.solve(sp.Eq(fs, 0), r)
    real_roots = [rr for rr in roots if abs(sp.im(complex(rr))) < 1e-9]
    r_h = max(float(sp.re(complex(rr))) for rr in real_roots)
    return {
        "name": f"Hayward (l={l_val})",
        "f": fs,
        "r_h": r_h,
        "M": M_val,
        "params": {"l": l_val},
    }


# convenient registry for scripted sweeps
def build_catalogue():
    cat = [schwarzschild(1.0)]
    for a_val in (0.2, 0.5, 1.0):
        cat.append(kazakov_solodukhin(1.0, a_val))
    return cat
