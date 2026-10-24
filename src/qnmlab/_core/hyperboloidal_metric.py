"""
hyperboloidal_metric.py -- the phase-readout operator for ANY static, spherically
symmetric, asymptotically flat black hole, not just Schwarzschild.

WHY
---
`hyperboloidal.minimal_gauge` hard-codes Schwarzschild.  The phase readout of
rayleigh.py therefore only ever ran on Schwarzschild, while the
Kazakov-Solodukhin (KS) results lived on the Leaver/Frobenius side (search
only).  This module derives the same minimal-gauge operator from a lapse f(r),
so the readout runs on any metric in qnm.metrics.

DERIVATION (scalar field, ds^2 = -f dt^2 + dr^2/f + r^2 dOmega^2, M = 1)
-----------------------------------------------------------------------
Compactify sigma = r_h / r in [0, 1]  (0 = future null infinity, 1 = horizon).
Tortoise derivative and potential:

    x'(sigma) = -r_h / (sigma^2 f) ,     V = f ( l(l+1)/r^2 + f_r / r ) .

MINIMAL GAUGE: the height function removes exactly the singular parts of x' at
both ends,

    h' = S_0(sigma) - S_1(sigma)
    S_0 = -r_h/sigma^2 - c1 r_h/sigma      c1 = d(1/f)/dsigma at sigma = 0
    S_1 = -1 / (f_r(r_h) (1 - sigma))      (surface-gravity term)

For any asymptotically flat f = 1 - 2M/r + O(1/r^2), c1 = 2M/r_h.  Then, with
the sign convention of hyperboloidal.py (all flipped so W > 0):

    p  = -1/x'  = sigma^2 f / r_h
    g  = -h'/x' = h' sigma^2 f / r_h
    W  = (1 - g^2) / p
    q  = -x' V  = l(l+1)/r_h + f_r(r)/sigma

and the wave equation is  W u_tt + 2 g u_ts + g' u_t = (p u_s)_s - q u, exactly
as for Schwarzschild.  The minimal-gauge identity g^2 + W p = 1 holds by
construction; the SELF-CHECKS that matter are that W, p, q come out REGULAR and
POSITIVE on [0,1] (they are evaluated at the endpoints by symbolic limits) and
that Schwarzschild is reproduced exactly (tested).

Validity: f must be analytic on [r_h, infinity) in sigma.  For KS,
f = sqrt(r^2 - a^2)/r - 2M/r, r_h = sqrt(4M^2 + a^2) > a, so
sqrt(r_h^2 - a^2 sigma^2)/r_h is analytic on [0,1] and spectral convergence is
retained.
"""
from functools import lru_cache

import numpy as np
import sympy as sp
from scipy.linalg import sqrtm

from . import metrics
from .hyperboloidal import cheb_lobatto
from .rayleigh import clenshaw_curtis_weights

_s = sp.Symbol("sigma", positive=True)

MODELS = {        # each takes the model's single deformation parameter (a, Q or l)
    "schwarzschild": lambda a=0.0: metrics.schwarzschild(1),
    "kazakov_solodukhin": lambda a=0.0: metrics.kazakov_solodukhin(1, sp.nsimplify(a)),
    "reissner_nordstrom": lambda a=0.0: metrics.reissner_nordstrom(1, sp.nsimplify(a)),
    "hayward": lambda a=0.0: metrics.hayward(1, sp.nsimplify(a)),
}

# Published scalar l=2 fundamentals (M w), used for validation only.
REFERENCE = {
    ("schwarzschild", 0.0): 0.483643872211 - 0.096758775978j,
    ("kazakov_solodukhin", 0.2): 0.482034 - 0.096654j,
    ("kazakov_solodukhin", 0.5): 0.473887 - 0.096109j,
    ("kazakov_solodukhin", 1.0): 0.448362 - 0.094248j,
}


SYMBOLIC = ("schwarzschild", "kazakov_solodukhin")     # validated path; others numeric


@lru_cache(maxsize=None)
def gauge_functions(model="schwarzschild", a=0.0, ell=2):
    """
    (W, p, g, g', q) for a metric, returned as numpy callables that are exact at
    the endpoints.  Cached: the work is done once per metric.
    Schwarzschild and Kazakov-Solodukhin use exact symbolic limits; other metrics
    use the high-precision numeric path (_gauge_numeric), which needs no symbolic
    simplification and handles numerically known horizons.
    """
    if model not in SYMBOLIC:
        return _gauge_numeric(model, a, ell)
    m = MODELS[model](a=a)
    r = metrics.r
    f_r = sp.sympify(m["f"])
    r_h = sp.sympify(m["r_h"])
    fs = sp.simplify(f_r.subs(r, r_h / _s))                       # f(sigma)
    dfdr = sp.diff(f_r, r)
    kappa2 = sp.simplify(dfdr.subs(r, r_h))                        # f_r(r_h) = 2 kappa
    c1 = sp.limit(sp.diff(1 / fs, _s), _s, 0)

    xp = -r_h / (_s ** 2 * fs)
    hp = (-r_h / _s ** 2 - c1 * r_h / _s) + 1 / (kappa2 * (1 - _s))
    p = _s ** 2 * fs / r_h
    g = sp.simplify(hp * _s ** 2 * fs / r_h)
    W = sp.simplify((1 - g ** 2) / p)
    q = sp.simplify(ell * (ell + 1) / r_h + dfdr.subs(r, r_h / _s) / _s)
    gp = sp.diff(g, _s)

    def make(expr):
        fn = sp.lambdify(_s, expr, "numpy")
        ends = {0.0: complex(sp.limit(expr, _s, 0, "+")),
                1.0: complex(sp.limit(expr, _s, 1, "-"))}
        for k, v in ends.items():
            if not np.isfinite(v) or abs(v.imag) > 1e-12:
                raise RuntimeError(f"{model} a={a}: gauge function not regular at sigma={k}")

        def ev(sig):
            sig = np.asarray(sig, dtype=float)
            out = np.empty_like(sig)
            for i, sv in enumerate(sig):
                if abs(sv) < 1e-14:
                    out[i] = ends[0.0].real
                elif abs(sv - 1) < 1e-14:
                    out[i] = ends[1.0].real
                else:
                    out[i] = float(fn(sv))
            return out
        return ev

    return {"W": make(W), "p": make(p), "g": make(g), "gp": make(gp), "q": make(q),
            "r_h": float(r_h), "name": m["name"]}


def _gauge_numeric(model, a, ell, dps=60):
    """
    The same minimal-gauge functions evaluated in `dps`-digit arithmetic.

    In closed form, with F = f(r_h/sigma), F_r = f'(r_h/sigma), kappa2 = f'(r_h),
    c1 = 2M/r_h (asymptotic flatness, M = 1):
        p = sigma^2 F / r_h
        g = -F - c1 sigma F + sigma^2 F / (r_h kappa2 (1 - sigma))
        W = (1 - g^2) / p ,   q = l(l+1)/r_h + F_r / sigma ,   g' = dg/dsigma.
    Endpoint values (0/0 at sigma = 0 for W and q, at sigma = 1 for g and W) are taken
    one high-precision step inside the interval, where the removable singularity
    costs ~dps/2 digits -- far more than double precision needs.
    """
    import mpmath as mp
    mp.mp.dps = dps
    m = MODELS[model](a=a)
    r = metrics.r
    f_r = sp.sympify(m["f"])
    f = sp.lambdify(r, f_r, "mpmath")
    # a horizon known only to double precision (e.g. Hayward's cubic root) leaves
    # f(r_h) ~ 1e-16, which corrupts the horizon endpoint: refine it to dps digits
    r_h = mp.findroot(f, mp.mpf(str(sp.N(m["r_h"], dps + 10))))
    df = sp.lambdify(r, sp.diff(f_r, r), "mpmath")
    kappa2 = df(r_h)
    c1 = 2 / r_h

    def F(s):
        return f(r_h / s)

    def g(s):
        Fs = F(s)
        return -Fs - c1 * s * Fs + s ** 2 * Fs / (r_h * kappa2 * (1 - s))

    funcs = {
        "p": lambda s: s ** 2 * F(s) / r_h,
        "g": g,
        "W": lambda s: (1 - g(s) ** 2) / (s ** 2 * F(s) / r_h),
        "q": lambda s: ell * (ell + 1) / r_h + df(r_h / s) / s,
        "gp": lambda s: mp.diff(g, s),
    }
    tiny = mp.mpf(10) ** (-dps // 4)          # (1 - g^2) ~ tiny^2: keeps ~dps/2 digits

    def make(fn):
        ends = {0.0: fn(tiny), 1.0: fn(1 - tiny)}
        for k, v in ends.items():
            if not mp.isfinite(v):
                raise RuntimeError(f"{model} a={a}: gauge function not regular at sigma={k}")

        def ev(sig):
            sig = np.asarray(sig, dtype=float)
            out = np.empty_like(sig)
            for i, sv in enumerate(sig):
                if abs(sv) < 1e-14:
                    out[i] = float(ends[0.0])
                elif abs(sv - 1) < 1e-14:
                    out[i] = float(ends[1.0])
                else:
                    out[i] = float(fn(mp.mpf(float(sv))))
            return out
        return ev

    return {k: make(v) for k, v in funcs.items()} | {"r_h": float(r_h), "name": m["name"]}


def coefficients(sigma, model="schwarzschild", a=0.0, ell=2):
    G = gauge_functions(model, float(a), ell)
    W, p, g, gp, q = (G[k](sigma) for k in ("W", "p", "g", "gp", "q"))
    if not np.allclose(g ** 2 + W * p, 1.0, atol=1e-9):
        raise RuntimeError("minimal-gauge identity g^2 + W p = 1 violated")
    if W.min() <= 0 or p.min() < -1e-14 or q.min() <= 0:
        raise RuntimeError("W, q must be > 0 and p >= 0 on [0,1] for the energy norm")
    return W, p, g, gp, q


def operator(n_pts, ell=2, model="schwarzschild", a=0.0):
    """Evolution generator A, d_tau (psi, psi_dot) = A (psi, psi_dot).  w = i eig(A)."""
    sigma, D = cheb_lobatto(n_pts)
    W, p, g, gp, q = coefficients(sigma, model, a, ell)
    n = n_pts + 1
    Winv = np.diag(1.0 / W)
    L1 = Winv @ (D @ np.diag(p) @ D - np.diag(q))
    L2 = -Winv @ (2.0 * np.diag(g) @ D + np.diag(gp))
    return np.block([[np.zeros((n, n)), np.eye(n)], [L1, L2]]), sigma


def energy_gram(n_pts, ell=2, model="schwarzschild", a=0.0):
    sigma, D = cheb_lobatto(n_pts)
    W, p, g, gp, q = coefficients(sigma, model, a, ell)
    wq = clenshaw_curtis_weights(n_pts)
    n = n_pts + 1
    G11 = D.T @ np.diag(wq * p) @ D + np.diag(wq * q)
    G = np.block([[G11, np.zeros((n, n))], [np.zeros((n, n)), np.diag(wq * W)]])
    G = (G + G.T) / 2.0
    if np.linalg.eigvalsh(G).min() <= 0:
        raise RuntimeError("energy Gram matrix is not positive definite")
    return G


def energy_normed_operator(n_pts, ell=2, model="schwarzschild", a=0.0):
    """G^{1/2} A G^{-1/2}: same spectrum, better-conditioned readout."""
    A, _ = operator(n_pts, ell, model, a)
    Gh = np.real(sqrtm(energy_gram(n_pts, ell, model, a)))
    return Gh @ A @ np.linalg.inv(Gh)


def fundamental(n_pts, ell=2, model="schwarzschild", a=0.0, near=0.48 - 0.097j):
    w = 1j * np.linalg.eigvals(operator(n_pts, ell, model, a)[0])
    w = w[np.isfinite(w) & (w.real > 0) & (w.imag < 0)]
    return w[np.argmin(np.abs(w - near))]


def qubits_to_npts(qubits):
    """Operator dimension 2(n_pts+1) = 2**qubits."""
    return 2 ** (qubits - 1) - 1
