"""
General P(w) builder for ANY lapse f(r) -- the "many black-hole models" axis.

The tridiagonal Leaver matrix in pencil.py is specific to Schwarzschild.  For a
generic metric (Kazakov-Solodukhin, Reissner-Nordstrom, Hayward, ...) we build
P(w) from a Frobenius expansion of the peeled master equation.

Method
------
Compact coordinate  s = 1 - r_h/r  (s = 0 horizon, s = 1 infinity).  Peel the
QNM asymptotics  Psi = e^{i w r} r^{2 i M w} s^{-i w/f'(r_h)} u(s)  and expand
u(s) = sum a_n s^n.  Substituting into  f^2 Psi'' + f f' Psi' + (w^2 - V)Psi = 0
gives, after clearing the horizon poles (multiply by s^2), a banded linear
system whose matrix P(w) is singular exactly at QNM frequencies.

Why this implementation is fully numeric
----------------------------------------
Earlier versions extracted the s-Taylor coefficients with sympy `expand` +
`series`.  Under sympy 1.14 those steps silently produce nan for lapses
containing sqrt (Kazakov-Solodukhin): Float-heavy expansions of the pole terms
generate 0*zoo artifacts.  Here sympy is used ONLY to lambdify the three simple
radial functions f(r), f'(r), V(r); everything else is closed-form numpy.

The peeling factor's logarithmic derivative is linear in w with a known shape:

    G1(s, w)      = w * g1(s),   g1 = i [ r' + 2 M r'/r - 1/(f'(r_h) s) ]
    G2 = G1'+G1^2 = w * g1'(s) + w^2 * g1(s)^2

so the three ODE coefficients, times s^2, split EXACTLY by powers of w:

    b2 : w^0  F2                        with F2 = f^2 (ds/dr)^2 s^2
    b1 : w^0  B s^2                          B  = f^2 d2s/dr2 + f f' ds/dr
         w^1  2 F2 g1
    b0 : w^0  -V s^2
         w^1  F2 g1' + B s^2 g1
         w^2  F2 g1^2 + s^2

Each of these functions of s is analytic in a disk around s = 0 (the 1/s,
1/s^2 poles are cleared by the explicit s^2 factors; the nearest true
singularity is the compactification point s = 1, and the metric's sqrt branch
point maps outside |s| < 1 for all KS deformations).  Their Taylor
coefficients are obtained numerically by the Cauchy integral evaluated with an
FFT on the circle |s| = rho:

    t_k = (1/(M rho^k)) sum_m F(rho e^{2 pi i m/M}) e^{-2 pi i k m/M}.

With rho = 0.35 and M = 512 sample points this is accurate to near machine
precision for the orders used here, independent of how gnarly f(r) is.

The pencil is exactly quadratic in w by construction.

A construction-time self-check compares the FFT-Taylor reconstruction of each
coefficient function against its direct numerical evaluation at interior
points, so a bad expansion fails loudly instead of silently.
"""

import numpy as np
import sympy as sp


class RecurrencePencil:
    """Callable P(w) for a given metric / multipole, built once and reused."""

    def __init__(self, f_expr, r_h, M, ell, spin="scalar", n_taylor=30,
                 rho=0.35, n_fft=512):
        r = None
        for sym in f_expr.free_symbols:
            if sym.name == "r":
                r = sym
        if r is None:
            r = sp.Symbol("r")
        self.n_taylor = n_taylor
        r_h = float(r_h)
        M = float(M)

        f = f_expr
        fp = sp.diff(f, r)
        fp_h = complex(fp.subs(r, r_h))
        if spin == "scalar":
            V = f * (ell * (ell + 1) / r**2 + fp / r)
        elif spin == "axial":
            V = f * (ell * (ell + 1) / r**2 - 6 * M / r**3)
        else:
            raise ValueError("spin must be 'scalar' or 'axial'")

        # the ONLY sympy->numpy step: three simple radial functions
        f_n = sp.lambdify(r, f, "numpy")
        fp_n = sp.lambdify(r, fp, "numpy")
        V_n = sp.lambdify(r, V, "numpy")

        # closed-form numpy assembly of the coefficient functions of s
        def pieces(s):
            s = np.asarray(s, dtype=complex)
            one = 1.0 - s
            r_s = r_h / one                     # r(s)
            rp = r_h / one**2                   # dr/ds
            rpp = 2.0 * r_h / one**3            # d2r/ds2
            fs = np.asarray(f_n(r_s), dtype=complex)
            fps = np.asarray(fp_n(r_s), dtype=complex)
            Vs = np.asarray(V_n(r_s), dtype=complex)
            spr = one**2 / r_h                  # ds/dr = r_h / r^2
            sppr = -2.0 * one**3 / r_h**2       # d2s/dr2 = -2 r_h / r^3
            F2 = fs**2 * spr**2 * s**2
            B = fs**2 * sppr + fs * fps * spr
            g1 = 1j * (rp + 2.0 * M * rp / r_s) - (1j / fp_h) / s
            g1p = 1j * (rpp + 2.0 * M * (rpp / r_s - (rp / r_s) ** 2)) \
                + (1j / fp_h) / s**2
            return {
                ("b2", 0): F2,
                ("b1", 0): B * s**2,
                ("b1", 1): 2.0 * F2 * g1,
                ("b0", 0): -Vs * s**2,
                ("b0", 1): F2 * g1p + B * s**2 * g1,
                ("b0", 2): F2 * g1**2 + s**2,
            }

        # Cauchy/FFT Taylor coefficients on |s| = rho
        theta = 2.0 * np.pi * np.arange(n_fft) / n_fft
        ring = rho * np.exp(1j * theta)
        vals = pieces(ring)
        ks = np.arange(n_taylor + 1)
        self._t = {name: np.zeros((3, n_taylor + 1), dtype=complex)
                   for name in ("b2", "b1", "b0")}
        for (name, m), arr in vals.items():
            fft = np.fft.fft(np.asarray(arr, dtype=complex)) / n_fft
            self._t[name][m, :] = fft[: n_taylor + 1] / rho**ks

        # self-check: reconstruction vs direct evaluation inside the disk
        probes = np.array([0.08 + 0.03j, 0.15 - 0.05j, 0.21 + 0.07j])
        direct = pieces(probes)
        powers = probes[None, :] ** ks[:, None]          # (n_taylor+1, npts)
        for (name, m), arr in direct.items():
            recon = self._t[name][m, :] @ powers
            scale = np.maximum(np.abs(arr), 1.0)
            err = np.max(np.abs(recon - arr) / scale)
            if err > 1e-8:
                raise RuntimeError(
                    f"Taylor self-check failed for {name}[w^{m}] (err {err:.2e});"
                    " increase n_taylor/n_fft or check the metric expression")

    # ------------------------------------------------------------------
    def _bands(self, w_val):
        w = complex(w_val)
        out = {}
        for name, t in self._t.items():
            out[name] = t[0] + w * t[1] + w * w * t[2]
        return out["b2"], out["b1"], out["b0"]

    def matrix(self, w_val, N):
        """Assemble the N x N banded pencil P(w_val).  Exactly quadratic in w."""
        b2, b1, b0 = self._bands(w_val)
        rows = N + 6
        big = np.zeros((rows, N), dtype=complex)
        for m in range(rows):
            for n in range(N):
                v = 0j
                if 0 <= m - n + 2 < len(b2):
                    v += b2[m - n + 2] * n * (n - 1)
                if 0 <= m - n + 1 < len(b1):
                    v += b1[m - n + 1] * n
                if 0 <= m - n < len(b0):
                    v += b0[m - n]
                big[m, n] = v
        nz = np.where(np.abs(big).sum(1) > 1e-12)[0][0]
        return big[nz:nz + N, :]

    def sigma_min(self, w_val, N):
        from numpy.linalg import svd
        return svd(self.matrix(w_val, N), compute_uv=False)[-1]
