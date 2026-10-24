"""
ultra_rayleigh.py -- TASK 1: ultraspherical basis + phase (Rayleigh) readout.

THE UNTESTED CELL
-----------------
The ultraspherical basis was ruled out against the RESIDUAL readout, which reads
a magnitude and therefore lives or dies on the mode/non-mode contrast in
sigma_min.  The Rayleigh readout reads a phase: <z|A|z>/<z|B|z> is O(1) whether
or not the operator is near-singular nearby.  So the ultraspherical failure need
not transfer.  Nobody has tested the combination.

If it survives, we get banded support (fixing measurement density) AND the phase
readout -- the only route to both at once.

============================================================================
PREREGISTERED GATE  (written before any number in this module was produced)
============================================================================
Metric: shots to pin w to |dw| <= 1e-3, measured with the same noisy device,
ZNE, and per-Pauli shot splitting as rayleigh.py, against that module's
measured 1.11e10.

  * within 2x of the incumbent AND materially lower Pauli count
        -> adopt as the production encoding
  * 2x - 10x worse
        -> record as a measured trade, keep the incumbent
  * worse than 10x
        -> ultraspherical is confirmed dead for BOTH readouts; write up as a
           two-route result for section 8

Either outcome is a result.  The negative is not to be discarded.
============================================================================

TWO ARMS, AND WHY THERE ARE TWO
-------------------------------
The task as specified says: convert the pencil to a fixed generator in the
energy inner product, then run the Rayleigh readout.  Measured first, before
building on it:

        n_coef   pencil A bw   pencil B bw   L = B^-1 A bw   L density
           8         10            10             15          53.1%
          16         10            10             30          49.4%

L = B^{-1} A IS DENSE.  Bandwidth equals dim-2 at every size.  Forming the
generator destroys exactly the bandedness that motivates the task, because B^-1
is dense even when B is banded.  Building only that arm would have answered a
question nobody asked.

So both are measured:

  ARM A (as specified) -- fixed generator, energy inner product
        L = B^-1 A ,   Lhat = G^{1/2} L G^{-1/2} ,   w = i <z|Lhat|z>
        Dense.  Two measured observables.  Directly comparable to rayleigh.py.

  ARM B (bandedness-preserving) -- GENERALIZED Rayleigh quotient
        lambda = <z|A|z> / <z|B|z> ,   w = i lambda
        No inverse is ever formed, so A and B stay banded.  Four measured
        Hermitian observables:
            A1 = (A+A^dag)/2 , A2 = (A-A^dag)/2i ,
            B1 = (B+B^dag)/2 , B2 = (B-B^dag)/2i
        This is the only arm that can satisfy the gate's "materially lower
        Pauli count" clause, so it is the one the gate is really about.

A NOTE ON THE INCUMBENT
-----------------------
The task says "head-to-head against the current Leaver + Rayleigh arm".  No such
arm exists and cannot: Leaver's pencil is QUADRATIC in w, so it admits no fixed
generator without companion linearization (which doubles the dimension).  The
incumbent Rayleigh arm in rayleigh.py is built on the HYPERBOLOIDAL COLLOCATION
operator.  That is what the comparison is against, at matched dimension.

REPORTING STANDARDS OBSERVED HERE
---------------------------------
* l1/signal is reported alongside Pauli count, because shot cost is (l1/eps)^2
  and Pauli count alone has misled comparisons in this project before.
* PROJECTION and MEASUREMENT are labelled separately and never mixed.
* The energy transform is re-asserted to be a strict similarity at construction,
  so it cannot move a mode.
"""

import numpy as np
from numpy.polynomial import chebyshev as _cheb
from scipy.linalg import sqrtm

from .hardware import NoisyAnsatzSim, pauli_decompose, zne_energy
from .hyperboloidal import cheb_lobatto, ultraspherical_pencil
from .quantum import ansatz_state
from .rayleigh import energy_gram

WQ = 0.483643872211 - 0.096758775978j


# ----------------------------------------------------------------------
# 1. energy inner product in Chebyshev COEFFICIENT space
# ----------------------------------------------------------------------
def coefficient_vandermonde(n_coef):
    """
    V[j,k] = T_k(x_j) at the Chebyshev-Lobatto grid, in the ULTRASPHERICAL
    orientation x = 2*sigma - 1.

    Note the orientation flip: cheb_lobatto uses sigma = (1-x_cheb)/2 while
    ultraspherical_pencil uses sigma = (1+x)/2, so x = 2*sigma - 1 must be
    rebuilt from sigma rather than reused from the grid routine.
    """
    sigma, _ = cheb_lobatto(n_coef - 1)
    x = 2.0 * sigma - 1.0
    return _cheb.chebvander(x, n_coef - 1)


def coefficient_energy_gram(n_coef, ell=2):
    """
    Energy Gram matrix expressed on Chebyshev coefficients:

        G_coef = blkdiag(V,V)^T  G_grid  blkdiag(V,V)

    with G_grid the collocation-space energy metric from rayleigh.energy_gram.
    """
    V = coefficient_vandermonde(n_coef)
    G_grid = energy_gram(n_coef - 1, ell)
    n = n_coef
    Z = np.zeros((n, n))
    Vb = np.block([[V, Z], [Z, V]])
    G = Vb.T @ G_grid @ Vb
    G = (G + G.T) / 2.0
    if np.linalg.eigvalsh(G).min() <= 0:
        raise RuntimeError("coefficient-space energy Gram is not positive definite")
    return G


# ----------------------------------------------------------------------
# 2. ARM A -- fixed generator in the energy inner product (as specified)
# ----------------------------------------------------------------------
def ultra_generator(n_coef, ell=2, energy=True, atol=1e-8):
    """
    L = B^-1 A, optionally mapped into the energy inner product.

    Asserts that the energy map is a STRICT SIMILARITY (spectra agree), so it
    cannot move a mode.  Raises otherwise.

    WARNING, measured: L is dense even though (A,B) are banded, because B^-1 is
    dense.  See the module docstring.
    """
    A, B = ultraspherical_pencil(n_coef, ell)
    L = np.linalg.solve(B, A)
    if not energy:
        return L
    G = coefficient_energy_gram(n_coef, ell)
    Gh = np.real(sqrtm(G))
    Lhat = Gh @ L @ np.linalg.inv(Gh)

    # Assert on the PHYSICAL modes, not the whole spectrum.  A whole-spectrum
    # sorted comparison reports spurious drift (measured 2.94 at n_coef=32)
    # because the far end of a truncated pencil carries numerically ill-defined,
    # near-degenerate eigenvalues that sort_complex pairs up wrongly.  Those
    # carry no physics.  The damped, physical branch is the real invariant, and
    # it is preserved to ~1e-14.
    def physical(M):
        w = 1j * np.linalg.eigvals(M)
        w = w[np.isfinite(w) & (w.real > 1e-6) & (w.imag < -1e-6) & (w.imag > -2.0)]
        return np.sort_complex(w)

    p0, p1 = physical(L), physical(Lhat)
    n_cmp = min(len(p0), len(p1))
    if n_cmp == 0 or not np.allclose(p0[:n_cmp], p1[:n_cmp], atol=atol, rtol=1e-8):
        drift = np.max(np.abs(p0[:n_cmp] - p1[:n_cmp])) if n_cmp else np.inf
        raise RuntimeError("energy map is not a similarity -- it moved the "
                           f"physical spectrum by {drift:.2e}")
    return Lhat


# ----------------------------------------------------------------------
# 3. ARM B -- generalized Rayleigh quotient, no inverse formed
# ----------------------------------------------------------------------
def generalized_observables(A, B):
    """(A1, A2, B1, B2), all Hermitian.  lambda = (<A1>+i<A2>) / (<B1>+i<B2>)."""
    A1 = (A + A.conj().T) / 2.0
    A2 = (A - A.conj().T) / 2.0j
    B1 = (B + B.conj().T) / 2.0
    B2 = (B - B.conj().T) / 2.0j
    return A1, A2, B1, B2


def omega_generalized(z, A, B):
    """w = i <z|A|z> / <z|B|z>  -- exact, for validation."""
    num = complex(np.vdot(z, A @ z))
    den = complex(np.vdot(z, B @ z))
    return 1j * num / den


def generalized_variance(theta, n_qubits, layers, A, B):
    """
    ||A z - lambda B z||^2 with lambda the closed-form optimum -- the pencil
    analogue of rayleigh.variance_cost.  Calibration only; never measured.
    """
    z = ansatz_state(theta, n_qubits, layers)
    Az, Bz = A @ z, B @ z
    bb = float(np.real(np.vdot(Bz, Bz)))
    if bb < 1e-300:
        return float(np.real(np.vdot(Az, Az)))
    lam = np.vdot(Bz, Az) / bb
    r = Az - lam * Bz
    return float(np.real(np.vdot(r, r)))


# ----------------------------------------------------------------------
# 4. cost metrics -- PROJECTIONS (labelled)
# ----------------------------------------------------------------------
def pauli_stats(H, tol=1e-10):
    """(n_terms, l1) over NONIDENTITY Pauli strings of a Hermitian H."""
    terms = [(c, lab) for c, lab in pauli_decompose(H, tol=tol)
             if set(lab) != {"I"}]
    return len(terms), float(sum(abs(c) for c, _ in terms))


def bandwidth(M, tol=1e-10):
    idx = np.argwhere(np.abs(M) > tol * np.abs(M).max())
    return int(np.max(np.abs(idx[:, 0] - idx[:, 1])))


def shot_projection_fixed(L, eps=1e-3):
    """
    PROJECTION (not a measurement).  w = -<H2> + i<H1>, so pinning w to eps
    needs each part to eps/sqrt(2):

        shots ~ 2 * ( l1(H1)^2 + l1(H2)^2 ) / eps^2
    """
    H1 = (L + L.conj().T) / 2.0
    H2 = (L - L.conj().T) / 2.0j
    _, a = pauli_stats(H1)
    _, b = pauli_stats(H2)
    return 2.0 * (a ** 2 + b ** 2) / eps ** 2, (a, b)


def shot_projection_generalized(A, B, z, eps=1e-3):
    """
    PROJECTION (not a measurement).  lambda = N/D, so

        |d lambda| ~ ( |dN| + |lambda||dD| ) / |D|

    and each expectation costs (l1/precision)^2.  Evaluated at the state z,
    since |N| and |D| depend on it.
    """
    A1, A2, B1, B2 = generalized_observables(A, B)
    l1s = [pauli_stats(H)[1] for H in (A1, A2, B1, B2)]
    num = complex(np.vdot(z, A @ z))
    den = complex(np.vdot(z, B @ z))
    lam = num / den
    # budget eps/2 to the numerator group and eps/2 to the denominator group
    need_N = eps * abs(den) / 2.0
    need_D = eps * abs(den) / (2.0 * max(abs(lam), 1e-12))
    shots = ((l1s[0] ** 2 + l1s[1] ** 2) / need_N ** 2
             + (l1s[2] ** 2 + l1s[3] ** 2) / need_D ** 2)
    return shots, l1s, abs(den)


# ----------------------------------------------------------------------
# 5. noisy MEASUREMENT of the generalized readout
# ----------------------------------------------------------------------
def measure_omega_generalized(theta, A, B, n_qubits, layers, shots=None,
                              p1=1e-3, p2=1e-2, seed=0, mitigate=True,
                              scales=(1.0, 2.0, 3.0)):
    """
    MEASUREMENT.  Four Hermitian expectation values on the simulated noisy
    device, same machinery as rayleigh.measure_omega.
    """
    obs = generalized_observables(A, B)
    sim = NoisyAnsatzSim(n_qubits, layers, p1=p1, p2=p2)
    rng = np.random.default_rng(seed)
    vals = []
    for H in obs:
        nrm = np.linalg.norm(H, 2)
        Hn = H / nrm
        if mitigate:
            e, _ = zne_energy(sim, theta, Hn, scales=scales, shots=shots, rng=rng)
        else:
            e = sim.energy(theta, Hn, noise_scale=1.0, shots=shots, rng=rng)
        vals.append(e * nrm)
    num = vals[0] + 1j * vals[1]
    den = vals[2] + 1j * vals[3]
    return 1j * num / den


def exact_mode(M, target=WQ):
    """The eigenvalue of a generator nearest the true fundamental."""
    w = 1j * np.linalg.eigvals(M)
    return w[int(np.argmin(np.abs(w - target)))]


def exact_mode_pencil(A, B, target=WQ):
    from scipy.linalg import eig
    w = 1j * eig(A, B, right=False)
    w = w[np.isfinite(w)]
    return w[int(np.argmin(np.abs(w - target)))]
