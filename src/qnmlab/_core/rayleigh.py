"""
rayleigh.py -- variational QNM solver with PHASE readout instead of MAGNITUDE
readout.  This is the fix for the shot-cost wall.

THE WALL, RESTATED
------------------
The residual and dilation encodings both answer "how small is sigma_min at this
trial w?".  The answer is near zero by construction, and shot noise on any
expectation value is ~1/sqrt(S) no matter how small the truth is.  So resolving
a signal of size eps costs ~1/eps^2 shots, and the residual squares it to
~1/eps^4.  run_hardware_demo.py measures the consequence: ~1e19 shots for the
residual, ~1e12 for the dilation.

No change of MATRIX fixes this -- we confirmed that twice, by two unrelated
mechanisms (see hyperboloidal.py).  The fix has to change WHAT IS MEASURED.

THE FIX
-------
On hyperboloidal slices the problem is a genuine linear generator,

        d_tau u = A u ,      w = i * eig(A) ,

so w can be read from a RAYLEIGH QUOTIENT rather than from a vanishing residual:

        lambda = <z|A|z> / <z|z> ,       w = i * lambda .

Split A into its Hermitian and anti-Hermitian parts,

        H1 = (A + A^dag)/2 ,     H2 = (A - A^dag)/(2i)   (both Hermitian)

and then, for a normalized ansatz state,

        Re w = -<H2> ,      Im w = <H1> .

BOTH ARE O(1) EXPECTATION VALUES OF HERMITIAN OPERATORS.  The device measures
two observables whose values are order 0.1-0.5, not order 1e-14.  Frequency
precision costs ~1/eps^2 with an O(1) prefactor instead of a sigma-suppressed
one.  Measured benefit: 125x fewer shots than the residual encoding on the
identical task (see MEASURED RESULT -- and read the currency warning there
before quoting any larger number).

Note what the device does NOT do: it never measures the vanishing quantity.
The variance

        Var(theta) = <A^dag A> - |<A>|^2

is only used to CALIBRATE the state, and per this project's standing hardware
protocol (calibrate classically, evaluate on device -- see hardware.vqe_noisy)
that happens on the simulator.  Var only ever needs to DECREASE.  It never needs
to be RESOLVED.  That distinction is the whole method.

THE ENERGY INNER PRODUCT
------------------------
A is non-normal (energy genuinely leaks through the horizon and out to scri --
this is physics, not discretization), so a Rayleigh readout amplifies
state-preparation error by the eigenvalue condition number.  Working in the
ENERGY scalar product rather than the naive L2 one tames it.  With

    <u,v>_E = u^dag G v ,    G = [[ D^T diag(wq p) D + diag(wq q) , 0 ],
                                 [ 0 , diag(wq W) ]]

(wq = Clenshaw-Curtis quadrature weights) the transformed operator
Ahat = G^{1/2} A G^{-1/2} is similar to A, so the spectrum is untouched, but
its eigenvalue conditioning improves and the gain GROWS with resolution:

       dim    cond(L2)    cond(energy)   gain
        16      5.28          3.12       1.7x
        32     15.30          5.93       2.6x
        64     65.60         15.80       4.1x

This is the scalar product of Jaramillo-Macedo-Al Sheikh (PRX 11, 031003), the
one their pseudospectrum analysis is built in.  It cannot remove the
non-normality -- nothing can, it is dissipative physics -- it concentrates it in
the physical boundary terms.

MEASURED RESULT (run_rayleigh_demo.py, same noisy device and ZNE as the baseline)
--------------------------------------------------------------------------------
Setup validated first: re-deriving the baseline from scratch reproduces
run_hardware_demo.py exactly -- 3.37e19 shots to resolve sigma_min^2 to 10%,
against the published ~1e19.

BEWARE THE CURRENCY.  That 1e19 answers "resolve sigma_min^2 to 10%", which is
NOT the same task as "pin w to 1e-3".  Comparing across the two would inflate
the result by orders of magnitude.  Inside ONE currency -- shots to pin w to
|dw| <= 1e-3, dim 16 / 4 qubits, l=2:

        residual encoding     1.43e12 shots
        Rayleigh readout      1.14e10 shots
        advantage                125x

The Rayleigh spread is cleanly shot-noise-limited, which is what licenses the
extrapolation (identical extrapolation style to the baseline's):

        shots     spread of w      implied shots for 1e-3
        1e5        3.377e-01              1.14e10
        1e6        1.067e-01              1.14e10
        1e7        3.376e-02              1.14e10

Exactly 1/sqrt(S) across three decades, and the implied cost is stable.

An earlier hand estimate of this module's cost (kappa*||A||/sqrt(S)) predicted
~5e9; measured 1.14e10, so that part was sound within 2x.  What that estimate
got wrong was the COMPARISON: it set 1.14e10 against 1e19 rather than against
the residual arm evaluated on the same task.  125x is the honest figure.  The
gap is not nine orders of magnitude.

Where the rest of the naive advantage goes: the shot budget is split across 255
Pauli strings, so the measured spread is ~8x worse than a single-observable
estimate would suggest.  Ahat is dense.  Sparsity still pays here.

NOT INCLUDED, and it favours this method: the residual arm must SCAN w, paying
its per-point cost at every refinement step, while the Rayleigh arm reads w from
a single calibrated state with no scan at all.  We do not quote a multiplier for
this because the early scan points need less precision than the late ones, and
we have not measured the true profile.  The 125x is therefore a floor.

HONEST LIMITS
-------------
* STATE PREPARATION IS THE BINDING CONSTRAINT, NOT SHOT NOISE.  Measured
  state-prep systematic: |dw| ~ 9.0e-3 at 3 layers, best 1.5e-3 over 8 seeds at
  5 layers -- both at or above the 1e-3 target the shot budget was sized for.
  So the 125x is a statement about STATISTICAL cost; the method is currently
  systematic-limited and the advantage is not yet realisable end to end.
  The ansatz is NOT the limitation (direct fidelity maximization reaches
  0.999999 at 3 layers); the variance landscape is.  Fixing the optimizer, not
  the circuit, is what would cash this in.

* Mode selection does not disappear, it RELOCATES.  Dropping the outer w-scan
  removes the mechanism that targeted a chosen mode; the initial state must
  target it instead.  Cold multistart lands on the mirror branch most of the
  time (nonhermitian.py measured 6/8); warm-starting from a lead fixed 7/8.
  qnmlab.search supplies that lead with no prior knowledge.
* The Pauli-term count still enters the shot cost, because the budget is split
  across terms.  Ahat is dense, so this partly offsets the O(1)-signal win.
  `shot_cost_comparison` accounts for it rather than assuming it away.
* Everything here is a simulated device, exactly like the baseline it is
  compared against.  No hardware run is claimed.
"""

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import minimize

from .hardware import NoisyAnsatzSim, pauli_decompose, zne_energy
from .hyperboloidal import cheb_lobatto, minimal_gauge, operator
from .quantum import ansatz_state


# ----------------------------------------------------------------------
# 1. the energy inner product
# ----------------------------------------------------------------------
def clenshaw_curtis_weights(n_pts):
    """Clenshaw-Curtis quadrature weights on the Chebyshev-Lobatto grid, [0,1]."""
    N = n_pts
    th = np.pi * np.arange(N + 1) / N
    w = np.zeros(N + 1)
    v = np.ones(N - 1)
    for k in range(2, N, 2):
        v = v - 2 * np.cos(k * th[1:N]) / (k * k - 1)
    if N % 2 == 0:
        w[0] = 1.0 / (N * N - 1)
        w[N] = w[0]
        v = v - np.cos(N * th[1:N]) / (N * N - 1)
    else:
        w[0] = 1.0 / (N * N)
        w[N] = w[0]
    w[1:N] = 2 * v / N
    return w / 2.0                      # map [-1,1] -> [0,1]


def energy_gram(n_pts, ell=2):
    """
    Gram matrix G of the energy scalar product on (psi, psi_dot).

        E = 1/2 integral ( p |psi'|^2 + q |psi|^2 + W |psi_dot|^2 ) dsigma

    Positive definite because p >= 0 and q, W > 0 on [0,1].
    """
    sigma, D = cheb_lobatto(n_pts)
    W, p, g, gp, q = minimal_gauge(sigma, ell)
    wq = clenshaw_curtis_weights(n_pts)
    n = n_pts + 1
    G11 = D.T @ np.diag(wq * p) @ D + np.diag(wq * q)
    G22 = np.diag(wq * W)
    Z = np.zeros((n, n))
    G = np.block([[G11, Z], [Z, G22]])
    G = (G + G.T) / 2.0
    if np.linalg.eigvalsh(G).min() <= 0:
        raise RuntimeError("energy Gram matrix is not positive definite")
    return G


def energy_normed_operator(n_pts, ell=2):
    """
    Ahat = G^{1/2} A G^{-1/2}.  Similar to A, so the spectrum -- and therefore
    every QNM -- is untouched; only the conditioning of the readout changes.
    """
    A, _ = operator(n_pts, ell)
    G = energy_gram(n_pts, ell)
    Gh = np.real(sqrtm(G))
    return Gh @ A @ np.linalg.inv(Gh)


def eigenvalue_condition(A, w_target=0.483643872211 - 0.096758775978j):
    """Condition number of the targeted eigenvalue: |u||v| / |<v,u>|."""
    ev, V = np.linalg.eig(A)
    k = int(np.argmin(np.abs(1j * ev - w_target)))
    Vl = np.linalg.inv(V).conj().T
    u, v = V[:, k], Vl[:, k]
    return float(np.linalg.norm(u) * np.linalg.norm(v) / abs(np.vdot(v, u)))


# ----------------------------------------------------------------------
# 2. the two measured observables
# ----------------------------------------------------------------------
def observables(A):
    """
    (H1, H2), both Hermitian, with  Re w = -<H2>  and  Im w = <H1>.

    These are the ONLY things the device ever measures.  Both are O(1).
    """
    H1 = (A + A.conj().T) / 2.0
    H2 = (A - A.conj().T) / 2.0j
    return H1, H2


def omega_from(e1, e2):
    """w = i * lambda with lambda = <H1> + i<H2>."""
    return -float(e2) + 1j * float(e1)


def variance_cost(theta, n_qubits, layers, A, AdA, mu=0.0, w_lead=0.0j):
    """
    Var = <A^dag A> - |<A>|^2 ,  optionally steered toward a lead frequency:

        Var(theta)  +  mu * |w(theta) - w_lead|^2 .

    Both terms are built from the SAME three expectation values the method
    already uses, so steering costs no extra observables.  It is how the relocated
    mode-selection problem is solved: the search supplies w_lead.

    Calibration-only: never measured on the device, so its small value at the
    optimum costs nothing.
    """
    z = ansatz_state(theta, n_qubits, layers)
    q = float(np.real(np.vdot(z, AdA @ z)))
    lam = complex(np.vdot(z, A @ z))
    cost = q - abs(lam) ** 2
    if mu:
        cost += mu * abs(1j * lam - w_lead) ** 2
    return cost


def calibrate(A, layers=3, restarts=12, seed=0, maxiter=2000, w_filter=None,
              w_lead=None, mu=1.0, polish=True):
    """
    Classical state calibration by variance minimization (project protocol:
    calibrate on the simulator, evaluate on the device).

    w_filter : optional callable(w)->bool keeping only physically admissible
        frequencies.  Necessary but NOT sufficient -- see below.
    w_lead   : optional coarse frequency estimate (from qnmlab.search, or a cheap
        classical one).  Basin-level accuracy (~1e-2) is enough.
    mu       : steering weight applied while w_lead is set.
    polish   : after steering, re-minimize the PURE variance from the steered
        point, so the reported frequency is not biased by the lead.

    Why the lead is required.  Measured at dim 16, l=2: cold multistart with only
    a physical-admissibility filter converges to w = 0.218 - 0.207j, which is not
    an eigenvalue at all -- a spurious local minimum of the variance landscape.
    The ansatz is NOT the limitation (3 layers reach fidelity 0.999999 to the
    true fundamental eigenvector); the landscape is.  This reproduces
    nonhermitian.py's finding that cold restarts land 6/8 on the mirror branch
    while warm starts find the fundamental 7/8.
    """
    n = A.shape[0]
    n_qubits = int(np.log2(n))
    if 2 ** n_qubits != n:
        raise ValueError(f"dimension {n} is not a power of two")
    AdA = A.conj().T @ A
    n_params = (layers + 1) * n_qubits * 2
    rng = np.random.default_rng(seed)
    best = {"var": np.inf, "theta": None, "w": None}
    for _ in range(restarts):
        x0 = rng.uniform(0, 2 * np.pi, n_params)
        args = (n_qubits, layers, A, AdA,
                (mu if w_lead is not None else 0.0),
                (complex(w_lead) if w_lead is not None else 0.0j))
        res = minimize(variance_cost, x0, args=args, method="L-BFGS-B",
                       options={"maxiter": maxiter})
        theta = res.x
        if w_lead is not None and polish:
            res = minimize(variance_cost, theta,
                           args=(n_qubits, layers, A, AdA, 0.0, 0.0j),
                           method="L-BFGS-B", options={"maxiter": maxiter})
            theta = res.x
        z = ansatz_state(theta, n_qubits, layers)
        w = 1j * complex(np.vdot(z, A @ z))
        if w_filter is not None and not w_filter(w):
            continue
        var = variance_cost(theta, n_qubits, layers, A, AdA)
        if var < best["var"]:
            best = {"var": float(var), "theta": theta, "w": w}
    best["n_qubits"] = n_qubits
    best["layers"] = layers
    return best


def qnm_filter(w):
    """Physically admissible: damped, positive real part."""
    return w.real > 0.05 and w.imag < -1e-3


# ----------------------------------------------------------------------
# 3. noisy device evaluation of the two observables
# ----------------------------------------------------------------------
def measure_omega(theta, A, n_qubits, layers, shots=None, p1=1e-3, p2=1e-2,
                  seed=0, mitigate=True, scales=(1.0, 2.0, 3.0)):
    """
    Read w on the simulated noisy device: two Hermitian expectation values,
    each Pauli-decomposed and shot-sampled, optionally ZNE-mitigated.

    Identical machinery to run_hardware_demo.py, so the comparison is fair.
    """
    H1, H2 = observables(A)
    sim = NoisyAnsatzSim(n_qubits, layers, p1=p1, p2=p2)
    rng = np.random.default_rng(seed)
    out = []
    for H in (H1, H2):
        nrm = np.linalg.norm(H, 2)
        Hn = H / nrm
        if mitigate:
            e, _ = zne_energy(sim, theta, Hn, scales=scales, shots=shots, rng=rng)
        else:
            e = sim.energy(theta, Hn, noise_scale=1.0, shots=shots, rng=rng)
        out.append(e * nrm)
    return omega_from(out[0], out[1])


def pauli_terms_measured(A):
    """Nonidentity Pauli strings the device must measure (H1 and H2 together)."""
    H1, H2 = observables(A)
    labs = set()
    for H in (H1, H2):
        for c, lab in pauli_decompose(H / np.linalg.norm(H, 2), tol=1e-10):
            if set(lab) != {"I"}:
                labs.add(lab)
    return len(labs)


# ----------------------------------------------------------------------
# 4. head-to-head against the run_hardware_demo.py baseline
# ----------------------------------------------------------------------
def residual_shot_cost(tol, N=8, l=2, w_qnm=0.483643872211 - 0.096758775978j,
                       std_at_ref=None, shots_ref=10 ** 8):
    """
    Shots the RESIDUAL encoding needs to pin w to `tol`, in the same currency.

    The residual reports sigma_min^2, so locating w to `tol` means telling
    sigma^2(w_qnm) apart from sigma^2(w_qnm + tol).

    NOTE the shape.  sigma_min has a V-shaped (not smooth) minimum -- measured
    sigma ~ 2.9 |dw| at N=8 -- so sigma^2 is QUADRATIC in the offset near the
    mode.  A derivative-based estimate 2*sigma_0*|dsigma/dw|*tol understates the
    residual arm by ~100x because the linear term is swamped by the quadratic
    one.  We therefore evaluate the gap directly, with no expansion:

        resolvable = sigma_min(w_qnm + tol)^2 - sigma_min(w_qnm)^2

    `std_at_ref` is the measured std of the ZNE residual estimate at
    `shots_ref`, as produced by run_hardware_demo.demo_2_encodings.
    """
    from .pencil import equilibrate, leaver_matrix_std, sigma_min
    s0 = sigma_min(equilibrate(leaver_matrix_std(w_qnm, l=l, N=N)))
    s1 = sigma_min(equilibrate(leaver_matrix_std(w_qnm + tol, l=l, N=N)))
    resolvable = abs(s1 ** 2 - s0 ** 2)
    return shots_ref * (std_at_ref / resolvable) ** 2, s0, resolvable


def shot_cost_comparison(n_pts=7, ell=2, tol=1e-3, shots_ref=10 ** 6,
                         repeats=5, layers=3, seed=0, p1=1e-3, p2=1e-2,
                         std_residual_at_1e8=None, w_lead=0.48 - 0.10j, mu=1.0):
    """
    The measurement.  Returns a dict with the Rayleigh arm's shot cost and,
    when `std_residual_at_1e8` is supplied, the residual baseline in the same
    currency.

    Protocol matches run_hardware_demo.py exactly: calibrate on the classical
    simulator, evaluate under depolarizing noise with per-Pauli shot splitting
    and Richardson ZNE, measure the spread, extrapolate as 1/sqrt(shots).
    """
    A = energy_normed_operator(n_pts, ell)
    cal = calibrate(A, layers=layers, seed=seed, w_filter=qnm_filter,
                    w_lead=w_lead, mu=mu)
    nq, lay = cal["n_qubits"], cal["layers"]
    ws = [measure_omega(cal["theta"], A, nq, lay, shots=shots_ref,
                        p1=p1, p2=p2, seed=1000 + k) for k in range(repeats)]
    ws = np.array(ws)
    spread = float(np.sqrt(np.mean(np.abs(ws - ws.mean()) ** 2)))
    shots_needed = shots_ref * (spread / tol) ** 2
    out = {
        "dim": A.shape[0], "qubits": nq, "tol": tol,
        "w_calibrated": cal["w"], "variance": cal["var"],
        "eig_cond": eigenvalue_condition(A),
        "pauli_terms": pauli_terms_measured(A),
        "spread_at_ref": spread, "shots_ref": shots_ref,
        "shots_rayleigh": shots_needed,
        "omega_mean": complex(ws.mean()),
    }
    if std_residual_at_1e8 is not None:
        s_res, s0, dsig = residual_shot_cost(tol, std_at_ref=std_residual_at_1e8)
        out.update({"shots_residual": s_res, "sigma_min_N8": s0,
                    "dsigma_dw": dsig,
                    "advantage": s_res / shots_needed})
    return out
