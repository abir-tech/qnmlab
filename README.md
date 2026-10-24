# qnmlab

**Black-hole quasinormal modes with quantum algorithms.**

`qnmlab` computes the ringing frequencies of black holes with variational quantum
algorithms. You choose a black hole, a size in qubits, a representation, and a simulated
or real quantum backend, and compare the result with trusted reference values.

The two ideas behind it:

- **Phase readout.** On hyperboloidal slices the modes are eigenvalues of one linear
  operator, so a mode's frequency is two measured averages,
  `Re ω = −⟨H₂⟩`, `Im ω = ⟨H₁⟩`. That costs about 130 times fewer shots than detecting
  where a matrix becomes singular.
- **Seedless search.** A thresholded scan of the complex frequency plane produces a
  shortlist of candidates, and the phase readout verifies each one. No starting guess is
  needed.

Classical methods compute these modes faster; `qnmlab` is for studying, benchmarking and
running the quantum formulation, not for replacing them.

## Install

```bash
pip install qnmlab                 # core: numpy, scipy, sympy
pip install "qnmlab[hardware]"     # + Qiskit, to export circuits for real devices
```

Latest development version:

```bash
pip install git+https://github.com/abir-tech/qnmlab.git
```

## Quick start

```python
import qnmlab as ql

bh = ql.KazakovSolodukhin(a=0.5)          # or ql.Schwarzschild(), ql.ReissnerNordstrom(Q), ql.Hayward(l)
ql.reference_modes(bh, ell=2, n_modes=2)  # classical ground truth

# read the fundamental on a simulated noisy 3-qubit device
r = ql.measure_mode(bh, qubits=3, noise=ql.NoiseModel.emerald_random(), shots=800)
print(r.omega, r.meta["error_vs_reference"])

# find modes with no starting guess (Schwarzschild, 4 qubits)
res = ql.find_modes(ql.Schwarzschild(), qubits=4)
print(res.modes)          # the fundamental and the first overtone
```

From the command line:

```bash
qnmlab modes   --model kazakov-solodukhin --parameter 0.5
qnmlab readout --model schwarzschild --qubits 3 --noise emerald-random --shots 800
qnmlab search  --model schwarzschild --qubits 4
```

## What you can choose

| choice | options |
|---|---|
| black hole | `Schwarzschild()`, `KazakovSolodukhin(a)` — validated against published frequencies; `ReissnerNordstrom(Q)`, `Hayward(l)` — experimental (cross-checked against an independent method, not against published values) |
| size | qubits; operator dimension `2**qubits` |
| readout representation | `"hyperboloidal"` (any model), `"ultraspherical"` (banded generalised pencil, Schwarzschild) |
| search representation | `"leaver"` (tridiagonal, Schwarzschild), `"frobenius"` (any model) |
| search solver | `"exact"` (fast) or `"variational"` (a variational solver at every grid point) |
| noise | `NoiseModel.ideal()`, `.emerald_random()`, `.emerald_coherent()`, or your own `NoiseModel(p2, eps, readout)` |

## Real hardware

```python
op = ql.readout_operator(ql.Schwarzschild(), qubits=3)
state = ql.prepare_state(op, lead=0.48 - 0.097j, layers=2)
job = ql.hardware.build_job(op, state)                 # 11 circuits
ok = ql.hardware.verify(job)                            # Qiskit export matches the simulation
circuits = [ql.hardware.to_qiskit(c, 3) for c in job["circuits"]]
# ... run them with your provider, collect raw counts by circuit name ...
result = ql.hardware.analyse_counts(job, raw_counts)    # measures bit order, mitigates readout
```

`analyse_counts` measures the provider's bit order from two calibration circuits and stops
if a qubit is stuck — both happened on real devices while this package was developed.

The raw counts of the 2026 IQM Emerald run ship with the package:

```python
ql.benchmarks.reanalyse_emerald_2026()   # (0.43919 - 0.10430i), exact 0.48186 - 0.09674i
```

## Scope and honesty

- The end-to-end search ran in simulation; on hardware only the readout was run, from a
  classically prepared state.
- The search costs about 10^13 measurement shots at 4 qubits.
- The hardware error (~0.04) is dominated by gate noise; more qubits help only once it falls.

## Citing

If you use `qnmlab`, please cite the accompanying paper (see `CITATION.cff`):
A. Chekroun, H. Al-Yoorby, M. A. Assal, Z. Lourghi, A. Boussahi,
*Phase Readout and Seedless Search for Black-Hole Quasinormal Modes on Quantum Processors* (2026).

## Development

```bash
pip install -e ".[dev]"
pytest -m "not slow"     # ~5 minutes
pytest                   # includes the ~40-minute end-to-end search
```

MIT licence.
