<div align="center">

# qnmlab

**Black-hole quasinormal modes with quantum algorithms**

[![PyPI](https://img.shields.io/pypi/v/qnmlab.svg)](https://pypi.org/project/qnmlab/)
[![Python](https://img.shields.io/pypi/pyversions/qnmlab.svg)](https://pypi.org/project/qnmlab/)
[![Tests](https://github.com/abir-tech/qnmlab/actions/workflows/tests.yml/badge.svg)](https://github.com/abir-tech/qnmlab/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Developed by **[AiQC — Ai and Quantum Community](https://www.aiqcommunity.org/)**

</div>

---

When a black hole is disturbed, it rings down like a struck bell. The tones it rings at — its
**quasinormal modes** — depend only on the black hole itself, so they are its fingerprint.
`qnmlab` measures these frequencies with **quantum algorithms**: you choose a black hole, a size
in qubits, a representation and a simulated or real quantum backend, and compare the result
with trusted reference values.

```bash
pip install qnmlab
```

## About AiQC

**AiQC (Ai and Quantum Community)** is a research community working on **quantum computing**
and **artificial intelligence**. Its motto is its method — *we do science* — and its mission is
to use the minds and the collective intellectual strength of its members to do research,
regardless of age, academic rank or background.

Recent work and recognition, from [aiqcommunity.org](https://www.aiqcommunity.org/):

- **First place globally** in the Advanced Rigetti Computing Challenge at the international
  Q-Volution Hackathon hosted by Girls in Quantum (March 2026), with a solution based on Pauli
  Correlation Encoding.
- Represented Algeria and Africa at **MIT's iQuHACK 2026** quantum hackathon in Cambridge,
  Massachusetts.
- Organised **AiQuest — Quantum Edition**, the first AI and quantum computing contest in Algeria
  and the Arab Maghreb, featured by **UNESCO** for the International Year of Quantum Science and
  Technology.
- Eleven research projects across quantum machine learning, quantum random number generation,
  quantum reservoir computing, physics-informed hybrid quantum networks and applied AI, several
  run on IBM, Xanadu and Rigetti quantum hardware.

Explore the projects at **[aiqcommunity.org/research](https://www.aiqcommunity.org/research)** ·
[GitHub](https://github.com/Ai-Quantum-Community) ·
[LinkedIn](https://www.linkedin.com/company/ai-quantum-community-aiqc/) ·
contact@aiqcommunity.org

### Authors

| | role |
|---|---|
| **Abir Chekroun** | AiQC Co-Director, Head of Quantum Projects — lead author |
| **Hamed Al-Yoorby** | AiQC member and research partner |
| **Mouadh Abderrahmane Assal** | AiQC Co-Director, Head of Advanced AI Research |
| **Zakaria Lourghi** | AiQC Co-Director, Principal AI Engineer |
| **Achraf Boussahi** | AiQC Co-Director, Chief Scientific Officer |

---

## How it works

![The qnmlab pipeline](https://raw.githubusercontent.com/abir-tech/qnmlab/main/docs/images/pipeline.png)

The pipeline has two stages and needs **no starting guess**.

**Stage 1 — seedless search.**

1. **Search region.** You give a rectangle of the complex frequency plane — nothing else.
2. **Leaver pencil.** At a trial frequency ω, a sparse, tridiagonal matrix P(ω) becomes singular
   exactly when ω is a mode.
3. **Thresholded scan.** At every grid point a variational quantum solver estimates how close
   P(ω) is to singular, and the scan keeps the points that pass a threshold. It thresholds
   instead of minimising, because near a mode the true value is so small that the minimum of
   a noisy map is set by noise. Only cells that pass are refined.

**Stage 2 — verification and phase readout.**

4. **Verification.** The scan returns a shortlist, false candidates included. For each one, a
   quantum state is steered towards it; a real mode keeps the state nearby, a false candidate
   lets it drift away and is rejected.
5. **Readout circuit.** On *hyperboloidal* slices, every mode is an eigenvalue of one fixed
   operator A, so the prepared state is (close to) an eigenvector.
6. **Mode frequency.** The frequency is read from two measured averages,

   &nbsp;&nbsp;&nbsp;&nbsp;**Re ω = −⟨H₂⟩, Im ω = ⟨H₁⟩**, with H₁ = (A + A†)/2 and H₂ = (A − A†)/2i.

Every result is scored against Leaver's continued fraction, the classical reference.

### Why read a phase instead of a magnitude

<img align="right" width="420" src="https://raw.githubusercontent.com/abir-tech/qnmlab/main/docs/images/readout_cost.png" alt="Shots needed by each readout">

The obvious quantum encoding asks the processor where a matrix becomes singular — a quantity
that is **zero at the answer**, which costs about 10¹⁹ shots for 10 % precision. The phase
readout measures two averages of order one instead. On the same task — fixing a frequency to
10⁻³ with four qubits — it needs **about 130 times fewer shots**, and the banded generalised
Rayleigh quotient is cheaper still.

<br clear="right"/>

### The search finds the modes on its own

![Search and verification](https://raw.githubusercontent.com/abir-tech/qnmlab/main/docs/images/search_verification.png)

Starting from a region of the plane, the search narrows to ten candidates; verification accepts
exactly the two true Schwarzschild modes — the fundamental to 10⁻⁶ and the first overtone to
3 × 10⁻⁵ — and rejects all eight false ones, with variational solvers in both stages.

### On a real quantum processor

![The hardware experiment](https://raw.githubusercontent.com/abir-tech/qnmlab/main/docs/images/hardware_workflow.png)

<img align="right" width="380" src="https://raw.githubusercontent.com/abir-tech/qnmlab/main/docs/images/hardware_readout.png" alt="Hardware result">

On **IQM Emerald**, a superconducting processor, the phase readout of a three-qubit state
returned **Mω = 0.439 − 0.104i** against the exact **0.482 − 0.097i**. It is 3.1 times closer to
the right mode than to the next one, and its error (0.043) is what a noise model predicted
before the run. The raw counts ship with the package:

```python
import qnmlab as ql
ql.benchmarks.reanalyse_emerald_2026()   # (0.43919 - 0.10430i)
```

<br clear="right"/>

### What more qubits buy — and cost

![Scaling with qubits](https://raw.githubusercontent.com/abir-tech/qnmlab/main/docs/images/scaling.png)

More qubits make the computed mode far more accurate, but each qubit multiplies the shots for a
fixed-precision readout by about 16. At three qubits the method's own error is already 20 times
smaller than today's hardware error.

---

## Install

```bash
pip install qnmlab                                   # core: numpy, scipy, sympy
pip install "qnmlab[hardware]"                       # + Qiskit, to run circuits on real devices
pip install git+https://github.com/abir-tech/qnmlab.git   # latest development version
```

## Quick start

```python
import qnmlab as ql

bh = ql.KazakovSolodukhin(a=0.5)          # or ql.Schwarzschild(), ql.ReissnerNordstrom(Q), ql.Hayward(l)
ql.reference_modes(bh, ell=2, n_modes=2)  # classical ground truth

# read the fundamental on a simulated noisy 3-qubit device
r = ql.measure_mode(bh, qubits=3, noise=ql.NoiseModel.emerald_random(), shots=800)
print(r.omega, r.meta["error_vs_reference"])

# find modes with no starting guess (about 40 minutes with verification)
res = ql.find_modes(ql.Schwarzschild(), qubits=4)
print(res.modes)          # the fundamental and the first overtone
```

From the command line:

```bash
qnmlab modes   --model kazakov-solodukhin --parameter 0.5
qnmlab readout --model schwarzschild --qubits 3 --noise emerald-random --shots 800
qnmlab search  --model schwarzschild --qubits 4
```

A guided tour is in [`examples/01_quickstart.ipynb`](examples/01_quickstart.ipynb).

## What you can choose

| choice | options |
|---|---|
| black hole | `Schwarzschild()`, `KazakovSolodukhin(a)` — validated against published frequencies; `ReissnerNordstrom(Q)`, `Hayward(l)` — experimental (cross-checked against an independent method) |
| size | number of qubits; operator dimension `2**qubits` |
| readout representation | `"hyperboloidal"` (any model), `"ultraspherical"` (banded generalised pencil, Schwarzschild) |
| search representation | `"leaver"` (tridiagonal, Schwarzschild), `"frobenius"` (any model) |
| search solver | `"exact"` (fast) or `"variational"` (a variational solver at every grid point) |
| noise | `NoiseModel.ideal()`, `.emerald_random()`, `.emerald_coherent()`, or `NoiseModel(p2, eps, readout)` |

## Running on real hardware

```python
op = ql.readout_operator(ql.Schwarzschild(), qubits=3)
state = ql.prepare_state(op, lead=0.48 - 0.097j)
job = ql.hardware.build_job(op, state)                  # 11 circuits
worst, ok = ql.hardware.verify(job)                     # Qiskit export matches the simulation
circuits = [ql.hardware.to_qiskit(c, 3) for c in job["circuits"]]
# ... run them with your provider and collect the raw counts by circuit name ...
result = ql.hardware.analyse_counts(job, raw_counts)    # measures bit order, mitigates readout
```

`analyse_counts` measures the provider's bit order from two calibration circuits and refuses
to continue if a qubit is stuck — both happened on real devices while this package was built.

## Scope

- Classical methods compute these modes faster; `qnmlab` is for studying, benchmarking and
  running the quantum formulation.
- The end-to-end search ran in simulation; on hardware only the readout was run, from a
  classically prepared state. The search costs about 10¹³ shots at four qubits.
- For models other than Schwarzschild, the search threshold is calibrated on the classical
  reference mode (recorded in the result).

## Citing

If you use `qnmlab`, please cite (see [`CITATION.cff`](CITATION.cff)):

> A. Chekroun, H. Al-Yoorby, M. A. Assal, Z. Lourghi, A. Boussahi,
> *Phase Readout and Seedless Search for Black-Hole Quasinormal Modes on Quantum Processors*, 2026.

## Development

```bash
pip install -e ".[dev]"
pytest -m "not slow"     # about 5 minutes
pytest                   # includes the 40-minute end-to-end search
```

MIT licence © 2026 the authors · [AiQC — Ai and Quantum Community](https://www.aiqcommunity.org/)
