# Changelog

## 0.1.1

- README: about AiQC (Ai and Quantum Community), the authors, and an illustrated explanation
  of the pipeline, the readout cost, the search, the hardware run and the scaling.
  No code changes.

## 0.1.0

First release.

- Black holes: Schwarzschild and Kazakov–Solodukhin (validated); Reissner–Nordström and
  Hayward (experimental).
- Reference modes from the hyperboloidal operator, any metric.
- Phase readout with hyperboloidal and ultraspherical representations; variance-minimising
  state preparation steered by a lead.
- Seedless search: thresholded scan (exact or variational solver) plus readout verification.
- Simulated noise: random (depolarising) and coherent (ZZ over-rotation), readout error,
  zero-noise extrapolation, qubit-wise and general-commuting measurement grouping.
- Hardware path: job building, Qiskit export and verification, bit-order detection, stuck-qubit
  detection, readout mitigation.
- Benchmarks: published frequencies and the raw counts of the 2026 IQM Emerald run.
- Command line: `qnmlab modes | readout | search`.
