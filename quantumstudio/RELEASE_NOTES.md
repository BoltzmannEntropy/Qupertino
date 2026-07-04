# Qupertino v1.0.2 Release Notes

## Overview

Qupertino is an MLX-native quantum circuit simulator and benchmarking studio for Apple Silicon, packaged as a self-contained macOS desktop app (Flutter UI + bundled Python/FastAPI backend, no external install required).

It runs in **two measured performance tiers** on the same hardware:

- **Pure-MLX tier** — every structured gate expressed as MLX array operations.
- **Hand-tuned Metal shader tier** (opt-in) — hand-written Metal kernels for every structured layer family (phase-LUT diagonals, GF(2) affine permutation gathers, fused tensor-product single-qubit layers, radix-4 QFT and Walsh–Hadamard butterflies, basis-conjugated XX/YY Trotter layers), reached through semantics-preserving fusion detectors.

## Measured performance (M1 Max, 25 qubits, gate-identical circuits)

The Metal shader tier is **fastest in all 18 comparison cells** against Qiskit Aer CPU and PennyLane `lightning.qubit`:

| Workload | Qupertino Metal | Aer CPU | PennyLane | Paired speedup |
|---|---:|---:|---:|---|
| QFT | 59 ms | 2.80 s | 5.61 s | 47× / 95× |
| TFIM Trotter (20 steps) | 0.50 s | 17.79 s | 32.95 s | 36× / 67× |
| Phase estimation | 0.105 s | 4.05 s | 6.06 s | 43× / 65× |
| GHZ | 22 ms | 0.69 s | 0.42 s | 31× / 19× |

Gate-stream QFT (59 ms) is faster than MLX's own `mx.fft` primitive. Across the full 29-workload suite the shader tier accelerates **26 of 29 workloads, up to 25×** over the pure tier.

## Features

- MLX state-vector and MPS backends; OpenQASM 2.0 import
- Benchmark families: QFT, QAOA, VQE, QCBM, Grover, phase estimation, Hamiltonian/time-evolution, Heisenberg/TFIM Trotter, and more
- Job queue, results viewer, CSV/JSON export, scaling plots
- MCP server for Claude Code integration
- Reproducible artifacts with raw timing distributions and run manifests

## Technical details

- **Version**: 1.0.2 (build 3)
- **Platform**: macOS (Apple Silicon optimized; Intel supported with reduced performance)
- **Framework**: Flutter with bundled Python 3.11 FastAPI backend
- **Minimum macOS**: 12.0 (Monterey)

## Installation

1. Download `Qupertino-1.0.2-macos.dmg`
2. Open the DMG and drag **Qupertino** to Applications
3. On first launch, right-click the app and select "Open" (macOS Gatekeeper), or open `System Settings → Privacy & Security → Open Anyway`

## System requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| macOS | 12.0 | 13.0+ |
| RAM | 8 GB | 16 GB+ (larger qubit counts) |
| Storage | 2 GB | 5 GB |
| CPU | Apple Silicon (M1+) | M1 Pro/Max or newer |

## Checksums

SHA256 checksums are provided alongside each asset (`*.sha256`).

## License

MIT License (`LICENSE`). Source and binaries are free and open source.

---

**Repository:** https://github.com/BoltzmannEntropy/Qupertino
**Website:** https://boltzmannentropy.github.io/QupertinoWEB/
