# QuantumStudio v1.0.0 Release Notes

**Release Date:** February 21, 2026

## Overview

QuantumStudio v1.0.0 is the initial release of the MLX Quantum Benchmarking Suite for Apple Silicon. This desktop application provides researchers with tools for running quantum computing benchmarks using the MLX framework on Apple's unified memory architecture.

## Features

### Quantum Benchmarking
- **MLX Integration**: Native Apple Silicon acceleration via MLX framework
- **Benchmark Suite**: Comprehensive quantum algorithm benchmarks
- **Performance Metrics**: Detailed timing and memory profiling
- **Multi-Qubit Support**: Scalable benchmarks from 1-25 qubits

### Supported Benchmarks
- **Gate Operations**: X, H, T, RX, RZ, CNOT, Toffoli
- **Quantum Fourier Transform (QFT)**: Multiple variants
- **QCBM**: Quantum Circuit Born Machine benchmarks
- **VQE**: Variational Quantum Eigensolver
- **Phase Estimation**: Quantum phase estimation algorithms

### Analysis Tools
- **Job Queue**: Background benchmark execution
- **Results Viewer**: Interactive benchmark analysis
- **Export Formats**: JSON, CSV, and yaoquantum.org compatible output
- **Visualization**: Performance plots and scaling analysis

### MCP Integration
- **MCP Server**: Claude Code integration via MCP protocol
- **Remote Control**: Run benchmarks via AI assistant
- **Status Monitoring**: Real-time job status queries

## Technical Details

- **Version**: 1.0.0 (build 1)
- **Platform**: macOS (Apple Silicon optimized)
- **Framework**: Flutter 3.x with Python FastAPI backend
- **Python**: 3.11.x bundled in DMG
- **MLX**: Apple's Machine Learning framework
- **Minimum macOS**: 12.0 (Monterey)

## Installation

1. Download `QuantumStudio-1.0.0-macos.dmg`
2. Open the DMG and drag QuantumStudio to Applications
3. On first launch, right-click the app and select "Open" (macOS Gatekeeper bypass)
4. If launch is still blocked, open `System Settings -> Privacy & Security -> Open Anyway`

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| macOS | 12.0 | 13.0+ |
| RAM | 8GB | 16GB+ (for larger qubit counts) |
| Storage | 2GB | 5GB |
| CPU | Apple Silicon (M1+) | M1 Pro/Max or newer |

## Checksums

SHA256 checksums are provided in `QuantumStudio-1.0.0-macos.dmg.sha256`

## Known Issues

- First launch requires Gatekeeper bypass (right-click > Open or Privacy & Security > Open Anyway)
- Large qubit benchmarks (>20) require significant memory and time
- Intel Macs supported but with reduced performance

## License

- Source code: Business Source License 1.1 (`LICENSE`)
- Binary distribution: Binary Distribution License (`BINARY-LICENSE.txt`)
- License overview: `LICENSE.md`

---

**Website:** https://qneura.ai/apps.html

For bug reports and feature requests, contact: solomon@qneura.ai
