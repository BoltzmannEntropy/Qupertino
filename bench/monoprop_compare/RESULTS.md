# Measured results

Regenerate with `make_figure.py`; raw series in `results/raw/`.


## 2D TFIM (upstream settings: dt=0.05, atol=1e-6, no cutoff, 21 timed points)

| Grid | Qubits | mlxQ Metal (s) | mlxQ pure (s) | monoprop (s) | Faster |
|---|---|---|---|---|---|
| 3x3 | 9 | 0.127 | 0.020 | 0.213 | mlxQ 1.7x |
| 4x4 | 16 | 0.019 | 0.040 | 1.242 | mlxQ 64.7x |
| 4x5 | 20 | 0.052 | 0.156 | 1.912 | mlxQ 36.5x |
| 5x5 | 25 | 0.836 | 5.304 | 2.438 | mlxQ 2.9x |
| 4x7 | 28 | 40.664 | — | 2.925 | monoprop 13.9x |
| 6x6 | 36 | — | — | 3.105 | monoprop only |

## Kicked Ising (upstream settings: 20 layers, cutoff=8, atol=1e-4)

| Qubits | mlxQ Metal (s) | monoprop (s) | Faster | Δexpval (trunc. error) |
|---|---|---|---|---|
| 12 | — | 0.351 | monoprop only | — |
| 14 | — | 0.263 | monoprop only | — |
| 16 | 0.026 | 0.273 | mlxQ 10.7x | 0.319 |
| 20 | 0.069 | 0.338 | mlxQ 4.9x | 0.188 |
| 24 | 0.370 | 0.271 | monoprop 1.4x | 0.025 |
| 28 | 47.844 | 0.188 | monoprop 254.3x | 0.290 |
| 127 | — | 0.683 | monoprop only | — |
