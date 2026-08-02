# Measured results

Median total runtime in seconds over dedicated-process repeats, with [min–max, n] spread. Selection protocol (shared-process and pre-fix exclusions) is documented in `make_figure.py`; every raw run is in `results/raw/`.


## 2D TFIM (upstream settings: dt=0.05, atol=1e-6, no cutoff, 21 timed points)

| Grid | Qubits | mlxQ Metal (s) | mlxQ pure (s) | monoprop (s) | Faster (median) |
|---|---|---|---|---|---|
| 3x3 | 9 | 0.023 [0.020–0.127, n=6] | 0.022 [0.020–0.059, n=6] | 0.245 [0.226–0.382, n=6] | mlxQ 10.6x |
| 4x4 | 16 | 0.027 [0.019–0.028, n=6] | 0.042 [0.040–0.056, n=6] | 1.795 [1.635–2.019, n=6] | mlxQ 65.9x |
| 4x5 | 20 | 0.062 [0.052–0.086, n=6] | 0.158 [0.154–0.164, n=6] | 2.302 [2.142–2.642, n=6] | mlxQ 37.3x |
| 5x5 | 25 | 0.807 [0.793–0.836, n=6] | 5.862 [5.304–7.872, n=6] | 2.770 [2.513–2.994, n=6] | mlxQ 3.4x |
| 4x7 | 28 | 49.813 [40.664–86.561, n=3] | — | 2.121 [2.017–2.230, n=3] | monoprop 23.5x |
| 6x6 | 36 | — | — | 3.604 [3.105–3.820, n=6] | monoprop only |

## Kicked Ising (upstream settings: 20 layers, cutoff=8, atol=1e-4)

| Qubits | mlxQ Metal (s) | monoprop (s) | Faster (median) | Δexpval (trunc. error) |
|---|---|---|---|---|
| 12 | — | 0.190 [0.185–0.351, n=3] | monoprop only | — |
| 14 | — | 0.176 [0.155–0.263, n=3] | monoprop only | — |
| 16 | 0.028 [0.026–0.589, n=3] | 0.160 [0.138–0.189, n=4] | mlxQ 5.8x | 0.319 |
| 20 | 0.069 [0.068–0.512, n=3] | 0.333 [0.328–0.435, n=3] | mlxQ 4.8x | 0.188 |
| 24 | 0.390 [0.370–1.288, n=3] | 0.288 [0.254–0.337, n=3] | monoprop 1.4x | 0.025 |
| 28 | 56.324 [47.844–135.906, n=3] | 0.167 [0.153–0.183, n=3] | monoprop 337.5x | 0.290 |
| 127 | — | 0.670 [0.620–0.683, n=3] | monoprop only | — |
