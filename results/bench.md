All-reduce of float32 vectors, gloo on one machine, one thread per rank. Times in ms: the slowest rank per call, best of 7–50 calls.

### p = 2

| model | α per round | 1/β per link | 1/γ reduce | 1/β_bus shared | median error | fastest picked right |
|---|---|---|---|---|---|---|
| links | 156 µs | 6.02 GB/s | 1.46 GB/s | — | 29% | 5 / 9 |
| shared | 154 µs | ∞ (not needed) | ∞ (not needed) | 2.18 GB/s | 21% | 6 / 9 |

| size | naive | tree | recursive_doubling | ring | rabenseifner | fastest | links model | shared model |
|---|---|---|---|---|---|---|---|---|
| 1 KB | 0.26 | 0.24 | 0.21 | 0.49 | 0.47 | recursive_doubling | recursive_doubling | recursive_doubling |
| 4 KB | 0.26 | 0.26 | 0.21 | 0.48 | 0.45 | recursive_doubling | recursive_doubling | recursive_doubling |
| 16 KB | 0.27 | 0.29 | 0.21 | 0.49 | 0.49 | recursive_doubling | recursive_doubling | recursive_doubling |
| 64 KB | 0.30 | 0.32 | 0.25 | 0.53 | 0.47 | recursive_doubling | recursive_doubling | recursive_doubling |
| 256 KB | 0.43 | 0.42 | 0.36 | 0.64 | 0.62 | recursive_doubling | recursive_doubling | recursive_doubling |
| 1 MB | 0.93 | 1.12 | 1.10 | 1.45 | 1.49 | naive | ring | recursive_doubling |
| 4 MB | 4.14 | 3.26 | 2.82 | 4.46 | 3.88 | recursive_doubling | ring | recursive_doubling |
| 16 MB | 16.80 | 15.60 | 16.70 | 19.78 | 18.77 | tree | ring | recursive_doubling |
| 64 MB | 85.35 | 86.48 | 91.66 | 110.39 | 82.43 | rabenseifner | ring | recursive_doubling |

### p = 4

| model | α per round | 1/β per link | 1/γ reduce | 1/β_bus shared | median error | fastest picked right |
|---|---|---|---|---|---|---|
| links | 203 µs | ∞ (not needed) | 0.90 GB/s | — | 35% | 5 / 9 |
| shared | 200 µs | 424.11 GB/s | 8.48 GB/s | 3.26 GB/s | 18% | 5 / 9 |

| size | naive | tree | recursive_doubling | ring | rabenseifner | fastest | links model | shared model |
|---|---|---|---|---|---|---|---|---|
| 1 KB | 0.85 | 0.69 | 0.64 | 1.98 | 1.31 | recursive_doubling | recursive_doubling | recursive_doubling |
| 4 KB | 0.76 | 0.73 | 0.55 | 2.10 | 1.52 | recursive_doubling | recursive_doubling | recursive_doubling |
| 16 KB | 0.91 | 0.77 | 0.73 | 2.23 | 1.68 | recursive_doubling | recursive_doubling | recursive_doubling |
| 64 KB | 1.15 | 0.90 | 0.85 | 2.16 | 1.83 | recursive_doubling | recursive_doubling | recursive_doubling |
| 256 KB | 1.64 | 1.17 | 1.15 | 2.24 | 1.69 | recursive_doubling | recursive_doubling | recursive_doubling |
| 1 MB | 3.56 | 2.40 | 2.96 | 3.00 | 3.01 | tree | rabenseifner | rabenseifner |
| 4 MB | 13.34 | 7.75 | 10.52 | 7.87 | 9.32 | tree | rabenseifner | rabenseifner |
| 16 MB | 45.28 | 30.86 | 45.47 | 31.28 | 32.77 | tree | rabenseifner | rabenseifner |
| 64 MB | 190.82 | 156.76 | 199.48 | 158.25 | 167.16 | tree | rabenseifner | rabenseifner |

### p = 8

| model | α per round | 1/β per link | 1/γ reduce | 1/β_bus shared | median error | fastest picked right |
|---|---|---|---|---|---|---|
| links | 213 µs | ∞ (not needed) | 0.78 GB/s | — | 46% | 3 / 9 |
| shared | 212 µs | 9.16 GB/s | ∞ (not needed) | 3.75 GB/s | 21% | 3 / 9 |

| size | naive | tree | recursive_doubling | ring | rabenseifner | fastest | links model | shared model |
|---|---|---|---|---|---|---|---|---|
| 1 KB | 1.77 | 1.25 | 1.39 | 5.38 | 3.08 | tree | recursive_doubling | recursive_doubling |
| 4 KB | 2.14 | 1.37 | 1.35 | 7.09 | 2.72 | recursive_doubling | recursive_doubling | recursive_doubling |
| 16 KB | 1.98 | 1.31 | 1.49 | 5.51 | 2.14 | tree | recursive_doubling | recursive_doubling |
| 64 KB | 2.31 | 1.83 | 1.41 | 6.87 | 2.84 | recursive_doubling | recursive_doubling | recursive_doubling |
| 256 KB | 3.46 | 2.24 | 2.81 | 7.76 | 4.24 | tree | rabenseifner | rabenseifner |
| 1 MB | 8.50 | 5.89 | 8.01 | 12.67 | 7.19 | tree | rabenseifner | rabenseifner |
| 4 MB | 29.32 | 17.40 | 24.07 | 24.61 | 21.95 | tree | rabenseifner | rabenseifner |
| 16 MB | 117.04 | 66.75 | 100.27 | 67.77 | 61.81 | rabenseifner | rabenseifner | rabenseifner |
| 64 MB | 420.38 | 251.86 | 451.14 | 222.23 | 250.64 | ring | rabenseifner | rabenseifner |
