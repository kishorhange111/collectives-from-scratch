# collectives-from-scratch

[![CI](https://github.com/kishorhange111/collectives-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/collectives-from-scratch/actions/workflows/ci.yml)

All-reduce, the collective that synchronises gradients in data-parallel training, written
from scratch on nothing but point-to-point `isend` / `irecv`. It covers the five classic
algorithms (naive, binomial tree, recursive doubling, ring and Rabenseifner) and the ring
reduce-scatter and all-gather that FSDP / ZeRO are built from. Every algorithm counts its
own rounds and bytes, the tests check those counts against the closed forms, and a
benchmark fits cost models to real multi-process timings.

## Algorithms ([`algos.py`](src/coll/algos.py))

For p ranks and a vector of n bytes, per rank:

| algorithm | rounds | bytes sent | bytes reduced |
|---|---|---|---|
| naive (reduce to rank 0, send back) | 2(p − 1) at the root | (p − 1)·n at the root | (p − 1)·n at the root |
| binomial tree (reduce + broadcast) | 2⌈log₂ p⌉ | up to n·⌈log₂ p⌉ (the root) | up to n·⌈log₂ p⌉ |
| recursive doubling | log₂ p | n·log₂ p | n·log₂ p |
| ring (reduce-scatter + all-gather) | 2(p − 1) | 2n(p − 1)/p | n(p − 1)/p |
| Rabenseifner (halving + doubling) | 2 log₂ p | 2n(p − 1)/p | n(p − 1)/p |

`2n(p − 1)/p` is the lower bound on bytes sent per rank, and ring and Rabenseifner reach
it. Recursive doubling uses the fewest rounds instead. The textbook α-β-γ model prices a
run at `rounds·α + bytes·β + reduced·γ`, which predicts recursive doubling for small
messages and the bandwidth-optimal pair for large ones.

## Results ([`bench.py`](src/coll/bench.py))

p = 2, 4 and 8 gloo processes on one 8-core machine (WSL2), one thread each, all-reduce
float32 vectors from 1 KB to 64 MB. A call's time is the slowest rank's, best of 7–50
calls. Two runs agree within 26% on 80% of the points, so gaps smaller than that are not
claimed below. All the tables are in [results/bench.md](results/bench.md).

p = 8, time in ms:

| size | naive | tree | recursive doubling | ring | Rabenseifner |
|---|---|---|---|---|---|
| 4 KB | 2.14 | 1.37 | **1.35** | 7.09 | 2.72 |
| 64 KB | 2.31 | 1.83 | **1.41** | 6.87 | 2.84 |
| 1 MB | 8.50 | **5.89** | 8.01 | 12.67 | 7.19 |
| 16 MB | 117.0 | 66.8 | 100.3 | 67.8 | **61.8** |
| 64 MB | 420.4 | 251.9 | 451.1 | **222.2** | 250.6 |

- **Latency rules up to about 256 KB.** A round costs α ≈ 150–210 µs (gloo over
  loopback TCP). Ring needs 2(p − 1) = 14 rounds against recursive doubling's 3, and it
  is 2.8–5.3× slower at p = 8 for every size up to 256 KB.
- **At 16–64 MB the bandwidth-optimal algorithms win.** Ring and Rabenseifner beat
  recursive doubling by 1.5–2.0× at p = 8, and by 1.26× at p = 4 and 64 MB.
- **Bytes per rank is the wrong measure on one machine.** The binomial tree's root sends
  n·log₂ p, which is 1.33× ring's per-rank bytes at p = 4 and 1.71× at p = 8. Even so, at
  p = 4 it matches ring within 2% from 4 MB to 64 MB (7.75 vs 7.87 ms, 156.8 vs 158.3 ms).
  At p = 8 it is 29% faster than ring at 4 MB and within 13% at 16–64 MB. The per-rank
  count misses it, but tree and ring move the same **total** traffic, 2(p − 1)·n, and on one host every "link" is a memory copy
  through the same caches and DRAM. Recursive doubling moves p·n·log₂ p in total, which
  is 1.33× ring's at p = 4 and 1.71× at p = 8. Its measured slowdown at 64 MB is 1.26×
  and 2.03×. The naive algorithm also moves 2(p − 1)·n, but serialises it through one
  root, and it is 1.9× slower than ring at p = 8 and 64 MB.

So the benchmark fits two models to all algorithms and sizes at once, by non-negative
least squares on relative error:

| p | per-link α-β-γ: median error | + shared bus T·β_bus: median error | fitted 1/β_bus |
|---|---|---|---|
| 2 | 29% | 21% | 2.2 GB/s |
| 4 | 35% | 18% | 3.3 GB/s |
| 8 | 46% | 21% | 3.8 GB/s |

T is the bytes sent by all ranks. At p = 4 and 8, the per-link model sets its bandwidth
term β to zero, so per-rank bytes explain none of the time. The shared-bus term halves
the error. Neither model reliably picks the fastest algorithm at large sizes, because
tree, ring and Rabenseifner sit within the 26% noise of each other there. That is the
practical lesson: the α-β ranking assumes independent links, as with NVLink or one
NIC per GPU. When ranks share a bus, choose by rounds first and total bytes second.

## Tests

`pytest -q` runs 26 tests on real gloo process groups for p = 2, 3, 4, 5 and 8 (with
vector lengths that p does not divide). They check:
- every algorithm gives the exact sum on every rank;
- reduce-scatter leaves each rank its own fully reduced chunk, and all-gather restores the
  whole vector, on uneven chunk sizes too;
- ring and Rabenseifner send exactly `2n(p − 1)/p` per rank, the bandwidth lower bound;
- the round counts equal the closed forms, and the naive root sends (p − 1) copies;
- the non-negative fit recovers known costs exactly.

```
pip install -e ".[dev]" && pytest -q
python -m coll.bench --out results --P 2 4 8
```
