"""Time every all-reduce on real processes and fit two cost models to the timings.

    python -m coll.bench --out results

p gloo processes (one thread each) all-reduce float32 vectors from 1 KB to 64 MB. The
time of one call is the slowest rank's, and the best of 7–50 calls is kept: between two
runs it reproduces within 26% for 80% of the points. For each algorithm and size, the
critical path costs ``R·α + B·β + C·γ``. R is the number of rounds, B the sum
over rounds of the largest message, and C the bytes the rank adds into its buffer, all
counted by ``Comm``. A single (α, β, γ) per p is fitted to all algorithms and sizes at
once by non-negative least squares on relative error. A second model adds
``T·β_bus``, where T is the bytes sent by *all* ranks: on one machine every "link" is a
memory copy through the same caches and DRAM, so traffic anywhere slows every rank. Each
model then predicts the fastest algorithm at each size.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .algos import ALLREDUCE, Comm

SIZES = [2**k for k in range(10, 27, 2)]  # bytes: 1 KB … 64 MB


def _worker(rank, p, init, sizes, out_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=p)
    try:
        rows = []
        for nbytes in sizes:
            n = nbytes // 4
            reps = max(7, min(50, int(4e7 // nbytes)))
            for name, fn in ALLREDUCE.items():
                times = []
                for rep in range(reps + 1):
                    x = torch.ones(n)
                    c = Comm()
                    dist.barrier()
                    t0 = time.perf_counter()
                    fn(x, c)
                    el = time.perf_counter() - t0
                    if rep:
                        times.append(el)
                rows.append({"name": name, "bytes": nbytes, "times": times,
                             "rounds": c.rounds, "path_bytes": c.path_bytes,
                             "sent": c.bytes_sent, "reduced": c.reduced})  # fmt: skip
        torch.save(rows, os.path.join(out_dir, f"{rank}.pt"))
    finally:
        dist.destroy_process_group()


def measure(p, sizes):
    with tempfile.TemporaryDirectory() as d:
        init = "file:///" + os.path.abspath(os.path.join(d, "rdzv")).replace("\\", "/").lstrip("/")
        mp.spawn(_worker, args=(p, init, sizes, d), nprocs=p, join=True)
        per = [torch.load(os.path.join(d, f"{r}.pt")) for r in range(p)]
    out = []
    for j, base in enumerate(per[0]):
        slow = np.max([per[r][j]["times"] for r in range(p)], axis=0)  # slowest rank per rep
        out.append({
            "p": p, "name": base["name"], "bytes": base["bytes"], "seconds": float(np.min(slow)),
            "median_seconds": float(np.median(slow)),
            "rounds": max(per[r][j]["rounds"] for r in range(p)),
            "path_bytes": max(per[r][j]["path_bytes"] for r in range(p)),
            "max_sent": max(per[r][j]["sent"] for r in range(p)),
            "reduced": max(per[r][j]["reduced"] for r in range(p)),
            "total_sent": sum(per[r][j]["sent"] for r in range(p)),
        })  # fmt: skip
    return out


MODELS = {  # feature columns of each cost model
    "links": ("rounds", "path_bytes", "reduced"),  # α-β-γ: every pair has its own link
    "shared": ("rounds", "path_bytes", "reduced", "total_sent"),  # + one host-wide bus
}


def fit(rows, cols):
    """Non-negative least squares on relative error (exact: tries every column subset)."""
    A = np.array([[r[c] for c in cols] for r in rows], float)
    t = np.array([r["seconds"] for r in rows])
    Aw, tw = A / t[:, None], np.ones_like(t)
    best = (np.inf, np.zeros(len(cols)))
    for mask in itertools.product([0, 1], repeat=len(cols)):
        idx = [j for j, m in enumerate(mask) if m]
        if not idx:
            continue
        c, *_ = np.linalg.lstsq(Aw[:, idx], tw, rcond=None)
        if (c < 0).any():
            continue
        coef = np.zeros(len(cols))
        coef[idx] = c
        res = float(np.sum((Aw @ coef - 1) ** 2))
        best = min(best, (res, coef), key=lambda b: b[0])
    return best[1]


def _gbps(c):
    return f"{1 / c / 1e9:.2f} GB/s" if c > 0 else "∞ (not needed)"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--P", type=int, nargs="+", default=[4, 8])
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    allrows, L = [], []
    for p in a.P:
        rows = measure(p, SIZES)
        start = len(L)
        L += [f"### p = {p}", "",
              "| model | α per round | 1/β per link | 1/γ reduce | 1/β_bus shared "
              "| median error | fastest picked right |", "|---" * 7 + "|"]  # fmt: skip
        for m, cols in MODELS.items():
            coef = fit(rows, cols)
            for r in rows:
                r["pred_" + m] = float(sum(k * r[c] for k, c in zip(coef, cols, strict=True)))
            err = np.median([abs(r["pred_" + m] / r["seconds"] - 1) for r in rows])
            right, key = 0, "pred_" + m
            for nb in SIZES:
                rs = [r for r in rows if r["bytes"] == nb]
                right += min(rs, key=lambda r: r["seconds"]) is min(rs, key=lambda r: r[key])
            bus = _gbps(coef[3]) if len(coef) > 3 else "—"
            L.append(f"| {m} | {coef[0] * 1e6:.0f} µs | {_gbps(coef[1])} | {_gbps(coef[2])} | "
                     f"{bus} | {100 * err:.0f}% | {right} / {len(SIZES)} |")  # fmt: skip
        L += ["", "| size | " + " | ".join(ALLREDUCE) + " | fastest | links model | shared model |",
              "|---" * (len(ALLREDUCE) + 4) + "|"]  # fmt: skip
        for nb in SIZES:
            rs = {r["name"]: r for r in rows if r["bytes"] == nb}
            picks = [min(rs, key=lambda n, k=k: rs[n][k])
                     for k in ("seconds", "pred_links", "pred_shared")]  # fmt: skip
            size = f"{nb // 1024} KB" if nb < 2**20 else f"{nb // 2**20} MB"
            cells = " | ".join(f"{rs[k]['seconds'] * 1e3:.2f}" for k in ALLREDUCE)
            L.append(f"| {size} | {cells} | " + " | ".join(picks) + " |")
        L += [""]
        allrows += rows
        print("\n".join(L[start:]), flush=True)
    with open(os.path.join(a.out, "bench.jsonl"), "w") as f:
        for r in allrows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(a.out, "bench.md"), "w", encoding="utf-8") as f:
        head = ("All-reduce of float32 vectors, gloo on one machine, one thread per rank. "
                "Times in ms: the slowest rank per call, best of 7–50 calls.\n\n")  # fmt: skip
        f.write(head + "\n".join(L))


if __name__ == "__main__":
    main()
