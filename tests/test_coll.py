import math
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from coll.algos import ALLREDUCE, Comm, chunks, ring_all_gather, ring_reduce_scatter


def _worker(rank, p, init, n, out_dir):
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=p)
    try:
        res = {}
        for name, fn in ALLREDUCE.items():
            if name in ("recursive_doubling", "rabenseifner") and p & (p - 1):
                continue
            g = torch.Generator().manual_seed(rank)
            x = torch.randint(-1000, 1000, (n,), generator=g).double()  # exact sums
            c = Comm()
            fn(x, c)
            res[name] = (x, c.bytes_sent, c.rounds)
        # reduce-scatter alone: which chunk each rank ends up owning
        x = torch.full((n,), float(rank + 1), dtype=torch.float64)
        c = Comm()
        own = ring_reduce_scatter(x, c)
        rs = (own, x[chunks(n, p)[own]].clone())
        ring_all_gather(x, Comm(), own)
        torch.save({"allreduce": res, "rs": rs, "ag": x}, os.path.join(out_dir, f"{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _run(p, n, tmp_path):
    init = "file:///" + os.path.abspath(tmp_path / "rdzv").replace("\\", "/").lstrip("/")
    mp.spawn(_worker, args=(p, init, n, str(tmp_path)), nprocs=p, join=True)
    return [torch.load(tmp_path / f"{r}.pt") for r in range(p)]


@pytest.fixture(scope="module", params=[(2, 37), (3, 40), (4, 1001), (5, 64), (8, 999)])
def runs(request, tmp_path_factory):
    p, n = request.param
    return p, n, _run(p, n, tmp_path_factory.mktemp(f"p{p}"))


def _expected(p, n):
    xs = [torch.randint(-1000, 1000, (n,), generator=torch.Generator().manual_seed(r)).double()
          for r in range(p)]  # fmt: skip
    return torch.stack(xs).sum(0)


def test_every_allreduce_gives_the_exact_sum_on_every_rank(runs):
    p, n, out = runs
    want = _expected(p, n)
    for name in out[0]["allreduce"]:
        for r in range(p):
            assert torch.equal(out[r]["allreduce"][name][0], want), (name, r)


def test_reduce_scatter_then_all_gather(runs):
    p, n, out = runs
    total = p * (p + 1) / 2
    owners = sorted(o["rs"][0] for o in out)
    assert owners == list(range(p))  # each chunk owned by exactly one rank
    for o in out:
        assert (o["rs"][1] == total).all()
        assert (o["ag"] == total).all()


def test_ring_and_rabenseifner_hit_the_bandwidth_lower_bound(runs):
    p, n, out = runs
    lower = 2 * (p - 1) / p * n * 8
    for name in ("ring", "rabenseifner"):
        if name not in out[0]["allreduce"]:
            continue
        sent = [o["allreduce"][name][1] for o in out]
        # chunks differ by at most one element, so each rank is within 2(p−1) elements
        assert all(abs(s - lower) <= 2 * (p - 1) * 8 for s in sent)
        assert abs(sum(sent) - p * lower) <= 1e-6 * p * lower  # exactly on average


def test_round_counts_match_the_closed_forms(runs):
    p, n, out = runs
    lg = math.ceil(math.log2(p))
    rounds = {name: [o["allreduce"][name][2] for o in out] for name in out[0]["allreduce"]}
    assert rounds["ring"] == [2 * (p - 1)] * p
    assert rounds["naive"][0] == 2 * (p - 1)
    assert rounds["tree"][0] == 2 * lg  # the root is on every level
    if p & (p - 1) == 0:
        assert rounds["recursive_doubling"] == [lg] * p
        assert rounds["rabenseifner"] == [2 * lg] * p
        sent = [o["allreduce"]["recursive_doubling"][1] for o in out]
        assert sent == [n * 8 * lg] * p


def test_naive_root_sends_p_minus_1_copies(runs):
    p, n, out = runs
    sent = [o["allreduce"]["naive"][1] for o in out]
    assert sent[0] == (p - 1) * n * 8 and sent[1:] == [n * 8] * (p - 1)


def test_fit_recovers_known_costs_and_keeps_them_non_negative():
    from coll.bench import MODELS, fit

    true = {"rounds": 2e-4, "path_bytes": 0.0, "reduced": 1e-9, "total_sent": 3e-10}
    g = torch.Generator().manual_seed(0)
    rows = []
    for _ in range(40):
        r = {c: float(torch.randint(1, 10**7, (1,), generator=g)) for c in true}
        r["seconds"] = sum(true[c] * r[c] for c in true)
        rows.append(r)
    coef = fit(rows, MODELS["shared"])
    for k, c in zip(coef, MODELS["shared"], strict=True):
        assert k >= 0 and math.isclose(k, true[c], rel_tol=1e-6, abs_tol=1e-15)
