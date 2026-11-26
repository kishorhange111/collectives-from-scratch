"""All-reduce and its building blocks, written on point-to-point sends only.

Every algorithm runs in **rounds**: in one round each rank sends at most one message and
receives at most one. The α-β model prices a round at ``α + bytes · β`` (latency plus
bandwidth). For p ranks and a vector of n bytes:

| algorithm | rounds | bytes sent per rank | note |
|---|---|---|---|
| reduce-to-root + broadcast | 2(p − 1) at root | (p − 1)·n at root | the naive baseline |
| binary tree | 2⌈log₂ p⌉ | up to n·⌈log₂ p⌉ (root) | latency-friendly, bandwidth-poor |
| recursive doubling | log₂ p | n·log₂ p | latency-optimal |
| ring (reduce-scatter + all-gather) | 2(p − 1) | 2n(p − 1)/p | bandwidth-optimal |
| Rabenseifner (halving + doubling) | 2 log₂ p | 2n(p − 1)/p | both, p a power of two |

The bandwidth lower bound for all-reduce is ``2n(p − 1)/p`` per rank: every rank must
send its contribution to (p − 1)/p of the result, and receive the rest. Ring and
Rabenseifner reach it. Recursive doubling trades log p times more bytes for the fewest
rounds, which wins for small messages; where it hands over depends on the fitted
α and β.

``Comm`` wraps ``isend`` / ``irecv`` and counts bytes and rounds per rank, so the tests
can check each closed form exactly.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist


class Comm:
    def __init__(self, group=None):
        self.group = group
        self.rank, self.p = dist.get_rank(group), dist.get_world_size(group)
        self.bytes_sent = 0
        self.rounds = 0
        self.path_bytes = 0  # Σ over rounds of the larger of the sent / received message
        self.reduced = 0  # bytes this rank added into its buffer (the γ term)

    def add(self, dst, src):
        dst += src
        self.reduced += src.numel() * src.element_size()

    def _peer(self, r):  # group rank → global rank for the p2p call
        return r if self.group is None else dist.get_global_rank(self.group, r)

    def round(self, send=None, dst=None, recv=None, src=None):
        """One round: optionally send ``send`` to dst and receive into ``recv`` from src."""
        ops, sb, rb = [], 0, 0
        if send is not None:
            send = send.contiguous()
            ops.append(dist.isend(send, self._peer(dst), self.group))
            sb = send.numel() * send.element_size()
            self.bytes_sent += sb
        if recv is not None:
            ops.append(dist.irecv(recv, self._peer(src), self.group))
            rb = recv.numel() * recv.element_size()
        self.path_bytes += max(sb, rb)
        for op in ops:
            op.wait()
        self.rounds += 1


def chunks(n: int, p: int) -> list[slice]:
    """p near-equal contiguous slices of range(n)."""
    b = [n * i // p for i in range(p + 1)]
    return [slice(b[i], b[i + 1]) for i in range(p)]


# --------------------------------------------------------------------------------- ring
def ring_reduce_scatter(x, c: Comm):
    """After p − 1 rounds rank r holds the full sum of chunk (r + 1) mod p."""
    r, p = c.rank, c.p
    cs = chunks(len(x), p)
    for s in range(p - 1):
        send_i, recv_i = (r - s) % p, (r - s - 1) % p
        buf = torch.empty_like(x[cs[recv_i]])
        c.round(x[cs[send_i]], (r + 1) % p, buf, (r - 1) % p)
        c.add(x[cs[recv_i]], buf)
    return (r + 1) % p


def ring_all_gather(x, c: Comm, owned: int):
    """Each rank starts owning chunk ``owned``; after p − 1 rounds all hold every chunk."""
    r, p = c.rank, c.p
    cs = chunks(len(x), p)
    for s in range(p - 1):
        send_i, recv_i = (owned - s) % p, (owned - s - 1) % p
        buf = torch.empty_like(x[cs[recv_i]])
        c.round(x[cs[send_i]], (r + 1) % p, buf, (r - 1) % p)
        x[cs[recv_i]] = buf


def ring_allreduce(x, c: Comm):
    ring_all_gather(x, c, ring_reduce_scatter(x, c))
    return x


# ------------------------------------------------------------------ recursive doubling
def _pow2(p):
    if p & (p - 1):
        raise ValueError(f"needs a power-of-two world size, got {p}")


def recursive_doubling_allreduce(x, c: Comm):
    _pow2(c.p)
    buf = torch.empty_like(x)
    d = 1
    while d < c.p:
        peer = c.rank ^ d
        c.round(x, peer, buf, peer)
        c.add(x, buf)  # both partners now hold the same partial sum (addition commutes)
        d *= 2
    return x


def rabenseifner_allreduce(x, c: Comm):
    """Recursive-halving reduce-scatter, then recursive-doubling all-gather."""
    _pow2(c.p)
    r, p = c.rank, c.p
    lo, hi = 0, len(x)
    steps = []
    d = p // 2
    while d >= 1:  # halving: keep the half on my side of the split, send the other half
        peer = r ^ d
        mid = (lo + hi) // 2
        keep, give = ((lo, mid), (mid, hi)) if r < peer else ((mid, hi), (lo, mid))
        buf = torch.empty(keep[1] - keep[0], dtype=x.dtype)
        c.round(x[give[0] : give[1]], peer, buf, peer)
        c.add(x[keep[0] : keep[1]], buf)
        steps.append((peer, keep, give))
        lo, hi = keep
        d //= 2
    for peer, keep, give in reversed(steps):  # doubling: swap the halves back
        buf = torch.empty(give[1] - give[0], dtype=x.dtype)
        c.round(x[keep[0] : keep[1]], peer, buf, peer)
        x[give[0] : give[1]] = buf
    return x


# -------------------------------------------------------------------------- tree / naive
def tree_allreduce(x, c: Comm):
    """Binomial-tree reduce to rank 0, then binomial-tree broadcast."""
    r, p = c.rank, c.p
    buf = torch.empty_like(x)
    d = 1
    while d < p:  # reduce: at distance d, ranks with bit d set send to r − d
        if r % (2 * d) == d:
            c.round(send=x, dst=r - d)
        elif r % (2 * d) == 0 and r + d < p:
            c.round(recv=buf, src=r + d)
            c.add(x, buf)
        d *= 2
    d = 1 << max(0, math.ceil(math.log2(p)) - 1) if p > 1 else 0
    while d >= 1:  # broadcast: the same tree, top down
        if r % (2 * d) == 0 and r + d < p:
            c.round(send=x, dst=r + d)
        elif r % (2 * d) == d:
            c.round(recv=x, src=r - d)
        d //= 2
    return x


def naive_allreduce(x, c: Comm):
    """Everyone sends to rank 0, which sums and sends the result back to everyone."""
    r, p = c.rank, c.p
    if r == 0:
        buf = torch.empty_like(x)
        for s in range(1, p):
            c.round(recv=buf, src=s)
            c.add(x, buf)
        for s in range(1, p):
            c.round(send=x, dst=s)
    else:
        c.round(send=x, dst=0)
        c.round(recv=x, src=0)
    return x


ALLREDUCE = {
    "naive": naive_allreduce,
    "tree": tree_allreduce,
    "recursive_doubling": recursive_doubling_allreduce,
    "ring": ring_allreduce,
    "rabenseifner": rabenseifner_allreduce,
}
