"""
grouping.py -- fewer measurement circuits via GENERAL-commuting (GC) grouping.

The hardware run used qubit-wise-commuting (QWC) groups: 28 circuits for the 63
Pauli strings of H1 and H2 at 3 qubits.  QWC cannot do better than 27 here,
because every one of the 27 weight-3 strings needs its own local basis.

General commutation is weaker than qubit-wise commutation, so groups can be much
larger.  For n qubits all 4^n - 1 non-identity Paulis partition into 2^n + 1
commuting families of 2^n - 1 (mutually unbiased bases): 9 circuits instead of
28 at 3 qubits.  The price is a short Clifford circuit that rotates each family
onto Z-strings, which adds two-qubit gates (found by search, minimising CX count).

Measured trade-off: see the accompanying paper (noise and mitigation section).
"""
import heapq
import itertools
import math

import numpy as np

from .hardware_run import _cx_matrix, _embed, _mat1

PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
         "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0, -1.0])}


def pmat(label):
    M = np.array([[1.0 + 0j]])
    for ch in label:
        M = np.kron(M, PAULI[ch])
    return M


def symp(label):
    """(x bits, z bits) of a Pauli label."""
    x = tuple(int(c in "XY") for c in label)
    z = tuple(int(c in "ZY") for c in label)
    return x, z


def commute(a, b):
    xa, za = symp(a)
    xb, zb = symp(b)
    return (sum(i * j for i, j in zip(xa, zb)) + sum(i * j for i, j in zip(za, xb))) % 2 == 0


def gc_partition(labels):
    """
    Partition Pauli labels into mutually commuting families with as few families
    as possible.  Exact-cover search over maximal commuting sets (fine for n <= 3;
    falls back to greedy colouring above that).
    """
    labels = list(labels)
    n = len(labels[0])
    if n <= 3:
        alls = ["".join(t) for t in itertools.product("IXYZ", repeat=n)][1:]
        fams = set()
        for gens in itertools.combinations(alls, n):
            if all(commute(a, b) for a, b in itertools.combinations(gens, 2)):
                grp = {_mul_label(c) for c in _span(gens)} - {"I" * n, None}
                if len(grp) == 2 ** n - 1:
                    fams.add(frozenset(grp))
        fams = list(fams)
        target = set(labels)
        best = [None]

        def search(rem, chosen):
            if best[0] is not None and len(chosen) >= len(best[0]):
                return
            if not rem:
                best[0] = list(chosen)
                return
            p = min(rem)
            for f in fams:
                if p in f and not (f & set().union(*chosen) if chosen else False):
                    search(rem - f, chosen + [f])
        search(target, [])
        return [sorted(set(f) & target) for f in best[0]]
    groups = []
    for lab in sorted(labels, key=lambda s: -sum(c != "I" for c in s)):
        for g in groups:
            if all(commute(lab, o) for o in g):
                g.append(lab)
                break
        else:
            groups.append([lab])
    return groups


def _span(gens):
    out = []
    for bits in itertools.product((0, 1), repeat=len(gens)):
        out.append(tuple(g for g, b in zip(gens, bits) if b))
    return out


def _mul_label(factors):
    n = len(factors[0]) if factors else 0
    if not factors:
        return None
    x = [0] * n
    z = [0] * n
    for f in factors:
        fx, fz = symp(f)
        x = [a ^ b for a, b in zip(x, fx)]
        z = [a ^ b for a, b in zip(z, fz)]
    return "".join("I" if (a, b) == (0, 0) else "X" if (a, b) == (1, 0)
                   else "Z" if (a, b) == (0, 1) else "Y" for a, b in zip(x, z))


# ---------------------------------------------------------------- diagonalising Cliffords
def _apply_sym(gate, vecs, n):
    out = []
    for x, z in vecs:
        x, z = list(x), list(z)
        if gate[0] == "h":
            q = gate[1]; x[q], z[q] = z[q], x[q]
        elif gate[0] == "s":
            q = gate[1]; z[q] ^= x[q]
        elif gate[0] == "cx":
            c, t = gate[1], gate[2]; x[t] ^= x[c]; z[c] ^= z[t]
        out.append((tuple(x), tuple(z)))
    return tuple(out)


def diagonalising_circuit(family, n):
    """Cheapest {H, S, CX} circuit (CX cost 1, single-qubit 0.01) mapping every
    member of a commuting family onto a Z-string."""
    gens = [symp(l) for l in family]
    moves = [("h", q) for q in range(n)] + [("s", q) for q in range(n)] + \
            [("cx", c, t) for c in range(n) for t in range(n) if c != t]
    start = tuple(gens)
    pq = [(0.0, 0, start, ())]
    seen = {start: 0.0}
    k = 0
    while pq:
        cost, _, st, path = heapq.heappop(pq)
        if all(not any(x) for x, _ in st):
            return list(path)
        if cost > 6:
            break
        for m in moves:
            nst = _apply_sym(m, st, n)
            nc = cost + (1.0 if m[0] == "cx" else 0.01)
            if nst not in seen or nc < seen[nst] - 1e-12:
                seen[nst] = nc
                k += 1
                heapq.heappush(pq, (nc, k, nst, path + (m,)))
    raise RuntimeError("no diagonalising circuit found")


def to_sim_gates(path):
    """Map {h, s, cx} to the simulator's gate set (S = RZ(pi/2) up to global phase)."""
    return [("h", g[1]) if g[0] == "h" else ("rz", g[1], math.pi / 2) if g[0] == "s"
            else ("cx", g[1], g[2]) for g in path]


def unitary(gates, n):
    U = np.eye(2 ** n, dtype=complex)
    for g in gates:
        if g[0] == "cx":
            G = _cx_matrix(g[1], g[2], n)
        else:
            G = _embed(_mat1(g[0], g[2] if len(g) > 2 else None), g[1], n)
        U = G @ U
    return U


def z_image(label, U, n):
    """U P U^dag = sign * Z-string.  Returns (sign, zlabel)."""
    M = U @ pmat(label) @ U.conj().T
    for zl in ("".join(t) for t in itertools.product("IZ", repeat=n)):
        c = np.trace(pmat(zl) @ M) / 2 ** n
        if abs(abs(c) - 1) < 1e-9:
            return (1.0 if np.real(c) > 0 else -1.0), zl
    raise RuntimeError(f"{label} is not diagonalised")
