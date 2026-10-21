"""Soft (posterior) alignment: forward-backward over the same interleaving lattice as the DP merge.
For every row of B we get P(i) = probability that exactly i rows of A precede it, and from it
expected neighbour-label features (expected label of previous/next labelled minutes, boundary prob...)."""
import numpy as np
import numba

KS = (1, 2, 3, 5, 10, 30)
SOFT_NAMES = ([f's_prev{k}' for k in KS] + [f's_next{k}' for k in KS] + [f's_both{k}' for k in KS] +
              ['s_inside', 's_boundary', 's_pmax', 's_entropy', 's_pos_std'])
NF = len(SOFT_NAMES)


@numba.njit(cache=True)
def _lae(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    if a > b:
        return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


@numba.njit(cache=True)
def _dist(x, y, w):
    s = 0.0
    for t in range(x.shape[0]):
        s += w[t] * abs(x[t] - y[t])
    return s


@numba.njit(cache=True)
def soft_day(A, B, yA, w, T):
    n, m = A.shape[0], B.shape[0]
    out = np.full((m, 23), np.nan)
    if m == 0 or n == 0:
        return out
    NI = -np.inf
    F = np.full((n + 1, m + 1, 2), NI)
    G = np.full((n + 1, m + 1, 2), NI)
    F[1, 0, 0] = 0.0
    F[0, 1, 1] = 0.0
    for i in range(n + 1):                      # forward (push)
        for j in range(m + 1):
            for k in range(2):
                v = F[i, j, k]
                if v == NI:
                    continue
                last = A[i - 1] if k == 0 else B[j - 1]
                if i < n:
                    F[i + 1, j, 0] = _lae(F[i + 1, j, 0], v - _dist(last, A[i], w) / T)
                if j < m:
                    F[i, j + 1, 1] = _lae(F[i, j + 1, 1], v - _dist(last, B[j], w) / T)
    G[n, m, 0] = 0.0
    G[n, m, 1] = 0.0
    for i in range(n, -1, -1):                  # backward
        for j in range(m, -1, -1):
            for k in range(2):
                if i == n and j == m:
                    continue
                if (k == 0 and i == 0) or (k == 1 and j == 0):
                    continue
                last = A[i - 1] if k == 0 else B[j - 1]
                v = NI
                if i < n:
                    v = _lae(v, G[i + 1, j, 0] - _dist(last, A[i], w) / T)
                if j < m:
                    v = _lae(v, G[i, j + 1, 1] - _dist(last, B[j], w) / T)
                G[i, j, k] = v
    logZ = _lae(F[n, m, 0], F[n, m, 1])
    CS = np.zeros(n + 1)
    for i in range(n):
        CS[i + 1] = CS[i] + yA[i]
    P = np.empty(n + 1)
    for j in range(m):
        tot = 0.0
        for i in range(n + 1):
            P[i] = np.exp(F[i, j + 1, 1] + G[i, j + 1, 1] - logZ)
            tot += P[i]
        if tot <= 0:
            continue
        for i in range(n + 1):
            P[i] /= tot
        acc = np.zeros(23)
        pmax = 0.0; ent = 0.0; mu = 0.0; m2 = 0.0
        for i in range(n + 1):
            p = P[i]
            if p < 1e-9:
                continue
            for q in range(6):
                k = KS[q]
                lo = max(i - k, 0); hi = min(i + k, n)
                pv = (CS[i] - CS[lo]) / max(i - lo, 1)
                nv = (CS[hi] - CS[i]) / max(hi - i, 1)
                bv = (CS[i] - CS[lo] + CS[hi] - CS[i]) / max(hi - lo, 1)
                acc[q] += p * pv; acc[6 + q] += p * nv; acc[12 + q] += p * bv
            yp = yA[i - 1] if i > 0 else 0.0
            yn = yA[i] if i < n else 0.0
            acc[18] += p * yp * yn                     # sits inside a labelled episode
            acc[19] += p * (yp + yn - 2 * yp * yn)     # sits on an episode boundary
            if p > pmax:
                pmax = p
            ent -= p * np.log(p)
            mu += p * i; m2 += p * i * i
        acc[20] = pmax; acc[21] = ent; acc[22] = np.sqrt(max(m2 - mu * mu, 0.0))
        out[j] = acc
    return out


def soft_features(SA, dayA, yA, SB, dayB, w, T):
    """SA/SB standardised sensors (day-sorted, own order), yA labels of A. Returns (len(SB), NF)."""
    out = np.full((len(SB), NF), np.nan, np.float32)
    days = np.union1d(dayA, dayB)
    ia0, ia1 = np.searchsorted(dayA, days, 'left'), np.searchsorted(dayA, days, 'right')
    ib0, ib1 = np.searchsorted(dayB, days, 'left'), np.searchsorted(dayB, days, 'right')
    for a0, a1, b0, b1 in zip(ia0, ia1, ib0, ib1):
        if b1 > b0:
            out[b0:b1] = soft_day(SA[a0:a1], SB[b0:b1], yA[a0:a1].astype(np.float64), w, T)
    return out
