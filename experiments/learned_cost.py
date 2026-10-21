"""Learned (likelihood) alignment cost: -log P(delta_k) per sensor from empirical lag-1 distributions."""
import numpy as np, numba
OFF = 3200; WIDTH = 2 * OFF + 1

def int_codes(dec):
    return np.column_stack([np.round(dec['x1'].values * 500), np.round(dec['x2'].values * 100),
                            dec['x3'].values, dec['x4'].values, dec['x5'].values]).astype(np.int64)

def fit_tables(I, day, smooth=0.5):
    """I: int codes in true time order (train file order); lag-1 deltas within each day."""
    same = day[1:] == day[:-1]
    T = np.zeros((5, WIDTH))
    for k in range(5):
        d = np.clip(I[1:, k] - I[:-1, k], -OFF, OFF)[same] + OFF
        c = np.bincount(d, minlength=WIDTH).astype(float)
        p = (c + smooth) / (c.sum() + smooth * WIDTH)
        T[k] = -np.log(p)
    T -= T.min(axis=1, keepdims=True)          # zero cost for the most likely change
    return T

@numba.njit(cache=True)
def _cost(a, b, T, alpha):
    s = 0.0
    for k in range(5):
        d = b[k] - a[k]
        if d > 3200: d = 3200
        if d < -3200: d = -3200
        s += alpha[k] * T[k, d + 3200]
    return s

@numba.njit(cache=True)
def merge_learned(A, B, T, alpha):
    n, m = A.shape[0], B.shape[0]; INF = 1e18
    C = np.full((n + 1, m + 1, 2), INF); P = np.zeros((n + 1, m + 1, 2), np.int8)
    if n > 0: C[1, 0, 0] = 0.0
    if m > 0: C[0, 1, 1] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            for k in range(2):
                c = C[i, j, k]
                if c >= INF: continue
                last = A[i - 1] if k == 0 else B[j - 1]
                if i < n:
                    v = c + _cost(last, A[i], T, alpha)
                    if v < C[i + 1, j, 0]: C[i + 1, j, 0] = v; P[i + 1, j, 0] = k
                if j < m:
                    v = c + _cost(last, B[j], T, alpha)
                    if v < C[i, j + 1, 1]: C[i, j + 1, 1] = v; P[i, j + 1, 1] = k
    k = 0 if C[n, m, 0] <= C[n, m, 1] else 1
    i, j, L = n, m, n + m; src = np.empty(n + m, np.int8)
    while i + j > 0:
        L -= 1; src[L] = k; pk = P[i, j, k]
        if k == 0: i -= 1
        else: j -= 1
        k = pk
    return src

@numba.njit(cache=True)
def merge_mixed(As, Bs, w, Ai, Bi, T, alpha, lam):
    """hand-made weighted L1 on standardised sensors + lam * learned -log P(delta)."""
    n, m = As.shape[0], Bs.shape[0]; INF = 1e18
    C = np.full((n + 1, m + 1, 2), INF); P = np.zeros((n + 1, m + 1, 2), np.int8)
    if n > 0: C[1, 0, 0] = 0.0
    if m > 0: C[0, 1, 1] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            for k in range(2):
                c = C[i, j, k]
                if c >= INF: continue
                if k == 0:
                    ls = As[i - 1]; li = Ai[i - 1]
                else:
                    ls = Bs[j - 1]; li = Bi[j - 1]
                if i < n:
                    v = c + lam * _cost(li, Ai[i], T, alpha)
                    for t in range(5): v += w[t] * abs(ls[t] - As[i, t])
                    if v < C[i + 1, j, 0]: C[i + 1, j, 0] = v; P[i + 1, j, 0] = k
                if j < m:
                    v = c + lam * _cost(li, Bi[j], T, alpha)
                    for t in range(5): v += w[t] * abs(ls[t] - Bs[j, t])
                    if v < C[i, j + 1, 1]: C[i, j + 1, 1] = v; P[i, j + 1, 1] = k
    k = 0 if C[n, m, 0] <= C[n, m, 1] else 1
    i, j, L = n, m, n + m; src = np.empty(n + m, np.int8)
    while i + j > 0:
        L -= 1; src[L] = k; pk = P[i, j, k]
        if k == 0: i -= 1
        else: j -= 1
        k = pk
    return src
