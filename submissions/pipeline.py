"""Shared pipeline: decode sensors, fix dates, rebuild the minute timeline, build features."""
import numpy as np
import pandas as pd
import numba

SENS = ['x1', 'x2', 'x3', 'x4', 'x5']


def load(data_dir):
    tr = pd.read_parquet(f'{data_dir}/train.parquet')
    te = pd.read_parquet(f'{data_dir}/test.parquet')
    tr['target'] = tr['target'].astype(int)
    return tr, te


def decode(df):
    """Undo the exp/log transforms applied by the organisers (verified: the results are integer/regular grids)."""
    out = pd.DataFrame(index=df.index)
    out['x1'] = np.log(df['X1'])                       # 0 .. 1.39 on a 0.002 grid
    out['x2'] = np.exp(df['X2'])                       # ~224 .. 255  (a voltage-like reading)
    out['x3'] = np.round(np.log(df['X3'])).astype(int)  # integer count 0 .. 88
    out['x4'] = np.round(np.log(df['X4'])).astype(int)  # integer count 0 .. 80
    out['x5'] = np.round(np.exp(df['X5'])).astype(int)  # integer 1 .. 32
    return out


def fix_dates(df):
    """Days <= 12 have month/day swapped. File order is the true time order, so walk the day-blocks and
    pick the candidate (original or swapped) that keeps the calendar increasing."""
    blk = (df['Date'] != df['Date'].shift()).cumsum() - 1
    firsts = df['Date'].groupby(blk).first()
    prev, fixed = None, []
    for d in firsts:
        cands = [d]
        if d.day <= 12:
            try:
                cands.append(pd.Timestamp(year=d.year, month=d.day, day=d.month))
            except ValueError:
                pass
        if prev is not None:
            cands = [c for c in cands if c > prev] or cands
            d = min(cands, key=lambda c: abs((c - prev).days))
        fixed.append(d)
        prev = d
    return blk.values, pd.Series(fixed).values[blk.values]


@numba.njit(cache=True)
def _merge_two(A, B, w):
    """Optimal interleaving of two ordered sequences A, B that minimises the total weighted L1 jump
    between consecutive readings (sensors are smooth in time). Returns src[k] in {0: A, 1: B}."""
    n = A.shape[0]
    m = B.shape[0]
    INF = 1e18
    C = np.full((n + 1, m + 1, 2), INF)
    P = np.zeros((n + 1, m + 1, 2), np.int8)
    if n > 0:
        C[1, 0, 0] = 0.0
    if m > 0:
        C[0, 1, 1] = 0.0
    nf = A.shape[1]
    for i in range(n + 1):
        for j in range(m + 1):
            for k in range(2):
                c = C[i, j, k]
                if c >= INF:
                    continue
                if k == 0:
                    last = A[i - 1]
                else:
                    last = B[j - 1]
                if i < n:
                    v = c
                    for t in range(nf):
                        v += w[t] * abs(last[t] - A[i, t])
                    if v < C[i + 1, j, 0]:
                        C[i + 1, j, 0] = v
                        P[i + 1, j, 0] = k
                if j < m:
                    v = c
                    for t in range(nf):
                        v += w[t] * abs(last[t] - B[j, t])
                    if v < C[i, j + 1, 1]:
                        C[i, j + 1, 1] = v
                        P[i, j + 1, 1] = k
    k = 0 if C[n, m, 0] <= C[n, m, 1] else 1
    i, j, L = n, m, n + m
    src = np.empty(n + m, np.int8)
    while i + j > 0:
        L -= 1
        src[L] = k
        pk = P[i, j, k]
        if k == 0:
            i -= 1
        else:
            j -= 1
        k = pk
    return src


MERGE_W = np.array([1.0, 2.0, 0.5, 0.5, 1.0])


def merge_timeline(SA, dayA, SB, dayB):
    """SA/SB: standardised sensor matrices in their own file order; dayA/dayB: (sorted) day ids.
    Returns integer timeline positions for every row of A and B (one global ordering)."""
    posA = np.empty(len(SA), np.int64)
    posB = np.empty(len(SB), np.int64)
    days = np.union1d(dayA, dayB)
    ia0, ia1 = np.searchsorted(dayA, days, 'left'), np.searchsorted(dayA, days, 'right')
    ib0, ib1 = np.searchsorted(dayB, days, 'left'), np.searchsorted(dayB, days, 'right')
    off = 0
    for a0, a1, b0, b1 in zip(ia0, ia1, ib0, ib1):
        src = _merge_two(SA[a0:a1], SB[b0:b1], MERGE_W)
        k = np.arange(len(src)) + off
        posA[a0:a1] = k[src == 0]
        posB[b0:b1] = k[src == 1]
        off += len(src)
    return posA, posB


def standardise(*mats):
    allm = np.vstack(mats)
    mu, sd = allm.mean(0), allm.std(0)
    return [(m - mu) / sd for m in mats]


def sensor_matrix(dec):
    return dec[SENS].to_numpy(np.float64)


# ----------------------------------------------------------------------------- features
class _Store(dict):
    def __init__(self, keep): super().__init__(); self.keep = keep
    def __setitem__(self, k, v): super().__setitem__(k, np.asarray(v, dtype=np.float32)[self.keep])


def sensor_features(tl, keep):
    f = _Store(keep)
    for c in SENS:
        s = tl[c].astype(float)
        f[c] = s
        for k in (1, 2, 3):
            f[f'{c}_lag{k}'] = s.shift(k) - s
            f[f'{c}_lead{k}'] = s.shift(-k) - s
        for w in (5, 15, 61, 241):
            r = s.rolling(w, center=True, min_periods=1)
            m = r.mean()
            f[f'{c}_m{w}'] = m
            f[f'{c}_dev{w}'] = s - m
            f[f'{c}_s{w}'] = r.std()
            if w in (5, 15, 61):
                f[f'{c}_mx{w}'] = r.max()
                f[f'{c}_mn{w}'] = r.min()
        for w in (15, 61):
            f[f'{c}_pdev{w}'] = s - s.rolling(w, min_periods=1).mean()            # vs. past window
            f[f'{c}_fdev{w}'] = s - s[::-1].rolling(w, min_periods=1).mean()[::-1]  # vs. future window
        f[f'{c}_ewm'] = s - s.ewm(span=10).mean()
        g = s.groupby(tl['day'])
        dm, ds = g.transform('mean'), g.transform('std')
        f[f'{c}_daymean'] = dm
        f[f'{c}_daystd'] = ds
        f[f'{c}_dayz'] = (s - dm) / (ds + 1e-9)
        f[f'{c}_dayrank'] = g.rank(pct=True)
    x2 = tl['x2']
    for w in (15, 61):
        f[f'x2_drop_past{w}'] = x2.rolling(w, min_periods=1).max() - x2
        f[f'x2_drop_fut{w}'] = x2[::-1].rolling(w, min_periods=1).max()[::-1] - x2
    f['x2_absdiff_s15'] = x2.diff().abs().rolling(15, center=True, min_periods=1).mean()
    f['x3_pos'] = (tl['x3'] > 0).astype(int)
    f['x4_pos'] = (tl['x4'] > 0).astype(int)
    f['x34'] = tl['x3'] + tl['x4']
    f['x3_runlen'] = _run_len(tl['x3'].to_numpy() > 0)
    f['x4_runlen'] = _run_len(tl['x4'].to_numpy() > 0)
    f['x3_runlen_b'] = _run_len(tl['x3'].to_numpy()[::-1] > 0)[::-1]
    f['x4_runlen_b'] = _run_len(tl['x4'].to_numpy()[::-1] > 0)[::-1]
    is1 = (tl['x1'] == 0).astype(int); f['x1_is1'] = is1
    f['x1_is1_m61'] = is1.rolling(61, center=True, min_periods=1).mean()
    # calendar / trend
    dt = pd.to_datetime(tl['date'])
    f['minute'] = tl['minute']
    f['month'] = dt.dt.month
    f['doy'] = dt.dt.dayofyear
    f['dow'] = dt.dt.dayofweek
    f['t_days'] = (dt - dt.min()).dt.days
    f['day_n'] = tl.groupby('day')['x1'].transform('size')
    return pd.DataFrame(dict(f), index=tl.index[keep])


def _run_len(b):
    """Length of the current run of True values (0 where False)."""
    out = np.zeros(len(b), np.int64)
    run = 0
    for i, v in enumerate(b):
        run = run + 1 if v else 0
        out[i] = run
    return out


def label_features(pos_all, lab_pos, lab_y, day_all, lab_day):
    """Neighbour-label features from the *labelled* rows only, always excluding the row itself.
    pos_all: timeline position of every row we need features for; lab_pos/lab_y: positions & labels
    of the labelled rows (sorted by position)."""
    o = np.argsort(lab_pos)
    P, Y = lab_pos[o], lab_y[o].astype(float)
    CS = np.r_[0, np.cumsum(Y)]
    n = len(P)
    kb = np.searchsorted(P, pos_all, 'left')   # labelled rows strictly before
    ka = np.searchsorted(P, pos_all, 'right')  # first labelled row strictly after
    f = {}
    for k in (1, 2, 3, 5, 10, 30, 120):
        lo = np.maximum(kb - k, 0)
        hi = np.minimum(ka + k, n)
        f[f'yprev{k}'] = (CS[kb] - CS[lo]) / np.maximum(kb - lo, 1)
        f[f'ynext{k}'] = (CS[hi] - CS[ka]) / np.maximum(hi - ka, 1)
        f[f'yboth{k}'] = (CS[kb] - CS[lo] + CS[hi] - CS[ka]) / np.maximum(kb - lo + hi - ka, 1)
    # gap (timeline steps) to the nearest labelled neighbour on each side
    f['gap_prev'] = np.where(kb > 0, pos_all - P[np.maximum(kb - 1, 0)], 999)
    f['gap_next'] = np.where(ka < n, P[np.minimum(ka, n - 1)] - pos_all, 999)
    # distance to nearest labelled anomaly on each side
    AP = P[Y == 1]
    jb = np.searchsorted(AP, pos_all, 'left')
    ja = np.searchsorted(AP, pos_all, 'right')
    f['dist_prev_anom'] = np.where(jb > 0, pos_all - AP[np.maximum(jb - 1, 0)], 10**6).clip(max=5000)
    f['dist_next_anom'] = np.where(ja < len(AP), AP[np.minimum(ja, len(AP) - 1)] - pos_all, 10**6).clip(max=5000)
    f['dist_anom'] = np.minimum(f['dist_prev_anom'], f['dist_next_anom'])
    f = pd.DataFrame(f)
    # leave-self-out daily anomaly rate
    lab_df = pd.DataFrame({'day': lab_day, 'y': lab_y})
    dsum = lab_df.groupby('day')['y'].sum()
    dcnt = lab_df.groupby('day')['y'].size()
    s = pd.Series(day_all).map(dsum).fillna(0).to_numpy()
    c = pd.Series(day_all).map(dcnt).fillna(0).to_numpy()
    is_lab = np.isin(pos_all, P)
    # subtract self for labelled rows
    self_y = np.zeros(len(pos_all))
    idx = np.searchsorted(P, pos_all)
    idx = np.minimum(idx, n - 1)
    self_y[is_lab] = Y[idx[is_lab]]
    f['day_rate'] = (s - self_y) / np.maximum(c - is_lab, 1)
    return f
