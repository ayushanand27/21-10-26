"""Test-like features for training rows: hide one chunk of train at a time, re-position it with the
DP merge exactly like the test rows, compute its features from the remaining labels.
usage: python build2.py DATA_DIR OUT_DIR {sim|final} K"""
import sys, os, time
import numpy as np, pandas as pd
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), p) for p in ('.', '../src', '../scripts')]
from pipeline import *

data_dir = os.path.join(os.environ.get('WORK', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work')), 'data')
t0 = time.time()
tr, te = load(data_dir)
dtr, dte = decode(tr), decode(te)
dtr['day'], dtr['date'] = fix_dates(tr)
dte['day'], dte['date'] = fix_dates(te)
dtr['y'] = tr['target'].values
dtr['src'], dte['src'] = 'train', 'test'
dte['ID'] = te['ID'].values
Str, Ste = standardise(sensor_matrix(dtr), sensor_matrix(dte))
day_tr, day_te = dtr['day'].values, dte['day'].values


def view(lab_idx, hidden_groups, want_idx, with_test=False):
    """lab_idx: train rows with known order+label. hidden_groups: list of train-row index arrays that
    are positioned by DP (in turn), then the real test rows. Returns features for train rows want_idx."""
    seq = np.sort(lab_idx)
    for g in hidden_groups:
        g = np.sort(g)
        pa, pb = merge_timeline(Str[seq], day_tr[seq], Str[g], day_tr[g])
        allr = np.r_[seq, g]
        seq = allr[np.argsort(np.r_[pa, pb], kind='stable')]
    p1, p2 = merge_timeline(Str[seq], day_tr[seq], Ste, day_te)
    pos_tr = np.empty(len(dtr), np.int64)
    pos_tr[seq] = p1
    tl = pd.concat([dtr.assign(pos=pos_tr, row=np.arange(len(dtr))), dte.assign(pos=p2, row=-1)], ignore_index=True)
    tl['is_lab'] = False
    tl.loc[lab_idx, 'is_lab'] = True
    tl = tl.sort_values('pos').reset_index(drop=True)
    tl['minute'] = tl.groupby('day').cumcount() / tl.groupby('day')['x1'].transform('size') * 1440
    keep = np.isin(tl['row'].values, want_idx) if want_idx is not None else (tl['src'] == 'test').values
    if with_test: keep = keep | (tl['src'] == 'test').values
    F = sensor_features(tl, keep)
    lab = tl[tl.is_lab]
    LF = label_features(tl['pos'].values, lab['pos'].values, lab['y'].values, tl['day'].values, lab['day'].values)
    LF.index = tl.index
    return pd.concat([tl.loc[keep, ['row', 'pos', 'day', 'src', 'ID', 'y']], F, LF[keep].astype(np.float32)], axis=1)



rng = np.random.default_rng(42)
allrows = np.arange(len(dtr))
hide = rng.random(len(dtr)) < 0.2
pool, H = allrows[~hide], allrows[hide]
