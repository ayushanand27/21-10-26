"""Shared, fixed evaluation protocol for hold-out experiments (sim world).
Usage:
    import sys; sys.path.insert(0, 'src'); from eval_utils import *     # WORK env var = work directory
    trn, ev, cols = load_sim()              # fit-pool rows (test-like OOF feats + soft), hold-out rows, model feature list
    p_base = baseline_pred(ev)              # cached baseline LightGBM (same params/seed/rounds as P, 1300 rounds)
    report('my idea', ev, p_new, p_base)   # paired comparison: AP, full F1, cross-fitted F1, week-block bootstrap
"""
import os, numpy as np, pandas as pd, lightgbm as lgb, warnings
from sklearn.metrics import average_precision_score, f1_score
warnings.filterwarnings('ignore')
S = os.environ.get('WORK', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work'))
F = f'{S}/feat'
META = ['row', 'pos', 'day', 'src', 'ID', 'y', 'chunk']
P = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_child_samples=50, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4, seed=42)
ROUNDS = 1300
GRID = np.linspace(0.05, 0.95, 181)

def load_sim(columns=None):
    trn = pd.read_parquet(f'{F}/tl_sim_train.parquet', columns=columns).join(pd.read_parquet(f'{F}/soft_sim_train_T1.0.parquet'), on='row')
    ev = pd.read_parquet(f'{F}/tl_sim_eval.parquet', columns=columns).join(pd.read_parquet(f'{F}/soft_sim_eval_T1.0.parquet'), on='row')
    cols = lgb.Booster(model_file=f'{F}/lgb_sim_soft.txt').feature_name()
    return trn, ev, cols

def baseline_pred(ev):
    cache = f'{F}/baseline_hold_pred.npy'
    if os.path.exists(cache):
        p = np.load(cache)
        if len(p) == len(ev): return p
    b = lgb.Booster(model_file=f'{F}/lgb_sim_soft.txt'); p = b.predict(ev[b.feature_name()]); np.save(cache, p); return p

def best_thr(y, p):
    f = [f1_score(y, p >= t) for t in GRID]; i = int(np.argmax(f)); return GRID[i], f[i]

def cross_fitted_f1(y, p, day):
    half = (np.asarray(day) // 7) % 2; out = []
    for a in (0, 1):
        A, B = half == a, half != a; t, _ = best_thr(y[A], p[A]); out.append(f1_score(y[B], p[B] >= t))
    return float(np.mean(out))

def report(name, ev, p_new, p_base, n_boot=200, seed=0, mask=None):
    """Paired comparison on the hold-out. mask: optional subset of hold-out rows (bool array)."""
    y = ev['y'].values.astype(int); day = ev['day'].values
    if mask is not None: y, day, p_new, p_base = y[mask], day[mask], p_new[mask], p_base[mask]
    tn, fn_ = best_thr(y, p_new); tb, fb = best_thr(y, p_base)
    cfn, cfb = cross_fitted_f1(y, p_new, day), cross_fitted_f1(y, p_base, day)
    wk = day // 7; W = np.unique(wk); r = np.random.default_rng(seed); groups = {w: np.flatnonzero(wk == w) for w in W}; d = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[w] for w in r.choice(W, len(W), replace=True)])
        d.append(f1_score(y[idx], p_new[idx] >= tn) - f1_score(y[idx], p_base[idx] >= tb))
    d = np.array(d)
    res = dict(name=name, AP_new=round(average_precision_score(y, p_new), 4), AP_base=round(average_precision_score(y, p_base), 4),
               F1_new=round(fn_, 4), F1_base=round(fb, 4), CF_F1_new=round(cfn, 4), CF_F1_base=round(cfb, 4),
               CF_gain=round(cfn - cfb, 4), boot_mean_gain=round(float(d.mean()), 4),
               boot_CI90=[round(float(np.quantile(d, .05)), 4), round(float(np.quantile(d, .95)), 4)],
               P_gain_pos=round(float((d > 0).mean()), 3), thr_new=round(float(tn), 3), thr_base=round(float(tb), 3))
    res['VERDICT'] = 'KEEP' if (res['CF_gain'] >= 0.0015 and res['P_gain_pos'] >= 0.8) else 'REJECT'
    print(res, flush=True)
    return res

def train_lgb(X, y, params=None, rounds=ROUNDS):
    return lgb.train({**P, **(params or {})}, lgb.Dataset(X, y), rounds)
