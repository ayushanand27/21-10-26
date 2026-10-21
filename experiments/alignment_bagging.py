"""Alignment bagging: hold-out features under several DP cost weightings, same model, average predictions."""
import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), p) for p in ('.', '../src', '../scripts')]
import pipeline
import build_lib as B
from softalign import soft_features, SOFT_NAMES
from eval_utils import load_sim, baseline_pred, report, F
t0=time.time()
def log(*a): print(*a, f'[{(time.time()-t0)/60:.1f} min]', flush=True)
trn, ev, cols = load_sim(); del trn
pb = baseline_pred(ev); bst = lgb.Booster(model_file=f'{F}/lgb_sim_soft.txt')
y_all = B.dtr['y'].values.astype(float); L = np.sort(B.pool); Hs = np.sort(B.H)
VARIANTS = {'eq': [1, 1, 1, 1, 1], 'mix': [2, 3, 1, 1, 2], 'x1heavy': [3, 2, .5, .5, 1], 'x2heavy': [1, 4, .5, .5, 1], 'counters': [1, 2, 1.5, 1.5, 2]}
preds = {'base': pb}
for name, w in VARIANTS.items():
    w = np.array(w, float); pipeline.MERGE_W[:] = w           # merge_timeline reads this module global
    V = B.view(B.pool, [B.H], B.H).set_index('row').reindex(ev.row.values)
    sf = pd.DataFrame(soft_features(B.Str[L], B.day_tr[L], y_all[L], B.Str[Hs], B.day_tr[Hs], w, 1.0), columns=SOFT_NAMES, index=Hs).reindex(ev.row.values)
    V[SOFT_NAMES] = sf.values.astype(np.float32)
    preds[name] = bst.predict(V[cols]); log('variant', name)
    report(f'single variant {name} vs base', ev, preds[name], pb, n_boot=100)
pipeline.MERGE_W[:] = np.array([1.0, 2.0, 0.5, 0.5, 1.0])
np.save(f'{F}/alignbag_preds.npy', preds)
allp = np.mean([preds[k] for k in preds], axis=0)
report('ALIGNMENT BAG (base + 5 variants) vs base', ev, allp, pb)
report('ALIGNMENT BAG (base + eq + mix) vs base', ev, np.mean([preds[k] for k in ('base', 'eq', 'mix')], axis=0), pb)
log('DONE')
