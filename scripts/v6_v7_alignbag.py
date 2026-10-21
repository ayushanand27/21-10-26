"""v6 = alignment bag (base, eq, mix DP weights) of LightGBM x3; v7 = avg(v6, v3). Thresholds from the hold-out."""
import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), p) for p in ('.', '../src', '../scripts')]
import pipeline
import build_lib as B
from softalign import soft_features, SOFT_NAMES
from eval_utils import P, ROUNDS, F
S = os.path.dirname(F); OUT = f'{S}/share'; t0 = time.time()
def log(*a): print(*a, f'[{(time.time()-t0)/60:.1f} min]', flush=True)
trF = pd.read_parquet(f'{F}/tl_final_train.parquet').join(pd.read_parquet(f'{F}/soft_final_train_T1.0.parquet'), on='row')
cols = lgb.Booster(model_file=f'{F}/lgb_sim_soft.txt').feature_name()
models = []
for s in range(3):
    m = f'{F}/lgb_final_soft_s{s}.txt'
    if os.path.exists(m): b = lgb.Booster(model_file=m)
    else: b = lgb.train({**P, 'seed': 42 + s}, lgb.Dataset(trF[cols], trF.y), ROUNDS); b.save_model(m)
    models.append(b); log('final model', s)
del trF
ids = B.dte['ID'].values; y_all = B.dtr['y'].values.astype(float); allr = np.arange(len(B.dtr))
teF = pd.read_parquet(f'{F}/tl_final_eval.parquet').join(pd.read_parquet(f'{F}/soft_final_test_T1.0.parquet'), on='ID').set_index('ID').reindex(ids)
preds = {'base': np.mean([b.predict(teF[cols]) for b in models], 0)}; log('base predicted')
for name, w in {'eq': [1, 1, 1, 1, 1], 'mix': [2, 3, 1, 1, 2]}.items():
    w = np.array(w, float); pipeline.MERGE_W[:] = w
    V = B.view(allr, [], None).set_index('ID').reindex(ids)
    V[SOFT_NAMES] = soft_features(B.Str, B.day_tr, y_all, B.Ste, B.day_te, w, 1.0).astype(np.float32)
    preds[name] = np.mean([b.predict(V[cols]) for b in models], 0); log('variant', name)
pipeline.MERGE_W[:] = np.array([1.0, 2.0, 0.5, 0.5, 1.0])
np.save(f'{F}/v6_test_preds.npy', preds)
v6 = np.mean([preds[k] for k in ('base', 'eq', 'mix')], 0)
v3 = pd.read_csv(f'{F}/v3_test_probs.csv').set_index('ID')['p_v3'].reindex(ids).values
v2a = np.load(f'{F}/v2_test.npy', allow_pickle=True).item()['lgb']
print('corr base-pred vs previous v2a test probs (sanity, should be ~1):', np.corrcoef(preds['base'], pd.Series(v2a, index=pd.read_parquet(f'{F}/tl_final_eval.parquet', columns=['ID']).ID.values).reindex(ids).values)[0, 1])
for name, p, t in (('submission_v6_alignbag', v6, 0.465), ('submission_v7_alignbag_plus_v3', (v6 + v3) / 2, 0.555)):
    sub = pd.DataFrame({'ID': ids, 'target': (p >= t).astype(int).astype(str)}); sub.to_csv(f'{OUT}/{name}.csv', index=False)
    prev = pd.read_csv(f'{OUT}/submission_v4_blend_v2_v3.csv').set_index('ID').target.reindex(ids).values
    log(name, 'thr', t, 'pred rate %.4f%%' % ((p >= t).mean() * 100), sub.target.value_counts().to_dict(), '| rows differing from v4:', int((sub.target.astype(int).values != prev).sum()))
log('DONE')
