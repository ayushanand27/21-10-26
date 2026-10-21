import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), p) for p in ('.', '../src', '../scripts')]
import build_lib as B
from pipeline import label_features, merge_timeline, MERGE_W
from softalign import soft_features, SOFT_NAMES
from sklearn.metrics import average_precision_score, f1_score
S=sys.argv[1]; F=f'{S}/feat'; t0=time.time()
def log(*a): print(*a, f'[{(time.time()-t0)/60:.1f} min]', flush=True)
y_all=B.dtr['y'].values; pool, H = B.pool, B.H
# ---- sim view: features for H + test rows, plus timeline positions of the pool rows
ev2=B.view(pool,[H],H,with_test=True).reset_index(drop=True)
seq=np.sort(pool); g=np.sort(H)
pa,pb=merge_timeline(B.Str[seq],B.day_tr[seq],B.Str[g],B.day_tr[g]); allr=np.r_[seq,g]; seq=allr[np.argsort(np.r_[pa,pb],kind='stable')]
p1,_=merge_timeline(B.Str[seq],B.day_tr[seq],B.Ste,B.day_te); pos_tr=np.empty(len(B.dtr),np.int64); pos_tr[seq]=p1
assert np.all(pos_tr[ev2.row.values[ev2.src.values=='train']]==ev2.pos.values[ev2.src.values=='train'])
# soft features: H from cache, test vs pool
sH=pd.read_parquet(f'{F}/soft_sim_eval_T1.0.parquet')
sT=pd.DataFrame(soft_features(B.Str[np.sort(pool)],B.day_tr[np.sort(pool)],y_all[np.sort(pool)],B.Ste,B.day_te,MERGE_W,1.0),columns=SOFT_NAMES,index=B.dte['ID'].values)
isH=(ev2.src=='train').values
soft=np.where(isH[:,None], sH.reindex(ev2.row.values).values, sT.reindex(ev2.ID.values).values)
ev2[SOFT_NAMES]=soft.astype(np.float32)
log('sim view built', ev2.shape)
# ---- model trained on the usual test-like training features
trS=pd.read_parquet(f'{F}/tl_sim_train.parquet').join(pd.read_parquet(f'{F}/soft_sim_train_T1.0.parquet'),on='row')
meta=['row','pos','day','src','ID','y','chunk']; cols=[c for c in trS.columns if c not in meta]
P=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_child_samples=50,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1,num_threads=4,seed=42)
bst=lgb.train(P,lgb.Dataset(trS[cols],trS.y),1300); bst.save_model(f'{F}/lgb_sim_soft.txt'); del trS
p=bst.predict(ev2[cols]); log('stage-1 predicted')
yH=ev2.y.values[isH].astype(int); dayH=ev2.day.values[isH]; half=(dayH//7)%2
def cf_f1(pp):
    grid=np.linspace(0.05,0.95,181); out=[]
    for a in (0,1):
        A_,B_=half==a,half!=a; t=grid[np.argmax([f1_score(yH[A_],pp[A_]>=g) for g in grid])]; out.append(f1_score(yH[B_],pp[B_]>=t))
    full=max(f1_score(yH,pp>=g) for g in grid)
    return round(np.mean(out),4), round(full,4), round(average_precision_score(yH,pp),4)
log('baseline (stage 1) cross-fitted/full F1, AP:', cf_f1(p[isH]))
LABC=[c for c in cols if c.startswith(('yprev','ynext','yboth','gap_','dist_','day_rate'))]
pos_pool=pos_tr[pool]; day_pool=B.day_tr[pool]
posU=ev2.pos.values; dayU=ev2.day.values
def recompute(pseudo_y, keep_mask, soft=False):
    lp=np.r_[pos_pool, posU[keep_mask]]; ly=np.r_[y_all[pool].astype(float), pseudo_y[keep_mask]]; ld=np.r_[day_pool, dayU[keep_mask]]
    LF=label_features(posU[isH], lp, ly, dayH, ld)
    if soft:   # distances must use hard anomalies
        LFh=label_features(posU[isH], lp, (ly>=0.5).astype(float), dayH, ld)
        for c in ('dist_prev_anom','dist_next_anom','dist_anom'): LF[c]=LFh[c].values
    X=ev2.loc[isH,cols].copy()
    for c in LABC: X[c]=LF[c].values.astype(np.float32)
    return bst.predict(X)
for hi,lo in ((0.95,0.02),(0.9,0.05),(0.8,0.1),(0.6,0.3),(0.5,0.5)):
    keep=(p>=hi)|(p<=lo); py=(p>=0.5).astype(float)
    prec=((py[isH&keep]==ev2.y.values[isH&keep])).mean()
    log(f'hard hi={hi} lo={lo}: labelled share of unlabelled {keep.mean():.3f}, pseudo accuracy on H {prec:.4f}, pos-precision {((ev2.y.values[isH&keep&(py==1)])==1).mean():.3f} ->', cf_f1(recompute(py,keep)))
log('soft (all unlabelled, y=p) ->', cf_f1(recompute(p.astype(float),np.ones(len(p),bool),soft=True)))
