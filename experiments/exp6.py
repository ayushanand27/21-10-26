import sys, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score, average_precision_score
S=sys.argv[1]
trn=pd.read_parquet(f'{S}/feat/tl_sim_train.parquet'); ev=pd.read_parquet(f'{S}/feat/tl_sim_eval.parquet')
meta=['row','pos','day','src','ID','y','chunk']; cols=[c for c in trn.columns if c not in meta]
P=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_child_samples=50,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1,num_threads=4)
# OOF preds for training rows by chunk (needed later for stacking tests), full-model preds for hold-out
oof=np.zeros(len(trn))
for k in range(0,10,2):   # 5 folds of 2 chunks
    m=trn.chunk.isin([k,k+1]).values
    b=lgb.train(P,lgb.Dataset(trn.loc[~m,cols],trn.y[~m]),900); oof[m]=b.predict(trn.loc[m,cols]); print('fold',k,flush=True)
b=lgb.train(P,lgb.Dataset(trn[cols],trn.y),900); p=b.predict(ev[cols])
trn[['row','pos','chunk','y']].assign(p=oof).to_parquet(f'{S}/feat/oof_sim.parquet'); ev[['row','pos','y','day','dist_anom','yboth1','gap_prev','gap_next']].assign(p=p).to_parquet(f'{S}/feat/pred_sim.parquet')
def best(y,p):
    ts=np.linspace(0.05,0.95,91); fs=[f1_score(y,p>t) for t in ts]; i=int(np.argmax(fs)); return round(fs[i],4),round(ts[i],2)
print('holdout AP',average_precision_score(ev.y,p),'F1',best(ev.y,p),'| OOF AP',average_precision_score(trn.y,oof),'F1',best(trn.y,oof))
