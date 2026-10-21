import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import load, decode, fix_dates, sensor_matrix, standardise, MERGE_W
from softalign import soft_features, SOFT_NAMES
from sklearn.metrics import average_precision_score, f1_score
S=sys.argv[1]; t0=time.time()
tr,te=load(f'{S}/data'); dtr,dte=decode(tr),decode(te); dtr['day'],_=fix_dates(tr); dte['day'],_=fix_dates(te)
Str,Ste=standardise(sensor_matrix(dtr),sensor_matrix(dte)); y=tr.target.values; day=dtr.day.values
hide=np.random.default_rng(42).random(len(dtr))<0.2; pool=np.flatnonzero(~hide); H=np.flatnonzero(hide)
for T in (2.0,4.0):
    sf=soft_features(Str[pool],day[pool],y[pool],Str[H],day[H],MERGE_W,T)
    print('T',T,'s_both1 AP',round(average_precision_score(y[H],np.nan_to_num(sf[:,SOFT_NAMES.index('s_both1')])),4),flush=True)
T=float(sys.argv[2]) if len(sys.argv)>2 else 1.0
chunk=np.random.default_rng(7).integers(0,10,len(pool))
parts=[]
for k in range(10):
    ck=pool[chunk==k]; lab=pool[chunk!=k]
    parts.append(pd.DataFrame(soft_features(Str[lab],day[lab],y[lab],Str[ck],day[ck],MERGE_W,T),columns=SOFT_NAMES).assign(row=ck))
SF_tr=pd.concat(parts).set_index('row')
SF_h=pd.DataFrame(soft_features(Str[pool],day[pool],y[pool],Str[H],day[H],MERGE_W,T),columns=SOFT_NAMES).assign(row=H).set_index('row')
SF_tr.to_parquet(f'{S}/feat/soft_sim_train_T{T}.parquet'); SF_h.to_parquet(f'{S}/feat/soft_sim_eval_T{T}.parquet')
print('soft feats built',f'{time.time()-t0:.0f}s',flush=True)
trn=pd.read_parquet(f'{S}/feat/tl_sim_train.parquet'); ev=pd.read_parquet(f'{S}/feat/tl_sim_eval.parquet')
meta=['row','pos','day','src','ID','y','chunk']; base=[c for c in trn.columns if c not in meta]
trn=trn.join(SF_tr,on='row'); ev=ev.join(SF_h,on='row')
P=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_child_samples=50,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1,num_threads=4,seed=42)
def best(yy,pp):
    ts=np.linspace(0.05,0.95,91); fs=[f1_score(yy,pp>t) for t in ts]; i=int(np.argmax(fs)); return round(fs[i],4),round(ts[i],2)
b=lgb.train(P,lgb.Dataset(trn[base+SOFT_NAMES],trn.y),1200); p=b.predict(ev[base+SOFT_NAMES])
print(f'T={T} base+soft: AP={average_precision_score(ev.y,p):.4f} F1={best(ev.y,p)}  (baseline AP=0.9491 F1=0.8836)',f'{time.time()-t0:.0f}s',flush=True)
np.save(f'{S}/feat/p_soft_sim_T{T}.npy',p)
imp=pd.Series(b.feature_importance('gain'),base+SOFT_NAMES).sort_values(ascending=False); print(imp.head(12).round(0).to_dict())
