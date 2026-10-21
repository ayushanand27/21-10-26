import sys, time, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score, average_precision_score
S=sys.argv[1]; t0=time.time()
trn=pd.read_parquet(f'{S}/feat/tl_sim_train.parquet'); ev=pd.read_parquet(f'{S}/feat/tl_sim_eval.parquet')
meta=['row','pos','day','src','ID','y','chunk']; cols=[c for c in trn.columns if c not in meta]
print(trn.shape, ev.shape, ev.y.mean())
P=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_child_samples=50,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1,num_threads=4)
ds=lgb.Dataset(trn[cols],trn.y); res={}
b=lgb.train(P,ds,1500,valid_sets=[lgb.Dataset(ev[cols],ev.y)],callbacks=[lgb.record_evaluation(res)],)
for it in (300,600,900,1200,1500):
    p=b.predict(ev[cols],num_iteration=it); ts=np.linspace(0.05,0.95,91); fs=[f1_score(ev.y,p>t) for t in ts]; i=int(np.argmax(fs))
    print(it,'AP',round(average_precision_score(ev.y,p),4),'F1',round(fs[i],4),ts[i], f'{time.time()-t0:.0f}s',flush=True)
