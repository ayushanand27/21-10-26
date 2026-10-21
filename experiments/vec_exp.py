# exact sensor-vector leave-out label statistics (out-of-fold, same chunks as the other features)
import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import load, decode
from sklearn.metrics import average_precision_score, f1_score
S=sys.argv[1]; t0=time.time()
tr,te=load(f'{S}/data'); dtr=decode(tr); y=tr.target.values
def keys(d):
    x1=np.round(d.x1.values*1000).astype(np.int64); x2=np.round(d.x2.values*100).astype(np.int64)
    k1=(((x1*40000+x2)*100+d.x3.values)*100+d.x4.values)*40+d.x5.values                      # exact vector
    k2=((np.round(d.x2.values).astype(np.int64)*100+d.x3.values)*100+d.x4.values)*40+d.x5.values  # coarse: int voltage + counters + level
    k3=(np.minimum(d.x3.values,99)*100+d.x4.values)*40+d.x5.values                              # counters + level only
    return [k1,k2,k3]
K=keys(dtr); PI=y.mean(); A=20.0
def stats(lab_rows, tgt_rows):
    out={}
    for n,k in zip(('v1','v2','v3'),K):
        g=pd.DataFrame({'k':k[lab_rows],'y':y[lab_rows]}).groupby('k')['y'].agg(['sum','size'])
        s=pd.Series(k[tgt_rows]).map(g['sum']).fillna(0).values; c=pd.Series(k[tgt_rows]).map(g['size']).fillna(0).values
        out[f'{n}_rate']=(s+A*PI)/(c+A); out[f'{n}_logcnt']=np.log1p(c)
    return pd.DataFrame(out,index=tgt_rows).astype(np.float32)
trn=pd.read_parquet(f'{S}/feat/tl_sim_train.parquet').join(pd.read_parquet(f'{S}/feat/soft_sim_train_T1.0.parquet'),on='row')
ev=pd.read_parquet(f'{S}/feat/tl_sim_eval.parquet').join(pd.read_parquet(f'{S}/feat/soft_sim_eval_T1.0.parquet'),on='row')
pool=np.sort(trn.row.values)
V=pd.concat([stats(np.setdiff1d(pool,trn.row.values[trn.chunk.values==k]), trn.row.values[trn.chunk.values==k]) for k in range(10)])
trn=trn.join(V,on='row'); ev=ev.join(stats(pool,ev.row.values),on='row')
VN=list(V.columns); meta=['row','pos','day','src','ID','y','chunk']; base=[c for c in trn.columns if c not in meta+VN]
print('built',f'{time.time()-t0:.0f}s', 'single-feature AP v1_rate', round(average_precision_score(ev.y,ev.v1_rate),4),flush=True)
P=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_child_samples=50,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1,num_threads=4,seed=42)
def best(yy,pp):
    ts=np.linspace(0.05,0.95,181); fs=[f1_score(yy,pp>=t) for t in ts]; i=int(np.argmax(fs)); return round(fs[i],4),round(ts[i],3)
half=(ev.day.values//7)%2
for name,cols in (('soft (baseline)',base),('soft + vector stats',base+VN)):
    p=lgb.train(P,lgb.Dataset(trn[cols],trn.y),1300).predict(ev[cols])
    cf=[]
    for a in (0,1):
        _,t=best(ev.y.values[half==a],p[half==a]); cf.append(f1_score(ev.y.values[half!=a],p[half!=a]>=t))
    print(f'{name:22s} AP={average_precision_score(ev.y,p):.4f} F1={best(ev.y,p)} cross-fitted F1={np.mean(cf):.4f}',f'{time.time()-t0:.0f}s',flush=True)
