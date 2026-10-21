# context-specific thresholds, cross-fitted on two halves of the hold-out
import sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score
S=sys.argv[1]; key=sys.argv[2] if len(sys.argv)>2 else 'lgb'
HP=np.load(f'{S}/feat/v2_hold.npy',allow_pickle=True).item(); p=HP[key] if key in HP else np.mean([HP[k] for k in HP],0)
ev=pd.read_parquet(f'{S}/feat/tl_sim_eval.parquet',columns=['row','y','day','dist_anom'])
sf=pd.read_parquet(f'{S}/feat/soft_sim_eval_T1.0.parquet',columns=['s_inside','s_boundary','s_both1'])
ev=ev.join(sf,on='row'); y=ev.y.values.astype(int)
half=(ev.day.values//7)%2
grid=np.linspace(0.05,0.95,91)
def bucket(scheme):
    d=ev.dist_anom.values
    if scheme=='dist3': return np.digitize(d,[1.5,10])            # edge / middle / isolated
    if scheme=='dist2': return (d>10).astype(int)                 # near / isolated
    if scheme=='soft':  return np.digitize(np.nan_to_num(ev.s_both1.values),[0.05,0.5])
def fit_thr(pp,yy,b):
    # coordinate ascent on per-bucket thresholds maximising global F1
    K=b.max()+1; t=np.full(K,0.5)
    for _ in range(3):
        for k in range(K):
            best=(-1,0.5)
            for g in grid:
                tt=t.copy(); tt[k]=g; f=f1_score(yy,pp>=tt[b])
                if f>best[0]: best=(f,g)
            t[k]=best[1]
    return t
res=[]
for scheme in ('global','dist2','dist3','soft'):
    fs=[]
    for a in (0,1):
        A,B=half==a,half!=a
        if scheme=='global':
            t=grid[np.argmax([f1_score(y[A],p[A]>=g) for g in grid])]; fs.append(f1_score(y[B],p[B]>=t)); tt=[t]
        else:
            b=bucket(scheme); tt=fit_thr(p[A],y[A],b[A]); fs.append(f1_score(y[B],p[B]>=tt[b[B]]))
    res.append((scheme,np.round(fs,4),round(np.mean(fs),4),np.round(tt,2)))
for r in res: print(r)
