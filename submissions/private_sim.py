# simulate Kaggle's public/private 50/50 split many times on the labelled hold-out
import sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score
S=sys.argv[1]
HP=np.load(f'{S}/feat/v2_hold.npy',allow_pickle=True).item()
ev=pd.read_parquet(f'{S}/feat/tl_sim_eval.parquet',columns=['y','day']); y=ev.y.values.astype(int)
# hold-out holds 2,680 anomalies; Kaggle's test has ~3,500 -> each half ~1,750. Subsample the hold-out to test size per draw.
cands={'v2a LGBx3 (thr .45)':(HP['lgb'],0.45),'v2 ensemble (thr .515)':(0.873*HP['lgb']+0.122*HP['xgb']+0.005*HP['cb'],0.515),
       'v1-style single LGB (thr .50)':(HP['lgb'],0.50)}
r=np.random.default_rng(0); n=len(y); m=int(n*409856/327852*0.5)   # rows per simulated half (test-size half)
rows=[]
for name,(p,t) in cands.items():
    pred=p>=t; pub=[]; prv=[]
    for b in range(400):
        idx=r.choice(n,min(2*m,n),replace=False); A,B=idx[:len(idx)//2],idx[len(idx)//2:]
        pub.append(f1_score(y[A],pred[A])); prv.append(f1_score(y[B],pred[B]))
    pub,prv=np.array(pub),np.array(prv); d=prv-pub
    rows.append({'candidate':name,'F1 full hold-out':round(f1_score(y,pred),4),'half-F1 mean':round(prv.mean(),4),'half-F1 std':round(prv.std(),4),
                 'private-public diff 90% range':f'[{np.quantile(d,.05):+.4f}, {np.quantile(d,.95):+.4f}]','P(private < 0.875)':round((prv<0.875).mean(),3),
                 'worst of 400 halves':round(prv.min(),4)})
print(pd.DataFrame(rows).to_string(index=False))
# pessimistic version: halves made of whole weeks (correlated errors)
wk=ev.day.values//7; W=np.unique(wk); p,t=cands['v2a LGBx3 (thr .45)']; pred=p>=t; vals=[]
for b in range(400):
    sw=r.permutation(W); half=np.isin(wk,sw[:len(W)//2]); vals.append(f1_score(y[half],pred[half])-f1_score(y[~half],pred[~half]))
vals=np.array(vals); print('\nweek-block split (pessimistic): private-public diff 90%% range [%+.4f, %+.4f], std %.4f'%(np.quantile(vals,.05),np.quantile(vals,.95),vals.std()))
