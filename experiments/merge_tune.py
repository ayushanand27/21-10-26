import sys, os, numpy as np, pandas as pd
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), p) for p in ('.', '../src', '../scripts')]
from pipeline import load, decode, fix_dates, sensor_matrix, standardise, _merge_two
tr,te=load(sys.argv[1]); dtr=decode(tr); dtr['day'],_=fix_dates(tr)
y=tr.target.values
rng=np.random.default_rng(1)
days=rng.choice(dtr.day.max()+1,120,replace=False)
def run(transform, w):
    M=transform(dtr)
    M=(M-M.mean(0))/M.std(0)
    e=[];agree=[]
    r2=np.random.default_rng(0)
    for dd in days:
        idx=np.flatnonzero(dtr.day.values==dd); N=len(idx); mask=r2.random(N)<0.2
        src=_merge_two(M[idx[~mask]],M[idx[mask]],w)
        posB=np.flatnonzero(src==1); e.append(np.abs(posB-np.flatnonzero(mask)))
        # label of the labelled row right before / after each hidden row (estimated vs true)
        yl=y[idx[~mask]]; kb_est=posB-np.arange(len(posB)); kb_true=np.flatnonzero(mask)-np.arange(mask.sum())
        est=(yl[np.clip(kb_est-1,0,len(yl)-1)]+yl[np.clip(kb_est,0,len(yl)-1)]); tru=(yl[np.clip(kb_true-1,0,len(yl)-1)]+yl[np.clip(kb_true,0,len(yl)-1)])
        agree.append(est==tru)
    e=np.concatenate(e); a=np.concatenate(agree)
    return e.mean(), (e==0).mean(), (e<=1).mean(), a.mean()
base=lambda d: d[['x1','x2','x3','x4','x5']].to_numpy(float)
ext=lambda d: np.column_stack([d[['x1','x2','x3','x4','x5']].to_numpy(float), (d.x3>0), (d.x4>0)])
for nm,tf,w in [('base',base,[1,2,.5,.5,1]),('x2heavy',base,[1,4,.5,.5,1]),('x2only',base,[0,1,0,0,0]),('eq',base,[1,1,1,1,1]),('x5heavy',base,[1,2,.5,.5,3]),('x1heavy',base,[3,2,.5,.5,1]),('ext',ext,[1,2,.5,.5,1,1,1]),('mix',base,[2,3,1,1,2])]:
    print(nm, np.round(run(tf,np.array(w,float)),4), flush=True)
