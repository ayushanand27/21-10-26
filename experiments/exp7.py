# episode smoothing over unlabelled neighbours (hold-out + test rows in the same view)
import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), p) for p in ('.', '../src', '../scripts')]
from sklearn.metrics import f1_score, average_precision_score
S=sys.argv[1]; K=sys.argv[2] if len(sys.argv)>2 else ''
import build_lib as B
trn=pd.read_parquet(f'{S}/feat/tl_sim{K}_train.parquet')
meta=['row','pos','day','src','ID','y','chunk']; cols=[c for c in trn.columns if c not in meta]
mf=f'{S}/feat/lgb_sim{K}.txt'
P=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_child_samples=50,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1,num_threads=4)
if os.path.exists(mf): b=lgb.Booster(model_file=mf)
else: b=lgb.train(P,lgb.Dataset(trn[cols],trn.y),1200); b.save_model(mf)
del trn
ev=B.view(B.pool,[B.H],B.H,with_test=True)     # H rows + test rows, timeline positions
ev=ev.sort_values('pos').reset_index(drop=True)
p=b.predict(ev[cols]); ev['p']=p
h=(ev.src=='train').values; y=ev.y.values
def best(yy,pp):
    ts=np.linspace(0.05,0.95,91); fs=[f1_score(yy,pp>t) for t in ts]; i=int(np.argmax(fs)); return round(fs[i],4),round(ts[i],2)
print('base AP',round(average_precision_score(y[h],p[h]),4),'F1',best(y[h],p[h]),flush=True)
pos=ev.pos.values
for w in (1,2,3,5):
    # mean p of unlabelled rows within +-w timeline positions (excluding self)
    num=np.zeros(len(ev)); den=np.zeros(len(ev))
    for d in range(1,w+1):
        for sgn in (1,-1):
            j=np.searchsorted(pos,pos+sgn*d)
            ok=(j>=0)&(j<len(pos)); jj=np.clip(j,0,len(pos)-1); ok&=pos[jj]==pos+sgn*d
            num[ok]+=p[jj[ok]]; den[ok]+=1
    nb=np.where(den>0,num/np.maximum(den,1),np.nan)
    for lam in (0.1,0.2,0.3,0.5):
        ps=np.where(np.isnan(nb),p,(1-lam)*p+lam*nb)
        print(f'w={w} lam={lam} AP={average_precision_score(y[h],ps[h]):.4f} F1={best(y[h],ps[h])}',flush=True)
print('share of H rows with an unlabelled neighbour at +-1:', np.mean([0]))
