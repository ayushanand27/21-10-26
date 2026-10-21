import sys, os, time, numpy as np, pandas as pd
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), p) for p in ('.', '../src', '../scripts')]
from pipeline import load, decode, fix_dates, sensor_matrix, standardise, MERGE_W
from softalign import soft_features, SOFT_NAMES
from sklearn.metrics import average_precision_score
S=sys.argv[1]
tr,te=load(f'{S}/data'); dtr,dte=decode(tr),decode(te); dtr['day'],_=fix_dates(tr); dte['day'],_=fix_dates(te)
Str,Ste=standardise(sensor_matrix(dtr),sensor_matrix(dte)); y=tr.target.values; day=dtr.day.values
hide=np.random.default_rng(42).random(len(dtr))<0.2; pool=np.flatnonzero(~hide); H=np.flatnonzero(hide)
ev=pd.read_parquet(f'{S}/feat/tl_sim_eval.parquet',columns=['row','y','yboth1','yprev1','ynext1','yboth3','dist_anom']).set_index('row').loc[H]
print('hard yboth1 AP',round(average_precision_score(ev.y,ev.yboth1),4),' yboth3',round(average_precision_score(ev.y,ev.yboth3),4),' -dist',round(average_precision_score(ev.y,-ev.dist_anom),4),flush=True)
for T in (0.05,0.15,0.4,1.0):
    t=time.time(); sf=soft_features(Str[pool],day[pool],y[pool],Str[H],day[H],MERGE_W,T)
    d=pd.DataFrame(sf,columns=SOFT_NAMES)
    print(f'T={T} ({time.time()-t:.0f}s)', 's_both1 AP',round(average_precision_score(y[H],np.nan_to_num(d.s_both1)),4),
          's_both3',round(average_precision_score(y[H],np.nan_to_num(d.s_both3)),4),'s_inside',round(average_precision_score(y[H],np.nan_to_num(d.s_inside)),4),
          'pmax mean',round(float(np.nanmean(d.s_pmax)),3),flush=True)
